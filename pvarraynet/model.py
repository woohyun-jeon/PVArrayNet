from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import CONVNEXT_ENCODERS, LoRAConvNeXt, LoRAViT

SDF_TRUNC = 8.0


def conv_block(cin, cout, stride=1):
    groups = 16 if cout % 16 == 0 else 8
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
        nn.GroupNorm(groups, cout),
        nn.GELU(),
    )


class PVArrayNet(nn.Module):
    def __init__(self, encoder="vitb16_lvd", pretrained_dir="pretrained", lora_r=16, lora_alpha=32,
                 grad_ckpt=False, split_head=True, detail_path=True, separator_embedding=False):
        super().__init__()
        self.detail_path = bool(detail_path)
        self.split_head = bool(split_head)
        self.separator_embedding = bool(separator_embedding)
        self.convnext = encoder in CONVNEXT_ENCODERS
        if self.convnext and not self.detail_path:
            raise ValueError("removing the detail path is defined for ViT encoders only")
        encoder_cls = LoRAConvNeXt if self.convnext else LoRAViT
        self.encoder = encoder_cls(encoder, pretrained_dir, lora_r, lora_alpha, grad_ckpt)

        if self.detail_path:
            self.hr1 = conv_block(3, 24, stride=1)
            self.hr2 = conv_block(24, 32, stride=2)
            self.hr4 = conv_block(32, 48, stride=2)

        if self.convnext:
            self.lat = nn.ModuleList(nn.Conv2d(c, 128, 1) for c in self.encoder.dims)
            self.fpn_out = conv_block(128, 96)
        else:
            self.tok = nn.Sequential(conv_block(self.encoder.dim, 256), conv_block(256, 128))
            self.up8 = conv_block(128, 96)
            self.up4 = conv_block(96, 96)

        if self.detail_path:
            self.fuse4 = conv_block(96 + 48, 96)
            self.fuse2 = conv_block(96 + 32, 64)
            self.fuse1 = conv_block(64 + 24, 32)
        else:
            self.fuse4 = conv_block(96, 96)
            self.fuse2 = conv_block(96, 64)
            self.fuse1 = conv_block(64, 32)

        if self.split_head:
            self.head_sem = nn.Sequential(conv_block(32, 32), nn.Conv2d(32, 1, 1))
            self.head_fld = nn.Sequential(conv_block(32, 32), nn.Conv2d(32, 4, 1))
        else:
            self.head = nn.Conv2d(32, 5, 1)

        if self.separator_embedding:
            self.proj = nn.Sequential(nn.Conv2d(64, 64, 1), nn.GELU(), nn.Conv2d(64, 64, 1))
            self.prototypes = nn.Parameter(torch.randn(3, 64) * 0.02)

    @staticmethod
    def resize(x, size):
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

    def forward(self, x):
        H, W = x.shape[-2:]
        if self.detail_path:
            h1 = self.hr1(x)
            h2 = self.hr2(h1)
            h4 = self.hr4(h2)

        if self.convnext:
            feats = self.encoder(x)
            t = self.lat[-1](feats[-1].to(h1.dtype))
            for lvl in range(len(feats) - 2, -1, -1):
                t = self.lat[lvl](feats[lvl].to(h1.dtype)) + self.resize(t, feats[lvl].shape[-2:])
            t = self.fpn_out(t)
        else:
            feats = self.encoder(x)
            t = self.tok(feats.to(h1.dtype) if self.detail_path else feats)
            t = self.up8(F.interpolate(t, scale_factor=2, mode="bilinear", align_corners=False))
            t = self.up4(self.resize(t, (H // 4, W // 4)))

        if self.detail_path:
            f4 = self.fuse4(torch.cat([t, h4], 1))
            f2 = self.fuse2(torch.cat([self.resize(f4, h2.shape[-2:]), h2], 1))
            f1 = self.fuse1(torch.cat([self.resize(f2, h1.shape[-2:]), h1], 1))
        else:
            f4 = self.fuse4(t)
            f2 = self.fuse2(self.resize(f4, (H // 2, W // 2)))
            f1 = self.fuse1(self.resize(f2, (H, W)))

        raw = torch.cat([self.head_sem(f1), self.head_fld(f1)], 1) if self.split_head else self.head(f1)
        return SimpleNamespace(
            sem=raw[:, 0:1],
            sdf=torch.tanh(raw[:, 1:2]),
            orient=raw[:, 2:4],
            vertex=raw[:, 4:5],
            emb=self.proj(f2) if self.separator_embedding else None,
        )


def count_parameters(module, trainable_only=True):
    return sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only)
