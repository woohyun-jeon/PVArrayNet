import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize

VIT_ENCODERS = {
    "vitb16_lvd": ("vit_base_patch16_dinov3_qkvb", "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"),
    "vitl16_lvd": ("vit_large_patch16_dinov3_qkvb", "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"),
    "vitl16_sat": ("vit_large_patch16_dinov3", "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"),
}
CONVNEXT_ENCODERS = {
    "convnextb_lvd": ("convnext_base.dinov3_lvd1689m", "dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth"),
}
ENCODERS = list(VIT_ENCODERS) + list(CONVNEXT_ENCODERS)


class LoRAWeight(nn.Module):
    def __init__(self, out_features, in_features, r=16, alpha=32):
        super().__init__()
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scale = alpha / r

    def forward(self, w):
        return w + (self.lora_B @ self.lora_A).to(w.dtype) * self.scale


def add_lora(linear, r, alpha):
    parametrize.register_parametrization(
        linear, "weight", LoRAWeight(linear.out_features, linear.in_features, r, alpha))


def checkpoint_path(pretrained_dir, filename):
    path = os.path.join(pretrained_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return path


class LoRAViT(nn.Module):
    def __init__(self, name, pretrained_dir, r=16, alpha=32, grad_ckpt=False):
        super().__init__()
        import timm

        timm_name, filename = VIT_ENCODERS[name]
        self.name = name
        self.checkpoint = checkpoint_path(pretrained_dir, filename)
        self.backbone = timm.create_model(
            timm_name, pretrained=True, pretrained_cfg_overlay=dict(file=self.checkpoint),
            dynamic_img_size=True, num_classes=0)
        self.patch_size = 16
        self.dim = int(self.backbone.embed_dim)
        for p in self.backbone.parameters():
            p.requires_grad = False
        if r > 0:
            for blk in self.backbone.blocks:
                add_lora(blk.attn.qkv, r, alpha)
                add_lora(blk.attn.proj, r, alpha)
        if grad_ckpt:
            self.backbone.set_grad_checkpointing(True)
        self.backbone.eval()

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, x):
        b, _, h, w = x.shape
        gh, gw = max(1, round(h / self.patch_size)), max(1, round(w / self.patch_size))
        target = (gh * self.patch_size, gw * self.patch_size)
        if (h, w) != target:
            x = F.interpolate(x, size=target, mode="bilinear", align_corners=False)
        tokens = self.backbone.forward_features(x)
        n_extra = tokens.shape[1] - gh * gw
        return tokens[:, n_extra:].transpose(1, 2).reshape(b, self.dim, gh, gw)


class LoRAConvNeXt(nn.Module):
    def __init__(self, name, pretrained_dir, r=16, alpha=32, grad_ckpt=False):
        super().__init__()
        import timm

        timm_name, filename = CONVNEXT_ENCODERS[name]
        self.name = name
        self.checkpoint = checkpoint_path(pretrained_dir, filename)
        self.backbone = timm.create_model(
            timm_name, pretrained=True, pretrained_cfg_overlay=dict(file=self.checkpoint),
            features_only=True)
        self.dims = self.backbone.feature_info.channels()
        for p in self.backbone.parameters():
            p.requires_grad = False
        if r > 0:
            for mod in self.backbone.modules():
                if type(mod).__name__ == "ConvNeXtBlock":
                    add_lora(mod.mlp.fc1, r, alpha)
                    add_lora(mod.mlp.fc2, r, alpha)
        if grad_ckpt and hasattr(self.backbone, "set_grad_checkpointing"):
            self.backbone.set_grad_checkpointing(True)
        self.backbone.eval()

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, x):
        return self.backbone(x)
