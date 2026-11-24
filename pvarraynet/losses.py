import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import SDF_TRUNC


def dice_bce(logits, target):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    prob = torch.sigmoid(logits)
    num = 2.0 * (prob * target).sum(dim=(1, 2, 3)) + 1.0
    den = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + 1.0
    return bce + (1.0 - num / den).mean()


def band(y, half_width):
    k = 2 * half_width + 1
    dil = F.max_pool2d(y, k, stride=1, padding=k // 2)
    ero = -F.max_pool2d(-y, k, stride=1, padding=k // 2)
    return (dil - ero).clamp(0.0, 1.0)


class PVArrayNetLoss(nn.Module):
    def __init__(self, w_band=1.0, w_sdf=1.0, w_orient=0.5, w_vertex=0.5, band_half_width=5,
                 boundary_weight=0.0, w_separator=0.3, separator_tau=0.1, separator_samples=2048):
        super().__init__()
        self.w_band = float(w_band)
        self.w_sdf, self.w_orient, self.w_vertex = float(w_sdf), float(w_orient), float(w_vertex)
        self.band_half_width = int(band_half_width)
        self.boundary_weight = float(boundary_weight)
        self.w_separator, self.tau, self.n_samples = float(w_separator), separator_tau, separator_samples

    def semantic(self, logits, y):
        if self.boundary_weight <= 0.0:
            loss = dice_bce(logits, y)
        else:
            wmap = 1.0 + (self.boundary_weight - 1.0) * band(y.float(), self.band_half_width)
            wmap = wmap / wmap.mean()
            bce = (F.binary_cross_entropy_with_logits(logits, y, reduction="none") * wmap).mean()
            prob = torch.sigmoid(logits)
            num = 2.0 * (prob * y * wmap).sum(dim=(1, 2, 3)) + 1.0
            den = (prob * wmap).sum(dim=(1, 2, 3)) + (y * wmap).sum(dim=(1, 2, 3)) + 1.0
            loss = bce + (1.0 - num / den).mean()
        if self.w_band > 0.0:
            b = band(y.float(), self.band_half_width)
            p, g = torch.sigmoid(logits) * b, y * b
            inter = (p * g).sum()
            loss = loss + self.w_band * (1.0 - (inter + 1e-6) / (p.sum() + g.sum() - inter + 1e-6))
        return loss

    def separator(self, out, tgt, prototypes):
        emb = F.normalize(out.emb.float(), dim=1, eps=1e-4)
        protos = F.normalize(prototypes.float(), dim=1, eps=1e-4)
        sdf2 = tgt["sdf"][:, :, ::2, ::2]
        classes = [F.max_pool2d(tgt["sep"], 2) > 0.5, sdf2 >= 0.5, sdf2 <= -0.5]
        flat = emb.permute(0, 2, 3, 1).reshape(-1, emb.shape[1])
        loss, used = emb.sum() * 0.0, 0
        for c, m in enumerate(classes):
            idx = m.reshape(-1).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            if idx.numel() > self.n_samples:
                idx = idx[torch.randint(idx.numel(), (self.n_samples,), device=idx.device)]
            logits = flat[idx] @ protos.T / self.tau
            labels = torch.full((idx.numel(),), c, dtype=torch.long, device=logits.device)
            loss = loss + F.cross_entropy(logits, labels)
            used += 1
        return loss / max(1, used)

    def forward(self, out, tgt, model=None):
        sem = self.semantic(out.sem.float(), tgt["label"])
        sdf_gt = tgt["sdf"]
        wmap = 1.0 + 3.0 * torch.exp(-(sdf_gt.abs() * SDF_TRUNC) / 2.0)
        sdf = (F.smooth_l1_loss(out.sdf.float(), sdf_gt, beta=0.1, reduction="none") * wmap).mean()
        ov = F.normalize(out.orient.float(), dim=1, eps=1e-4)
        valid = tgt["ovalid"]
        dot = (ov * tgt["orient"]).sum(1, keepdim=True)
        orient = ((1.0 - dot) * valid).sum() / valid.sum().clamp(min=1.0)
        vertex = dice_bce(out.vertex.float(), tgt["vertex"])
        total = sem + self.w_sdf * sdf + self.w_orient * orient + self.w_vertex * vertex
        parts = {"sem": float(sem), "sdf": float(sdf), "orient": float(orient), "vertex": float(vertex)}
        if out.emb is not None:
            sep = self.separator(out, tgt, model.prototypes)
            total = total + self.w_separator * sep
            parts["separator"] = float(sep)
        return total, parts
