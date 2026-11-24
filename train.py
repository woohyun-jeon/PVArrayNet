import argparse
import json
import os
import random
import time

import numpy as np
import torch

from pvarraynet.data import PVTiles, read_split
from pvarraynet.encoder import ENCODERS
from pvarraynet.losses import PVArrayNetLoss, dice_bce
from pvarraynet.model import PVArrayNet, count_parameters


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def validate(model, root, names, device, batch_size, workers):
    was_training = model.training
    model.eval()
    loader = torch.utils.data.DataLoader(PVTiles(root, names), batch_size=batch_size, shuffle=False,
                                         num_workers=workers, pin_memory=True)
    ious, loss_sum, n = [], 0.0, 0
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(image)
        loss_sum += float(dice_bce(out.sem.float(), label)) * image.shape[0]
        n += image.shape[0]
        pred = (torch.sigmoid(out.sem.float()) > 0.5)[:, 0].cpu().numpy()
        gt = batch["label"][:, 0].numpy() > 0.5
        for i in range(pred.shape[0]):
            inter = (pred[i] & gt[i]).sum()
            ious.append(float(inter / (pred[i].sum() + gt[i].sum() - inter + 1e-6)))
    if was_training:
        model.train()
    return loss_sum / max(1, n), float(np.mean(ious))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--pretrained-dir", default="pretrained")
    p.add_argument("--out", required=True)
    p.add_argument("--encoder", default="vitb16_lvd", choices=ENCODERS)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--no-split-head", action="store_true")
    p.add_argument("--no-detail-path", action="store_true")
    p.add_argument("--w-band", type=float, default=1.0)
    p.add_argument("--band-half-width", type=int, default=5)
    p.add_argument("--boundary-weight", type=float, default=0.0)
    p.add_argument("--w-sdf", type=float, default=1.0)
    p.add_argument("--w-orient", type=float, default=0.5)
    p.add_argument("--w-vertex", type=float, default=0.5)
    p.add_argument("--separator", action="store_true")
    p.add_argument("--w-separator", type=float, default=0.3)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-ckpt", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device)
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    train_names = read_split(args.data_root, "train")
    valid_names = read_split(args.data_root, "valid")
    model = PVArrayNet(args.encoder, args.pretrained_dir, args.lora_r, args.lora_alpha,
                       grad_ckpt=args.grad_ckpt, split_head=not args.no_split_head,
                       detail_path=not args.no_detail_path,
                       separator_embedding=args.separator).to(device)
    criterion = PVArrayNetLoss(w_band=args.w_band, w_sdf=args.w_sdf, w_orient=args.w_orient,
                               w_vertex=args.w_vertex, band_half_width=args.band_half_width,
                               boundary_weight=args.boundary_weight, w_separator=args.w_separator)
    optim = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad],
                              lr=args.lr, weight_decay=args.weight_decay)
    g = torch.Generator()
    g.manual_seed(args.seed)
    loader = torch.utils.data.DataLoader(
        PVTiles(args.data_root, train_names, train=True, separator=args.separator),
        batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True,
        drop_last=True, generator=g, persistent_workers=args.workers > 0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(1, args.epochs * len(loader)))

    config = {**vars(args), "params_total": count_parameters(model, False),
              "params_trained": count_parameters(model)}
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(config, f, indent=1)
    print(f"trained {config['params_trained'] / 1e6:.2f} M of {config['params_total'] / 1e6:.2f} M parameters")

    keys = ("label", "sdf", "orient", "ovalid", "vertex") + (("sep",) if args.separator else ())
    best_iou = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, losses = time.time(), []
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            tgt = {k: batch[k].to(device, non_blocking=True) for k in keys}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(image)
            loss, _ = criterion(out, tgt, model)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            sched.step()
            losses.append(loss.detach().item())
        val_loss, val_iou = validate(model, args.data_root, valid_names, device, 4, 4)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "valid_loss": val_loss,
               "valid_iou": val_iou, "sec": round(time.time() - t0, 1)}
        with open(os.path.join(args.out, "log.jsonl"), "a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        state = {"model": model.state_dict(), "epoch": epoch, "valid_iou": val_iou, "config": config}
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(state, os.path.join(args.out, "best.pt"))
        torch.save(state, os.path.join(args.out, "last.pt"))


if __name__ == "__main__":
    main()
