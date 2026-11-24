import argparse
import json
import os

import numpy as np
import torch
from PIL import Image

from pvarraynet.capacity import CapacityScorer
from pvarraynet.data import PVTiles, load_instances, load_mask, read_split
from pvarraynet.decode import connected_components, field_decode
from pvarraynet.metrics import InstanceScorer, SemanticScorer
from pvarraynet.model import SDF_TRUNC, PVArrayNet


def load_model(checkpoint, pretrained_dir, device):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = state["config"]
    model = PVArrayNet(cfg["encoder"], pretrained_dir, cfg["lora_r"], cfg["lora_alpha"],
                       split_head=not cfg["no_split_head"], detail_path=not cfg["no_detail_path"],
                       separator_embedding=cfg["separator"])
    missing, unexpected = model.load_state_dict(state["model"], strict=False)
    frozen = [k for k in missing if k.startswith("encoder.backbone.") and "lora_" not in k]
    if unexpected or len(frozen) != len(missing):
        raise RuntimeError(f"checkpoint mismatch: missing {sorted(set(missing) - set(frozen))}, "
                           f"unexpected {unexpected}")
    return model.to(device).eval()


@torch.no_grad()
def predict(model, root, names, device, batch_size, workers):
    loader = torch.utils.data.DataLoader(PVTiles(root, names), batch_size=batch_size, shuffle=False,
                                         num_workers=workers, pin_memory=True)
    for batch in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(batch["image"].to(device, non_blocking=True))
        sem = torch.sigmoid(out.sem.float())[:, 0].cpu().numpy()
        sdf = out.sdf.float()[:, 0].cpu().numpy() * SDF_TRUNC
        for i in range(sem.shape[0]):
            yield batch["name"][i], sem[i], sdf[i]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--checkpoint")
    p.add_argument("--masks")
    p.add_argument("--pretrained-dir", default="pretrained")
    p.add_argument("--out", required=True)
    p.add_argument("--save-predictions")
    p.add_argument("--false-alarms", action="store_true")
    p.add_argument("--instances", action="store_true")
    p.add_argument("--gsd", type=float, default=0.25)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    if bool(args.checkpoint) == bool(args.masks):
        raise SystemExit("give exactly one of --checkpoint or --masks")

    names = read_split(args.data_root, args.split)
    semantic = SemanticScorer(false_alarms=args.false_alarms)
    instance = InstanceScorer()
    capacity = {"cc_decode": CapacityScorer(args.gsd)}
    if args.checkpoint:
        device = torch.device(args.device)
        model = load_model(args.checkpoint, args.pretrained_dir, device)
        source = predict(model, args.data_root, names, device, args.batch_size, args.workers)
        capacity["field_decode"] = CapacityScorer(args.gsd)
    else:
        source = ((n, (np.array(Image.open(os.path.join(args.masks, n)).convert("L")) > 127), None)
                  for n in names)
    if args.save_predictions:
        os.makedirs(args.save_predictions, exist_ok=True)

    for name, sem, sdf in source:
        pred = sem > 0.5 if sdf is not None else sem
        semantic.add(pred, load_mask(args.data_root, name))
        if args.instances:
            gt_inst = load_instances(args.data_root, name).astype(np.int32)
            capacity["cc_decode"].add(gt_inst, connected_components(pred))
            if sdf is not None:
                inst = field_decode(sem, sdf)
                instance.add(gt_inst, inst)
                capacity["field_decode"].add(gt_inst, inst)
        if args.save_predictions:
            Image.fromarray(pred.astype(np.uint8) * 255).save(os.path.join(args.save_predictions, name))

    result = {"split": args.split, "semantic": semantic.result()}
    if args.instances:
        if args.checkpoint:
            result["instance"] = instance.result()
        result["capacity"] = {k: v.result() for k, v in capacity.items()}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps(result["semantic"], indent=1))


if __name__ == "__main__":
    main()
