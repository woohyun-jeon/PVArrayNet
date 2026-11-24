# PVArrayNet

Code for

> W. Jeon and S. Hong, "Structural separation of adjacent photovoltaic arrays for area-based capacity
> attribution from high-resolution remote sensing imagery," *Applied Energy*, 2026.
> https://doi.org/10.1016/j.apenergy.2026.129278

PVArrayNet segments photovoltaic (PV) arrays and separates adjacent arrays into individual instances, so
that the area-based capacity proxy can be attributed per array. It adapts a frozen DINOv3 encoder with
low-rank adapters, carries thin boundaries through a full-resolution detail path, supervises contour
placement with an overlap loss restricted to a ground-truth contour band, and recovers arrays with a
deterministic watershed on a predicted signed distance field.

## Installation

```bash
pip install -r requirements.txt
```

The code was run with Python 3.10, PyTorch 2.11 and timm 1.0.28.

## Pretrained encoders

Download the DINOv3 checkpoints from the [official DINOv3 release](https://github.com/facebookresearch/dinov3)
and place them in `pretrained/` (or pass `--pretrained-dir`). File names must be kept as released.

| `--encoder` | File |
|---|---|
| `vitb16_lvd` (default) | `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth` |
| `vitl16_lvd` | `dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` |
| `vitl16_sat` | `dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth` |
| `convnextb_lvd` | `dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth` |

## Data

The prepared NGII and PV01 datasets used in the paper, with the splits and instance maps below, are
available from the authors on request.

Each dataset is a folder with one file per tile in `images/`, `masks/` and `instances/`, plus split lists:

```
<data-root>/
  images/       RGB tiles, PNG
  masks/        semantic masks, 0 = background, 255 = PV
  instances/    16-bit instance maps, 0 = background, 1..N = arrays
  train.txt  valid.txt  test.txt
```

- **NGII**: 7,645 aerial tiles of 1024 x 1024 px at 0.25 m, split 4,844 / 1,472 / 1,329 by parent scene.
- **PV01**: the PV01 subset of [Jiang et al. (2021)](https://doi.org/10.5194/essd-13-5389-2021), 645 patches of
  256 x 256 px at 0.1 m, with the map-disjoint split 387 / 129 / 129. PV01 has no instance annotation, so
  `instances/` holds the 8-connected components of the mask, with components below 20 px removed.

## Training

```bash
python train.py --data-root data/NGII --out runs/pvarraynet_s42 --seed 42 --device cuda:0
```

All models in the paper were trained for 100 epochs at batch size 4 with AdamW (learning rate 3e-4,
weight decay 1e-4, cosine schedule) and seeds 42, 123, 777, 2025 and 2026. `best.pt` is the epoch with the
highest validation IoU. PV01 uses the same command with `--data-root data/PV01`.

The arms of Table 3 and Section 3.3.3 add the following flags to the command above.

| Arm | Flags |
|---|---|
| Base (no boundary lever) | `--no-split-head --w-band 0` |
| + split head | `--w-band 0` |
| + GT-band IoU loss | `--no-split-head` |
| + both (adopted) | none |
| Boundary-weighted CE control | `--no-split-head --w-band 0 --boundary-weight 5` |
| Fields: distance only | `--no-split-head --w-band 0 --w-orient 0 --w-vertex 0` |
| No field heads | `--no-split-head --w-band 0 --w-sdf 0 --w-orient 0 --w-vertex 0` |
| No detail path | `--no-detail-path` |
| Base + separator InfoNCE | `--no-split-head --w-band 0 --separator` |
| Adopted + separator InfoNCE | `--separator` |
| Frozen encoder, no LoRA | `--lora-r 0` |
| Other encoders (Table 5) | `--encoder vitl16_lvd`, `--encoder vitl16_sat` or `--encoder convnextb_lvd` |

## Evaluation

```bash
python evaluate.py --data-root data/NGII --checkpoint runs/pvarraynet_s42/best.pt \
    --out runs/pvarraynet_s42/test.json --instances --false-alarms
```

The output reports, on the chosen split:

- `semantic`: tile-averaged IoU, mIoU, GT-band IoU and Boundary IoU (band half-width 2 px); pooled IoU,
  precision, recall and F1; the adjacent merge rate (AMR) over ground-truth pairs at most 16 px apart;
  and, with `--false-alarms`, the false-positive area and alarm rate on panel-free 512 x 512 quadrants.
- `instance`: instance precision, recall and F1 at mask IoU 0.5 and 0.75 after the field decode.
- `capacity`: detection rate, merged array rate, merged capacity, per-array capacity error and array count
  bias at a constant density of 0.20 kWp/m², for the field decode and for connected components of the same
  mask, over all tiles and over the dense stratum (tiles with a ground-truth pair at most 2 px apart).

Binary masks from any other model can be scored the same way, with connected components as the instance
decode:

```bash
python evaluate.py --data-root data/NGII --masks <mask-folder> --out baseline.json --instances --false-alarms
```

## Citation

```bibtex
@article{jeon2026pvarraynet,
  title   = {Structural separation of adjacent photovoltaic arrays for area-based capacity attribution
             from high-resolution remote sensing imagery},
  author  = {Jeon, Woohyun and Hong, Seunghyeok},
  journal = {Applied Energy},
  year    = {2026},
  doi     = {10.1016/j.apenergy.2026.129278}
}
```
