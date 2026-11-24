import os

import albumentations as A
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .model import SDF_TRUNC

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def read_split(root, split):
    with open(os.path.join(root, f"{split}.txt")) as f:
        return [line.strip() for line in f if line.strip()]


def load_image(root, name):
    return np.array(Image.open(os.path.join(root, "images", name)).convert("RGB"))


def load_mask(root, name):
    return np.array(Image.open(os.path.join(root, "masks", name)).convert("L")) > 127


def load_instances(root, name):
    stem = os.path.splitext(name)[0]
    return np.array(Image.open(os.path.join(root, "instances", stem + ".png")))


def boundary_map(inst):
    b = np.zeros(inst.shape, bool)
    b[:, :-1] |= inst[:, :-1] != inst[:, 1:]
    b[:, 1:] |= inst[:, :-1] != inst[:, 1:]
    b[:-1, :] |= inst[:-1, :] != inst[1:, :]
    b[1:, :] |= inst[:-1, :] != inst[1:, :]
    return b


def field_targets(inst):
    h, w = inst.shape
    fg = inst > 0
    if not fg.any():
        return (np.full((h, w), -1.0, np.float32), np.zeros((2, h, w), np.float32),
                np.zeros((h, w), np.float32), np.zeros((h, w), np.float32))
    dist = cv2.distanceTransform((~boundary_map(inst)).astype(np.uint8), cv2.DIST_L2, 3)
    sdf = np.clip(np.where(fg, dist, -dist), -SDF_TRUNC, SDF_TRUNC) / SDF_TRUNC
    orient = np.zeros((2, h, w), np.float32)
    vertex = np.zeros((h, w), np.uint8)
    for i in np.unique(inst[fg]):
        comp = (inst == i).astype(np.uint8)
        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        _, (rw, rh), angle = cv2.minAreaRect(contour)
        theta = np.deg2rad(angle + (90.0 if rw < rh else 0.0))
        m = comp.astype(bool)
        orient[0][m], orient[1][m] = np.cos(2 * theta), np.sin(2 * theta)
        for v in cv2.approxPolyDP(contour, 2.0, True)[:, 0]:
            cv2.circle(vertex, (int(v[0]), int(v[1])), 2, 1, -1)
    return sdf.astype(np.float32), orient, fg.astype(np.float32), vertex.astype(np.float32)


def separator_targets(inst, radius=3):
    from scipy import ndimage

    fg = inst > 0
    if not fg.any():
        return np.zeros(inst.shape, np.float32)
    count = np.zeros(inst.shape, np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)
    h, w = inst.shape
    for i, sl in enumerate(ndimage.find_objects(inst.astype(np.int32)), start=1):
        if sl is None:
            continue
        y0, y1 = max(0, sl[0].start - radius), min(h, sl[0].stop + radius)
        x0, x1 = max(0, sl[1].start - radius), min(w, sl[1].stop + radius)
        blob = (inst[y0:y1, x0:x1] == i).astype(np.uint8)
        count[y0:y1, x0:x1] += cv2.dilate(blob, kernel)
    return ((~fg) & (count >= 2)).astype(np.float32)


def to_tensor(image):
    x = image.astype(np.float32) / 255.0
    x = (x - np.array(MEAN, np.float32)) / np.array(STD, np.float32)
    return torch.from_numpy(x.transpose(2, 0, 1))


class PVTiles(Dataset):
    def __init__(self, root, names, train=False, separator=False):
        self.root = root
        self.names = list(names)
        self.train = train
        self.separator = separator
        self.augment = A.Compose(
            [A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5),
             A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5)],
            additional_targets={"inst": "mask"}) if train else None

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        name = self.names[i]
        image = load_image(self.root, name)
        mask = load_mask(self.root, name).astype(np.uint8)
        if not self.train:
            return {"image": to_tensor(image),
                    "label": torch.from_numpy(mask[None].astype(np.float32)),
                    "name": name}
        out = self.augment(image=image, mask=mask, inst=load_instances(self.root, name))
        sdf, orient, ovalid, vertex = field_targets(out["inst"])
        item = {
            "image": to_tensor(out["image"]),
            "label": torch.from_numpy(out["mask"][None].astype(np.float32)),
            "sdf": torch.from_numpy(sdf[None]),
            "orient": torch.from_numpy(orient),
            "ovalid": torch.from_numpy(ovalid[None]),
            "vertex": torch.from_numpy(vertex[None]),
            "name": name,
        }
        if self.separator:
            item["sep"] = torch.from_numpy(separator_targets(out["inst"])[None])
        return item
