import cv2
import numpy as np
from scipy import ndimage
from skimage.segmentation import watershed

MIN_AREA = 20


def field_decode(sem_prob, sdf_px, t_sem=0.5, t_seed=2.0, min_area=MIN_AREA):
    fg = sem_prob > t_sem
    seed_mask = (sdf_px >= t_seed) & fg
    n, seeds = cv2.connectedComponents(seed_mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.zeros_like(sdf_px, dtype=np.int32)
    inst = watershed(-sdf_px, markers=seeds, mask=fg).astype(np.int32)
    ids, counts = np.unique(inst[inst > 0], return_counts=True)
    for i, c in zip(ids, counts):
        if c < min_area:
            inst[inst == i] = 0
    return inst


def connected_components(mask, min_area=MIN_AREA):
    lab, n = ndimage.label(mask, structure=ndimage.generate_binary_structure(2, 2))
    if n:
        counts = np.bincount(lab.ravel())
        drop = np.where(counts < min_area)[0]
        if len(drop):
            lab[np.isin(lab, drop)] = 0
    return lab.astype(np.int32)
