import cv2
import numpy as np

MIN_AREA = 20
ADJ_MAX_GAP = 16
MATCH_MIN_OVERLAP = 0.3
TAU = 2
EPS = 1e-6


def box(t):
    return np.ones((2 * t + 1, 2 * t + 1), np.uint8)


def iou(pred, gt):
    inter = float((pred & gt).sum())
    return inter / (float(pred.sum()) + float(gt.sum()) - inter + EPS)


def background_iou(pred, gt):
    npred, ngt = ~pred, ~gt
    inter = float((npred & ngt).sum())
    return inter / (float(npred.sum() + ngt.sum() - inter) + EPS)


def gtband_iou(pred, gt, t=TAU):
    g8 = gt.astype(np.uint8)
    band = cv2.dilate(g8, box(t)) - cv2.erode(g8, box(t))
    p, g = pred.astype(np.uint8) * band, g8 * band
    inter = float((p * g).sum())
    return inter / (float(p.sum()) + float(g.sum()) - inter + EPS)


def boundary_iou(pred, gt, d=TAU):
    g8, p8 = gt.astype(np.uint8), pred.astype(np.uint8)
    gd = g8 - cv2.erode(g8, box(d))
    pd = p8 - cv2.erode(p8, box(d))
    return float((gd & pd).sum()) / (float((gd | pd).sum()) + EPS)


def components(mask, min_area=MIN_AREA):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    return labels, stats, keep


def bbox_gap(stats, i, j):
    xi, yi, wi, hi = stats[i, 0], stats[i, 1], stats[i, 2], stats[i, 3]
    xj, yj, wj, hj = stats[j, 0], stats[j, 1], stats[j, 2], stats[j, 3]
    dx = max(0, max(xi, xj) - min(xi + wi, xj + wj))
    dy = max(0, max(yi, yj) - min(yi + hi, yj + hj))
    return max(dx, dy)


def merge_counts(gt, pred):
    lab_gt, stats_gt, comps = components(gt)
    lab_pr, _, _ = components(pred)
    dom, matched = {}, {}
    for i in comps:
        region = lab_pr[lab_gt == i]
        labels, counts = np.unique(region[region > 0], return_counts=True)
        if len(labels) == 0:
            dom[i], matched[i] = 0, False
            continue
        dom[i] = int(labels[counts.argmax()])
        matched[i] = counts.max() / (lab_gt == i).sum() >= MATCH_MIN_OVERLAP
    n_pairs = n_merged = n_detected = 0
    h, w = gt.shape
    for a in range(len(comps)):
        for b in range(a + 1, len(comps)):
            i, j = comps[a], comps[b]
            if bbox_gap(stats_gt, i, j) > ADJ_MAX_GAP:
                continue
            x0 = max(0, min(stats_gt[i, 0], stats_gt[j, 0]) - ADJ_MAX_GAP)
            y0 = max(0, min(stats_gt[i, 1], stats_gt[j, 1]) - ADJ_MAX_GAP)
            x1 = min(w, max(stats_gt[i, 0] + stats_gt[i, 2], stats_gt[j, 0] + stats_gt[j, 2]) + ADJ_MAX_GAP)
            y1 = min(h, max(stats_gt[i, 1] + stats_gt[i, 3], stats_gt[j, 1] + stats_gt[j, 3]) + ADJ_MAX_GAP)
            sub_i = (lab_gt[y0:y1, x0:x1] == i).astype(np.uint8)
            sub_j = (lab_gt[y0:y1, x0:x1] == j).astype(np.uint8)
            vals = cv2.distanceTransform(1 - sub_i, cv2.DIST_L2, 3)[sub_j == 1]
            if len(vals) == 0:
                continue
            if max(0.0, float(vals.min()) - 1.0) > ADJ_MAX_GAP:
                continue
            n_pairs += 1
            both = matched[i] and matched[j]
            n_detected += int(both)
            n_merged += int(both and dom[i] == dom[j] and dom[i] != 0)
    return n_pairs, n_detected, n_merged


class SemanticScorer:
    def __init__(self, false_alarms=False):
        self.tp = self.fp = self.fn = self.tn = 0
        self.fg, self.bg, self.gtband, self.boundary = [], [], [], []
        self.pairs = self.detected = self.merged = 0
        self.false_alarms = false_alarms
        self.fa_pixels = self.fa_area = self.fa_alarms = self.fa_quadrants = 0

    def add(self, pred, gt):
        self.tp += int((pred & gt).sum())
        self.fp += int((pred & ~gt).sum())
        self.fn += int((~pred & gt).sum())
        self.tn += int((~pred & ~gt).sum())
        self.fg.append(iou(pred, gt))
        self.bg.append(background_iou(pred, gt))
        self.gtband.append(gtband_iou(pred, gt))
        self.boundary.append(boundary_iou(pred, gt))
        n_pairs, n_detected, n_merged = merge_counts(gt, pred)
        self.pairs += n_pairs
        self.detected += n_detected
        self.merged += n_merged
        if self.false_alarms:
            h, w = gt.shape[0] // 2, gt.shape[1] // 2
            for ys in (slice(0, h), slice(h, 2 * h)):
                for xs in (slice(0, w), slice(w, 2 * w)):
                    if gt[ys, xs].any():
                        continue
                    sub = pred[ys, xs]
                    self.fa_pixels += int(sub.sum())
                    self.fa_area += sub.size
                    self.fa_quadrants += 1
                    self.fa_alarms += int(sub.sum() >= MIN_AREA)

    def result(self):
        precision = self.tp / (self.tp + self.fp + EPS)
        recall = self.tp / (self.tp + self.fn + EPS)
        out = {
            "n_tiles": len(self.fg),
            "iou": float(np.mean(self.fg)),
            "miou": (float(np.mean(self.fg)) + float(np.mean(self.bg))) / 2,
            "gtband_iou": float(np.mean(self.gtband)),
            "boundary_iou": float(np.mean(self.boundary)),
            "pooled_iou": self.tp / (self.tp + self.fp + self.fn + EPS),
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall + EPS),
            "amr": self.merged / self.detected if self.detected else None,
            "n_adjacent_pairs": self.pairs,
            "n_detected_pairs": self.detected,
        }
        if self.false_alarms and self.fa_quadrants:
            out["false_alarm"] = {
                "n_quadrants": self.fa_quadrants,
                "fp_area_pct": 100 * self.fa_pixels / self.fa_area,
                "alarm_rate_pct": 100 * self.fa_alarms / self.fa_quadrants,
            }
        return out


def gt_ids(inst):
    return [i for i in np.unique(inst) if i > 0 and (inst == i).sum() >= MIN_AREA]


def greedy_match(gt, pred, thr=0.5):
    gids = gt_ids(gt)
    pids = [int(i) for i in np.unique(pred) if i > 0]
    cand = []
    for gi in gids:
        g = gt == gi
        for pi in np.unique(pred[g]):
            if pi == 0:
                continue
            p = pred == pi
            inter = (g & p).sum()
            v = inter / (g.sum() + p.sum() - inter)
            if v >= thr:
                cand.append((float(v), int(gi), int(pi)))
    cand.sort(reverse=True)
    used_g, used_p, pairs = set(), set(), []
    for v, gi, pi in cand:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        pairs.append((gi, pi, v))
    return pairs, gids, pids


def adjacent_pairs(inst):
    ids = gt_ids(inst)
    out = []
    for a in range(len(ids)):
        di = cv2.distanceTransform((inst != ids[a]).astype(np.uint8), cv2.DIST_L2, 3)
        for b in range(a + 1, len(ids)):
            gap = float(max(0.0, di[inst == ids[b]].min() - 1.0))
            if gap <= ADJ_MAX_GAP:
                out.append((ids[a], ids[b], gap))
    return out


def dominant(pred, region):
    v = pred[region]
    v = v[v > 0]
    if len(v) == 0:
        return 0
    ids, cnt = np.unique(v, return_counts=True)
    return int(ids[cnt.argmax()])


class InstanceScorer:
    def __init__(self):
        self.tp = {0.5: 0, 0.75: 0}
        self.n_gt = self.n_pred = 0

    def add(self, gt, pred):
        for thr in self.tp:
            pairs, gids, pids = greedy_match(gt, pred, thr)
            self.tp[thr] += len(pairs)
        self.n_gt += len(gids)
        self.n_pred += len(pids)

    def result(self):
        out = {"n_gt": self.n_gt, "n_pred": self.n_pred}
        for thr, tp in self.tp.items():
            out[f"precision@{thr}"] = tp / max(1, self.n_pred)
            out[f"recall@{thr}"] = tp / max(1, self.n_gt)
            out[f"f1@{thr}"] = 2 * tp / max(1, self.n_gt + self.n_pred)
        return out
