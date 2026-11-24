import numpy as np

from .metrics import adjacent_pairs, dominant, greedy_match, gt_ids

KWP_PER_M2 = 0.20
COUNT_MIN_PX = (20, 100, 400, 1600)
STRATA = ("all", "dense", "remainder")


class CapacityScorer:
    def __init__(self, gsd=0.25, dense_gap=2):
        self.px_area = gsd * gsd
        self.dense_gap = dense_gap
        self.acc = {k: {"tiles": 0, "n_gt": 0, "n_pred": 0, "matched": 0, "merged_arrays": 0,
                        "gt_area": 0.0, "merged_area": 0.0, "cap_abs": [],
                        "sweep": {t: [0, 0] for t in COUNT_MIN_PX}} for k in STRATA}

    def add(self, gt, pred):
        gids = gt_ids(gt)
        p_ids = [i for i in np.unique(pred) if i > 0]
        dense = any(g <= self.dense_gap for _, _, g in adjacent_pairs(gt))
        claim = {}
        for gi in gids:
            pi = dominant(pred, gt == gi)
            if pi:
                claim.setdefault(pi, []).append(gi)
        merged = [gi for v in claim.values() if len(v) >= 2 for gi in v]
        pairs, _, _ = greedy_match(gt, pred)
        cap_abs = [abs((pred == pi).sum() * self.px_area - (gt == gi).sum() * self.px_area) * KWP_PER_M2
                   for gi, pi, _ in pairs]
        gt_area = sum(int((gt == gi).sum()) for gi in gids) * self.px_area
        merged_area = sum(int((gt == gi).sum()) for gi in merged) * self.px_area
        g_areas = np.array([(gt == gi).sum() for gi in gids]) if gids else np.array([])
        p_areas = np.array([(pred == pi).sum() for pi in p_ids]) if p_ids else np.array([])
        for key in ("all", "dense" if dense else "remainder"):
            s = self.acc[key]
            s["tiles"] += 1
            s["n_gt"] += len(gids)
            s["n_pred"] += len(p_ids)
            s["matched"] += len(pairs)
            s["merged_arrays"] += len(merged)
            s["gt_area"] += gt_area
            s["merged_area"] += merged_area
            s["cap_abs"].extend(cap_abs)
            for t in COUNT_MIN_PX:
                s["sweep"][t][0] += int((g_areas >= t).sum())
                s["sweep"][t][1] += int((p_areas >= t).sum())

    def result(self):
        out = {}
        for key, s in self.acc.items():
            if not s["tiles"]:
                continue
            out[key] = {
                "tiles": s["tiles"],
                "n_gt": s["n_gt"],
                "detection_rate": s["matched"] / s["n_gt"],
                "merged_array_rate_pct": 100 * s["merged_arrays"] / s["n_gt"],
                "merged_capacity_pct": 100 * s["merged_area"] / s["gt_area"],
                "per_array_mae_kwp": float(np.mean(s["cap_abs"])) if s["cap_abs"] else None,
                "count_bias_pct": 100 * (s["n_pred"] - s["n_gt"]) / s["n_gt"],
                "count_bias_pct_by_min_area": {
                    f"{t}px": 100 * (n_pred - n_gt) / max(1, n_gt)
                    for t, (n_gt, n_pred) in s["sweep"].items()},
            }
        return out
