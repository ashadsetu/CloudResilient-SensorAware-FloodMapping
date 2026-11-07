"""Metrics on valid pixels only (y != 255). Definitions mirror the training notebook (section 14.1)."""
from __future__ import annotations

import numpy as np

IGNORE_INDEX = 255


def counts(pred: np.ndarray, y: np.ndarray) -> dict:
    """pred: bool [H,W]; y: uint8 [H,W] (0 land, 1 flood, 255 ignore)."""
    valid = y != IGNORE_INDEX
    truth = y == 1
    return dict(
        tp=int((pred & truth).sum()),
        fp=int((pred & ~truth & valid).sum()),
        fn=int((~pred & truth).sum()),
        tn=int((~pred & ~truth & valid).sum()),
    )


def has_flood(y: np.ndarray) -> bool:
    return bool((y == 1).any())


def iou(c: dict) -> float:
    den = c["tp"] + c["fp"] + c["fn"]
    return float("nan") if den == 0 else c["tp"] / den


def miss_fraction(c: dict) -> float:
    """FN / (TP + FN); undefined (NaN) when the scene has no flood pixels."""
    den = c["tp"] + c["fn"]
    return float("nan") if den == 0 else c["fn"] / den


def scene_metrics(pred: np.ndarray, y: np.ndarray) -> dict:
    c = counts(pred, y)
    return dict(iou=iou(c), miss=miss_fraction(c), fp=c["fp"], fn=c["fn"], tp=c["tp"], tn=c["tn"],
                has_flood=has_flood(y))


def flooded_share(pred: np.ndarray, y: np.ndarray) -> float:
    valid = y != IGNORE_INDEX
    n = int(valid.sum())
    return float("nan") if n == 0 else float((pred & valid).sum() / n)


def flooded_area_km2(pred: np.ndarray, y: np.ndarray, area_m2: float) -> float:
    """Approximate: flood pixels x area of one 128-px-grid cell. Ignore pixels are excluded."""
    valid = y != IGNORE_INDEX
    return float((pred & valid).sum() * float(area_m2) / 1e6)


def agreement_iou(a: np.ndarray, b: np.ndarray, y: np.ndarray | None = None) -> float:
    """IoU between two binary maps (optionally restricted to valid pixels)."""
    if y is not None:
        v = y != IGNORE_INDEX
        a, b = a & v, b & v
    union = (a | b).sum()
    return float("nan") if union == 0 else float((a & b).sum() / union)


def error_map(pred: np.ndarray, y: np.ndarray) -> np.ndarray:
    """RGBA image. Correct flood = green-blue, false alarm = orange, missed = red, ignore = transparent."""
    h, w = y.shape
    out = np.zeros((h, w, 4), np.float32)
    valid = y != IGNORE_INDEX
    truth = y == 1
    out[valid, :3] = (0.93, 0.95, 0.97); out[valid, 3] = 1.0           # land / correct no-flood
    out[pred & truth, :3] = (0.10, 0.62, 0.62)                          # correct flood
    out[pred & ~truth & valid, :3] = (0.96, 0.60, 0.10)                 # false alarm
    out[~pred & truth, :3] = (0.85, 0.15, 0.15)                         # missed flood
    return out
