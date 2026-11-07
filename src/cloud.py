"""Synthetic cloud simulator. Copied verbatim from the training notebook (sections 2.1 and 7.1)."""
import hashlib

import numpy as np
import torch
import torch.nn.functional as F

BASE_SEED = 12345


def stable_seed(text):
    """Process-independent 32-bit seed from a string (unlike hash())."""
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def cloud_mask_np(h, w, rng):
    small = rng.random((max(4, h//16), max(4, w//16)), dtype=np.float32)
    m = F.interpolate(torch.from_numpy(small)[None, None], size=(h, w), mode="bilinear", align_corners=False)[0, 0].numpy()
    m = (m - m.min()) / (m.max() - m.min() + 1e-6)
    return np.clip((m - 0.20) / 0.80, 0, 1).astype(np.float32)

def cloud_corrupt(optical, severity, rng):
    """optical: [C,H,W] numpy. Identical to the original simulator."""
    if severity <= 0: return optical.copy()
    c, h, w = optical.shape
    alpha = np.clip(cloud_mask_np(h, w, rng) * float(severity), 0, 0.95)[None]
    return np.clip(optical * (1 - alpha) + alpha, 0, 1.5).astype(np.float32)


def eval_cloud_s2(s2_chip, cid, severity, base_seed):
    """Deterministic evaluation corruption for one chip."""
    if severity <= 0: return s2_chip
    rng = np.random.default_rng(stable_seed(f"{base_seed}|{cid}|cloud={severity:.3f}"))
    return cloud_corrupt(s2_chip, severity, rng)
