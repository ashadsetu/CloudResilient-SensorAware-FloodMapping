"""Generate tiny synthetic stand-ins for the exported assets.

Every file is named *_PLACEHOLDER* (or the chip id starts with PLACEHOLDER_) so it can never be
mistaken for a real export. They exist ONLY so the app and the tests run before the real
`app_assets/` folder is copied in. The model weights are randomly initialised: predictions are
meaningless, and all numbers in the placeholder CSVs are fake.

    python -m src.placeholders
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .model import SensorAwareFusionUNet

A = Path(__file__).resolve().parent.parent / "assets"


def _blob(rng, h=128, w=128, frac=0.25):
    small = rng.random((8, 8)).astype(np.float32)
    t = torch.nn.functional.interpolate(torch.from_numpy(small)[None, None], size=(h, w), mode="bilinear",
                                        align_corners=False)[0, 0].numpy()
    thr = np.quantile(t, 1 - frac) if frac > 0 else np.inf
    return t >= thr


def make_chip(i, frac, ignore=False):
    rng = np.random.default_rng(100 + i)
    flood = _blob(rng, frac=frac)
    s1 = np.clip(0.55 + 0.05 * rng.standard_normal((2, 128, 128)) - 0.25 * flood, 0, 1).astype(np.float32)
    s2 = np.clip(0.15 + 0.03 * rng.standard_normal((13, 128, 128)) + 0.05 * (~flood), 0, 1.5).astype(np.float32)
    y = flood.astype(np.uint8)
    if ignore:
        y[:16, :] = 255
    return s1, s2, y


def main():
    for d in ("models", "demo_chips", "thumbs", "results"):
        (A / d).mkdir(parents=True, exist_ok=True)
    cfg = dict(base=16, drop_p=0.1, flags={}, s2_channels=13, img_size=128)
    torch.manual_seed(0)
    m = SensorAwareFusionUNet(cfg["base"], cfg["s2_channels"], cfg["drop_p"])
    torch.save(dict(model_state_dict=m.state_dict(), model_class="SensorAwareFusionUNet", config=cfg,
                    preproc=dict(img_size=128, placeholder=True)), A / "models/aware_fusion_PLACEHOLDER.pt")

    specs = [("PLACEHOLDER_001", "Synthetic wide flood", "wide_flood", 0.35, False),
             ("PLACEHOLDER_002", "Synthetic small flood", "small_flood", 0.08, False),
             ("PLACEHOLDER_003", "Synthetic scene with ignore pixels", "ignore_pixels", 0.20, True),
             ("PLACEHOLDER_004", "Synthetic scene with no flood", "no_flood", 0.0, False)]
    manifest = []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for i, (cid, title, tag, frac, ign) in enumerate(specs):
        s1, s2, y = make_chip(i, frac, ign)
        np.savez_compressed(A / f"demo_chips/{cid}.npz", s1=s1, s2=s2, y=y, area_m2=np.float32(900.0 * 40 * 40 / 1.0),
                            chip_id=cid, event="PLACEHOLDER", split="test")
        fig, ax = plt.subplots(1, 3, figsize=(6, 2))
        ax[0].imshow(s1[0], cmap="gray"); ax[1].imshow(np.clip(s2[[3, 2, 1]].transpose(1, 2, 0) * 3, 0, 1))
        ax[2].imshow(np.where(y == 255, np.nan, y), cmap="Blues")
        for a in ax: a.axis("off")
        fig.savefig(A / f"thumbs/{cid}.png", dpi=60, bbox_inches="tight"); plt.close(fig)
        v = y[y != 255]
        manifest.append(dict(chip_id=cid, tag=tag, title=title, reason="PLACEHOLDER data - replace with your export",
                             event="PLACEHOLDER", split="test", flood_fraction=float((v == 1).mean()),
                             precomputed_iou=None))
    (A / "manifest_PLACEHOLDER.json").write_text(json.dumps(manifest, indent=2))
    thr = {t: {s: dict(threshold=0.5, feasible=True, n_cal_flood_chips=0, n_cal_chips=0)
               for s in ("both", "radar_only", "optical_only")} for t in ("0.05", "0.10", "0.20")}
    (A / "thresholds_PLACEHOLDER.json").write_text(json.dumps(thr, indent=2))
    (A / "model_card_PLACEHOLDER.json").write_text(json.dumps(dict(note="PLACEHOLDER", dataset="Sen1Floods11"), indent=2))

    # fake results tables (column names = the schema the Results page reads)
    import pandas as pd
    rng = np.random.default_rng(1)
    rows = []
    for mdl, states in (("radar_unet", ["radar_only"]), ("optical_unet", ["optical_only"]), ("early_fusion", ["both"]),
                        ("dropout_fusion", ["both", "radar_only", "optical_only"]),
                        ("aware_fusion", ["both", "radar_only", "optical_only"])):
        for st in states:
            for sev in np.round(np.arange(0, 1.01, 0.2), 1):
                rows.append(dict(model=mdl, state=st, severity=sev, IoU=float(rng.uniform(0.3, 0.8)),
                                 miss_rate_chip_macro=float(rng.uniform(0.05, 0.4)), FP=int(rng.integers(100, 5000))))
    pd.DataFrame(rows).to_csv(A / "results/cloud_robustness_summary_PLACEHOLDER.csv", index=False)
    pd.DataFrame([dict(model=m, state="both", params=0, IoU_mean=0.0, IoU_sd=0.0)
                  for m in ("radar_unet", "optical_unet", "early_fusion", "dropout_fusion", "aware_fusion")]
                 ).to_csv(A / "results/main_evaluation_summary_PLACEHOLDER.csv", index=False)
    pd.DataFrame([dict(variant="aware_no_gate", IoU_mean=0.0)]).to_csv(A / "results/ablation_summary_PLACEHOLDER.csv", index=False)
    pd.DataFrame([dict(state="both", temperature=1.0, ece_raw=0.0, ece_ts=0.0)]).to_csv(
        A / "results/calibration_tables_PLACEHOLDER.csv", index=False)
    fig, ax = plt.subplots(figsize=(4, 2.5)); ax.text(0.5, 0.5, "PLACEHOLDER figure", ha="center"); ax.axis("off")
    fig.savefig(A / "results/gate_weights_PLACEHOLDER.png", dpi=80); plt.close(fig)
    print("placeholder assets written to", A)


if __name__ == "__main__":
    main()
