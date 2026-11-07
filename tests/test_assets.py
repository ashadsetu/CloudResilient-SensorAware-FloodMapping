import json

import numpy as np

from src.inference import TARGETS, asset_path, load_chip, load_thresholds


def test_manifest_chips_load_with_documented_shapes(manifest):
    assert manifest, "manifest is empty"
    for m in manifest:
        c = load_chip(m["chip_id"])
        assert c["s1"].shape == (2, 128, 128) and c["s1"].dtype == np.float32
        assert c["s2"].shape == (13, 128, 128) and c["s2"].dtype == np.float32
        assert c["y"].shape == (128, 128) and set(np.unique(c["y"]).tolist()) <= {0, 1, 255}
        assert c["area_m2"] > 0


def test_thresholds_parse():
    assert json.loads(asset_path("thresholds.json").read_text())
    t = load_thresholds()
    for tg in TARGETS:
        for s in ("both", "radar_only", "optical_only"):
            e = t[f"{tg:.2f}"][s]
            assert 0.0 <= e["threshold"] <= 1.0 and "feasible" in e


def test_no_blank_demo_chips(manifest):
    """A sensor array of all zeros means a blank / no-data tile: it would make a meaningless showcase."""
    for m in manifest:
        c = load_chip(m["chip_id"])
        assert np.any(c["s1"]) and np.any(c["s2"]), f"{m['chip_id']}: an all-zero sensor array"
