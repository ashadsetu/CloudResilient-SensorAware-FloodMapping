import numpy as np
import torch

from src.inference import asset_path, run_model
from src.model import STATES, SensorAwareFusionUNet


def test_model_loads_strict():
    ck = torch.load(asset_path("models/aware_fusion.pt"), map_location="cpu", weights_only=False)
    cfg = ck["config"]
    m = SensorAwareFusionUNet(cfg["base"], cfg["s2_channels"], cfg["drop_p"], **cfg.get("flags", {}))
    m.load_state_dict(ck["model_state_dict"], strict=True)


def test_radar_only_bit_identical_across_cloud(model, chip):
    a = run_model(model, chip["s1"], chip["s2"], "radar_only", 0.0, chip["chip_id"])["logits"]
    b = run_model(model, chip["s1"], chip["s2"], "radar_only", 0.9, chip["chip_id"])["logits"]
    assert np.array_equal(a, b)


def test_cloud_changes_optical_and_fused(model, chip):
    for s in ("optical_only", "both"):
        a = run_model(model, chip["s1"], chip["s2"], s, 0.0, chip["chip_id"])["logits"]
        b = run_model(model, chip["s1"], chip["s2"], s, 0.9, chip["chip_id"])["logits"]
        assert not np.array_equal(a, b)


def test_optical_only_invariant_to_radar(model, chip):
    a = run_model(model, chip["s1"], chip["s2"], "optical_only", 0.5, chip["chip_id"])["logits"]
    b = run_model(model, np.random.default_rng(0).random(chip["s1"].shape).astype(np.float32), chip["s2"],
                  "optical_only", 0.5, chip["chip_id"])["logits"]
    assert np.array_equal(a, b)


def test_gate_weights_sum_to_one_and_missing_is_zero(model, chip):
    for s, (r, o) in STATES.items():
        g = run_model(model, chip["s1"], chip["s2"], s, 0.3, chip["chip_id"])["gates"]   # [scale, (radar, optical)]
        assert np.allclose(g.sum(1), 1.0, atol=1e-5)
        if not r:
            assert np.all(g[:, 0] == 0)
        if not o:
            assert np.all(g[:, 1] == 0)
