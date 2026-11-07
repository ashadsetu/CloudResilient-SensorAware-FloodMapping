import time

import numpy as np

from src.inference import analyze, lookup_precomputed, predict, threshold_info
from src.metrics import iou, miss_fraction, counts, error_map
from src.model import STATES


def test_predict_smoke_all_chips_all_states(manifest):
    for m in manifest:
        for s in STATES:
            p = predict(m["chip_id"], s, 0.5)
            assert p["prob"].shape == (128, 128) and np.isfinite(p["prob"]).all()
            assert ((p["prob"] >= 0) & (p["prob"] <= 1)).all()


def test_prediction_under_two_seconds(model, chip):
    t = time.perf_counter()
    predict(chip["chip_id"], "both", 0.37)
    assert time.perf_counter() - t < 2.0


def test_analyze_all_chips(manifest):
    for m in manifest:
        r = analyze(m["chip_id"], "both", 0.6, 0.10)
        assert r["explanation"] and 0.0 <= r["share"] <= 1.0


def test_metrics_valid_pixels_only_and_undefined_cases():
    y = np.array([[1, 1, 0, 255]], np.uint8)
    pred = np.array([[1, 0, 1, 1]], bool)
    c = counts(pred, y)
    assert c == dict(tp=1, fp=1, fn=1, tn=0)           # the ignore pixel is not counted
    assert iou(c) == 1 / 3 and miss_fraction(c) == 0.5
    no_flood = counts(np.zeros((1, 4), bool), np.zeros((1, 4), np.uint8))
    assert np.isnan(miss_fraction(no_flood)) and np.isnan(iou(no_flood))
    assert error_map(pred, y)[0, 3, 3] == 0              # ignore pixel transparent


def test_threshold_info_fallback():
    assert threshold_info(0.33, "both")["feasible"] is None


def test_lookup_precomputed_layouts():
    flat = {"precomputed_iou": {"iou_both_cloud0.6": 0.77, "iou_radar_only": 0.5}}
    assert lookup_precomputed(flat, "both", 0.6) == 0.77
    assert lookup_precomputed(flat, "radar_only", 0.9) == 0.5       # radar-only ignores cloud
    assert lookup_precomputed(flat, "optical_only", 0.6) is None     # not stored -> computed live
    assert lookup_precomputed({"precomputed_iou": 0.8}, "both", 0.0) == 0.8
    assert lookup_precomputed({}, "both", 0.0) is None
