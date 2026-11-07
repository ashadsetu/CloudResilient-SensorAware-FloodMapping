"""Uploaded scenes through the real model: all sensor states, cloud invariance, analysis without labels."""
import numpy as np

from src import uploads as U
from src.inference import analyze, predict, register_scene
from src.model import STATES
from tests.test_uploads import CID, _pair, _read


def _scene(files=None):
    (sc,) = U.parse_files(files or _pair())
    return register_scene(sc), sc


def test_uploaded_geotiff_predicts_in_every_state():
    cid, sc = _scene()
    for s in STATES:
        p = predict(cid, s, 0.3)
        assert p["prob"].shape == (128, 128) and np.isfinite(p["prob"]).all()
        assert ((p["prob"] >= 0) & (p["prob"] <= 1)).all()


def test_uploaded_radar_only_is_cloud_invariant_and_optical_is_not():
    cid, _ = _scene()
    assert np.array_equal(predict(cid, "radar_only", 0.0)["logits"], predict(cid, "radar_only", 0.9)["logits"])
    assert not np.array_equal(predict(cid, "optical_only", 0.0)["logits"], predict(cid, "optical_only", 0.9)["logits"])


def test_single_sensor_uploads_run_only_their_state():
    cid, sc = _scene([_read(f"{CID}_S1Hand.tif")])
    r = analyze(cid, "radar_only", 0.6, 0.10)
    assert list(r["per"]) == ["radar_only"] and not r["has_label"]
    p = predict(cid, "radar_only", 0.0)
    assert np.isfinite(p["prob"]).all() and np.all(p["gates"][:, 1] == 0)


def test_analysis_of_unlabelled_scene_reports_no_accuracy():
    cid, sc = _scene()
    r = analyze(cid, "both", 0.6, 0.10)
    assert not r["has_label"] and np.isnan(r["sel"]["m"]["iou"]) and np.isnan(r["sel"]["m"]["miss"])
    assert 0.0 <= r["share"] <= 1.0 and r["area_km2"] > 0
    assert "No ground-truth label" in r["explanation"]
