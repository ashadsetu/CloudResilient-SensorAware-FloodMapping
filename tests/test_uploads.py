"""Upload parsing (GeoTIFF + npz). Torch-free except the resize-parity test, which is skipped without torch."""
import io
from pathlib import Path

import numpy as np
import pytest

from src import uploads as U

SAMPLES = Path(__file__).resolve().parent.parent / "assets" / "sample_tiffs"
CID = "Ghana_141271"


def _read(name):
    return (name, (SAMPLES / name).read_bytes())


def _pair(cid=CID):
    return [_read(f"{cid}_S1Hand.tif"), _read(f"{cid}_S2Hand.tif")]


def _npz(**arrays):
    b = io.BytesIO()
    np.savez(b, **arrays)
    return b.getvalue()


def test_geotiff_pair_gives_model_ready_chip():
    (sc,) = U.parse_files(_pair())
    assert sc["s1"].shape == (2, 128, 128) and sc["s1"].dtype == np.float32
    assert sc["s2"].shape == (13, 128, 128) and sc["s2"].dtype == np.float32
    assert sc["y"].shape == (128, 128) and not sc["has_label"]
    assert 0.0 <= sc["s1"].min() and sc["s1"].max() <= 1.0
    assert 0.0 <= sc["s2"].min() and sc["s2"].max() <= 1.5
    assert U.available_states(sc) == ["both", "radar_only", "optical_only"]
    assert sc["chip_id"].startswith("up-")


def test_cell_area_is_plausible_for_a_10m_512px_chip():
    (sc,) = U.parse_files(_pair())
    assert 1400 < sc["area_m2"] < 1700            # (4 x ~10 m)^2 = ~1600 m^2 per 128-px cell


def test_single_sensor_scenes_restrict_states():
    (r,) = U.parse_files([_read(f"{CID}_S1Hand.tif")])
    (o,) = U.parse_files([_read(f"{CID}_S2Hand.tif")])
    assert U.available_states(r) == ["radar_only"] and not r["has_optical"]
    assert U.available_states(o) == ["optical_only"] and not o["has_radar"]


def test_files_with_unrelated_names_are_paired_by_band_count():
    (sc,) = U.parse_files([("a_S1Hand.tif", _pair()[0][1]), ("b_S2Hand.tif", _pair()[1][1])])
    assert sc["has_radar"] and sc["has_optical"] and any("Paired" in w for w in sc["warnings"])


def test_two_different_chips_give_two_scenes():
    scenes = U.parse_files(_pair("Ghana_141271") + _pair("Ghana_1078550"))
    assert len(scenes) == 2 and scenes[0]["chip_id"] != scenes[1]["chip_id"]


def test_npz_in_demo_format_roundtrips():
    (sc,) = U.parse_files(_pair())
    y = np.zeros((128, 128), np.uint8); y[40:80, 30:90] = 1; y[:4] = 255
    (n,) = U.parse_files([("chip.npz", _npz(s1=sc["s1"], s2=sc["s2"], y=y, area_m2=1500.0))])
    assert np.array_equal(n["s1"], sc["s1"]) and np.array_equal(n["s2"], sc["s2"])
    assert n["has_label"] and n["area_m2"] == 1500.0 and np.array_equal(n["y"], y)


def test_raw_npz_matches_geotiff_path():
    s1, _ = U.read_tiff(_pair()[0][1])
    s2, _ = U.read_tiff(_pair()[1][1])
    (a,) = U.parse_files(_pair())
    (b,) = U.parse_files([("raw.npz", _npz(s1=s1, s2=s2))])      # raw dB / raw digital numbers
    assert np.allclose(a["s1"], b["s1"]) and np.allclose(a["s2"], b["s2"])


def test_label_tif_is_used_for_scoring():
    lab = np.zeros((1, 512, 512), np.int16); lab[0, 100:300, 100:300] = 1; lab[0, :20] = -1
    import tifffile
    buf = io.BytesIO(); tifffile.imwrite(buf, lab)
    (sc,) = U.parse_files(_pair() + [(f"{CID}_LabelHand.tif", buf.getvalue())])
    assert sc["has_label"] and set(np.unique(sc["y"]).tolist()) == {0, 1, 255}


@pytest.mark.parametrize("files, text", [
    ([], "No files"),
    ([("x.zip", b"123")], "Unsupported"),
    ([("x.tif", b"not a tiff")], "cannot read"),
    ([("x.npz", _npz(foo=np.ones(3)))], "needs an 's1'"),
])
def test_bad_files_give_readable_errors(files, text):
    with pytest.raises(U.UploadError, match=text):
        U.parse_files(files)


def test_wrong_content_is_rejected_not_silently_processed():
    rng = np.random.default_rng(0)
    with pytest.raises(U.UploadError, match="all zero"):
        U.build_scene(s2=np.zeros((13, 64, 64), np.float32))
    with pytest.raises(U.UploadError, match="already scaled"):
        U.build_scene(s2=rng.random((13, 64, 64)).astype(np.float32))
    with pytest.raises(U.UploadError, match="dB"):
        U.build_scene(s1=(rng.random((2, 64, 64)) * 1000 + 100).astype(np.float32))
    with pytest.raises(U.UploadError, match="2 bands"):
        U.build_scene(s1=np.zeros((3, 64, 64), np.float32))
    with pytest.raises(U.UploadError, match="13"):
        U.build_scene(s2=np.zeros((4, 64, 64), np.float32))


def test_resize_matches_torch():
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F
    rng = np.random.default_rng(1)
    for h, w in ((512, 512), (300, 300), (128, 128), (200, 333)):
        a = rng.random((3, h, w)).astype(np.float32)
        ref = F.interpolate(torch.from_numpy(a)[None], size=(128, 128), mode="bilinear", align_corners=False)[0].numpy()
        assert np.allclose(U.resize_bilinear(a), ref, atol=1e-5)
        lab = rng.integers(0, 3, (1, h, w)).astype(np.float32)
        ref = F.interpolate(torch.from_numpy(lab)[None], size=(128, 128), mode="nearest")[0].numpy()
        assert np.array_equal(U.resize_nearest(lab), ref)
