"""Turn user-supplied files into the chip dict the model expects.

Supported inputs
----------------
* **Sen1Floods11-style GeoTIFFs**: ``<id>_S1Hand.tif`` (2 bands VV, VH, backscatter in dB) and/or
  ``<id>_S2Hand.tif`` (13 bands, reflectance x 10000, Sen1Floods11 band order), optionally
  ``<id>_LabelHand.tif`` (1 band: 1 flood, 0 land, -1 no data). One sensor alone is fine.
* **.npz** in the demo-chip format (``s1``, ``s2``, optional ``y`` and ``area_m2``). Arrays that are
  already normalised (the demo-chip format) are used as they are; raw dB / raw digital numbers are
  normalised exactly like the training notebook.

The preprocessing mirrors the training notebook (section 5.1, ``PREPROC`` v2.0): S1 is clipped to
[-35, 5] dB and scaled to 0-1, S2 is divided by 10000 and clipped to [0, 1.5], images are resized to
128x128 with bilinear interpolation (``align_corners=False``), labels with nearest-neighbour.

This module deliberately has no torch/streamlit dependency, so it can be unit-tested anywhere.
Everything happens in memory: nothing is written to disk or sent over the network.
"""
from __future__ import annotations

import hashlib
import io
import math
import re
from dataclasses import dataclass

import numpy as np
import tifffile

IMG_SIZE = 128
S1_CLIP = (-35.0, 5.0)
S2_SCALE = 10000.0
S2_CLIP_MAX = 1.5
IGNORE_INDEX = 255
MAX_FILE_BYTES = 150 * 1024 * 1024      # per file
MAX_PIXELS = 4096 * 4096                # per band; larger rasters must be cropped first
S1_BANDS, S2_BANDS = 2, 13


class UploadError(ValueError):
    """A problem with the uploaded files that the user can fix (message is shown in the UI)."""


# ----------------------------------------------------------------------------- resizing
def _axis_weights(n_in: int, n_out: int):
    """Source indices/weights of torch's bilinear interpolation with align_corners=False."""
    scale = n_in / n_out
    src = np.maximum((np.arange(n_out, dtype=np.float64) + 0.5) * scale - 0.5, 0.0)
    i0 = np.minimum(np.floor(src).astype(np.int64), n_in - 1)
    i1 = np.minimum(i0 + 1, n_in - 1)
    return i0, i1, (src - i0).astype(np.float32)


def resize_bilinear(arr: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    """[C,H,W] -> [C,size,size]; equals ``F.interpolate(mode='bilinear', align_corners=False)``."""
    arr = np.asarray(arr, np.float32)
    _, h, w = arr.shape
    y0, y1, wy = _axis_weights(h, size)
    x0, x1, wx = _axis_weights(w, size)
    rows = arr[:, y0, :] * (1 - wy)[None, :, None] + arr[:, y1, :] * wy[None, :, None]
    return (rows[:, :, x0] * (1 - wx)[None, None, :] + rows[:, :, x1] * wx[None, None, :]).astype(np.float32)


def resize_nearest(arr: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    """[C,H,W] -> [C,size,size]; equals ``F.interpolate(mode='nearest')`` (used for labels)."""
    _, h, w = arr.shape
    yi = np.minimum((np.arange(size) * (h / size)).astype(np.int64), h - 1)
    xi = np.minimum((np.arange(size) * (w / size)).astype(np.int64), w - 1)
    return arr[:, yi][:, :, xi]


# ----------------------------------------------------------------------------- GeoTIFF reading
@dataclass
class GeoInfo:
    sx: float                 # pixel size in x (degrees or metres)
    sy: float
    top_y: float              # y coordinate of the top edge (degrees or metres)
    geographic: bool          # True: degrees (e.g. EPSG:4326), False: metres


def _geo_info(page) -> GeoInfo | None:
    tags = page.tags
    scale, tie = tags.get(33550), tags.get(33922)          # ModelPixelScale, ModelTiepoint
    if scale is None or tie is None:
        return None
    try:
        sx, sy = abs(float(scale.value[0])), abs(float(scale.value[1]))
        i, j, _, _, y, _ = (float(v) for v in tie.value[:6])
    except Exception:
        return None
    geographic = None
    keys = tags.get(34735)                                  # GeoKeyDirectory: header(4) + entries of 4 shorts
    if keys is not None:
        v = list(keys.value)
        for k in range(4, len(v) - 3, 4):
            if v[k] == 1024:                                # GTModelTypeGeoKey: 1 projected, 2 geographic
                geographic = v[k + 3] == 2
    if geographic is None:
        geographic = sx < 0.1
    return GeoInfo(sx, sy, y + j * sy, geographic)


def read_tiff(data: bytes, name: str = "file.tif"):
    """Return (array [bands,H,W], GeoInfo | None) from the bytes of a TIFF."""
    if len(data) > MAX_FILE_BYTES:
        raise UploadError(f"{name}: file is larger than {MAX_FILE_BYTES // 2**20} MB.")
    try:
        with tifffile.TiffFile(io.BytesIO(data)) as tf:
            series = tf.series[0]
            arr, axes = series.asarray(), series.axes
            geo = _geo_info(tf.pages[0])
    except Exception as e:
        raise UploadError(f"{name}: cannot read this TIFF ({type(e).__name__}: {e}). "
                          "If it uses an unusual compression, re-save it as plain/deflate GeoTIFF.") from e
    if arr.ndim == 2:
        arr = arr[None]
    elif arr.ndim == 3 and axes.endswith("S"):              # pixel-interleaved: H,W,C -> C,H,W
        arr = np.moveaxis(arr, -1, 0)
    if arr.ndim != 3:
        raise UploadError(f"{name}: expected a [bands, rows, cols] raster, got shape {arr.shape}.")
    if arr.shape[1] * arr.shape[2] > MAX_PIXELS:
        raise UploadError(f"{name}: raster is {arr.shape[2]}x{arr.shape[1]} px; please crop to a chip "
                          "(Sen1Floods11 chips are 512x512 px, about 5x5 km).")
    return arr, geo


def cell_area_m2(geo: GeoInfo | None, h: int, w: int, size: int = IMG_SIZE) -> float:
    """Approximate ground area (m^2) of one resized (size x size) cell; NaN when there is no geo-referencing."""
    if geo is None:
        return float("nan")
    if geo.geographic:
        lat = geo.top_y - geo.sy * h / 2
        mx, my = geo.sx * 111_320.0 * math.cos(math.radians(lat)), geo.sy * 111_132.0
    else:
        mx, my = geo.sx, geo.sy
    return float((mx * w / size) * (my * h / size))


# ----------------------------------------------------------------------------- normalisation
def normalise_s1(raw: np.ndarray, name: str = "radar") -> np.ndarray:
    """Raw dB [2,H,W] -> 0-1 (same as the notebook)."""
    lo, hi = S1_CLIP
    a = np.asarray(raw, np.float32)
    med = float(np.nanmedian(a)) if np.isfinite(a).any() else float("nan")
    if not np.isfinite(med):
        raise UploadError(f"{name}: radar band contains no valid values.")
    if not (-60.0 <= med <= 10.0):
        raise UploadError(f"{name}: radar values look wrong (median {med:.1f}). Expected Sentinel-1 backscatter "
                          "in dB (typically -30 to 0), not linear power or digital numbers.")
    a = np.clip(np.nan_to_num(a, nan=lo, posinf=hi, neginf=lo), lo, hi)
    return ((a - lo) / (hi - lo)).astype(np.float32)


def normalise_s2(raw: np.ndarray, name: str = "optical") -> np.ndarray:
    """Raw reflectance x 10000 [13,H,W] -> reflectance clipped to [0, 1.5]."""
    a = np.nan_to_num(np.asarray(raw, np.float32), nan=0.0, posinf=S2_SCALE, neginf=0.0)
    step = max(1, max(a.shape[1:]) // 512)                  # subsample big rasters; only a sanity check
    p99 = float(np.percentile(a[:, ::step, ::step], 99))
    if p99 <= 0:
        raise UploadError(f"{name}: optical bands are all zero / no-data.")
    if p99 <= 2.0:
        raise UploadError(f"{name}: optical values look already scaled to 0-1 (99th percentile {p99:.2f}). "
                          "The model expects Sentinel-2 reflectance multiplied by 10000 (as in Sen1Floods11); "
                          "multiply by 10000 and re-upload, or supply a normalised .npz.")
    return np.clip(a / S2_SCALE, 0.0, S2_CLIP_MAX).astype(np.float32)


# ----------------------------------------------------------------------------- scene assembly
def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-._")[:24] or "scene"


def build_scene(*, s1=None, s2=None, label=None, area_m2=float("nan"), name="scene", source="upload",
                warnings=None, s1_norm=False, s2_norm=False) -> dict:
    """Assemble a chip dict. ``s1``/``s2`` are raw [bands,H,W] unless ``*_norm`` says they are normalised;
    ``label`` is [1,H,W] with 1 flood, 0 land, anything else = ignore."""
    warnings = list(warnings or [])
    if s1 is None and s2 is None:
        raise UploadError("Provide at least one sensor (radar S1 or optical S2).")
    if s1 is not None:
        if s1.shape[0] != S1_BANDS:
            raise UploadError(f"Radar needs {S1_BANDS} bands (VV, VH); found {s1.shape[0]}.")
        s1 = s1.astype(np.float32) if s1_norm else normalise_s1(s1, name)
    if s2 is not None:
        if s2.shape[0] != S2_BANDS:
            raise UploadError(f"Optical needs {S2_BANDS} Sentinel-2 bands (Sen1Floods11 order); found {s2.shape[0]}.")
        s2 = s2.astype(np.float32) if s2_norm else normalise_s2(s2, name)
    for arr in (s1, s2, label):
        if arr is not None and min(arr.shape[1:]) < 16:
            raise UploadError("Image is too small (< 16 px on a side).")
    if s1 is not None and s2 is not None and s1.shape[1:] != s2.shape[1:]:
        warnings.append(f"Radar is {s1.shape[2]}x{s1.shape[1]} px but optical is {s2.shape[2]}x{s2.shape[1]} px; "
                        "both were resized independently, assuming they cover the same footprint.")
    for arr in (s1, s2):
        if arr is not None:
            h, w = arr.shape[1:]
            if abs(h - w) / max(h, w) > 0.01:
                warnings.append(f"Image is not square ({w}x{h}); it was stretched to {IMG_SIZE}x{IMG_SIZE}.")
                break
    for arr in (s1, s2):
        if arr is not None and min(arr.shape[1:]) > 1024:
            warnings.append("Large raster: it is heavily downsampled to 128x128 without anti-aliasing, "
                            "so results may be unreliable. The model was trained on ~5 km chips.")
            break

    r1 = resize_bilinear(s1) if s1 is not None else np.zeros((S1_BANDS, IMG_SIZE, IMG_SIZE), np.float32)
    r2 = resize_bilinear(s2) if s2 is not None else np.zeros((S2_BANDS, IMG_SIZE, IMG_SIZE), np.float32)
    if s1 is not None and not np.any(r1):
        raise UploadError(f"{name}: the radar image is all zeros (blank or no-data tile).")
    if s2 is not None and not np.any(r2):
        raise UploadError(f"{name}: the optical image is all zeros (blank or no-data tile).")

    has_label = label is not None
    if has_label:
        y = resize_nearest(np.where(label == 1, 1, np.where(label == 0, 0, IGNORE_INDEX)).astype(np.uint8))[0]
        if not np.any(y != IGNORE_INDEX):
            warnings.append("The label has no valid pixels, so it was ignored.")
            has_label, y = False, np.zeros((IMG_SIZE, IMG_SIZE), np.uint8)
    else:
        y = np.zeros((IMG_SIZE, IMG_SIZE), np.uint8)          # every pixel counts as valid; no metrics are shown

    digest = hashlib.sha256(r1.tobytes() + r2.tobytes() + y.tobytes()).hexdigest()[:8]
    return dict(s1=r1, s2=r2, y=y.astype(np.uint8), area_m2=float(area_m2), has_label=has_label,
                has_radar=s1 is not None, has_optical=s2 is not None, chip_id=f"up-{_slug(name)}-{digest}",
                event="upload", split="upload", title=name, source=source, warnings=warnings)


def _classify(name: str, arr: np.ndarray) -> str:
    n = arr.shape[0]
    if n == S1_BANDS:
        return "s1"
    if n == S2_BANDS:
        return "s2"
    if n == 1:
        return "label"
    raise UploadError(f"{name}: {n} bands. Expected 2 (radar VV,VH), 13 (Sentinel-2) or 1 (flood label).")


_SUFFIX = re.compile(r"[_-]?(s1hand|s2hand|labelhand|s1|s2|label|radar|optical)$", re.I)


def _key(name: str) -> str:
    stem = re.sub(r"\.(tiff?|npz)$", "", name, flags=re.I)
    return _SUFFIX.sub("", stem) or stem


def scene_from_npz(name: str, data: bytes) -> dict:
    if len(data) > MAX_FILE_BYTES:
        raise UploadError(f"{name}: file is larger than {MAX_FILE_BYTES // 2**20} MB.")
    try:
        z = np.load(io.BytesIO(data), allow_pickle=False)
        keys = set(z.files)
    except Exception as e:
        raise UploadError(f"{name}: cannot read this .npz ({type(e).__name__}: {e}).") from e
    if not keys & {"s1", "s2"}:
        raise UploadError(f"{name}: the .npz needs an 's1' [2,H,W] and/or an 's2' [13,H,W] array "
                          f"(found keys: {sorted(keys)}).")
    s1 = np.asarray(z["s1"], np.float32) if "s1" in keys else None
    s2 = np.asarray(z["s2"], np.float32) if "s2" in keys else None
    for k, a in (("s1", s1), ("s2", s2)):
        if a is not None and a.ndim != 3:
            raise UploadError(f"{name}: '{k}' must be [bands,H,W], got shape {a.shape}.")
    # already-normalised arrays (the demo-chip format) are kept; anything else is treated as raw
    s1_norm = s1 is not None and np.isfinite(s1).all() and float(s1.min()) >= 0.0 and float(s1.max()) <= 1.0001
    s2_norm = s2 is not None and np.isfinite(s2).all() and float(s2.min()) >= 0.0 and float(s2.max()) <= S2_CLIP_MAX + 1e-4
    label = None
    if "y" in keys:
        y = np.asarray(z["y"])
        label = (y[None] if y.ndim == 2 else y[:1]).astype(np.int16)
        label = np.where((label == 0) | (label == 1), label, -1)
    area = float(z["area_m2"]) if "area_m2" in keys else float("nan")
    return build_scene(s1=s1, s2=s2, label=label, area_m2=area, name=re.sub(r"\.npz$", "", name, flags=re.I),
                       source=".npz", s1_norm=s1_norm, s2_norm=s2_norm)


def scenes_from_tiffs(files: list[tuple[str, bytes]]) -> list[dict]:
    groups: dict[str, dict] = {}
    for name, data in files:
        arr, geo = read_tiff(data, name)
        kind = _classify(name, arr)
        g = groups.setdefault(_key(name), {})
        if kind in g:
            raise UploadError(f"Two {kind.upper()} files for '{_key(name)}' ({g[kind][0]} and {name}).")
        g[kind] = (name, arr, geo)
    # lone S1 + lone S2 with different names: pair them (the usual 'I dropped two files' case)
    notes: dict[str, str] = {}
    s1_only = [k for k, g in groups.items() if "s1" in g and "s2" not in g]
    s2_only = [k for k, g in groups.items() if "s2" in g and "s1" not in g]
    if len(s1_only) == 1 and len(s2_only) == 1 and len(groups) == 2:
        a, b = s1_only[0], s2_only[0]
        groups = {a: {**groups[a], **groups[b]}}
        notes[a] = f"Paired '{groups[a]['s1'][0]}' with '{groups[a]['s2'][0]}' by band count (names did not match)."
    scenes = []
    for key, g in groups.items():
        if "label" in g and "s1" not in g and "s2" not in g:
            raise UploadError(f"'{g['label'][0]}' is a label with no matching radar/optical file.")
        ref = g.get("s1") or g.get("s2")
        geo, (h, w) = ref[2] or (g.get("s2") or ref)[2], ref[1].shape[1:]
        warnings = [notes[key]] if key in notes else []
        if "s1" in g and "s2" not in g:
            warnings.append("Only radar was provided: only the 'Radar only' sensor state is available.")
        if "s2" in g and "s1" not in g:
            warnings.append("Only optical was provided: only the 'Optical only' sensor state is available.")
        scenes.append(build_scene(
            s1=g["s1"][1] if "s1" in g else None, s2=g["s2"][1] if "s2" in g else None,
            label=g["label"][1] if "label" in g else None, area_m2=cell_area_m2(geo, h, w), name=key,
            source="GeoTIFF", warnings=warnings))
    return scenes


def parse_files(files: list[tuple[str, bytes]]) -> list[dict]:
    """Entry point for the UI: list of (filename, bytes) -> list of scene dicts (one per scene found)."""
    if not files:
        raise UploadError("No files provided.")
    npz = [f for f in files if f[0].lower().endswith(".npz")]
    tif = [f for f in files if f[0].lower().endswith((".tif", ".tiff"))]
    other = [f[0] for f in files if f not in npz and f not in tif]
    if other:
        raise UploadError(f"Unsupported file type: {', '.join(other)}. Use .tif/.tiff or .npz.")
    scenes = [scene_from_npz(n, d) for n, d in npz]
    if tif:
        scenes += scenes_from_tiffs(tif)
    return scenes


def available_states(scene: dict) -> list[str]:
    """Sensor states this scene supports (demo chips have no flags, so all three)."""
    r, o = scene.get("has_radar", True), scene.get("has_optical", True)
    return (["both"] if r and o else []) + (["radar_only"] if r else []) + (["optical_only"] if o else [])


if __name__ == "__main__":      # python -m src.uploads  S1.tif S2.tif [Label.tif]  out.npz
    import sys
    if len(sys.argv) < 3:
        sys.exit("usage: python -m src.uploads <S1.tif> [<S2.tif>] [<Label.tif>] <out.npz>")
    *ins, out = sys.argv[1:]
    sc = parse_files([(p.split("/")[-1], open(p, "rb").read()) for p in ins])[0]
    np.savez_compressed(out, s1=sc["s1"], s2=sc["s2"], y=sc["y"], area_m2=sc["area_m2"], chip_id=sc["chip_id"],
                        event="upload", split="upload")
    print(f"wrote {out}: {sc['chip_id']} (radar={sc['has_radar']}, optical={sc['has_optical']}, label={sc['has_label']})")
