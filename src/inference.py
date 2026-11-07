"""Model loading, prediction, thresholds. CPU only, no network access."""
from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

try:
    import streamlit as st
    cache_resource, cache_data = st.cache_resource, st.cache_data
except Exception:  # allows importing without Streamlit (tests, notebooks)
    def cache_resource(f=None, **_):
        return f if f is not None else (lambda g: g)
    cache_data = cache_resource

from .cloud import BASE_SEED, eval_cloud_s2
from .model import STATES, SensorAwareFusionUNet
from .uploads import available_states

ASSETS = Path(__file__).resolve().parent.parent / "assets"
TARGETS = (0.05, 0.10, 0.20)
STATE_LABELS = {"both": "Both sensors", "radar_only": "Radar only", "optical_only": "Optical only"}


def asset_path(rel: str) -> Path:
    """Real asset if present, otherwise its *_PLACEHOLDER* stand-in (see README)."""
    p = ASSETS / rel
    if p.exists():
        return p
    alt = p.with_name(f"{p.stem}_PLACEHOLDER{p.suffix}")
    return alt if alt.exists() else p


def using_placeholders() -> bool:
    return any(not (ASSETS / r).exists() for r in ("models/aware_fusion.pt", "manifest.json", "thresholds.json"))


def _read_json(rel: str, default=None):
    p = asset_path(rel)
    try:
        return json.loads(p.read_text())
    except Exception:
        return default


@cache_data
def load_manifest() -> list:
    return _read_json("manifest.json", []) or []


@cache_data
def load_thresholds() -> dict:
    return _read_json("thresholds.json", {}) or {}


@cache_data
def load_model_card() -> dict:
    return _read_json("model_card.json", {}) or {}


def threshold_info(target: float, state: str) -> dict:
    """Entry of thresholds.json for (target, state); falls back to 0.5 / feasible=None when missing."""
    t = load_thresholds()
    try:
        return dict(t[f"{target:.2f}"][state])
    except Exception:
        return dict(threshold=0.5, feasible=None, n_cal_flood_chips=0, n_cal_chips=0)


@cache_resource
def load_model() -> SensorAwareFusionUNet:
    ck = torch.load(asset_path("models/aware_fusion.pt"), map_location="cpu", weights_only=False)
    cfg = ck["config"]
    model = SensorAwareFusionUNet(cfg["base"], cfg["s2_channels"], cfg["drop_p"], **cfg.get("flags", {}))
    model.load_state_dict(ck["model_state_dict"], strict=True)
    return model.eval()


_SCENES: "OrderedDict[str, dict]" = OrderedDict()      # uploaded scenes, RAM only (content-hash ids), newest last
_MAX_SCENES = 16


def register_scene(scene: dict) -> str:
    """Make an uploaded scene (see ``uploads.parse_files``) available to ``predict``/``analyze``; returns its id.
    Scenes live only in this process's memory (nothing is written to disk); the 16 most recent are kept."""
    cid = scene["chip_id"]
    _SCENES[cid] = scene
    _SCENES.move_to_end(cid)
    while len(_SCENES) > _MAX_SCENES:
        _SCENES.popitem(last=False)
    return cid


@cache_data
def _load_demo_chip(chip_id: str) -> dict:
    z = np.load(asset_path(f"demo_chips/{chip_id}.npz"), allow_pickle=False)
    return dict(s1=z["s1"].astype(np.float32), s2=z["s2"].astype(np.float32), y=z["y"].astype(np.uint8),
                area_m2=float(z["area_m2"]), chip_id=str(z["chip_id"]), event=str(z["event"]), split=str(z["split"]),
                has_label=True, has_radar=True, has_optical=True)


def load_chip(chip_id: str) -> dict:
    """A demo chip from assets/demo_chips, or an uploaded scene registered with ``register_scene``."""
    if chip_id in _SCENES:
        return _SCENES[chip_id]
    if chip_id.startswith("up-"):
        raise KeyError("This uploaded scene is no longer in memory (the app restarted or newer uploads replaced it). "
                       "Please upload the files again.")
    return _load_demo_chip(chip_id)


def optical_as_seen(chip: dict, state: str, severity: float):
    """Optical array exactly as the model receives it (None when optical is unavailable)."""
    if STATES[state][1] == 0:
        return None
    return eval_cloud_s2(chip["s2"], chip["chip_id"], float(severity), BASE_SEED)


def run_model(model, s1, s2, state: str, severity: float, chip_id: str) -> dict:
    avail = STATES[state]
    radar = np.asarray(s1, np.float32) * avail[0]
    opt = eval_cloud_s2(s2, chip_id, float(severity), BASE_SEED) if avail[1] else np.zeros_like(s2)
    opt = np.asarray(opt, np.float32) * avail[1]
    with torch.no_grad():
        logits, aux = model(torch.from_numpy(radar)[None], torch.from_numpy(opt)[None],
                            torch.tensor([avail], dtype=torch.float32))
    gates = [g[0].mean(dim=(1, 2)).numpy() for g in aux["gates"]]       # per scale: [radar, optical]
    lg = logits[0, 0].numpy()
    return dict(logits=lg, prob=1.0 / (1.0 + np.exp(-np.clip(lg, -50, 50))), gates=np.stack(gates))


@cache_data(show_spinner=False, max_entries=256)
def predict(chip_id: str, state: str, severity: float) -> dict:
    """Probabilities for one chip. Cloud is applied only when optical is available."""
    chip = load_chip(chip_id)
    return run_model(load_model(), chip["s1"], chip["s2"], state, severity, chip["chip_id"])


def lookup_precomputed(entry: dict, state: str, severity: float):
    """IoU stored in manifest.json['precomputed_iou'] (computed at the 10 % target in the notebook export).

    Supported layouts: flat keys such as 'iou_both_cloud0.6', 'iou_optical_only_cloud0.0', 'iou_radar_only'
    (radar-only is cloud-independent); a plain number (both sensors, clear sky); or {state: number | {severity: number}}.
    Returns None when the value is not stored, so the caller computes it live."""
    v = entry.get("precomputed_iou")
    if isinstance(v, (int, float)):
        return float(v) if state == "both" and severity == 0 else None
    if not isinstance(v, dict):
        return None
    key = "iou_radar_only" if state == "radar_only" else f"iou_{state}_cloud{severity:.1f}"
    if key in v:
        return float(v[key])
    if state in v:
        s = v[state]
        if isinstance(s, (int, float)):
            return float(s) if severity == 0 else None
        if isinstance(s, dict):
            for k in (f"{severity:.1f}", str(severity), f"{severity:.2f}"):
                if k in s:
                    return float(s[k])
    return None


def get_entry(chip_id: str) -> dict:
    if chip_id in _SCENES:
        sc = _SCENES[chip_id]
        return dict(chip_id=chip_id, title=sc.get("title", chip_id), event=sc.get("event", "upload"), split="upload", upload=True)
    return next((m for m in load_manifest() if m.get("chip_id") == chip_id), {"chip_id": chip_id})


def _fmt(v: float) -> str:
    return "undefined" if v != v else f"{v:.2f}"


def _pc(v: float) -> str:
    return "n/a" if v != v else f"{v * 100:.1f} %"


def analyze(chip_id: str, state: str, severity: float, target: float) -> dict:
    """Everything the results panel needs: masks/metrics for every available sensor state + explanation text."""
    from .metrics import agreement_iou, flooded_area_km2, flooded_share, scene_metrics

    chip, entry = load_chip(chip_id), get_entry(chip_id)
    states = available_states(chip)
    if state not in states:
        raise ValueError(f"“{STATE_LABELS[state]}” is not available for this scene "
                         f"(available: {', '.join(STATE_LABELS[x] for x in states)}).")
    per = {}
    for s in states:
        p = predict(chip_id, s, float(severity))
        ti = threshold_info(target, s)
        mask = p["prob"] >= float(ti["threshold"])
        m = scene_metrics(mask, chip["y"])
        if not chip.get("has_label", True):      # no ground truth: accuracy numbers would be meaningless
            m.update(iou=float("nan"), miss=float("nan"), fp=0, fn=0, tp=0, has_flood=False)
        per[s] = dict(prob=p["prob"], mask=mask, thr=ti, gates=p["gates"], m=m)
    sel = per[state]
    both_pair = {"radar_only", "both"} <= set(per)
    res = dict(chip=chip, chip_id=chip_id, entry=entry, state=state, severity=float(severity), target=float(target),
               per=per, sel=sel, has_label=bool(chip.get("has_label", True)),
               share=flooded_share(sel["mask"], chip["y"]),
               area_km2=flooded_area_km2(sel["mask"], chip["y"], chip["area_m2"]),
               agreement=agreement_iou(per["radar_only"]["mask"], per["both"]["mask"], chip["y"]) if both_pair else float("nan"))
    res["explanation"] = _explain(res)
    return res


def _explain(res: dict) -> str:
    """Plain-language summary built only from numbers computed here (or the manifest)."""
    from .metrics import flooded_share, scene_metrics

    chip, entry, per, state, sel = res["chip"], res["entry"], res["per"], res["state"], res["sel"]
    severity, target, cid = res["severity"], res["target"], res["chip_id"]
    tgt = int(round(target * 100))

    if not res["has_label"]:                         # uploaded scene without ground truth: no accuracy claims
        parts = "; ".join(f"{STATE_LABELS[s].lower()} {_pc(flooded_share(r['mask'], chip['y']))}" for s, r in per.items())
        txt = ("No ground-truth label was provided, so IoU and the missed-flood fraction cannot be computed and nothing "
               f"here says how accurate the map is. Share of the scene flagged as flooded at the {tgt} % target: {parts}.")
        if "optical_only" in per and severity > 0:
            pc = predict(cid, "optical_only", 0.0)
            s0 = flooded_share(pc["prob"] >= float(threshold_info(target, "optical_only")["threshold"]), chip["y"])
            cur = flooded_share(per["optical_only"]["mask"], chip["y"])
            txt += (f" With {severity * 100:.0f} % synthetic cloud the optical-only map flags {_pc(cur)} "
                    f"instead of {_pc(s0)} with clear sky.")
        if res["agreement"] == res["agreement"]:      # not NaN
            txt += f" The radar-only and fused maps agree with IoU {res['agreement']:.2f}."
        return txt

    names = {"both": "fused (both sensors)", "radar_only": "radar-only", "optical_only": "optical-only"}
    if set(per) != set(STATES):                      # labelled upload with a single sensor
        if not sel["m"]["has_flood"]:
            return ("This scene has no flood pixels, so IoU and the missed-flood fraction are undefined; any predicted "
                    f"flood here is a false alarm. At the {tgt} % target the {names[state]} map flags {int(sel['mask'].sum())} pixels.")
        return (f"The {names[state]} map scores IoU {_fmt(sel['m']['iou'])} and misses {_pc(sel['m']['miss'])} of the "
                f"flooded pixels on this scene (target {tgt} %, an average over scenes, not a per-scene guarantee).")

    clear = {}
    for s in ("optical_only", "both"):
        # manifest IoUs were exported at the 10 % target (verified to match live values); other targets are computed live
        pre = lookup_precomputed(entry, s, 0.0) if abs(float(target) - 0.10) < 1e-9 else None
        if pre is not None:
            clear[s] = pre
        else:
            pc = predict(cid, s, 0.0)
            clear[s] = scene_metrics(pc["prob"] >= float(threshold_info(target, s)["threshold"]), chip["y"])["iou"]
    i_opt, i_both, i_rad = (per[s]["m"]["iou"] for s in ("optical_only", "both", "radar_only"))
    if not per["both"]["m"]["has_flood"]:
        return ("This scene has no flood pixels, so IoU and the missed-flood fraction are undefined; "
                f"any predicted flood here is a false alarm. At the {tgt} % target the "
                f"{names[state]} map flags {int(sel['mask'].sum())} pixels.")
    if severity > 0:
        txt = (f"With {severity * 100:.0f} % synthetic cloud, optical-only IoU went from {_fmt(clear['optical_only'])} "
               f"(clear sky) to {_fmt(i_opt)}; the fused model scored {_fmt(i_both)} (clear sky: {_fmt(clear['both'])}) "
               f"and radar-only, which cloud cannot affect, scored {_fmt(i_rad)}.")
        rw_clear = float(predict(cid, "both", 0.0)["gates"][:, 0].mean())
        rw_cloud = float(per["both"]["gates"][:, 0].mean())
        if rw_cloud > rw_clear + 0.05:
            txt += f" The fused model shifted weight toward radar (mean radar gate {rw_clear:.2f} → {rw_cloud:.2f})."
        return txt
    return (f"With clear sky the three sensor states score IoU {_fmt(i_both)} (both), {_fmt(i_rad)} (radar only) and "
            f"{_fmt(i_opt)} (optical only). Raise the cloud severity to see how each state degrades.")
