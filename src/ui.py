"""UI building blocks: navbar, hero, cards, predict form, results panel, history, footer."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import streamlit as st

from .inference import (ASSETS, STATE_LABELS, TARGETS, asset_path, load_manifest, register_scene,
                        using_placeholders)
from .uploads import UploadError, available_states, parse_files

def _fit() -> dict:
    """Full-width kwargs that work on old and new Streamlit (use_container_width is deprecated)."""
    try:
        major, minor = (int(x) for x in st.__version__.split(".")[:2])
    except Exception:
        return dict(use_container_width=True)
    return dict(width="stretch") if (major, minor) >= (1, 52) else dict(use_container_width=True)


FIT = _fit()
CSS_PATH = Path(__file__).resolve().parent / "styles.css"

DISCLAIMER = ("⚠ <b>Disclaimer:</b> This tool is a <b>research demonstration only</b> — it is <b>NOT</b> an operational "
              "flood-warning or decision-making tool. Do not use it for emergency or safety decisions.")

GUARANTEE_MD = """
**What the missed-flood target means — and what it does not**

- The threshold for each sensor state is chosen by **chip-level conformal risk control**. It bounds the
  *expected per-chip missed-flood fraction over flood-containing chips that are exchangeable with the calibration chips*.
- It is **not** a per-pixel, per-image or per-event guarantee. It may be exceeded on individual scenes, and it is
  **not expected to hold under distribution shift** (for example the held-out Bolivia scenes).
- Scenes with no flood have an undefined missed-flood rate.
- Chips are 128×128 resizes of Sen1Floods11 scenes. Clouds are a **synthetic stress test**, not real atmosphere.
- Area values are approximate. The model was trained on Sen1Floods11 only; there is no Bangladesh validation.
"""


def inject_css() -> None:
    st.markdown(f"<style>{CSS_PATH.read_text()}</style>", unsafe_allow_html=True)


def anchor(name: str) -> None:
    st.markdown(f'<div id="{name}" class="anchor"></div>', unsafe_allow_html=True)


def navbar() -> None:
    st.markdown("""
<div class="fs-nav">
  <div class="fs-brand"><div class="logo">🌊</div><span>FloodSense <span class="sub">| Flood Mapper</span></span></div>
  <div class="fs-links"><a href="#home">Home</a><a href="#predict">Predict</a><a href="#upload">Upload</a><a href="#results">Results</a>
    <a href="#history">History</a><a href="#about">About</a></div>
  <div class="fs-actions"><a class="pill" href="#predict">Live check</a><a class="pill pill-fill" href="#predict">Check now</a></div>
</div>""", unsafe_allow_html=True)


def hero() -> None:
    st.markdown("""
<div class="hero">
  <span class="badge">PyTorch • Sensor-aware • Controlled missed-flood rate</span>
  <h1>Map floods from satellite images — <span class="grad">even under cloud</span></h1>
  <p>Not just a yes/no answer: the model shows a <b>probability map</b>, a <b>calibrated flood map</b>,
  <b>how much it trusts each sensor</b>, and <b>what happens when a sensor is missing</b>.</p>
  <div class="btns"><a class="btn btn-fill" href="#predict">Start prediction</a><a class="btn" href="#upload">Upload a scene</a><a class="btn" href="#about">Learn more</a></div>
  <p class="hero-note">Try a demo scene, or upload your own Sen1Floods11-style <b>GeoTIFFs</b> (<code>.tif</code>) or a <b>.npz</b> chip — radar, optical, or both.</p>
</div>""", unsafe_allow_html=True)


def feature_cards() -> None:
    st.markdown("""
<div class="cards3">
  <div class="fcard"><div class="ic">☁️</div><h4>Works under cloud</h4>
    <p>Radar keeps working when optical imagery is blocked by cloud.</p></div>
  <div class="fcard"><div class="ic">🎯</div><h4>Controlled missed-flood rate</h4>
    <p>A threshold per sensor state, set by conformal risk control to target a chosen missed-flood level.</p></div>
  <div class="fcard"><div class="ic">⚠️</div><h4>Research demo</h4>
    <p>Not an operational flood-warning tool — a demonstration on pre-selected scenes.</p></div>
</div>""", unsafe_allow_html=True)


def placeholder_banner() -> None:
    if using_placeholders():
        st.markdown('<div class="ph-banner">⚙ Placeholder assets in use (<code>*_PLACEHOLDER*</code> files): the model is '
                    'randomly initialised and every number is meaningless until you copy your exported '
                    '<code>app_assets/</code> into <code>assets/</code>.</div>', unsafe_allow_html=True)


# ----------------------------------------------------------------------------- predict form
DEFAULTS = dict(state="both", severity=0.6, target=0.10)


def _reset_form() -> None:
    st.session_state.update(f_state=DEFAULTS["state"], f_sev=DEFAULTS["severity"], f_target=DEFAULTS["target"])
    st.session_state["result"] = None
    st.session_state.update(f_source="demo", f_sample=_NO_SAMPLE, upload_id=None,
                            upl_nonce=st.session_state.get("upl_nonce", 0) + 1)
    manifest = load_manifest()
    if manifest:
        st.session_state["chip_id"] = manifest[0]["chip_id"]


def _random_scene() -> None:
    manifest = load_manifest()
    if manifest:
        st.session_state["chip_id"] = random.choice(manifest)["chip_id"]


def _select(cid: str) -> None:
    st.session_state["chip_id"] = cid


_NO_SAMPLE = "— none —"


def _sample_pairs() -> dict:
    """Bundled sample GeoTIFF pairs: {chip id: [S1 path, S2 path]} from assets/sample_tiffs."""
    d, out = ASSETS / "sample_tiffs", {}
    if d.is_dir():
        for s1 in sorted(d.glob("*_S1Hand.tif")):
            cid = s1.name[: -len("_S1Hand.tif")]
            if (d / f"{cid}_S2Hand.tif").exists():
                out[cid] = [s1, d / f"{cid}_S2Hand.tif"]
    return out


@st.cache_data(show_spinner=False)
def _read_bytes(path: str) -> bytes:
    return Path(path).read_bytes()


@st.cache_data(show_spinner="Reading files…", max_entries=8)
def _parse_cached(files: tuple) -> list:
    return parse_files(list(files))


def _preview(sc: dict) -> None:
    """Small preview of an uploaded scene exactly as it enters the model (128x128, normalised)."""
    cols = st.columns(3)
    if sc.get("has_radar", True):
        cols[0].image(sc["s1"][0], caption="Radar VV (normalised)", clamp=True, **FIT)
    if sc.get("has_optical", True):
        cols[1].image(np.clip(sc["s2"][[3, 2, 1]].transpose(1, 2, 0) / 0.3, 0, 1), caption="Optical RGB", clamp=True, **FIT)
    if sc.get("has_label"):
        lab = np.ones((*sc["y"].shape, 3), np.float32)
        lab[sc["y"] == 0] = 0.93
        lab[sc["y"] == 1] = (0.17, 0.5, 0.82)
        cols[2].image(lab, caption="Ground truth (blue = flood)", **FIT)
    area = sc["area_m2"]
    sensors = " + ".join(n for n, ok in (("radar", sc.get("has_radar", True)), ("optical", sc.get("has_optical", True))) if ok)
    st.caption(f"**{sc['title']}** · {sc['source']} · sensors: {sensors} · ground truth: {'yes' if sc['has_label'] else 'no'} · "
               + (f"≈ {area:,.0f} m² per 128-px cell" if area == area else "no geo-referencing, so area is n/a"))
    for w in sc.get("warnings", []):
        st.warning(w)


def _upload_section(ss) -> dict | None:
    """File uploader + sample picker. Returns the chosen (registered) scene, or None."""
    st.markdown("**Upload your own scene**")
    st.caption("**GeoTIFF:** `<id>_S1Hand.tif` (2 bands VV, VH in dB) and/or `<id>_S2Hand.tif` (13 Sentinel-2 bands, "
               "reflectance × 10000), optionally `<id>_LabelHand.tif` (1 = flood, 0 = land) for scoring.  "
               "**NPZ:** the demo-chip format (`s1`, `s2`, optional `y`, `area_m2`).  A single sensor works too. "
               "Files are processed in memory only — nothing is stored or sent anywhere.")
    files = st.file_uploader("Drop .tif / .tiff / .npz files", type=["tif", "tiff", "npz"], accept_multiple_files=True,
                             key=f"f_upload_{ss.get('upl_nonce', 0)}", label_visibility="collapsed")
    samples = _sample_pairs()
    if samples:
        st.selectbox("…or try a bundled sample GeoTIFF pair (Sen1Floods11 test chips, Ghana)", [_NO_SAMPLE, *samples], key="f_sample")
        with st.expander("Download the sample files (to test the uploader)"):
            for cid, paths in samples.items():
                dl = st.columns(len(paths))
                for col, p in zip(dl, paths):
                    col.download_button(f"⬇ {p.name}", _read_bytes(str(p)), file_name=p.name, mime="image/tiff",
                                        key=f"dl_{p.name}", **FIT)
    raw = [(f.name, f.getvalue()) for f in files or []]
    if not raw and samples and ss.get("f_sample") in samples:
        raw = [(p.name, _read_bytes(str(p))) for p in samples[ss["f_sample"]]]
    ss["upload_id"] = None
    if not raw:
        st.info("No file yet — drop a GeoTIFF pair or a .npz above, or pick a sample.")
        return None
    try:
        scenes = _parse_cached(tuple(raw))
    except UploadError as e:
        st.error(str(e))
        return None
    except Exception as e:                       # never crash on an odd file
        st.error(f"Could not read these files: {type(e).__name__}: {e}")
        return None
    idx = 0
    if len(scenes) > 1:
        idx = st.selectbox("Scenes found", list(range(len(scenes))), format_func=lambda i: scenes[i]["title"], key="f_upload_scene")
    sc = scenes[idx]
    ss["upload_id"] = register_scene(sc)
    _preview(sc)
    return sc


def predict_form(on_predict) -> None:
    """Card 'Choose a scene and sensors'. Calls on_predict(chip_id, state, severity, target) on Predict."""
    manifest = load_manifest()
    ss = st.session_state
    ss.setdefault("f_state", DEFAULTS["state"]); ss.setdefault("f_sev", DEFAULTS["severity"])
    ss.setdefault("f_target", DEFAULTS["target"]); ss.setdefault("f_source", "demo"); ss.setdefault("upload_id", None)
    if manifest:
        ss.setdefault("chip_id", manifest[0]["chip_id"])

    st.markdown('<div class="sec-title">Choose a scene and sensors</div>'
                '<div class="sec-sub">Pick a demo scene or upload your own, choose which sensors are available, '
                'and set the missed-flood target.</div>', unsafe_allow_html=True)
    anchor("upload")
    with st.container(border=True):
        st.radio("Scene source", ["demo", "upload"], key="f_source", horizontal=True,
                 format_func={"demo": "Demo scenes", "upload": "Upload your own (.tif / .npz)"}.get)
        scene, active_id = None, None
        if ss["f_source"] == "upload":
            scene = _upload_section(ss)
            active_id = ss.get("upload_id")
        else:
            top = st.columns([4, 1])
            top[0].markdown("**Demo scenes**")
            top[1].button("🎲 Random scene", on_click=_random_scene, **FIT)
            if not manifest:
                st.warning("No demo scenes found: copy your exported `manifest.json` and `demo_chips/` into `assets/`.")
            else:
                active_id = ss.get("chip_id")
                ncol = 4
                for r in range(0, len(manifest), ncol):
                    cols = st.columns(ncol)
                    for col, m in zip(cols, manifest[r:r + ncol]):
                        with col.container(border=True):
                            thumb = asset_path(f"thumbs/{m['chip_id']}.png")
                            if thumb.exists():
                                st.image(str(thumb), **FIT)
                            st.markdown(f"**{m.get('title', m['chip_id'])}**")
                            st.markdown(f'<span class="chip-tag">{m.get("event", "?")}</span>'
                                        f'<span class="chip-tag">{m.get("split", "?")}</span>', unsafe_allow_html=True)
                            st.markdown(f'<div class="chip-reason">{m.get("reason", "")}</div>', unsafe_allow_html=True)
                            sel = ss.get("chip_id") == m["chip_id"]
                            st.button("✓ Selected" if sel else "Select", key=f"sel_{m['chip_id']}", disabled=sel,
                                      on_click=_select, args=(m["chip_id"],), **FIT)

        opts = available_states(scene) if scene else ["both", "radar_only", "optical_only"]
        if ss["f_state"] not in opts:            # e.g. a radar-only upload: only 'Radar only' is possible
            ss["f_state"] = opts[0]
        st.markdown("---")
        c1, c2, c3 = st.columns([1.2, 1.2, 1])
        with c1:
            st.radio("Sensor availability", opts, key="f_state", horizontal=True, format_func=STATE_LABELS.get)
        with c2:
            radar = ss["f_state"] == "radar_only"
            st.slider("Cloud severity (synthetic)", 0.0, 1.0, step=0.1, key="f_sev", disabled=radar)
            if radar:
                st.caption("Not used: radar is unaffected by cloud, so this slider is disabled.")
        with c3:
            st.radio("Target missed-flood level", list(TARGETS), key="f_target", horizontal=True,
                     format_func=lambda t: f"{int(round(t * 100))} %")
        b1, b2, _ = st.columns([1, 1, 4])
        b1.button("↺ Reset", on_click=_reset_form, **FIT)
        if b2.button("Predict", type="primary", **FIT, disabled=not active_id):
            sev = float(ss["f_sev"])  # radar-only ignores cloud; the value is still used for the comparison strip
            on_predict(active_id, ss["f_state"], sev, float(ss["f_target"]))


# ----------------------------------------------------------------------------- results panel
import io
from datetime import datetime

import numpy as np
import plotly.graph_objects as go
from PIL import Image

from .inference import STATES, analyze
from .metrics import error_map


def _show_fig(fig) -> None:
    try:
        st.pyplot(fig, **FIT)
    except TypeError:
        st.pyplot(fig)


def _composite(rgba: np.ndarray) -> np.ndarray:
    """RGBA float -> RGB uint8 on white (ignore pixels become white)."""
    a = rgba[..., 3:4]
    return (np.clip(rgba[..., :3] * a + (1 - a), 0, 1) * 255).astype(np.uint8)


def _overlay(chip: dict, mask: np.ndarray) -> np.ndarray:
    """RGB float image: predicted flood (blue) over optical RGB, or over radar when there is no optical."""
    if chip.get("has_optical", True):
        base = np.clip(chip["s2"][[3, 2, 1]].transpose(1, 2, 0) / 0.3, 0, 1)
    else:
        base = np.repeat(chip["s1"][0][..., None], 3, axis=-1)
    out, m = base.astype(np.float32).copy(), np.asarray(mask, bool)
    out[m] = 0.45 * out[m] + 0.55 * np.array([0.17, 0.5, 0.82], np.float32)
    return out


def _pct(v: float) -> str:
    return "n/a" if v != v else f"{v * 100:.1f} %"


def _card(k: str, v: str, n: str = "") -> str:
    return f'<div class="mcard"><div class="k">{k}</div><div class="v">{v}</div><div class="n">{n}</div></div>'


def render_panels(res: dict) -> None:
    import matplotlib.pyplot as plt

    chip, state, sel = res["chip"], res["state"], res["sel"]
    from .inference import optical_as_seen
    fig, ax = plt.subplots(1, 5, figsize=(16, 3.8))
    titles = ["Radar (VV)", "Optical RGB (as the model saw it)", "Probability heatmap",
              f"Calibrated flood map (t = {sel['thr']['threshold']:.3f})",
              "Error map vs ground truth" if res.get("has_label", True) else "Flood overlay (no ground truth)"]
    if STATES[state][0]:
        ax[0].imshow(chip["s1"][0], cmap="gray")
    else:
        ax[0].imshow(np.full((8, 8), 0.8), cmap="gray", vmin=0, vmax=1); ax[0].text(3.5, 3.5, "radar not used", ha="center", va="center", color="#555")
    opt = optical_as_seen(chip, state, res["severity"])
    if opt is not None:
        ax[1].imshow(np.clip(opt[[3, 2, 1]].transpose(1, 2, 0) / 0.3, 0, 1))
    else:
        ax[1].imshow(np.full((8, 8), 0.8), cmap="gray", vmin=0, vmax=1); ax[1].text(3.5, 3.5, "optical not available", ha="center", va="center", color="#555")
    im = ax[2].imshow(sel["prob"], cmap="YlGnBu", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax[2], fraction=0.046, pad=0.02)
    valid = chip["y"] != 255
    ax[3].imshow(np.where(valid, sel["mask"], np.nan), cmap="Blues", vmin=0, vmax=1.4)
    if res.get("has_label", True):
        ax[4].imshow(np.ones_like(chip["y"], float), cmap="gray", vmin=0, vmax=1)
        ax[4].imshow(error_map(sel["mask"], chip["y"]))
    else:
        ax[4].imshow(_overlay(chip, sel["mask"]))
    for a, t in zip(ax, titles):
        a.set_title(t, fontsize=9); a.axis("off")
    fig.tight_layout()
    _show_fig(fig)
    plt.close(fig)
    if res.get("has_label", True):
        st.markdown('<span class="chip-reason">Error map: <b style="color:#1a9e9e">■</b> correct flood · '
                    '<b style="color:#f5991a">■</b> false alarm · <b style="color:#d92626">■</b> missed flood · '
                    'ignore pixels are transparent.</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="chip-reason"><b style="color:#2B7FD0">■</b> predicted flood over the optical image '
                    '(or radar if optical is absent). No ground truth was provided, so errors cannot be shown.</span>',
                    unsafe_allow_html=True)


def render_metrics(res: dict) -> None:
    m, thr = res["sel"]["m"], res["sel"]["thr"]
    labelled = res.get("has_label", True)
    na = "n/a — no ground truth"
    miss = na if not labelled else "n/a — no flood in this scene" if not m["has_flood"] else _pct(m["miss"])
    iou = na if not labelled else "n/a — no flood in this scene" if not m["has_flood"] else f"{m['iou']:.3f}"
    fp = "n/a" if not labelled else f"{m['fp']:,}"
    km2 = res["area_km2"]
    val = lambda v: f'<span style="font-size:1.05rem">{v}</span>' if v.startswith("n/a") else v
    cards = [_card("Predicted flooded share", _pct(res["share"]), "of valid pixels"),
             _card("Approx. flooded area", f"{km2:.2f} km²" if km2 == km2 else val("n/a"),
                   "approximate" if km2 == km2 else "no geo-referencing"),
             _card("Threshold used", f"{thr['threshold']:.3f}", f"{int(round(res['target'] * 100))} % target · {STATE_LABELS[res['state']]}"),
             _card("Missed-flood fraction (this scene)", val(miss), "FN / (TP + FN)"),
             _card("False-alarm pixels", val(fp), "predicted flood, truly dry"),
             _card("IoU vs ground truth", val(iou), "valid pixels only")]
    st.markdown(f'<div class="mgrid">{"".join(cards)}</div>', unsafe_allow_html=True)
    if thr.get("feasible") is False:
        st.warning(f"The {int(round(res['target'] * 100))} % missed-flood target cannot be certified for "
                   f"“{STATE_LABELS[res['state']]}” with this calibration set; the threshold shown is a fallback.")
    if res["entry"].get("upload"):
        st.info("Uploaded scene: the thresholds were calibrated on Sen1Floods11 chips, so the missed-flood target comes "
                "with **no guarantee** for your scene (different sensor processing, region or season = distribution shift).")


def render_gates(res: dict) -> None:
    g = res["sel"]["gates"]                         # [scale, (radar, optical)]
    scales = ["Scale 1 (fine)", "Scale 2", "Scale 3 (coarse)"]
    fig = go.Figure([go.Bar(name="Radar", x=scales, y=g[:, 0], marker_color="#2B7FD0"),
                     go.Bar(name="Optical", x=scales, y=g[:, 1], marker_color="#2fb98c")])
    fig.update_layout(barmode="group", height=300, margin=dict(l=10, r=10, t=30, b=10), yaxis=dict(range=[0, 1], title="Mean gate weight"),
                      title="What the model trusted", legend=dict(orientation="h", y=1.15), plot_bgcolor="white")
    st.plotly_chart(fig, **FIT)
    st.caption("Mean per-pixel gate weight for each sensor at each scale. Weights over the available sensors sum to 1; "
               "a missing sensor gets exactly zero weight.")


def render_comparison(res: dict) -> None:
    from .metrics import flooded_share

    chip, per, labelled = res["chip"], res["per"], res.get("has_label", True)
    st.markdown("**Same scene, " + ("three sensor states**" if len(per) == 3 else "available sensor states**")
                + f" (cloud severity {res['severity']:.1f}, target {int(round(res['target'] * 100))} %)")
    cols = st.columns(len(per))
    for col, (s, r) in zip(cols, per.items()):
        if labelled:
            iou = r["m"]["iou"]
            col.image(_composite(error_map(r["mask"], chip["y"])), **FIT,
                      caption=f"{STATE_LABELS[s]} — IoU {'n/a' if iou != iou else f'{iou:.3f}'}")
        else:
            col.image((_overlay(chip, r["mask"]) * 255).astype(np.uint8), **FIT,
                      caption=f"{STATE_LABELS[s]} — {_pct(flooded_share(r['mask'], chip['y']))} flagged as flood")
    if {"radar_only", "both"} <= set(per):
        ag = res["agreement"]
        st.metric("Agreement between radar-only and fused map (IoU)", "n/a" if ag != ag else f"{ag:.3f}")


def render_results(params: dict | None) -> None:
    st.markdown('<div class="sec-title" style="text-align:left">Prediction</div>', unsafe_allow_html=True)
    if not params:
        st.markdown('<div class="empty">No results yet — choose a scene and click Predict.</div>', unsafe_allow_html=True)
        return
    try:
        res = analyze(**params)
    except Exception as e:  # never crash on a missing/odd asset
        st.error(f"Could not run this prediction: {e}")
        return
    st.caption(f"Scene **{res['entry'].get('title', res['chip']['chip_id'])}** · {STATE_LABELS[res['state']]} · "
               f"cloud {res['severity']:.1f}" + (" (ignored: radar only)" if res["state"] == "radar_only" else ""))
    render_panels(res)
    render_metrics(res)
    left, right = st.columns([1, 1])
    with left:
        render_gates(res)
    with right:
        st.markdown(f'<div class="explain"><b>In plain words.</b> {res["explanation"]}</div>', unsafe_allow_html=True)
    render_comparison(res)
    with st.expander("Missed-flood guarantee"):
        st.markdown(GUARANTEE_MD)


# ----------------------------------------------------------------------------- history
def _thumb_bytes(res: dict) -> bytes:
    rgb = (_composite(error_map(res["sel"]["mask"], res["chip"]["y"])) if res.get("has_label", True)
           else (_overlay(res["chip"], res["sel"]["mask"]) * 255).astype(np.uint8))
    img = Image.fromarray(rgb).resize((64, 64), Image.NEAREST)
    buf = io.BytesIO(); img.save(buf, "PNG")
    return buf.getvalue()


def add_history(params: dict) -> None:
    res = analyze(**params)
    m = res["sel"]["m"]
    entry = dict(params=dict(params), title=res["entry"].get("title", params["chip_id"]), iou=m["iou"], share=res["share"],
                 miss=m["miss"], thr=res["sel"]["thr"]["threshold"], thumb=_thumb_bytes(res),
                 when=datetime.now().strftime("%H:%M:%S"))
    h = st.session_state.setdefault("history", [])
    h.insert(0, entry)
    del h[20:]


def _view(i: int) -> None:
    st.session_state["current"] = dict(st.session_state["history"][i]["params"])


def _clear_history() -> None:
    st.session_state["history"] = []


def render_history() -> None:
    top = st.columns([5, 1])
    top[0].markdown('<div class="sec-title" style="text-align:left">🕘 Previous results</div>', unsafe_allow_html=True)
    h = st.session_state.get("history", [])
    top[1].button("Clear history", on_click=_clear_history, disabled=not h, **FIT)
    if not h:
        st.markdown('<div class="empty" style="padding:28px">No previous results yet.</div>', unsafe_allow_html=True)
        return
    for i, e in enumerate(h):
        p = e["params"]
        c = st.columns([1, 6, 1.2])
        c[0].image(e["thumb"])
        iou = "n/a" if e["iou"] != e["iou"] else f"{e['iou']:.3f}"
        c[1].markdown(f"**{e['title']}** · {STATE_LABELS[p['state']]} · cloud {p['severity']:.1f} · target "
                      f"{int(round(p['target'] * 100))} %  \n<span class='chip-reason'>IoU {iou} · flooded share {_pct(e['share'])} · "
                      f"threshold {e['thr']:.3f} · {e['when']}</span>", unsafe_allow_html=True)
        c[2].button("View", key=f"view_{i}_{e['when']}", on_click=_view, args=(i,), **FIT)


# ----------------------------------------------------------------------------- disclaimer + footer
def disclaimer_and_footer() -> None:
    st.markdown(f'<div class="disclaimer"><div>{DISCLAIMER}<br><i>Research demonstration only — NOT an operational '
                'flood-warning tool.</i></div></div>', unsafe_allow_html=True)
    anchor("about")
    st.markdown("""
<div class="footer">
  <div><b style="font-size:1.1rem">🌊 FloodSense</b><p>A research demo of a sensor-aware fusion U-Net that maps floods from
    Sentinel-1 radar and Sentinel-2 optical imagery and keeps working when either sensor is missing. Runs on pre-selected
    Sen1Floods11 chips; it does not process live satellite data.</p></div>
  <div><h5>Quick links</h5><a href="#home">Home</a><a href="#predict">Predict form</a><a href="#results">Results &amp; explanation</a><a href="#history">History</a></div>
  <div><h5>About the model</h5><a>Sensor-aware fusion U-Net</a><a>Chip-level conformal risk control</a><a>Temperature scaling</a><a>Sen1Floods11</a></div>
</div>
<div class="copy">© 2026 FloodSense · MIT License · Trained on Sen1Floods11 only; no Bangladesh validation.</div>""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------- results & explanation page
import pandas as pd
import plotly.express as px

from .inference import ASSETS

CLOUD_MODEL_ORDER = ["radar_unet", "optical_unet", "early_fusion", "dropout_fusion", "aware_fusion"]
MODEL_NAMES = {"radar_unet": "Radar U-Net", "optical_unet": "Optical U-Net", "early_fusion": "Early fusion",
               "dropout_fusion": "Plain-dropout fusion", "aware_fusion": "Proposed (sensor-aware)"}


def _results_file(stem: str, ext: str = "csv") -> Path | None:
    """assets/results/<stem>*.csv (real export first, then *_PLACEHOLDER*), e.g. cloud_robustness_summary_one_day.csv."""
    files = sorted((ASSETS / "results").glob(f"{stem}*.{ext}"))
    real = [f for f in files if "PLACEHOLDER" not in f.name]
    pick = real or files
    return pick[0] if pick else None


def _read_csv(stem: str) -> pd.DataFrame | None:
    p = _results_file(stem)
    if p is None:
        return None
    try:
        return pd.read_csv(p)
    except Exception:
        return None


def _missing(what: str) -> None:
    st.info(f"{what} not found in `assets/results/` — copy your exported results to see it here.")


def render_results_page() -> None:
    st.markdown('<div class="sec-title">Results &amp; explanation</div>'
                '<div class="sec-sub">Training and evaluation outputs exported from the notebook. Clouds are a synthetic stress test.</div>',
                unsafe_allow_html=True)
    df = _read_csv("cloud_robustness_summary")
    st.subheader("Cloud robustness")
    need = {"model", "state", "severity"}
    if df is None or not need <= set(df.columns):
        _missing("Cloud-robustness summary (`cloud_robustness_summary*.csv` with columns model, state, severity, …)")
    else:
        df = df.copy()
        if "threshold_source" in df.columns:
            sources = list(df["threshold_source"].unique())
            default = sources.index("crc_chip") if "crc_chip" in sources else 0
            src = st.selectbox("Threshold rule", sources, index=default,
                               help="crc_chip is the rule used by the app (chip-level conformal risk control).")
            df = df[df["threshold_source"] == src]
        df["model"] = df["model"].map(lambda m: MODEL_NAMES.get(m, m))
        for col, title in (("IoU", "IoU"), ("miss_rate_chip_macro", "Chip-macro missed-flood rate"),
                           ("FP", "False-positive pixels")):
            y = col if col in df.columns else (f"{col}_mean" if f"{col}_mean" in df.columns else None)
            if y is None:
                st.caption(f"Column `{col}` missing from the cloud-robustness table.")
                continue
            sd = f"{col}_sd" if f"{col}_sd" in df.columns else None
            fig = px.line(df, x="severity", y=y, color="model", facet_col="state", markers=True, title=title,
                          error_y=sd, labels={y: title, "severity": "Synthetic cloud severity"})
            fig.update_layout(height=340, margin=dict(l=10, r=10, t=50, b=10), plot_bgcolor="white")
            st.plotly_chart(fig, **FIT)
        st.caption("Lines: mean over seeds; error bars: ± 1 sd across seeds (when available).")

    for stem, title in (("eval_main_summary", "Main evaluation (parameter counts, seed mean ± sd)"),
                        ("ablation_summary", "Ablations"),
                        ("probability_calibration", "Calibration (temperature scaling, ECE, Brier)"),
                        ("threshold_report", "Thresholds chosen on the calibration chips")):
        st.subheader(title)
        t = _read_csv(stem)
        if t is None:
            _missing(f"`{stem}*.csv`")
        else:
            st.dataframe(t.round(4), hide_index=True, **FIT)

    st.subheader("Figures (reliability diagrams, gate weights)")
    pngs = sorted((ASSETS / "results").glob("*.png"))
    real = [p for p in pngs if "PLACEHOLDER" not in p.name]
    shown = real or pngs
    if not shown:
        _missing("Result figures (`*.png`)")
    for i in range(0, len(shown), 2):
        cols = st.columns(2)
        for col, p in zip(cols, shown[i:i + 2]):
            col.image(str(p), caption=p.stem.replace("_", " "), **FIT)
