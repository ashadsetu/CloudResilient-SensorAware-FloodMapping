"""End-to-end: the Streamlit page renders and Predict works on a demo chip (uses streamlit's AppTest)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_app_renders_and_predicts_a_demo_scene():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
    assert not at.exception, at.exception
    next(b for b in at.button if b.label == "Predict").click().run()
    assert not at.exception, at.exception
    assert at.session_state["current"]["state"] == "both"
