"""FloodSense — Streamlit entry point (page assembly only)."""
import streamlit as st

from src import ui

st.set_page_config(page_title="FloodSense | Flood Mapper", page_icon="🌊", layout="wide",
                   initial_sidebar_state="collapsed")
ui.inject_css()
ui.navbar()
ui.anchor("home")
ui.hero()
ui.feature_cards()
ui.placeholder_banner()


def on_predict(chip_id, state, severity, target):
    params = dict(chip_id=chip_id, state=state, severity=severity, target=target)
    st.session_state["current"] = params
    ui.add_history(params)


ui.anchor("predict")
ui.predict_form(on_predict)
ui.render_results(st.session_state.get("current"))
ui.anchor("results")
ui.render_results_page()
ui.anchor("history")
ui.render_history()
ui.disclaimer_and_footer()
