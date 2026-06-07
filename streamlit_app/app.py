"""TruthLens Streamlit application entry point."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from streamlit_app.components import (  # noqa: E402
    render_about_page,
    render_detector_page,
    render_methodology_page,
    render_results_page,
    render_sidebar,
)
from streamlit_app.utils.paths import ASSETS_DIR  # noqa: E402


PAGE_RENDERERS = {
    "Detector": render_detector_page,
    "Methodology": render_methodology_page,
    "Results": render_results_page,
    "About": render_about_page,
}


def _load_css() -> None:
    css_path = ASSETS_DIR / "styles.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)


def main() -> None:
    """Run the Streamlit app."""
    st.set_page_config(
        page_title="TruthLens",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _load_css()

    selected_page = render_sidebar()
    renderer = PAGE_RENDERERS[selected_page]
    renderer()


if __name__ == "__main__":
    main()
