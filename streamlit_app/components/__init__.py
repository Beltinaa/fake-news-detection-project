"""Streamlit page components for TruthLens."""

from .about import render_about_page
from .detector import render_detector_page
from .methodology import render_methodology_page
from .results import render_results_page
from .sidebar import render_sidebar

__all__ = [
    "render_about_page",
    "render_detector_page",
    "render_methodology_page",
    "render_results_page",
    "render_sidebar",
]
