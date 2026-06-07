"""Sidebar navigation and context for TruthLens."""

from __future__ import annotations

import streamlit as st


PAGES = ["Detector", "Methodology", "Results", "About"]


def render_sidebar() -> str:
    """Render the application sidebar and return the selected page."""
    with st.sidebar:
        st.markdown("## TruthLens")
        st.caption("Machine Learning-Based Fake News Detection")
        st.markdown(
            """
            <div class="tl-sidebar-card">
                <p><strong>Demo context</strong></p>
                <p>Bachelor thesis presentation prototype</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        selected_page = st.radio("Navigation", PAGES, label_visibility="collapsed")

        st.markdown("### Academic context")
        st.caption(
            "Universiteti Politeknik i Tiranës · Fakulteti i Teknologjisë së Informacionit"
        )
        st.caption("Student: Beltina Manallari")

        st.markdown("### Deployed pipeline")
        st.caption("Saved `TF-IDF` vectorizer → saved `Linear SVM` classifier")
        st.caption("Verified mapping: `0 = REAL`, `1 = FAKE`")

        st.info(
            "TruthLens is an academic prototype. Use the output as decision support, not as "
            "definitive verification."
        )

    return selected_page
