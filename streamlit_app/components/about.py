"""About page for TruthLens."""

from __future__ import annotations

import streamlit as st

from streamlit_app.utils.confidence import CONFIDENCE_NOTE


def render_about_page() -> None:
    """Render the academic context and limitations page."""
    st.markdown("## About")
    st.markdown(
        """
        TruthLens is the practical demonstration component of the Bachelor thesis
        **“Zbulimi i Lajmeve të Rreme duke përdorur teknika të Machine Learning dhe Natural Language
        Processing”** in the Computer Engineering programme at the Polytechnic University of Tirana.

        The application demonstrates how a classical natural-language-processing workflow can load a
        saved TF-IDF vectorizer and a saved Linear SVM classifier to support binary fake-news text
        classification for English-language content.
        """
    )

    st.markdown("### Project purpose")
    st.markdown(
        """
        The prototype exists to present the thesis pipeline in an interactive form for academic
        demonstration. It is intended as a decision-support interface that exposes how the trained
        thesis model behaves on submitted English news text.
        """
    )

    st.markdown("### Classification vs. fact verification")
    st.markdown(
        """
        TruthLens performs text classification, not independent fact verification. The model learns
        linguistic and dataset-specific patterns associated with the WELFake labels. It does not
        retrieve external evidence, inspect original sources, validate author credibility, or check
        claims against real-time fact-checking databases.
        """
    )

    st.markdown("### Ethical limitations")
    st.markdown(
        """
        - Trained on English-language data only
        - Binary Fake/Real classification only
        - Possible source bias in the benchmark dataset
        - Possible topic bias in the benchmark dataset
        - Learns linguistic and dataset patterns rather than objective truth
        - No external evidence retrieval
        - No source-authority verification
        - No real-time fact-checking
        - May perform poorly on recent, domain-shifted, or out-of-distribution content
        - Confidence estimate is derived from the Linear SVM decision margin and is not a calibrated probability
        """
    )
    st.caption(CONFIDENCE_NOTE)

    st.warning(
        "TruthLens is an academic prototype and a decision-support tool. Its output must not be "
        "treated as definitive proof that a news article is true or false."
    )
