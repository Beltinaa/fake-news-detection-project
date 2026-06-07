"""Main detector page for the TruthLens Streamlit application."""

from __future__ import annotations

from datetime import datetime

import streamlit as st

from streamlit_app.config import (
    CLEANED_ARTICLE_COUNT,
    DATASET_NAME,
    LINEAR_SVM_ACCURACY_PERCENT,
    MODEL_NAME,
)
from streamlit_app.components.result_display import render_result_display
from streamlit_app.services.model_service import ModelServiceError, get_model_status, predict_text
from streamlit_app.utils.demo_examples import load_demo_examples
from streamlit_app.utils.validation import MAX_INPUT_CHARS, count_words, validate_input


HISTORY_LIMIT = 5


def _short_preview(text: str, limit: int = 96) -> str:
    clean_text = " ".join(text.split())
    if len(clean_text) <= limit:
        return clean_text
    return f"{clean_text[: limit - 1]}…"


def _set_detector_input(text: str) -> None:
    st.session_state.detector_input = text
    st.session_state.pop("latest_result", None)


def _render_hero() -> None:
    st.markdown(
        """
        <div class="tl-hero">
            <div class="tl-badge">Bachelor Thesis Demo</div>
            <h1>TruthLens</h1>
            <p class="tl-subtitle">Machine Learning-Based Fake News Detection</p>
            <p class="tl-university">
                Universiteti Politeknik i Tiranës — Fakulteti i Teknologjisë së Informacionit.
            </p>
            <p class="tl-body">
                Ky prototip akademik përdor TF-IDF dhe Linear SVM për të klasifikuar përmbajtje
                lajmesh në gjuhën angleze si Fake ose Real.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_top_metrics() -> None:
    cards = [
        ("Dataset", DATASET_NAME),
        ("Cleaned articles", f"{CLEANED_ARTICLE_COUNT:,}"),
        ("Best model", MODEL_NAME),
        ("Test accuracy", f"{LINEAR_SVM_ACCURACY_PERCENT:.2f}%"),
    ]
    columns = st.columns(4)
    for column, (label, value) in zip(columns, cards):
        column.markdown(
            f"""
            <div class="tl-stat-card">
                <p class="tl-stat-label">{label}</p>
                <p class="tl-stat-value">{value}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _append_history(result: dict[str, object], text: str) -> None:
    history_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "prediction": result["prediction"],
        "preview": _short_preview(text),
    }
    history = st.session_state.get("analysis_history", [])
    st.session_state.analysis_history = [history_entry, *history][:HISTORY_LIMIT]


def _render_history() -> None:
    history = st.session_state.get("analysis_history", [])
    header_columns = st.columns([3, 1])
    header_columns[0].markdown("### Recent analyses")
    if header_columns[1].button("Clear history", use_container_width=True):
        st.session_state.analysis_history = []
        st.experimental_rerun()

    if not history:
        st.caption("Session history is empty. The last five analyses are stored only in memory.")
        return

    for item in history:
        st.markdown(
            f"""
            <div class="tl-history-item">
                <div>
                    <p class="tl-history-prediction">{item['prediction']}</p>
                    <p class="tl-history-preview">{item['preview']}</p>
                </div>
                <p class="tl-history-time">{item['timestamp']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_detector_page() -> None:
    """Render the interactive detector page."""
    _render_hero()
    _render_top_metrics()

    if "detector_input" not in st.session_state:
        st.session_state.detector_input = ""
    if "analysis_history" not in st.session_state:
        st.session_state.analysis_history = []

    st.markdown("### Detector")
    st.caption(
        "Paste an English news headline, excerpt, or full article here. The deployed prototype "
        "uses the saved TF-IDF vectorizer and saved Linear SVM thesis model without retraining."
    )

    model_status = get_model_status()
    if model_status["ready"]:
        st.success(str(model_status["message"]))
    else:
        st.error(str(model_status["message"]))

    demo_examples = load_demo_examples()
    if demo_examples:
        st.markdown("#### Load demo example")
        st.caption(
            "This sample comes from the held-out WELFake test set and is provided only for model "
            "demonstration."
        )
        selected_name = st.selectbox(
            "Load demo example",
            options=[example["name"] for example in demo_examples],
            key="demo_example_name",
            label_visibility="collapsed",
        )
        selected_example = next(
            example for example in demo_examples if example["name"] == selected_name
        )
        demo_columns = st.columns([2, 1])
        demo_columns[0].caption(
            f"Stored labels: actual `{selected_example['actualLabel']}`, predicted "
            f"`{selected_example['predictedLabel']}`"
        )
        demo_columns[1].button(
            "Load selected example",
            use_container_width=True,
            on_click=_set_detector_input,
            args=(selected_example["text"],),
        )

    input_text = st.text_area(
        "Article text",
        key="detector_input",
        height=280,
        placeholder="Paste an English news headline, excerpt, or full article here…",
    )

    normalized_word_count = count_words(input_text.strip()) if input_text else 0
    char_count = len(input_text)
    count_columns = st.columns(2)
    count_columns[0].metric("Word count", normalized_word_count)
    count_columns[1].metric("Character count", char_count)

    live_validation = validate_input(input_text)
    for warning in live_validation.warnings:
        st.warning(warning)
    st.caption(
        f"Safe maximum length: {MAX_INPUT_CHARS:,} characters. Meaningful article text usually "
        "produces more reliable classification than a one-line phrase."
    )

    action_columns = st.columns([2, 1])
    analyze_clicked = action_columns[0].button(
        "Analyze Article",
        type="primary",
        use_container_width=True,
        disabled=not bool(model_status["ready"]),
    )
    action_columns[1].button(
        "Clear",
        use_container_width=True,
        on_click=_set_detector_input,
        args=("",),
    )

    if analyze_clicked:
        validation_result = validate_input(input_text)
        if not validation_result.is_valid:
            st.session_state.pop("latest_result", None)
            for error in validation_result.errors:
                st.error(error)
        else:
            try:
                with st.spinner("Running the saved TF-IDF + Linear SVM inference pipeline..."):
                    result = predict_text(validation_result.normalized_text)
            except ModelServiceError as exc:
                st.session_state.pop("latest_result", None)
                st.error(str(exc))
            else:
                st.session_state.latest_result = result
                _append_history(result=result, text=validation_result.normalized_text)

    latest_result = st.session_state.get("latest_result")
    if latest_result:
        render_result_display(latest_result)

    _render_history()
