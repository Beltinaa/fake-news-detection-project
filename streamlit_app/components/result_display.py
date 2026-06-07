"""Detector result rendering helpers."""

from __future__ import annotations

from typing import Any, Mapping

import streamlit as st

from streamlit_app.utils.confidence import (
    CONFIDENCE_METHOD_NOTE,
    CONFIDENCE_NOTE,
    CONFIDENCE_THRESHOLD_NOTE,
)
from streamlit_app.utils.labels import VERDICT_EXPLANATIONS


PIPELINE_TEXT = (
    "`Input text` → `TF-IDF transformation` → `220,275-dimensional sparse feature space` "
    "→ `Linear SVM` → `prediction`"
)


def render_result_display(result: Mapping[str, Any]) -> None:
    """Render a thesis-aligned classification result."""
    prediction = result["prediction"]
    verdict_class = "verdict-fake" if prediction == "FAKE" else "verdict-real"
    verdict_text = VERDICT_EXPLANATIONS[prediction]

    st.markdown(
        f"""
        <div class="tl-result-card {verdict_class}">
            <div class="tl-result-label">{prediction}</div>
            <p class="tl-result-text">
                {verdict_text}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info(
        "This is a statistical classification result and not an independent verification of the "
        "factual claims in the article."
    )

    confidence_estimate = result.get("confidence_estimate")
    confidence_label = result.get("confidence_label") or "Unavailable"
    decision_score = result.get("decision_score")
    absolute_decision_score = result.get("absolute_decision_score")

    metric_columns = st.columns(4)
    metric_columns[0].metric("Predicted label", prediction)
    metric_columns[1].metric(
        "Decision-strength estimate",
        f"{confidence_estimate * 100:.1f}%" if confidence_estimate is not None else "Unavailable",
    )
    metric_columns[2].metric("Qualitative strength", confidence_label)
    metric_columns[3].metric(
        "Decision margin",
        f"{decision_score:.4f}" if decision_score is not None else "Unavailable",
    )

    if confidence_estimate is not None:
        st.caption(CONFIDENCE_NOTE)
        st.caption(CONFIDENCE_THRESHOLD_NOTE)
        st.caption(CONFIDENCE_METHOD_NOTE)

    with st.expander("Technical details", expanded=False):
        st.markdown(
            "\n".join(
                [
                    f"- Model: `{result['model']}`",
                    f"- Vectorizer: `{result['vectorizer']}`",
                    f"- Predicted numeric class: `{result['label']}`",
                    f"- Predicted display label: `{prediction}`",
                    f"- Verified label mapping: `{result['class_mapping']}`",
                    f"- model.classes_: `{result['model_classes']}`",
                    f"- Submitted word count: `{result['word_count']}`",
                    f"- Raw decision score: `{decision_score:.4f}`"
                    if decision_score is not None
                    else "- Raw decision score: `Unavailable`",
                    f"- Absolute decision score: `{absolute_decision_score:.4f}`"
                    if absolute_decision_score is not None
                    else "- Absolute decision score: `Unavailable`",
                    f"- Decision-strength estimate: `{confidence_estimate * 100:.2f}%`"
                    if confidence_estimate is not None
                    else "- Decision-strength estimate: `Unavailable`",
                    f"- Confidence category: `{confidence_label}`",
                    f"- Vectorizer feature count: `{result['vectorizer_feature_count']}`",
                    f"- Held-out test accuracy: `{result['test_accuracy']:.2f}%`",
                    f"- Margin interpretation: {result['margin_direction_note']}",
                    "- Prediction language scope: `English news text`",
                    f"- Inference pipeline: {PIPELINE_TEXT}",
                ]
            )
        )

    st.markdown("**Inference pipeline**")
    st.markdown(PIPELINE_TEXT)
