"""Confidence-estimate tests for Linear SVM decision margins."""

from __future__ import annotations

from streamlit_app.utils.confidence import (
    CONFIDENCE_METHOD_NOTE,
    CONFIDENCE_NOTE,
    confidence_label_from_score,
    estimate_from_decision_score,
)


def test_confidence_estimate_is_bounded_between_fifty_and_one_hundred_percent():
    estimate = estimate_from_decision_score(0.0)
    assert estimate is not None
    assert 0.50 <= estimate < 1.0


def test_confidence_estimate_increases_with_absolute_margin():
    small = estimate_from_decision_score(0.5)
    large = estimate_from_decision_score(2.5)
    assert small is not None and large is not None
    assert large > small


def test_confidence_estimate_depends_on_absolute_margin_not_sign():
    positive = estimate_from_decision_score(1.5)
    negative = estimate_from_decision_score(-1.5)
    assert positive == negative


def test_confidence_labels_follow_verified_threshold_bands():
    assert confidence_label_from_score(0.1) == "Low"
    assert confidence_label_from_score(1.5) == "Moderate"
    assert confidence_label_from_score(3.0) == "High"


def test_confidence_notes_do_not_claim_calibrated_probability():
    combined_note = f"{CONFIDENCE_NOTE} {CONFIDENCE_METHOD_NOTE}".lower()
    assert "calibrated probability" in combined_note
