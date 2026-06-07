"""Decision-score interpretation helpers for Linear SVM output."""

from __future__ import annotations

import math

from streamlit_app.config import CONFIDENCE_MARGIN_THRESHOLDS


CONFIDENCE_NOTE = (
    "Derived from the Linear SVM decision margin; not a calibrated probability."
)
CONFIDENCE_THRESHOLD_NOTE = (
    "Low / Moderate / High bands are derived from the 50th and 90th percentiles of absolute "
    "decision margins among correctly classified held-out test samples."
)
CONFIDENCE_METHOD_NOTE = (
    "The confidence estimate is derived from the absolute Linear SVM decision score and is not a "
    "calibrated probability."
)


def estimate_from_decision_score(decision_score: float | None) -> float | None:
    """Convert the absolute decision score into a bounded display estimate."""
    if decision_score is None:
        return None

    magnitude = abs(float(decision_score))
    estimate = 1.0 / (1.0 + math.exp(-magnitude))
    return max(0.50, min(0.99, estimate))


def confidence_label_from_score(decision_score: float | None) -> str | None:
    """Map score magnitude to a qualitative confidence label."""
    if decision_score is None:
        return None

    magnitude = abs(float(decision_score))
    if magnitude < CONFIDENCE_MARGIN_THRESHOLDS["moderate_lower"]:
        return "Low"
    if magnitude < CONFIDENCE_MARGIN_THRESHOLDS["high_lower"]:
        return "Moderate"
    return "High"
