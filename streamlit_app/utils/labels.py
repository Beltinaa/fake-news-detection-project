"""Verified label-mapping and margin interpretation helpers."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


EXPECTED_MODEL_CLASSES = (0, 1)
VERIFIED_LABEL_MAPPING = {
    0: "REAL",
    1: "FAKE",
}
VERDICT_COLORS = {
    "REAL": "#15803D",
    "FAKE": "#B91C1C",
}
VERDICT_EXPLANATIONS = {
    "REAL": "The submitted text was classified as REAL by the trained Linear SVM model.",
    "FAKE": "The submitted text was classified as FAKE by the trained Linear SVM model.",
}


def validate_model_classes(model_classes: Sequence[int] | np.ndarray | None) -> tuple[int, int]:
    """Validate that the saved binary classifier matches the verified thesis classes."""
    if model_classes is None:
        raise ValueError(
            "The saved classifier does not expose model.classes_. Expected binary classes [0, 1]."
        )

    classes_array = np.asarray(model_classes)
    if classes_array.ndim != 1 or classes_array.size != 2:
        raise ValueError(
            f"Expected a binary classifier with classes {list(EXPECTED_MODEL_CLASSES)}, "
            f"but received {classes_array.tolist()}."
        )

    normalized = tuple(int(value) for value in classes_array.tolist())
    if normalized != EXPECTED_MODEL_CLASSES:
        raise ValueError(
            f"Unexpected model.classes_ order {list(normalized)}. "
            f"Expected {list(EXPECTED_MODEL_CLASSES)} to match the verified thesis mapping."
        )

    return normalized


def display_label_for_class(class_id: int) -> str:
    """Return the verified display label for a numeric class id."""
    try:
        return VERIFIED_LABEL_MAPPING[int(class_id)]
    except KeyError as exc:
        raise ValueError(
            f"Unexpected class id {class_id}. Verified thesis mapping is {VERIFIED_LABEL_MAPPING}."
        ) from exc


def mapping_as_display_dict() -> Mapping[int, str]:
    """Return the verified numeric-to-display-label mapping."""
    return VERIFIED_LABEL_MAPPING.copy()


def display_labels_for_classes(model_classes: Sequence[int] | np.ndarray) -> list[str]:
    """Return display labels in the exact order used by the classifier."""
    validated = validate_model_classes(model_classes)
    return [display_label_for_class(class_id) for class_id in validated]


def favored_class_id_from_margin(
    decision_score: float | None,
    model_classes: Sequence[int] | np.ndarray,
) -> int | None:
    """Interpret the binary LinearSVC margin sign using the verified class order."""
    if decision_score is None:
        return None

    class_zero, class_one = validate_model_classes(model_classes)
    if decision_score < 0:
        return class_zero
    if decision_score > 0:
        return class_one
    return None


def favored_display_label_from_margin(
    decision_score: float | None,
    model_classes: Sequence[int] | np.ndarray,
) -> str | None:
    """Return the display label favored by the margin sign."""
    favored_class_id = favored_class_id_from_margin(decision_score, model_classes)
    if favored_class_id is None:
        return None
    return display_label_for_class(favored_class_id)


def margin_direction_note(model_classes: Sequence[int] | np.ndarray) -> str:
    """Return a user-facing note describing binary margin direction."""
    class_zero, class_one = validate_model_classes(model_classes)
    negative_label = display_label_for_class(class_zero)
    positive_label = display_label_for_class(class_one)
    return (
        f"With model.classes_ = {list((class_zero, class_one))}, negative decision scores favor "
        f"{negative_label} and positive decision scores favor {positive_label}."
    )
