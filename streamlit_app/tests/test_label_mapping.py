"""Label mapping and margin interpretation tests."""

from __future__ import annotations

import numpy as np
import pytest

from streamlit_app.utils.labels import (
    VERIFIED_LABEL_MAPPING,
    display_label_for_class,
    favored_class_id_from_margin,
    validate_model_classes,
)


def test_verified_numeric_mapping_matches_audit():
    assert VERIFIED_LABEL_MAPPING[0] == "REAL"
    assert VERIFIED_LABEL_MAPPING[1] == "FAKE"


def test_display_label_lookup_uses_verified_mapping():
    assert display_label_for_class(0) == "REAL"
    assert display_label_for_class(1) == "FAKE"


def test_validate_model_classes_accepts_verified_binary_order():
    assert validate_model_classes(np.array([0, 1])) == (0, 1)


def test_validate_model_classes_rejects_unexpected_order():
    with pytest.raises(ValueError, match="Unexpected model.classes_ order"):
        validate_model_classes(np.array([1, 0]))


def test_negative_margin_favors_class_zero():
    assert favored_class_id_from_margin(-0.25, np.array([0, 1])) == 0


def test_positive_margin_favors_class_one():
    assert favored_class_id_from_margin(0.25, np.array([0, 1])) == 1
