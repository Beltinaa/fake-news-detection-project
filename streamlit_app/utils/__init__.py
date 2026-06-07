"""Utility helpers for TruthLens."""

from .confidence import (
    CONFIDENCE_METHOD_NOTE,
    CONFIDENCE_NOTE,
    CONFIDENCE_THRESHOLD_NOTE,
    confidence_label_from_score,
    estimate_from_decision_score,
)
from .demo_examples import load_demo_examples
from .labels import (
    EXPECTED_MODEL_CLASSES,
    VERIFIED_LABEL_MAPPING,
    display_label_for_class,
    display_labels_for_classes,
    favored_class_id_from_margin,
    favored_display_label_from_margin,
    margin_direction_note,
    mapping_as_display_dict,
    validate_model_classes,
)
from .paths import (
    ASSETS_DIR,
    DATA_DIR,
    DEMO_EXAMPLES_FILE,
    FIGURES_DIR,
    MODELS_DIR,
    PROCESSED_DATA_FILE,
    RAW_DATA_FILE,
    REPO_ROOT,
    REQUIRED_MODEL_FILES,
    SPLITS_DIR,
)
from .validation import MAX_INPUT_CHARS, ValidationResult, count_words, validate_input

__all__ = [
    "ASSETS_DIR",
    "CONFIDENCE_METHOD_NOTE",
    "CONFIDENCE_NOTE",
    "CONFIDENCE_THRESHOLD_NOTE",
    "DATA_DIR",
    "DEMO_EXAMPLES_FILE",
    "EXPECTED_MODEL_CLASSES",
    "FIGURES_DIR",
    "MAX_INPUT_CHARS",
    "MODELS_DIR",
    "PROCESSED_DATA_FILE",
    "RAW_DATA_FILE",
    "REPO_ROOT",
    "REQUIRED_MODEL_FILES",
    "SPLITS_DIR",
    "ValidationResult",
    "VERIFIED_LABEL_MAPPING",
    "confidence_label_from_score",
    "count_words",
    "display_label_for_class",
    "display_labels_for_classes",
    "estimate_from_decision_score",
    "favored_class_id_from_margin",
    "favored_display_label_from_margin",
    "load_demo_examples",
    "margin_direction_note",
    "mapping_as_display_dict",
    "validate_model_classes",
    "validate_input",
]
