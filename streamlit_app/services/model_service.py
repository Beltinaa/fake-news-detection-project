"""Cached thesis-model loading and inference for TruthLens."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import warnings

import joblib
import numpy as np
import streamlit as st

from streamlit_app.config import (
    LINEAR_SVM_ACCURACY_PERCENT,
    MODEL_NAME,
    TFIDF_FEATURE_COUNT,
    VECTORIZER_NAME,
)
from streamlit_app.utils.confidence import confidence_label_from_score, estimate_from_decision_score
from streamlit_app.utils.labels import (
    display_label_for_class,
    favored_class_id_from_margin,
    margin_direction_note,
    mapping_as_display_dict,
    validate_model_classes,
)
from streamlit_app.utils.paths import REQUIRED_MODEL_FILES
from streamlit_app.utils.validation import count_words


class ModelServiceError(RuntimeError):
    """Raised when the saved inference pipeline cannot be loaded or used."""


@dataclass(frozen=True)
class LoadedArtifacts:
    """Validated model artifacts required for inference."""

    vectorizer: Any
    model: Any
    model_classes: tuple[int, int]
    has_decision_function: bool
    vectorizer_feature_count: int
    vectorizer_name: str = VECTORIZER_NAME
    model_name: str = MODEL_NAME


def _artifact_paths(
    vectorizer_path: Path | None = None,
    model_path: Path | None = None,
) -> tuple[Path, Path]:
    resolved_vectorizer = vectorizer_path or REQUIRED_MODEL_FILES["vectorizer"]
    resolved_model = model_path or REQUIRED_MODEL_FILES["classifier"]
    return resolved_vectorizer, resolved_model


def _validate_loaded_objects(vectorizer: Any, model: Any) -> LoadedArtifacts:
    if not hasattr(vectorizer, "transform"):
        raise ModelServiceError(
            "The saved TF-IDF artifact is not compatible with the deployed inference pipeline."
        )

    if not hasattr(model, "predict"):
        raise ModelServiceError(
            "The saved classifier artifact is not compatible with the deployed inference pipeline."
        )

    try:
        model_classes = validate_model_classes(getattr(model, "classes_", None))
    except ValueError as exc:
        raise ModelServiceError(str(exc)) from exc

    vectorizer_feature_count = len(getattr(vectorizer, "vocabulary_", {})) or TFIDF_FEATURE_COUNT
    return LoadedArtifacts(
        vectorizer=vectorizer,
        model=model,
        model_classes=model_classes,
        has_decision_function=hasattr(model, "decision_function"),
        vectorizer_feature_count=vectorizer_feature_count,
    )


def load_artifacts(
    vectorizer_path: Path | None = None,
    model_path: Path | None = None,
) -> LoadedArtifacts:
    """Load and validate the saved thesis vectorizer and classifier."""
    resolved_vectorizer, resolved_model = _artifact_paths(
        vectorizer_path=vectorizer_path,
        model_path=model_path,
    )

    missing_names = [
        path.name for path in (resolved_vectorizer, resolved_model) if not path.exists()
    ]
    if missing_names:
        formatted = ", ".join(sorted(missing_names))
        raise ModelServiceError(
            "Required thesis model artifacts are missing. "
            f"Expected files: {formatted}."
        )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vectorizer = joblib.load(resolved_vectorizer)
            model = joblib.load(resolved_model)
    except Exception as exc:  # pragma: no cover - rare loader failure path
        raise ModelServiceError(
            "The saved thesis artifacts could not be loaded. "
            "Verify that the repository contains the original TF-IDF vectorizer and Linear SVM model."
        ) from exc

    return _validate_loaded_objects(vectorizer=vectorizer, model=model)


@st.cache_resource(show_spinner=False)
def load_cached_artifacts(
    vectorizer_path: str | None = None,
    model_path: str | None = None,
) -> LoadedArtifacts:
    """Cache the validated inference pipeline for the Streamlit runtime."""
    return load_artifacts(
        vectorizer_path=Path(vectorizer_path) if vectorizer_path else None,
        model_path=Path(model_path) if model_path else None,
    )


def _extract_decision_score(model: Any, features: Any) -> float | None:
    if not hasattr(model, "decision_function"):
        return None

    decision_values = np.asarray(model.decision_function(features)).ravel()
    if decision_values.size == 0:
        return None
    return float(decision_values[0])


def get_model_status() -> dict[str, str | bool]:
    """Return a user-friendly artifact health summary."""
    try:
        artifacts = load_cached_artifacts()
    except ModelServiceError as exc:
        return {"ready": False, "message": str(exc)}

    return {
        "ready": True,
        "message": (
            f"{artifacts.vectorizer_name} vectorizer and {artifacts.model_name} classifier loaded "
            "from the saved thesis artifacts. Verified mapping: 0=REAL, 1=FAKE."
        ),
    }


def predict_text(
    text: str,
    *,
    vectorizer: Any | None = None,
    model: Any | None = None,
    vectorizer_path: Path | None = None,
    model_path: Path | None = None,
) -> dict[str, Any]:
    """Run thesis-aligned inference on submitted text."""
    if vectorizer is None or model is None:
        artifacts = load_cached_artifacts(
            vectorizer_path=str(vectorizer_path) if vectorizer_path else None,
            model_path=str(model_path) if model_path else None,
        )
        vectorizer = artifacts.vectorizer
        model = artifacts.model
        has_decision_function = artifacts.has_decision_function
    else:
        artifacts = _validate_loaded_objects(vectorizer=vectorizer, model=model)
        has_decision_function = artifacts.has_decision_function

    try:
        transformed_text = vectorizer.transform([text])
        raw_prediction = model.predict(transformed_text)
    except Exception as exc:
        raise ModelServiceError(
            "The saved TF-IDF and Linear SVM pipeline could not classify the submitted text."
        ) from exc

    flattened_prediction = np.asarray(raw_prediction).ravel()
    if flattened_prediction.size == 0:
        raise ModelServiceError("The classifier returned an empty prediction.")

    numeric_label = int(flattened_prediction[0])
    try:
        prediction = display_label_for_class(numeric_label)
    except ValueError as exc:
        raise ModelServiceError(str(exc)) from exc

    decision_score = _extract_decision_score(model=model, features=transformed_text)
    favored_class_id = favored_class_id_from_margin(decision_score, artifacts.model_classes)
    if favored_class_id is not None and favored_class_id != numeric_label:
        raise ModelServiceError(
            "The classifier prediction and the Linear SVM decision score disagree. "
            "This indicates a model-consistency problem."
        )

    absolute_decision_score = abs(decision_score) if decision_score is not None else None
    confidence_estimate = estimate_from_decision_score(decision_score)
    confidence_label = confidence_label_from_score(decision_score)

    return {
        "prediction": prediction,
        "label": numeric_label,
        "class_mapping": mapping_as_display_dict(),
        "model_classes": list(artifacts.model_classes),
        "decision_score": decision_score if has_decision_function else None,
        "absolute_decision_score": absolute_decision_score,
        "margin_direction_note": margin_direction_note(artifacts.model_classes),
        "confidence_estimate": confidence_estimate,
        "confidence_label": confidence_label,
        "model": MODEL_NAME,
        "vectorizer": VECTORIZER_NAME,
        "vectorizer_feature_count": artifacts.vectorizer_feature_count,
        "test_accuracy": LINEAR_SVM_ACCURACY_PERCENT,
        "word_count": count_words(text),
    }
