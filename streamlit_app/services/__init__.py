"""Inference services for TruthLens."""

from .model_service import (
    MODEL_NAME,
    VECTORIZER_NAME,
    ModelServiceError,
    get_model_status,
    load_artifacts,
    load_cached_artifacts,
    predict_text,
)

__all__ = [
    "MODEL_NAME",
    "VECTORIZER_NAME",
    "ModelServiceError",
    "get_model_status",
    "load_artifacts",
    "load_cached_artifacts",
    "predict_text",
]
