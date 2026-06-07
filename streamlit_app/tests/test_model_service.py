"""Model service tests for TruthLens."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pytest

from src.consistency_utils import load_consistency_context
from streamlit_app.services.model_service import ModelServiceError, load_artifacts, predict_text


class DummyVectorizer:
    def transform(self, texts):
        return np.asarray([[len(texts[0].split())]])


class DummyModel:
    classes_ = np.array([0, 1])

    def __init__(self, prediction: int = 1, score: float = 2.5):
        self.prediction = prediction
        self.score = score

    def predict(self, transformed_text):
        return np.asarray([self.prediction])

    def decision_function(self, transformed_text):
        return np.asarray([self.score])


class InvalidModel:
    classes_ = np.array([0, 1])


class WrongClassOrderModel:
    classes_ = np.array([1, 0])

    def predict(self, transformed_text):
        return np.asarray([1])


def test_predict_text_returns_expected_schema():
    result = predict_text(
        "Example article text for inference.",
        vectorizer=DummyVectorizer(),
        model=DummyModel(prediction=1, score=2.5),
    )

    expected_keys = {
        "prediction",
        "label",
        "class_mapping",
        "model_classes",
        "decision_score",
        "absolute_decision_score",
        "confidence_estimate",
        "confidence_label",
        "model",
        "vectorizer",
        "vectorizer_feature_count",
        "test_accuracy",
        "word_count",
    }
    assert expected_keys.issubset(result.keys())
    assert result["prediction"] == "FAKE"
    assert result["label"] == 1
    assert result["decision_score"] == 2.5
    assert result["absolute_decision_score"] == 2.5


def test_prediction_and_margin_sign_mismatch_raises_error():
    with pytest.raises(ModelServiceError, match="decision score disagree"):
        predict_text(
            "Example article text for inference.",
            vectorizer=DummyVectorizer(),
            model=DummyModel(prediction=1, score=-0.5),
        )


def test_missing_model_artifacts_raise_friendly_error(tmp_path: Path):
    with pytest.raises(ModelServiceError, match="Required thesis model artifacts are missing"):
        load_artifacts(
            vectorizer_path=tmp_path / "missing_vectorizer.pkl",
            model_path=tmp_path / "missing_model.pkl",
        )


def test_invalid_model_object_is_rejected(tmp_path: Path):
    vectorizer_path = tmp_path / "vectorizer.pkl"
    model_path = tmp_path / "model.pkl"
    joblib.dump(DummyVectorizer(), vectorizer_path)
    joblib.dump(InvalidModel(), model_path)

    with pytest.raises(ModelServiceError, match="saved classifier artifact"):
        load_artifacts(vectorizer_path=vectorizer_path, model_path=model_path)


def test_unexpected_model_class_order_is_rejected(tmp_path: Path):
    vectorizer_path = tmp_path / "vectorizer.pkl"
    model_path = tmp_path / "model.pkl"
    joblib.dump(DummyVectorizer(), vectorizer_path)
    joblib.dump(WrongClassOrderModel(), model_path)

    with pytest.raises(ModelServiceError, match="Unexpected model.classes_ order"):
        load_artifacts(vectorizer_path=vectorizer_path, model_path=model_path)


def test_real_model_classes_match_verified_mapping():
    context = load_consistency_context()
    assert list(context.model_classes) == [0, 1]
