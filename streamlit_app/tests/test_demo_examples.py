"""Held-out demo example integrity tests."""

from __future__ import annotations

from pathlib import Path

from streamlit_app.services.model_service import predict_text
from streamlit_app.utils.demo_examples import load_demo_examples
from streamlit_app.utils.paths import DEMO_EXAMPLES_FILE


def _example_by_name(name: str) -> dict[str, object]:
    examples = load_demo_examples()
    return next(example for example in examples if example["name"] == name)


def test_demo_examples_file_exists_and_is_valid():
    assert DEMO_EXAMPLES_FILE.exists()
    examples = load_demo_examples()
    assert isinstance(examples, list)
    assert len(examples) == 6


def test_high_confidence_fake_example_is_consistent():
    example = _example_by_name("High-confidence FAKE example")
    assert example["actualLabel"] == "FAKE"
    assert example["predictedLabel"] == "FAKE"
    result = predict_text(str(example["text"]))
    assert result["prediction"] == "FAKE"


def test_high_confidence_real_example_is_consistent():
    example = _example_by_name("High-confidence REAL example")
    assert example["actualLabel"] == "REAL"
    assert example["predictedLabel"] == "REAL"
    result = predict_text(str(example["text"]))
    assert result["prediction"] == "REAL"


def test_borderline_examples_preserve_verified_labels():
    real_example = _example_by_name("Borderline REAL example")
    fake_example = _example_by_name("Borderline FAKE example")
    assert real_example["actualLabel"] == "REAL"
    assert real_example["predictedLabel"] == "REAL"
    assert fake_example["actualLabel"] == "FAKE"
    assert fake_example["predictedLabel"] == "FAKE"
