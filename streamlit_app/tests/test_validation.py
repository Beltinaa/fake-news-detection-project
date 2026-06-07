"""Validation tests for TruthLens detector input handling."""

from streamlit_app.utils.validation import MAX_INPUT_CHARS, validate_input


def test_empty_text_is_rejected():
    result = validate_input("")
    assert not result.is_valid
    assert "Please enter English news text" in result.errors[0]


def test_whitespace_only_text_is_rejected():
    result = validate_input("   \n\t  ")
    assert not result.is_valid
    assert result.normalized_text == ""


def test_very_short_text_triggers_warning_but_remains_valid():
    result = validate_input("Fake news?")
    assert result.is_valid
    assert result.warnings


def test_text_above_maximum_length_is_rejected():
    oversized = "a" * (MAX_INPUT_CHARS + 1)
    result = validate_input(oversized)
    assert not result.is_valid
    assert "safe maximum length" in result.errors[0]
