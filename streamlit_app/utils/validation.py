"""Input validation helpers for the detector page."""

from __future__ import annotations

from dataclasses import dataclass, field


MAX_INPUT_CHARS = 25_000
SHORT_TEXT_WORD_THRESHOLD = 5
SHORT_TEXT_CHAR_THRESHOLD = 40


def count_words(text: str) -> int:
    """Count whitespace-delimited words in normalized text."""
    return len([token for token in text.split() if token])


@dataclass(frozen=True)
class ValidationResult:
    """Structured validation result for detector submissions."""

    normalized_text: str
    word_count: int
    char_count: int
    errors: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_valid(self) -> bool:
        return not self.errors


def validate_input(text: str | None, max_chars: int = MAX_INPUT_CHARS) -> ValidationResult:
    """Validate user input without changing the thesis preprocessing pipeline."""
    raw_text = text or ""
    normalized_text = raw_text.strip()
    char_count = len(raw_text)
    word_count = count_words(normalized_text)
    errors: list[str] = []
    warnings: list[str] = []

    if not raw_text or not normalized_text:
        errors.append("Please enter English news text before running the classifier.")

    if char_count > max_chars:
        errors.append(
            f"Input exceeds the safe maximum length of {max_chars:,} characters. "
            "Please shorten the article text and try again."
        )

    if normalized_text and (
        word_count < SHORT_TEXT_WORD_THRESHOLD or len(normalized_text) < SHORT_TEXT_CHAR_THRESHOLD
    ):
        warnings.append(
            "Very short input can be classified, but longer article text usually produces a more "
            "reliable statistical result than a single short phrase."
        )

    return ValidationResult(
        normalized_text=normalized_text,
        word_count=word_count,
        char_count=char_count,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )
