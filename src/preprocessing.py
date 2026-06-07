"""Experimental preprocessing helper.

This function is not used by the deployed TruthLens inference pipeline. The
saved thesis vectorizer is applied directly at prediction time to remain
consistent with the notebook experiment.
"""

import string
from functools import lru_cache

from nltk.corpus import stopwords


@lru_cache(maxsize=1)
def _english_stopwords():
    try:
        return set(stopwords.words("english"))
    except LookupError as exc:  # pragma: no cover - depends on local NLTK data
        raise RuntimeError(
            "NLTK English stopwords are required to use src.preprocessing.clean_text."
        ) from exc


def clean_text(text):
    if not isinstance(text, str):
        return ""

    stop_words = _english_stopwords()
    text = text.lower()
    text = "".join(char for char in text if char not in string.punctuation)
    text = " ".join(word for word in text.split() if word not in stop_words)
    return text
