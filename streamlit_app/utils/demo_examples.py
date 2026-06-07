"""Demo-example loading helpers."""

from __future__ import annotations

import json
from typing import Any

from streamlit_app.utils.paths import DEMO_EXAMPLES_FILE


def load_demo_examples() -> list[dict[str, Any]]:
    """Load sanitized demo examples extracted from the held-out test split."""
    if not DEMO_EXAMPLES_FILE.exists():
        return []

    data = json.loads(DEMO_EXAMPLES_FILE.read_text())
    if not isinstance(data, list):
        raise ValueError("Demo examples file must contain a JSON list.")

    return [dict(item) for item in data]
