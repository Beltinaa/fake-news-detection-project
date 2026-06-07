"""Shared helpers for saved-model consistency auditing and demo extraction."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any
import warnings

import joblib
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from streamlit_app.config import DEMO_EXAMPLE_SOURCE  # noqa: E402
from streamlit_app.utils.confidence import confidence_label_from_score, estimate_from_decision_score  # noqa: E402
from streamlit_app.utils.labels import (  # noqa: E402
    display_label_for_class,
    favored_class_id_from_margin,
    validate_model_classes,
)
from streamlit_app.utils.paths import (  # noqa: E402
    DEMO_EXAMPLES_FILE,
    PROCESSED_DATA_FILE,
    RAW_DATA_FILE,
    REQUIRED_MODEL_FILES,
    SPLITS_DIR,
)


@dataclass(frozen=True)
class ConsistencyContext:
    """Loaded repository artifacts needed for consistency analysis."""

    raw_dataset: pd.DataFrame
    processed_dataset: pd.DataFrame
    x_train: pd.Series
    x_test: pd.Series
    y_train: pd.Series
    y_test: pd.Series
    vectorizer: Any
    model: Any
    model_classes: tuple[int, int]


def _read_split_csv(name: str) -> pd.Series:
    frame = pd.read_csv(SPLITS_DIR / name)
    return frame.iloc[:, 0]


def _load_joblib(path: Path) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return joblib.load(path)


def load_consistency_context() -> ConsistencyContext:
    """Load datasets, splits, vectorizer, and saved Linear SVM classifier."""
    raw_dataset = pd.read_csv(RAW_DATA_FILE)
    processed_dataset = pd.read_csv(PROCESSED_DATA_FILE)
    x_train = _read_split_csv("X_train.csv")
    x_test = _read_split_csv("X_test.csv")
    y_train = _read_split_csv("y_train.csv").astype(int)
    y_test = _read_split_csv("y_test.csv").astype(int)
    vectorizer = _load_joblib(REQUIRED_MODEL_FILES["vectorizer"])
    model = _load_joblib(REQUIRED_MODEL_FILES["classifier"])
    model_classes = validate_model_classes(getattr(model, "classes_", None))

    return ConsistencyContext(
        raw_dataset=raw_dataset,
        processed_dataset=processed_dataset,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        vectorizer=vectorizer,
        model=model,
        model_classes=model_classes,
    )


def build_evaluation_frame(context: ConsistencyContext) -> pd.DataFrame:
    """Create a per-sample evaluation table for the held-out test set."""
    transformed = context.vectorizer.transform(context.x_test.astype(str))
    predictions = context.model.predict(transformed)
    decision_scores = np.asarray(context.model.decision_function(transformed)).ravel()

    evaluation = pd.DataFrame(
        {
            "text": context.x_test.astype(str).tolist(),
            "actual_label_id": context.y_test.to_numpy(dtype=int),
            "predicted_label_id": predictions.astype(int),
            "decision_score": decision_scores.astype(float),
        }
    )
    evaluation["absolute_score"] = evaluation["decision_score"].abs()
    evaluation["is_correct"] = evaluation["actual_label_id"] == evaluation["predicted_label_id"]
    evaluation["actual_label"] = evaluation["actual_label_id"].map(display_label_for_class)
    evaluation["predicted_label"] = evaluation["predicted_label_id"].map(display_label_for_class)
    evaluation["margin_favored_class_id"] = evaluation["decision_score"].apply(
        lambda score: favored_class_id_from_margin(score, context.model_classes)
    )
    evaluation["margin_favored_label"] = evaluation["margin_favored_class_id"].map(
        lambda class_id: display_label_for_class(class_id) if class_id is not None else None
    )
    evaluation["confidence_estimate"] = evaluation["decision_score"].apply(estimate_from_decision_score)
    evaluation["confidence_category"] = evaluation["decision_score"].apply(confidence_label_from_score)
    return evaluation


def distribution_summary(series: pd.Series | np.ndarray) -> dict[str, float]:
    """Return descriptive statistics used for confidence auditing."""
    values = np.asarray(series, dtype=float)
    percentiles = np.percentile(values, [0, 25, 50, 75, 90, 95, 99, 100])
    return {
        "min": float(percentiles[0]),
        "p25": float(percentiles[1]),
        "median": float(percentiles[2]),
        "p75": float(percentiles[3]),
        "p90": float(percentiles[4]),
        "p95": float(percentiles[5]),
        "p99": float(percentiles[6]),
        "max": float(percentiles[7]),
    }


def representative_rows(dataset: pd.DataFrame, label: int, sample_size: int = 3) -> list[dict[str, Any]]:
    """Return small label-specific samples for auditing."""
    records: list[dict[str, Any]] = []
    sample = dataset.loc[dataset["label"] == label, ["title", "text"]].head(sample_size)
    for _, row in sample.iterrows():
        records.append(
            {
                "title": _normalize_text(row["title"])[:180],
                "text": _normalize_text(row["text"])[:260],
            }
        )
    return records


def _normalize_text(text: Any) -> str:
    return " ".join(str(text).split())


def _record_from_row(name: str, row: pd.Series) -> dict[str, Any]:
    return {
        "name": name,
        "text": _normalize_text(row["text"]),
        "actualNumericLabel": int(row["actual_label_id"]),
        "predictedNumericLabel": int(row["predicted_label_id"]),
        "actualLabel": str(row["actual_label"]),
        "predictedLabel": str(row["predicted_label"]),
        "decisionScore": float(row["decision_score"]),
        "absoluteScore": float(row["absolute_score"]),
        "confidenceEstimate": float(row["confidence_estimate"]),
        "confidenceCategory": str(row["confidence_category"]),
        "source": DEMO_EXAMPLE_SOURCE,
    }


def select_demo_examples(evaluation: pd.DataFrame) -> list[dict[str, Any]]:
    """Extract a small, verified demo set from the held-out test split."""
    case_definitions = [
        (
            "High-confidence REAL example",
            lambda frame: frame[
                (frame["is_correct"]) & (frame["actual_label_id"] == 0) & (frame["predicted_label_id"] == 0)
            ].sort_values("decision_score", ascending=True),
        ),
        (
            "High-confidence FAKE example",
            lambda frame: frame[
                (frame["is_correct"]) & (frame["actual_label_id"] == 1) & (frame["predicted_label_id"] == 1)
            ].sort_values("decision_score", ascending=False),
        ),
        (
            "Borderline REAL example",
            lambda frame: frame[
                (frame["is_correct"]) & (frame["actual_label_id"] == 0) & (frame["predicted_label_id"] == 0)
            ].sort_values("absolute_score", ascending=True),
        ),
        (
            "Borderline FAKE example",
            lambda frame: frame[
                (frame["is_correct"]) & (frame["actual_label_id"] == 1) & (frame["predicted_label_id"] == 1)
            ].sort_values("absolute_score", ascending=True),
        ),
        (
            "False positive for FAKE",
            lambda frame: frame[
                (~frame["is_correct"]) & (frame["actual_label_id"] == 0) & (frame["predicted_label_id"] == 1)
            ].sort_values("absolute_score", ascending=True),
        ),
        (
            "False negative for FAKE",
            lambda frame: frame[
                (~frame["is_correct"]) & (frame["actual_label_id"] == 1) & (frame["predicted_label_id"] == 0)
            ].sort_values("absolute_score", ascending=True),
        ),
    ]

    selected: list[dict[str, Any]] = []
    for name, selector in case_definitions:
        subset = selector(evaluation)
        if subset.empty:
            raise RuntimeError(f"Could not find a held-out sample for '{name}'.")
        selected.append(_record_from_row(name=name, row=subset.iloc[0]))

    return selected


def write_demo_examples(examples: list[dict[str, Any]]) -> Path:
    """Write selected demo examples to the Streamlit app directory."""
    DEMO_EXAMPLES_FILE.write_text(json.dumps(examples, indent=2, ensure_ascii=False))
    return DEMO_EXAMPLES_FILE
