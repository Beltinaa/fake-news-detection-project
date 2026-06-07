"""Regenerate saved confusion-matrix figures with verified label ordering."""

from __future__ import annotations

from pathlib import Path
import sys
import warnings

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from streamlit_app.utils.labels import display_labels_for_classes, validate_model_classes  # noqa: E402
from streamlit_app.utils.paths import FIGURES_DIR, MODELS_DIR, SPLITS_DIR  # noqa: E402


MODEL_FILES = [
    ("logistic_regression.pkl", "Confusion Matrix - Logistic Regression", "confusion_matrix_lr.png"),
    ("naive_bayes.pkl", "Confusion Matrix - Naive Bayes", "confusion_matrix_nb.png"),
    ("linear_svm.pkl", "Confusion Matrix - Linear SVM", "confusion_matrix_svm.png"),
    ("random_forest.pkl", "Confusion Matrix - Random Forest", "confusion_matrix_rf.png"),
]


def _load_joblib(path: Path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return joblib.load(path)


def main() -> int:
    x_test = pd.read_csv(SPLITS_DIR / "X_test.csv").iloc[:, 0]
    y_test = pd.read_csv(SPLITS_DIR / "y_test.csv").iloc[:, 0]
    vectorizer = _load_joblib(MODELS_DIR / "tfidf_vectorizer.pkl")
    x_test_tfidf = vectorizer.transform(x_test)

    for model_file, title, output_name in MODEL_FILES:
        model = _load_joblib(MODELS_DIR / model_file)
        model_classes = validate_model_classes(getattr(model, "classes_", None))
        predictions = model.predict(x_test_tfidf)
        matrix = confusion_matrix(y_test, predictions)
        tick_labels = display_labels_for_classes(model_classes)

        plt.figure(figsize=(6, 4))
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=tick_labels,
            yticklabels=tick_labels,
        )
        plt.xlabel("Predicted label")
        plt.ylabel("Actual label")
        plt.title(title)
        plt.savefig(FIGURES_DIR / output_name, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Updated {output_name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
