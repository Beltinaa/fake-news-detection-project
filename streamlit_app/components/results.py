"""Results page for TruthLens."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from streamlit_app.config import LINEAR_SVM_ACCURACY_PERCENT
from streamlit_app.utils.paths import FIGURES_DIR


RESULTS_TABLE = pd.DataFrame(
    [
        {
            "Model": "Linear SVM",
            "Accuracy": "96.34%",
            "Approximate F1-score": "0.96",
            "Interpretation": "Best-performing model",
        },
        {
            "Model": "Logistic Regression",
            "Accuracy": "94.90%",
            "Approximate F1-score": "0.95",
            "Interpretation": "Strong linear baseline",
        },
        {
            "Model": "Random Forest",
            "Accuracy": "94.08%",
            "Approximate F1-score": "0.94",
            "Interpretation": "Competitive ensemble model",
        },
        {
            "Model": "Naive Bayes",
            "Accuracy": "86.81%",
            "Approximate F1-score": "0.87",
            "Interpretation": "Fast but weaker",
        },
    ]
)


FIGURE_DETAILS = [
    {
        "file": "confusion_matrix_svm.png",
        "title": "Figure 1. Confusion Matrix - Linear SVM",
        "interpretation": (
            "Linear SVM yields the strongest error profile in the thesis experiment, with 6,691 "
            "real articles and 7,200 fake articles correctly classified once the verified label "
            "mapping is applied."
        ),
    },
    {
        "file": "confusion_matrix_lr.png",
        "title": "Figure 2. Confusion Matrix - Logistic Regression",
        "interpretation": (
            "Logistic Regression performs strongly as a linear baseline and shows a balanced "
            "confusion structure, but still trails Linear SVM overall."
        ),
    },
    {
        "file": "confusion_matrix_rf.png",
        "title": "Figure 3. Confusion Matrix - Random Forest",
        "interpretation": (
            "Random Forest remains competitive, but sparse high-dimensional textual features are "
            "typically more efficient for linear models than tree ensembles."
        ),
    },
    {
        "file": "confusion_matrix_nb.png",
        "title": "Figure 4. Confusion Matrix - Naive Bayes",
        "interpretation": (
            "Naive Bayes is fast and simple, but its conditional-independence assumption limits "
            "performance on richer textual patterns."
        ),
    },
    {
        "file": "roc_curves_comparison.png",
        "title": "Figure 5. ROC Curve Comparison of Machine Learning Models",
        "interpretation": (
            "ROC comparison provides threshold-based discrimination context, complementing accuracy "
            "and class-wise Precision, Recall, and F1-score."
        ),
    },
]


def _render_figure(path: Path, title: str, interpretation: str) -> None:
    if not path.exists():
        st.warning(f"{path.name} is not available in the repository figures folder.")
        return

    st.markdown(f"**{title}**")
    image_columns = st.columns([0.08, 0.84, 0.08])
    with image_columns[1]:
        st.image(str(path), use_column_width=True)
    st.caption(interpretation)


def render_results_page() -> None:
    """Render the thesis results page."""
    st.markdown("## Results")
    st.caption("Performance values and figures below follow the saved notebook outputs and figures.")

    st.table(RESULTS_TABLE)

    st.markdown(
        """
        Linear SVM performs especially well on high-dimensional sparse TF-IDF features because the
        separating hyperplane can exploit informative lexical signals without the computational cost
        of dense neural representations. Logistic Regression remains a strong baseline for the same
        reason: it is stable, interpretable, and well suited to sparse text vectors.

        Naive Bayes is computationally efficient, but its conditional-independence assumption often
        limits accuracy on complex article language. Random Forest can capture nonlinear patterns,
        yet it is generally more computationally expensive and less naturally matched to very large
        sparse feature spaces.

        Accuracy alone is not sufficient for thesis evaluation. Precision, Recall, and F1-score are
        important because they describe how the classifier behaves on each class, especially when
        false positives and false negatives carry different interpretive risks.

        The verified saved model uses `model.classes_ = [0, 1]`, which in this repository maps to
        `REAL` and `FAKE` respectively. That means negative Linear SVM decision scores favor
        `REAL`, and positive scores favor `FAKE`.
        """
    )

    st.markdown("### Linear SVM confusion-matrix breakdown")
    st.markdown(
        """
        - `6691` REAL articles correctly classified as REAL
        - `315` REAL articles classified as FAKE
        - `213` FAKE articles classified as REAL
        - `7200` FAKE articles correctly classified as FAKE
        """
    )
    st.caption(
        f"Linear SVM held-out accuracy reproduced from the saved model artifacts: "
        f"{LINEAR_SVM_ACCURACY_PERCENT:.2f}%."
    )

    st.markdown("### Saved evaluation figures")
    for figure in FIGURE_DETAILS:
        _render_figure(
            path=FIGURES_DIR / figure["file"],
            title=figure["title"],
            interpretation=figure["interpretation"],
        )
