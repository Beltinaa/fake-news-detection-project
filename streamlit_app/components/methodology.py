"""Methodology page content for TruthLens."""

from __future__ import annotations

import streamlit as st


from streamlit_app.config import (
    TEST_SAMPLE_COUNT,
    TFIDF_FEATURE_COUNT,
    TRAIN_SAMPLE_COUNT,
)


def render_methodology_page() -> None:
    """Render the thesis methodology overview."""
    st.markdown("## Methodology")
    st.caption(
        "This page summarizes the experiment pipeline recorded in `notebooks/eda.ipynb` and the "
        "saved thesis artifacts."
    )

    st.markdown("### Dataset")
    st.markdown(
        """
        The thesis uses the WELFake dataset for binary fake-news classification. After the
        notebook cleaning steps, the final dataset contains **72,095** articles with the retained
        columns `title`, `text`, `label`, and `content`. The binary label encoding is kept
        consistent throughout the repository:

        - `0 = REAL`
        - `1 = FAKE`

        Because WELFake is a curated benchmark dataset, its class balance, source coverage, and
        topical distribution can introduce source bias or topic bias that may not generalize to
        every real-world news domain.
        """
    )

    st.markdown("### Data preparation")
    st.markdown(
        """
        The executed notebook combines article title and body text into a single input field:
        `content = title + " " + text`. Missing title values are filled with an empty string before
        concatenation, missing `text` rows are removed, and the serial/index column is dropped
        when present. No stemming, lemmatization, punctuation stripping, or manual stopword
        removal is applied outside the saved TF-IDF vectorizer in the deployed inference pipeline.
        """
    )

    st.markdown("### Train/test split")
    st.markdown(
        f"""
        The dataset is split with an **80/20** train/test strategy using:

        - `test_size=0.2`
        - `random_state=42`
        - `stratify=y`

        This yields **{TRAIN_SAMPLE_COUNT:,}** training samples and **{TEST_SAMPLE_COUNT:,}** testing samples. The fixed random
        state supports reproducibility, while stratification preserves the class ratio in both
        partitions for fair model comparison.
        """
    )

    st.markdown("### TF-IDF")
    st.markdown(
        f"""
        The thesis vectorizer is:

        - `TfidfVectorizer(stop_words="english", max_df=0.7)`

        TF-IDF increases the weight of terms that are informative for a document while reducing the
        influence of very common words. The vectorizer is fitted on the training split only, which
        helps prevent data leakage. The resulting training matrix has shape **({TRAIN_SAMPLE_COUNT:,}, {TFIDF_FEATURE_COUNT:,})**,
        representing a high-dimensional sparse feature space suitable for linear text classifiers.
        """
    )

    st.markdown("### Models")
    st.markdown(
        """
        Four classical machine-learning models were evaluated in the notebook:

        - Logistic Regression
        - Multinomial Naive Bayes
        - Linear SVM using `LinearSVC`
        - Random Forest
        """
    )

    st.markdown("### Metrics")
    st.markdown(
        """
        The thesis evaluation uses Accuracy, Precision, Recall, F1-score, Confusion Matrix, ROC
        Curve, and AUC. Accuracy gives an overall correctness rate, but Precision and Recall help
        explain class-specific tradeoffs, while F1-score balances both. The confusion matrix shows
        where false positives and false negatives occur, and ROC/AUC summarize ranking behavior
        across decision thresholds.
        """
    )

    st.markdown("### Final deployment pipeline")
    st.markdown(
        """
        The Streamlit application does not retrain any model. During inference it loads the saved
        `models/tfidf_vectorizer.pkl` and `models/linear_svm.pkl` artifacts with `joblib`, applies
        the vectorizer directly to submitted text, and uses the saved Linear SVM to generate the
        final binary prediction. The verified saved model exposes `model.classes_ = [0, 1]`, so
        negative decision scores favor `REAL` and positive decision scores favor `FAKE`.
        """
    )
