"""Central verified configuration for the TruthLens application."""

from __future__ import annotations


APP_NAME = "TruthLens"
APP_SUBTITLE = "Machine Learning-Based Fake News Detection"
DATASET_NAME = "WELFake"
MODEL_NAME = "Linear SVM"
VECTORIZER_NAME = "TF-IDF"

CLEANED_ARTICLE_COUNT = 72_095
TRAIN_SAMPLE_COUNT = 57_676
TEST_SAMPLE_COUNT = 14_419
TFIDF_FEATURE_COUNT = 220_275

LINEAR_SVM_ACCURACY = 0.9633816492128442
LINEAR_SVM_ACCURACY_PERCENT = 96.34
LINEAR_SVM_CONFUSION_MATRIX = (
    (6691, 315),
    (213, 7200),
)

# These thresholds are derived from the held-out test-set distribution of
# absolute Linear SVM decision margins among correctly classified samples:
# - 50th percentile: 1.3262801441810486
# - 90th percentile: 2.2964537111040015
CONFIDENCE_MARGIN_THRESHOLDS = {
    "moderate_lower": 1.3262801441810486,
    "high_lower": 2.2964537111040015,
}

DEMO_EXAMPLE_SOURCE = "Held-out WELFake test split"
