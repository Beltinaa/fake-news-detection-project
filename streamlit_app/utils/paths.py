"""Repository-relative path helpers for the TruthLens app."""

from pathlib import Path


STREAMLIT_APP_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
ASSETS_DIR = STREAMLIT_APP_DIR / "assets"
DEMO_EXAMPLES_FILE = STREAMLIT_APP_DIR / "demo_examples.json"
FIGURES_DIR = REPO_ROOT / "figures"
MODELS_DIR = REPO_ROOT / "models"
SRC_DIR = REPO_ROOT / "src"
RAW_DATA_FILE = RAW_DATA_DIR / "WELFake_Dataset.csv"
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "welfake_clean.csv"

REQUIRED_MODEL_FILES = {
    "vectorizer": MODELS_DIR / "tfidf_vectorizer.pkl",
    "classifier": MODELS_DIR / "linear_svm.pkl",
}
