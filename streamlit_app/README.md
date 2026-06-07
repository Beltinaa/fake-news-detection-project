# TruthLens Streamlit Application

## Purpose

This directory contains the Streamlit demonstration interface for the Bachelor thesis project
**TruthLens**. The app presents the saved TF-IDF + Linear SVM thesis pipeline in a structured,
presentation-ready UI.

## Pages

- `Detector`: submit English news text for classification
- `Methodology`: summarize the thesis experiment pipeline
- `Results`: compare evaluated models and show saved figures
- `About`: explain academic scope, limitations, and disclaimers

## Verified label mapping

The app uses the repository-verified mapping:

- `0 = REAL`
- `1 = FAKE`

The saved classifier exposes:

```text
model.classes_ = [0, 1]
```

Therefore:

- negative Linear SVM decision scores favor `REAL`
- positive Linear SVM decision scores favor `FAKE`

## Inference behavior

The production app uses:

```text
submitted text
→ models/tfidf_vectorizer.pkl
→ models/linear_svm.pkl
→ predicted label
```

Important constraints:

- no model retraining at startup
- no LLM classification
- no external fake-news API
- no database persistence
- no URL scraping in this version

## Confidence display

The UI shows a **Decision-strength estimate** derived from the absolute Linear SVM decision margin.
It is:

- bounded for display
- useful for comparing stronger vs. weaker margins
- **not** a calibrated probability
- distinct from the global held-out accuracy of `96.34%`

## Demo examples

The Detector page can load a small set of held-out demonstration samples from:

- [demo_examples.json](/Users/beltinaa/fake-news-detection-thesis/streamlit_app/demo_examples.json)

These samples come from the real saved test split and are intended only for model demonstration.

## Key files

- [app.py](/Users/beltinaa/fake-news-detection-thesis/streamlit_app/app.py)
- [config.py](/Users/beltinaa/fake-news-detection-thesis/streamlit_app/config.py)
- [services/model_service.py](/Users/beltinaa/fake-news-detection-thesis/streamlit_app/services/model_service.py)
- [components/detector.py](/Users/beltinaa/fake-news-detection-thesis/streamlit_app/components/detector.py)
- [components/result_display.py](/Users/beltinaa/fake-news-detection-thesis/streamlit_app/components/result_display.py)
- [utils/labels.py](/Users/beltinaa/fake-news-detection-thesis/streamlit_app/utils/labels.py)
- [utils/confidence.py](/Users/beltinaa/fake-news-detection-thesis/streamlit_app/utils/confidence.py)
- [tests/](/Users/beltinaa/fake-news-detection-thesis/streamlit_app/tests)

## Run locally

From the repository root:

```bash
source venv/bin/activate
streamlit run streamlit_app/app.py
```

## Run tests

From the repository root:

```bash
source venv/bin/activate
pytest -v
```

## Notes

- Path handling is repository-relative via `pathlib.Path`.
- Recent analyses are stored only in `st.session_state` and capped at five entries.
- The app fails clearly if the loaded model classes do not match the verified thesis mapping.
