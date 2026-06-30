# TruthLens: Fake News Detection Thesis Project

Practical Bachelor thesis repository for:

- **Universiteti Politeknik i Tiranës**
- **Fakulteti i Teknologjisë së Informacionit**
- **Departamenti i Inxhinierisë Informatike**
- **Programi:** Inxhinieri Informatike
- **Student:** Beltina Manallari
- **Thesis title:** *Zbulimi i Lajmeve të Rreme duke përdorur teknika të Machine Learning dhe Natural Language Processing*

## Problem statement

TruthLens studies whether classical Machine Learning and Natural Language Processing techniques can
classify English-language news text from the WELFake benchmark dataset as `REAL` or `FAKE`. The
practical objective is to compare five TF-IDF-based models (four classical plus one neural network)
and deploy the best-performing saved classifier in a Streamlit application for academic
demonstration. An additional ablation study evaluates the impact of Stanza lemmatization on all
five models.

## Academic source of truth

The executed notebook [notebooks/eda.ipynb](notebooks/eda.ipynb)
is the experiment source of truth for:

- data cleaning
- label usage
- train/test split
- TF-IDF configuration
- model training (Logistic Regression, Naive Bayes, Linear SVM, Random Forest)
- saved artifact filenames
- reported evaluation outputs

[notebooks/mlp_model.ipynb](notebooks/mlp_model.ipynb) is the source of truth for:

- MLP Neural Network training and evaluation
- chi² feature selection (30,000 features)
- MLP hyperparameters

[notebooks/lemmatizer_comparison.ipynb](notebooks/lemmatizer_comparison.ipynb) is the source of
truth for:

- Stanza lemmatization pipeline
- lemmatized TF-IDF vectorizer
- with-lemmatization results for all five models

The production application does **not** retrain at startup. It loads the saved artifacts:

- `models/tfidf_vectorizer.pkl`
- `models/linear_svm.pkl`

## Dataset description

- Dataset: **WELFake**
- Raw source file: `data/raw/WELFake_Dataset.csv`
- Final cleaned dataset size: **72,095**
- Retained columns: `title`, `text`, `label`, `content`
- Combined model input: `content = title + " " + text`

## Data-preparation summary

The notebook:

1. loads `WELFake_Dataset.csv`
2. creates `content` from `title` and `text`
3. fills missing titles with an empty string before concatenation
4. drops rows with missing `text`
5. drops the serial/index column when present
6. saves `data/processed/welfake_clean.csv`

The deployed inference path does **not** add extra stemming, lemmatization, punctuation stripping,
or manual stopword removal outside the saved TF-IDF vectorizer.

## Train/test split

The saved experiment uses:

- `test_size=0.2`
- `random_state=42`
- `stratify=y`

Saved split sizes:

- Training samples: **57,676**
- Testing samples: **14,419**

These split artifacts are preserved in `data/splits/` for reproducibility, but they are not
required by the Streamlit app at runtime.

## TF-IDF configuration

```python
TfidfVectorizer(
    stop_words="english",
    max_df=0.7
)
```

Verified training matrix shape:

```text
(57676, 220275)
```

## Models evaluated

1. Logistic Regression
2. Multinomial Naive Bayes
3. Linear SVM using `LinearSVC`
4. Random Forest
5. MLP Neural Network (with chi² feature selection, k=30,000)

## Verified results — without lemmatization

| Model              | Accuracy | F1-score | AUC   | Interpretation              |
|--------------------|--------: |---------:|------:|-----------------------------|
| Linear SVM         | 96.34%   | 0.9634   | 0.994 | Best-performing model       |
| MLP Neural Network | 95.74%   | 0.9574   | 0.991 | Second-best, neural network |
| Logistic Regression| 94.90%   | 0.9489   | 0.988 | Strong linear baseline      |
| Random Forest      | 94.08%   | 0.9407   | 0.989 | Competitive ensemble model  |
| Naive Bayes        | 86.81%   | 0.8681   | 0.940 | Fast but weaker             |

Exact notebook-aligned accuracy values:

- Logistic Regression: `0.9489562382966918`
- Naive Bayes: `0.8680907136417227`
- Linear SVM: `0.9633816492128442`
- Random Forest: `0.9407725917192593`
- MLP Neural Network: `0.9574172966225120`

## Verified results — with Stanza lemmatization

All five models were retrained on lemmatized text using the same train/test split and
hyperparameters. Results:

| Model              | Accuracy (no lemma) | Accuracy (lemma) | Δ Accuracy | AUC (lemma) |
|--------------------|--------------------:|-----------------:|-----------:|------------:|
| Linear SVM         | 0.9634              | **0.9739**       | +0.0105    | 0.997       |
| MLP Neural Network | 0.9574              | **0.9678**       | +0.0104    | 0.995       |
| Logistic Regression| 0.9490              | **0.9619**       | +0.0129    | 0.993       |
| Random Forest      | 0.9408              | **0.9505**       | +0.0097    | 0.991       |
| Naive Bayes        | 0.8681              | **0.8774**       | +0.0093    | 0.948       |

Lemmatization improves all five models consistently (~+1.06 pp average Accuracy gain) with no
metric regressions. Lemmatized artifacts are saved separately and do not overwrite the primary
pipeline artifacts.

## Model configurations

```python
# Logistic Regression
LogisticRegression(max_iter=1000, random_state=42)

# Multinomial Naive Bayes
MultinomialNB()

# Linear SVM
LinearSVC(random_state=42)

# Random Forest
RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)

# MLP Neural Network (after chi² feature selection, k=30000)
MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation='relu',
    solver='adam',
    alpha=0.0003,
    batch_size=512,
    learning_rate='adaptive',
    learning_rate_init=0.002,
    max_iter=40,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=4,
    random_state=42
)
```

MLP training converged after **7 iterations** (early stopping) with final training loss `0.002972`.

## Verified Linear SVM confusion matrix

Class order confirmed as `[0, 1] = [REAL, FAKE]`:

```text
[[6691  315]
 [ 213 7200]]
```

Interpreted by label:

- `6691` REAL articles correctly classified as REAL
- `315` REAL articles classified as FAKE
- `213` FAKE articles classified as REAL
- `7200` FAKE articles correctly classified as FAKE

## MLP confusion matrix

Class order `[0, 1] = [REAL, FAKE]`:

```text
[[6702  304]
 [ 310 7103]]
```

## Label-mapping verification

The repository audit confirmed that the production mapping is:

- `0 = REAL`
- `1 = FAKE`

Evidence:

- representative label `0` rows in the raw dataset are Reuters/AP-style factual news articles
- representative label `1` rows are sensational or conspiratorial fake-news examples
- processed and split label counts preserve the same numeric values
- the saved Linear SVM exposes `model.classes_ = [0, 1]`
- reproducing the held-out test predictions matches the notebook confusion matrix exactly

The previous Streamlit UI mapping was reversed and has been corrected.

## Decision-score interpretation

The saved classifier is a binary `LinearSVC` with:

```text
model.classes_ = [0, 1]
```

For this saved model:

- negative `decision_function()` scores favor class `0` → `REAL`
- positive `decision_function()` scores favor class `1` → `FAKE`

This relation was verified programmatically across the held-out test set with zero sign violations.

## Accuracy vs. confidence estimate

Held-out **test accuracy** and per-sample **decision strength** are different concepts.

- Test accuracy (`96.34%`) is a global evaluation metric measured across the entire held-out test split.
- The UI confidence estimate is a bounded visualization of the **absolute Linear SVM decision margin** for one submitted text.
- The displayed estimate is **not** a calibrated probability.
- A high global test accuracy does **not** guarantee that every individual prediction is high confidence.

TruthLens uses:

```python
strength = 1 / (1 + exp(-abs(score)))
```

and labels the result as a **Decision-strength estimate**. Qualitative bands are derived from the
held-out test-set distribution of correctly classified absolute margins:

- `Low`: below the 50th percentile
- `Moderate`: 50th to 90th percentile
- `High`: above the 90th percentile

## Confidence audit summary

Held-out Linear SVM margin statistics:

- Absolute scores, correctly classified samples:
  - median: `1.3262801441810486`
  - 90th percentile: `2.2964537111040015`
  - max: `5.300617059286541`
- Absolute scores, misclassified samples:
  - median: `0.25739530979573483`
  - 90th percentile: `0.7717152446432826`
  - max: `2.167169878162828`

This confirms that larger absolute margins are generally associated with more reliable
classifications, while many misclassifications occur very close to the decision boundary.

## Demo-example extraction

The repository contains a small sanitized demonstration set derived from the real held-out test
split:

- `streamlit_app/demo_examples.json`

The examples include:

- strongest correctly classified REAL sample
- strongest correctly classified FAKE sample
- borderline correctly classified REAL sample
- borderline correctly classified FAKE sample
- one false positive for FAKE
- one false negative for FAKE

They are intended for thesis demonstration only. They inherit the dataset labels from WELFake and
are **not** independent fact-checking results.

## Application architecture

The Streamlit app lives under `streamlit_app/` and is organized into:

- `app.py` for entry-point routing
- `components/` for Detector, Methodology, Results, About, and sidebar rendering
- `services/model_service.py` for cached saved-model inference
- `utils/` for paths, validation, label mapping, confidence logic, and demo-example loading
- `assets/styles.css` for the UI
- `tests/` for unit and integration checks

Inference path:

```text
User text
→ saved TF-IDF vectorizer
→ saved Linear SVM model
→ predicted label
```

## Repository structure

```text
fake-news-detection-thesis/
├── data/
│   ├── raw/                        # WELFake_Dataset.csv (not tracked by Git)
│   ├── processed/                  # welfake_clean.csv, welfake_lemmatized.csv
│   └── splits/                     # X_train, X_test, y_train, y_test CSV files
├── figures/
├── models/
│   ├── tfidf_vectorizer.pkl        # Primary TF-IDF vectorizer (all 4 classical models)
│   ├── linear_svm.pkl
│   ├── logistic_regression.pkl
│   ├── naive_bayes.pkl
│   ├── random_forest.pkl
│   ├── mlp_model.pkl               # MLP Neural Network
│   ├── chi2_selector_mlp.pkl       # chi² feature selector for MLP
│   └── tfidf_vectorizer_lemmatized.pkl  # Separate vectorizer for lemmatized pipeline
├── notebooks/
│   ├── eda.ipynb                   # Primary pipeline: 4 classical models
│   ├── mlp_model.ipynb             # MLP Neural Network
│   └── lemmatizer_comparison.ipynb # Stanza lemmatization ablation study
├── src/
│   ├── audit_model_consistency.py
│   ├── consistency_utils.py
│   ├── extract_demo_examples.py
│   └── regenerate_confusion_matrix_figures.py
├── streamlit_app/
│   ├── app.py
│   ├── demo_examples.json
│   ├── services/
│   ├── components/
│   ├── utils/
│   ├── tests/
│   └── README.md
├── AGENTS.md
├── README.md
└── requirements.txt
```

## Notebook execution order

Execute notebooks in this order for full reproducibility:

1. `notebooks/eda.ipynb` — data cleaning, TF-IDF, 4 classical models, saves splits and vectorizer
2. `notebooks/mlp_model.ipynb` — MLP training (requires splits and vectorizer from step 1)
3. `notebooks/lemmatizer_comparison.ipynb` — lemmatization ablation (requires splits from step 1)

## Environment setup

From the repository root:

```bash
source venv/bin/activate
pip install -r requirements.txt
```

Use the repository virtual environment so the saved `scikit-learn==1.3.0` artifacts remain
compatible with the loading environment.

## How to run the model-consistency audit

```bash
source venv/bin/activate
python src/audit_model_consistency.py
```

## How to refresh demo examples

```bash
source venv/bin/activate
python src/extract_demo_examples.py
```

## How to run the Streamlit application

```bash
source venv/bin/activate
streamlit run streamlit_app/app.py
```

## How to run tests

```bash
source venv/bin/activate
pytest -v
```

## Required model files

The application requires:

- `models/tfidf_vectorizer.pkl`
- `models/linear_svm.pkl`

The repository also preserves:

- `models/logistic_regression.pkl`
- `models/naive_bayes.pkl`
- `models/random_forest.pkl`
- `models/mlp_model.pkl`
- `models/chi2_selector_mlp.pkl`
- `models/tfidf_vectorizer_lemmatized.pkl`

## Dataset availability

This thesis expects local dataset artifacts under `data/raw/`, `data/processed/`, and `data/splits/`.
These research files are preserved locally but ignored by Git.

## Reproducibility information

- Execute notebooks in order: `eda.ipynb` → `mlp_model.ipynb` → `lemmatizer_comparison.ipynb`
- Use `scikit-learn==1.3.0` for compatibility with the saved `.pkl` artifacts
- Keep the label mapping fixed as `0 = REAL`, `1 = FAKE`
- Keep training and deployment separate
- Do not overwrite the saved model binaries
- The lemmatization step (~72,000 articles via Stanza on CPU) takes approximately 2 hours;
  checkpointing saves progress every 2,000 rows to `data/processed/welfake_lemmatized_partial.csv`

## Limitations

- Trained on English-language data only
- Binary REAL/FAKE classification only
- Evaluated on a single train/test split (random_state=42); cross-validation not performed
- Possible source bias in WELFake (aggregated from Kaggle, McIntire, Reuters, BuzzFeed)
- Possible topic bias in WELFake
- Learns dataset patterns rather than objective truth
- No external evidence retrieval
- No source-authority verification
- No live fact-checking
- No statistical significance testing for performance differences between models
- Decision-strength estimate is derived from the Linear SVM margin and is **not** a calibrated probability

## Academic disclaimer

TruthLens is an academic prototype and a decision-support tool. Its output must not be treated as
definitive proof that a news article is true or false.
