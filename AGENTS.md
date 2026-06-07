# AGENTS.md

Future coding agents working in this repository must follow these constraints:

1. Preserve all notebook outputs in `notebooks/eda.ipynb`.
2. Preserve datasets, processed files, split files, figures, and saved model artifacts.
3. Treat the executed notebook as the experiment source of truth.
4. Never alter the verified label mapping: `0 = REAL`, `1 = FAKE`.
5. Never replace the trained classifier with an LLM, external fake-news API, or fabricated output.
6. Keep training and inference separate. The deployed app must load saved artifacts only.
7. Do not invent metrics, figures, or evaluation claims that are not supported by the notebook.
8. Avoid unsupported preprocessing changes that would break compatibility with the saved vectorizer.
9. Maintain academic disclaimers and avoid language implying factual certainty.
10. Do not commit datasets, model binaries, virtual environments, caches, or secrets.
11. Run feasible tests and verification checks after changes.
