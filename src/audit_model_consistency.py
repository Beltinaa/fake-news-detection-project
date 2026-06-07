"""Audit label mapping, score interpretation, and saved-model consistency."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from streamlit_app.config import (  # noqa: E402
    LINEAR_SVM_ACCURACY,
    LINEAR_SVM_CONFUSION_MATRIX,
)
from streamlit_app.utils.labels import (  # noqa: E402
    EXPECTED_MODEL_CLASSES,
    VERIFIED_LABEL_MAPPING,
    display_labels_for_classes,
)
from src.consistency_utils import (  # noqa: E402
    build_evaluation_frame,
    distribution_summary,
    load_consistency_context,
    representative_rows,
    select_demo_examples,
)


def main() -> int:
    context = load_consistency_context()
    evaluation = build_evaluation_frame(context)

    errors: list[str] = []

    raw_labels = sorted(int(value) for value in context.raw_dataset["label"].dropna().unique())
    processed_labels = sorted(int(value) for value in context.processed_dataset["label"].dropna().unique())
    train_labels = sorted(int(value) for value in context.y_train.dropna().unique())
    test_labels = sorted(int(value) for value in context.y_test.dropna().unique())

    print("Confirmed dataset label mapping:")
    print(f"  Raw dataset labels: {VERIFIED_LABEL_MAPPING}")
    print(f"  Processed dataset labels: {VERIFIED_LABEL_MAPPING}")
    print(f"  Split labels: {VERIFIED_LABEL_MAPPING}")
    print()

    print("Representative raw-dataset rows:")
    for label_id in EXPECTED_MODEL_CLASSES:
        print(f"  Label {label_id} ({VERIFIED_LABEL_MAPPING[label_id]}):")
        for sample in representative_rows(context.raw_dataset, label_id):
            print(f"    - title={sample['title']}")
            print(f"      text={sample['text']}")
    print()

    print(f"model.classes_: {list(context.model_classes)}")
    print(f"Display class order: {display_labels_for_classes(context.model_classes)}")

    if raw_labels != list(EXPECTED_MODEL_CLASSES):
        errors.append(f"Unexpected raw dataset labels: {raw_labels}")
    if processed_labels != list(EXPECTED_MODEL_CLASSES):
        errors.append(f"Unexpected processed dataset labels: {processed_labels}")
    if train_labels != list(EXPECTED_MODEL_CLASSES):
        errors.append(f"Unexpected y_train labels: {train_labels}")
    if test_labels != list(EXPECTED_MODEL_CLASSES):
        errors.append(f"Unexpected y_test labels: {test_labels}")

    accuracy = accuracy_score(evaluation["actual_label_id"], evaluation["predicted_label_id"])
    matrix = confusion_matrix(evaluation["actual_label_id"], evaluation["predicted_label_id"])
    report = classification_report(
        evaluation["actual_label_id"],
        evaluation["predicted_label_id"],
        digits=6,
    )

    print()
    print(f"Reproduced Linear SVM accuracy: {accuracy}")
    print("Reproduced confusion matrix:")
    print(matrix)
    print()
    print(report)

    if not np.isclose(accuracy, LINEAR_SVM_ACCURACY, atol=1e-12):
        errors.append(
            f"Accuracy mismatch: reproduced {accuracy} vs expected {LINEAR_SVM_ACCURACY}"
        )
    if matrix.tolist() != [list(row) for row in LINEAR_SVM_CONFUSION_MATRIX]:
        errors.append(
            f"Confusion-matrix mismatch: reproduced {matrix.tolist()} vs expected {LINEAR_SVM_CONFUSION_MATRIX}"
        )

    sign_violations = evaluation.loc[
        evaluation["margin_favored_class_id"] != evaluation["predicted_label_id"]
    ]
    print(f"Decision-score sign violations: {len(sign_violations)}")
    if not sign_violations.empty:
        errors.append(
            f"Decision-score sign interpretation failed for {len(sign_violations)} samples."
        )

    print()
    print("Decision-score distribution summary:")
    print(f"  all scores: {distribution_summary(evaluation['decision_score'])}")
    print(f"  absolute scores: {distribution_summary(evaluation['absolute_score'])}")
    print(
        "  correctly classified absolute scores: "
        f"{distribution_summary(evaluation.loc[evaluation['is_correct'], 'absolute_score'])}"
    )
    print(
        "  misclassified absolute scores: "
        f"{distribution_summary(evaluation.loc[~evaluation['is_correct'], 'absolute_score'])}"
    )
    print(
        f"  actual REAL scores: {distribution_summary(evaluation.loc[evaluation['actual_label_id'] == 0, 'decision_score'])}"
    )
    print(
        f"  actual FAKE scores: {distribution_summary(evaluation.loc[evaluation['actual_label_id'] == 1, 'decision_score'])}"
    )

    examples = select_demo_examples(evaluation)
    print()
    print("Selected held-out demo examples:")
    for example in examples:
        print(
            f"  - {example['name']}: actual={example['actualLabel']}, "
            f"predicted={example['predictedLabel']}, score={example['decisionScore']:.6f}"
        )

    if errors:
        print()
        print("Critical consistency errors detected:")
        for error in errors:
            print(f"  - {error}")
        return 1

    print()
    print("Audit passed: dataset labels, model classes, decision-score signs, and saved metrics are consistent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
