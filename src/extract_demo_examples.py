"""Extract a small set of verified held-out demo examples for the Streamlit app."""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.consistency_utils import (  # noqa: E402
    build_evaluation_frame,
    load_consistency_context,
    select_demo_examples,
    write_demo_examples,
)


def main() -> int:
    context = load_consistency_context()
    evaluation = build_evaluation_frame(context)
    examples = select_demo_examples(evaluation)
    output_path = write_demo_examples(examples)

    print(f"Saved {len(examples)} demo examples to {output_path.relative_to(REPO_ROOT)}")
    for example in examples:
        print(
            f"- {example['name']}: actual={example['actualLabel']}, "
            f"predicted={example['predictedLabel']}, score={example['decisionScore']:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
