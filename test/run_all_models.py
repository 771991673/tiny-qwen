import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from test_text_plus_image import DEFAULT_MODELS, run_reference_check


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Qwen 3.5 multimodal reference checks across model variants."
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=DEFAULT_MODELS,
        help="Model ids to evaluate.",
    )
    parser.add_argument(
        "--reference-json",
        type=Path,
        default=Path("test/data/qwen35_transformers_mm_outputs.json"),
        help="Transformers reference JSON file.",
    )
    parser.add_argument(
        "--match-mode",
        choices=["semantic", "exact"],
        default="semantic",
        help="Comparison mode.",
    )
    parser.add_argument(
        "--min-ratio",
        type=float,
        default=0.35,
        help="Minimum ratio for semantic mode.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results = run_reference_check(
        reference_json=args.reference_json,
        models=args.models,
        match_mode=args.match_mode,
        min_ratio=args.min_ratio,
    )
    return 0 if all(r["match"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
