import argparse
import json
import sys
from difflib import SequenceMatcher
from pathlib import Path

import pytest
import torch
from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.model import Qwen3VL
from model.processor import Processor

REFERENCE_JSON = Path("test/data/qwen35_transformers_mm_outputs.json")
DEFAULT_MODELS = [
    "Qwen/Qwen3.5-0.8B",
    "Qwen/Qwen3.5-2B",
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3.5-9B",
    "Qwen/Qwen3.5-27B",
    "Qwen/Qwen3.5-35B-A3B",
]


def _get_stop_tokens(processor: Processor) -> list[int]:
    tokens = []
    for tok in ("<|im_end|>", "<|im_start|>", "<|endoftext|>", "<|eot_id|>"):
        tok_id = processor.tokenizer.token_to_id(tok)
        if tok_id is not None:
            tokens.append(tok_id)
    return sorted(set(tokens))


def _normalize_output(text: str) -> str:
    text = text.strip()
    if text.startswith("<think>\n\n</think>\n\n"):
        text = text[len("<think>\n\n</think>\n\n") :]
    if text.startswith("<think>\n"):
        text = text[len("<think>\n") :]

    # If generation spills into a new pseudo-turn, keep only assistant's first answer.
    for marker in ("\nuser\n", "\n<|im_start|>user\n", "\nassistant\n"):
        if marker in text:
            text = text.split(marker, 1)[0]
    return text.strip()


def _semantic_match(expected: str, actual: str, min_ratio: float) -> tuple[bool, float, bool]:
    expected_norm = _normalize_output(expected).lower()
    actual_norm = _normalize_output(actual).lower()
    ratio = SequenceMatcher(a=actual_norm, b=expected_norm).ratio()

    # We intentionally allow wording differences; enforce core visual facts instead.
    core_ok = (
        ("sunflower" in actual_norm)
        and ("bee" in actual_norm)
        and (("field" in actual_norm) or ("background" in actual_norm))
    )
    return (ratio >= min_ratio and core_ok), ratio, core_ok


def _is_match(
    expected: str, actual: str, match_mode: str, min_ratio: float
) -> tuple[bool, float, bool]:
    if match_mode == "exact":
        return (_normalize_output(actual) == _normalize_output(expected)), 1.0, True
    return _semantic_match(expected=expected, actual=actual, min_ratio=min_ratio)


def _generate_tiny_output(
    model_name: str,
    prompt: str,
    image_path: str,
    max_new_tokens: int,
    enable_thinking: bool,
) -> tuple[bool, str]:
    try:
        device_map = {"": 0} if torch.cuda.is_available() else "auto"
        weights_path = snapshot_download(repo_id=model_name, cache_dir=".cache")
        model = Qwen3VL.from_pretrained(weights_path=weights_path, device_map=device_map)
        model.eval()
        processor = Processor.from_pretrained(model_name)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        device = next(model.parameters()).device
        inputs = processor(
            messages,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
            device=device,
        )

        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixels=inputs.get("pixels"),
            d_image=inputs.get("d_image"),
            max_new_tokens=max_new_tokens,
            stop_tokens=_get_stop_tokens(processor),
        )
        prompt_len = inputs["input_ids"].shape[1]
        continuation_ids = output_ids[0][prompt_len:].tolist()
        return True, processor.tokenizer.decode(continuation_ids)
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def _load_reference(reference_json: Path) -> dict:
    if not reference_json.exists():
        raise FileNotFoundError(f"Missing reference JSON: {reference_json}")
    return json.loads(reference_json.read_text())


def run_reference_check(
    reference_json: Path,
    models: list[str] | None = None,
    match_mode: str = "semantic",
    min_ratio: float = 0.35,
) -> list[dict]:
    ref = _load_reference(reference_json)
    requested_models = models or list(ref.get("outputs", {}).keys())

    print(
        "NOTE: This check uses semantic matching against Transformers reference output. "
        "A token-identical match is not expected."
    )

    results = []
    for model_name in requested_models:
        expected = ref["outputs"].get(model_name)
        if expected is None:
            result = {
                "model": model_name,
                "loaded": False,
                "match": False,
                "ratio": 0.0,
                "core_ok": False,
                "expected": f"<missing in {reference_json}>",
                "actual": "<not run>",
            }
            results.append(result)
            continue

        loaded, actual = _generate_tiny_output(
            model_name=model_name,
            prompt=ref["prompt"],
            image_path=ref["image_path"],
            max_new_tokens=int(ref["max_new_tokens"]),
            enable_thinking=bool(ref.get("enable_thinking", True)),
        )

        match, ratio, core_ok = (False, 0.0, False)
        if loaded:
            match, ratio, core_ok = _is_match(
                expected=expected,
                actual=actual,
                match_mode=match_mode,
                min_ratio=min_ratio,
            )

        result = {
            "model": model_name,
            "loaded": loaded,
            "match": match,
            "ratio": ratio,
            "core_ok": core_ok,
            "expected": expected,
            "actual": actual,
            "normalized_expected": _normalize_output(expected),
            "normalized_actual": _normalize_output(actual),
        }
        results.append(result)

    for result in results:
        print(f"\n[{result['model']}] loaded={result['loaded']}")
        print(f"match={result['match']}")
        print(
            f"mode={match_mode} ratio={result['ratio']:.3f} core_ok={result['core_ok']}"
        )
        print(f"expected={result['expected']!r}")
        print(f"actual={result['actual']!r}")
        if "normalized_expected" in result:
            print(f"normalized_expected={result['normalized_expected']!r}")
            print(f"normalized_actual={result['normalized_actual']!r}")

    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate Tiny Qwen image outputs against a Transformers-generated "
            "Qwen 3.5 reference JSON."
        )
    )
    parser.add_argument(
        "--reference-json",
        type=Path,
        default=REFERENCE_JSON,
        help="Path to Transformers reference JSON.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional subset of model ids to evaluate.",
    )
    parser.add_argument(
        "--match-mode",
        choices=["semantic", "exact"],
        default="semantic",
        help="How to compare Tiny output to reference.",
    )
    parser.add_argument(
        "--min-ratio",
        type=float,
        default=0.35,
        help="Minimum SequenceMatcher ratio for semantic mode.",
    )
    return parser.parse_args()


@pytest.mark.parametrize("model_name", DEFAULT_MODELS)
def test_text_plus_image_reference(model_name: str):
    if not REFERENCE_JSON.exists():
        pytest.skip(f"Missing reference JSON: {REFERENCE_JSON}")

    results = run_reference_check(
        reference_json=REFERENCE_JSON,
        models=[model_name],
        match_mode="semantic",
        min_ratio=0.35,
    )
    result = results[0]
    assert result["loaded"], f"Model failed to load: {result['actual']}"
    assert result["match"], (
        f"Mismatch for {model_name}\n"
        f"ratio={result['ratio']:.3f} core_ok={result['core_ok']}\n"
        f"expected={result['expected']!r}\n"
        f"actual={result['actual']!r}"
    )


def main() -> int:
    args = _parse_args()
    results = run_reference_check(
        reference_json=args.reference_json,
        models=args.models,
        match_mode=args.match_mode,
        min_ratio=args.min_ratio,
    )
    return 0 if all(r["match"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
