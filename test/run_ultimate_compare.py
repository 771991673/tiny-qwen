import argparse
import sys
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoProcessor

try:
    from transformers import AutoModelForImageTextToText as AutoModelForVL
except ImportError:
    from transformers import AutoModelForVision2Seq as AutoModelForVL

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from model.model import Qwen3VL
from model.processor import Processor

DEFAULT_MODELS = [
    "Qwen/Qwen3.5-0.8B",
    "Qwen/Qwen3.5-2B",
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3.5-9B",
    "Qwen/Qwen3.5-27B",
    "Qwen/Qwen3.5-35B-A3B",
]


def _messages(image_path: str, prompt: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def _get_stop_tokens(processor: Processor) -> list[int]:
    tokens = []
    for tok in ("<|im_end|>", "<|im_start|>", "<|endoftext|>", "<|eot_id|>"):
        tok_id = processor.tokenizer.token_to_id(tok)
        if tok_id is not None:
            tokens.append(tok_id)
    return sorted(set(tokens))


def _transformers_output(
    model_name: str,
    image_path: str,
    prompt: str,
    max_new_tokens: int,
    enable_thinking: bool,
) -> str:
    kwargs = {"device_map": "auto"}
    if torch.cuda.is_available():
        kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForVL.from_pretrained(model_name, **kwargs)
    model.eval()
    processor = AutoProcessor.from_pretrained(model_name)
    messages = _messages(image_path=image_path, prompt=prompt)

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
        return_dict=True,
        return_tensors="pt",
    )
    inputs.pop("token_type_ids", None)
    inputs = inputs.to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    del model, processor, inputs, generated_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return text


def _tiny_output(
    model_name: str,
    image_path: str,
    prompt: str,
    max_new_tokens: int,
    enable_thinking: bool,
) -> str:
    weights_path = snapshot_download(repo_id=model_name, cache_dir=".cache")
    device_map = {"": 0} if torch.cuda.is_available() else "auto"
    model = Qwen3VL.from_pretrained(weights_path=weights_path, device_map=device_map)
    model.eval()
    processor = Processor.from_pretrained(model_name)
    messages = _messages(image_path=image_path, prompt=prompt)

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
    text = processor.tokenizer.decode(continuation_ids)

    del model, processor, inputs, output_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one side-by-side multimodal inference pass per model and print "
            "Transformers output next to Tiny-Qwen output."
        )
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Hugging Face model ids to evaluate.",
    )
    parser.add_argument(
        "--image-path",
        type=Path,
        default=Path("test/data/test-img-1.jpg"),
        help="Local image path.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Describe this image accurately in 2-3 sentences.",
        help="Prompt text.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum generated tokens.",
    )
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Pass enable_thinking to both pipelines.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    failures = []
    image_path = str(args.image_path)

    for model_name in args.models:
        print("\n" + "=" * 88)
        print(f"MODEL: {model_name}")
        print("=" * 88)

        tf_text = None
        tiny_text = None
        tf_err = None
        tiny_err = None

        try:
            tf_text = _transformers_output(
                model_name=model_name,
                image_path=image_path,
                prompt=args.prompt,
                max_new_tokens=args.max_new_tokens,
                enable_thinking=args.enable_thinking,
            )
        except Exception as exc:  # noqa: BLE001
            tf_err = f"{type(exc).__name__}: {exc}"

        try:
            tiny_text = _tiny_output(
                model_name=model_name,
                image_path=image_path,
                prompt=args.prompt,
                max_new_tokens=args.max_new_tokens,
                enable_thinking=args.enable_thinking,
            )
        except Exception as exc:  # noqa: BLE001
            tiny_err = f"{type(exc).__name__}: {exc}"

        print("[Transformers]")
        print(tf_text if tf_err is None else f"ERROR: {tf_err}")
        print("\n[Tiny-Qwen]")
        print(tiny_text if tiny_err is None else f"ERROR: {tiny_err}")

        if tf_err is not None or tiny_err is not None:
            failures.append(model_name)

    if failures:
        print("\nFailed models:")
        for model_name in failures:
            print(f"- {model_name}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
