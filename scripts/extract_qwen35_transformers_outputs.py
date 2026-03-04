import argparse
import json
from pathlib import Path

import torch
from transformers import AutoProcessor

try:
    from transformers import AutoModelForImageTextToText as AutoModelForVL
except ImportError:
    from transformers import AutoModelForVision2Seq as AutoModelForVL

DEFAULT_MODELS = [
    "Qwen/Qwen3.5-0.8B",
    "Qwen/Qwen3.5-2B",
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3.5-9B",
    "Qwen/Qwen3.5-27B",
    "Qwen/Qwen3.5-35B-A3B",
]
def _build_messages(image_path: str, prompt: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def run_inference_for_model(
    model_name: str,
    image_path: str,
    prompt: str,
    max_new_tokens: int,
    attn_implementation: str,
    enable_thinking: bool,
) -> str:
    model = AutoModelForVL.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation=attn_implementation,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(model_name)

    messages = _build_messages(image_path=image_path, prompt=prompt)
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
    return processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract Qwen 3.5 multimodal outputs from Hugging Face Transformers."
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
        help="Local image path used in the prompt.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Describe this image accurately in 2-3 sentences.",
        help="Prompt text appended to the image content.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum number of generated tokens.",
    )
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default="sdpa",
        choices=["sdpa", "eager"],
        help="Attention implementation to use in Transformers.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("test/data/qwen35_transformers_mm_outputs.json"),
        help="Output path for captured generation results.",
    )
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass enable_thinking to chat-template processing.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    image_path = str(args.image_path)
    results = {}

    for model_name in args.models:
        print(f"\n=== Running {model_name} ===", flush=True)
        text = run_inference_for_model(
            model_name=model_name,
            image_path=image_path,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            attn_implementation=args.attn_implementation,
            enable_thinking=args.enable_thinking,
        )
        results[model_name] = text
        print(f"[{model_name}] {text}", flush=True)

    payload = {
        "image_path": image_path,
        "prompt": args.prompt,
        "max_new_tokens": args.max_new_tokens,
        "attn_implementation": args.attn_implementation,
        "enable_thinking": args.enable_thinking,
        "outputs": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nSaved outputs to {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
