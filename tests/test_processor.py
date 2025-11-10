import torch
from PIL import Image
from transformers import AutoProcessor

from model.processor import Processor, VisionConfig


def test_processor():
    """Test our processor against HuggingFace's reference implementation."""
    model_name = "Qwen/Qwen2.5-VL-3B-Instruct"

    # Initialize processors
    hf_processor = AutoProcessor.from_pretrained(model_name)
    vision_config = VisionConfig(
        n_embed=1280,
        n_layer=32,
        n_heads=16,
        output_n_embed=1280,
        in_channels=3,
        spatial_merge_size=2,
        spatial_patch_size=14,
        temporal_patch_size=2,
    )
    our_processor = Processor(repo_id=model_name, vision_config=vision_config)

    # Load test images
    image_1 = Image.open("data/test-img-1.jpg")
    image_2 = Image.open("data/test-img-2.jpg")

    # Prepare input for HF processor
    text_for_hf = (
        "<|im_start|>user\n"
        "<|vision_start|><|image_pad|><|vision_end|>What's in image 1?\n"
        "<|vision_start|><|image_pad|><|vision_end|>Now what's in image 2?<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    # Prepare input for our processor
    text_for_ours = [
        "<|im_start|>user\n<|vision_start|>",
        image_1,
        "<|vision_end|>What's in image 1?\n<|vision_start|>",
        image_2,
        "<|vision_end|>Now what's in image 2?<|im_end|>\n<|im_start|>assistant\n",
    ]

    # Process with HF processor
    hf_processed = hf_processor(
        text=[text_for_hf],
        images=[image_1, image_2],
        return_tensors="pt",
    )
    hf_input_ids = hf_processed["input_ids"]
    hf_pixel_values = hf_processed["pixel_values"]
    hf_grid_thw = hf_processed["image_grid_thw"]

    # Process with our processor
    our_processed = our_processor(text_for_ours)
    our_input_ids = our_processed["input_ids"]
    our_pixel_values = our_processed["pixels"]
    our_grid_thw = our_processed["d_image"]

    # Assert shapes match
    assert (
        hf_input_ids.shape == our_input_ids.shape
    ), f"input_ids shape mismatch: HF {hf_input_ids.shape} vs Ours {our_input_ids.shape}"
    assert (
        hf_pixel_values.shape == our_pixel_values.shape
    ), f"pixel_values shape mismatch: HF {hf_pixel_values.shape} vs Ours {our_pixel_values.shape}"
    assert (
        hf_grid_thw.shape == our_grid_thw.shape
    ), f"grid_thw shape mismatch: HF {hf_grid_thw.shape} vs Ours {our_grid_thw.shape}"

    # Assert input_ids match exactly
    assert torch.equal(hf_input_ids, our_input_ids.cpu()), "input_ids do not match"

    # Assert pixel_values are close (allowing for small floating point differences)
    diff = torch.abs(hf_pixel_values - our_pixel_values.cpu())

    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    median_diff = diff.median().item()
    pct_large_diff = (diff > 0.01).float().mean().item() * 100

    # Assert strict bounds on differences
    assert max_diff < 0.05, f"Max difference {max_diff:.6f} exceeds 0.05"
    assert mean_diff < 0.001, f"Mean difference {mean_diff:.6f} exceeds 0.001"
    assert median_diff < 0.001, f"Median difference {median_diff:.6f} exceeds 0.001"
    assert (
        pct_large_diff < 1.0
    ), f"Too many pixels differ: {pct_large_diff:.2f}% have diff > 0.01"

    # Assert grid_thw matches exactly
    assert torch.equal(
        hf_grid_thw, our_grid_thw.cpu()
    ), "grid_thw (d_image) does not match"


def test_message_conversion():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in the image?"},
                {"type": "image", "image": Image.open("data/test-img-1.jpg")},
            ],
        },
        {"role": "assistant", "content": "The image contains a bee on a sunflower."},
        {"role": "user", "content": "what's the temperature in paris?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "name": "get_current_temperature",
                    "arguments": {"location": "Paris, France", "unit": "celsius"},
                }
            ],
        },
        {
            "role": "tool",
            "content": '{"location":"Paris, France","unit":"celsius","temperature":18.2}',
        },
        {
            "role": "assistant",
            "content": "It's 18.2 °C in Paris right now.",
        },
    ]

    processor = Processor(repo_id="Qwen/Qwen3-VL-30B-A3B-Instruct")

    correct_rendered_messages = """<|im_start|>user
What's in the image?<|vision_start|><|image_pad|><|vision_end|><|im_end|>
<|im_start|>assistant
The image contains a bee on a sunflower.<|im_end|>
<|im_start|>user
what's the temperature in paris?<|im_end|>
<|im_start|>assistant
<tool_call>
{"name": "get_current_temperature", "arguments": {"location": "Paris, France", "unit": "celsius"}}
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
{"location":"Paris, France","unit":"celsius","temperature":18.2}
</tool_response><|im_end|>
<|im_start|>assistant
It's 18.2 °C in Paris right now.<|im_end|>
<|im_start|>assistant

"""

    rendered_messages = processor(messages)
    assert rendered_messages == correct_rendered_messages, "Rendered messages do not match"
    print("Test passed")


if __name__ == "__main__":
    test_processor()
