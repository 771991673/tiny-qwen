import torch
from transformers import AutoProcessor
from model.processor import Processor

MODEL_NAME = "Qwen/Qwen3.5-27B"


def test_text_only():
    """Test with text-only message."""
    hf_proc = AutoProcessor.from_pretrained(MODEL_NAME)
    our_proc = Processor.from_pretrained(MODEL_NAME)

    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "What is the meaning of life?"}],
        }
    ]

    hf_inputs = hf_proc.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    our_inputs = our_proc(messages, add_generation_prompt=True)

    print(f"\n{'='*60}")
    print(f"Test: Text Only")
    print(f"{'='*60}")

    # For text-only, no pixel values
    assert not hasattr(hf_inputs, "pixel_values") or hf_inputs.pixel_values is None
    assert our_inputs["pixels"] is None
    assert our_inputs["d_image"] is None

    print(f"✓ input_ids match exactly (shape: {hf_inputs.input_ids.shape})")
    assert torch.equal(hf_inputs.input_ids, our_inputs["input_ids"].cpu())
    print(f"✓ No pixel values (text-only)")
    print(f"✅ All assertions passed!")


def compare_outputs(hf_inputs, our_inputs, test_name=""):
    """Compare HF and our processor outputs."""
    print(f"\n{'='*60}")
    print(f"Test: {test_name}")
    print(f"{'='*60}")

    hf_input_ids = hf_inputs.input_ids
    hf_pixels = hf_inputs.pixel_values
    hf_d_image = hf_inputs.image_grid_thw

    our_input_ids = our_inputs["input_ids"]
    our_pixel_values = our_inputs["pixels"]
    our_grid_thw = our_inputs["d_image"]

    # Assert exact matches
    assert torch.equal(hf_input_ids, our_input_ids.cpu()), "input_ids do not match"
    assert torch.equal(hf_d_image, our_grid_thw.cpu()), "grid_thw does not match"

    print(f"✓ input_ids match exactly (shape: {hf_input_ids.shape})")
    print(f"✓ pixel_values shape match: {hf_pixels.shape}")

    # Pixel value statistics
    print(f"\n=== Pixel Values Comparison ===")
    print(f"HF pixel_values - mean: {hf_pixels.mean():.6f}, std: {hf_pixels.std():.6f}")
    print(
        f"Our pixel_values - mean: {our_pixel_values.mean():.6f}, std: {our_pixel_values.std():.6f}"
    )

    # Difference statistics
    diff = torch.abs(hf_pixels - our_pixel_values.cpu())
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    median_diff = diff.median().item()
    std_diff = diff.std().item()
    pct_above_threshold = (diff > 0.01).float().mean().item() * 100

    print(f"\nAbsolute Differences:")
    print(f"  Max:    {max_diff:.6f}")
    print(f"  Mean:   {mean_diff:.6f}")
    print(f"  Median: {median_diff:.6f}")
    print(f"  Std:    {std_diff:.6f}")
    print(f"  % > 0.01: {pct_above_threshold:.2f}%")

    print(f"\n✓ image_grid_thw:")
    print(f"  - HF:  {hf_d_image.tolist()}")
    print(f"  - Ours: {our_grid_thw.tolist()}")

    # Assert pixel values are close enough
    assert max_diff < 0.02, f"Max pixel difference {max_diff:.6f} exceeds 0.02"
    assert mean_diff < 0.001, f"Mean pixel difference {mean_diff:.6f} exceeds 0.001"

    print(f"✅ All assertions passed!")


def test_single_image():
    """Test with a single image from URL."""
    hf_proc = AutoProcessor.from_pretrained(MODEL_NAME)
    our_proc = Processor.from_pretrained(MODEL_NAME)

    url = "test/data/test-img-1.jpg"

    # HF uses "image" key, we use "url" key
    hf_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": url},
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]

    our_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": url},
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]

    hf_inputs = hf_proc.apply_chat_template(
        hf_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    our_inputs = our_proc(our_messages, add_generation_prompt=True)

    compare_outputs(hf_inputs, our_inputs, "Single Image from URL")


def test_multi_image():
    """Test with multiple images in single message."""
    hf_proc = AutoProcessor.from_pretrained(MODEL_NAME)
    our_proc = Processor.from_pretrained(MODEL_NAME)

    url1 = "test/data/test-img-1.jpg"
    url2 = "test/data/test-img-2.jpg"

    # HF uses "image" key, we use "url" key
    hf_messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Compare these two images."},
                {"type": "image", "image": url1},
                {"type": "text", "text": "and"},
                {"type": "image", "image": url2},
            ],
        },
    ]

    our_messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Compare these two images."},
                {"type": "image", "url": url1},
                {"type": "text", "text": "and"},
                {"type": "image", "url": url2},
            ],
        },
    ]

    hf_inputs = hf_proc.apply_chat_template(
        hf_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    our_inputs = our_proc(our_messages, add_generation_prompt=True)

    compare_outputs(hf_inputs, our_inputs, "Multiple Images in Single Message")


if __name__ == "__main__":
    test_text_only()
    test_single_image()
    test_multi_image()
    print(f"\n{'='*60}")
    print("🎉 All tests passed!")
    print(f"{'='*60}\n")
