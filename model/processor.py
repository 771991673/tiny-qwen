import torch
import requests
from io import BytesIO
from PIL import Image
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from tokenizers import Tokenizer

from .vision import VisionConfig


@dataclass
class ModelInputs:
    input_ids: torch.Tensor  # (B, T)
    attention_mask: torch.Tensor  # (B, T)
    # ^ tell model which tokens to attend to. good for ignoring padding tokens in a batch padded to the same length.
    pixel_values: torch.Tensor  # (B, C, H, W)
    image_grid_thw: torch.Tensor  # (B, T, 3, H, W)


class Processor:
    def __init__(self, repo_id: str, vision_config: Optional[VisionConfig] = None):
        self.tokenizer = Tokenizer.from_pretrained(repo_id)
        self.vision_config = vision_config

        if self.vision_config is not None:
            # Vision-specific setup
            image_pad_token = "<|image_pad|>"
            vision_start_token = "<|vision_start|>"
            vision_end_token = "<|vision_end|>"
            self.tokenizer.add_special_tokens([image_pad_token])
            self.vision_start_token_id = self.tokenizer.encode(vision_start_token).ids[
                0
            ]
            self.image_pad_token_id = self.tokenizer.encode(image_pad_token).ids[0]
            self.vision_end_token_id = self.tokenizer.encode(vision_end_token).ids[0]

            # Constants for image processing
            self.MIN_PIXELS = 3136
            self.MAX_PIXELS = 12845056
            self.IMAGE_MEAN = np.array(
                [0.48145466, 0.4578275, 0.40821073], dtype=np.float32
            )
            self.IMAGE_STD = np.array(
                [0.26862954, 0.26130258, 0.27577711], dtype=np.float32
            )

    # Turn openai harmony style messages into model input tensors.
    def __call__(self, messages: List[dict]) -> ModelInputs:
        messages_str = self._flatten_messages(messages)
        input_ids = self.tokenizer.encode(messages_str).ids
        attention_mask = torch.ones_like(input_ids)

        pixels_list = []
        d_image_list = []

        # Identify if we have vision support
        has_vision = self.vision_config is not None
        if not has_vision:
            # If we have no vision_config, do text-only
            for item in inputs:
                if isinstance(item, str):
                    input_ids.extend(self.tokenizer.encode(item).ids)
                else:
                    raise ValueError(
                        f"Images are not supported by a text-only model. Got {type(item)}"
                    )
        else:
            # Vision + text model
            merge_size = self.vision_config.spatial_merge_size
            image_pad_token_id = self.image_pad_token_id

            for item in inputs:
                if isinstance(item, str):
                    # Handle text
                    input_ids.extend(self.tokenizer.encode(item).ids)
                elif isinstance(item, Image.Image):
                    # Handle image
                    patches, t, h, w = self._process_image(item)
                    pixels_list.append(patches)
                    d_image_list.append([t, h, w])

                    pad_token_count = (t * h * w) // (merge_size**2)
                    pad_tokens = [image_pad_token_id] * pad_token_count
                    input_ids.extend(pad_tokens)
                else:
                    raise ValueError(f"Unsupported input type: {type(item)}")

        # Convert accumulated ids to tensor
        input_ids = torch.tensor([input_ids], dtype=torch.long)

        # Convert all images to tensor if there are any
        if pixels_list:
            pixels_np = np.concatenate(pixels_list, axis=0)
            pixels = torch.tensor(pixels_np, dtype=torch.float)
            d_image = torch.tensor(d_image_list, dtype=torch.long)
        else:
            pixels = None
            d_image = None

        # # Move to device (if given)
        # if device is not None:
        #     input_ids = input_ids.to(device)
        #     if pixels is not None:
        #         pixels = pixels.to(device)

        attention_mask = torch.ones_like(input_ids)

        return ModelInputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixels,
            image_grid_thw=d_image,
        )

    def _flatten_messages(self, messages: List[dict]) -> str:
        flattened_messages = [self._flatten_message(message) for message in messages]
        return "".join(flattened_messages)

    def _flatten_message(self, message: dict) -> str:
        role, content = message["role"], message["content"]
        content_items = [self._flatten_content(item) for item in content]
        if role == "system":
            return f"<|im_start|>system\n{content_items}<|im_end|>\n"
        elif role == "user":
            return f"<|im_start|>user\n{content_items}<|im_end|>\n"
        elif role == "assistant":
            return f"<|im_start|>assistant\n{content_items}<|im_end|>\n"
        else:
            raise ValueError(f"Unsupported role: {role}")

    def _flatten_content(
        self, content: List[str | dict[str, str | Image.Image]]
    ) -> str:
        if isinstance(content, str):
            return content
        elif content["type"] == "text":
            return content["text"]
        elif content["type"] == "image":
            return "<|vision_start|><|image_pad|><|vision_end|>"
        elif content["type"] == "tool_use":
            tool_name, tool_id, tool_args = content["name"], content["args"]
            return f"<|tool_use|>{tool_name}<|tool_use_id|>{tool_id}<|args|>{tool_args}<|tool_use_end|>"
        elif content["type"] == "tool_result":
            tool_id, result = content["tool"], content["result"]
            return f"<|tool_result|>{tool_id}<|result|>{result}<|tool_result_end|>"
        else:
            raise ValueError(f"Unsupported content type: {content['type']}")

    def _fetch_img_through_url(self, url: str) -> Image.Image:
        response = requests.get(url)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))

    def _process_image(self, image: Image.Image) -> Tuple[np.ndarray, int, int, int]:
        SPATIAL_PATCH_SIZE = self.vision_config.spatial_patch_size
        TEMPORAL_PATCH_SIZE = self.vision_config.temporal_patch_size
        SPATIAL_MERGE_SIZE = self.vision_config.spatial_merge_size

        image_np = np.array(image, dtype=np.float32)
        height, width = image_np.shape[:2]
        resized_height, resized_width = self._resize_image(
            height,
            width,
            factor=SPATIAL_PATCH_SIZE * SPATIAL_MERGE_SIZE,
        )
        image_resized = image.resize(
            (resized_width, resized_height), resample=Image.BICUBIC
        )
        image_np_resized = np.array(image_resized, dtype=np.float32)

        # Normalize
        image_np_resized = image_np_resized / 255.0
        mean = self.IMAGE_MEAN.reshape(1, 1, -1)
        std = self.IMAGE_STD.reshape(1, 1, -1)
        image_np_resized = (image_np_resized - mean) / std

        # Convert to channels-first and add batch dimension
        image_np_resized = np.transpose(image_np_resized, (2, 0, 1))
        image_np_resized = image_np_resized[np.newaxis, ...]

        # Handle temporal dimension
        if image_np_resized.shape[0] == 1:
            image_np_resized = np.tile(image_np_resized, (TEMPORAL_PATCH_SIZE, 1, 1, 1))

        # Extract patches
        batch_size, channels, height, width = image_np_resized.shape
        grid_t = batch_size // TEMPORAL_PATCH_SIZE
        grid_h = resized_height // SPATIAL_PATCH_SIZE
        grid_w = resized_width // SPATIAL_PATCH_SIZE

        patches = image_np_resized.reshape(
            grid_t,
            TEMPORAL_PATCH_SIZE,
            channels,
            grid_h // SPATIAL_MERGE_SIZE,
            SPATIAL_MERGE_SIZE,
            SPATIAL_PATCH_SIZE,
            grid_w // SPATIAL_MERGE_SIZE,
            SPATIAL_MERGE_SIZE,
            SPATIAL_PATCH_SIZE,
        )

        patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
        flatten_patches = patches.reshape(
            grid_t * grid_h * grid_w,
            channels * TEMPORAL_PATCH_SIZE * SPATIAL_PATCH_SIZE * SPATIAL_PATCH_SIZE,
        )

        return flatten_patches.astype(np.float32), grid_t, grid_h, grid_w

    def _resize_image(
        self, height: int, width: int, factor: int = 28
    ) -> Tuple[int, int]:
        if height < factor or width < factor:
            raise ValueError(
                f"height:{height} or width:{width} must be larger than factor:{factor}"
            )
        elif max(height, width) / min(height, width) > 200:
            raise ValueError(
                f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
            )

        h_bar = round(height / factor) * factor
        w_bar = round(width / factor) * factor

        if h_bar * w_bar > self.MAX_PIXELS:
            beta = np.sqrt((height * width) / self.MAX_PIXELS)
            h_bar = int(np.floor(height / beta / factor) * factor)
            w_bar = int(np.floor(width / beta / factor) * factor)
        elif h_bar * w_bar < self.MIN_PIXELS:
            beta = np.sqrt(self.MIN_PIXELS / (height * width))
            h_bar = int(np.ceil(height * beta / factor) * factor)
            w_bar = int(np.ceil(width * beta / factor) * factor)

        return h_bar, w_bar


if __name__ == "__main__":
    from transformers import AutoProcessor

    model_name = "Qwen/Qwen2-VL-2B-Instruct"
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

    image_1 = Image.open("test-images/test-image.webp")
    image_2 = Image.open("test-images/test-image.jpeg")

    text_for_hf = (
        "<|im_start|>user\n"
        "<|vision_start|><|image_pad|><|vision_end|>What's in image 1?\n"
        "<|vision_start|><|image_pad|><|vision_end|>Now what's in image 2?<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

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
    our_pixel_values = our_processed["pixel_values"]
    our_grid_thw = our_processed["grid_thw"]

    # Print shapes
    print("\nOutput Shapes:")
    print("-" * 50)
    print("pixel_values:")
    print("  HF:", hf_pixel_values.shape)
    print("  Ours:", our_pixel_values.shape)

    print("\ninput_ids:")
    print("  HF:", hf_input_ids.shape)
    print("  Ours:", our_input_ids.shape)

    print("\ngrid_thw:")
    print("  HF:", hf_grid_thw.shape)
    print("  Ours:", our_grid_thw.shape)

    # Print differences
    print("\nDifferences:")
    print("-" * 50)

    # pixel_values
    diff_pixel_values = torch.abs(hf_pixel_values - our_pixel_values.cpu()).sum().item()
    numel_pixel_values = hf_pixel_values.numel()
    print("pixel_values:")
    print("  Total difference:", diff_pixel_values)
    print("  Average difference:", diff_pixel_values / numel_pixel_values)

    # input_ids
    diff_input_ids = torch.abs(hf_input_ids - our_input_ids.cpu()).sum().item()
    numel_input_ids = hf_input_ids.numel()
    print("\ninput_ids:")
    print("  Total difference:", diff_input_ids)
    print("  Average difference:", diff_input_ids / numel_input_ids)

    # grid_thw
    diff_grid = torch.abs(hf_grid_thw - our_grid_thw.cpu()).sum().item()
    numel_grid = hf_grid_thw.numel()
    print("\ngrid_thw:")
    print("  Total difference:", diff_grid)
    print("  Average difference:", diff_grid / numel_grid)
