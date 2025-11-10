import json
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


USER_MESSAGE_TEMPLATE = "<|im_start|>user\n{content}<|im_end|>\n"
ASSISTANT_MESSAGE_TEMPLATE = "<|im_start|>assistant\n{content}{tool_calls}<|im_end|>\n"
TOOL_MESSAGE_TEMPLATE = (
    "<|im_start|>user\n<tool_response>\n{content}\n</tool_response><|im_end|>\n"
)
SYSTEM_MESSAGE_TEMPLATE = "<|im_start|>system\n{content}<|im_end|>\n"

IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"

TOOL_CALL_TEMPLATE = (
    '<tool_call>\n{{"name": "{name}", "arguments": {arguments}}}\n</tool_call>'
)
TOOL_RESPONSE_TEMPLATE = (
    "<|im_start|>user\n<tool_response>\n{content}\n</tool_response><|im_end|>\n"
)


class Processor:
    def __init__(self, model_config):
        self.tokenizer = Tokenizer.from_pretrained(model_config.repo_id)

        if model_config.vision_config is not None:
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
        messages_str = self._render_messages(messages)
        # input_ids = self.tokenizer.encode(messages_str).ids
        # attention_mask = torch.ones_like(input_ids)
        return messages_str
        # pixels_list = []
        # d_image_list = []

        # # Identify if we have vision support
        # has_vision = self.vision_config is not None
        # if not has_vision:
        #     # If we have no vision_config, do text-only
        #     for item in inputs:
        #         if isinstance(item, str):
        #             input_ids.extend(self.tokenizer.encode(item).ids)
        #         else:
        #             raise ValueError(
        #                 f"Images are not supported by a text-only model. Got {type(item)}"
        #             )
        # else:
        #     # Vision + text model
        #     merge_size = self.vision_config.spatial_merge_size
        #     image_pad_token_id = self.image_pad_token_id

        #     for item in inputs:
        #         if isinstance(item, str):
        #             # Handle text
        #             input_ids.extend(self.tokenizer.encode(item).ids)
        #         elif isinstance(item, Image.Image):
        #             # Handle image
        #             patches, t, h, w = self._process_image(item)
        #             pixels_list.append(patches)
        #             d_image_list.append([t, h, w])

        #             pad_token_count = (t * h * w) // (merge_size**2)
        #             pad_tokens = [image_pad_token_id] * pad_token_count
        #             input_ids.extend(pad_tokens)
        #         else:
        #             raise ValueError(f"Unsupported input type: {type(item)}")

        # # Convert accumulated ids to tensor
        # input_ids = torch.tensor([input_ids], dtype=torch.long)

        # # Convert all images to tensor if there are any
        # if pixels_list:
        #     pixels_np = np.concatenate(pixels_list, axis=0)
        #     pixels = torch.tensor(pixels_np, dtype=torch.float)
        #     d_image = torch.tensor(d_image_list, dtype=torch.long)
        # else:
        #     pixels = None
        #     d_image = None

        # attention_mask = torch.ones_like(input_ids)

        # return ModelInputs(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     pixel_values=pixels,
        #     image_grid_thw=d_image,
        # )

    def _render_messages(self, messages: List[dict]) -> str:
        rendered = [self._render_message(message) for message in messages]
        return "".join(rendered)

    def _render_message(self, message: dict) -> str:
        assert isinstance(message, dict), f"Message must be a dict, got {type(message)}"
        assert "role" in message, f"Message must have a role, got {message}"

        role = message["role"]
        content = message.get("content", "")

        tool_calls = message.get("tool_calls", [])
        tool_calls = [self._render_tool_call(tool_call) for tool_call in tool_calls]
        tool_call_str = "".join(tool_calls)

        if isinstance(content, str):
            content_str = content
        elif isinstance(content, list):
            content_str = "".join([self._render_content(item) for item in content])
        else:
            content_str = ""

        if role == "system":
            return SYSTEM_MESSAGE_TEMPLATE.format(content=content_str)
        elif role == "user":
            return USER_MESSAGE_TEMPLATE.format(content=content_str)
        elif role == "assistant":
            return ASSISTANT_MESSAGE_TEMPLATE.format(
                content=content_str, tool_calls=tool_call_str
            )
        elif role == "tool":
            return TOOL_RESPONSE_TEMPLATE.format(content=content_str)
        else:
            raise ValueError(f"Unsupported role: {role}")

    def _render_content(self, content: dict | str) -> str:
        assert isinstance(content, dict) or isinstance(
            content, str
        ), f"Content must be a string or a dict, got {type(content)}"
        assert (
            isinstance(content, dict) and "type" in content
        ), f"Content must be a dict with a type, got {content}"

        if isinstance(content, str):
            return content
        if content["type"] == "text":
            return content["text"]
        elif content["type"] == "image":
            return IMAGE_PLACEHOLDER
        else:
            raise ValueError(f"Unsupported content type: {content['type']}")

    def _render_tool_call(self, tool_call: dict) -> str:
        assert isinstance(
            tool_call, dict
        ), f"Tool call must be a dict, got {type(tool_call)}"
        assert (
            "name" in tool_call and "arguments" in tool_call
        ), f"Tool call must have a name and arguments, got {tool_call}"
        return TOOL_CALL_TEMPLATE.format(
            name=tool_call["name"],
            arguments=tool_call["arguments"],
        )

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
