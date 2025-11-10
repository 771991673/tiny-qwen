import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from dataclasses import dataclass


@dataclass
class Qwen3VLVisionEncoderConfig:
    n_embed: int
    n_heads: int
    n_kv_heads: int
    n_layer: int
    n_mlp: int


@dataclass
class Qwen3VLTextDecoderConfig:
    repo_id: str

    n_embed: int
    n_heads: int
    n_kv_heads: int
    n_layer: int
    n_mlp: int
    rope_theta: float
    rms_norm_eps: float
    vocab_size: int
    tie_word_embeddings: bool

    vision_encoder_config: Qwen3VLVisionEncoderConfig

    # MoE parameters
    num_experts: Optional[int] = None
    num_experts_per_tok: Optional[int] = None
    moe_intermediate_size: Optional[int] = None
