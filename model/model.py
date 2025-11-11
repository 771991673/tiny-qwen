import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from dataclasses import dataclass
from .utils import rename_dict_keys, load_pretrained_model
from .vision import VisionEncoder, VisionConfig


@dataclass
class ModelConfig:
    repo_id: str
    n_embed: int
    n_heads: int
    n_kv_heads: int
    n_layer: int
    n_dense_mlp: int
    rope_theta: float
    rms_norm_eps: float
    n_vocab: int
    tie_word_embeddings: bool
    max_position_embeddings: int
    max_window_layers: int
    eos_token_id: int
    head_dim: Optional[int] = None
    n_moe_mlp: Optional[int] = None

    # MoE parameters
    n_experts: Optional[int] = None
    n_experts_per_tok: Optional[int] = None

    @classmethod
    def from_pretrained(cls, config: dict):
        config = rename_dict_keys(config, HF_TO_LM_CONFIG)
        return cls(**config)


class RotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Use explicit head_dim if provided, otherwise calculate
        d = (
            config.head_dim
            if config.head_dim is not None
            else (config.n_embed // config.n_heads)
        )
        t = config.rope_theta
        r = torch.arange(0, d, 2)
        inv_freq = 1.0 / (t ** (r / d)).float()
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # M-RoPE section for multimodal (temporal, height, width)
        # Default from Qwen3-VL: [24, 20, 20] for 64-dim head
        self.mrope_section = [16, 24, 24]  # For typical 64-dim head

    def forward(self, x, position_ids):
        """
        Args:
            x: hidden states
            position_ids: 2D [batch, seq] for text-only, or 3D [3, batch, seq] for multimodal

        Returns:
            Tuple of (cos, sin) tensors
        """
        # Handle 2D (text-only) vs 3D (multimodal) position_ids
        if position_ids.ndim == 2:
            # Text-only mode: standard RoPE
            inv_freq = self.inv_freq.to(x.device)
            position_ids = position_ids.unsqueeze(-1).float()
            freqs = position_ids * inv_freq
            emb = torch.cat([freqs, freqs], dim=-1)
            cos = emb.cos().to(x.dtype)
            sin = emb.sin().to(x.dtype)
            return cos, sin
        else:
            # Multimodal mode: M-RoPE with 3D position_ids [3, batch, seq]
            inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
            position_ids_expanded = position_ids[:, :, None, :].float()  # shape (3, bs, 1, positions)

            device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
            with torch.autocast(device_type=device_type, enabled=False):  # Force float32
                freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
                freqs = self._apply_interleaved_mrope(freqs, self.mrope_section)
                emb = torch.cat((freqs, freqs), dim=-1)
                cos = emb.cos()
                sin = emb.sin()

            return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    def _apply_interleaved_mrope(self, freqs, mrope_section):
        """Apply interleaved MRoPE to 3D rotary embeddings.
        Reorganizes frequency layout from chunked [TTT...HHH...WWW] to
        interleaved [THTHWHTHW...TT], preserving frequency continuity.

        Args:
            freqs: (3, bs, seq_len, head_dim // 2)
            mrope_section: list of 3 ints [t_section, h_section, w_section]

        Returns:
            freqs_t: (bs, seq_len, head_dim // 2)
        """
        freqs_t = freqs[0]  # Start with temporal frequencies
        for dim, offset in enumerate((1, 2), start=1):  # H, W dimensions
            length = mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t


class Qwen3DenseAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads

        self.n_embed = config.n_embed
        self.n_embed_per_head = config.n_embed // config.n_heads
        self.n_kv_embed = config.n_kv_heads * self.n_embed_per_head

        self.q_proj = nn.Linear(self.n_embed, self.n_embed, bias=False)
        self.k_proj = nn.Linear(self.n_embed, self.n_kv_embed, bias=False)
        self.v_proj = nn.Linear(self.n_embed, self.n_kv_embed, bias=False)
        self.o_proj = nn.Linear(self.n_embed, self.n_embed, bias=False)

        # Qwen3 specific: q_norm and k_norm on head dimension
        self.q_norm = Qwen3RMSNorm(self.n_embed_per_head, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.n_embed_per_head, eps=config.rms_norm_eps)

    def forward(self, x, cos, sin):
        B, T, C = x.size()

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(B, T, self.n_heads, self.n_embed_per_head).transpose(1, 2)
        k = k.view(B, T, self.n_kv_heads, self.n_embed_per_head).transpose(1, 2)
        v = v.view(B, T, self.n_kv_heads, self.n_embed_per_head).transpose(1, 2)

        # Apply normalization to q and k before RoPE (Qwen3 specific)
        q = self.q_norm(q.transpose(1, 2)).transpose(1, 2)
        k = self.k_norm(k.transpose(1, 2)).transpose(1, 2)

        q, k = self._apply_rotary_pos_emb(q, k, cos, sin)

        if self.n_kv_heads < self.n_heads:
            num_repeat = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(num_repeat, dim=1)
            v = v.repeat_interleave(num_repeat, dim=1)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.o_proj(y)
        return y

    @staticmethod
    def _apply_rotary_pos_emb(q, k, cos, sin):
        if cos.dim() == 4:
            # shape [B, 3, T, D] -> multi-modal
            cos = Qwen3DenseAttention._process_rotary_component(cos)
            sin = Qwen3DenseAttention._process_rotary_component(sin)
        else:
            # shape [B, T, D] -> text-only
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)

        q_embed = (q * cos) + (Qwen3DenseAttention._rotate_half(q) * sin)
        k_embed = (k * cos) + (Qwen3DenseAttention._rotate_half(k) * sin)
        return q_embed, k_embed

    @staticmethod
    def _process_rotary_component(x):
        # Split into sections and select appropriate indices
        sections = x.split([16, 24, 24, 16, 24, 24], dim=-1)
        processed = [m[i % 3] for i, m in enumerate(sections)]
        # Combine and add dimension
        return torch.cat(processed, dim=-1).unsqueeze(1)

    @staticmethod
    def _rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)


class Qwen3MoeAttention(nn.Module):
    """Qwen3 MoE attention with explicit head_dim support"""

    def __init__(self, config):
        super().__init__()

        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_embed = config.n_embed

        # Use explicit head_dim if provided, otherwise calculate
        self.head_dim = (
            config.head_dim
            if config.head_dim is not None
            else (config.n_embed // config.n_heads)
        )

        self.q_proj = nn.Linear(self.n_embed, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(
            self.n_embed, self.n_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.n_embed, self.n_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, self.n_embed, bias=False)

        # Qwen3 specific: q_norm and k_norm on head dimension
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(self, x, cos, sin):
        B, T, C = x.size()

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply normalization to q and k before RoPE (Qwen3 specific)
        q = self.q_norm(q.transpose(1, 2)).transpose(1, 2)
        k = self.k_norm(k.transpose(1, 2)).transpose(1, 2)

        q, k = self._apply_rotary_pos_emb(q, k, cos, sin)

        if self.n_kv_heads < self.n_heads:
            num_repeat = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(num_repeat, dim=1)
            v = v.repeat_interleave(num_repeat, dim=1)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        y = self.o_proj(y)
        return y

    @staticmethod
    def _apply_rotary_pos_emb(q, k, cos, sin):
        if cos.dim() == 4:
            # shape [B, 3, T, D] -> multi-modal
            cos = Qwen3MoeAttention._process_rotary_component(cos)
            sin = Qwen3MoeAttention._process_rotary_component(sin)
        else:
            # shape [B, T, D] -> text-only
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)

        q_embed = (q * cos) + (Qwen3MoeAttention._rotate_half(q) * sin)
        k_embed = (k * cos) + (Qwen3MoeAttention._rotate_half(k) * sin)
        return q_embed, k_embed

    @staticmethod
    def _process_rotary_component(x):
        # Split into sections and select appropriate indices
        sections = x.split([16, 24, 24, 16, 24, 24], dim=-1)
        processed = [m[i % 3] for i, m in enumerate(sections)]
        # Combine and add dimension
        return torch.cat(processed, dim=-1).unsqueeze(1)

    @staticmethod
    def _rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)


class Qwen3RMSNorm(nn.Module):
    def __init__(self, n_embed, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_embed))
        self.variance_epsilon = eps

    def forward(self, x):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * x.to(input_dtype)


class Qwen3DenseMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.n_embed, config.n_mlp, bias=False)
        self.up_proj = nn.Linear(config.n_embed, config.n_mlp, bias=False)
        self.down_proj = nn.Linear(config.n_mlp, config.n_embed, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3MoEMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts_per_tok = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.gate = nn.Linear(config.n_embed, config.num_experts, bias=False)

        # Expert layers with proper naming to match checkpoint
        self.experts = nn.ModuleList()
        for _ in range(config.num_experts):
            expert = nn.Module()
            expert.gate_proj = nn.Linear(
                config.n_embed, config.moe_intermediate_size, bias=False
            )
            expert.up_proj = nn.Linear(
                config.n_embed, config.moe_intermediate_size, bias=False
            )
            expert.down_proj = nn.Linear(
                config.moe_intermediate_size, config.n_embed, bias=False
            )
            self.experts.append(expert)

    def forward(self, x):
        b, seq_len, embed_dim = x.shape
        scores = self.gate(x)  # (b, seq_len, num_experts)
        topk_scores, topk_indices = torch.topk(scores, self.num_experts_per_tok, dim=-1)
        topk_probs = torch.softmax(topk_scores, dim=-1)

        expert_outputs = []
        for e in range(self.num_experts):
            expert = self.experts[e]
            hidden = F.silu(expert.gate_proj(x)) * expert.up_proj(x)
            out = expert.down_proj(hidden)
            expert_outputs.append(out.unsqueeze(-2))
        expert_outputs = torch.cat(
            expert_outputs, dim=-2
        )  # (b, t, num_experts, emb_dim)

        gating_probs = torch.zeros_like(scores)

        for i in range(self.num_experts_per_tok):
            indices = topk_indices[..., i : i + 1]
            prob = topk_probs[..., i : i + 1]
            gating_probs.scatter_(dim=-1, index=indices, src=prob)
        gating_probs = gating_probs.unsqueeze(-1)  # (b, t, num_experts, 1)

        # Weighted sum over experts
        y = (gating_probs * expert_outputs).sum(dim=-2)
        return y


class Qwen3DenseBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        n_embed, eps = config.n_embed, config.rms_norm_eps
        self.input_layernorm = Qwen3RMSNorm(n_embed=n_embed, eps=eps)
        self.self_attn = Qwen3DenseAttention(config)
        self.post_attention_layernorm = Qwen3RMSNorm(n_embed=n_embed, eps=eps)
        self.mlp = Qwen3DenseMLP(config)

    def forward(self, x, cos, sin):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Qwen3MoEBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        n_embed, eps = config.n_embed, config.rms_norm_eps
        self.input_layernorm = Qwen3RMSNorm(n_embed=n_embed, eps=eps)
        self.self_attn = Qwen3MoeAttention(config)
        self.post_attention_layernorm = Qwen3RMSNorm(n_embed=n_embed, eps=eps)

        # Use MoE if experts are configured, otherwise regular MLP
        if config.num_experts and config.num_experts > 0:
            self.mlp = Qwen3MoEMLP(config)
        else:
            self.mlp = Qwen3DenseMLP(config)

    def forward(self, x, cos, sin):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Qwen3DenseModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.n_embed)
        self.rotary_emb = RotaryEmbedding(config)

        # Use Qwen3Block with proper attention
        self.layers = nn.ModuleList(
            Qwen3DenseBlock(config) for _ in range(config.n_layer)
        )
        self.norm = Qwen3RMSNorm(config.n_embed, eps=config.rms_norm_eps)

        # Store config for convenience
        self.config = config

    def forward(self, x, position_ids, visual_pos_masks=None, deepstack_visual_embeds=None):
        """
        Args:
            x: input embeddings
            position_ids: 2D [batch, seq] or 3D [3, batch, seq] for multimodal
            visual_pos_masks: optional tensor marking visual token positions
            deepstack_visual_embeds: optional list of visual features to inject
        """
        cos, sin = self.rotary_emb(x, position_ids)
        for layer_idx, layer in enumerate(self.layers):
            x = layer(x, cos, sin)

            # Inject deepstack visual features if provided
            if deepstack_visual_embeds is not None and layer_idx < len(deepstack_visual_embeds):
                if deepstack_visual_embeds[layer_idx] is not None:
                    x = self._deepstack_process(x, visual_pos_masks, deepstack_visual_embeds[layer_idx])

        x = self.norm(x)
        return x

    def _deepstack_process(self, hidden_states, visual_pos_masks, visual_embeds):
        """Inject visual features into hidden states at visual token positions"""
        visual_pos_masks = visual_pos_masks.to(hidden_states.device)
        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
        hidden_states = hidden_states.clone()
        hidden_states[visual_pos_masks] = visual_embeds
        return hidden_states


class Qwen3Dense(nn.Module):
    """Qwen3 dense model - text-only version"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model = Qwen3DenseModel(config)

        self.lm_head = None
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)

    def _get_position_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        device = input_ids.device
        position_ids = torch.arange(T, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(B, -1)
        return position_ids

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.model.embed_tokens(input_ids)
        position_ids = self._get_position_ids(input_ids)
        x = self.model(x=x, position_ids=position_ids)

        if self.lm_head is None:
            logits = torch.matmul(x, self.model.embed_tokens.weight.T)
        else:
            logits = self.lm_head(x)
        return logits

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 1,
        stop_tokens: list = None,
        stream: bool = False,
    ):
        if stop_tokens is None:
            stop_tokens = [
                151645,
                151644,
                151643,
            ]  # <|im_end|>, <|im_start|>, <|endoftext|>

        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = self.forward(input_ids=input_ids)
                last_logits = logits[:, -1, :]
                probs = F.softmax(last_logits, dim=-1)
                next_token = probs.argmax(dim=-1, keepdim=True)
                input_ids = torch.cat([input_ids, next_token], dim=1)

                # If streaming, yield the new token
                if stream:
                    yield next_token.item()

                # Check if we hit a stop token
                if next_token.item() in stop_tokens:
                    break

        # If not streaming, return the full input_ids
        if not stream:
            return input_ids

    @classmethod
    def get_config_class(cls):
        return ModelConfig

    @classmethod
    def from_pretrained(cls, repo_id: str, device_map: str = "auto"):
        return load_pretrained_model(cls, repo_id, device_map=device_map)


class Qwen3MoEModel(nn.Module):
    """MoE for Mixture of Experts"""

    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.n_embed)
        self.rotary_emb = RotaryEmbedding(config)

        # Use Qwen3MoeBlock with proper attention and MoE
        self.layers = nn.ModuleList(
            Qwen3MoEBlock(config) for _ in range(config.n_layer)
        )
        self.norm = Qwen3RMSNorm(config.n_embed, eps=config.rms_norm_eps)

        # Store config for convenience
        self.config = config

    def forward(self, x, position_ids, visual_pos_masks=None, deepstack_visual_embeds=None):
        """
        Args:
            x: input embeddings
            position_ids: 2D [batch, seq] or 3D [3, batch, seq] for multimodal
            visual_pos_masks: optional tensor marking visual token positions
            deepstack_visual_embeds: optional list of visual features to inject
        """
        cos, sin = self.rotary_emb(x, position_ids)
        for layer_idx, layer in enumerate(self.layers):
            x = layer(x, cos, sin)

            # Inject deepstack visual features if provided
            if deepstack_visual_embeds is not None and layer_idx < len(deepstack_visual_embeds):
                if deepstack_visual_embeds[layer_idx] is not None:
                    x = self._deepstack_process(x, visual_pos_masks, deepstack_visual_embeds[layer_idx])

        x = self.norm(x)
        return x

    def _deepstack_process(self, hidden_states, visual_pos_masks, visual_embeds):
        """Inject visual features into hidden states at visual token positions"""
        visual_pos_masks = visual_pos_masks.to(hidden_states.device)
        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
        hidden_states = hidden_states.clone()
        hidden_states[visual_pos_masks] = visual_embeds
        return hidden_states


class Qwen3MoE(nn.Module):
    """Qwen3 MoE model - text-only version with mixture of experts"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model = Qwen3MoEModel(config)

        self.lm_head = None
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)

    def _get_position_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        device = input_ids.device
        position_ids = torch.arange(T, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(B, -1)
        return position_ids

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.model.embed_tokens(input_ids)
        position_ids = self._get_position_ids(input_ids)
        x = self.model(x=x, position_ids=position_ids)

        if self.lm_head is None:
            logits = torch.matmul(x, self.model.embed_tokens.weight.T)
        else:
            logits = self.lm_head(x)
        return logits

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 1,
        stop_tokens: list = None,
        stream: bool = False,
    ):
        if stop_tokens is None:
            stop_tokens = [
                151645,
                151644,
                151643,
            ]  # <|im_end|>, <|im_start|>, <|endoftext|>

        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = self.forward(input_ids=input_ids)
                last_logits = logits[:, -1, :]
                probs = F.softmax(last_logits, dim=-1)
                next_token = probs.argmax(dim=-1, keepdim=True)
                input_ids = torch.cat([input_ids, next_token], dim=1)

                # If streaming, yield the new token
                if stream:
                    yield next_token.item()

                # Check if we hit a stop token
                if next_token.item() in stop_tokens:
                    break

        # If not streaming, return the full input_ids
        if not stream:
            return input_ids

    @classmethod
    def get_config_class(cls):
        return ModelConfig

    @classmethod
    def from_pretrained(cls, repo_id: str, device_map: str = "auto"):
        return load_pretrained_model(cls, repo_id, device_map=device_map)


# ============================================================================
# Multimodal VL Models
# ============================================================================


@dataclass
class VLConfig:
    """Configuration for multimodal VL models"""
    text_config: ModelConfig
    vision_config: VisionConfig
    repo_id: str = None

    @classmethod
    def from_pretrained(cls, config: dict):
        text_cfg = ModelConfig.from_pretrained(config.get("text_config", config))
        vision_cfg = VisionConfig.from_pretrained(config.get("vision_config", {}))
        return cls(text_config=text_cfg, vision_config=vision_cfg, repo_id=config.get("repo_id"))


class Qwen3VLDense(nn.Module):
    """Qwen3-VL Dense multimodal model"""

    def __init__(self, config: VLConfig):
        super().__init__()
        self.config = config
        self.vision_encoder = VisionEncoder(config.vision_config)
        self.language_model = Qwen3DenseModel(config.text_config)

        self.lm_head = None
        if not config.text_config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.text_config.n_embed, config.text_config.n_vocab, bias=False)

    def get_rope_index(
        self,
        input_ids: torch.Tensor,
        image_grid_thw: Optional[torch.Tensor] = None,
    ):
        """
        Generate M-RoPE position indices based on image grids.

        Returns:
            position_ids: 3D tensor [3, batch, seq] with temporal/height/width positions
            mrope_position_deltas: Position deltas for tracking offset
        """
        # Build position IDs - simple sequential for batch size 1
        seq_len = input_ids.shape[1]
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).unsqueeze(0).expand(3, input_ids.shape[0], -1)

        mrope_position_deltas = position_ids[0].max(dim=-1, keepdim=True)[0] + 1 - input_ids.shape[1]

        return position_ids, mrope_position_deltas

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass for multimodal input.

        Args:
            input_ids: Text token IDs [batch, seq]
            pixel_values: Image pixels (optional)
            image_grid_thw: Image grid dimensions [num_images, 3] (temporal, height, width)

        Returns:
            logits: Language model logits
        """
        # Process vision inputs
        visual_features = None
        deepstack_visual_embeds = None
        visual_pos_masks = None

        if pixel_values is not None and image_grid_thw is not None:
            # Run vision encoder
            visual_features, deepstack_features_list = self.vision_encoder(pixel_values, image_grid_thw)

            # Prepare deepstack embeddings (pad to match number of layers)
            num_layers = len(self.language_model.layers)
            deepstack_visual_embeds = [None] * num_layers
            if self.vision_encoder.deepstack_visual_indexes:
                for idx, layer_idx in enumerate(self.vision_encoder.deepstack_visual_indexes):
                    if layer_idx < num_layers:
                        deepstack_visual_embeds[layer_idx] = deepstack_features_list[idx]

            # Create visual position mask (marks where visual tokens are in the sequence)
            # This is a simplified version - in practice you'd track actual positions
            visual_pos_masks = torch.zeros(input_ids.shape[0], input_ids.shape[1], dtype=torch.bool, device=input_ids.device)

        # Embed text tokens
        inputs_embeds = self.language_model.embed_tokens(input_ids)

        # Merge visual features into text embeddings if present
        # Note: This is simplified - real implementation needs proper position tracking
        if visual_features is not None:
            pass  # TODO: Implement visual feature merging based on special tokens

        # Generate position IDs
        position_ids, _ = self.get_rope_index(input_ids, image_grid_thw)

        # Run language model
        hidden_states = self.language_model(
            inputs_embeds,
            position_ids=position_ids,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )

        # Generate logits
        if self.lm_head is None:
            logits = torch.matmul(hidden_states, self.language_model.embed_tokens.weight.T)
        else:
            logits = self.lm_head(hidden_states)

        return logits

    @classmethod
    def from_pretrained(cls, repo_id: str, device_map: str = "auto"):
        return load_pretrained_model(cls, repo_id, device_map=device_map)


class Qwen3VLMoE(nn.Module):
    """Qwen3-VL MoE multimodal model"""

    def __init__(self, config: VLConfig):
        super().__init__()
        self.config = config
        self.vision_encoder = VisionEncoder(config.vision_config)
        self.language_model = Qwen3MoEModel(config.text_config)

        self.lm_head = None
        if not config.text_config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.text_config.n_embed, config.text_config.n_vocab, bias=False)

    def get_rope_index(
        self,
        input_ids: torch.Tensor,
        image_grid_thw: Optional[torch.Tensor] = None,
    ):
        """Generate M-RoPE position indices - same as Qwen3VLDense"""
        # Build position IDs - simple sequential for batch size 1
        seq_len = input_ids.shape[1]
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).unsqueeze(0).expand(3, input_ids.shape[0], -1)

        mrope_position_deltas = position_ids[0].max(dim=-1, keepdim=True)[0] + 1 - input_ids.shape[1]

        return position_ids, mrope_position_deltas

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
    ):
        """Forward pass - same structure as Qwen3VLDense"""
        # Process vision inputs
        visual_features = None
        deepstack_visual_embeds = None
        visual_pos_masks = None

        if pixel_values is not None and image_grid_thw is not None:
            # Run vision encoder
            visual_features, deepstack_features_list = self.vision_encoder(pixel_values, image_grid_thw)

            num_layers = len(self.language_model.layers)
            deepstack_visual_embeds = [None] * num_layers
            if self.vision_encoder.deepstack_visual_indexes:
                for idx, layer_idx in enumerate(self.vision_encoder.deepstack_visual_indexes):
                    if layer_idx < num_layers:
                        deepstack_visual_embeds[layer_idx] = deepstack_features_list[idx]

            visual_pos_masks = torch.zeros(input_ids.shape[0], input_ids.shape[1], dtype=torch.bool, device=input_ids.device)

        inputs_embeds = self.language_model.embed_tokens(input_ids)

        if visual_features is not None:
            pass  # TODO: Implement visual feature merging

        position_ids, _ = self.get_rope_index(input_ids, image_grid_thw)

        hidden_states = self.language_model(
            inputs_embeds,
            position_ids=position_ids,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )

        if self.lm_head is None:
            logits = torch.matmul(hidden_states, self.language_model.embed_tokens.weight.T)
        else:
            logits = self.lm_head(hidden_states)

        return logits

    @classmethod
    def from_pretrained(cls, repo_id: str, device_map: str = "auto"):
        return load_pretrained_model(cls, repo_id, device_map=device_map)


# Maps: HuggingFace config key -> tiny-qwen config key
HF_TO_LM_CONFIG = {
    "hidden_size": "n_embed",
    "num_attention_heads": "n_heads",
    "num_key_value_heads": "n_kv_heads",
    "num_hidden_layers": "n_layer",
    "intermediate_size": "n_dense_mlp",
    "rope_theta": "rope_theta",
    "rms_norm_eps": "rms_norm_eps",
    "vocab_size": "n_vocab",
    "tie_word_embeddings": "tie_word_embeddings",
    "max_position_embeddings": "max_position_embeddings",
    "max_window_layers": "max_window_layers",
    "eos_token_id": "eos_token_id",
    "head_dim": "head_dim",
    "num_experts": "n_experts",
    "num_experts_per_tok": "n_experts_per_tok",
    "moe_intermediate_size": "n_moe_mlp",
}


# Maps: HuggingFace component name -> tiny-qwen component name
HF_TO_LM_WEIGHTS = {}
