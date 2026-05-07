#!/usr/bin/env python3
"""
export_weights.py - Export Qwen3.5-0.8B model weights for WebGPU demo.

Usage:
    .venv/bin/python webgpu-demo/export_weights.py

Outputs:
    weights.bin      - packed float16 weights (big-endian layout)
    metadata.json    - weight shapes, model config, kernel info
"""

import json, os, struct, sys
from pathlib import Path

import torch
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from huggingface_hub import snapshot_download
from huggingface_hub.utils import disable_progress_bars
disable_progress_bars()

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from contextlib import redirect_stderr
import io

from model.model import Qwen3_5, ModelConfig
from model.vision import VisionConfig

HF_REPO = "Qwen/Qwen3.5-0.8B"
OUT_DIR = Path(__file__).resolve().parent


def export_weights():
    print(f"[1/3] Downloading model: {HF_REPO}")
    with redirect_stderr(io.StringIO()):
        weights_path = snapshot_download(repo_id=HF_REPO, cache_dir="../.cache")
    
    model_path = Path(weights_path)
    
    # Load config
    with open(model_path / "config.json") as f:
        hf_config = json.load(f)
    
    # Save full config for reference
    with open(OUT_DIR / "hf_config.json", "w") as f:
        json.dump(hf_config, f, indent=2)
    
    print(f"[2/3] Loading model weights...")
    
    # Load model on CPU without accelerate device_map
    with open(model_path / "config.json") as f:
        hf_config_inner = json.load(f)
    
    llm_config_inner = hf_config_inner["text_config"]
    vis_config_inner = hf_config_inner.get("vision_config")
    
    # Build config
    n_mlp = llm_config_inner.get("intermediate_size")
    if n_mlp is None:
        n_mlp = llm_config_inner.get("moe_intermediate_size")
    
    config = ModelConfig(
        n_embed=llm_config_inner["hidden_size"],
        n_heads=llm_config_inner["num_attention_heads"],
        n_kv_heads=llm_config_inner["num_key_value_heads"],
        n_layer=llm_config_inner["num_hidden_layers"],
        n_mlp=n_mlp,
        n_vocab=llm_config_inner["vocab_size"],
        tie_word_embeddings=hf_config_inner["tie_word_embeddings"],
        rope_theta=llm_config_inner.get("rope_parameters", {}).get("rope_theta", 10000000.0),
        rms_norm_eps=llm_config_inner.get("rms_norm_eps", 1e-6),
        d_head=llm_config_inner.get("head_dim"),
        n_experts=llm_config_inner.get("num_experts"),
        n_experts_per_token=llm_config_inner.get("num_experts_per_tok"),
        n_moe_mlp=llm_config_inner.get("moe_intermediate_size"),
        n_shared_expert_mlp=llm_config_inner.get("shared_expert_intermediate_size"),
        layer_types=llm_config_inner.get("layer_types"),
        n_linear_k_heads=llm_config_inner.get("linear_num_key_heads"),
        n_linear_v_heads=llm_config_inner.get("linear_num_value_heads"),
        d_linear_k=llm_config_inner.get("linear_key_head_dim"),
        d_linear_v=llm_config_inner.get("linear_value_head_dim"),
        linear_conv_kernel=llm_config_inner.get("linear_conv_kernel_dim", 4),
        partial_rotary_factor=llm_config_inner.get("partial_rotary_factor", 1.0),
    )
    
    vision_cfg = None
    if vis_config_inner is not None:
        vision_cfg = VisionConfig(
            n_embed=vis_config_inner["hidden_size"],
            n_layer=vis_config_inner["depth"],
            n_heads=vis_config_inner["num_heads"],
            n_output_embed=vis_config_inner["out_hidden_size"],
            n_mlp=vis_config_inner["intermediate_size"],
            num_position_embeddings=vis_config_inner["num_position_embeddings"],
            in_channels=vis_config_inner["in_channels"],
            temporal_patch_size=vis_config_inner["temporal_patch_size"],
            patch_size=vis_config_inner["patch_size"],
            spatial_merge_size=vis_config_inner["spatial_merge_size"],
        )
    
    model = Qwen3_5(config, vision_config=vision_cfg)
    
    # Load state dict from safetensors directly
    from safetensors import safe_open
    safetensors_files = sorted(model_path.glob("*.safetensors"))
    if not safetensors_files:
        safetensors_files = sorted(model_path.glob("model-*.safetensors"))
    
    print(f"   Found {len(safetensors_files)} safetensors file(s)")
    state = {}
    for st_file in safetensors_files:
        print(f"   Loading: {st_file.name}")
        with safe_open(st_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                state[key] = f.get_tensor(key).to(torch.float16).numpy()
    
    print(f"   Extracted {len(state)} tensors")
    
    # Also extract model config info for export
    llm_config = hf_config["text_config"]
    vis_config = hf_config.get("vision_config")
    
    print(f"[3/3] Writing output files...")
    
    # Write weights.bin
    weights_bin = bytearray()
    weight_meta = []
    offset = 0
    
    # Sort keys for deterministic order
    for key in sorted(state.keys()):
        arr = np.ascontiguousarray(state[key])
        raw = arr.tobytes()
        weight_meta.append({
            "name": key,
            "offset": offset,
            "size": len(raw),
            "shape": list(arr.shape),
            "dtype": "float16",
        })
        weights_bin.extend(raw)
        offset += len(raw)
    
    with open(OUT_DIR / "weights.bin", "wb") as f:
        f.write(weights_bin)
    
    print(f"   weights.bin: {len(weights_bin)} bytes ({len(weights_bin)/1024/1024:.1f} MB)")
    
    # Build metadata
    metadata = {
        "model_name": "Qwen3.5-0.8B",
        "model_type": "multimodal",
        "description": "Qwen3.5-0.8B multimodal model (vision + text)",
        "weight_count": len(weight_meta),
        "weight_bytes": len(weights_bin),
        "use_fp16": True,
        "weights": weight_meta,
        "text_config": {
            "hidden_size": llm_config["hidden_size"],
            "num_attention_heads": llm_config["num_attention_heads"],
            "num_key_value_heads": llm_config["num_key_value_heads"],
            "num_hidden_layers": llm_config["num_hidden_layers"],
            "intermediate_size": llm_config.get("intermediate_size", 
                llm_config.get("shared_expert_intermediate_size")),
            "vocab_size": llm_config["vocab_size"],
            "head_dim": llm_config.get("head_dim", 
                llm_config["hidden_size"] // llm_config["num_attention_heads"]),
            "rms_norm_eps": llm_config["rms_norm_eps"],
            "rope_theta": llm_config.get("rope_parameters", {}).get("rope_theta", 10000000.0),
            "partial_rotary_factor": llm_config.get("rope_parameters", {}).get("partial_rotary_factor", 0.25),
            "layer_types": llm_config.get("layer_types", []),
            "tie_word_embeddings": hf_config.get("tie_word_embeddings", True),
            # Linear attention params
            "linear_num_key_heads": llm_config.get("linear_num_key_heads", 16),
            "linear_num_value_heads": llm_config.get("linear_num_value_heads", 16),
            "linear_key_head_dim": llm_config.get("linear_key_head_dim", 128),
            "linear_value_head_dim": llm_config.get("linear_value_head_dim", 128),
            "linear_conv_kernel": llm_config.get("linear_conv_kernel_dim", 4),
            "image_token_id": hf_config.get("image_token_id", 248056),
        },
    }
    
    if vis_config:
        metadata["vision_config"] = {
            "hidden_size": vis_config["hidden_size"],
            "num_heads": vis_config["num_heads"],
            "num_layers": vis_config["depth"],
            "intermediate_size": vis_config["intermediate_size"],
            "out_hidden_size": vis_config["out_hidden_size"],
            "patch_size": vis_config["patch_size"],
            "temporal_patch_size": vis_config["temporal_patch_size"],
            "spatial_merge_size": vis_config["spatial_merge_size"],
            "num_position_embeddings": vis_config["num_position_embeddings"],
            "in_channels": vis_config["in_channels"],
        }
    
    with open(OUT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"   metadata.json")
    print(f"\n✅ Done! Files written to {OUT_DIR}")
    print(f"   Use these with the WebGPU demo (index.html)")


if __name__ == "__main__":
    export_weights()
