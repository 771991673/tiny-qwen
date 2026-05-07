#!/usr/bin/env python3
"""
export_vision_weights.py — Export ONLY the VisionEncoder weights from
Qwen/Qwen3.5-0.8B for the WebGPU demo.

Outputs:
    weights.bin      - packed float16 weights (vision encoder only)
    metadata.json    - weight shapes + model config
"""

import json, os, sys
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from huggingface_hub import snapshot_download
from huggingface_hub.utils import disable_progress_bars
disable_progress_bars()
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_VERBOSITY"] = "error"

HF_REPO = "Qwen/Qwen3.5-0.8B"
OUT_DIR = Path(__file__).resolve().parent


def export():
    print(f"[1/3] Downloading model: {HF_REPO}")
    weights_path = snapshot_download(repo_id=HF_REPO)
    model_path = Path(weights_path)

    with open(model_path / "config.json") as f:
        hf_config = json.load(f)

    # Save full config
    with open(OUT_DIR / "hf_config.json", "w") as f:
        json.dump(hf_config, f, indent=2)

    llm_config = hf_config["text_config"]
    vis_config = hf_config.get("vision_config")

    if not vis_config:
        print("[ERROR] No vision config found!")
        return

    print(f"[2/3] Loading vision encoder weights from safetensors...")
    from safetensors import safe_open
    safetensors_files = sorted(model_path.glob("*.safetensors"))
    print(f"   Found {len(safetensors_files)} safetensors file(s)")

    # Only extract vision encoder weights (not token embeddings)
    vision_prefixes = ["model.visual."]

    state = {}
    for st_file in safetensors_files:
        print(f"   Scanning: {st_file.name}")
        with safe_open(st_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                for prefix in vision_prefixes:
                    if key.startswith(prefix):
                        state[key] = f.get_tensor(key).to(torch.float16).numpy()
                        break

    print(f"   Extracted {len(state)} vision tensors")

    # Print weight names for debugging
    for k in sorted(state.keys()):
        print(f"     {k}: {state[k].shape}")

    print(f"[3/3] Writing output files...")

    # Write weights.bin
    weights_bin = bytearray()
    weight_meta = []
    offset = 0

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

    mb = len(weights_bin) / 1024 / 1024
    print(f"   weights.bin: {len(weights_bin)} bytes ({mb:.1f} MB)")

    # Build metadata
    metadata = {
        "model_name": "Qwen3.5-0.8B-Vision",
        "model_type": "vision",
        "description": "Qwen3.5-0.8B VisionEncoder (image understanding)",
        "weight_count": len(weight_meta),
        "weight_bytes": len(weights_bin),
        "use_fp16": True,
        "weights": weight_meta,
        "vision_config": {
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
        },
        "text_config": {
            "hidden_size": llm_config["hidden_size"],
            "vocab_size": llm_config["vocab_size"],
            "head_dim": llm_config.get("head_dim",
                llm_config["hidden_size"] // llm_config["num_attention_heads"]),
            "rms_norm_eps": llm_config["rms_norm_eps"],
        },
    }

    with open(OUT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"   metadata.json")
    print(f"\n✅ Done! VisionEncoder weights exported to {OUT_DIR}")
    print(f"   Open index.html in a WebGPU-capable browser to run inference.")


if __name__ == "__main__":
    export()
