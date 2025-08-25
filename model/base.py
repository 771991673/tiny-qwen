import os
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union, Dict, Any
import torch
import torch.nn as nn
from huggingface_hub import snapshot_download, HfApi, logging
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from safetensors.torch import save_file

logging.set_verbosity_error()


class PretrainedModelMixin(ABC):
    """
    Mixin class for all pretrained models with HuggingFace-style loading/saving capabilities.
    
    Each model should implement:
    - get_config_class(): Return the config class for this model
    - convert_hf_config(): Convert HF config dict to our format (optional)
    - get_loading_strategy(): Return loading strategy (optional)
    """

    @classmethod
    @abstractmethod
    def get_config_class(cls):
        """Return the config class for this model."""
        pass

    @classmethod
    def convert_hf_config(cls, hf_config: dict) -> dict:
        """
        Convert HuggingFace config format to our format.
        Override this method for model-specific config conversion.
        """
        return cls._convert_llm_config(hf_config)

    @classmethod
    def get_loading_strategy(cls) -> dict:
        """
        Return loading strategy for this model.
        Override for model-specific loading (e.g., MoE, vision models).
        """
        return {
            "use_empty_weights": True,
            "dtype": torch.bfloat16,
            "no_split_module_classes": ["Block"]
        }

    @classmethod
    def get_weight_map(cls) -> Optional[Dict[str, str]]:
        """
        Return weight key mapping from HF format to our format.
        Override for models with different weight naming.
        """
        return None

    @classmethod
    def from_pretrained(
        cls,
        repo_id: Union[str, Path],
        device_map: Optional[Union[str, Dict[str, Union[int, str]]]] = "auto",
        cache_dir: Optional[str] = None,
        local_files_only: bool = False,
        force_download: bool = False,
        **kwargs,
    ):
        """Load a pretrained model from HuggingFace Hub or local directory."""
        # Determine model path
        if os.path.isdir(repo_id):
            model_path = Path(repo_id)
        else:
            model_path = Path(
                snapshot_download(
                    repo_id=repo_id,
                    cache_dir=cache_dir,
                    local_files_only=local_files_only,
                    force_download=force_download,
                )
            )

        # Load and convert config
        with open(model_path / "config.json", "r") as f:
            config_data = json.load(f)

        config_cls = cls.get_config_class()
        converted_config = cls.convert_hf_config(config_data)
        converted_config = cls._filter_dict_by_dataclass(converted_config, config_cls)
        model_config = config_cls(**converted_config)

        # Get loading strategy
        loading_strategy = cls.get_loading_strategy()
        use_empty_weights = loading_strategy.get("use_empty_weights", True)
        dtype = loading_strategy.get("dtype", torch.bfloat16)
        no_split_modules = loading_strategy.get("no_split_module_classes", ["Block"])

        # Create and load model
        if use_empty_weights:
            try:
                with init_empty_weights():
                    model = cls(model_config)
            except Exception:
                # Fallback for models that don't work with empty weights
                model = cls(model_config)
        else:
            model = cls(model_config)

        # Load weights with fallback for problematic models
        try:
            model = load_checkpoint_and_dispatch(
                model,
                model_path,
                device_map=device_map,
                dtype=dtype,
                no_split_module_classes=no_split_modules,
            )
        except (NotImplementedError, AttributeError) as e:
            if "Cannot copy out of meta tensor" in str(e) or "get_output_embeddings" in str(e):
                # Fallback: try without empty weights
                print("Meta tensor/accelerate issue detected, trying alternative loading...")
                model = cls(model_config)
                model = load_checkpoint_and_dispatch(
                    model,
                    model_path,
                    device_map=device_map,
                    dtype=dtype,
                    no_split_module_classes=no_split_modules,
                )
            else:
                raise

        # Apply weight mapping if needed
        weight_map = cls.get_weight_map()
        if weight_map:
            # TODO: Implement weight key remapping during loading
            pass

        return model

    def save_pretrained(
        self,
        save_directory: Union[str, Path],
        push_to_hub: bool = False,
        repo_id: Optional[str] = None,
        token: Optional[str] = None,
        max_shard_size: str = "10GB",
        **kwargs,
    ):
        """Save the model and its configuration to a directory or HuggingFace Hub."""
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        # Save model weights as safetensors
        self._save_safetensors(save_directory, max_shard_size)

        # Save config
        if self.config is not None:
            config_dict = self._config_to_dict(self.config)
            with open(save_directory / "config.json", "w") as f:
                json.dump(config_dict, f, indent=2)

        # Push to hub if requested
        if push_to_hub and repo_id is not None:
            api = HfApi(token=token)
            api.upload_folder(
                folder_path=str(save_directory),
                repo_id=repo_id,
                repo_type="model",
                token=token,
            )

    def _save_safetensors(self, save_directory: Path, max_shard_size: str = "10GB"):
        """Save model weights as safetensors files."""
        state_dict = self.state_dict()
        
        # Convert max_shard_size to bytes
        if max_shard_size.endswith("GB"):
            max_size_bytes = int(float(max_shard_size[:-2]) * 1024 * 1024 * 1024)
        elif max_shard_size.endswith("MB"):
            max_size_bytes = int(float(max_shard_size[:-2]) * 1024 * 1024)
        else:
            max_size_bytes = int(max_shard_size)

        # Group tensors by size
        current_size = 0
        file_index = 1
        tensors = {}
        
        for key, tensor in state_dict.items():
            tensor_size = tensor.numel() * tensor.element_size()
            
            if current_size + tensor_size > max_size_bytes and tensors:
                # Save current shard
                save_path = save_directory / f"model-{file_index:05d}-of-XXXXX.safetensors"
                save_file(tensors, save_path)
                tensors = {}
                current_size = 0
                file_index += 1
            
            tensors[key] = tensor
            current_size += tensor_size

        # Save final shard
        if tensors:
            save_path = save_directory / f"model-{file_index:05d}-of-XXXXX.safetensors"
            save_file(tensors, save_path)

        # Rename files with correct total count
        total_files = file_index
        for idx in range(1, total_files + 1):
            old_path = save_directory / f"model-{idx:05d}-of-XXXXX.safetensors"
            new_path = save_directory / f"model-{idx:05d}-of-{total_files:05d}.safetensors"
            old_path.rename(new_path)

    def _config_to_dict(self, config) -> dict:
        """Convert config object to dictionary for JSON serialization."""
        if hasattr(config, '__dict__'):
            config_dict = config.__dict__.copy()
            # Handle nested configs (like vision_config)
            for key, value in config_dict.items():
                if hasattr(value, '__dict__'):
                    config_dict[key] = value.__dict__
            return config_dict
        return {}

    @staticmethod
    def _convert_llm_config(hf_config: dict) -> dict:
        """Convert HuggingFace LLM config to our format."""
        llm_config_key_mapping = {
            "hidden_size": "n_embed",
            "num_attention_heads": "n_heads",
            "num_key_value_heads": "n_kv_heads",
            "num_hidden_layers": "n_layer",
            "intermediate_size": "n_mlp",
            "rms_norm_eps": "rms_norm_eps",
            "vocab_size": "vocab_size",
            "rope_theta": "rope_theta",
            "tie_word_embeddings": "tie_word_embeddings",
            "head_dim": "head_dim",
            # MoE parameters
            "num_experts": "num_experts",
            "num_experts_per_tok": "num_experts_per_tok",
            "moe_intermediate_size": "moe_intermediate_size",
        }
        return PretrainedModelMixin._rename_dict_keys(hf_config, llm_config_key_mapping)

    @staticmethod  
    def _rename_dict_keys(original_dict: dict, key_mapping: dict) -> dict:
        """Rename keys in a dictionary according to a provided mapping."""
        new_dict = {}
        for key, value in original_dict.items():
            new_key = key_mapping.get(key, key)
            new_dict[new_key] = value
        return new_dict

    @staticmethod
    def _filter_dict_by_dataclass(params: dict, dataclass_type) -> dict:
        """Filter dictionary to only include keys that exist in the dataclass."""
        return {k: v for k, v in params.items() if k in dataclass_type.__annotations__}