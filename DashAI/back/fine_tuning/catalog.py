from copy import deepcopy
from typing import Any

MODEL_CATALOG: dict[str, dict[str, Any]] = {
    "qwen2.5-0.5b-instruct": {
        "key": "qwen2.5-0.5b-instruct",
        "name": "Qwen2.5 0.5B Instruct",
        "repository": "Qwen/Qwen2.5-0.5B-Instruct",
        "parameter_count_billions": 0.5,
        "recommended_vram_gb": 6,
    },
    "qwen2.5-1.5b-instruct": {
        "key": "qwen2.5-1.5b-instruct",
        "name": "Qwen2.5 1.5B Instruct",
        "repository": "Qwen/Qwen2.5-1.5B-Instruct",
        "parameter_count_billions": 1.5,
        "recommended_vram_gb": 8,
    },
}

PRESET_CATALOG: dict[str, dict[str, Any]] = {
    "quick_test": {
        "name": "Quick test",
        "description": "32 examples and three optimizer steps.",
        "parameters": {
            "preset": "quick_test",
            "max_samples": 32,
            "max_steps": 3,
            "num_train_epochs": 1.0,
            "max_length": 128,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "learning_rate": 0.0002,
            "lora_r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "target_modules": "all-linear",
            "seed": 42,
            "logging_steps": 1,
            "save_steps": 1,
            "eval_steps": 1,
        },
    },
    "qlora_8gb": {
        "name": "QLoRA 8 GB",
        "description": "Conservative QLoRA preset for an 8 GB NVIDIA GPU.",
        "parameters": {
            "preset": "qlora_8gb",
            "max_samples": 1000,
            "max_steps": 100,
            "num_train_epochs": 1.0,
            "max_length": 256,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "learning_rate": 0.0002,
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "target_modules": "all-linear",
            "seed": 42,
            "logging_steps": 1,
            "save_steps": 25,
            "eval_steps": 25,
        },
    },
    "lora_small": {
        "name": "LoRA small",
        "description": "FP16 LoRA for small models when 4-bit is unavailable.",
        "parameters": {
            "preset": "lora_small",
            "max_samples": 1000,
            "max_steps": 100,
            "num_train_epochs": 1.0,
            "max_length": 256,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "learning_rate": 0.0002,
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "target_modules": "all-linear",
            "seed": 42,
            "logging_steps": 1,
            "save_steps": 25,
            "eval_steps": 25,
        },
    },
}


def resolve_model(model_id: str) -> dict[str, Any]:
    """Resolve a curated catalog key without accepting arbitrary repositories."""
    try:
        return deepcopy(MODEL_CATALOG[model_id])
    except KeyError as exc:
        raise ValueError(f"Unsupported base model: {model_id}") from exc


def get_catalog() -> dict[str, Any]:
    return {
        "models": deepcopy(list(MODEL_CATALOG.values())),
        "presets": deepcopy(PRESET_CATALOG),
        "capabilities": {
            "methods": ["lora", "qlora"],
            "dataset_formats": ["text", "prompt_completion", "messages"],
            "distributed_training": False,
            "gguf_export": False,
            "unsloth": False,
        },
    }
