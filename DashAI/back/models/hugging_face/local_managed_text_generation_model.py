from pathlib import Path
from typing import List, Optional

from DashAI.back.core.schema_fields import (
    BaseSchema,
    enum_field,
    float_field,
    int_field,
    none_type,
    schema_field,
)
from DashAI.back.core.utils import MultilingualString
from DashAI.back.models.text_to_text_generation_model import (
    TextToTextGenerationTaskModel,
)


class LocalManagedTextGenerationSchema(BaseSchema):
    max_new_tokens: schema_field(
        int_field(ge=1, le=4096),
        placeholder=128,
        description=MultilingualString(
            en="Maximum number of tokens generated for each response.",
            es="Número máximo de tokens generados por respuesta.",
        ),
        alias=MultilingualString(en="Max new tokens", es="Tokens nuevos máximos"),
    )  # type: ignore
    temperature: schema_field(
        float_field(ge=0.0, le=2.0),
        placeholder=0.7,
        description=MultilingualString(
            en="Sampling randomness; zero uses deterministic decoding.",
            es="Aleatoriedad del muestreo; cero usa decodificación determinista.",
        ),
        alias=MultilingualString(en="Temperature", es="Temperatura"),
    )  # type: ignore
    top_p: schema_field(
        float_field(gt=0.0, le=1.0),
        placeholder=0.9,
        description=MultilingualString(
            en="Cumulative probability used for nucleus sampling.",
            es="Probabilidad acumulada usada por el muestreo nucleus.",
        ),
        alias=MultilingualString(en="Top p", es="Top p"),
    )  # type: ignore
    top_k: schema_field(
        int_field(ge=0, le=1000),
        placeholder=50,
        description=MultilingualString(
            en="Number of highest-probability tokens kept when sampling; "
            "zero disables the filter.",
            es="Cantidad de tokens de mayor probabilidad conservados al "
            "muestrear; cero desactiva el filtro.",
        ),
        alias=MultilingualString(en="Top k", es="Top k"),
        default=50,
    )  # type: ignore
    repetition_penalty: schema_field(
        float_field(ge=0.1, le=3.0),
        placeholder=1.05,
        description=MultilingualString(
            en="Penalty applied to repeated tokens.",
            es="Penalización aplicada a tokens repetidos.",
        ),
        alias=MultilingualString(
            en="Repetition penalty", es="Penalización de repetición"
        ),
    )  # type: ignore
    seed: schema_field(
        none_type(int_field(ge=0, le=2**31 - 1)),
        placeholder=42,
        description=MultilingualString(
            en="Seed for reproducible generation; ignored when empty.",
            es="Semilla para generación reproducible; se ignora si está vacía.",
        ),
        alias=MultilingualString(en="Seed", es="Semilla"),
        default=None,
    )  # type: ignore
    context_window: schema_field(
        int_field(ge=64, le=32768),
        placeholder=2048,
        description=MultilingualString(
            en="Maximum prompt context retained for generation.",
            es="Contexto máximo del prompt conservado para generar.",
        ),
        alias=MultilingualString(en="Context window", es="Ventana de contexto"),
    )  # type: ignore
    device: schema_field(
        enum_field(["auto", "cuda", "cpu"]),
        placeholder="auto",
        description=MultilingualString(
            en="Hardware used to load the model; CPU is slower but always available.",
            es="Hardware usado para cargar el modelo; CPU es más lento pero "
            "siempre está disponible.",
        ),
        alias=MultilingualString(en="Device", es="Dispositivo"),
    )  # type: ignore


class LocalManagedTextGenerationModel(TextToTextGenerationTaskModel):
    """Base-model inference backed by DashAI-managed local storage.

    Internal component: sessions reference a `ManagedLocalModel` inventory
    id, and the job layer injects the resolved storage path. It never
    accepts arbitrary paths from clients.
    """

    SCHEMA = LocalManagedTextGenerationSchema
    COMPATIBLE_COMPONENTS = ["TextToTextGenerationTask"]
    DISPLAY_NAME = MultilingualString(
        en="Managed local model", es="Modelo local administrado"
    )
    DESCRIPTION = MultilingualString(
        en="Run a catalog base model downloaded by DashAI Fine-tuning.",
        es="Ejecuta un modelo base del catálogo descargado por Fine-tuning de DashAI.",
    )
    COLOR = "#3f7f5f"

    def __init__(self, **kwargs):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        base_model_path = Path(kwargs.pop("_base_model_path"))
        values = self.validate_and_transform(kwargs)
        self.max_new_tokens = values.get("max_new_tokens", 128)
        self.temperature = values.get("temperature", 0.7)
        self.top_p = values.get("top_p", 0.9)
        self.top_k = values.get("top_k", 50)
        self.repetition_penalty = values.get("repetition_penalty", 1.05)
        self.seed: Optional[int] = values.get("seed")
        self.context_window = values.get("context_window", 2048)
        requested_device = values.get("device", "auto")
        use_cuda = torch.cuda.is_available() and requested_device != "cpu"
        if requested_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")

        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path, local_files_only=True, trust_remote_code=False
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        model_kwargs = {
            "local_files_only": True,
            "trust_remote_code": False,
            "attn_implementation": "eager",
        }
        if use_cuda:
            model_kwargs.update({"device_map": {"": 0}, "dtype": torch.float16})
        else:
            model_kwargs["dtype"] = torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_path, **model_kwargs
        )
        self.model.eval()

    def generate(self, prompt: list) -> List[str]:
        import torch

        if self.seed is not None:
            torch.manual_seed(self.seed)
        inputs = self.tokenizer.apply_chat_template(
            prompt,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        inputs = inputs[:, -self.context_window :].to(self.model.device)
        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
            "repetition_penalty": self.repetition_penalty,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if self.temperature > 0:
            generation_kwargs["temperature"] = self.temperature
            generation_kwargs["top_p"] = self.top_p
            if self.top_k and self.top_k > 0:
                generation_kwargs["top_k"] = self.top_k
        with torch.inference_mode():
            output = self.model.generate(inputs, **generation_kwargs)
        generated = output[0, inputs.shape[-1] :]
        return [self.tokenizer.decode(generated, skip_special_tokens=True).strip()]
