import gc
import inspect
import json
import math
import platform
import shutil
import time
from functools import partial
from pathlib import Path
from typing import Any

from DashAI.back.api.api_v1.schemas.fine_tuning_params import FineTuningMethod
from DashAI.back.fine_tuning.base import (
    CancellationCallback,
    FineTuningBackend,
    FineTuningRequest,
    FineTuningResult,
    ProgressCallback,
    TrainingCanceledError,
    TrainingHealthError,
)
from DashAI.back.fine_tuning.model_store import directory_size, ensure_managed_path
from DashAI.back.fine_tuning.preflight import dependency_versions, hardware_info


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (bool, str, int)) or value is None:
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    # NaN and infinities are not valid JSON and hide numeric failures.
    return number if math.isfinite(number) else None


def _non_finite_warning(field: str, value: Any, step: Any) -> dict[str, Any] | None:
    """Return a structured warning when a logged training value is not finite."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isfinite(number):
        return None
    return {
        "step": step,
        "field": field,
        "value": str(value),
        "message": (
            f"{field} is not finite (NaN or infinite) at step {step}. "
            "The training is not numerically stable on this "
            "hardware/configuration."
        ),
    }


def _training_text(example: dict[str, Any], tokenizer) -> str:
    if "text" in example:
        return example["text"]
    if "messages" in example:
        return tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
    messages = [
        {"role": "user", "content": example["prompt"]},
        {"role": "assistant", "content": example["completion"]},
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    return f"{example['prompt']}\n{example['completion']}"


def _training_text_row(example: dict[str, Any], tokenizer) -> dict[str, str]:
    return {"text": _training_text(example, tokenizer)}


def _sft_config_kwargs(
    parameters, output_path: Path, has_validation: bool, use_fp16: bool
) -> dict:
    from trl import SFTConfig

    signature = inspect.signature(SFTConfig).parameters
    is_quick_test = parameters.preset == "quick_test"
    values = {
        "output_dir": str(output_path),
        "max_steps": parameters.max_steps,
        "num_train_epochs": parameters.num_train_epochs,
        "per_device_train_batch_size": parameters.per_device_train_batch_size,
        "gradient_accumulation_steps": parameters.gradient_accumulation_steps,
        "learning_rate": parameters.learning_rate,
        "logging_steps": parameters.logging_steps,
        "save_steps": parameters.save_steps,
        "eval_steps": parameters.eval_steps,
        "seed": parameters.seed,
        "fp16": use_fp16,
        "bf16": False,
        "optim": "adamw_torch",
        "gradient_checkpointing": not is_quick_test,
        "report_to": "none",
        "skip_memory_metrics": False,
        "dataset_text_field": "text",
        "max_length": parameters.max_length,
        "packing": False,
        "save_strategy": "no" if is_quick_test else "steps",
        "eval_strategy": ("steps" if has_validation and not is_quick_test else "no"),
    }
    if "max_length" not in signature and "max_seq_length" in signature:
        values["max_seq_length"] = values.pop("max_length")
    if "eval_strategy" not in signature and "evaluation_strategy" in signature:
        values["evaluation_strategy"] = values.pop("eval_strategy")
    return {key: value for key, value in values.items() if key in signature}


class HuggingFaceFineTuningBackend(FineTuningBackend):
    """Transformers/TRL implementation for LoRA and 4-bit QLoRA."""

    def train(
        self,
        request: FineTuningRequest,
        progress: ProgressCallback,
        is_canceled: CancellationCallback,
    ) -> FineTuningResult:
        import torch
        from peft import LoraConfig, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            TrainerCallback,
        )
        from trl import SFTConfig, SFTTrainer

        if is_canceled():
            raise TrainingCanceledError

        try:
            return self._train(
                request,
                progress,
                is_canceled,
                torch=torch,
                lora_config_cls=LoraConfig,
                prepare_model_for_kbit_training=prepare_model_for_kbit_training,
                auto_model_cls=AutoModelForCausalLM,
                auto_tokenizer_cls=AutoTokenizer,
                bitsandbytes_config_cls=BitsAndBytesConfig,
                trainer_callback_cls=TrainerCallback,
                sft_config_cls=SFTConfig,
                sft_trainer_cls=SFTTrainer,
            )
        finally:
            # _train's finally already dropped its heavy references from the
            # (possibly traceback-held) frame, so the caches can be emptied
            # now even when training failed or was canceled.
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _train(
        self,
        request: FineTuningRequest,
        progress: ProgressCallback,
        is_canceled: CancellationCallback,
        torch,
        lora_config_cls,
        prepare_model_for_kbit_training,
        auto_model_cls,
        auto_tokenizer_cls,
        bitsandbytes_config_cls,
        trainer_callback_cls,
        sft_config_cls,
        sft_trainer_cls,
    ) -> FineTuningResult:
        started = time.perf_counter()
        tokenizer = None
        model = None
        trainer = None
        train_dataset = None
        validation_dataset = None
        try:
            progress(0.08, "Loading tokenizer", {})
            tokenizer = auto_tokenizer_cls.from_pretrained(
                request.model_path, local_files_only=True, trust_remote_code=False
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            train_dataset = request.dataset.train.map(
                partial(_training_text_row, tokenizer=tokenizer),
                remove_columns=request.dataset.train.column_names,
            )
            if request.dataset.validation is not None:
                validation_dataset = request.dataset.validation.map(
                    partial(_training_text_row, tokenizer=tokenizer),
                    remove_columns=request.dataset.validation.column_names,
                )

            model_kwargs: dict[str, Any] = {
                "local_files_only": True,
                "trust_remote_code": False,
                "attn_implementation": "eager",
            }
            if request.method == FineTuningMethod.QLORA:
                model_kwargs.update(
                    {
                        "quantization_config": bitsandbytes_config_cls(
                            load_in_4bit=True,
                            bnb_4bit_quant_type="nf4",
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_compute_dtype=torch.float16,
                        ),
                        "device_map": {"": 0},
                        "dtype": torch.float16,
                    }
                )
            elif torch.cuda.is_available():
                model_kwargs.update({"device_map": {"": 0}, "dtype": torch.float16})
            else:
                model_kwargs["dtype"] = torch.float32

            progress(0.12, "Loading base model", {})
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            model = auto_model_cls.from_pretrained(request.model_path, **model_kwargs)
            model.config.use_cache = False
            if request.method == FineTuningMethod.QLORA:
                model = prepare_model_for_kbit_training(
                    model,
                    use_gradient_checkpointing=(
                        request.parameters.preset != "quick_test"
                    ),
                )

            peft_config = lora_config_cls(
                r=request.parameters.lora_r,
                lora_alpha=request.parameters.lora_alpha,
                lora_dropout=request.parameters.lora_dropout,
                target_modules=request.parameters.target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            )

            class ProgressAndCancellationCallback(trainer_callback_cls):
                """Tracks progress, numeric health and cooperative cancel."""

                def __init__(self):
                    self.canceled = False
                    self.logs: dict[str, Any] = {}
                    self.health_warnings: list[dict[str, Any]] = []
                    self.step_durations: list[float] = []
                    self._step_started: float | None = None

                def on_step_begin(self, args, state, control, **kwargs):
                    self._step_started = time.perf_counter()
                    return control

                def on_log(self, args, state, control, logs=None, **kwargs):
                    if logs:
                        self.logs.update(_json_safe(logs))
                        for field in ("loss", "grad_norm"):
                            warning = _non_finite_warning(
                                field,
                                logs.get(field),
                                getattr(state, "global_step", None),
                            )
                            if warning is None:
                                continue
                            if field == "loss":
                                # A non-finite loss invalidates the adapter:
                                # stop now so it is never published as a
                                # valid artifact.
                                raise TrainingHealthError(warning["message"])
                            last = (
                                self.health_warnings[-1]
                                if self.health_warnings
                                else None
                            )
                            if (
                                last is None
                                or last["field"] != field
                                or last["step"] != warning["step"]
                            ):
                                self.health_warnings.append(warning)
                    return control

                def on_step_end(self, args, state, control, **kwargs):
                    if is_canceled():
                        self.canceled = True
                        control.should_training_stop = True
                    if self._step_started is not None:
                        self.step_durations.append(
                            round(time.perf_counter() - self._step_started, 3)
                        )
                        self._step_started = None
                    fraction = state.global_step / max(1, state.max_steps)
                    payload: dict[str, Any] = {"step": state.global_step, **self.logs}
                    if self.health_warnings:
                        payload["health_warnings"] = self.health_warnings
                    progress(
                        0.15 + fraction * 0.75,
                        f"Training step {state.global_step}/{state.max_steps}",
                        payload,
                    )
                    return control

            callback = ProgressAndCancellationCallback()
            if request.output_path.exists():
                raise FileExistsError(f"Artifact already exists: {request.output_path}")
            temporary = request.output_path.with_name(f"{request.output_path.name}.tmp")
            ensure_managed_path(temporary, request.output_path.parent)
            if temporary.exists():
                shutil.rmtree(temporary)
            temporary.mkdir(parents=True)
            trainer_output = temporary / "trainer"
            config_kwargs = _sft_config_kwargs(
                request.parameters,
                trainer_output,
                validation_dataset is not None,
                torch.cuda.is_available(),
            )
            trainer_kwargs = {
                "model": model,
                "args": sft_config_cls(**config_kwargs),
                "train_dataset": train_dataset,
                "eval_dataset": validation_dataset,
                "peft_config": peft_config,
                "callbacks": [callback],
            }
            trainer_signature = inspect.signature(sft_trainer_cls).parameters
            if "processing_class" in trainer_signature:
                trainer_kwargs["processing_class"] = tokenizer
            else:
                trainer_kwargs["tokenizer"] = tokenizer

            trainer = sft_trainer_cls(**trainer_kwargs)
            trainable_dtypes: dict[str, int] = {}
            for parameter in trainer.model.parameters():
                if not parameter.requires_grad:
                    continue
                if parameter.dtype != torch.float32:
                    parameter.data = parameter.data.to(torch.float32)
                dtype_name = str(parameter.dtype).removeprefix("torch.")
                trainable_dtypes[dtype_name] = trainable_dtypes.get(dtype_name, 0) + 1
            progress(0.15, "Starting optimizer", {})
            try:
                train_output = trainer.train()
            except Exception:
                shutil.rmtree(temporary, ignore_errors=True)
                raise
            if callback.canceled or is_canceled():
                shutil.rmtree(temporary)
                raise TrainingCanceledError

            progress(0.92, "Saving adapter and manifest", {})
            adapter_path = temporary / "adapter"
            trainer.model.save_pretrained(adapter_path)
            tokenizer.save_pretrained(adapter_path)
            trainer.state.save_to_json(str(temporary / "trainer_state.json"))
            chat_template = getattr(tokenizer, "chat_template", None)
            if chat_template:
                (temporary / "chat_template.txt").write_text(
                    chat_template, encoding="utf-8"
                )

            metrics = _json_safe(train_output.metrics)
            metrics["health_warnings"] = callback.health_warnings
            metrics["last_logged"] = callback.logs
            peak_vram = 0
            if torch.cuda.is_available():
                peak_vram = torch.cuda.max_memory_allocated()
            step_durations = callback.step_durations
            runtime_metadata = {
                "duration_seconds": round(time.perf_counter() - started, 3),
                "peak_vram_bytes": peak_vram,
                "hardware": hardware_info(),
                "versions": dependency_versions(),
                "python": platform.python_version(),
                "platform": platform.platform(),
                "trainable_parameter_dtypes": trainable_dtypes,
                "health": {
                    "step_durations_seconds": step_durations,
                    "mean_step_seconds": (
                        round(sum(step_durations) / len(step_durations), 3)
                        if step_durations
                        else None
                    ),
                    "non_finite_warnings": callback.health_warnings,
                },
            }
            manifest = {
                "schema_version": 1,
                "run_id": request.run_id,
                "base_model": {
                    "id": request.model_id,
                    "requested_revision": request.model_revision,
                    "resolved_revision": request.resolved_revision,
                    "local_path": str(request.model_path),
                },
                "dataset": {
                    "fingerprint": request.dataset.fingerprint,
                    "source_rows": request.dataset.source_rows,
                    "train_rows": len(request.dataset.train),
                    "validation_rows": (
                        len(request.dataset.validation)
                        if request.dataset.validation is not None
                        else 0
                    ),
                    "mapping": request.mapping.model_dump(mode="json"),
                },
                "method": request.method.value,
                "parameters": request.parameters.model_dump(mode="json"),
                "seed": request.parameters.seed,
                "metrics": metrics,
                "runtime": runtime_metadata,
                "artifact_size_bytes": directory_size(adapter_path),
            }
            (temporary / "manifest.json").write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            temporary.replace(request.output_path)
            progress(1.0, "Training completed", metrics)
            return FineTuningResult(
                artifact_path=request.output_path,
                metrics=metrics,
                runtime_metadata=runtime_metadata,
            )
        finally:
            del trainer, model, tokenizer, train_dataset, validation_dataset
