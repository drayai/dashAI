"""Optional Unsloth Core execution backend.

Unsloth is an execution backend for LoRA/QLoRA (not an alternative method):
it accelerates the same adapter training produced by the Transformers
backend. All heavy imports are deferred to ``train`` so DashAI starts
normally without Unsloth installed.

Availability is detected without importing Unsloth (``find_spec`` plus
package metadata): importing it patches global state, which must only
happen inside the worker that actually trains.
"""

import importlib.metadata
import importlib.util
import json
import platform
import shutil
import time
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

NOT_INSTALLED_MESSAGE = (
    "Unsloth Core is not installed. Install it in the DashAI environment "
    "(see the finetuning-unsloth notes in docs/fine_tuning_prototype_findings.md) "
    "and make sure the installed versions do not replace the pinned "
    "fine-tuning dependencies."
)


def unsloth_installed() -> tuple[bool, str | None]:
    """Return (installed, version) without importing Unsloth."""
    if importlib.util.find_spec("unsloth") is None:
        return False, None
    try:
        return True, importlib.metadata.version("unsloth")
    except importlib.metadata.PackageNotFoundError:
        return True, None


def unsloth_capabilities() -> dict[str, Any]:
    """Report Unsloth installation status without importing it.

    Hardware suitability is checked in preflight, which already inspects
    the GPU; the catalog endpoint must stay free of heavy imports.
    """
    installed, version = unsloth_installed()
    return {
        "available": installed,
        "version": version,
        "reason": None if installed else NOT_INSTALLED_MESSAGE,
    }


class UnslothFineTuningBackend(FineTuningBackend):
    """LoRA/QLoRA training accelerated by Unsloth Core."""

    def train(
        self,
        request: FineTuningRequest,
        progress: ProgressCallback,
        is_canceled: CancellationCallback,
    ) -> FineTuningResult:
        installed, _version = unsloth_installed()
        if not installed:
            raise RuntimeError(NOT_INSTALLED_MESSAGE)
        if is_canceled():
            raise TrainingCanceledError

        # Deferred imports keep this module importable at startup and avoid
        # circular imports with the Transformers backend and preflight.
        # Deferred on purpose: importing Unsloth patches global torch state.
        from unsloth import FastLanguageModel
        from unsloth import is_bfloat16_supported as unsloth_bf16_supported

        from DashAI.back.fine_tuning.huggingface_backend import (
            _json_safe,
            _non_finite_warning,
            _sft_config_kwargs,
            _training_text_row,
        )
        from DashAI.back.fine_tuning.model_store import (
            directory_size,
            ensure_managed_path,
        )
        from DashAI.back.fine_tuning.preflight import (
            dependency_versions,
            hardware_info,
        )

        started = time.perf_counter()
        progress(0.08, "Loading tokenizer and model with Unsloth", {})
        load_in_4bit = request.method == FineTuningMethod.QLORA
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(request.model_path),
            max_seq_length=request.parameters.max_length,
            dtype=None,
            load_in_4bit=load_in_4bit,
            local_files_only=True,
            trust_remote_code=False,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = FastLanguageModel.get_peft_model(
            model,
            r=request.parameters.lora_r,
            lora_alpha=request.parameters.lora_alpha,
            lora_dropout=request.parameters.lora_dropout,
            target_modules=request.parameters.target_modules,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=request.parameters.seed,
        )

        from datasets import Dataset as HFDataset
        from transformers import TrainerCallback
        from trl import SFTConfig, SFTTrainer

        train_dataset = HFDataset.from_list(
            [
                {"text": _training_text_row(example, tokenizer)["text"]}
                for example in request.dataset.train
            ]
        )
        validation_dataset = None
        if request.dataset.validation is not None:
            validation_dataset = HFDataset.from_list(
                [
                    {"text": _training_text_row(example, tokenizer)["text"]}
                    for example in request.dataset.validation
                ]
            )

        class HealthCallback(TrainerCallback):
            def __init__(self):
                self.logs: dict[str, Any] = {}
                self.health_warnings: list[dict[str, Any]] = []

            def on_log(self, args, state, control, logs=None, **kwargs):
                if not logs:
                    return control
                self.logs.update(_json_safe(logs))
                for field in ("loss", "grad_norm"):
                    warning = _non_finite_warning(
                        field, logs.get(field), getattr(state, "global_step", None)
                    )
                    if warning is None:
                        continue
                    if field == "loss":
                        raise TrainingHealthError(warning["message"])
                    self.health_warnings.append(warning)
                return control

        callback = HealthCallback()
        if request.output_path.exists():
            raise FileExistsError(f"Artifact already exists: {request.output_path}")
        temporary = request.output_path.with_name(f"{request.output_path.name}.tmp")
        ensure_managed_path(temporary, request.output_path.parent)
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir(parents=True)

        config_kwargs = _sft_config_kwargs(
            request.parameters,
            temporary / "trainer",
            validation_dataset is not None,
            unsloth_bf16_supported(),
        )
        trainer = SFTTrainer(
            model=model,
            args=SFTConfig(**config_kwargs),
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            processing_class=tokenizer,
            callbacks=[callback],
        )
        progress(0.15, "Starting Unsloth training", {})
        try:
            train_output = trainer.train()
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        if is_canceled():
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
        runtime_metadata = {
            "backend": "unsloth",
            "duration_seconds": round(time.perf_counter() - started, 3),
            "hardware": hardware_info(),
            "versions": dependency_versions(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "health": {"non_finite_warnings": callback.health_warnings},
        }
        manifest = {
            "schema_version": 1,
            "run_id": request.run_id,
            "backend": "unsloth",
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
