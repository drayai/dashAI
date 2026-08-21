import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, HTTPException, status
from kink import di
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from DashAI.back.api.api_v1.schemas.fine_tuning_params import (
    FineTuningRunCreate,
    FineTuningRunRead,
    LocalModelInfo,
    PreflightReport,
    PreflightRequest,
)
from DashAI.back.core.enums.status import DatafileStatus, FineTuningStatus
from DashAI.back.dependencies.database.models import (
    Dataset,
    FineTuningRun,
    GenerativeSession,
    ManagedLocalModel,
)
from DashAI.back.fine_tuning.catalog import MODEL_CATALOG, get_catalog, resolve_model
from DashAI.back.fine_tuning.model_store import (
    directory_size,
    ensure_managed_path,
    read_model_metadata,
)
from DashAI.back.fine_tuning.preflight import run_preflight
from DashAI.back.fine_tuning.resource_lock import training_lock
from DashAI.back.job.fine_tuning_job import FineTuningJob
from DashAI.back.job.managed_model_job import ManagedModelDownloadJob

if TYPE_CHECKING:
    from sqlalchemy.orm import sessionmaker

    from DashAI.back.dependencies.job_queues.base_job_queue import BaseJobQueue

router = APIRouter()
ACTIVE_STATUSES = {FineTuningStatus.QUEUED, FineTuningStatus.RUNNING}


def _config():
    return di["config"]


@router.get("/catalog")
async def catalog():
    return get_catalog()


@router.post("/preflight", response_model=PreflightReport)
async def preflight(
    request: PreflightRequest,
    session_factory: "sessionmaker" = Depends(lambda: di["session_factory"]),
):
    with session_factory() as db:
        dataset = db.get(Dataset, request.dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset does not exist.")
        return run_preflight(request, dataset, Path(_config()["LLM_MODELS_PATH"]))


@router.post(
    "/runs", response_model=FineTuningRunRead, status_code=status.HTTP_201_CREATED
)
async def create_run(
    params: FineTuningRunCreate,
    session_factory: "sessionmaker" = Depends(lambda: di["session_factory"]),
):
    try:
        resolve_model(params.base_model_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    with session_factory() as db:
        if not db.get(Dataset, params.dataset_id):
            raise HTTPException(status_code=404, detail="Dataset does not exist.")
        run = FineTuningRun(
            name=params.name,
            dataset_id=params.dataset_id,
            base_model_id=params.base_model_id,
            base_model_revision=params.base_model_revision,
            method=params.method.value,
            dataset_mapping=params.dataset_mapping.model_dump(mode="json"),
            training_parameters=params.training_parameters.model_dump(mode="json"),
        )
        db.add(run)
        try:
            db.commit()
        except IntegrityError as exc:
            db.rollback()
            raise HTTPException(
                status_code=409,
                detail=f"Fine-tuning run '{params.name}' already exists.",
            ) from exc
        db.refresh(run)
        return run


@router.get("/runs", response_model=list[FineTuningRunRead])
async def list_runs(
    session_factory: "sessionmaker" = Depends(lambda: di["session_factory"]),
):
    with session_factory() as db:
        return list(
            db.scalars(select(FineTuningRun).order_by(FineTuningRun.created.desc()))
        )


@router.get("/runs/{run_id}", response_model=FineTuningRunRead)
async def get_run(
    run_id: int,
    session_factory: "sessionmaker" = Depends(lambda: di["session_factory"]),
):
    with session_factory() as db:
        run = db.get(FineTuningRun, run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Fine-tuning run not found.")
        return run


@router.post("/runs/{run_id}/start", response_model=FineTuningRunRead)
async def start_run(
    run_id: int,
    session_factory: "sessionmaker" = Depends(lambda: di["session_factory"]),
    job_queue: "BaseJobQueue" = Depends(lambda: di["job_queue"]),
):
    with session_factory() as db:
        run = db.get(FineTuningRun, run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Fine-tuning run not found.")
        if run.status in ACTIVE_STATUSES or run.status == FineTuningStatus.COMPLETED:
            raise HTTPException(
                status_code=409, detail=f"Run cannot start from '{run.status.value}'."
            )
        # Advisory check only: the worker re-checks under the lock to close
        # the race between concurrent start requests.
        lock = training_lock(_config())
        if lock.is_locked():
            raise HTTPException(status_code=409, detail=lock.busy_message())
        run.progress = 0.0
        run.artifact_path = None
        run.metrics = None
        run.runtime_metadata = None
        run.mark_queued()
        db.commit()

    job = FineTuningJob(run_id=run_id)
    try:
        queued = job_queue.put(job)
        huey_id = str(queued.id)
    except Exception as exc:
        with session_factory() as db:
            run = db.get(FineTuningRun, run_id)
            run.mark_failed(f"Could not enqueue job: {exc}")
            db.commit()
        raise HTTPException(status_code=500, detail="Could not enqueue job.") from exc

    with session_factory() as db:
        run = db.get(FineTuningRun, run_id)
        run.huey_id = huey_id
        db.commit()
        db.refresh(run)
        return run


@router.post("/runs/{run_id}/cancel", response_model=FineTuningRunRead)
async def cancel_run(
    run_id: int,
    session_factory: "sessionmaker" = Depends(lambda: di["session_factory"]),
    job_queue: "BaseJobQueue" = Depends(lambda: di["job_queue"]),
):
    with session_factory() as db:
        run = db.get(FineTuningRun, run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Fine-tuning run not found.")
        if run.status not in ACTIVE_STATUSES:
            raise HTTPException(status_code=409, detail="Run is not active.")
        run.cancellation_requested = True
        huey_id = run.huey_id
        queued = run.status == FineTuningStatus.QUEUED
        db.commit()

    if queued and huey_id:
        job_queue.delete_from_db(huey_id)
        with session_factory() as db:
            run = db.get(FineTuningRun, run_id)
            run.mark_canceled()
            db.commit()
            db.refresh(run)
            return run
    with session_factory() as db:
        run = db.get(FineTuningRun, run_id)
        db.refresh(run)
        return run


@router.delete("/runs/{run_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_run(
    run_id: int,
    session_factory: "sessionmaker" = Depends(lambda: di["session_factory"]),
):
    artifact = None
    with session_factory() as db:
        run = db.get(FineTuningRun, run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Fine-tuning run not found.")
        if run.status in ACTIVE_STATUSES:
            raise HTTPException(
                status_code=409, detail="Active runs cannot be deleted."
            )
        referenced = db.scalar(
            select(GenerativeSession.id).where(
                GenerativeSession.fine_tuning_run_id == run_id
            )
        )
        if referenced:
            raise HTTPException(
                status_code=409,
                detail="The adapter is used by a generative session.",
            )
        if run.artifact_path:
            artifact = ensure_managed_path(
                Path(run.artifact_path), Path(_config()["FINE_TUNING_PATH"])
            )
        db.delete(run)
        db.commit()
    if artifact and artifact.exists():
        shutil.rmtree(artifact)


def _base_inventory(db, models_root: Path) -> dict[str, LocalModelInfo]:
    """Build the base-model portion of the inventory keyed by inventory key.

    Merges three sources: catalog entries (never downloaded), database
    download rows (downloading/ready/error), and filesystem snapshots
    downloaded by earlier flows without a database row.
    """
    entries: dict[str, LocalModelInfo] = {}
    for model_key, model in MODEL_CATALOG.items():
        revision = "main"
        key = f"base:{model_key}:{revision}"
        entries[key] = LocalModelInfo(
            key=key,
            kind="base",
            name=model["name"],
            source=model["repository"],
            status="not_downloaded",
            recommended_vram_gb=model["recommended_vram_gb"],
            downloadable=True,
        )

    rows = db.scalars(select(ManagedLocalModel)).all()
    for row in rows:
        model = MODEL_CATALOG.get(row.model_key)
        if model is None:
            continue
        key = f"base:{row.model_key}:{row.base_model_revision}"
        path = models_root / row.model_key / row.base_model_revision
        files_present = path.exists() and (path / "config.json").exists()
        if row.status == DatafileStatus.READY and not files_present:
            # Reported as an error without mutating state during a GET.
            row_status = "error"
        else:
            row_status = row.status.value
        active = db.scalar(
            select(FineTuningRun.id).where(
                FineTuningRun.base_model_id == row.model_key,
                FineTuningRun.status.in_(ACTIVE_STATUSES),
            )
        )
        used_by_session = db.scalar(
            select(GenerativeSession.id).where(
                GenerativeSession.local_model_id == row.id
            )
        )
        entries[key] = LocalModelInfo(
            key=key,
            kind="base",
            name=model["name"],
            source=model["repository"],
            path=str(path) if files_present else None,
            size_bytes=row.size_bytes or 0,
            status=row_status,
            local_model_id=row.id,
            recommended_vram_gb=model["recommended_vram_gb"],
            downloadable=row_status == "error",
            in_use=bool(active or used_by_session),
        )

    # Snapshots present on disk without a database row (downloaded through
    # preflight before tracked downloads existed) are reported as ready; the
    # download endpoint adopts them into a row without re-downloading.
    tracked_keys = {f"base:{row.model_key}:{row.base_model_revision}" for row in rows}
    for metadata_path in models_root.glob("*/*/.dashai_model.json"):
        path = metadata_path.parent
        metadata = read_model_metadata(path)
        if not metadata:
            continue
        model_key = metadata["model_id"]
        if model_key not in MODEL_CATALOG:
            continue
        key = f"base:{model_key}:{metadata['requested_revision']}"
        if key in tracked_keys:
            continue
        active = db.scalar(
            select(FineTuningRun.id).where(
                FineTuningRun.base_model_id == model_key,
                FineTuningRun.status.in_(ACTIVE_STATUSES),
            )
        )
        entries[key] = LocalModelInfo(
            key=key,
            kind="base",
            name=resolve_model(model_key)["name"],
            source=metadata["repository"],
            path=str(path),
            size_bytes=directory_size(path),
            status="ready",
            recommended_vram_gb=MODEL_CATALOG[model_key]["recommended_vram_gb"],
            downloadable=False,
            in_use=bool(active),
        )
    return entries


def _inventory(db) -> list[LocalModelInfo]:
    result: list[LocalModelInfo] = list(
        _base_inventory(db, Path(_config()["LLM_MODELS_PATH"])).values()
    )
    for run in db.scalars(
        select(FineTuningRun).where(FineTuningRun.artifact_path.is_not(None))
    ):
        path = Path(run.artifact_path)
        if not path.exists():
            continue
        result.append(
            LocalModelInfo(
                key=f"adapter:{run.id}",
                kind="adapter",
                name=run.name,
                source=run.base_model_id,
                path=str(path),
                size_bytes=directory_size(path),
                status=run.status.value,
                run_id=run.id,
                in_use=bool(run.generative_sessions),
            )
        )
    return result


@router.get("/models", response_model=list[LocalModelInfo])
async def list_models(
    session_factory: "sessionmaker" = Depends(lambda: di["session_factory"]),
):
    with session_factory() as db:
        return _inventory(db)


def _parse_base_key(key: str) -> tuple[str, str]:
    parts = key.split(":", 2)
    if len(parts) != 3 or parts[0] != "base":
        raise HTTPException(
            status_code=400,
            detail="Only DashAI-managed base models use this endpoint.",
        )
    model_key, revision = parts[1], parts[2]
    try:
        resolve_model(model_key)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return model_key, revision


@router.post("/models/{key}/download", status_code=status.HTTP_202_ACCEPTED)
async def download_model(
    key: str,
    session_factory: "sessionmaker" = Depends(lambda: di["session_factory"]),
    job_queue: "BaseJobQueue" = Depends(lambda: di["job_queue"]),
):
    model_key, revision = _parse_base_key(key)
    with session_factory() as db:
        row = db.scalar(
            select(ManagedLocalModel).where(
                ManagedLocalModel.model_key == model_key,
                ManagedLocalModel.base_model_revision == revision,
            )
        )
        if row and row.status in (DatafileStatus.DOWNLOADING, DatafileStatus.READY):
            raise HTTPException(
                status_code=409,
                detail=f"The model is already {row.status.value}.",
            )
        if row is None:
            row = ManagedLocalModel(
                model_key=model_key,
                base_model_revision=revision,
                status=DatafileStatus.DOWNLOADING,
            )
            db.add(row)
        else:
            row.status = DatafileStatus.DOWNLOADING
            row.error_message = None
        try:
            db.commit()
        except IntegrityError as exc:
            db.rollback()
            raise HTTPException(
                status_code=409, detail="The model is already downloading."
            ) from exc
        db.refresh(row)
        row_id = row.id

    job = ManagedModelDownloadJob(managed_model_id=row_id, model_key=model_key)
    try:
        job_queue.put(job)
    except Exception as exc:
        with session_factory() as db:
            row = db.get(ManagedLocalModel, row_id)
            row.status = DatafileStatus.ERROR
            row.error_message = f"Could not enqueue download: {exc}"
            db.commit()
        raise HTTPException(
            status_code=500, detail="Could not enqueue model download."
        ) from exc
    return {"detail": f"Download of {model_key} started.", "local_model_id": row_id}


@router.delete("/models/{key}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_model(
    key: str,
    session_factory: "sessionmaker" = Depends(lambda: di["session_factory"]),
):
    model_key, revision = _parse_base_key(key)
    with session_factory() as db:
        row = db.scalar(
            select(ManagedLocalModel).where(
                ManagedLocalModel.model_key == model_key,
                ManagedLocalModel.base_model_revision == revision,
            )
        )
        if row and row.status == DatafileStatus.DOWNLOADING:
            raise HTTPException(
                status_code=409, detail="The model is currently downloading."
            )
        active = db.scalar(
            select(FineTuningRun.id).where(
                FineTuningRun.base_model_id == model_key,
                FineTuningRun.status.in_(ACTIVE_STATUSES),
            )
        )
        if active:
            raise HTTPException(
                status_code=409, detail="The model is used by an active run."
            )
        if row:
            referenced = db.scalar(
                select(GenerativeSession.id).where(
                    GenerativeSession.local_model_id == row.id
                )
            )
            if referenced:
                raise HTTPException(
                    status_code=409,
                    detail="The model is used by a generative session.",
                )
            db.delete(row)
            db.commit()
    candidates = [
        item
        for item in Path(_config()["LLM_MODELS_PATH"]).glob("*/*")
        if (metadata := read_model_metadata(item))
        and metadata["model_id"] == model_key
        and metadata["requested_revision"] == revision
    ]
    if not candidates:
        if row is None:
            raise HTTPException(status_code=404, detail="Managed model not found.")
        return
    path = ensure_managed_path(candidates[0], Path(_config()["LLM_MODELS_PATH"]))
    shutil.rmtree(path)
