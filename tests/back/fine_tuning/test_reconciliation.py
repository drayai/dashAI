import json

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from DashAI.back.core.enums.status import FineTuningStatus
from DashAI.back.dependencies.database.models import Base, FineTuningRun
from DashAI.back.fine_tuning.reconciliation import (
    INTERRUPTED_RUNNING_MESSAGE,
    MISSING_TASK_MESSAGE,
    clear_stale_gpu_lock,
    quarantine_orphan_temp_dirs,
    reconcile_fine_tuning_runs,
)


class FakeQueue:
    """Queue stub exposing only the interface reconciliation needs."""

    def __init__(self, pending_ids=()):
        self.pending = set(pending_ids)

    def task_snapshot(self, job_id):
        return {"pending": job_id in self.pending, "mirror_status": None}


class OpaqueQueue:
    """Queue without task introspection: queued runs must be kept."""


@pytest.fixture(name="session_factory")
def session_factory_fixture(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'reconciliation.db'}")
    Base.metadata.create_all(engine, tables=[FineTuningRun.__table__], checkfirst=True)
    yield sessionmaker(bind=engine)
    engine.dispose()


def _make_run(name: str, status: FineTuningStatus, huey_id=None) -> FineTuningRun:
    return FineTuningRun(
        name=name,
        dataset_id=1,
        base_model_id="qwen2.5-0.5b-instruct",
        base_model_revision="main",
        method="lora",
        dataset_mapping={
            "format": "text",
            "text_column": "text",
            "validation_split": 0.1,
        },
        training_parameters={"preset": "quick_test"},
        status=status,
        huey_id=huey_id,
    )


def test_running_run_fails_after_restart(session_factory):
    with session_factory() as db:
        db.add(_make_run("interrupted", FineTuningStatus.RUNNING, huey_id="task-1"))
        db.commit()

    summary = reconcile_fine_tuning_runs(session_factory, FakeQueue())

    with session_factory() as db:
        run = db.query(FineTuningRun).one()
        assert run.status == FineTuningStatus.FAILED
        assert INTERRUPTED_RUNNING_MESSAGE in run.error_message
    assert summary["failed_running"] == [run.id]


def test_queued_run_is_kept_when_task_is_pending(session_factory):
    with session_factory() as db:
        run = _make_run("waiting", FineTuningStatus.QUEUED, huey_id="task-2")
        db.add(run)
        db.commit()
        run_id = run.id

    summary = reconcile_fine_tuning_runs(session_factory, FakeQueue({"task-2"}))

    with session_factory() as db:
        assert db.query(FineTuningRun).one().status == FineTuningStatus.QUEUED
    assert summary["kept_queued"] == [run_id]
    assert summary["failed_without_task"] == []


def test_queued_run_fails_when_task_is_missing(session_factory):
    with session_factory() as db:
        run = _make_run("orphan", FineTuningStatus.QUEUED, huey_id="task-3")
        db.add(run)
        db.commit()
        run_id = run.id

    summary = reconcile_fine_tuning_runs(session_factory, FakeQueue())

    with session_factory() as db:
        saved = db.query(FineTuningRun).one()
        assert saved.status == FineTuningStatus.FAILED
        assert MISSING_TASK_MESSAGE in saved.error_message
    assert summary["failed_without_task"] == [run_id]


def test_queued_run_is_kept_when_queue_cannot_be_introspected(session_factory):
    with session_factory() as db:
        db.add(_make_run("opaque", FineTuningStatus.QUEUED, huey_id="task-4"))
        db.commit()

    reconcile_fine_tuning_runs(session_factory, OpaqueQueue())

    with session_factory() as db:
        assert db.query(FineTuningRun).one().status == FineTuningStatus.QUEUED


def test_reconciliation_is_idempotent(session_factory):
    with session_factory() as db:
        db.add(_make_run("interrupted", FineTuningStatus.RUNNING, huey_id="t-1"))
        db.add(_make_run("orphan", FineTuningStatus.QUEUED, huey_id="t-2"))
        db.add(_make_run("waiting", FineTuningStatus.QUEUED, huey_id="t-3"))
        db.commit()

    first = reconcile_fine_tuning_runs(session_factory, FakeQueue({"t-3"}))
    second = reconcile_fine_tuning_runs(session_factory, FakeQueue({"t-3"}))

    assert first["failed_running"]
    assert first["failed_without_task"]
    assert second == {
        "failed_running": [],
        "failed_without_task": [],
        "kept_queued": [_id_of(session_factory, "waiting")],
    }


def _id_of(session_factory, name):
    with session_factory() as db:
        return db.query(FineTuningRun).filter_by(name=name).one().id


def test_valid_run_dirs_are_not_quarantined(tmp_path):
    valid = tmp_path / "run-1"
    valid.mkdir()
    (valid / "manifest.json").write_text("{}", encoding="utf-8")

    assert quarantine_orphan_temp_dirs(tmp_path) == []
    assert valid.exists()


def test_orphan_tmp_dir_is_quarantined_and_idempotent(tmp_path):
    orphan = tmp_path / "run-2.tmp"
    orphan.mkdir()
    (orphan / "adapter").mkdir()
    (orphan / "adapter" / "weights.safetensors").write_bytes(b"x")

    quarantined = quarantine_orphan_temp_dirs(tmp_path)

    assert len(quarantined) == 1
    target = tmp_path / quarantined[0]
    assert target.name.startswith("orphaned-run-2.tmp-")
    assert not orphan.exists()
    assert (target / "adapter" / "weights.safetensors").read_bytes() == b"x"
    # Second pass is a no-op: the suffix no longer matches.
    assert quarantine_orphan_temp_dirs(tmp_path) == []


def test_tmp_dirs_outside_managed_path_are_untouched(tmp_path):
    fine_tuning_root = tmp_path / "fine_tuning"
    fine_tuning_root.mkdir()
    outside = tmp_path / "elsewhere"
    foreign = outside / "run-9.tmp"
    foreign.mkdir(parents=True)

    quarantine_orphan_temp_dirs(fine_tuning_root)

    # Only the managed root is scanned: the foreign directory stays intact.
    assert foreign.exists()
    assert list(fine_tuning_root.iterdir()) == []


def test_stale_gpu_lock_is_cleared_at_startup(tmp_path):
    config = {"FINE_TUNING_PATH": str(tmp_path)}
    lock_path = tmp_path / "gpu.lock"
    lock_path.write_text(
        json.dumps({"owner": "fine_tuning_run_1", "pid": -1}), encoding="utf-8"
    )

    assert clear_stale_gpu_lock(config) is True
    assert not lock_path.exists()
    assert clear_stale_gpu_lock(config) is False
