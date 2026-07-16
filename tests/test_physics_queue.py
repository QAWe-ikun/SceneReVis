import json
import time
from pathlib import Path

import pytest

from script.pretreatment import enqueue_isaac_jobs
from utils.physics_queue import (
    atomic_write_json,
    build_job,
    claim_next_job,
    finish_job,
    heartbeat_path,
    load_json,
    publish_job,
    queue_status,
    recover_stale_jobs,
    relative_to_shared_root,
    resolve_shared_path,
)


def _job(sample_id: str = "sample-001", max_retries: int = 2):
    return build_job(
        sample_id=sample_id,
        manifest_relpath=f"outputs/hap/{sample_id}/manifest.json",
        max_retries=max_retries,
    )


def test_publish_is_idempotent_and_refreshes_terminal_job(tmp_path: Path):
    queue = tmp_path / "queue"
    job = _job()
    assert publish_job(queue, job) == "pending"
    assert publish_job(queue, job) == "pending"
    assert queue_status(queue)["counts"]["pending"] == 1

    claimed = claim_next_job(queue, "worker-0", gpu=0)
    assert claimed is not None
    assert finish_job(queue, claimed, succeeded=True, result_relpath="result.json") == "done"
    assert publish_job(queue, job) == "done"
    assert publish_job(queue, job, refresh_terminal=True) == "pending"


def test_only_one_worker_claims_a_job(tmp_path: Path):
    queue = tmp_path / "queue"
    publish_job(queue, _job())

    first = claim_next_job(queue, "worker-0", gpu=0)
    second = claim_next_job(queue, "worker-1", gpu=1)

    assert first is not None
    assert second is None
    assert publish_job(queue, _job()) == "running"
    claimed = load_json(first)
    assert claimed["attempt"] == 1
    assert claimed["worker_id"] == "worker-0"
    assert claimed["gpu"] == 0
    assert queue_status(queue)["counts"] == {
        "pending": 0,
        "done": 0,
        "failed": 0,
        "running": 1,
    }


def test_execution_failure_retries_then_becomes_failed(tmp_path: Path):
    queue = tmp_path / "queue"
    publish_job(queue, _job(max_retries=1))

    first = claim_next_job(queue, "worker-0")
    assert first is not None
    assert finish_job(queue, first, succeeded=False, error="first failure") == "pending"

    second = claim_next_job(queue, "worker-1")
    assert second is not None
    assert load_json(second)["attempt"] == 2
    assert finish_job(queue, second, succeeded=False, error="second failure") == "failed"
    failed = next((queue / "failed").glob("*.json"))
    assert load_json(failed)["last_error"] == "second failure"


def test_stale_worker_job_is_requeued(tmp_path: Path):
    queue = tmp_path / "queue"
    publish_job(queue, _job(max_retries=2))
    claimed = claim_next_job(queue, "worker-0", gpu=0)
    assert claimed is not None

    heartbeat = load_json(heartbeat_path(queue, "worker-0"))
    heartbeat["timestamp_epoch"] = time.time() - 1000.0
    atomic_write_json(heartbeat_path(queue, "worker-0"), heartbeat)

    recovered = recover_stale_jobs(queue, stale_after_seconds=10.0)
    assert recovered == {"workers_stale": 1, "requeued": 1, "failed": 0}
    assert queue_status(queue)["counts"]["pending"] == 1
    assert not claimed.exists()


def test_stale_worker_exhausts_retry_budget(tmp_path: Path):
    queue = tmp_path / "queue"
    publish_job(queue, _job(max_retries=0))
    claimed = claim_next_job(queue, "worker-0")
    assert claimed is not None
    heartbeat = load_json(heartbeat_path(queue, "worker-0"))
    heartbeat["timestamp_epoch"] = 0.0
    atomic_write_json(heartbeat_path(queue, "worker-0"), heartbeat)

    recovered = recover_stale_jobs(queue, stale_after_seconds=1.0)
    assert recovered["failed"] == 1
    assert queue_status(queue)["counts"]["failed"] == 1


def test_shared_paths_are_portable_and_cannot_escape(tmp_path: Path):
    shared = tmp_path / "shared"
    manifest = shared / "outputs" / "sample" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({}), encoding="utf-8")

    relative = relative_to_shared_root(manifest, shared)
    assert relative == "outputs/sample/manifest.json"
    assert resolve_shared_path(relative, shared) == manifest
    with pytest.raises(ValueError):
        resolve_shared_path("../outside.json", shared)
    with pytest.raises(ValueError):
        relative_to_shared_root(tmp_path / "outside.json", shared)


def test_enqueue_producer_publishes_portable_manifest_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    shared = tmp_path / "shared"
    sample_dir = shared / "outputs" / "work" / "sample-001"
    sample_dir.mkdir(parents=True)
    (sample_dir / "geometry.npz").write_bytes(b"geometry")
    manifest = {
        "sample_id": "sample-001",
        "geometry_npz": "geometry.npz",
        "result_json": "isaac_result.json",
    }
    manifest_path = sample_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    input_path = shared / "outputs" / "release.json"
    input_path.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "sample_id": "sample-001",
                        "valid_output": True,
                        "physics_manifest": "outputs/work/sample-001/manifest.json",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    queue = shared / "outputs" / "physics_queue"
    monkeypatch.setattr(
        "sys.argv",
        [
            "enqueue_isaac_jobs.py",
            "--input_json",
            str(input_path),
            "--shared_root",
            str(shared),
            "--queue_root",
            str(queue),
        ],
    )

    enqueue_isaac_jobs.main()

    job_path = next((queue / "pending").glob("*.json"))
    job = load_json(job_path)
    assert job["manifest_relpath"] == "outputs/work/sample-001/manifest.json"
    assert job["metadata"]["source_result_relpath"] == "outputs/release.json"
    assert not Path(job["manifest_relpath"]).is_absolute()


@pytest.mark.parametrize("worker_id", ["../worker", "worker/name", "worker name", ""])
def test_worker_id_must_be_path_safe(tmp_path: Path, worker_id: str):
    with pytest.raises(ValueError):
        heartbeat_path(tmp_path / "queue", worker_id)
