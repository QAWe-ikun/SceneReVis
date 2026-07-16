"""Durable cross-platform file queue for Windows Isaac Sim consumers."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Mapping, Optional


QUEUE_STATES = ("pending", "running", "done", "failed")
JOB_SCHEMA = "hap_place_physics_job_v1"


def _now_fields() -> Dict[str, Any]:
    now = time.time()
    return {
        "timestamp_epoch": now,
        "timestamp_utc": datetime.fromtimestamp(now, timezone.utc).isoformat(),
    }


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return data


def ensure_queue(queue_root: Path) -> None:
    for state in QUEUE_STATES:
        (queue_root / state).mkdir(parents=True, exist_ok=True)


def make_job_id(sample_id: str) -> str:
    digest = hashlib.sha1(sample_id.encode("utf-8")).hexdigest()[:16]
    return f"job_{digest}"


def relative_to_shared_root(path: Path, shared_root: Path) -> str:
    resolved_path = path.resolve()
    resolved_root = shared_root.resolve()
    try:
        relative = resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"{resolved_path} is outside shared root {resolved_root}") from exc
    return relative.as_posix()


def resolve_shared_path(relative_path: str, shared_root: Path) -> Path:
    pure = PurePosixPath(relative_path)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"Shared path must be a safe relative POSIX path: {relative_path!r}")
    return shared_root.joinpath(*pure.parts)


def build_job(
    sample_id: str,
    manifest_relpath: str,
    max_retries: int = 2,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")
    job = {
        "schema": JOB_SCHEMA,
        "job_id": make_job_id(sample_id),
        "sample_id": sample_id,
        "manifest_relpath": manifest_relpath,
        "status": "pending",
        "attempt": 0,
        "max_retries": int(max_retries),
        "producer_host": socket.gethostname(),
        "created": _now_fields(),
    }
    if metadata:
        job["metadata"] = dict(metadata)
    return job


def _job_name(job: Mapping[str, Any]) -> str:
    job_id = str(job["job_id"])
    if not job_id or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in job_id):
        raise ValueError(f"Unsafe job_id: {job_id!r}")
    return f"{job_id}.json"


def _find_job_path(queue_root: Path, job_id: str) -> Optional[Path]:
    name = f"{job_id}.json"
    for state in ("pending", "done", "failed"):
        candidate = queue_root / state / name
        if candidate.exists():
            return candidate
    running_root = queue_root / "running"
    if running_root.exists():
        for worker_dir in running_root.iterdir():
            candidate = worker_dir / name
            if candidate.exists():
                return candidate
    return None


def publish_job(queue_root: Path, job: Mapping[str, Any], refresh_terminal: bool = False) -> str:
    ensure_queue(queue_root)
    payload = dict(job)
    if payload.get("schema") != JOB_SCHEMA:
        raise ValueError(f"Unsupported job schema: {payload.get('schema')!r}")
    name = _job_name(payload)
    existing = _find_job_path(queue_root, str(payload["job_id"]))
    if existing is not None:
        state = "running" if existing.parent.parent == queue_root / "running" else existing.parent.name
        if not refresh_terminal or state not in {"done", "failed"}:
            return state
        existing.unlink()
    payload["status"] = "pending"
    payload["published"] = _now_fields()
    atomic_write_json(queue_root / "pending" / name, payload)
    return "pending"


def heartbeat_path(queue_root: Path, worker_id: str) -> Path:
    if not worker_id or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        for character in worker_id
    ):
        raise ValueError(f"Unsafe worker_id: {worker_id!r}")
    return queue_root / "running" / worker_id / "heartbeat.json"


def write_heartbeat(
    queue_root: Path,
    worker_id: str,
    current_job_id: Optional[str],
    gpu: Optional[int] = None,
) -> None:
    payload = {
        "schema": "hap_place_worker_heartbeat_v1",
        "worker_id": worker_id,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "current_job_id": current_job_id,
        "gpu": gpu,
        **_now_fields(),
    }
    atomic_write_json(heartbeat_path(queue_root, worker_id), payload)


def claim_next_job(queue_root: Path, worker_id: str, gpu: Optional[int] = None) -> Optional[Path]:
    ensure_queue(queue_root)
    worker_dir = heartbeat_path(queue_root, worker_id).parent
    worker_dir.mkdir(parents=True, exist_ok=True)
    for pending_path in sorted((queue_root / "pending").glob("*.json")):
        claimed_path = worker_dir / pending_path.name
        try:
            pending_path.rename(claimed_path)
        except (FileNotFoundError, FileExistsError, PermissionError, OSError):
            continue
        job = load_json(claimed_path)
        job["status"] = "running"
        job["worker_id"] = worker_id
        job["gpu"] = gpu
        job["attempt"] = int(job.get("attempt", 0)) + 1
        job["lease_token"] = uuid.uuid4().hex
        job["started"] = _now_fields()
        atomic_write_json(claimed_path, job)
        write_heartbeat(queue_root, worker_id, str(job["job_id"]), gpu=gpu)
        return claimed_path
    write_heartbeat(queue_root, worker_id, None, gpu=gpu)
    return None


def finish_job(
    queue_root: Path,
    claimed_path: Path,
    succeeded: bool,
    result_relpath: Optional[str] = None,
    error: Optional[str] = None,
) -> str:
    if not claimed_path.exists():
        return "lease_lost"
    job = load_json(claimed_path)
    worker_id = str(job.get("worker_id", claimed_path.parent.name))
    if succeeded:
        state = "done"
        job["result_relpath"] = result_relpath
        job.pop("last_error", None)
    else:
        job["last_error"] = error or "unknown consumer error"
        attempt = int(job.get("attempt", 0))
        state = "pending" if attempt <= int(job.get("max_retries", 0)) else "failed"
    job["status"] = state
    job["finished_attempt"] = _now_fields()
    job.pop("lease_token", None)
    atomic_write_json(claimed_path, job)
    destination = queue_root / state / claimed_path.name
    if destination.exists():
        destination.unlink()
    claimed_path.replace(destination)
    write_heartbeat(queue_root, worker_id, None, gpu=job.get("gpu"))
    return state


def recover_stale_jobs(queue_root: Path, stale_after_seconds: float) -> Dict[str, int]:
    if stale_after_seconds <= 0:
        raise ValueError("stale_after_seconds must be positive")
    ensure_queue(queue_root)
    now = time.time()
    counts = {"workers_stale": 0, "requeued": 0, "failed": 0}
    for worker_dir in (queue_root / "running").iterdir():
        if not worker_dir.is_dir():
            continue
        heartbeat = worker_dir / "heartbeat.json"
        try:
            heartbeat_data = load_json(heartbeat)
            heartbeat_time = float(heartbeat_data["timestamp_epoch"])
        except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            heartbeat_time = worker_dir.stat().st_mtime
        if now - heartbeat_time <= stale_after_seconds:
            continue
        counts["workers_stale"] += 1
        for claimed_path in list(worker_dir.glob("*.json")):
            if claimed_path.name == "heartbeat.json":
                continue
            state = finish_job(
                queue_root=queue_root,
                claimed_path=claimed_path,
                succeeded=False,
                error=f"worker heartbeat stale for more than {stale_after_seconds:.1f}s",
            )
            if state == "pending":
                counts["requeued"] += 1
            elif state == "failed":
                counts["failed"] += 1
        heartbeat.unlink(missing_ok=True)
    return counts


def queue_status(queue_root: Path) -> Dict[str, Any]:
    ensure_queue(queue_root)
    counts = {
        "pending": len(list((queue_root / "pending").glob("*.json"))),
        "done": len(list((queue_root / "done").glob("*.json"))),
        "failed": len(list((queue_root / "failed").glob("*.json"))),
        "running": 0,
    }
    workers = {}
    for worker_dir in (queue_root / "running").iterdir():
        if not worker_dir.is_dir():
            continue
        running_jobs = [path for path in worker_dir.glob("*.json") if path.name != "heartbeat.json"]
        counts["running"] += len(running_jobs)
        heartbeat = worker_dir / "heartbeat.json"
        try:
            workers[worker_dir.name] = load_json(heartbeat) if heartbeat.exists() else None
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            workers[worker_dir.name] = {"error": str(exc)}
    return {"counts": counts, "workers": workers}
