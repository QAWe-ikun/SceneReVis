"""Persistent Windows Isaac Sim consumer for the shared HAP-Place queue."""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
import traceback
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent.parent))
sys.path.insert(0, str(SCRIPT_DIR))

from utils.physics_queue import (
    claim_next_job,
    finish_job,
    load_json,
    relative_to_shared_root,
    resolve_shared_path,
    write_heartbeat,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persistent HAP-Place Isaac queue consumer")
    parser.add_argument("--shared_root", required=True)
    parser.add_argument("--queue_root", required=True)
    parser.add_argument("--worker_id", default=f"{socket.gethostname()}-{os.getpid()}")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu_threads", type=int, default=16)
    parser.add_argument("--poll_seconds", type=float, default=2.0)
    parser.add_argument("--heartbeat_seconds", type=float, default=10.0)
    parser.add_argument("--idle_exit_seconds", type=float, default=0.0)
    parser.add_argument("--max_jobs", type=int, default=0)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _manifest_result_path(manifest_path: Path) -> Path:
    manifest = load_json(manifest_path)
    path = Path(manifest["result_json"])
    return path if path.is_absolute() else manifest_path.parent / path


def _event(name: str, **values: object) -> None:
    print(json.dumps({"event": name, **values}, ensure_ascii=False, allow_nan=False), flush=True)


def main() -> None:
    args = parse_args()
    if args.gpu < 0 or args.cpu_threads <= 0:
        raise ValueError("--gpu must be non-negative and --cpu_threads must be positive")
    if args.poll_seconds <= 0 or args.heartbeat_seconds <= 0:
        raise ValueError("poll and heartbeat intervals must be positive")

    shared_root = Path(args.shared_root).resolve()
    queue_root = Path(args.queue_root).resolve()
    relative_to_shared_root(queue_root, shared_root)

    from isaacsim.simulation_app import SimulationApp

    simulation_app = SimulationApp(
        {
            "headless": args.headless,
            "active_gpu": args.gpu,
            "physics_gpu": args.gpu,
            "multi_gpu": False,
            "limit_cpu_threads": args.cpu_threads,
        }
    )
    from run_isaac_settle import _run_manifest

    processed = 0
    idle_since = time.monotonic()
    current_job_id = None
    try:
        write_heartbeat(queue_root, args.worker_id, None, gpu=args.gpu)
        _event("consumer_started", worker_id=args.worker_id, gpu=args.gpu)
        while simulation_app.is_running():
            claimed_path = claim_next_job(queue_root, args.worker_id, gpu=args.gpu)
            if claimed_path is None:
                if args.idle_exit_seconds > 0 and time.monotonic() - idle_since >= args.idle_exit_seconds:
                    break
                time.sleep(args.poll_seconds)
                continue

            idle_since = time.monotonic()
            job = load_json(claimed_path)
            current_job_id = str(job["job_id"])
            manifest_path = resolve_shared_path(str(job["manifest_relpath"]), shared_root)
            last_heartbeat = 0.0

            def progress() -> None:
                nonlocal last_heartbeat
                now = time.monotonic()
                if now - last_heartbeat >= args.heartbeat_seconds:
                    write_heartbeat(
                        queue_root,
                        args.worker_id,
                        current_job_id,
                        gpu=args.gpu,
                    )
                    last_heartbeat = now

            _event("job_started", job_id=current_job_id, manifest=str(manifest_path))
            try:
                progress()
                result = _run_manifest(manifest_path, simulation_app, progress_callback=progress)
                if result.get("error"):
                    state = finish_job(
                        queue_root,
                        claimed_path,
                        succeeded=False,
                        error=str(result["error"]),
                    )
                    _event("job_execution_failed", job_id=current_job_id, state=state)
                else:
                    result_path = _manifest_result_path(manifest_path).resolve()
                    result_relpath = relative_to_shared_root(result_path, shared_root)
                    state = finish_job(
                        queue_root,
                        claimed_path,
                        succeeded=True,
                        result_relpath=result_relpath,
                    )
                    _event(
                        "job_completed",
                        job_id=current_job_id,
                        state=state,
                        sim_ready=result.get("sim_ready") is True,
                    )
            except Exception as exc:
                state = finish_job(
                    queue_root,
                    claimed_path,
                    succeeded=False,
                    error=f"{exc}\n{traceback.format_exc()}",
                )
                _event("job_execution_failed", job_id=current_job_id, state=state, error=str(exc))

            processed += 1
            current_job_id = None
            if args.max_jobs > 0 and processed >= args.max_jobs:
                break
    finally:
        write_heartbeat(queue_root, args.worker_id, current_job_id, gpu=args.gpu)
        simulation_app.close()
        _event("consumer_stopped", worker_id=args.worker_id, processed=processed)


if __name__ == "__main__":
    main()
