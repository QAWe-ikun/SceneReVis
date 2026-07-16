"""Inspect, recover, and merge the shared HAP-Place Isaac Sim queue."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent.parent))
sys.path.insert(0, str(SCRIPT_DIR))

from utils.physics_queue import (
    atomic_write_json,
    queue_status,
    recover_stale_jobs,
    relative_to_shared_root,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manage the HAP-Place Isaac queue")
    subparsers = parser.add_subparsers(dest="command", required=True)

    status = subparsers.add_parser("status")
    status.add_argument("--queue_root", required=True)

    recover = subparsers.add_parser("recover")
    recover.add_argument("--queue_root", required=True)
    recover.add_argument("--stale_seconds", type=float, default=600.0)
    recover.add_argument(
        "--watch_interval",
        type=float,
        default=0.0,
        help="Repeat recovery at this interval; zero performs one pass",
    )

    merge = subparsers.add_parser("merge")
    merge.add_argument("--input_json", required=True)
    merge.add_argument("--output_json", required=True)
    merge.add_argument("--shared_root", required=True)
    return parser.parse_args()


def _resolve_manifest(raw_path: str, input_path: Path, shared_root: Path) -> Path:
    path = Path(raw_path)
    candidates = [path] if path.is_absolute() else [shared_root / path, Path.cwd() / path, input_path.parent / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _result_path(manifest_path: Path) -> Path:
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    path = Path(manifest["result_json"])
    return path if path.is_absolute() else manifest_path.parent / path


def _merge(args: argparse.Namespace) -> None:
    from run_isaac_batch import _merge_physics, _refresh_summary

    input_path = Path(args.input_json).resolve()
    output_path = Path(args.output_json).resolve()
    shared_root = Path(args.shared_root).resolve()
    relative_to_shared_root(input_path, shared_root)
    relative_to_shared_root(output_path, shared_root)
    with open(input_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    for result in payload.get("results", []):
        manifest_text = result.get("physics_manifest")
        if result.get("valid_output") is not True or not manifest_text:
            continue
        manifest_path = _resolve_manifest(str(manifest_text), input_path, shared_root)
        physics_path = _result_path(manifest_path)
        if not physics_path.exists():
            result["status"] = "physics_pending"
            result["sim_ready"] = False
            result["failure_code"] = "missing_isaac_result"
            continue
        with open(physics_path, "r", encoding="utf-8") as handle:
            physics = json.load(handle)
        _merge_physics(result, physics)

    payload.setdefault("config", {})["physics_backend"] = "isaac_queue"
    _refresh_summary(payload)
    atomic_write_json(output_path, payload)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    if args.command == "status":
        print(json.dumps(queue_status(Path(args.queue_root)), ensure_ascii=False, indent=2))
        return
    if args.command == "merge":
        _merge(args)
        return
    if args.stale_seconds <= 0 or args.watch_interval < 0:
        raise ValueError("Recovery intervals must be valid positive durations")
    while True:
        recovered = recover_stale_jobs(Path(args.queue_root), args.stale_seconds)
        print(json.dumps(recovered, ensure_ascii=False), flush=True)
        if args.watch_interval == 0:
            return
        time.sleep(args.watch_interval)


if __name__ == "__main__":
    main()
