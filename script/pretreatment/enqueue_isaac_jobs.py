"""Publish HAP-Place release manifests to the shared Isaac Sim queue."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from utils.physics_queue import (
    build_job,
    publish_job,
    relative_to_shared_root,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enqueue HAP-Place Isaac Sim jobs")
    parser.add_argument("--input_json", required=True, help="Release-stage HAP-Place JSON")
    parser.add_argument("--shared_root", required=True, help="Root shared by WSL and Windows")
    parser.add_argument("--queue_root", required=True, help="Queue directory on the shared drive")
    parser.add_argument("--max_retries", type=int, default=2)
    parser.add_argument(
        "--refresh_terminal",
        action="store_true",
        help="Republish jobs currently in done or failed; running jobs are never replaced",
    )
    return parser.parse_args()


def _candidate_paths(raw_path: str, input_path: Path, shared_root: Path) -> Iterable[Path]:
    path = Path(raw_path)
    if path.is_absolute():
        yield path
        return
    yield shared_root / path
    yield Path.cwd() / path
    yield input_path.parent / path


def _resolve_existing(raw_path: str, input_path: Path, shared_root: Path) -> Path:
    for candidate in _candidate_paths(raw_path, input_path, shared_root):
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not resolve shared file: {raw_path}")


def _resolve_manifest_file(manifest_path: Path, path_text: str) -> Path:
    path = Path(path_text)
    return path.resolve() if path.is_absolute() else (manifest_path.parent / path).resolve()


def main() -> None:
    args = parse_args()
    if args.max_retries < 0:
        raise ValueError("--max_retries must be non-negative")

    input_path = Path(args.input_json).resolve()
    shared_root = Path(args.shared_root).resolve()
    queue_root = Path(args.queue_root).resolve()
    relative_to_shared_root(queue_root, shared_root)
    source_relpath = relative_to_shared_root(input_path, shared_root)

    with open(input_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    counts: Counter[str] = Counter()
    for result in payload.get("results", []):
        if result.get("valid_output") is not True:
            counts["invalid_skipped"] += 1
            continue
        manifest_text = result.get("physics_manifest")
        if not manifest_text:
            counts["missing_manifest_skipped"] += 1
            continue
        try:
            manifest_path = _resolve_existing(str(manifest_text), input_path, shared_root)
            manifest_relpath = relative_to_shared_root(manifest_path, shared_root)
            with open(manifest_path, "r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            geometry_path = _resolve_manifest_file(manifest_path, manifest["geometry_npz"])
            result_path = _resolve_manifest_file(manifest_path, manifest["result_json"])
            if not geometry_path.exists():
                raise FileNotFoundError(geometry_path)
            relative_to_shared_root(geometry_path, shared_root)
            relative_to_shared_root(result_path, shared_root)
        except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            counts["invalid_manifest_skipped"] += 1
            print(f"Skipping {result.get('sample_id')}: {exc}", file=sys.stderr)
            continue

        sample_id = str(result.get("sample_id") or manifest.get("sample_id") or "")
        if not sample_id:
            counts["missing_sample_id_skipped"] += 1
            continue
        job = build_job(
            sample_id=sample_id,
            manifest_relpath=manifest_relpath,
            max_retries=args.max_retries,
            metadata={"source_result_relpath": source_relpath},
        )
        state = publish_job(
            queue_root=queue_root,
            job=job,
            refresh_terminal=args.refresh_terminal,
        )
        counts[f"state_{state}"] += 1

    print(json.dumps(dict(counts), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
