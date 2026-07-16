"""Run all HAP-Place physics manifests in one Isaac Sim process and merge results."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from utils.hap_place import decompose_transform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch HAP-Place Isaac Sim projection")
    parser.add_argument("--input_json", required=True, help="Release-stage HAP-Place result JSON")
    parser.add_argument("--output_json", required=True, help="Merged simulator-ready result JSON")
    parser.add_argument("--isaac_python", required=True, help="Isaac Sim python.sh")
    parser.add_argument("--timeout", type=float, default=7200.0)
    parser.add_argument(
        "--skip_completed",
        action="store_true",
        help="Reuse existing successful v2 Isaac result files; default reruns every manifest",
    )
    return parser.parse_args()


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_payload(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
    temporary.replace(path)


def _isaac_result_path(manifest_path: Path) -> Path:
    manifest = _load_json(manifest_path)
    result_path = Path(manifest["result_json"])
    return result_path if result_path.is_absolute() else manifest_path.parent / result_path


def _is_completed(manifest_path: Path) -> bool:
    result_path = _isaac_result_path(manifest_path)
    if not result_path.exists():
        return False
    result = _load_json(result_path)
    return result.get("schema") == "hap_place_isaac_result_v2" and not result.get("error")


def _merge_physics(result: Dict[str, Any], physics: Mapping[str, Any]) -> None:
    result["physics"] = dict(physics)
    sim_ready = physics.get("sim_ready") is True
    result["sim_ready"] = sim_ready
    result["status"] = "simulator_ready" if sim_ready else "physics_failed"
    result["failure_code"] = None if sim_ready else str(
        physics.get("failure_code") or "physics_failed"
    )
    result.setdefault("timing_seconds", {})["physics"] = physics.get("worker_wall_seconds")

    simulator_record = result.get("simulator_record")
    final_transform = physics.get("final_transform_original_to_world")
    if not isinstance(simulator_record, dict) or final_transform is None:
        return
    transform = np.asarray(final_transform, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        result["status"] = "physics_failed"
        result["sim_ready"] = False
        result["failure_code"] = "invalid_isaac_transform"
        return

    pose = decompose_transform(transform)
    simulator_record["pose_stage"] = "settled"
    simulator_record["sim_ready"] = sim_ready
    target = simulator_record.get("target_object")
    if isinstance(target, dict):
        target["position_xyz"] = pose["position_xyz"]
        target["rotation_xyzw"] = pose["rotation_xyzw"]
        target["scale_xyz"] = pose["scale_xyz"]
        target["original_to_world"] = pose["transform"]


def _refresh_summary(payload: Dict[str, Any]) -> None:
    results: Sequence[Mapping[str, Any]] = payload.get("results", [])
    statuses = Counter(str(result.get("status", "unknown")) for result in results)
    failures = Counter(
        str(result["failure_code"])
        for result in results
        if result.get("failure_code") is not None
    )
    payload["summary"] = {
        "processed": len(results),
        "status_counts": dict(statuses),
        "failure_counts": dict(failures),
        "sim_ready": sum(result.get("sim_ready") is True for result in results),
    }


def main() -> None:
    args = parse_args()
    if args.timeout <= 0:
        raise ValueError("--timeout must be positive")
    isaac_python = Path(args.isaac_python)
    if not isaac_python.exists():
        raise FileNotFoundError(isaac_python)

    input_path = Path(args.input_json)
    output_path = Path(args.output_json)
    payload = _load_json(input_path)
    manifests = []
    for result in payload.get("results", []):
        manifest_text = result.get("physics_manifest")
        if result.get("valid_output") is not True or not manifest_text:
            continue
        manifest_path = Path(manifest_text)
        if not manifest_path.exists():
            result["status"] = "physics_failed"
            result["sim_ready"] = False
            result["failure_code"] = "missing_physics_manifest"
            continue
        if args.skip_completed and _is_completed(manifest_path):
            continue
        manifests.append(manifest_path)

    if manifests:
        list_path = output_path.parent / f".{output_path.stem}_manifests.json"
        list_path.parent.mkdir(parents=True, exist_ok=True)
        list_path.write_text(
            json.dumps([str(path.resolve()) for path in manifests], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        worker = Path(__file__).with_name("run_isaac_settle.py")
        completed = subprocess.run(
            [
                str(isaac_python),
                str(worker),
                "--manifest_list",
                str(list_path),
            ],
            check=False,
            timeout=args.timeout,
        )
        if completed.returncode != 0:
            print(
                "Isaac worker reported one or more errors; merging all result files that were produced.",
                file=sys.stderr,
            )

    for result in payload.get("results", []):
        manifest_text = result.get("physics_manifest")
        if result.get("valid_output") is not True or not manifest_text:
            continue
        manifest_path = Path(manifest_text)
        if not manifest_path.exists():
            continue
        physics_path = _isaac_result_path(manifest_path)
        if not physics_path.exists():
            result["status"] = "physics_failed"
            result["sim_ready"] = False
            result["failure_code"] = "missing_isaac_result"
            continue
        _merge_physics(result, _load_json(physics_path))

    payload.setdefault("config", {})["physics_backend"] = "isaac_batch"
    payload["config"]["isaac_python"] = str(isaac_python)
    _refresh_summary(payload)
    _write_payload(output_path, payload)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
