"""Run the complete HAP-Place release-search and physics handoff pipeline.

This command is intended to run in the WSL ``scenerevis`` environment. It
produces one result JSON and keeps simulator handoff files in a hidden work
directory next to that JSON.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import trimesh
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from script.pretreatment.components.scene_builder import SceneBuilder
from utils.hap_place import (
    CameraModel,
    FirstHitProjector,
    SceneReVisPose,
    VoxelGridSpec,
    compose_release_transform,
    concatenate_scene_mesh,
    decompose_transform,
    load_scenerevis_pose_records,
    lookup_scenerevis_pose,
    prepare_target_mesh,
    score_ordered_release_search,
    voxelize_scene,
    voxelize_target_kernel,
)
from utils.placement_heatmap import PlacementHeatmap, load_trainable_heatmap_state_dict


class SampleFailure(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HAP-Place end to end")
    parser.add_argument("--data_dir", required=True, help="Heatmap dataset root")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--checkpoint", required=True, help="Heatmap checkpoint")
    parser.add_argument("--scene_json_dir", required=True, help="Original scene JSON directory")
    parser.add_argument("--model_dir", required=True, help="3D-FUTURE model directory")
    parser.add_argument(
        "--scenerevis_results",
        default=None,
        help="JSON keyed by sample_id containing SceneReVis add_object tool calls",
    )
    parser.add_argument(
        "--allow_metadata_pose",
        action="store_true",
        help="Debug only: use removed_object rotation/size when SceneReVis pose is missing",
    )
    parser.add_argument("--output_json", required=True, help="Single pipeline result JSON")
    parser.add_argument("--camera_json", default=None, help="Optional calibrated camera JSON")
    parser.add_argument("--sample_id", action="append", default=None, help="Run selected sample id; repeatable")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=0, help="0 means all remaining samples")

    parser.add_argument("--voxel_resolution", type=int, default=256)
    parser.add_argument("--minimum_release_height_voxels", type=int, default=1)
    parser.add_argument("--max_candidates", type=int, default=65536)
    parser.add_argument("--minimum_heatmap_score", type=float, default=None)

    parser.add_argument("--physics_backend", default="none", choices=["none", "isaac"])
    parser.add_argument(
        "--isaac_python",
        default=None,
        help="Isaac Sim python.sh used when --physics_backend=isaac",
    )
    parser.add_argument("--physics_timeout", type=float, default=300.0)
    parser.add_argument("--simulation_seconds", type=float, default=5.0)
    parser.add_argument("--physics_hz", type=float, default=120.0)
    parser.add_argument("--stable_frames", type=int, default=60)
    parser.add_argument("--support_frames", type=int, default=30)
    parser.add_argument("--linear_velocity_threshold", type=float, default=0.01)
    parser.add_argument("--angular_velocity_threshold", type=float, default=0.05)
    parser.add_argument("--support_normal_min_dot", type=float, default=0.7)
    parser.add_argument("--support_contact_height_tolerance", type=float, default=0.03)
    parser.add_argument("--penetration_threshold", type=float, default=0.005)
    parser.add_argument("--tilt_threshold_degrees", type=float, default=15.0)
    parser.add_argument("--max_horizontal_displacement", type=float, default=0.1)
    parser.add_argument("--mass", type=float, default=1.0)
    parser.add_argument("--save_physics_usd", action="store_true")
    return parser.parse_args()


def _load_heatmap_model(checkpoint_path: Path, device: torch.device) -> Tuple[PlacementHeatmap, Dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = dict(checkpoint.get("args", {}))
    room_encoder = config.get("room_encoder", "siglip")
    image_size = int(config.get("image_size", 384))
    room_image_size = config.get("room_image_size")
    object_image_size = config.get("object_image_size", image_size)
    model = PlacementHeatmap(
        heatmap_res=256,
        room_encoder=room_encoder,
        dino_model=config.get("dino_model"),
        hidden_dim=int(config.get("hidden_dim", 256)),
        room_image_size=room_image_size,
        object_image_size=object_image_size,
        decoder_layers=int(config.get("decoder_layers", 3)),
        num_heads=int(config.get("num_heads", 8)),
        mlp_ratio=float(config.get("mlp_ratio", 4.0)),
        decoder_dropout=float(config.get("decoder_dropout", 0.0)),
    ).to(device)
    missing, unexpected = load_trainable_heatmap_state_dict(model, checkpoint["model_state_dict"])
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    model.eval()
    config["checkpoint_epoch"] = checkpoint.get("epoch")
    config["best_peak_acc"] = checkpoint.get("best_peak_acc")
    return model, config


def _predict_heatmap(
    model: PlacementHeatmap,
    room_path: Path,
    object_path: Path,
    description: str,
    device: torch.device,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    room_image = Image.open(room_path).convert("RGB")
    object_image = Image.open(object_path).convert("RGB")
    room_tensor = model.preprocess_room_image(room_image).unsqueeze(0).to(device)
    object_tensor = model.preprocess_object_image(object_image).unsqueeze(0).to(device)
    with torch.no_grad():
        heatmap = model.forward_tensor(
            room_image=room_tensor,
            object_desc=description,
            object_image=object_tensor,
        )[0]
    array = heatmap.detach().float().cpu().numpy()
    if array.ndim != 2 or not np.any(np.isfinite(array)):
        raise SampleFailure("invalid_heatmap", "Model returned an invalid heatmap")
    return array, room_image.size


def _extract_bounds(scene_data: Mapping[str, Any]) -> Tuple[Sequence[Sequence[float]], Sequence[Sequence[float]]]:
    envelope = scene_data.get("room_envelope", {})
    bottom = scene_data.get("bounds_bottom", envelope.get("bounds_bottom", []))
    top = scene_data.get("bounds_top", envelope.get("bounds_top", []))
    if not bottom:
        raise SampleFailure("missing_room_bounds", "Scene has no bounds_bottom")
    if not top:
        bottom_array = np.asarray(bottom, dtype=np.float64)
        top_array = bottom_array.copy()
        top_array[:, 1] += 3.0
        top = top_array.tolist()
    return bottom, top


def _add_room_walls(
    scene: trimesh.Scene,
    bounds_bottom: Sequence[Sequence[float]],
    bounds_top: Sequence[Sequence[float]],
) -> None:
    """Add open-top room walls so arbitrary calibrated views have visible surfaces."""
    bottom = np.asarray(bounds_bottom, dtype=np.float64)
    top = np.asarray(bounds_top, dtype=np.float64)
    if bottom.shape != top.shape or bottom.ndim != 2 or bottom.shape[1] != 3:
        raise SampleFailure("invalid_room_bounds", "Top and bottom room polygons must have matching XYZ vertices")
    vertices = []
    faces = []
    for index in range(len(bottom)):
        next_index = (index + 1) % len(bottom)
        base = len(vertices)
        vertices.extend([bottom[index], bottom[next_index], top[next_index], top[index]])
        faces.extend([[base, base + 1, base + 2], [base, base + 2, base + 3]])
    wall_mesh = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )
    scene.add_geometry(wall_mesh, geom_name="room_walls")


def _metadata_pose(sample: Mapping[str, Any]) -> Optional[SceneReVisPose]:
    removed = sample.get("removed_object", {})
    rotation = removed.get("rot")
    size = removed.get("size")
    if not isinstance(rotation, list) or len(rotation) != 4:
        return None
    if not isinstance(size, list) or len(size) != 3:
        return None
    return SceneReVisPose(
        rotation_xyzw=tuple(float(v) for v in rotation),
        target_size_xyz=tuple(float(v) for v in size),
        source="metadata_debug_fallback",
    )


def _load_camera_config(path: Optional[str]) -> Optional[Any]:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _camera_for_sample(
    camera_config: Optional[Any],
    scene_name: str,
    sample_id: str,
    bounds_bottom: Sequence[Sequence[float]],
    image_size: Tuple[int, int],
) -> CameraModel:
    if camera_config is None:
        return CameraModel.top_down_from_bounds(bounds_bottom, image_size=image_size)
    data = camera_config
    if isinstance(data, Mapping) and "cameras" in data:
        data = data["cameras"]
    if isinstance(data, Mapping) and "projection" not in data:
        data = data.get(sample_id, data.get(scene_name))
    if data is None:
        raise SampleFailure("missing_camera", f"No camera calibration for {sample_id}")
    return CameraModel.from_dict(data)


def _find_target(objects: Sequence[Any], sample: Mapping[str, Any]) -> Any:
    removed = sample.get("removed_object", {})
    instance_id = removed.get("instance_id")
    jid = removed.get("jid")
    if isinstance(instance_id, int):
        for obj in objects:
            if obj.instance_id == instance_id:
                return obj
    for obj in objects:
        if jid and obj.jid == jid:
            return obj
    raise SampleFailure("missing_target_mesh", "Removed target object was not loaded from the source scene")


def _load_scene_data(scene_json_dir: Path, scene_name: str) -> Dict[str, Any]:
    path = scene_json_dir / f"{scene_name}.json"
    if not path.exists():
        raise SampleFailure("missing_scene_json", f"Scene JSON does not exist: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _export_physics_handoff(
    work_dir: Path,
    sample_id: str,
    scene_mesh: trimesh.Trimesh,
    prepared_target: trimesh.Trimesh,
    release_bottom_world_xyz: Sequence[float],
    original_to_prepared: np.ndarray,
    args: argparse.Namespace,
) -> Path:
    sample_dir = work_dir / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    geometry_path = sample_dir / "geometry.npz"
    np.savez_compressed(
        geometry_path,
        scene_vertices=np.asarray(scene_mesh.vertices, dtype=np.float32),
        scene_faces=np.asarray(scene_mesh.faces, dtype=np.int32),
        target_vertices=np.asarray(prepared_target.vertices, dtype=np.float32),
        target_faces=np.asarray(prepared_target.faces, dtype=np.int32),
        original_to_prepared=np.asarray(original_to_prepared, dtype=np.float64),
    )
    manifest = {
        "sample_id": sample_id,
        "geometry_npz": geometry_path.name,
        "result_json": "isaac_result.json",
        "release_bottom_world_xyz": [float(v) for v in release_bottom_world_xyz],
        "simulation_seconds": args.simulation_seconds,
        "physics_hz": args.physics_hz,
        "stable_frames": args.stable_frames,
        "support_frames": args.support_frames,
        "linear_velocity_threshold": args.linear_velocity_threshold,
        "angular_velocity_threshold": args.angular_velocity_threshold,
        "support_normal_min_dot": args.support_normal_min_dot,
        "support_contact_height_tolerance": args.support_contact_height_tolerance,
        "penetration_threshold": args.penetration_threshold,
        "tilt_threshold_degrees": args.tilt_threshold_degrees,
        "max_horizontal_displacement": args.max_horizontal_displacement,
        "mass": args.mass,
    }
    if args.save_physics_usd:
        manifest["debug_usd"] = "settled.usda"
    manifest_path = sample_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return manifest_path


def _run_isaac(manifest_path: Path, args: argparse.Namespace) -> Dict[str, Any]:
    if not args.isaac_python:
        raise SampleFailure("missing_isaac_python", "--isaac_python is required for the Isaac backend")
    isaac_python = Path(args.isaac_python)
    if not isaac_python.exists():
        raise SampleFailure("missing_isaac_python", f"Isaac Python does not exist: {isaac_python}")
    script_path = Path(__file__).with_name("run_isaac_settle.py")
    command = [str(isaac_python), str(script_path), "--manifest", str(manifest_path)]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=args.physics_timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise SampleFailure(
            "isaac_timeout",
            f"Isaac Sim exceeded the {args.physics_timeout:.1f}s worker timeout",
        ) from exc
    if completed.returncode != 0:
        tail = (completed.stderr or completed.stdout or "")[-4000:]
        raise SampleFailure("isaac_error", f"Isaac Sim exited with {completed.returncode}: {tail}")
    result_text = json.loads(manifest_path.read_text(encoding="utf-8"))["result_json"]
    result_path = Path(result_text)
    if not result_path.is_absolute():
        result_path = manifest_path.parent / result_path
    if not result_path.exists():
        raise SampleFailure("missing_isaac_result", f"Isaac Sim did not create {result_path}")
    with open(result_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_output(path: Path, config: Dict[str, Any], results: Sequence[Dict[str, Any]]) -> None:
    counts = Counter(result.get("status", "unknown") for result in results)
    failures = Counter(
        result.get("failure_code")
        for result in results
        if result.get("failure_code") is not None
    )
    payload = {
        "schema": "hap_place_results_v1",
        "config": config,
        "summary": {
            "processed": len(results),
            "status_counts": dict(counts),
            "failure_counts": dict(failures),
            "sim_ready": sum(result.get("sim_ready") is True for result in results),
        },
        "results": list(results),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
    temporary.replace(path)


def _process_sample(
    sample: Mapping[str, Any],
    args: argparse.Namespace,
    model: PlacementHeatmap,
    device: torch.device,
    builder: SceneBuilder,
    scene_json_dir: Path,
    pose_records: Mapping[str, Any],
    camera_config: Optional[Any],
    work_dir: Path,
) -> Dict[str, Any]:
    sample_id = str(sample["sample_id"])
    started = time.perf_counter()
    scene_name = str(sample.get("scene_name", ""))
    if not scene_name:
        raise SampleFailure("missing_scene_name", "Sample has no scene_name")

    pose = lookup_scenerevis_pose(pose_records, sample_id)
    if pose is None and args.allow_metadata_pose:
        pose = _metadata_pose(sample)
    if pose is None:
        raise SampleFailure(
            "missing_scenerevis_pose",
            "No SceneReVis rotation and scale/size were found for this sample",
        )

    data_root = Path(args.data_dir)
    sample_dir = data_root / sample["scene_dir"]
    room_path = sample_dir / sample["plane_image_path"]
    object_path = sample_dir / sample["object_image_path"]
    if not room_path.exists() or not object_path.exists():
        raise SampleFailure("missing_input_image", f"Missing room or target image for {sample_id}")

    heatmap_started = time.perf_counter()
    heatmap, image_size = _predict_heatmap(
        model=model,
        room_path=room_path,
        object_path=object_path,
        description=str(sample.get("object_desc", "")),
        device=device,
    )
    heatmap_seconds = time.perf_counter() - heatmap_started

    scene_data = _load_scene_data(scene_json_dir, scene_name)
    bounds_bottom, bounds_top = _extract_bounds(scene_data)
    scene, objects = builder.build_scene(scene_data)
    expected_objects = builder.extract_objects(scene_data)
    if len(objects) != len(expected_objects):
        raise SampleFailure(
            "missing_scene_mesh_assets",
            f"Loaded {len(objects)} of {len(expected_objects)} object meshes for {scene_name}",
        )
    target = _find_target(objects, sample)
    if target.geom_name not in scene.geometry:
        raise SampleFailure("missing_target_geometry", f"Scene does not contain {target.geom_name}")
    scene.delete_geometry(target.geom_name)
    _add_room_walls(scene, bounds_bottom, bounds_top)

    prepared = prepare_target_mesh(target.mesh, pose)
    grid_spec = VoxelGridSpec.from_room_bounds(
        bounds_bottom=bounds_bottom,
        bounds_top=bounds_top,
        resolution=args.voxel_resolution,
    )
    voxel_started = time.perf_counter()
    occupancy, voxel_stats = voxelize_scene(scene, grid_spec)
    if voxel_stats["failures"]:
        raise SampleFailure(
            "scene_voxelization_failed",
            f"Failed to voxelize {len(voxel_stats['failures'])} scene meshes",
        )
    voxel_stats.update(occupancy.mark_outside_room(bounds_bottom, bounds_top))
    target_kernel = voxelize_target_kernel(
        prepared.mesh,
        pitch=grid_spec.pitch,
        minimum_release_height_voxels=args.minimum_release_height_voxels,
    )
    voxel_seconds = time.perf_counter() - voxel_started

    scene_mesh = concatenate_scene_mesh(scene)
    projector = FirstHitProjector(scene_mesh=scene_mesh, occupancy=occupancy)
    camera = _camera_for_sample(
        camera_config=camera_config,
        scene_name=scene_name,
        sample_id=sample_id,
        bounds_bottom=bounds_bottom,
        image_size=image_size,
    )
    search_started = time.perf_counter()
    search = score_ordered_release_search(
        heatmap=heatmap,
        camera=camera,
        projector=projector,
        target_kernel=target_kernel,
        max_candidates=args.max_candidates,
        minimum_score=args.minimum_heatmap_score,
    )
    search_seconds = time.perf_counter() - search_started
    if search is None:
        raise SampleFailure("no_feasible_release", "No score-ordered candidate admitted a collision-free release")

    release_transform = compose_release_transform(
        release_bottom_world_xyz=search.release_bottom_world_xyz,
        original_to_prepared=prepared.original_to_prepared,
    )
    manifest_path = _export_physics_handoff(
        work_dir=work_dir,
        sample_id=sample_id,
        scene_mesh=scene_mesh,
        prepared_target=prepared.mesh,
        release_bottom_world_xyz=search.release_bottom_world_xyz,
        original_to_prepared=prepared.original_to_prepared,
        args=args,
    )

    physics = None
    status = "release_ready"
    sim_ready = None
    failure_code = None
    if args.physics_backend == "isaac":
        physics = _run_isaac(manifest_path, args)
        sim_ready = bool(physics.get("sim_ready"))
        status = "simulator_ready" if sim_ready else "physics_failed"
        failure_code = None if sim_ready else str(physics.get("failure_code") or "physics_failed")

    final_transform = release_transform
    pose_stage = "release"
    if physics and physics.get("final_transform_original_to_world") is not None:
        final_transform = np.asarray(physics["final_transform_original_to_world"], dtype=np.float64)
        pose_stage = "settled"
    decomposed_pose = decompose_transform(final_transform)
    removed = sample.get("removed_object", {})
    simulator_record = {
        "source_scene": scene_name,
        "sample_id": sample_id,
        "pose_stage": pose_stage,
        "sim_ready": sim_ready,
        "target_object": {
            "asset_id": target.model_id,
            "source_jid": removed.get("jid", target.jid),
            "desc": removed.get("desc", target.desc),
            "position_xyz": decomposed_pose["position_xyz"],
            "rotation_xyzw": decomposed_pose["rotation_xyzw"],
            "scale_xyz": decomposed_pose["scale_xyz"],
            "local_size_xyz": prepared.mesh.extents.tolist(),
            "original_to_world": decomposed_pose["transform"],
        },
    }

    return {
        "sample_id": sample_id,
        "scene_name": scene_name,
        "status": status,
        "failure_code": failure_code,
        "sim_ready": sim_ready,
        "valid_output": True,
        "initial_collision_free": True,
        "scenerevis_pose": pose.to_dict(),
        "camera": camera.to_dict(),
        "voxel_grid": grid_spec.to_dict(),
        "voxelization": {
            **voxel_stats,
            "target_voxels": int(len(target_kernel.offsets_xyz)),
            "minimum_release_height_voxels": args.minimum_release_height_voxels,
        },
        "release_search": search.to_dict(),
        "release_transform_original_to_world": release_transform.tolist(),
        "simulator_record": simulator_record,
        "physics_manifest": str(manifest_path),
        "physics": physics,
        "timing_seconds": {
            "heatmap": heatmap_seconds,
            "voxelization": voxel_seconds,
            "release_search": search_seconds,
            "total": time.perf_counter() - started,
        },
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    if args.voxel_resolution <= 0:
        raise ValueError("--voxel_resolution must be positive")
    if args.minimum_release_height_voxels < 0:
        raise ValueError("--minimum_release_height_voxels must be non-negative")
    if args.max_candidates <= 0:
        raise ValueError("--max_candidates must be positive")
    if args.physics_timeout <= 0 or args.simulation_seconds <= 0 or args.physics_hz <= 0:
        raise ValueError("Physics timeout, simulation duration, and frequency must be positive")
    if args.stable_frames <= 0 or args.support_frames <= 0:
        raise ValueError("--stable_frames and --support_frames must be positive")
    if args.mass <= 0:
        raise ValueError("--mass must be positive")
    if args.penetration_threshold < 0:
        raise ValueError("--penetration_threshold must be non-negative")
    if not 0.0 <= args.support_normal_min_dot <= 1.0:
        raise ValueError("--support_normal_min_dot must be in [0, 1]")
    if args.support_contact_height_tolerance <= 0:
        raise ValueError("--support_contact_height_tolerance must be positive")
    if args.tilt_threshold_degrees <= 0:
        raise ValueError("--tilt_threshold_degrees must be positive")
    if args.max_horizontal_displacement <= 0:
        raise ValueError("--max_horizontal_displacement must be positive")
    if args.physics_backend == "isaac" and not args.isaac_python:
        raise ValueError("--isaac_python is required when --physics_backend=isaac")
    if not args.scenerevis_results and not args.allow_metadata_pose:
        raise ValueError("--scenerevis_results is required unless --allow_metadata_pose is set")

    output_path = Path(args.output_json)
    work_dir = output_path.parent / f".{output_path.stem}_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    split_json = data_dir / args.split / f"{args.split}.json"
    with open(split_json, "r", encoding="utf-8") as handle:
        samples = json.load(handle)
    if args.sample_id:
        selected = set(args.sample_id)
        samples = [sample for sample in samples if sample.get("sample_id") in selected]
    else:
        samples = samples[max(0, args.start_index):]
        if args.num_samples > 0:
            samples = samples[:args.num_samples]

    pose_records = (
        load_scenerevis_pose_records(args.scenerevis_results)
        if args.scenerevis_results
        else {}
    )
    camera_config = _load_camera_config(args.camera_json)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", device)
    model, model_config = _load_heatmap_model(Path(args.checkpoint), device)
    builder = SceneBuilder(Path(args.model_dir))
    scene_json_dir = Path(args.scene_json_dir)

    config = {
        "data_dir": str(data_dir),
        "split": args.split,
        "checkpoint": args.checkpoint,
        "heatmap_model": model_config,
        "scene_json_dir": args.scene_json_dir,
        "model_dir": args.model_dir,
        "scenerevis_results": args.scenerevis_results,
        "allow_metadata_pose": args.allow_metadata_pose,
        "camera_json": args.camera_json,
        "voxel_resolution": args.voxel_resolution,
        "minimum_release_height_voxels": args.minimum_release_height_voxels,
        "max_candidates": args.max_candidates,
        "minimum_heatmap_score": args.minimum_heatmap_score,
        "physics_backend": args.physics_backend,
        "simulation_seconds": args.simulation_seconds,
        "physics_hz": args.physics_hz,
        "stable_frames": args.stable_frames,
        "support_frames": args.support_frames,
        "linear_velocity_threshold": args.linear_velocity_threshold,
        "angular_velocity_threshold": args.angular_velocity_threshold,
        "support_normal_min_dot": args.support_normal_min_dot,
        "support_contact_height_tolerance": args.support_contact_height_tolerance,
        "penetration_threshold": args.penetration_threshold,
        "tilt_threshold_degrees": args.tilt_threshold_degrees,
        "max_horizontal_displacement": args.max_horizontal_displacement,
        "mass": args.mass,
        "save_physics_usd": args.save_physics_usd,
    }
    results = []
    for index, sample in enumerate(samples, start=1):
        sample_id = str(sample.get("sample_id", f"index_{index}"))
        logging.info("[%d/%d] %s", index, len(samples), sample_id)
        try:
            result = _process_sample(
                sample=sample,
                args=args,
                model=model,
                device=device,
                builder=builder,
                scene_json_dir=scene_json_dir,
                pose_records=pose_records,
                camera_config=camera_config,
                work_dir=work_dir,
            )
            logging.info(
                "  %s: score=%.4f, candidates=%d, releases=%d",
                result["status"],
                result["release_search"]["heatmap_score"],
                result["release_search"]["tested_candidates"],
                result["release_search"]["tested_release_poses"],
            )
        except SampleFailure as exc:
            logging.warning("  %s: %s", exc.code, exc)
            result = {
                "sample_id": sample_id,
                "scene_name": sample.get("scene_name"),
                "status": "failed",
                "failure_code": exc.code,
                "error": str(exc),
                "sim_ready": False,
                "valid_output": False,
            }
        except Exception as exc:
            logging.exception("  unexpected error")
            result = {
                "sample_id": sample_id,
                "scene_name": sample.get("scene_name"),
                "status": "failed",
                "failure_code": "unexpected_error",
                "error": str(exc),
                "sim_ready": False,
                "valid_output": False,
            }
        results.append(result)
        _write_output(output_path, config, results)

    logging.info("Wrote %d results to %s", len(results), output_path)


if __name__ == "__main__":
    main()
