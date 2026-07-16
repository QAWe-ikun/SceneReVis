"""Drop and settle HAP-Place release poses in Isaac Sim 6.x.

Run this file with Isaac Sim's ``python.sh``, not the training environment.
Multiple manifests can be processed in one process to amortize Isaac startup.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import traceback
from collections import deque
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


PROJECT_TO_ISAAC_3 = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)
PROJECT_TO_ISAAC_4 = np.eye(4, dtype=np.float64)
PROJECT_TO_ISAAC_4[:3, :3] = PROJECT_TO_ISAAC_3
ISAAC_TO_PROJECT_4 = np.linalg.inv(PROJECT_TO_ISAAC_4)
ISAAC_WORLD_UP = np.array([0.0, 0.0, 1.0], dtype=np.float64)


def _project_points_to_isaac(points: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=np.float64) @ PROJECT_TO_ISAAC_3.T


def _isaac_pose_to_project(transform_isaac: np.ndarray) -> np.ndarray:
    return ISAAC_TO_PROJECT_4 @ transform_isaac @ PROJECT_TO_ISAAC_4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Isaac Sim drop-and-settle worker")
    parser.add_argument("--manifest", action="append", default=[])
    parser.add_argument(
        "--manifest_list",
        default=None,
        help="JSON array or newline-delimited file containing manifest paths",
    )
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _write_result(path: Path, result: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
    temporary.replace(path)


def _load_manifest_paths(args: argparse.Namespace) -> List[Path]:
    paths = [Path(value) for value in args.manifest]
    if args.manifest_list:
        list_path = Path(args.manifest_list)
        content = list_path.read_text(encoding="utf-8")
        try:
            values = json.loads(content)
        except json.JSONDecodeError:
            values = [line.strip() for line in content.splitlines() if line.strip()]
        if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
            raise ValueError("--manifest_list must contain a JSON string array or one path per line")
        paths.extend(Path(value) for value in values)
    if not paths:
        raise ValueError("At least one --manifest or --manifest_list entry is required")
    return paths


def _create_triangle_mesh(stage, path: str, vertices: np.ndarray, faces: np.ndarray):
    from pxr import UsdGeom, Vt

    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(vertices, dtype=np.float32)))
    mesh.CreateFaceVertexCountsAttr([3] * len(faces))
    mesh.CreateFaceVertexIndicesAttr(np.asarray(faces, dtype=np.int32).reshape(-1).tolist())
    mesh.CreateSubdivisionSchemeAttr("none")
    return mesh


def _pose_from_xform(xformable) -> Tuple[np.ndarray, np.ndarray]:
    from pxr import Usd

    matrix = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    translation = matrix.ExtractTranslation()
    quaternion = matrix.ExtractRotationQuat()
    imaginary = quaternion.GetImaginary()
    position = np.array([translation[0], translation[1], translation[2]], dtype=np.float64)
    quat_wxyz = np.array(
        [quaternion.GetReal(), imaginary[0], imaginary[1], imaginary[2]],
        dtype=np.float64,
    )
    quat_wxyz /= max(np.linalg.norm(quat_wxyz), 1e-12)
    return position, quat_wxyz


def _matrix_from_pose(position: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quat_wxyz, dtype=np.float64)
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = np.asarray(position, dtype=np.float64)
    return matrix


def _box_corners(vertices: np.ndarray) -> np.ndarray:
    minimum = np.min(vertices, axis=0)
    maximum = np.max(vertices, axis=0)
    return np.array(
        [
            [x, y, z]
            for x in (minimum[0], maximum[0])
            for y in (minimum[1], maximum[1])
            for z in (minimum[2], maximum[2])
        ],
        dtype=np.float64,
    )


def _target_bottom_coordinate(
    position: np.ndarray,
    quaternion_wxyz: np.ndarray,
    local_bounds_corners: np.ndarray,
) -> float:
    transform = _matrix_from_pose(position, quaternion_wxyz)
    world = local_bounds_corners @ transform[:3, :3].T + transform[:3, 3]
    return float(np.min(world @ ISAAC_WORLD_UP))


def _extract_contact_records(contact_headers: Iterable[Any], contact_data: Sequence[Any]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for header in contact_headers:
        offset = int(header.contact_data_offset)
        count = int(header.num_contact_data)
        for index in range(offset, offset + count):
            contact = contact_data[index]
            records.append(
                {
                    "position": np.asarray(contact.position, dtype=np.float64),
                    "normal": np.asarray(contact.normal, dtype=np.float64),
                    "impulse": np.asarray(contact.impulse, dtype=np.float64),
                    "separation": float(contact.separation),
                }
            )
    return records


def _contact_frame_metrics(
    records: Sequence[Mapping[str, Any]],
    target_bottom_coordinate: float,
    support_normal_min_dot: float,
    support_contact_height_tolerance: float,
    dt: float,
) -> Dict[str, Any]:
    max_penetration = 0.0
    max_force = 0.0
    support_contact = False
    for record in records:
        normal = np.asarray(record["normal"], dtype=np.float64)
        position = np.asarray(record["position"], dtype=np.float64)
        impulse = np.asarray(record["impulse"], dtype=np.float64)
        normal_norm = float(np.linalg.norm(normal))
        vertical_dot = abs(float(np.dot(normal, ISAAC_WORLD_UP))) / max(normal_norm, 1e-12)
        contact_height = float(np.dot(position, ISAAC_WORLD_UP))
        near_bottom = contact_height <= target_bottom_coordinate + support_contact_height_tolerance
        support_contact = support_contact or (
            vertical_dot >= support_normal_min_dot and near_bottom
        )
        max_penetration = max(max_penetration, max(0.0, -float(record["separation"])))
        max_force = max(max_force, float(np.linalg.norm(impulse)) / dt)
    return {
        "in_contact": bool(records),
        "support_contact": support_contact,
        "max_penetration": max_penetration,
        "max_force": max_force,
        "contact_points": len(records),
    }


def _update_support_latch(
    support_contact: bool,
    low_motion: bool,
    target_bottom_coordinate: float,
    support_latched: bool,
    latched_bottom_coordinate: float | None,
    height_tolerance: float,
) -> Tuple[bool, float | None]:
    if support_contact:
        return True, target_bottom_coordinate
    if not support_latched or latched_bottom_coordinate is None:
        return False, None
    if not low_motion or abs(target_bottom_coordinate - latched_bottom_coordinate) > height_tolerance:
        return False, None
    return True, latched_bottom_coordinate


def _tilt_degrees(quaternion_wxyz: np.ndarray) -> float:
    rotation = _matrix_from_pose(np.zeros(3, dtype=np.float64), quaternion_wxyz)[:3, :3]
    prepared_up_world = rotation @ ISAAC_WORLD_UP
    cosine = float(np.clip(np.dot(prepared_up_world, ISAAC_WORLD_UP), -1.0, 1.0))
    return float(math.degrees(math.acos(cosine)))


def _physics_failure_code(metrics: Mapping[str, Any], manifest: Mapping[str, Any]) -> str | None:
    if not metrics["stable"]:
        return "unstable_timeout"
    if not metrics["support_valid"]:
        return "support_invalid"
    if metrics["max_penetration"] > float(manifest["penetration_threshold"]):
        return "penetration_exceeded"
    if metrics["tilt_degrees"] > float(manifest["tilt_threshold_degrees"]):
        return "tilt_exceeded"
    if metrics["horizontal_displacement"] > float(manifest["max_horizontal_displacement"]):
        return "displacement_exceeded"
    return None


def _validate_manifest(manifest: Mapping[str, Any]) -> None:
    positive = (
        "simulation_seconds",
        "physics_hz",
        "stable_frames",
        "support_frames",
        "linear_velocity_threshold",
        "angular_velocity_threshold",
        "support_contact_height_tolerance",
        "tilt_threshold_degrees",
        "max_horizontal_displacement",
        "mass",
    )
    for name in positive:
        if float(manifest[name]) <= 0:
            raise ValueError(f"{name} must be positive")
    if float(manifest["penetration_threshold"]) < 0:
        raise ValueError("penetration_threshold must be non-negative")
    normal_dot = float(manifest["support_normal_min_dot"])
    if not 0.0 <= normal_dot <= 1.0:
        raise ValueError("support_normal_min_dot must be in [0, 1]")


def _resolve_manifest_path(manifest_path: Path, path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else manifest_path.parent / path


def _run_manifest(
    manifest_path: Path,
    simulation_app: Any,
    progress_callback: Optional[Callable[[], None]] = None,
) -> Dict[str, Any]:
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    result_path = _resolve_manifest_path(manifest_path, manifest["result_json"])
    result: Dict[str, Any] = {
        "schema": "hap_place_isaac_result_v2",
        "sample_id": manifest.get("sample_id"),
        "stable": False,
        "support_valid": False,
        "sim_ready": False,
        "failure_code": "isaac_error",
        "max_penetration": None,
    }
    started = time.perf_counter()
    simulation = None
    try:
        _validate_manifest(manifest)

        import omni.usd
        from isaacsim.core.api import SimulationContext
        from omni.physx import get_physx_simulation_interface
        from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

        geometry_path = _resolve_manifest_path(manifest_path, manifest["geometry_npz"])
        geometry = np.load(geometry_path)
        scene_vertices_project = np.asarray(geometry["scene_vertices"], dtype=np.float64)
        scene_faces = np.asarray(geometry["scene_faces"], dtype=np.int64)
        target_vertices_project = np.asarray(geometry["target_vertices"], dtype=np.float64)
        target_faces = np.asarray(geometry["target_faces"], dtype=np.int64)
        original_to_prepared = np.asarray(geometry["original_to_prepared"], dtype=np.float64)
        scene_vertices = _project_points_to_isaac(scene_vertices_project)
        target_vertices = _project_points_to_isaac(target_vertices_project)
        target_bounds_corners = _box_corners(target_vertices)

        omni.usd.get_context().new_stage()
        stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.Xform.Define(stage, "/World")

        physics_scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
        physics_scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
        physics_scene.CreateGravityMagnitudeAttr(9.81)

        scene_mesh = _create_triangle_mesh(stage, "/World/StaticScene", scene_vertices, scene_faces)
        UsdPhysics.CollisionAPI.Apply(scene_mesh.GetPrim()).CreateCollisionEnabledAttr(True)
        UsdPhysics.MeshCollisionAPI.Apply(scene_mesh.GetPrim()).CreateApproximationAttr().Set("none")

        target_xform = UsdGeom.Xform.Define(stage, "/World/Target")
        release_project = np.asarray(manifest["release_bottom_world_xyz"], dtype=np.float64)
        release = PROJECT_TO_ISAAC_3 @ release_project
        target_xform.AddTranslateOp().Set(Gf.Vec3d(*release.tolist()))
        target_mesh = _create_triangle_mesh(
            stage,
            "/World/Target/Mesh",
            target_vertices,
            target_faces,
        )
        UsdPhysics.CollisionAPI.Apply(target_mesh.GetPrim()).CreateCollisionEnabledAttr(True)
        UsdPhysics.MeshCollisionAPI.Apply(target_mesh.GetPrim()).CreateApproximationAttr().Set(
            "convexDecomposition"
        )

        rigid_body = UsdPhysics.RigidBodyAPI.Apply(target_xform.GetPrim())
        rigid_body.CreateRigidBodyEnabledAttr(True)
        rigid_body.CreateKinematicEnabledAttr(False)
        rigid_body.CreateVelocityAttr(Gf.Vec3f(0.0, 0.0, 0.0))
        rigid_body.CreateAngularVelocityAttr(Gf.Vec3f(0.0, 0.0, 0.0))
        UsdPhysics.MassAPI.Apply(target_xform.GetPrim()).CreateMassAttr(float(manifest["mass"]))
        PhysxSchema.PhysxContactReportAPI.Apply(target_xform.GetPrim()).CreateThresholdAttr(0.0)

        physics_hz = float(manifest["physics_hz"])
        dt = 1.0 / physics_hz
        max_steps = int(math.ceil(float(manifest["simulation_seconds"]) * physics_hz))
        stable_frames_required = int(manifest["stable_frames"])
        support_frames_required = int(manifest["support_frames"])
        linear_threshold = float(manifest["linear_velocity_threshold"])
        angular_threshold = float(manifest["angular_velocity_threshold"])
        support_normal_min_dot = float(manifest["support_normal_min_dot"])
        support_height_tolerance = float(manifest["support_contact_height_tolerance"])

        simulation = SimulationContext(
            physics_dt=dt,
            rendering_dt=dt,
            stage_units_in_meters=1.0,
            physics_prim_path="/World/PhysicsScene",
            set_defaults=False,
        )
        simulation.initialize_physics()
        simulation.play()
        simulation_app.update()
        if progress_callback is not None:
            progress_callback()
        physx_simulation = get_physx_simulation_interface()

        previous_position, previous_quaternion = _pose_from_xform(target_xform)
        stable_frames = 0
        support_frames = 0
        contact_frames = 0
        support_contact_frames = 0
        support_latched_frames = 0
        support_latched = False
        latched_support_bottom_coordinate: float | None = None
        max_contact_force = 0.0
        peak_penetration = 0.0
        final_linear_speed = float("inf")
        final_angular_speed = float("inf")
        final_frame_metrics: Dict[str, Any] = {
            "max_penetration": 0.0,
            "contact_points": 0,
        }
        stable_window_penetrations: deque[float] = deque(
            maxlen=max(stable_frames_required, support_frames_required)
        )
        steps_run = 0

        for step in range(max_steps):
            simulation.step(render=False)
            if progress_callback is not None:
                progress_callback()
            steps_run = step + 1
            position, quaternion = _pose_from_xform(target_xform)
            final_linear_speed = float(np.linalg.norm(position - previous_position) / dt)
            quat_dot = float(np.clip(abs(np.dot(quaternion, previous_quaternion)), 0.0, 1.0))
            final_angular_speed = float(2.0 * math.acos(quat_dot) / dt)

            contact_headers, contact_data = get_physx_simulation_interface().get_contact_report()
            records = _extract_contact_records(contact_headers, contact_data)
            bottom_coordinate = _target_bottom_coordinate(
                position,
                quaternion,
                target_bounds_corners,
            )
            final_frame_metrics = _contact_frame_metrics(
                records=records,
                target_bottom_coordinate=bottom_coordinate,
                support_normal_min_dot=support_normal_min_dot,
                support_contact_height_tolerance=support_height_tolerance,
                dt=dt,
            )
            in_contact = bool(final_frame_metrics["in_contact"])
            support_contact = bool(final_frame_metrics["support_contact"])
            if in_contact:
                contact_frames += 1
            if support_contact:
                support_contact_frames += 1
            max_contact_force = max(max_contact_force, float(final_frame_metrics["max_force"]))
            peak_penetration = max(
                peak_penetration,
                float(final_frame_metrics["max_penetration"]),
            )

            low_motion = (
                final_linear_speed <= linear_threshold
                and final_angular_speed <= angular_threshold
            )
            support_latched, latched_support_bottom_coordinate = _update_support_latch(
                support_contact=support_contact,
                low_motion=low_motion,
                target_bottom_coordinate=bottom_coordinate,
                support_latched=support_latched,
                latched_bottom_coordinate=latched_support_bottom_coordinate,
                height_tolerance=support_height_tolerance,
            )
            effective_support = support_contact or support_latched
            effective_contact = in_contact or effective_support
            if effective_support and not support_contact:
                support_latched_frames += 1

            if low_motion and effective_contact:
                stable_frames += 1
                stable_window_penetrations.append(float(final_frame_metrics["max_penetration"]))
            else:
                stable_frames = 0
                stable_window_penetrations.clear()
            if low_motion and effective_support:
                support_frames += 1
            else:
                support_frames = 0

            previous_position = position
            previous_quaternion = quaternion
            if stable_frames >= stable_frames_required and support_frames >= support_frames_required:
                break

        final_position, final_quaternion = _pose_from_xform(target_xform)
        final_prepared_to_isaac = _matrix_from_pose(final_position, final_quaternion)
        final_prepared_to_world = _isaac_pose_to_project(final_prepared_to_isaac)
        final_original_to_world = final_prepared_to_world @ original_to_prepared
        stable = stable_frames >= stable_frames_required
        support_valid = support_frames >= support_frames_required
        max_penetration = max(stable_window_penetrations, default=float(final_frame_metrics["max_penetration"]))
        horizontal_displacement = float(np.linalg.norm((final_position - release)[[0, 1]]))
        final_position_project = final_prepared_to_world[:3, 3]
        displacement_xyz = final_position_project - release_project
        tilt_degrees = _tilt_degrees(final_quaternion)

        metrics = {
            "stable": stable,
            "support_valid": support_valid,
            "max_penetration": float(max_penetration),
            "tilt_degrees": tilt_degrees,
            "horizontal_displacement": horizontal_displacement,
        }
        failure_code = _physics_failure_code(metrics, manifest)
        result.update(
            {
                **metrics,
                "sim_ready": failure_code is None,
                "failure_code": failure_code,
                "penetration_measurement": "physx_contact_separation",
                "peak_penetration_during_settle": peak_penetration,
                "steps_run": steps_run,
                "simulation_seconds": steps_run * dt,
                "stable_frames": stable_frames,
                "support_frames": support_frames,
                "contact_frames": contact_frames,
                "support_contact_frames": support_contact_frames,
                "support_latched_frames": support_latched_frames,
                "support_measurement": "physx_contact_with_sleep_latch",
                "final_contact_points": int(final_frame_metrics["contact_points"]),
                "max_contact_force": max_contact_force,
                "final_linear_speed": final_linear_speed,
                "final_angular_speed": final_angular_speed,
                "coordinate_system": {
                    "simulation": "isaac_z_up",
                    "output": "scenerevis_y_up",
                    "project_to_isaac": PROJECT_TO_ISAAC_4.tolist(),
                },
                "release_position_prepared_xyz": release_project.tolist(),
                "final_position_prepared_xyz": final_position_project.tolist(),
                "settle_displacement_xyz": displacement_xyz.tolist(),
                "release_position_isaac_xyz": release.tolist(),
                "final_position_isaac_xyz": final_position.tolist(),
                "final_quaternion_prepared_isaac_wxyz": final_quaternion.tolist(),
                "final_transform_prepared_to_isaac": final_prepared_to_isaac.tolist(),
                "final_transform_prepared_to_world": final_prepared_to_world.tolist(),
                "final_transform_original_to_world": final_original_to_world.tolist(),
            }
        )

        debug_usd = manifest.get("debug_usd")
        if debug_usd:
            debug_path = _resolve_manifest_path(manifest_path, debug_usd)
            debug_path.parent.mkdir(parents=True, exist_ok=True)
            stage.GetRootLayer().Export(str(debug_path))
            result["debug_usd"] = str(debug_path)
    except Exception as exc:
        result.update(
            {
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "failure_code": "isaac_error",
            }
        )
    finally:
        if simulation is not None:
            try:
                simulation.stop()
            except Exception:
                pass
        try:
            from isaacsim.core.api import SimulationContext

            SimulationContext.clear_instance()
        except Exception:
            pass
        result["worker_wall_seconds"] = time.perf_counter() - started
        _write_result(result_path, result)
    return result


def main() -> None:
    args = parse_args()
    manifest_paths = _load_manifest_paths(args)

    import isaacsim
    from isaacsim.simulation_app import SimulationApp

    del isaacsim
    simulation_app = SimulationApp({"headless": args.headless})
    failed = 0
    try:
        for manifest_path in manifest_paths:
            result = _run_manifest(manifest_path, simulation_app)
            print(json.dumps(result, ensure_ascii=False, allow_nan=False))
            if result.get("error"):
                failed += 1
    finally:
        simulation_app.close()
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
