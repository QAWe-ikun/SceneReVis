"""Offline tests for the Isaac Sim settle protocol and result merge."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from script.pretreatment.run_isaac_batch import _merge_physics
from script.pretreatment.run_isaac_settle import (
    ISAAC_TO_PROJECT_4,
    PROJECT_TO_ISAAC_4,
    _contact_frame_metrics,
    _extract_contact_records,
    _isaac_pose_to_project,
    _physics_failure_code,
    _project_points_to_isaac,
    _tilt_degrees,
    _update_support_latch,
)


def _manifest_thresholds():
    return {
        "penetration_threshold": 0.005,
        "tilt_threshold_degrees": 15.0,
        "max_horizontal_displacement": 0.1,
    }


def test_contact_report_separation_and_bottom_support_metrics():
    header = SimpleNamespace(contact_data_offset=0, num_contact_data=2)
    contacts = [
        SimpleNamespace(
            position=(0.0, 0.0, 0.01),
            normal=(0.0, 0.0, 1.0),
            impulse=(0.0, 0.0, 0.2),
            separation=-0.003,
        ),
        SimpleNamespace(
            position=(0.5, 0.5, 0.0),
            normal=(1.0, 0.0, 0.0),
            impulse=(0.1, 0.0, 0.0),
            separation=-0.001,
        ),
    ]

    records = _extract_contact_records([header], contacts)
    metrics = _contact_frame_metrics(
        records=records,
        target_bottom_coordinate=0.0,
        support_normal_min_dot=0.7,
        support_contact_height_tolerance=0.03,
        dt=0.01,
    )

    assert metrics["in_contact"]
    assert metrics["support_contact"]
    assert np.isclose(metrics["max_penetration"], 0.003)
    assert np.isclose(metrics["max_force"], 20.0)
    assert metrics["contact_points"] == 2


def test_side_contact_does_not_count_as_support():
    metrics = _contact_frame_metrics(
        records=[
            {
                "position": np.array([0.5, 0.2, 0.0]),
                "normal": np.array([1.0, 0.0, 0.0]),
                "impulse": np.array([0.1, 0.0, 0.0]),
                "separation": -0.002,
            }
        ],
        target_bottom_coordinate=0.0,
        support_normal_min_dot=0.7,
        support_contact_height_tolerance=0.03,
        dt=0.01,
    )

    assert metrics["in_contact"]
    assert not metrics["support_contact"]


def test_support_contact_latches_while_sleeping_at_the_same_height():
    latched, bottom_y = _update_support_latch(
        support_contact=True,
        low_motion=True,
        target_bottom_coordinate=0.0,
        support_latched=False,
        latched_bottom_coordinate=None,
        height_tolerance=0.03,
    )
    assert latched

    latched, bottom_y = _update_support_latch(
        support_contact=False,
        low_motion=True,
        target_bottom_coordinate=0.001,
        support_latched=latched,
        latched_bottom_coordinate=bottom_y,
        height_tolerance=0.03,
    )
    assert latched
    assert bottom_y == 0.0


def test_support_latch_clears_on_motion_or_height_change():
    assert _update_support_latch(False, False, 0.0, True, 0.0, 0.03) == (False, None)
    assert _update_support_latch(False, True, 0.04, True, 0.0, 0.03) == (False, None)


def test_tilt_is_measured_relative_to_prepared_up_axis():
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    rotation_x_30 = np.array(
        [np.cos(np.deg2rad(15.0)), np.sin(np.deg2rad(15.0)), 0.0, 0.0]
    )
    assert np.isclose(_tilt_degrees(identity), 0.0)
    assert np.isclose(_tilt_degrees(rotation_x_30), 30.0)


def test_project_y_up_maps_to_isaac_z_up_without_reflection():
    converted = _project_points_to_isaac(np.array([[1.0, 2.0, 3.0]]))
    assert np.allclose(converted, [[1.0, -3.0, 2.0]])
    assert np.isclose(np.linalg.det(PROJECT_TO_ISAAC_4[:3, :3]), 1.0)


def test_isaac_pose_basis_change_round_trips_to_project_coordinates():
    project_pose = np.eye(4)
    project_pose[:3, :3] = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ]
    )
    project_pose[:3, 3] = [1.0, 2.0, 3.0]
    isaac_pose = PROJECT_TO_ISAAC_4 @ project_pose @ ISAAC_TO_PROJECT_4
    assert np.allclose(_isaac_pose_to_project(isaac_pose), project_pose)


def test_physics_failure_codes_are_conservative_and_ordered():
    manifest = _manifest_thresholds()
    valid = {
        "stable": True,
        "support_valid": True,
        "max_penetration": 0.001,
        "tilt_degrees": 2.0,
        "horizontal_displacement": 0.01,
    }
    assert _physics_failure_code(valid, manifest) is None

    for field, value, expected in (
        ("stable", False, "unstable_timeout"),
        ("support_valid", False, "support_invalid"),
        ("max_penetration", 0.006, "penetration_exceeded"),
        ("tilt_degrees", 16.0, "tilt_exceeded"),
        ("horizontal_displacement", 0.11, "displacement_exceeded"),
    ):
        metrics = dict(valid)
        metrics[field] = value
        assert _physics_failure_code(metrics, manifest) == expected


def test_batch_merge_updates_settled_simulator_pose():
    result = {
        "status": "release_ready",
        "sim_ready": None,
        "simulator_record": {
            "pose_stage": "release",
            "sim_ready": None,
            "target_object": {},
        },
    }
    transform = np.eye(4)
    transform[:3, :3] = np.diag([2.0, 3.0, 4.0])
    transform[:3, 3] = [1.0, 2.0, 3.0]
    physics = {
        "sim_ready": True,
        "failure_code": None,
        "worker_wall_seconds": 1.5,
        "final_transform_original_to_world": transform.tolist(),
    }

    _merge_physics(result, physics)

    assert result["status"] == "simulator_ready"
    assert result["sim_ready"] is True
    assert result["simulator_record"]["pose_stage"] == "settled"
    assert result["simulator_record"]["target_object"]["position_xyz"] == [1.0, 2.0, 3.0]
    assert result["simulator_record"]["target_object"]["scale_xyz"] == [2.0, 3.0, 4.0]
