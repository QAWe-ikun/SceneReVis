"""Focused tests for the HAP-Place geometry and release-search contract."""

import json
import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.hap_place import (
    CameraModel,
    FirstHitProjector,
    PackedVoxelGrid,
    RayHit,
    SceneReVisPose,
    TargetVoxelKernel,
    VoxelGridSpec,
    conservative_voxelize_mesh,
    decompose_transform,
    parse_scenerevis_pose,
    prepare_target_mesh,
    score_ordered_release_search,
    voxelize_scene,
)


def test_packed_grid_uses_xyz_public_indices_and_zyx_storage():
    spec = VoxelGridSpec(origin_xyz=(0.0, 0.0, 0.0), shape_xyz=(130, 8, 6), pitch=0.1)
    grid = PackedVoxelGrid(spec)
    occupied = np.array([[0, 1, 2], [64, 3, 4], [129, 7, 5]], dtype=np.int64)
    assert grid.set_indices(occupied) == 3
    assert grid.contains_index([0, 1, 2])
    assert grid.contains_index([64, 3, 4])
    assert grid.contains_index([129, 7, 5])
    assert not grid.contains_index([63, 3, 4])
    assert grid.nbytes == 6 * 8 * 3 * 8


def test_room_prism_exterior_is_explicitly_occupied():
    spec = VoxelGridSpec(origin_xyz=(-1.0, -1.0, -1.0), shape_xyz=(4, 4, 4), pitch=1.0)
    grid = PackedVoxelGrid(spec)
    bottom = [[0, 0, 0], [2, 0, 0], [2, 0, 2], [0, 0, 2]]
    top = [[0, 2, 0], [2, 2, 0], [2, 2, 2], [0, 2, 2]]

    stats = grid.mark_outside_room(bottom, top)

    assert stats["inside_room_voxels"] == 8
    assert stats["outside_room_voxels"] == 56
    assert not grid.contains_index([1, 1, 1])
    assert not grid.contains_index([2, 2, 2])
    assert grid.contains_index([0, 1, 1])
    assert grid.contains_index([1, 0, 1])
    assert grid.contains_index([1, 1, 3])
    assert not grid.intersects_offsets([1, 1, 1], np.array([[0, 0, 0]]))
    assert grid.intersects_offsets([2, 1, 1], np.array([[1, 0, 0]]))


def test_room_exterior_mask_respects_concave_polygon_not_only_aabb():
    spec = VoxelGridSpec(origin_xyz=(0.0, 0.0, 0.0), shape_xyz=(4, 2, 4), pitch=1.0)
    grid = PackedVoxelGrid(spec)
    polygon_xz = [(0, 0), (3, 0), (3, 1), (1, 1), (1, 3), (0, 3)]
    bottom = [[x, 0, z] for x, z in polygon_xz]
    top = [[x, 2, z] for x, z in polygon_xz]

    grid.mark_outside_room(bottom, top)

    assert not grid.contains_index([0, 0, 2])
    assert not grid.contains_index([2, 1, 0])
    assert grid.contains_index([2, 0, 2])


def test_bottom_empty_layer_avoids_surface_voxel_overlap():
    spec = VoxelGridSpec(origin_xyz=(0.0, 0.0, 0.0), shape_xyz=(8, 8, 8), pitch=1.0)
    grid = PackedVoxelGrid(spec)
    surface = np.array([3, 2, 3], dtype=np.int64)
    grid.set_indices(surface.reshape(1, 3))

    no_clearance = np.array([[0, 0, 0]], dtype=np.int64)
    one_empty_layer = np.array([[0, 1, 0]], dtype=np.int64)
    assert grid.intersects_offsets(surface, no_clearance)
    assert not grid.intersects_offsets(surface, one_empty_layer)


def test_target_bitset_overlap_matches_sparse_offsets_across_word_boundary():
    spec = VoxelGridSpec(origin_xyz=(0.0, 0.0, 0.0), shape_xyz=(130, 8, 8), pitch=1.0)
    grid = PackedVoxelGrid(spec)
    grid.set_indices(np.array([[64, 3, 4], [129, 3, 4]], dtype=np.int64))
    kernel = TargetVoxelKernel(
        offsets_xyz=np.array([[0, 1, 0], [65, 1, 0]], dtype=np.int64),
        minimum_release_height_voxels=1,
        pitch=1.0,
    )

    for anchor in ([63, 2, 4], [64, 2, 4], [0, 2, 4]):
        assert grid.intersects_kernel(anchor, kernel) == grid.intersects_offsets(
            anchor,
            kernel.offsets_xyz,
        )


def test_conservative_voxelization_marks_triangle_box_overlap_without_vertex_hit():
    mesh = trimesh.Trimesh(
        vertices=np.array(
            [
                [0.1, 0.1, 0.5],
                [2.9, 0.1, 0.5],
                [0.1, 2.9, 0.5],
            ]
        ),
        faces=np.array([[0, 1, 2]]),
        process=False,
    )
    occupied, stats = conservative_voxelize_mesh(
        mesh=mesh,
        pitch=1.0,
        origin_xyz=(0.0, 0.0, 0.0),
        shape_xyz=(3, 3, 2),
        fill_watertight=False,
    )
    occupied_set = {tuple(index) for index in occupied.tolist()}
    assert (1, 1, 0) in occupied_set
    assert (2, 2, 0) not in occupied_set
    assert stats["triangle_count"] == 1
    assert stats["sat_candidate_tests"] > 0


def test_conservative_voxelization_includes_both_cells_at_shared_boundary():
    mesh = trimesh.Trimesh(
        vertices=np.array(
            [
                [1.0, 0.1, 0.1],
                [1.0, 0.9, 0.1],
                [1.0, 0.1, 0.9],
            ]
        ),
        faces=np.array([[0, 1, 2]]),
        process=False,
    )
    occupied, _ = conservative_voxelize_mesh(
        mesh=mesh,
        pitch=1.0,
        origin_xyz=(0.0, 0.0, 0.0),
        shape_xyz=(2, 1, 1),
        fill_watertight=False,
    )
    occupied_set = {tuple(index) for index in occupied.tolist()}
    assert occupied_set == {(0, 0, 0), (1, 0, 0)}


def test_conservative_voxelization_fills_watertight_interior():
    mesh = trimesh.creation.box(extents=[4.0, 4.0, 4.0])
    mesh.apply_translation([3.0, 3.0, 3.0])
    occupied, stats = conservative_voxelize_mesh(
        mesh=mesh,
        pitch=1.0,
        origin_xyz=(0.0, 0.0, 0.0),
        shape_xyz=(7, 7, 7),
        fill_watertight=True,
    )
    occupied_set = {tuple(index) for index in occupied.tolist()}
    assert (3, 3, 3) in occupied_set
    assert stats["occupied_voxels"] > stats["surface_voxels"]


def test_top_down_camera_matches_training_image_axes():
    bounds = [[-2, 0, -2], [2, 0, -2], [2, 0, 2], [-2, 0, 2]]
    camera = CameraModel.top_down_from_bounds(bounds, image_size=(100, 100))
    origin, direction, image_pixel = camera.ray_for_heatmap_pixel(1, 1, (3, 3))
    assert np.allclose(origin[[0, 2]], [0.0, 0.0])
    assert np.allclose(direction, [0.0, -1.0, 0.0])
    assert np.allclose(image_pixel, [49.5, 49.5])


def test_perspective_camera_center_ray_for_arbitrary_view():
    camera = CameraModel(
        projection="perspective",
        image_width=640,
        image_height=480,
        camera_to_world=tuple(tuple(float(v) for v in row) for row in np.eye(4)),
        convention="opencv",
        intrinsics=((500.0, 0.0, 319.5), (0.0, 500.0, 239.5), (0.0, 0.0, 1.0)),
    )
    origin, direction, _ = camera.ray_for_heatmap_pixel(0, 0, (1, 1))
    assert np.allclose(origin, [0.0, 0.0, 0.0])
    assert np.allclose(direction, [0.0, 0.0, 1.0])


def test_parse_scenerevis_pose_from_tool_call():
    response = """
<think>Place it beside the desk.</think>
<tool_calls>
[
  {
    "name": "add_object",
    "arguments": {
      "position": [1.0, 0.0, 2.0],
      "rotation": [0.0, 0.7071068, 0.0, 0.7071068],
      "size": [0.8, 1.2, 0.6]
    }
  }
]
</tool_calls>
"""
    pose = parse_scenerevis_pose(response)
    assert pose is not None
    assert np.allclose(pose.normalized_rotation, [0.0, 2 ** -0.5, 0.0, 2 ** -0.5])
    assert pose.target_size_xyz == (0.8, 1.2, 0.6)


def test_prepare_target_mesh_uses_bottom_center_anchor():
    mesh = trimesh.creation.box(extents=[1.0, 2.0, 3.0])
    pose = SceneReVisPose(
        rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
        target_size_xyz=(2.0, 4.0, 6.0),
    )
    prepared = prepare_target_mesh(mesh, pose)
    minimum, maximum = prepared.mesh.bounds
    assert np.allclose(prepared.mesh.extents, [2.0, 4.0, 6.0])
    assert np.isclose(minimum[1], 0.0)
    assert np.isclose((minimum[0] + maximum[0]) * 0.5, 0.0)
    assert np.isclose((minimum[2] + maximum[2]) * 0.5, 0.0)


def test_transform_decomposition_roundtrip_components():
    transform = np.eye(4)
    transform[:3, :3] = np.diag([2.0, 3.0, 4.0])
    transform[:3, 3] = [1.0, 2.0, 3.0]
    pose = decompose_transform(transform)
    assert np.allclose(pose["position_xyz"], [1.0, 2.0, 3.0])
    assert np.allclose(pose["scale_xyz"], [2.0, 3.0, 4.0])
    assert np.allclose(pose["rotation_xyzw"], [0.0, 0.0, 0.0, 1.0])


def test_exact_mesh_first_hit_and_score_ordered_release():
    floor = trimesh.creation.box(extents=[4.0, 0.1, 4.0])
    floor.apply_translation([0.0, -0.05, 0.0])
    scene = trimesh.Scene(floor)
    spec = VoxelGridSpec(origin_xyz=(-3.0, -1.0, -3.0), shape_xyz=(12, 8, 12), pitch=0.5)
    occupancy, _ = voxelize_scene(scene, spec)
    projector = FirstHitProjector(scene_mesh=floor, occupancy=occupancy)
    camera = CameraModel.top_down_from_bounds(
        [[-2, 0, -2], [2, 0, -2], [2, 0, 2], [-2, 0, 2]],
        image_size=(96, 96),
    )
    kernel = TargetVoxelKernel(
        offsets_xyz=np.array([[0, 1, 0]], dtype=np.int64),
        minimum_release_height_voxels=1,
        pitch=0.5,
    )
    heatmap = np.zeros((3, 3), dtype=np.float32)
    heatmap[1, 1] = 1.0
    result = score_ordered_release_search(
        heatmap=heatmap,
        camera=camera,
        projector=projector,
        target_kernel=kernel,
        max_candidates=1,
    )
    assert result is not None
    assert result.first_hit_source == "mesh_ray"
    assert result.heatmap_index_rc == (1, 1)
    assert np.isclose(result.surface_hit_world_xyz[1], 0.0, atol=1e-6)
    assert result.release_bottom_world_xyz[1] > result.surface_hit_world_xyz[1]
    assert result.tested_release_poses == 1


def test_release_search_tries_next_candidate_instead_of_traversing_height():
    spec = VoxelGridSpec(origin_xyz=(0.0, 0.0, 0.0), shape_xyz=(8, 8, 8), pitch=1.0)
    occupancy = PackedVoxelGrid(spec)
    occupancy.set_indices(np.array([[2, 3, 2]], dtype=np.int64))
    kernel = TargetVoxelKernel(
        offsets_xyz=np.array([[0, 1, 0]], dtype=np.int64),
        minimum_release_height_voxels=1,
        pitch=1.0,
    )

    class StubCamera:
        def ray_for_heatmap_pixel(self, row, col, heatmap_shape):
            del heatmap_shape
            return (
                np.array([float(col), float(row), 0.0]),
                np.array([0.0, -1.0, 0.0]),
                (float(col), float(row)),
            )

    class StubProjector:
        def __init__(self):
            self.occupancy = occupancy

        def first_hit(self, origin, direction):
            del direction
            x = 2 if int(origin[0]) == 0 else 5
            return RayHit(
                index_xyz=(x, 2, 2),
                point_world_xyz=(float(x), 2.0, 2.0),
                distance=1.0,
                source="stub",
            )

    heatmap = np.array([[1.0, 0.5]], dtype=np.float32)
    result = score_ordered_release_search(
        heatmap=heatmap,
        camera=StubCamera(),
        projector=StubProjector(),
        target_kernel=kernel,
        max_candidates=2,
    )

    assert result is not None
    assert result.heatmap_index_rc == (0, 1)
    assert result.surface_hit_index_xyz == (5, 2, 2)
    assert result.tested_candidates == 2
    assert result.tested_release_poses == 2
    assert "lift_voxels" not in result.to_dict()
