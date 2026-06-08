"""
Unit tests for placement_mask module.
"""

import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.placement_mask import (
    compute_mask, grid_to_world, world_to_grid,
    _room_mask, _collision_mask, _clearance_mask,
    find_best_position, _is_table_like, _rotated_footprint
)


def _make_rect_scene(x_min=-3, x_max=3, z_min=-3, z_max=3, objects=None):
    """Create a simple rectangular room scene."""
    scene = {
        "room_envelope": {
            "bounds_bottom": [
                [x_min, 0, z_max], [x_max, 0, z_max],
                [x_max, 0, z_min], [x_min, 0, z_min]
            ],
            "bounds_top": [
                [x_min, 3, z_max], [x_max, 3, z_max],
                [x_max, 3, z_min], [x_min, 3, z_min]
            ]
        },
        "objects": objects or []
    }
    return scene


def test_empty_room_all_feasible():
    """Empty rectangular room -> all cells inside room bounds should be feasible."""
    scene = _make_rect_scene()
    mask, ortho_scale, cx, cz = compute_mask(scene, heatmap_res=64)

    # Grid covers ortho_scale (1.2x room), so cells outside room bounds are False.
    # Check that cells INSIDE room bounds are all True.
    # Room: X=[-3,3], Z=[-3,3]. ortho_scale=7.2, center=(0,0).
    # Grid spans [-3.6, 3.6]. Room interior in grid: ~[5, 58] out of 64.
    margin = 6  # cells outside room bounds (plus safety)
    assert mask.shape == (64, 64)
    assert ortho_scale > 0
    assert mask[margin:-margin, margin:-margin].all(), \
        "All cells inside room bounds should be feasible"
    print("PASS: test_empty_room_all_feasible")


def test_object_creates_collision():
    """Place object in center -> center cells should be infeasible."""
    obj = {
        "desc": "box",
        "size": [2, 1, 2],
        "pos": [0, 0.5, 0],
        "rot": [0, 0, 0, 1],  # no rotation
    }
    scene = _make_rect_scene(objects=[obj])
    mask, ortho_scale, cx, cz = compute_mask(scene, heatmap_res=64, clearance=0)

    # Some cells should be masked out (collision)
    collision_count = (~mask).sum()
    assert collision_count > 0, f"Object should create collision cells, got {collision_count}"

    # Cells near room corners (inside room bounds) should still be feasible
    margin = 6  # cells outside room bounds (plus safety)
    assert mask[margin, margin], f"Room corner should be feasible"
    assert mask[-margin, -margin], f"Room corner should be feasible"
    print(f"PASS: test_object_cre_collision ({collision_count} collision cells)")


def test_clearance_expands_collision():
    """Clearance should expand collision region."""
    obj = {
        "desc": "box",
        "size": [1, 1, 1],
        "pos": [0, 0.5, 0],
        "rot": [0, 0, 0, 1],
    }
    scene = _make_rect_scene(objects=[obj])

    mask_no_clearance, _, _, _ = compute_mask(scene, heatmap_res=64, clearance=0)
    mask_clearance, _, _, _ = compute_mask(scene, heatmap_res=64, clearance=0.5)

    clearance_count_0 = (~mask_no_clearance).sum()
    clearance_count_05 = (~mask_clearance).sum()

    assert clearance_count_05 > clearance_count_0, \
        f"Clearance=0.5 should mask more cells: {clearance_count_05} vs {clearance_count_0}"
    print(f"PASS: test_clearance_expands_collision ({clearance_count_0} -> {clearance_count_05} cells)")


def test_table_height_layering():
    """Table-like object on floor should not mask entire footprint."""
    table = {
        "desc": "wooden desk",
        "size": [1.5, 0.75, 0.8],
        "pos": [0, 0.375, 0],  # bottom at y=0
        "rot": [0, 0, 0, 1],
    }
    scene = _make_rect_scene(objects=[table])
    mask, _, _, _ = compute_mask(scene, heatmap_res=64, clearance=0)

    # For a table-like object, the center (where the table is) should still
    # have SOME feasible cells (because only legs are masked, not the top)
    # The center region around (32, 32) in a 64x64 grid
    center_region = mask[24:40, 24:40]
    center_feasible = center_region.sum()
    assert center_feasible > 0, \
        f"Table should leave some center cells feasible, got {center_feasible}"
    print(f"PASS: test_table_height_layering ({center_feasible} center cells feasible)")


def test_grid_world_roundtrip():
    """grid_to_world and world_to_grid should be inverses."""
    cx, cz = 0, 0
    ortho_scale = 6.0
    res = 64

    # Match rendered top-view convention: top image row corresponds to max Z.
    _, z_top = grid_to_world(0, 0, cx, cz, ortho_scale, res)
    _, z_bottom = grid_to_world(res - 1, 0, cx, cz, ortho_scale, res)
    assert z_top > cz, f"row=0 should map to max Z side, got {z_top}"
    assert z_bottom < cz, f"last row should map to min Z side, got {z_bottom}"

    # Test several positions
    for gi, gj in [(0, 0), (32, 32), (63, 63), (10, 50)]:
        x, z = grid_to_world(gi, gj, cx, cz, ortho_scale, res)
        gi2, gj2 = world_to_grid(x, z, cx, cz, ortho_scale, res)
        assert gi == gi2, f"gi mismatch: {gi} vs {gi2}"
        assert gj == gj2, f"gj mismatch: {gj} vs {gj2}"

    print("PASS: test_grid_world_roundtrip")


def test_find_best_position():
    """find_best_position should return valid coordinates."""
    scene = _make_rect_scene()
    mask, ortho_scale, cx, cz = compute_mask(scene, heatmap_res=64)

    positions = find_best_position(mask, ortho_scale, cx, cz, 64, top_k=3)

    assert len(positions) == 3, f"Should return 3 positions, got {len(positions)}"
    for pos in positions:
        assert len(pos) == 3, "Each position should be [x, y, z]"
        assert pos[1] == 0.0, "Y should be 0 for floor placement"

    # Positions should be different (min_distance=0.5)
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            dist = np.sqrt(
                (positions[i][0] - positions[j][0]) ** 2 +
                (positions[i][2] - positions[j][2]) ** 2
            )
            assert dist >= 0.4, f"Positions should be separated: {dist}"

    print(f"PASS: test_find_best_position ({positions})")


def test_is_table_like():
    """Table-like detection should work on description keywords."""
    assert _is_table_like({"desc": "wooden desk"})
    assert _is_table_like({"desc": "nightstand"})
    assert _is_table_like({"desc": "office table"})
    assert not _is_table_like({"desc": "queen bed"})
    assert not _is_table_like({"desc": "wardrobe"})
    print("PASS: test_is_table_like")


def test_rotated_footprint():
    """Rotated footprint should differ from unrotated."""
    pos = [0, 0.5, 0]
    size = [2, 1, 1]

    # No rotation
    fp_no_rot = _rotated_footprint(pos, size, [0, 0, 0, 1])

    # 45-degree rotation around Y axis
    from scipy.spatial.transform import Rotation as R
    rot_45 = R.from_euler('y', 45, degrees=True).as_quat()
    fp_rot = _rotated_footprint(pos, size, rot_45)

    # The footprints should be different
    assert not np.allclose(fp_no_rot, fp_rot), \
        "Rotated footprint should differ from unrotated"

    print(f"PASS: test_rotated_footprint")


def test_no_feasible_position():
    """Tiny room with large object should have no feasible position."""
    obj = {
        "desc": "huge bed",
        "size": [10, 1, 10],  # Larger than room
        "pos": [0, 0.5, 0],
        "rot": [0, 0, 0, 1],
    }
    scene = _make_rect_scene(-3, 3, -3, 3, [obj])
    mask, _, _, _ = compute_mask(scene, heatmap_res=32, clearance=0)

    positions = find_best_position(mask, 7.2, 0, 0, 32, top_k=1)
    assert len(positions) == 0, f"Should have no feasible positions, got {positions}"
    print("PASS: test_no_feasible_position")


if __name__ == '__main__':
    test_empty_room_all_feasible()
    test_object_creates_collision()
    test_clearance_expands_collision()
    test_table_height_layering()
    test_grid_world_roundtrip()
    test_find_best_position()
    test_is_table_like()
    test_rotated_footprint()
    test_no_feasible_position()
    print("\nAll tests passed!")
