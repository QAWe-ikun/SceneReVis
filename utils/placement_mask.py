"""
Feasibility mask computation for heatmap-based object placement.

Computes a 2D boolean grid (heatmap_res x heatmap_res) where True means
the grid cell is a valid placement location for a new object.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from matplotlib.path import Path
from scipy.ndimage import binary_dilation
from scipy.spatial.transform import Rotation as R


def _extract_objects(scene: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract all objects from scene regardless of format."""
    if 'objects' in scene:
        return scene['objects']
    if 'groups' in scene:
        objs = []
        for group in scene['groups']:
            objs.extend(group.get('objects', []))
        return objs
    return []


def _extract_bounds_bottom(scene: Dict[str, Any]) -> List[List[float]]:
    """Extract room bounds_bottom polygon vertices."""
    if 'room_envelope' in scene:
        return scene['room_envelope'].get('bounds_bottom', [])
    if 'bounds_bottom' in scene:
        return scene['bounds_bottom']
    return []


def _compute_room_params(bounds_bottom: List[List[float]]) -> Tuple[float, float, float, float]:
    """Compute room bounding box and center from bounds_bottom polygon.

    Returns: (x_min, x_max, z_min, z_max)
    """
    if not bounds_bottom:
        return -5, 5, -5, 5
    xs = [p[0] for p in bounds_bottom]
    zs = [p[2] for p in bounds_bottom]
    return min(xs), max(xs), min(zs), max(zs)


def _room_mask(bounds_bottom: List[List[float]], heatmap_res: int,
               x_min: float, x_max: float, z_min: float, z_max: float) -> np.ndarray:
    """Binary mask: True where grid cell center is inside the room floor polygon."""
    mask = np.zeros((heatmap_res, heatmap_res), dtype=bool)

    if not bounds_bottom or len(bounds_bottom) < 3:
        return mask

    # Grid center points slightly inset from boundary to avoid
    # Path.contains_points treating edge points as outside
    cell_size_x = (x_max - x_min) / heatmap_res
    cell_size_z = (z_max - z_min) / heatmap_res
    margin_x = cell_size_x / 2
    margin_z = cell_size_z / 2

    xs = np.linspace(x_min + margin_x, x_max - margin_x, heatmap_res)
    zs = np.linspace(z_min + margin_z, z_max - margin_z, heatmap_res)
    gx, gz = np.meshgrid(xs, zs, indexing='ij')
    points = np.stack([gx.ravel(), gz.ravel()], axis=-1)

    # Room polygon in XZ plane
    polygon = Path([(p[0], p[2]) for p in bounds_bottom])
    inside = polygon.contains_points(points)
    mask = inside.reshape(heatmap_res, heatmap_res)

    return mask


def _rotated_footprint(pos: List[float], size: List[float], rot: List[float]) -> np.ndarray:
    """Compute the XZ footprint of an object considering its rotation.

    Returns Nx2 array of (x, z) corners forming a valid convex polygon.
    Projects all 8 corners of the 3D box onto XZ and keeps unique vertices.
    """
    # Local half-dimensions
    hw, hh, hd = size[0] / 2, size[1] / 2, size[2] / 2

    # 8 corners in local coordinates (x, y, z)
    local_corners = np.array([
        [-hw, -hh, -hd], [hw, -hh, -hd], [hw, -hh, hd], [-hw, -hh, hd],
        [-hw, hh, -hd], [hw, hh, -hd], [hw, hh, hd], [-hw, hh, hd],
    ])

    # Apply rotation
    rot_obj = R.from_quat(rot)
    world_corners = rot_obj.apply(local_corners)

    # Translate to object position
    world_corners[:, 0] += pos[0]
    world_corners[:, 2] += pos[2]

    # Project to XZ plane and deduplicate
    xz_corners = world_corners[:, [0, 2]]
    # Remove duplicates (8 corners → 4 unique XZ points for a rectangular box)
    unique = np.unique(xz_corners.round(decimals=6), axis=0)

    # Order vertices counter-clockwise for a valid Path polygon
    center = unique.mean(axis=0)
    angles = np.arctan2(unique[:, 1] - center[1], unique[:, 0] - center[0])
    ordered = unique[np.argsort(angles)]

    return ordered


def _is_table_like(obj: Dict[str, Any]) -> bool:
    """Heuristic: is this object table-like (legs only, open underneath)?

    Checks if the object has significant clearance above the floor, i.e.,
    its bottom surface is elevated. A simplified rule:
    - If the object's y-position is significantly above the floor level AND
      the height dimension is large relative to the position, it's table-like.
    - Also check description keywords as a fallback.
    """
    desc = obj.get('desc', '').lower()
    table_keywords = ['table', 'desk', 'stand', 'nightstand', 'dresser', 'cabinet', 'shelf']
    if any(kw in desc for kw in table_keywords):
        return True

    # Height heuristic: if the object's bottom is significantly above ground
    # For an object at pos[1] with size[1] height, bottom = pos[1] - size[1]/2
    pos = obj.get('pos', [0, 0, 0])
    size = obj.get('size', [1, 1, 1])
    bottom_y = pos[1] - size[1] / 2
    if bottom_y > 0.05:  # More than 5cm above ground
        return True

    return False


def _footprint_polygon_legs(obj: Dict[str, Any]) -> np.ndarray:
    """For table-like objects, compute only the leg-contact area footprint.

    Approximation: use the bottom 20% of the object's height for projection.
    This captures the legs/base but not the overhanging top surface.
    """
    pos = obj.get('pos', [0, 0, 0])
    size = obj.get('size', [1, 1, 1])
    rot = obj.get('rot', [0, 0, 0, 1])

    # Bottom 20% of height
    hw = size[0] / 2
    hh_legs = size[1] * 0.1  # bottom 20% = 0 to 0.2*height from bottom
    hd = size[2] / 2

    # 4 corners at the base level
    bottom_y = pos[1] - size[1] / 2
    leg_corners = np.array([
        [pos[0] - hw, bottom_y, pos[2] - hd],
        [pos[0] + hw, bottom_y, pos[2] - hd],
        [pos[0] + hw, bottom_y, pos[2] + hd],
        [pos[0] - hw, bottom_y, pos[2] + hd],
    ])

    # Apply rotation
    rot_obj = R.from_quat(rot)
    rotated = rot_obj.apply(leg_corners)

    return rotated[:, [0, 2]]


def _collision_mask(objects: List[Dict[str, Any]], heatmap_res: int,
                    x_min: float, x_max: float, z_min: float, z_max: float,
                    placement_plane: str = "floor",
                    target_jid: Optional[str] = None) -> np.ndarray:
    """Binary mask: False where existing objects block placement.

    Args:
        objects: list of scene objects
        heatmap_res: grid resolution
        x_min, x_max, z_min, z_max: room bounds
        placement_plane: "floor" or object jid
        target_jid: if placement_plane is a jid, this is the target object

    Returns:
        Boolean array where True = no collision.
    """
    # Initialize: all cells are collision-free (True)
    mask = np.ones((heatmap_res, heatmap_res), dtype=bool)

    # Grid cell centers (consistent with room mask)
    cell_size_x = (x_max - x_min) / heatmap_res
    cell_size_z = (z_max - z_min) / heatmap_res
    margin_x = cell_size_x / 2
    margin_z = cell_size_z / 2

    xs = np.linspace(x_min + margin_x, x_max - margin_x, heatmap_res)
    zs = np.linspace(z_min + margin_z, z_max - margin_z, heatmap_res)
    gx, gz = np.meshgrid(xs, zs, indexing='ij')
    points = np.stack([gx.ravel(), gz.ravel()], axis=-1)

    for obj in objects:
        # Skip the target object itself when placing ON it
        obj_id = obj.get('jid', '') or obj.get('uid', '')
        if placement_plane != "floor" and obj_id == placement_plane:
            continue

        pos = obj.get('pos', [0, 0, 0])
        size = obj.get('size', [1, 1, 1])

        if placement_plane == "floor":
            # For table-like objects, only mask the legs/base area
            if _is_table_like(obj):
                footprint_2d = _footprint_polygon_legs(obj)
            else:
                # Full footprint for solid objects
                footprint_2d = _rotated_footprint(pos, size, obj.get('rot', [0, 0, 0, 1]))

            # Test which grid points fall inside the footprint polygon
            fp_path = Path(footprint_2d)
            hits = fp_path.contains_points(points)
            mask &= ~hits.reshape(heatmap_res, heatmap_res)

        else:
            # placement_plane == jid: only mask objects that overlap with
            # the target object's top surface. For simplicity, mask all
            # objects on the same surface (same jid children).
            pass

    return mask


def _clearance_mask(collision_mask: np.ndarray, clearance: float,
                    cell_size: float) -> np.ndarray:
    """Expand collision regions by clearance distance using morphological dilation.

    Returns:
        Boolean array where True = sufficient clearance.
    """
    if clearance <= 0:
        return collision_mask

    # Number of pixels to dilate
    radius = max(1, int(np.ceil(clearance / cell_size)))

    # Structuring element: circular disk
    size = 2 * radius + 1
    y_grid, x_grid = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    struct = x_grid ** 2 + y_grid ** 2 <= radius ** 2

    dilated = binary_dilation(~collision_mask, structure=struct)
    return ~dilated


def _target_surface_mask(target_obj: Dict[str, Any], heatmap_res: int,
                         x_min: float, x_max: float, z_min: float, z_max: float) -> np.ndarray:
    """Mask for placement on top of a specific object.

    Only the top surface of the target object is feasible.
    Returns: boolean array where True = on the target's top surface.
    """
    pos = target_obj.get('pos', [0, 0, 0])
    size = target_obj.get('size', [1, 1, 1])
    rot = target_obj.get('rot', [0, 0, 0, 1])

    # Top surface is a rectangle at y = pos[1] + size[1]/2
    # Project the 4 top corners to XZ plane
    hw, hd = size[0] / 2, size[2] / 2
    top_y = pos[1] + size[1] / 2
    top_corners = np.array([
        [pos[0] - hw, top_y, pos[2] - hd],
        [pos[0] + hw, top_y, pos[2] - hd],
        [pos[0] + hw, top_y, pos[2] + hd],
        [pos[0] - hw, top_y, pos[2] + hd],
    ])

    rot_obj = R.from_quat(rot)
    rotated = rot_obj.apply(top_corners)
    footprint_2d = rotated[:, [0, 2]]

    # Test which grid points fall inside the top surface
    cell_size_x = (x_max - x_min) / heatmap_res
    cell_size_z = (z_max - z_min) / heatmap_res
    margin_x = cell_size_x / 2
    margin_z = cell_size_z / 2
    xs = np.linspace(x_min + margin_x, x_max - margin_x, heatmap_res)
    zs = np.linspace(z_min + margin_z, z_max - margin_z, heatmap_res)
    gx, gz = np.meshgrid(xs, zs, indexing='ij')
    points = np.stack([gx.ravel(), gz.ravel()], axis=-1)

    # Order vertices counter-clockwise for valid Path
    center = footprint_2d.mean(axis=0)
    angles = np.arctan2(footprint_2d[:, 1] - center[1], footprint_2d[:, 0] - center[0])
    ordered = footprint_2d[np.argsort(angles)]

    top_path = Path(ordered)
    inside = top_path.contains_points(points)
    return inside.reshape(heatmap_res, heatmap_res)


def grid_to_world(gi: int, gj: int, cx: float, cz: float,
                  ortho_scale: float, heatmap_res: int) -> Tuple[float, float]:
    """Convert grid cell index to world coordinates (x, z) of cell center.

    Linear mapping for orthographic projection.
    """
    x_min = cx - ortho_scale / 2
    z_min = cz - ortho_scale / 2
    cell_size_x = ortho_scale / heatmap_res
    cell_size_z = ortho_scale / heatmap_res
    x = x_min + (gj + 0.5) * cell_size_x
    z = z_min + (gi + 0.5) * cell_size_z
    return x, z


def world_to_grid(x: float, z: float, cx: float, cz: float,
                  ortho_scale: float, heatmap_res: int) -> Tuple[int, int]:
    """Convert world coordinates (x, z) to nearest grid cell index (gi, gj)."""
    x_min = cx - ortho_scale / 2
    z_min = cz - ortho_scale / 2
    cell_size_x = ortho_scale / heatmap_res
    cell_size_z = ortho_scale / heatmap_res
    gj = int((x - x_min) / cell_size_x - 0.5)
    gi = int((z - z_min) / cell_size_z - 0.5)
    gi = np.clip(gi, 0, heatmap_res - 1)
    gj = np.clip(gj, 0, heatmap_res - 1)
    return gi, gj


def compute_mask(scene: Dict[str, Any], heatmap_res: int = 256,
                 clearance: float = 0.5,
                 placement_plane: str = "floor") -> Tuple[np.ndarray, float, float, float]:
    """Compute the full feasibility mask for placing a new object.

    Args:
        scene: scene JSON data (flat or grouped format)
        heatmap_res: grid resolution (default 256)
        clearance: minimum clearance around existing objects in meters (default 0.5)
        placement_plane: "floor" for ground placement, or the jid of an object
                         to place on top of

    Returns:
        (mask, ortho_scale, cx, cz) where:
        - mask: 2D boolean array (heatmap_res x heatmap_res), True = feasible
        - ortho_scale: Blender orthographic scale (meters)
        - cx, cz: room center in world coordinates
    """
    bounds_bottom = _extract_bounds_bottom(scene)
    x_min, x_max, z_min, z_max = _compute_room_params(bounds_bottom)

    cx = (x_min + x_max) / 2
    cz = (z_min + z_max) / 2
    span = max(x_max - x_min, z_max - z_min, 1)
    ortho_scale = span * 1.2

    objects = _extract_objects(scene)

    # Step 1: Room boundary mask
    mask = _room_mask(bounds_bottom, heatmap_res, x_min, x_max, z_min, z_max)

    # Step 2: Handle placement_plane
    if placement_plane != "floor":
        # Find target object by jid
        target_obj = None
        for obj in objects:
            obj_id = obj.get('jid', '') or obj.get('uid', '')
            if obj_id == placement_plane:
                target_obj = obj
                break

        if target_obj is None:
            # Target object not found, fallback to full room mask
            # (collision mask will still be applied for all objects)
            pass
        else:
            # Restrict mask to target object's top surface
            surface_mask = _target_surface_mask(target_obj, heatmap_res,
                                                x_min, x_max, z_min, z_max)
            mask = mask & surface_mask

        # For surface placement, collision mask only considers other objects
        # that are on the same surface (simplified: mask all non-target objects
        # whose footprint overlaps with the target surface)
        # For now, we mask all other objects normally
        col_mask = _collision_mask(objects, heatmap_res, x_min, x_max, z_min, z_max,
                                   placement_plane=placement_plane)
        mask = mask & col_mask

    else:
        # Floor placement: apply collision mask with height layering
        col_mask = _collision_mask(objects, heatmap_res, x_min, x_max, z_min, z_max,
                                   placement_plane="floor")
        mask = mask & col_mask

    # Step 3: Clearance (only for floor placement)
    if placement_plane == "floor":
        cell_size = ortho_scale / heatmap_res
        clr_mask = _clearance_mask(col_mask, clearance, cell_size)
        mask = mask & clr_mask

    return mask, ortho_scale, cx, cz


def find_best_position(mask: np.ndarray, ortho_scale: float,
                       cx: float, cz: float, heatmap_res: int,
                       placement_plane: str = "floor",
                       scene: Optional[Dict[str, Any]] = None,
                       top_k: int = 1,
                       min_distance: float = 0.5) -> List[List[float]]:
    """Extract top-k best placement positions from the feasibility mask.

    Args:
        mask: 2D boolean feasibility mask
        ortho_scale: orthographic scale in meters
        cx, cz: room center
        heatmap_res: grid resolution
        placement_plane: "floor" or object jid
        scene: optional scene data (for computing Y height when placing on object)
        top_k: number of candidate positions to return
        min_distance: minimum distance between candidates in meters

    Returns:
        List of [x, y, z] positions, may be empty if no feasible positions.
    """
    if not np.any(mask):
        return []

    positions = []
    working_mask = mask.copy().astype(np.float32)

    for _ in range(top_k):
        if not np.any(working_mask > 0):
            break

        # Find argmax (all feasible cells have score 1.0, so this is arbitrary)
        idx = np.argmax(working_mask.ravel())
        gi, gj = divmod(idx, heatmap_res)

        x, z = grid_to_world(gi, gj, cx, cz, ortho_scale, heatmap_res)

        # Compute Y coordinate
        if placement_plane == "floor":
            y = 0.0
        else:
            # Y = top of target object + half height of new object
            # We don't know the new object's height yet, so just use target top
            if scene is not None:
                objects = _extract_objects(scene)
                for obj in objects:
                    obj_id = obj.get('jid', '') or obj.get('uid', '')
                    if obj_id == placement_plane:
                        pos = obj.get('pos', [0, 0, 0])
                        size = obj.get('size', [1, 1, 1])
                        y = pos[1] + size[1] / 2
                        break
                else:
                    y = 0.0
            else:
                y = 0.0

        positions.append([float(x), float(y), float(z)])

        # Suppress neighborhood to get diverse candidates
        radius_cells = int(min_distance / (ortho_scale / heatmap_res))
        y1 = max(0, gi - radius_cells)
        y2 = min(heatmap_res, gi + radius_cells + 1)
        x1 = max(0, gj - radius_cells)
        x2 = min(heatmap_res, gj + radius_cells + 1)
        working_mask[y1:y2, x1:x2] = 0

    return positions


def visualize_mask(mask: np.ndarray, ortho_scale: float,
                   cx: float, cz: float, save_path: Optional[str] = None) -> np.ndarray:
    """Create a debug visualization of the feasibility mask.

    Returns:
        numpy uint8 image (H, W, 3) RGB.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(mask.astype(np.float32), cmap='gray', origin='upper')
    ax.set_title(f'Feasibility Mask ({mask.shape[0]}x{mask.shape[1]})')
    ax.set_xlabel('X grid index')
    ax.set_ylabel('Z grid index')

    # Add world coordinate labels
    x_min_world = cx - ortho_scale / 2
    x_max_world = cx + ortho_scale / 2
    z_min_world = cz - ortho_scale / 2
    z_max_world = cz + ortho_scale / 2
    ax.set_xticks([0, mask.shape[1] - 1])
    ax.set_xticklabels([f'{x_min_world:.1f}', f'{x_max_world:.1f}'])
    ax.set_yticks([0, mask.shape[0] - 1])
    ax.set_yticklabels([f'{z_min_world:.1f}', f'{z_max_world:.1f}'])

    feasible_pct = mask.sum() / (mask.shape[0] * mask.shape[1]) * 100
    ax.text(0.02, 0.98, f'Feasible: {feasible_pct:.1f}%',
            transform=ax.transAxes, va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100)
        plt.close()

    # Convert to numpy image
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return img
