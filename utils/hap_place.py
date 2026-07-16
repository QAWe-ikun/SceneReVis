"""Geometry and release-search primitives for the HAP-Place pipeline.

The project uses an XYZ world frame with +Y as up. Dense arrays are stored in
ZYX order, while public coordinates and voxel indices always use XYZ order.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Sequence, Tuple

import numpy as np
import trimesh
from scipy import ndimage
from scipy.spatial.transform import Rotation


EPS = 1e-9
SAT_CHUNK_SIZE = 65536


def _vec3(value: Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain three finite values, got {value!r}")
    return array


def _matrix4(value: Sequence[Sequence[float]], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (4, 4) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    return array


def _points_in_polygon_xz(
    x_coordinates: np.ndarray,
    z_coordinates: np.ndarray,
    polygon_xz: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    """Vectorized even-odd test with polygon edges treated as inside."""
    x = np.asarray(x_coordinates, dtype=np.float64)
    z = np.asarray(z_coordinates, dtype=np.float64)
    polygon = np.asarray(polygon_xz, dtype=np.float64)
    inside = np.zeros(np.broadcast_shapes(x.shape, z.shape), dtype=bool)
    boundary = np.zeros_like(inside)
    for index in range(len(polygon)):
        x0, z0 = polygon[index]
        x1, z1 = polygon[(index + 1) % len(polygon)]
        dx = x1 - x0
        dz = z1 - z0
        edge_length = max(float(np.hypot(dx, dz)), EPS)
        cross = (x - x0) * dz - (z - z0) * dx
        projection = (x - x0) * dx + (z - z0) * dz
        boundary |= (
            (np.abs(cross) <= tolerance * edge_length)
            & (projection >= -tolerance)
            & (projection <= edge_length * edge_length + tolerance)
        )
        crosses = (z0 > z) != (z1 > z)
        denominator = dz if abs(dz) > EPS else 1.0
        intersection_x = x0 + (z - z0) * dx / denominator
        inside ^= crosses & (x < intersection_x)
    return inside | boundary


@dataclass(frozen=True)
class VoxelGridSpec:
    """A cubic-pitch voxel grid described in XYZ order."""

    origin_xyz: Tuple[float, float, float]
    shape_xyz: Tuple[int, int, int]
    pitch: float

    def __post_init__(self) -> None:
        if len(self.shape_xyz) != 3 or any(int(v) <= 0 for v in self.shape_xyz):
            raise ValueError(f"shape_xyz must be positive, got {self.shape_xyz}")
        if not math.isfinite(self.pitch) or self.pitch <= 0:
            raise ValueError(f"pitch must be positive, got {self.pitch}")
        _vec3(self.origin_xyz, "origin_xyz")

    @property
    def origin(self) -> np.ndarray:
        return np.asarray(self.origin_xyz, dtype=np.float64)

    @property
    def shape(self) -> np.ndarray:
        return np.asarray(self.shape_xyz, dtype=np.int64)

    @property
    def bounds_max(self) -> np.ndarray:
        return self.origin + self.shape * self.pitch

    @property
    def memory_bytes(self) -> int:
        sx, sy, sz = self.shape_xyz
        return int(math.ceil(sx / 64) * sy * sz * 8)

    def world_to_grid(self, points_xyz: Sequence[float] | np.ndarray) -> np.ndarray:
        points = np.asarray(points_xyz, dtype=np.float64)
        return np.floor((points - self.origin) / self.pitch).astype(np.int64)

    def grid_to_world(
        self,
        indices_xyz: Sequence[int] | np.ndarray,
        center: bool = True,
    ) -> np.ndarray:
        indices = np.asarray(indices_xyz, dtype=np.float64)
        offset = 0.5 if center else 0.0
        return self.origin + (indices + offset) * self.pitch

    def contains(self, indices_xyz: Sequence[int] | np.ndarray) -> np.ndarray:
        indices = np.asarray(indices_xyz, dtype=np.int64)
        return np.all((indices >= 0) & (indices < self.shape), axis=-1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "origin_xyz": list(self.origin_xyz),
            "shape_xyz": list(self.shape_xyz),
            "pitch": float(self.pitch),
            "bounds_max_xyz": self.bounds_max.tolist(),
            "packed_memory_bytes": self.memory_bytes,
        }

    @classmethod
    def from_room_bounds(
        cls,
        bounds_bottom: Sequence[Sequence[float]],
        bounds_top: Sequence[Sequence[float]],
        resolution: int = 256,
        padding_fraction: float = 0.02,
    ) -> "VoxelGridSpec":
        bottom = np.asarray(bounds_bottom, dtype=np.float64)
        top = np.asarray(bounds_top, dtype=np.float64)
        if bottom.ndim != 2 or bottom.shape[1] != 3 or len(bottom) < 3:
            raise ValueError("bounds_bottom must contain at least three XYZ vertices")
        all_points = bottom if top.size == 0 else np.concatenate([bottom, top], axis=0)
        minimum = all_points.min(axis=0)
        maximum = all_points.max(axis=0)
        span = maximum - minimum
        cube_span = max(float(span.max()), 1e-3)
        pad = cube_span * max(0.0, float(padding_fraction))
        cube_span += 2.0 * pad
        center = (minimum + maximum) * 0.5
        origin = center - cube_span * 0.5
        return cls(
            origin_xyz=tuple(float(v) for v in origin),
            shape_xyz=(int(resolution), int(resolution), int(resolution)),
            pitch=float(cube_span / resolution),
        )


@dataclass(frozen=True)
class RayHit:
    index_xyz: Tuple[int, int, int]
    point_world_xyz: Tuple[float, float, float]
    distance: float
    source: str


class PackedVoxelGrid:
    """A 3D occupancy grid packed into uint64 words along world X."""

    def __init__(self, spec: VoxelGridSpec):
        self.spec = spec
        sx, sy, sz = spec.shape_xyz
        self.words = np.zeros((sz, sy, math.ceil(sx / 64)), dtype=np.uint64)

    @property
    def nbytes(self) -> int:
        return int(self.words.nbytes)

    def set_indices(self, indices_xyz: np.ndarray) -> int:
        indices = np.asarray(indices_xyz, dtype=np.int64).reshape(-1, 3)
        if len(indices) == 0:
            return 0
        valid = self.spec.contains(indices)
        indices = indices[valid]
        if len(indices) == 0:
            return 0
        indices = np.unique(indices, axis=0)
        x, y, z = indices.T
        word = x // 64
        bit = x % 64
        masks = np.left_shift(np.uint64(1), bit.astype(np.uint64))
        np.bitwise_or.at(self.words, (z, y, word), masks)
        return int(len(indices))

    def mark_outside_room(
        self,
        bounds_bottom: Sequence[Sequence[float]],
        bounds_top: Sequence[Sequence[float]],
    ) -> Dict[str, Any]:
        """Set every in-grid voxel outside the extruded room envelope to one."""
        bottom = np.asarray(bounds_bottom, dtype=np.float64)
        top = np.asarray(bounds_top, dtype=np.float64)
        if bottom.ndim != 2 or bottom.shape[1] != 3 or len(bottom) < 3:
            raise ValueError("bounds_bottom must contain at least three XYZ vertices")
        if top.ndim != 2 or top.shape[1] != 3 or len(top) < 3:
            raise ValueError("bounds_top must contain at least three XYZ vertices")

        polygon_xz = bottom[:, [0, 2]]
        shifted = np.roll(polygon_xz, -1, axis=0)
        signed_area_twice = np.sum(
            polygon_xz[:, 0] * shifted[:, 1] - shifted[:, 0] * polygon_xz[:, 1]
        )
        if abs(float(signed_area_twice)) <= EPS:
            raise ValueError("Room XZ polygon is degenerate")

        sx, sy, sz = self.spec.shape_xyz
        x_centers = self.spec.origin[0] + (np.arange(sx) + 0.5) * self.spec.pitch
        y_centers = self.spec.origin[1] + (np.arange(sy) + 0.5) * self.spec.pitch
        z_centers = self.spec.origin[2] + (np.arange(sz) + 0.5) * self.spec.pitch
        x_grid, z_grid = np.meshgrid(x_centers, z_centers, indexing="xy")
        inside_xz = _points_in_polygon_xz(
            x_grid,
            z_grid,
            polygon_xz,
            tolerance=max(self.spec.pitch * 1e-6, EPS),
        )

        floor_y = float(np.max(bottom[:, 1]))
        ceiling_y = float(np.min(top[:, 1]))
        if floor_y >= ceiling_y:
            raise ValueError("Room floor must be below its ceiling")
        inside_y = (y_centers >= floor_y) & (y_centers <= ceiling_y)

        word_count = self.words.shape[2]
        padded_outside_xz = np.zeros((sz, word_count * 64), dtype=bool)
        padded_outside_xz[:, :sx] = ~inside_xz
        chunks = padded_outside_xz.reshape(sz, word_count, 64).astype(np.uint64)
        bit_weights = np.left_shift(np.uint64(1), np.arange(64, dtype=np.uint64))
        outside_xz_words = np.bitwise_or.reduce(
            chunks * bit_weights[None, None, :],
            axis=2,
        )

        for y_index in np.flatnonzero(inside_y):
            self.words[:, y_index, :] |= outside_xz_words

        full_x_words = np.full(word_count, np.iinfo(np.uint64).max, dtype=np.uint64)
        remainder = sx % 64
        if remainder:
            full_x_words[-1] = np.left_shift(np.uint64(1), np.uint64(remainder)) - np.uint64(1)
        for y_index in np.flatnonzero(~inside_y):
            self.words[:, y_index, :] = full_x_words

        inside_xz_count = int(np.count_nonzero(inside_xz))
        inside_y_count = int(np.count_nonzero(inside_y))
        inside_room_voxels = inside_xz_count * inside_y_count
        total_voxels = sx * sy * sz
        return {
            "room_mask_method": "extruded_xz_polygon_v1",
            "room_floor_y": floor_y,
            "room_ceiling_y": ceiling_y,
            "inside_room_voxels": inside_room_voxels,
            "outside_room_voxels": total_voxels - inside_room_voxels,
        }

    def contains_index(self, index_xyz: Sequence[int]) -> bool:
        index = np.asarray(index_xyz, dtype=np.int64)
        if not bool(self.spec.contains(index)):
            return False
        x, y, z = (int(v) for v in index)
        return bool((self.words[z, y, x // 64] >> np.uint64(x % 64)) & np.uint64(1))

    def contains_indices(self, indices_xyz: np.ndarray) -> np.ndarray:
        indices = np.asarray(indices_xyz, dtype=np.int64).reshape(-1, 3)
        result = np.zeros(len(indices), dtype=bool)
        valid = self.spec.contains(indices)
        if not np.any(valid):
            return result
        selected = indices[valid]
        x, y, z = selected.T
        values = self.words[z, y, x // 64]
        result[valid] = ((values >> (x % 64).astype(np.uint64)) & np.uint64(1)) != 0
        return result

    def intersects_offsets(
        self,
        anchor_xyz: Sequence[int],
        offsets_xyz: np.ndarray,
        outside_is_collision: bool = True,
    ) -> bool:
        offsets = np.asarray(offsets_xyz, dtype=np.int64).reshape(-1, 3)
        global_indices = offsets + np.asarray(anchor_xyz, dtype=np.int64)
        valid = self.spec.contains(global_indices)
        if outside_is_collision and not np.all(valid):
            return True
        if not np.any(valid):
            return False
        return bool(np.any(self.contains_indices(global_indices[valid])))

    def intersects_kernel(
        self,
        anchor_xyz: Sequence[int],
        kernel: "TargetVoxelKernel",
        outside_is_collision: bool = True,
    ) -> bool:
        """Test a translated target kernel using shifted uint64 row masks."""
        anchor = np.asarray(anchor_xyz, dtype=np.int64)
        if anchor.shape != (3,):
            raise ValueError("anchor_xyz must contain three indices")

        x_min = int(anchor[0]) + kernel.x_min
        x_max = int(anchor[0]) + kernel.x_max
        rows_yz = kernel.row_offsets_yz + anchor[[1, 2]]
        sx, sy, sz = self.spec.shape_xyz
        inside_x = x_min >= 0 and x_max < sx
        inside_yz = np.all(
            (rows_yz[:, 0] >= 0)
            & (rows_yz[:, 0] < sy)
            & (rows_yz[:, 1] >= 0)
            & (rows_yz[:, 1] < sz)
        )
        if not inside_x or not inside_yz:
            return bool(outside_is_collision)

        word_base, bit_shift = divmod(x_min, 64)
        target_words = kernel.packed_x_words
        relative_words = np.arange(target_words.shape[1], dtype=np.int64)
        scene_word_indices = word_base + relative_words
        y = rows_yz[:, 0, None]
        z = rows_yz[:, 1, None]

        low_masks = target_words << np.uint64(bit_shift)
        scene_low = self.words[z, y, scene_word_indices[None, :]]
        if np.any(np.bitwise_and(scene_low, low_masks)):
            return True

        if bit_shift == 0:
            return False
        high_masks = target_words >> np.uint64(64 - bit_shift)
        valid_high = scene_word_indices + 1 < self.words.shape[2]
        if not np.any(valid_high):
            return False
        scene_high = self.words[
            z,
            y,
            (scene_word_indices[valid_high] + 1)[None, :],
        ]
        return bool(np.any(np.bitwise_and(scene_high, high_masks[:, valid_high])))

@dataclass(frozen=True)
class CameraModel:
    """Calibrated perspective or orthographic camera."""

    projection: str
    image_width: int
    image_height: int
    camera_to_world: Tuple[Tuple[float, ...], ...]
    convention: str = "opengl"
    intrinsics: Optional[Tuple[Tuple[float, ...], ...]] = None
    ortho_width: Optional[float] = None
    ortho_height: Optional[float] = None

    def __post_init__(self) -> None:
        if self.projection not in {"perspective", "orthographic"}:
            raise ValueError(f"Unsupported projection: {self.projection}")
        if self.convention not in {"opengl", "opencv"}:
            raise ValueError(f"Unsupported camera convention: {self.convention}")
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("Image dimensions must be positive")
        _matrix4(self.camera_to_world, "camera_to_world")
        if self.projection == "perspective":
            intrinsics = np.asarray(self.intrinsics, dtype=np.float64)
            if intrinsics.shape != (3, 3):
                raise ValueError("Perspective cameras require a 3x3 intrinsics matrix")
        elif not self.ortho_width or not self.ortho_height:
            raise ValueError("Orthographic cameras require ortho_width and ortho_height")

    @property
    def transform(self) -> np.ndarray:
        return np.asarray(self.camera_to_world, dtype=np.float64)

    def ray_for_heatmap_pixel(
        self,
        row: int,
        col: int,
        heatmap_shape: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float]]:
        heatmap_h, heatmap_w = heatmap_shape
        u = (float(col) + 0.5) * self.image_width / heatmap_w - 0.5
        v = (float(row) + 0.5) * self.image_height / heatmap_h - 0.5
        transform = self.transform
        rotation = transform[:3, :3]
        translation = transform[:3, 3]

        if self.projection == "perspective":
            k = np.asarray(self.intrinsics, dtype=np.float64)
            if self.convention == "opencv":
                local = np.linalg.inv(k) @ np.array([u, v, 1.0], dtype=np.float64)
            else:
                x = (u - k[0, 2]) / k[0, 0]
                y = -(v - k[1, 2]) / k[1, 1]
                local = np.array([x, y, -1.0], dtype=np.float64)
            origin = translation
            direction = rotation @ local
        else:
            x = ((u + 0.5) / self.image_width - 0.5) * float(self.ortho_width)
            if self.convention == "opencv":
                y = ((v + 0.5) / self.image_height - 0.5) * float(self.ortho_height)
                direction_local = np.array([0.0, 0.0, 1.0])
            else:
                y = (0.5 - (v + 0.5) / self.image_height) * float(self.ortho_height)
                direction_local = np.array([0.0, 0.0, -1.0])
            origin = translation + rotation[:, 0] * x + rotation[:, 1] * y
            direction = rotation @ direction_local

        direction = direction / np.linalg.norm(direction)
        return origin, direction, (u, v)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        return data

    @classmethod
    def top_down_from_bounds(
        cls,
        bounds_bottom: Sequence[Sequence[float]],
        image_size: Tuple[int, int],
        height: float = 20.0,
    ) -> "CameraModel":
        bounds = np.asarray(bounds_bottom, dtype=np.float64)
        if bounds.ndim != 2 or bounds.shape[1] != 3 or len(bounds) < 3:
            raise ValueError("bounds_bottom must contain at least three vertices")
        x_min, x_max = float(bounds[:, 0].min()), float(bounds[:, 0].max())
        z_min, z_max = float(bounds[:, 2].min()), float(bounds[:, 2].max())
        cx, cz = (x_min + x_max) * 0.5, (z_min + z_max) * 0.5
        span = max(x_max - x_min, z_max - z_min, 1.0) * 1.2
        floor_y = float(np.median(bounds[:, 1]))
        transform = np.eye(4, dtype=np.float64)
        transform[:3, 0] = [1.0, 0.0, 0.0]
        transform[:3, 1] = [0.0, 0.0, -1.0]
        transform[:3, 2] = [0.0, 1.0, 0.0]
        transform[:3, 3] = [cx, floor_y + height, cz]
        width, height_px = image_size
        return cls(
            projection="orthographic",
            image_width=int(width),
            image_height=int(height_px),
            camera_to_world=tuple(tuple(float(v) for v in row) for row in transform),
            convention="opengl",
            ortho_width=float(span),
            ortho_height=float(span),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CameraModel":
        return cls(
            projection=str(data["projection"]),
            image_width=int(data["image_width"]),
            image_height=int(data["image_height"]),
            camera_to_world=tuple(tuple(float(v) for v in row) for row in data["camera_to_world"]),
            convention=str(data.get("convention", "opengl")),
            intrinsics=(
                tuple(tuple(float(v) for v in row) for row in data["intrinsics"])
                if data.get("intrinsics") is not None
                else None
            ),
            ortho_width=(float(data["ortho_width"]) if data.get("ortho_width") is not None else None),
            ortho_height=(float(data["ortho_height"]) if data.get("ortho_height") is not None else None),
        )


@dataclass(frozen=True)
class SceneReVisPose:
    rotation_xyzw: Tuple[float, float, float, float]
    scale_xyz: Optional[Tuple[float, float, float]] = None
    target_size_xyz: Optional[Tuple[float, float, float]] = None
    source: str = "scenerevis"

    def __post_init__(self) -> None:
        quaternion = np.asarray(self.rotation_xyzw, dtype=np.float64)
        if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
            raise ValueError("rotation_xyzw must contain four finite values")
        if float(np.linalg.norm(quaternion)) <= EPS:
            raise ValueError("rotation_xyzw cannot be zero")
        if self.scale_xyz is None and self.target_size_xyz is None:
            raise ValueError("SceneReVis pose must contain scale_xyz or target_size_xyz")
        if self.scale_xyz is not None and np.any(_vec3(self.scale_xyz, "scale_xyz") <= 0):
            raise ValueError("scale_xyz must be positive")
        if self.target_size_xyz is not None and np.any(_vec3(self.target_size_xyz, "target_size_xyz") <= 0):
            raise ValueError("target_size_xyz must be positive")

    @property
    def normalized_rotation(self) -> np.ndarray:
        quaternion = np.asarray(self.rotation_xyzw, dtype=np.float64)
        return quaternion / np.linalg.norm(quaternion)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _balanced_json_blocks(text: str) -> Iterator[str]:
    for start_char, end_char in (("[", "]"), ("{", "}")):
        for match in re.finditer(re.escape(start_char), text):
            start = match.start()
            depth = 0
            in_string = False
            escaped = False
            for index in range(start, len(text)):
                char = text[index]
                if in_string:
                    if escaped:
                        escaped = False
                    elif char == "\\":
                        escaped = True
                    elif char == '"':
                        in_string = False
                    continue
                if char == '"':
                    in_string = True
                elif char == start_char:
                    depth += 1
                elif char == end_char:
                    depth -= 1
                    if depth == 0:
                        yield text[start:index + 1]
                        break


def _find_add_object_arguments(value: Any) -> Optional[Mapping[str, Any]]:
    if isinstance(value, list):
        for item in value:
            found = _find_add_object_arguments(item)
            if found is not None:
                return found
        return None
    if not isinstance(value, Mapping):
        return None
    name = str(value.get("name", value.get("tool", ""))).lower()
    arguments = value.get("arguments")
    if name in {"add_object", "place_object", "replace_object"} and isinstance(arguments, Mapping):
        return arguments
    if any(
        key in value
        for key in (
            "rotation",
            "rotation_xyzw",
            "rot",
            "scale",
            "scale_xyz",
            "size",
            "target_size",
            "target_size_xyz",
        )
    ):
        return value
    for child in value.values():
        found = _find_add_object_arguments(child)
        if found is not None:
            return found
    return None


def _rotation_to_quaternion(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    if array.shape == (4,):
        norm = float(np.linalg.norm(array))
        return array / norm if norm > EPS else None
    if array.shape == (3,):
        return Rotation.from_euler("xyz", array, degrees=False).as_quat()
    if array.shape in {(), (1,)}:
        return Rotation.from_euler("y", float(array.reshape(-1)[0]), degrees=False).as_quat()
    return None


def _positive_vec3(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    if array.shape in {(), (1,)}:
        array = np.repeat(float(array.reshape(-1)[0]), 3)
    if array.shape != (3,) or not np.all(np.isfinite(array)) or np.any(array <= 0):
        return None
    return array


def parse_scenerevis_pose(record: Any, source: str = "scenerevis") -> Optional[SceneReVisPose]:
    """Parse rotation and scale/size from common SceneReVis result formats."""
    candidates: list[Any] = []
    if isinstance(record, str):
        tool_match = re.search(r"<tool_calls>\s*(.*?)\s*</tool_calls>", record, flags=re.DOTALL)
        if tool_match:
            candidates.append(tool_match.group(1))
        candidates.extend(_balanced_json_blocks(record))
    else:
        candidates.append(record)

    for candidate in candidates:
        if isinstance(candidate, str):
            try:
                candidate = json.loads(candidate)
            except (TypeError, json.JSONDecodeError):
                continue
        arguments = _find_add_object_arguments(candidate)
        if arguments is None:
            continue
        rotation = _rotation_to_quaternion(
            arguments.get(
                "rotation",
                arguments.get("rotation_xyzw", arguments.get("rot", arguments.get("new_rotation"))),
            )
        )
        scale = _positive_vec3(
            arguments.get("scale", arguments.get("scale_xyz", arguments.get("scale_factor")))
        )
        size = _positive_vec3(
            arguments.get("size", arguments.get("target_size", arguments.get("target_size_xyz")))
        )
        if rotation is None or (scale is None and size is None):
            continue
        return SceneReVisPose(
            rotation_xyzw=tuple(float(v) for v in rotation),
            scale_xyz=tuple(float(v) for v in scale) if scale is not None else None,
            target_size_xyz=tuple(float(v) for v in size) if size is not None else None,
            source=source,
        )
    return None


def load_scenerevis_pose_records(path: Path | str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, Mapping) and isinstance(data.get("results"), list):
        data = data["results"]
    if isinstance(data, list):
        records: Dict[str, Any] = {}
        for item in data:
            if not isinstance(item, Mapping):
                continue
            key = item.get("sample_id", item.get("id", item.get("cache_key")))
            if key is not None:
                records[str(key)] = item
        return records
    if isinstance(data, Mapping):
        return {str(key): value for key, value in data.items()}
    raise ValueError(f"Unsupported SceneReVis result format in {path}")


def lookup_scenerevis_pose(records: Mapping[str, Any], sample_id: str) -> Optional[SceneReVisPose]:
    keys = [sample_id]
    if sample_id.startswith("obj_"):
        keys.append(sample_id[4:])
    for key in keys:
        if key not in records:
            continue
        record = records[key]
        if isinstance(record, Mapping) and record.get("response"):
            parsed = parse_scenerevis_pose(record["response"], source=f"scenerevis:{key}")
            if parsed is not None:
                return parsed
        parsed = parse_scenerevis_pose(record, source=f"scenerevis:{key}")
        if parsed is not None:
            return parsed
    return None


@dataclass
class PreparedTarget:
    mesh: trimesh.Trimesh
    original_to_prepared: np.ndarray
    bottom_center_before_normalization: np.ndarray


def prepare_target_mesh(mesh: trimesh.Trimesh, pose: SceneReVisPose) -> PreparedTarget:
    """Apply SceneReVis scale/size and rotation, then move the bottom center to zero."""
    prepared = mesh.copy()
    scale = np.ones(3, dtype=np.float64)
    if pose.target_size_xyz is not None:
        extents = np.asarray(prepared.extents, dtype=np.float64)
        if np.any(extents <= EPS):
            raise ValueError("Target mesh has a degenerate extent")
        scale = np.asarray(pose.target_size_xyz, dtype=np.float64) / extents
    elif pose.scale_xyz is not None:
        scale = np.asarray(pose.scale_xyz, dtype=np.float64)

    scale_matrix = np.eye(4, dtype=np.float64)
    scale_matrix[:3, :3] = np.diag(scale)
    rotation_matrix = np.eye(4, dtype=np.float64)
    rotation_matrix[:3, :3] = Rotation.from_quat(pose.normalized_rotation).as_matrix()
    prepared.apply_transform(scale_matrix)
    prepared.apply_transform(rotation_matrix)

    minimum, maximum = prepared.bounds
    bottom_center = np.array(
        [(minimum[0] + maximum[0]) * 0.5, minimum[1], (minimum[2] + maximum[2]) * 0.5],
        dtype=np.float64,
    )
    normalization = np.eye(4, dtype=np.float64)
    normalization[:3, 3] = -bottom_center
    prepared.apply_transform(normalization)
    original_to_prepared = normalization @ rotation_matrix @ scale_matrix
    return PreparedTarget(
        mesh=prepared,
        original_to_prepared=original_to_prepared,
        bottom_center_before_normalization=bottom_center,
    )


def _triangle_box_overlap(
    triangle_xyz: np.ndarray,
    box_centers_xyz: np.ndarray,
    half_extent: float,
) -> np.ndarray:
    """Vectorized 13-axis SAT test for one triangle against equal AABBs."""
    triangle = np.asarray(triangle_xyz, dtype=np.float64)
    centers = np.asarray(box_centers_xyz, dtype=np.float64).reshape(-1, 3)
    if triangle.shape != (3, 3):
        raise ValueError("triangle_xyz must have shape (3, 3)")
    if not math.isfinite(half_extent) or half_extent <= 0:
        raise ValueError("half_extent must be positive")

    if len(centers) == 0:
        return np.zeros(0, dtype=bool)

    relative = triangle[None, :, :] - centers[:, None, :]
    box_overlap = np.all(
        (relative.min(axis=1) <= half_extent + 1e-12)
        & (relative.max(axis=1) >= -half_extent - 1e-12),
        axis=1,
    )
    if not np.any(box_overlap):
        return box_overlap

    edges = np.array(
        [
            triangle[1] - triangle[0],
            triangle[2] - triangle[1],
            triangle[0] - triangle[2],
        ]
    )
    box_axes = np.eye(3, dtype=np.float64)
    normal = np.cross(edges[0], triangle[2] - triangle[0])
    cross_axes = np.cross(edges[:, None, :], box_axes[None, :, :]).reshape(-1, 3)
    axes = np.concatenate([normal.reshape(1, 3), cross_axes], axis=0)
    axis_norm = np.linalg.norm(axes, axis=1)
    valid_axes = axis_norm > EPS
    axes = axes[valid_axes]
    axis_norm = axis_norm[valid_axes]
    if len(axes) == 0:
        return box_overlap

    selected = np.flatnonzero(box_overlap)
    projections = np.einsum("nvi,ai->nav", relative[selected], axes)
    projection_min = projections.min(axis=2)
    projection_max = projections.max(axis=2)
    box_radius = half_extent * np.abs(axes).sum(axis=1)
    tolerance = np.maximum(axis_norm * half_extent, 1.0) * 1e-12
    axis_overlap = (
        (projection_min <= box_radius[None, :] + tolerance[None, :])
        & (projection_max >= -box_radius[None, :] - tolerance[None, :])
    )
    box_overlap[selected] &= np.all(axis_overlap, axis=1)
    return box_overlap


def conservative_voxelize_mesh(
    mesh: trimesh.Trimesh,
    pitch: float,
    origin_xyz: Sequence[float],
    shape_xyz: Optional[Sequence[int]] = None,
    fill_watertight: bool = True,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Mark every voxel whose closed AABB intersects a mesh triangle."""
    if not math.isfinite(pitch) or pitch <= 0:
        raise ValueError("pitch must be positive")
    origin = _vec3(origin_xyz, "origin_xyz")
    shape = None if shape_xyz is None else np.asarray(shape_xyz, dtype=np.int64)
    if shape is not None and (shape.shape != (3,) or np.any(shape <= 0)):
        raise ValueError("shape_xyz must contain three positive integers")

    triangles = np.asarray(mesh.triangles, dtype=np.float64)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3) or len(triangles) == 0:
        raise ValueError("Mesh contains no valid triangles")

    mesh_lower = np.floor(
        (triangles.min(axis=(0, 1)) - origin - pitch * 1e-10) / pitch
    ).astype(np.int64)
    mesh_upper = np.floor(
        (triangles.max(axis=(0, 1)) - origin + pitch * 1e-10) / pitch
    ).astype(np.int64)
    if shape is not None:
        mesh_lower = np.maximum(mesh_lower, 0)
        mesh_upper = np.minimum(mesh_upper, shape - 1)
    if np.any(mesh_lower > mesh_upper):
        raise ValueError("Mesh does not intersect the voxel grid")

    mesh_shape = mesh_upper - mesh_lower + 1
    dense_size = int(np.prod(mesh_shape, dtype=np.int64))
    if dense_size > 64_000_000:
        raise MemoryError(
            f"Conservative surface rasterization requires {dense_size:,} cells; reduce voxel resolution"
        )
    surface_dense = np.zeros(tuple(int(v) for v in mesh_shape), dtype=bool)
    candidate_tests = 0
    boundary_epsilon = pitch * 1e-10
    for triangle in triangles:
        lower = np.floor((triangle.min(axis=0) - origin - boundary_epsilon) / pitch).astype(np.int64)
        upper = np.floor((triangle.max(axis=0) - origin + boundary_epsilon) / pitch).astype(np.int64)
        if shape is not None:
            lower = np.maximum(lower, 0)
            upper = np.minimum(upper, shape - 1)
        lower = np.maximum(lower, mesh_lower)
        upper = np.minimum(upper, mesh_upper)
        if np.any(lower > upper):
            continue

        counts = upper - lower + 1
        total = int(np.prod(counts, dtype=np.int64))
        yz_count = int(counts[1] * counts[2])
        for start in range(0, total, SAT_CHUNK_SIZE):
            flat = np.arange(start, min(start + SAT_CHUNK_SIZE, total), dtype=np.int64)
            local_x = flat // yz_count
            remainder = flat % yz_count
            local_y = remainder // counts[2]
            local_z = remainder % counts[2]
            indices = np.column_stack((local_x, local_y, local_z)) + lower
            local_indices = indices - mesh_lower
            pending = ~surface_dense[tuple(local_indices.T)]
            if not np.any(pending):
                continue
            indices = indices[pending]
            local_indices = local_indices[pending]
            centers = origin + (indices.astype(np.float64) + 0.5) * pitch
            intersects = _triangle_box_overlap(triangle, centers, pitch * 0.5)
            candidate_tests += len(indices)
            if np.any(intersects):
                selected_local = local_indices[intersects]
                surface_dense[tuple(selected_local.T)] = True

    if not np.any(surface_dense):
        raise ValueError("Conservative voxelization produced no occupied cells")
    surface = np.argwhere(surface_dense).astype(np.int64) + mesh_lower
    occupied = surface

    if fill_watertight and mesh.is_watertight:
        padded_size = int(np.prod(mesh_shape + 2, dtype=np.int64))
        if padded_size > 64_000_000:
            raise MemoryError(
                f"Watertight fill requires {padded_size:,} cells; reduce voxel resolution"
            )
        filled = ndimage.binary_fill_holes(np.pad(surface_dense, 1, mode="constant"))
        occupied = np.argwhere(filled).astype(np.int64) - 1 + mesh_lower
        if shape is not None:
            occupied = occupied[np.all((occupied >= 0) & (occupied < shape), axis=1)]

    return occupied, {
        "triangle_count": int(len(triangles)),
        "sat_candidate_tests": int(candidate_tests),
        "surface_voxels": int(len(surface)),
        "occupied_voxels": int(len(occupied)),
    }


@dataclass(frozen=True)
class TargetVoxelKernel:
    offsets_xyz: np.ndarray
    minimum_release_height_voxels: int
    pitch: float
    row_offsets_yz: np.ndarray = field(init=False, repr=False)
    packed_x_words: np.ndarray = field(init=False, repr=False)
    x_min: int = field(init=False)
    x_max: int = field(init=False)

    def __post_init__(self) -> None:
        offsets = np.asarray(self.offsets_xyz, dtype=np.int64)
        if offsets.ndim != 2 or offsets.shape[1] != 3 or len(offsets) == 0:
            raise ValueError("offsets_xyz must be a non-empty Nx3 array")
        if self.minimum_release_height_voxels < 0:
            raise ValueError("minimum_release_height_voxels must be non-negative")
        if not math.isfinite(self.pitch) or self.pitch <= 0:
            raise ValueError("pitch must be positive")

        offsets = np.unique(offsets, axis=0)
        x_min = int(offsets[:, 0].min())
        x_max = int(offsets[:, 0].max())
        rows_yz, inverse = np.unique(offsets[:, [1, 2]], axis=0, return_inverse=True)
        local_x = offsets[:, 0] - x_min
        word_indices = local_x // 64
        bit_indices = local_x % 64
        packed = np.zeros(
            (len(rows_yz), int(word_indices.max()) + 1),
            dtype=np.uint64,
        )
        masks = np.left_shift(np.uint64(1), bit_indices.astype(np.uint64))
        np.bitwise_or.at(packed, (inverse, word_indices), masks)

        object.__setattr__(self, "offsets_xyz", offsets)
        object.__setattr__(self, "row_offsets_yz", rows_yz)
        object.__setattr__(self, "packed_x_words", packed)
        object.__setattr__(self, "x_min", x_min)
        object.__setattr__(self, "x_max", x_max)


def voxelize_target_kernel(
    prepared_mesh: trimesh.Trimesh,
    pitch: float,
    minimum_release_height_voxels: int = 1,
    fill_watertight: bool = True,
) -> TargetVoxelKernel:
    offsets, _ = conservative_voxelize_mesh(
        mesh=prepared_mesh,
        pitch=pitch,
        origin_xyz=(-0.5 * pitch, -0.5 * pitch, -0.5 * pitch),
        shape_xyz=None,
        fill_watertight=fill_watertight,
    )
    offsets[:, 1] -= int(offsets[:, 1].min())
    offsets[:, 1] += int(minimum_release_height_voxels)
    offsets = np.unique(offsets, axis=0)
    return TargetVoxelKernel(
        offsets_xyz=offsets,
        minimum_release_height_voxels=minimum_release_height_voxels,
        pitch=pitch,
    )


def _iter_world_meshes(scene: trimesh.Scene | trimesh.Trimesh) -> Iterable[trimesh.Trimesh]:
    if isinstance(scene, trimesh.Trimesh):
        yield scene
        return
    dumped = scene.dump(concatenate=False)
    if isinstance(dumped, trimesh.Trimesh):
        yield dumped
        return
    for geometry in dumped:
        if isinstance(geometry, trimesh.Trimesh) and len(geometry.faces) > 0:
            yield geometry


def concatenate_scene_mesh(scene: trimesh.Scene | trimesh.Trimesh) -> trimesh.Trimesh:
    meshes = list(_iter_world_meshes(scene))
    if not meshes:
        raise ValueError("Scene contains no triangle meshes")
    return trimesh.util.concatenate(meshes)


def voxelize_scene(
    scene: trimesh.Scene | trimesh.Trimesh,
    spec: VoxelGridSpec,
    fill_watertight: bool = True,
) -> Tuple[PackedVoxelGrid, Dict[str, Any]]:
    grid = PackedVoxelGrid(spec)
    mesh_count = 0
    input_points = 0
    set_voxels = 0
    failures = []
    triangle_count = 0
    sat_candidate_tests = 0
    surface_voxels = 0
    for mesh_index, mesh in enumerate(_iter_world_meshes(scene)):
        mesh_count += 1
        try:
            indices, stats = conservative_voxelize_mesh(
                mesh=mesh,
                pitch=spec.pitch,
                origin_xyz=spec.origin_xyz,
                shape_xyz=spec.shape_xyz,
                fill_watertight=fill_watertight,
            )
            input_points += len(indices)
            set_voxels += grid.set_indices(indices)
            triangle_count += stats["triangle_count"]
            sat_candidate_tests += stats["sat_candidate_tests"]
            surface_voxels += stats["surface_voxels"]
        except Exception as exc:
            failures.append({"mesh_index": mesh_index, "error": str(exc)})
    return grid, {
        "mesh_count": mesh_count,
        "voxelized_points": int(input_points),
        "set_voxel_operations": int(set_voxels),
        "triangle_count": int(triangle_count),
        "sat_candidate_tests": int(sat_candidate_tests),
        "surface_voxels": int(surface_voxels),
        "voxelization_method": "conservative_triangle_aabb_sat_v1",
        "packed_memory_bytes": grid.nbytes,
        "failures": failures,
    }


class FirstHitProjector:
    """First-hit projection using exact triangle-mesh ray intersections."""

    def __init__(
        self,
        scene_mesh: trimesh.Trimesh,
        occupancy: PackedVoxelGrid,
    ):
        self.scene_mesh = scene_mesh
        self.occupancy = occupancy
        try:
            import rtree  # noqa: F401
            from trimesh.ray.ray_triangle import RayMeshIntersector

            self.intersector = RayMeshIntersector(scene_mesh)
            # Build the spatial index now instead of failing halfway through a run.
            _ = self.intersector.intersects_any(
                np.array([[0.0, 0.0, 0.0]]),
                np.array([[1.0, 0.0, 0.0]]),
            )
        except Exception as exc:
            raise RuntimeError(
                "Exact first-hit projection requires trimesh with rtree support. "
                "Install it in the WSL environment with `pip install rtree`."
            ) from exc

    def first_hit(
        self,
        origin_world_xyz: Sequence[float],
        direction_world_xyz: Sequence[float],
    ) -> Optional[RayHit]:
        origin = _vec3(origin_world_xyz, "ray origin")
        direction = _vec3(direction_world_xyz, "ray direction")
        direction = direction / np.linalg.norm(direction)
        locations, ray_ids, _ = self.intersector.intersects_location(
            ray_origins=origin.reshape(1, 3),
            ray_directions=direction.reshape(1, 3),
            multiple_hits=False,
        )
        if not len(locations) or not len(ray_ids):
            return None
        point = np.asarray(locations[0], dtype=np.float64)
        index = self.occupancy.spec.world_to_grid(point)
        if not bool(self.occupancy.spec.contains(index)):
            return None
        return RayHit(
            index_xyz=tuple(int(v) for v in index),
            point_world_xyz=tuple(float(v) for v in point),
            distance=float(np.linalg.norm(point - origin)),
            source="mesh_ray",
        )


@dataclass(frozen=True)
class ReleaseSearchResult:
    heatmap_pixel_xy: Tuple[float, float]
    heatmap_index_rc: Tuple[int, int]
    heatmap_score: float
    image_pixel_xy: Tuple[float, float]
    surface_hit_index_xyz: Tuple[int, int, int]
    surface_hit_world_xyz: Tuple[float, float, float]
    release_kernel_anchor_xyz: Tuple[int, int, int]
    release_bottom_world_xyz: Tuple[float, float, float]
    tested_candidates: int
    tested_release_poses: int
    duplicate_surface_hits: int
    no_surface_hits: int
    first_hit_source: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def score_ordered_release_search(
    heatmap: np.ndarray,
    camera: CameraModel,
    projector: FirstHitProjector,
    target_kernel: TargetVoxelKernel,
    max_candidates: int = 65536,
    minimum_score: Optional[float] = None,
) -> Optional[ReleaseSearchResult]:
    """Return the highest-scoring candidate feasible at the fixed release height."""
    values = np.asarray(heatmap, dtype=np.float64)
    if values.ndim != 2 or values.size == 0:
        raise ValueError("heatmap must be a non-empty 2D array")
    finite = np.isfinite(values)
    if not np.any(finite):
        return None
    flat = np.where(finite, values, -np.inf).reshape(-1)
    order = np.argsort(flat, kind="stable")[::-1]
    max_candidates = min(max(0, int(max_candidates)), len(order))

    seen_surface_hits: set[Tuple[int, int, int]] = set()
    tested_candidates = 0
    tested_release_poses = 0
    duplicate_surface_hits = 0
    no_surface_hits = 0
    heatmap_h, heatmap_w = values.shape

    for flat_index in order[:max_candidates]:
        score = float(flat[flat_index])
        if minimum_score is not None and score < minimum_score:
            break
        row, col = divmod(int(flat_index), heatmap_w)
        tested_candidates += 1
        origin, direction, image_pixel = camera.ray_for_heatmap_pixel(
            row=row,
            col=col,
            heatmap_shape=(heatmap_h, heatmap_w),
        )
        hit = projector.first_hit(origin, direction)
        if hit is None:
            no_surface_hits += 1
            continue
        if hit.index_xyz in seen_surface_hits:
            duplicate_surface_hits += 1
            continue
        seen_surface_hits.add(hit.index_xyz)

        anchor = np.asarray(hit.index_xyz, dtype=np.int64)
        tested_release_poses += 1
        if projector.occupancy.intersects_kernel(anchor, target_kernel):
            continue

        anchor_world = projector.occupancy.spec.grid_to_world(anchor)
        release_bottom = anchor_world + np.array(
            [
                0.0,
                target_kernel.minimum_release_height_voxels * target_kernel.pitch,
                0.0,
            ],
            dtype=np.float64,
        )
        return ReleaseSearchResult(
            heatmap_pixel_xy=(float(col), float(row)),
            heatmap_index_rc=(int(row), int(col)),
            heatmap_score=score,
            image_pixel_xy=(float(image_pixel[0]), float(image_pixel[1])),
            surface_hit_index_xyz=hit.index_xyz,
            surface_hit_world_xyz=hit.point_world_xyz,
            release_kernel_anchor_xyz=tuple(int(v) for v in anchor),
            release_bottom_world_xyz=tuple(float(v) for v in release_bottom),
            tested_candidates=tested_candidates,
            tested_release_poses=tested_release_poses,
            duplicate_surface_hits=duplicate_surface_hits,
            no_surface_hits=no_surface_hits,
            first_hit_source=hit.source,
        )
    return None


def compose_release_transform(
    release_bottom_world_xyz: Sequence[float],
    original_to_prepared: np.ndarray,
) -> np.ndarray:
    prepared_to_world = np.eye(4, dtype=np.float64)
    prepared_to_world[:3, 3] = _vec3(release_bottom_world_xyz, "release_bottom_world_xyz")
    return prepared_to_world @ _matrix4(original_to_prepared, "original_to_prepared")


def decompose_transform(transform: np.ndarray) -> Dict[str, Any]:
    """Decompose an affine transform into translation, quaternion, and XYZ scale."""
    matrix = _matrix4(transform, "transform")
    linear = matrix[:3, :3]
    scale = np.linalg.norm(linear, axis=0)
    if np.any(scale <= EPS):
        raise ValueError("Cannot decompose a transform with a degenerate scale")
    rotation_matrix = linear / scale
    if np.linalg.det(rotation_matrix) < 0:
        axis = int(np.argmax(scale))
        scale[axis] *= -1.0
        rotation_matrix[:, axis] *= -1.0
    quaternion = Rotation.from_matrix(rotation_matrix).as_quat()
    return {
        "position_xyz": matrix[:3, 3].tolist(),
        "rotation_xyzw": quaternion.tolist(),
        "scale_xyz": scale.tolist(),
        "transform": matrix.tolist(),
    }
