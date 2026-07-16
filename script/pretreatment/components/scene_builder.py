"""
场景构建模块
从 JSON 数据构建 3D 场景，加载网格文件。
"""
import trimesh
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..models import ObjectInfo
from scipy.spatial.transform import Rotation as R


class SceneBuilder:
    """场景构建器"""

    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        self._mesh_cache: Dict[Path, trimesh.Trimesh] = {}

    def find_model_path(self, model_id: str) -> Optional[Path]:
        """查找模型文件"""
        model_path = self.model_dir / model_id
        if not model_path.exists():
            return None
        if (model_path / "normalized_model.glb").exists():
            return model_path / "normalized_model.glb"
        if (model_path / "raw_model.glb").exists():
            return model_path / "raw_model.glb"
        glbs = list(model_path.glob("*.glb"))
        if glbs:
            return glbs[0]
        if (model_path / "raw_model.obj").exists():
            return model_path / "raw_model.obj"
        return None

    def parse_jid(self, jid: str) -> Tuple[str, float, float, float]:
        """解析 jid 获取模型 ID 和缩放

        格式: model_id-(sx)-(sy)-(sz)
        """
        parts = jid.split('-(')
        if len(parts) == 1:
            return jid, 1.0, 1.0, 1.0
        model_id = parts[0]
        try:
            sx = float(parts[1].rstrip(')'))
            sy = float(parts[2].rstrip(')'))
            sz = float(parts[3].rstrip(')'))
        except (ValueError, IndexError):
            return model_id, 1.0, 1.0, 1.0
        return model_id, sx, sy, sz

    def load_mesh(self, model_path: Path) -> Optional[trimesh.Trimesh]:
        """加载网格"""
        model_path = Path(model_path)
        cached = self._mesh_cache.get(model_path)
        if cached is not None:
            return cached.copy()
        try:
            loaded = trimesh.load(model_path, force='mesh')
            if not isinstance(loaded, trimesh.Trimesh):
                return None
            self._mesh_cache[model_path] = loaded.copy()
            return loaded
        except Exception:
            return None

    @staticmethod
    def triangulate_polygon(n: int, reverse: bool = False) -> np.ndarray:
        """扇形三角化多边形"""
        faces = []
        for i in range(1, n - 1):
            if reverse:
                faces.append([0, i + 1, i])
            else:
                faces.append([0, i, i + 1])
        return np.array(faces)

    @staticmethod
    def extract_bounds_bottom(scene_data: Dict) -> List[List[float]]:
        """从 scene JSON 中提取 bounds_bottom，兼容 flat 和 grouped 格式"""
        if 'room_envelope' in scene_data:
            return scene_data['room_envelope'].get('bounds_bottom', [])
        if 'bounds_bottom' in scene_data:
            return scene_data['bounds_bottom']
        return []

    @staticmethod
    def extract_objects(scene_data: Dict) -> List[Dict]:
        """从 scene JSON 中提取物体列表，兼容 flat 和 grouped 格式"""
        if 'objects' in scene_data:
            return scene_data['objects']
        if 'groups' in scene_data:
            objects = []
            for group in scene_data['groups']:
                objects.extend(group.get('objects', []))
            return objects
        return []

    def build_scene(self, scene_data: Dict) -> Tuple[trimesh.Scene, List[ObjectInfo]]:
        """构建完整场景，返回 trimesh.Scene 和物体列表"""
        scene = trimesh.Scene()

        bounds_bottom = self.extract_bounds_bottom(scene_data)

        # 添加地板
        if bounds_bottom:
            bounds_bottom_arr = np.array(bounds_bottom)
            floor_v = bounds_bottom_arr
            floor_f = self.triangulate_polygon(len(bounds_bottom))
            scene.add_geometry(
                trimesh.Trimesh(vertices=floor_v, faces=floor_f, process=False),
                geom_name="floor"
            )

        # 加载所有家具
        objects = []
        for obj_idx, obj_data in enumerate(self.extract_objects(scene_data)):
            desc = obj_data.get('desc', '')
            jid = obj_data.get('jid', '')
            pos = obj_data.get('pos', [0, 0, 0])
            rot = obj_data.get('rot', [0, 0, 0, 1])
            size = obj_data.get('size', [1, 1, 1])

            model_id, sx, sy, sz = self.parse_jid(jid)
            model_path = self.find_model_path(model_id)
            if not model_path or not model_path.exists():
                continue

            mesh = self.load_mesh(model_path)
            if mesh is None:
                continue

            # 应用 jid 缩放
            mesh.apply_scale([sx, sy, sz])

            # 判断是否在地面上
            floor_y = bounds_bottom[0][1] if bounds_bottom else 0.0
            is_on_floor = abs(pos[1] - floor_y) < 0.01

            # 创建变换副本加入场景
            mesh_transformed = mesh.copy()
            if len(rot) == 4 and not np.allclose(rot, [0, 0, 0, 1]):
                R_mat = np.eye(4)
                R_mat[:3, :3] = R.from_quat(rot).as_matrix()
                mesh_transformed.apply_transform(R_mat)

            T = np.eye(4)
            T[:3, 3] = pos
            mesh_transformed.apply_transform(T)

            geom_name = f"obj_{obj_idx:04d}_{jid}"
            scene.add_geometry(mesh_transformed, geom_name=geom_name)

            objects.append(ObjectInfo(
                instance_id=obj_idx,
                geom_name=geom_name,
                jid=jid,
                model_id=model_id,
                desc=desc,
                pos=pos,
                rot=rot,
                size=size,
                scale_jid=(sx, sy, sz),
                mesh=mesh,          # 原始 mesh（仅应用了 jid 缩放）
                is_on_floor=is_on_floor,
            ))

        return scene, objects
