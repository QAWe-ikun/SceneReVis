"""
正交投影渲染模块
使用 pyrender 离屏渲染生成俯视图，与 SceneReVis Blender 正交相机保持一致。

相机参数对齐:
  ortho_scale = span * 1.2
  span = max(x_max - x_min, z_max - z_min)  (from bounds_bottom AABB)
  cx, cz = bounds_bottom AABB center

像素映射 (线性, 图像约定):
  col = ((world_x - cx) / ortho_scale + 0.5) * image_size   # 列 → world X
  row = ((world_z - cz) / ortho_scale + 0.5) * image_size   # 行 → world Z

  即: col=0 对应 min_x (图像左侧), row=0 对应 min_z (图像顶部)
"""
import os
import logging
import numpy as np
import trimesh
from pathlib import Path
from typing import List, Optional

# 无头环境使用 EGL 后端 (需要 libegl1-mesa-dev)
if not os.environ.get("PYOPENGL_PLATFORM"):
    os.environ["PYOPENGL_PLATFORM"] = "egl"

logger = logging.getLogger(__name__)


class OrthoRenderer:
    """正交投影场景渲染器"""

    def __init__(self, image_size: int = 1024):
        self.image_size = image_size

    @staticmethod
    def compute_room_params(bounds_bottom: List[List[float]]):
        """计算房间参数: (cx, cz, ortho_scale)

        与 SceneReVis blender_renderer.py 和 placement_mask.py 保持一致。
        """
        bounds = np.array(bounds_bottom)
        xs = bounds[:, 0]
        zs = bounds[:, 2]
        x_min, x_max = xs.min(), xs.max()
        z_min, z_max = zs.min(), zs.max()
        cx = (x_min + x_max) / 2
        cz = (z_min + z_max) / 2
        span = max(x_max - x_min, z_max - z_min, 1.0)
        ortho_scale = span * 1.2
        return cx, cz, ortho_scale

    def _build_topdown_camera_pose(self, cx: float, cz: float, floor_y: float):
        """构建正交俯视相机位姿矩阵 (4x4)

        相机位于 (cx, floor_y + height, cz)，朝下看。
        图像方向: 右=world +X, 上=world -Z
        """
        camera_height = 20.0  # 正交投影下高度不影响投影，只需高于场景
        right = np.array([1.0, 0.0, 0.0])
        up = np.array([0.0, 0.0, -1.0])
        back = np.array([0.0, 1.0, 0.0])  # -forward
        pos = np.array([cx, floor_y + camera_height, cz])

        cam_transform = np.eye(4)
        cam_transform[:3, 0] = right
        cam_transform[:3, 1] = up
        cam_transform[:3, 2] = back
        cam_transform[:3, 3] = pos
        return cam_transform

    def render_scene_ortho(
        self,
        scene: trimesh.Scene,
        cx: float,
        cz: float,
        ortho_scale: float,
        floor_y: float = 0.0,
    ) -> Optional[np.ndarray]:
        """用正交投影渲染场景

        Args:
            scene: trimesh 场景
            cx, cz: 房间中心 (世界坐标)
            ortho_scale: 正交缩放 (米)，相机可见范围 = ortho_scale x ortho_scale
            floor_y: 地面 Y 坐标

        Returns:
            RGB 图像 numpy 数组 (H, W, 3)，或 None
        """
        import pyrender

        try:
            cam_transform = self._build_topdown_camera_pose(cx, cz, floor_y)

            py_scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0])

            for geom_name, geom in scene.geometry.items():
                if isinstance(geom, trimesh.Trimesh):
                    try:
                        mesh = pyrender.Mesh.from_trimesh(geom)
                        py_scene.add(mesh, name=geom_name)
                    except Exception as e:
                        logger.debug(f"跳过网格 {geom_name}: {e}")
                        continue

            # 正交相机: xmag/ymag = ortho_scale / 2
            camera = pyrender.OrthographicCamera(
                xmag=ortho_scale / 2,
                ymag=ortho_scale / 2,
            )
            py_scene.add(camera, pose=cam_transform)

            # 光源与相机共位
            light = pyrender.DirectionalLight(
                color=[1.0, 1.0, 1.0],
                intensity=3.0,
            )
            py_scene.add(light, pose=cam_transform)

            r = pyrender.OffscreenRenderer(
                viewport_width=self.image_size,
                viewport_height=self.image_size,
            )
            color, _ = r.render(py_scene)
            r.delete()

            return color

        except Exception as e:
            logger.debug(f"pyrender 渲染失败: {e}")
            return None

    def render_top_view(
        self,
        scene: trimesh.Scene,
        bounds_bottom: List[List[float]],
    ) -> Optional[np.ndarray]:
        """渲染俯视图

        自动从 bounds_bottom 计算相机参数。
        """
        bounds = np.array(bounds_bottom)
        floor_y = bounds[0, 1]
        cx, cz, ortho_scale = self.compute_room_params(bounds_bottom)
        return self.render_scene_ortho(scene, cx, cz, ortho_scale, floor_y)

    def render_object_reference(
        self,
        mesh: trimesh.Trimesh,
        bounds_bottom: List[List[float]],
    ) -> Optional[np.ndarray]:
        """渲染物体参考图 (居中于原点，正交俯视)

        使用与房间相同的 ortho_scale，保持比例一致。
        """
        obj_mesh = mesh.copy()
        obj_mesh.apply_translation(-obj_mesh.centroid)

        obj_scene = trimesh.Scene()
        obj_scene.add_geometry(obj_mesh, geom_name="object")

        cx, cz, ortho_scale = self.compute_room_params(bounds_bottom)

        # 物体居中在原点，floor_y=0
        return self.render_scene_ortho(obj_scene, 0.0, 0.0, ortho_scale, 0.0)
