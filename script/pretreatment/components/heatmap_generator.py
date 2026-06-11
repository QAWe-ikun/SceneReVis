"""
GT 热力图生成模块 (正交投影, 图像像素约定)

生成的 GT 热力图与渲染的房间俯视图像素一一对应:
  heatmap[row, col] 其中:
    row (第一轴) → world_z 方向, row=0 对应 min_z (图像顶部)
    col (第二轴) → world_x 方向, col=0 对应 min_x (图像左侧)

像素映射 (与 renderer.py 一致):
  col = ((world_x - cx) / ortho_scale + 0.5) * image_size
  row = ((world_z - cz) / ortho_scale + 0.5) * image_size

反变换:
  world_x = cx + (col / image_size - 0.5) * ortho_scale
  world_z = cz + (row / image_size - 0.5) * ortho_scale

推理时 placement_mask 也需要使用相同的图像像素约定,
score = heatmap * mask 才能正确逐元素相乘。
"""
import numpy as np
from typing import List, Tuple


class HeatmapGenerator:
    """GT 热力图生成器 (图像像素约定)"""

    def __init__(
        self,
        image_size: int = 1024,
        sigma: float = 15.0,
    ):
        self.image_size = image_size
        self.sigma = sigma

    @staticmethod
    def compute_room_params(bounds_bottom: List[List[float]]) -> Tuple[float, float, float]:
        """计算房间参数，与 OrthoRenderer / placement_mask 保持一致"""
        bounds = np.array(bounds_bottom)
        xs = bounds[:, 0]
        zs = bounds[:, 2]
        cx = (xs.min() + xs.max()) / 2
        cz = (zs.min() + zs.max()) / 2
        span = max(xs.max() - xs.min(), zs.max() - zs.min(), 1.0)
        ortho_scale = span * 1.2
        return cx, cz, ortho_scale

    def world_to_pixel(
        self,
        world_x: float,
        world_z: float,
        cx: float,
        cz: float,
        ortho_scale: float,
    ) -> Tuple[float, float]:
        """世界坐标 → 图像像素坐标 (与 renderer.py 一致)

        Returns:
            (pixel_row, pixel_col) 浮点像素坐标
        """
        pixel_col = ((world_x - cx) / ortho_scale + 0.5) * self.image_size
        pixel_row = ((world_z - cz) / ortho_scale + 0.5) * self.image_size
        return pixel_row, pixel_col

    def generate(
        self,
        target_pos: List[float],
        bounds_bottom: List[List[float]],
        sigma: float = None,
    ) -> np.ndarray:
        """生成 GT 高斯热力图 (图像像素约定)

        输出 heatmap[row, col] 与渲染的房间俯视图像素一一对应:
          - row=0 在图像顶部 (对应 min world_z)
          - col=0 在图像左侧 (对应 min world_x)

        Args:
            target_pos: 物体世界坐标 [x, y, z]
            bounds_bottom: 房间地面多边形顶点
            sigma: 高斯标准差 (像素数)，默认使用 self.sigma

        Returns:
            float32 数组 (image_size, image_size)，值域 [0, 1]
        """
        if sigma is None:
            sigma = self.sigma

        cx, cz, ortho_scale = self.compute_room_params(bounds_bottom)
        peak_row, peak_col = self.world_to_pixel(
            target_pos[0], target_pos[2], cx, cz, ortho_scale
        )
        return self.generate_from_pixel(peak_row, peak_col, sigma=sigma)

    def generate_from_pixel(
        self,
        peak_row: float,
        peak_col: float,
        sigma: float = None,
    ) -> np.ndarray:
        """Generate a Gaussian heatmap directly from image pixel coordinates."""
        if sigma is None:
            sigma = self.sigma

        row_coords, col_coords = np.ogrid[:self.image_size, :self.image_size]
        heatmap = np.exp(
            -((row_coords - peak_row) ** 2 + (col_coords - peak_col) ** 2)
            / (2 * sigma ** 2)
        ).astype(np.float32)

        max_val = heatmap.max()
        if max_val > 0:
            heatmap /= max_val

        return heatmap

    def object_sigma_pixels(
        self,
        obj_size: List[float],
        bounds_bottom: List[List[float]],
    ) -> float:
        """Return the adaptive Gaussian sigma for an object's XZ footprint."""
        _, _, ortho_scale = self.compute_room_params(bounds_bottom)
        pixels_per_meter = self.image_size / ortho_scale
        diagonal = np.sqrt(obj_size[0] ** 2 + obj_size[2] ** 2)
        return max(diagonal * pixels_per_meter, 5.0)

    def generate_with_object_sigma(
        self,
        target_pos: List[float],
        obj_size: List[float],
        bounds_bottom: List[List[float]],
    ) -> np.ndarray:
        """根据物体尺寸自适应 sigma

        sigma = 物体 XZ 对角线在像素空间的投影
        """
        sigma_pixels = self.object_sigma_pixels(obj_size, bounds_bottom)
        return self.generate(target_pos, bounds_bottom, sigma=sigma_pixels)
