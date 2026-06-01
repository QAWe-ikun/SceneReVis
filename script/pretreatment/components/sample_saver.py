"""
样本保存模块 (精简版)
保存训练样本图片和元数据 JSON。

输出结构:
  {output_dir}/{split}/{scene_name}/
    plane_images/obj_{jid}.png      # 剔除目标物体后的房间俯视图
    object_images/obj_{jid}_object.png  # 目标物体参考图 (居中)
    masks/obj_{jid}_mask.png        # GT 高斯热力图 (灰度)
"""
import json
import logging
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class SampleSaver:
    """样本保存器"""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.sample_counter = 0
        self.samples_by_split: Dict[str, List[dict]] = {
            "train": [], "val": [], "test": []
        }

    def save_sample(
        self,
        scene_dir: Path,
        obj_id: str,
        plane_image: np.ndarray,
        heatmap: np.ndarray,
        object_desc: str,
        object_image: Optional[np.ndarray] = None,
        split: str = "train",
        scene_name: Optional[str] = None,
        original_image: Optional[np.ndarray] = None,
        removed_object: Optional[Dict] = None,
        text_source: str = "text_processor",
    ):
        """保存单个训练样本

        Args:
            scene_dir: 场景输出目录 (如 output/train/scene_001)
            obj_id: 物体 jid
            plane_image: 房间俯视图 RGB (H, W, 3)
            heatmap: GT 热力图 float (H, W) 值域 [0, 1]
            object_desc: 物体描述文本
            object_image: 物体参考图 RGB (H, W, 3)，可选
            split: 数据集划分
            scene_name: 场景名称，可选
            original_image: 完整场景俯视图 RGB (H, W, 3)，可选 (VLM 启用时)
            removed_object: 被移除物体的 3D 信息字典，可选
            text_source: 文本来源 ("vlm" 或 "text_processor")
        """
        self.sample_counter += 1
        sample_id = f"obj_{obj_id}"

        plane_dir = scene_dir / "plane_images"
        mask_dir = scene_dir / "masks"
        object_dir = scene_dir / "object_images"
        plane_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        if object_image is not None:
            object_dir.mkdir(parents=True, exist_ok=True)

        plane_path = plane_dir / f"{sample_id}.png"
        mask_path = mask_dir / f"{sample_id}_mask.png"
        object_path = object_dir / f"{sample_id}_object.png" if object_image is not None else None

        # 保存房间俯视图 (RGB)
        Image.fromarray(plane_image).save(plane_path)

        # 保存热力图 (灰度 uint8)
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        Image.fromarray(heatmap_uint8).save(mask_path)

        # 保存物体参考图 (RGB)
        if object_image is not None:
            Image.fromarray(object_image).save(object_path)

        # 保存完整场景俯视图 (可选)
        original_path = None
        if original_image is not None:
            original_dir = scene_dir / "original_images"
            original_dir.mkdir(parents=True, exist_ok=True)
            original_path = original_dir / f"{sample_id}_original.png"
            Image.fromarray(original_image).save(original_path)

        # 记录元数据
        metadata = {
            "sample_id": sample_id,
            "scene_dir": str(scene_dir.relative_to(self.output_dir)),
            "plane_image_path": f"plane_images/{sample_id}.png",
            "mask_path": f"masks/{sample_id}_mask.png",
            "object_desc": object_desc,
            "split": split,
        }
        if object_image is not None:
            metadata["object_image_path"] = f"object_images/{sample_id}_object.png"
        if scene_name is not None:
            metadata["scene_name"] = scene_name
        if original_image is not None:
            metadata["original_image_path"] = f"original_images/{sample_id}_original.png"
        if removed_object is not None:
            metadata["removed_object"] = removed_object
        metadata["text_source"] = text_source

        self.samples_by_split[split].append(metadata)

    def save_split_json(self, split_name: str) -> int:
        """保存 {split}.json 元数据文件 (追加模式)"""
        split_samples = self.samples_by_split[split_name]
        if not split_samples:
            return 0

        split_dir = self.output_dir / split_name
        output_path = split_dir / f"{split_name}.json"

        if output_path.exists():
            with open(output_path, 'r', encoding='utf-8') as f:
                existing = json.load(f)
            existing.extend(split_samples)
            total_count = len(existing)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(existing, f, ensure_ascii=False, indent=2)
        else:
            total_count = len(split_samples)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(split_samples, f, ensure_ascii=False, indent=2)

        logger.info(f"保存 {split_name} 数据: {len(split_samples)} 个样本 (总计 {total_count})")
        self.samples_by_split[split_name] = []  # 清空缓存避免重复追加
        return total_count

    def clear_split_samples(self, split_name: str):
        """清空指定 split 的缓存"""
        self.samples_by_split[split_name] = []

    def clear_all_samples(self):
        """清空所有 split 的缓存"""
        self.samples_by_split = {"train": [], "val": [], "test": []}
