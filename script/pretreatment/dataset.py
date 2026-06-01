"""
热力图放置模型数据集
加载训练样本：房间俯视图 + 物体参考图 + 文本描述 + GT 热力图
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class HeatmapPlacementDataset(Dataset):
    """热力图放置数据集

    数据格式:
        {split}.json 包含样本列表，每个样本:
        {
            "sample_id": "obj_xxx",
            "scene_dir": "train/scene_001",
            "plane_image_path": "plane_images/obj_xxx.png",
            "mask_path": "masks/obj_xxx_mask.png",
            "object_image_path": "object_images/obj_xxx_object.png",
            "object_desc": "a wooden chair",
            "split": "train"
        }
    """

    def __init__(
        self,
        data_dir: Path,
        split: str = "train",
        image_size: int = 1024,
        mask_size: int = 256,
        normalize: bool = True,
    ):
        """初始化数据集

        Args:
            data_dir: 数据根目录 (如 output/heatmap_data)
            split: 数据集划分 (train/val/test)
            image_size: 输入图像分辨率 (正方形)
            mask_size: 热力图分辨率 (正方形)
            normalize: 是否归一化图像到 [-1, 1]
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.image_size = image_size
        self.mask_size = mask_size
        self.normalize = normalize

        # 加载元数据 JSON
        json_path = self.data_dir / split / f"{split}.json"
        if not json_path.exists():
            raise FileNotFoundError(f"数据集文件不存在: {json_path}")

        with open(json_path, 'r', encoding='utf-8') as f:
            self.samples = json.load(f)

        logger.info(f"加载 {split} 数据集: {len(self.samples)} 个样本")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取单个样本

        Returns:
            dict:
                - room_image: (3, H, W) float32 归一化到 [-1, 1]
                - object_image: (3, H, W) float32 归一化到 [-1, 1]
                - mask: (1, mask_size, mask_size) float32 值域 [0, 1]
                - object_desc: str
                - sample_id: str
        """
        sample = self.samples[idx]
        scene_dir = self.data_dir / sample["scene_dir"]

        # 加载房间俯视图
        plane_image_path = scene_dir / sample["plane_image_path"]
        room_image = self._load_image(plane_image_path, target_size=self.image_size)

        # 加载物体参考图
        object_image_path = scene_dir / sample["object_image_path"]
        object_image = self._load_image(object_image_path, target_size=self.image_size)

        # 加载 GT 热力图
        mask_path = scene_dir / sample["mask_path"]
        mask = self._load_mask(mask_path, target_size=self.mask_size)

        # 图像归一化: [0, 255] -> [-1, 1]
        if self.normalize:
            room_image = (room_image / 127.5) - 1.0
            object_image = (object_image / 127.5) - 1.0

        # 转换为 tensor (H, W, C) -> (C, H, W)
        room_image = torch.from_numpy(room_image.transpose(2, 0, 1)).float()
        object_image = torch.from_numpy(object_image.transpose(2, 0, 1)).float()
        mask = torch.from_numpy(mask).float().unsqueeze(0)  # (1, H, W)

        result = {
            "room_image": room_image,
            "object_image": object_image,
            "mask": mask,
            "object_desc": sample["object_desc"],
            "sample_id": sample["sample_id"],
        }

        # 透传新增的可选元数据字段 (scene_name, removed_object, text_source)
        for key in ("scene_name", "removed_object", "text_source"):
            if key in sample:
                result[key] = sample[key]

        return result

    def _load_image(self, path: Path, target_size: int) -> np.ndarray:
        """加载 RGB 图像并调整尺寸

        Returns:
            numpy 数组 (H, W, 3) uint8
        """
        img = Image.open(path).convert("RGB")
        if img.size != (target_size, target_size):
            img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
        return np.array(img)

    def _load_mask(self, path: Path, target_size: int) -> np.ndarray:
        """加载热力图并调整尺寸

        Returns:
            numpy 数组 (H, W) float32 值域 [0, 1]
        """
        mask = Image.open(path).convert("L")
        if mask.size != (target_size, target_size):
            mask = mask.resize((target_size, target_size), Image.Resampling.LANCZOS)
        return np.array(mask).astype(np.float32) / 255.0


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """自定义 collate 函数，处理文本描述

    Args:
        batch: 样本列表

    Returns:
        dict:
            - room_image: (B, 3, H, W)
            - object_image: (B, 3, H, W)
            - mask: (B, 1, mask_size, mask_size)
            - object_desc: list of str
            - sample_id: list of str
    """
    result = {
        "room_image": torch.stack([b["room_image"] for b in batch]),
        "object_image": torch.stack([b["object_image"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "object_desc": [b["object_desc"] for b in batch],
        "sample_id": [b["sample_id"] for b in batch],
    }

    # 透传新增的可选元数据字段
    for key in ("scene_name", "removed_object", "text_source"):
        if key in batch[0]:
            result[key] = [b[key] for b in batch]

    return result


if __name__ == "__main__":
    """测试数据集加载"""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./output/heatmap_data")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument("--mask_size", type=int, default=256)
    args = parser.parse_args()

    dataset = HeatmapPlacementDataset(
        data_dir=Path(args.data_dir),
        split=args.split,
        image_size=args.image_size,
        mask_size=args.mask_size,
    )

    print(f"数据集大小: {len(dataset)}")
    sample = dataset[0]
    print(f"room_image shape: {sample['room_image'].shape}")
    print(f"object_image shape: {sample['object_image'].shape}")
    print(f"mask shape: {sample['mask'].shape}")
    print(f"object_desc: {sample['object_desc']}")
    print(f"sample_id: {sample['sample_id']}")

    # 测试 DataLoader
    from torch.utils.data import DataLoader

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    batch = next(iter(loader))
    print(f"\nBatch shapes:")
    print(f"room_image: {batch['room_image'].shape}")
    print(f"object_image: {batch['object_image'].shape}")
    print(f"mask: {batch['mask'].shape}")
    print(f"object_desc: {batch['object_desc']}")
