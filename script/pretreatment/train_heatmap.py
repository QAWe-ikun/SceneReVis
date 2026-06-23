"""
SceneReVis 热力图放置模型训练脚本

训练数据格式 (由 generate_data.py 生成):
  {data_dir}/{split}/{split}.json
  每个样本:
    {
      "sample_id": "obj_xxx",
      "scene_dir": "train/scene_001",
      "plane_image_path": "plane_images/obj_xxx.png",
      "mask_path": "masks/obj_xxx_mask.png",
      "object_image_path": "object_images/obj_xxx_object.png",
      "object_desc": "a wooden chair",
      "split": "train"
    }

用法:
  python train_heatmap.py --data_dir /path/to/heatmap_data --epochs 100 --batch_size 8
"""
import os
import sys
import json
import argparse
import logging
from collections import deque
from pathlib import Path
from datetime import datetime
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.placement_heatmap import (
    DINO_IMAGE_SIZE,
    SIGLIP_IMAGE_SIZE,
    PlacementHeatmap,
    build_dino_image_transform,
    build_siglip_image_transform,
    load_trainable_heatmap_state_dict,
    trainable_heatmap_state_dict,
)


# ============================================================================
# Dataset
# ============================================================================

class HeatmapDataset(Dataset):
    """热力图训练数据集"""

    def __init__(
        self,
        data_dir: Path,
        split: str = "train",
        image_size: int = SIGLIP_IMAGE_SIZE,
        room_encoder: str = "siglip",
        room_image_size: int | None = None,
        object_image_size: int | None = None,
    ):
        """
        Args:
            data_dir: 数据根目录
            split: train/val/test
            image_size: 图像分辨率 (应匹配 SigLIP 模型输入尺寸 384)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.image_size = image_size
        self.room_encoder = room_encoder.lower()
        self.room_image_size = room_image_size or (
            DINO_IMAGE_SIZE if self.room_encoder == "dinov2" else image_size
        )
        self.object_image_size = object_image_size or image_size

        # 加载元数据
        json_path = self.data_dir / split / f"{split}.json"
        with open(json_path, 'r', encoding='utf-8') as f:
            self.samples = json.load(f)

        logging.info(f"Loaded {len(self.samples)} samples from {json_path}")

        # 图像预处理 (SigLIP 标准)
        # 注意：图像会被 resize 到 image_size (默认 384，匹配 SigLIP 输入)
        if self.room_encoder == "dinov2":
            self.room_transform = build_dino_image_transform(self.room_image_size)
        else:
            self.room_transform = build_siglip_image_transform(self.room_image_size)
        self.object_transform = build_siglip_image_transform(self.object_image_size)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        scene_dir = self.data_dir / sample["scene_dir"]

        # 加载房间俯视图
        room_path = scene_dir / sample["plane_image_path"]
        room_img = Image.open(room_path).convert("RGB")
        room_tensor = self.room_transform(room_img)

        # 加载物体参考图
        object_path = scene_dir / sample["object_image_path"]
        object_img = Image.open(object_path).convert("RGB")
        object_tensor = self.object_transform(object_img)

        # 加载 GT 热力图 (mask)
        mask_path = scene_dir / sample["mask_path"]
        mask_img = Image.open(mask_path).convert("L")
        mask_tensor = T.functional.to_tensor(mask_img)  # [1, H, W], 值域 [0, 1]

        # 文本描述
        object_desc = sample["object_desc"]

        result = {
            "room_image": room_tensor,       # [3, H, W]
            "object_image": object_tensor,   # [3, H, W]
            "mask": mask_tensor,             # [1, H, W]
            "object_desc": object_desc,
            "sample_id": sample["sample_id"],
        }

        # 透传新增的可选元数据字段 (scene_name, removed_object, text_source)
        for key in ("scene_name", "removed_object", "text_source"):
            if key in sample:
                result[key] = sample[key]

        return result


def heatmap_collate_fn(batch: List[Dict]) -> Dict:
    """Collate tensors for training while keeping metadata as plain lists."""
    result = {
        "room_image": torch.stack([item["room_image"] for item in batch]),
        "object_image": torch.stack([item["object_image"] for item in batch]),
        "mask": torch.stack([item["mask"] for item in batch]),
        "object_desc": [item["object_desc"] for item in batch],
        "sample_id": [item["sample_id"] for item in batch],
    }

    for key in ("scene_name", "removed_object", "text_source"):
        if any(key in item for item in batch):
            result[key] = [item.get(key) for item in batch]

    return result


def heatmap_bce_loss(
    pred_heatmap: torch.Tensor,
    target_heatmap: torch.Tensor,
    pos_weight: float,
) -> torch.Tensor:
    """Weighted BCE loss for sparse Gaussian placement heatmaps."""
    weight = torch.where(
        target_heatmap > 0.1,
        torch.as_tensor(pos_weight, device=target_heatmap.device),
        torch.as_tensor(1.0, device=target_heatmap.device),
    )
    return F.binary_cross_entropy(pred_heatmap, target_heatmap, weight=weight)


def peak_distance(pred_heatmap: torch.Tensor, target_heatmap: torch.Tensor) -> float:
    """Return argmax distance in heatmap pixels for a single prediction."""
    pred_peak = torch.argmax(pred_heatmap.flatten()).item()
    gt_peak = torch.argmax(target_heatmap.flatten()).item()
    W = pred_heatmap.shape[-1]
    pred_y, pred_x = divmod(pred_peak, W)
    gt_y, gt_x = divmod(gt_peak, W)
    return ((pred_y - gt_y) ** 2 + (pred_x - gt_x) ** 2) ** 0.5


# ============================================================================
# Training Loop
# ============================================================================

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    batch_scheduler=None,
    pos_weight: float = 10.0,
    peak_tolerance: float = 32.0,
    peak_window: int = 200,
) -> tuple[float, float]:
    """训练一个 epoch

    Args:
        batch_scheduler: 如果提供, 每个 batch 后步进学习率 (test_lr 模式)
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    peak_correct = 0
    peak_total = 0
    recent_peak = deque(maxlen=max(1, peak_window))
    recent_dist = deque(maxlen=max(1, peak_window))

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]", leave=False)

    for batch in progress_bar:
        current_lr = optimizer.param_groups[0]['lr']
        room_images = batch["room_image"].to(device)
        object_images = batch["object_image"].to(device)
        masks = batch["mask"].to(device)
        object_descs = batch["object_desc"]

        optimizer.zero_grad()

        # 前向传播 (使用 tensor 接口)
        # 当前文本编码器不支持批量，逐样本处理
        batch_size = room_images.size(0)
        batch_loss = 0.0

        for i in range(batch_size):
            room_img = room_images[i:i+1]
            obj_img = object_images[i:i+1]
            mask = masks[i:i+1]
            desc = object_descs[i]

            pred_heatmap = model.forward_tensor(
                room_image=room_img,
                object_desc=desc,
                object_image=obj_img,
            )  # [1, H, W]

            # 计算损失
            mask_resized = F.interpolate(
                mask,
                size=pred_heatmap.shape[-2:],
                mode='bilinear',
                align_corners=False
            ).squeeze(1)  # [1, H, W]

            loss = heatmap_bce_loss(pred_heatmap, mask_resized, pos_weight=pos_weight)
            batch_loss += loss

            # 统计峰值准确率
            with torch.no_grad():
                dist = peak_distance(pred_heatmap[0], mask_resized[0])
                is_correct = dist < peak_tolerance
                if is_correct:
                    peak_correct += 1
                peak_total += 1
                recent_peak.append(float(is_correct))
                recent_dist.append(dist)

        batch_loss = batch_loss / batch_size
        batch_loss.backward()
        optimizer.step()

        # test_lr 模式: 每个 batch 后步进学习率
        if batch_scheduler is not None:
            batch_scheduler.step()

        total_loss += batch_loss.item()
        num_batches += 1

        avg_loss = total_loss / num_batches
        peak_acc = peak_correct / peak_total if peak_total > 0 else 0.0
        recent_peak_acc = sum(recent_peak) / len(recent_peak) if recent_peak else 0.0
        recent_dist_avg = sum(recent_dist) / len(recent_dist) if recent_dist else 0.0
        hm_min = pred_heatmap.min().item()
        hm_max = pred_heatmap.max().item()

        progress_bar.set_postfix({
            "loss": f"{batch_loss.item():.4f}",
            "avg": f"{avg_loss:.4f}",
            "peak": f"{peak_acc:.2%}",
            f"p{peak_window}": f"{recent_peak_acc:.2%}",
            f"d{peak_window}": f"{recent_dist_avg:.1f}",
            "hm": f"[{hm_min:.2f},{hm_max:.2f}]",
            "lr": f"{current_lr:.1e}",
        })

    avg_loss = total_loss / num_batches
    peak_acc = peak_correct / peak_total if peak_total > 0 else 0.0
    return avg_loss, peak_acc


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    epoch: int,
    pos_weight: float = 10.0,
    peak_tolerance: float = 32.0,
) -> tuple[float, float]:
    """验证，返回 (loss, peak_accuracy)

    peak_accuracy: 预测峰值位置与 GT 峰值位置的距离 < 32 像素的比例
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    peak_correct = 0
    peak_total = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]", leave=False)

    for batch in progress_bar:
        room_images = batch["room_image"].to(device)
        object_images = batch["object_image"].to(device)
        masks = batch["mask"].to(device)
        object_descs = batch["object_desc"]

        batch_size = room_images.size(0)
        batch_loss = 0.0

        for i in range(batch_size):
            room_img = room_images[i:i+1]
            obj_img = object_images[i:i+1]
            mask = masks[i:i+1]
            desc = object_descs[i]

            pred_heatmap = model.forward_tensor(
                room_image=room_img,
                object_desc=desc,
                object_image=obj_img,
            )

            mask_resized = F.interpolate(
                mask,
                size=pred_heatmap.shape[-2:],
                mode='bilinear',
                align_corners=False
            ).squeeze(1)

            loss = heatmap_bce_loss(pred_heatmap, mask_resized, pos_weight=pos_weight)
            batch_loss += loss

            # 计算峰值准确率
            dist = peak_distance(pred_heatmap[0], mask_resized[0])
            if dist < peak_tolerance:
                peak_correct += 1
            peak_total += 1

        batch_loss = batch_loss / batch_size
        total_loss += batch_loss.item()
        num_batches += 1

        progress_bar.set_postfix({"loss": f"{batch_loss.item():.4f}"})

    avg_loss = total_loss / num_batches
    peak_acc = peak_correct / peak_total if peak_total > 0 else 0.0
    return avg_loss, peak_acc


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train heatmap placement model")
    parser.add_argument("--data_dir", type=str, required=True, help="数据目录")
    parser.add_argument("--output_dir", type=str, default="checkpoints/heatmap", help="输出目录")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=4, help="批量大小")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="权重衰减")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载线程数")
    parser.add_argument("--image_size", type=int, default=384, help="图像分辨率 (应匹配 SigLIP 输入)")
    parser.add_argument("--resume", type=str, default=None, help="恢复训练的检查点路径")
    parser.add_argument("--log_interval", type=int, default=10, help="日志间隔")
    parser.add_argument("--pos_weight", type=float, default=10.0,
                       help="GT heatmap positive-region BCE weight")
    parser.add_argument("--peak_tolerance", type=float, default=32.0,
                       help="Peak accuracy tolerance in 256x256 heatmap pixels")
    parser.add_argument("--peak_window", type=int, default=200,
                       help="Recent sample window for train progress peak/distance diagnostics")
    parser.add_argument("--early_stop_patience", type=int, default=0,
                       help="Stop after N epochs without val-loss improvement; 0 disables")
    parser.add_argument("--test_lr", action="store_true",
                       help="LR 测试模式: 1 epoch 内 cosine 衰减, 快速观察不同 lr 效果")
    parser.add_argument("--room_encoder", type=str, default="siglip", choices=["siglip", "dinov2"],
                       help="Room/top-view encoder: siglip or dinov2")
    parser.add_argument("--dino_model", type=str, default=None,
                       help="DINOv2 model path or HF id for room_encoder=dinov2")
    parser.add_argument("--room_image_size", type=int, default=None,
                       help="Room image input size; defaults to 518 for DINOv2 and image_size for SigLIP")
    parser.add_argument("--object_image_size", type=int, default=None,
                       help="Object image input size; defaults to image_size")
    parser.add_argument("--hidden_dim", type=int, default=256,
                       help="Trainable attention hidden dimension")
    args = parser.parse_args()

    # 设置日志
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"train_heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Training arguments: {args}")

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 数据集
    room_image_size = args.room_image_size or (
        DINO_IMAGE_SIZE if args.room_encoder == "dinov2" else args.image_size
    )
    object_image_size = args.object_image_size or args.image_size
    train_dataset = HeatmapDataset(
        Path(args.data_dir),
        split="train",
        image_size=args.image_size,
        room_encoder=args.room_encoder,
        room_image_size=room_image_size,
        object_image_size=object_image_size,
    )
    val_dataset = HeatmapDataset(
        Path(args.data_dir),
        split="val",
        image_size=args.image_size,
        room_encoder=args.room_encoder,
        room_image_size=room_image_size,
        object_image_size=object_image_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=heatmap_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=heatmap_collate_fn,
    )

    # 模型
    model = PlacementHeatmap(
        heatmap_res=256,
        room_encoder=args.room_encoder,
        dino_model=args.dino_model,
        hidden_dim=args.hidden_dim,
        room_image_size=room_image_size,
        object_image_size=object_image_size,
    ).to(device)
    logging.info(f"Model initialized: {sum(p.numel() for p in model.parameters()):,} parameters")

    # 优化器
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # 学习率调度器
    # test_lr 模式: 1 epoch 内 cosine 衰减, T_max = 总 batch 数
    if args.test_lr:
        total_batches = len(train_loader)
        scheduler = CosineAnnealingLR(optimizer, T_max=total_batches, eta_min=1e-6)
        args.epochs = 1
        logging.info(f"LR 测试模式: 1 epoch, {total_batches} batches, lr {args.lr} -> 1e-6")
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # 恢复训练
    start_epoch = 0
    best_val_loss = float('inf')
    best_peak_acc = 0.0
    epochs_without_improvement = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        missing_keys, unexpected_keys = load_trainable_heatmap_state_dict(
            model,
            checkpoint["model_state_dict"],
        )
        if missing_keys or unexpected_keys:
            logging.warning(
                "Checkpoint loaded with missing_keys=%s, unexpected_keys=%s",
                missing_keys,
                unexpected_keys,
            )
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint.get("best_val_loss", float('inf'))
        best_peak_acc = checkpoint.get("best_peak_acc", 0.0)
        logging.info(
            f"Resumed from epoch {start_epoch}, "
            f"best_val_loss={best_val_loss:.4f}, best_peak_acc={best_peak_acc:.2%}"
        )

    # 训练循环
    for epoch in range(start_epoch, args.epochs):
        logging.info(f"\n{'='*60}")
        logging.info(f"Epoch {epoch+1}/{args.epochs}")
        logging.info(f"{'='*60}")

        # 训练
        # test_lr 模式: 传入 scheduler, 每个 batch 后步进学习率
        batch_scheduler = scheduler if args.test_lr else None
        train_loss, train_peak_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch + 1,
            batch_scheduler,
            pos_weight=args.pos_weight,
            peak_tolerance=args.peak_tolerance,
            peak_window=args.peak_window,
        )
        logging.info(f"Train Loss: {train_loss:.4f}, Peak Acc: {train_peak_acc:.2%}")

        # 验证
        val_loss, peak_acc = validate(
            model,
            val_loader,
            device,
            epoch + 1,
            pos_weight=args.pos_weight,
            peak_tolerance=args.peak_tolerance,
        )
        logging.info(f"Val Loss: {val_loss:.4f}, Peak Acc: {peak_acc:.2%}")

        # 更新学习率 (test_lr 模式已在 batch 内步进, 跳过 epoch 级调度)
        if not args.test_lr:
            scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f"Learning Rate: {current_lr:.6f}")

        # 保存检查点
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": trainable_heatmap_state_dict(model),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "best_peak_acc": best_peak_acc,
            "args": vars(args),
        }

        # 保存最佳检查点
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            checkpoint["best_val_loss"] = best_val_loss
            torch.save(checkpoint, output_dir / "best.pth")
            logging.info(f"✓ New best model saved (val_loss={val_loss:.4f})")
        else:
            epochs_without_improvement += 1

        if peak_acc > best_peak_acc:
            best_peak_acc = peak_acc
            checkpoint["best_peak_acc"] = best_peak_acc
            torch.save(checkpoint, output_dir / "best_peak.pth")
            logging.info(f"✓ New best peak model saved (peak_acc={peak_acc:.2%})")

        checkpoint["best_val_loss"] = best_val_loss
        checkpoint["best_peak_acc"] = best_peak_acc

        # 保存最新检查点
        torch.save(checkpoint, output_dir / "latest.pth")

        # 定期保存
        if (epoch + 1) % 10 == 0:
            torch.save(checkpoint, output_dir / f"epoch_{epoch+1}.pth")

        if args.early_stop_patience > 0 and epochs_without_improvement >= args.early_stop_patience:
            logging.info(
                f"Early stopping: no val-loss improvement for "
                f"{epochs_without_improvement} epochs"
            )
            break

    logging.info(f"\n{'='*60}")
    logging.info(f"Training completed! Best val loss: {best_val_loss:.4f}")
    logging.info(f"Best peak accuracy: {best_peak_acc:.2%}")
    logging.info(f"Checkpoints saved to: {output_dir}")
    logging.info(f"{'='*60}")


if __name__ == "__main__":
    main()
