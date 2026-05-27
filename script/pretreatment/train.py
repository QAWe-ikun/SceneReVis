"""
热力图放置模型训练脚本

用法:
    python train.py --data_dir ./output/heatmap_data --epochs 100 --batch_size 4
"""
import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# 添加项目根目录到 path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from dataset import HeatmapPlacementDataset, collate_fn
from utils.placement_heatmap import PlacementHeatmap

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(
            f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
            encoding="utf-8",
        ),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="热力图放置模型训练")

    # 数据
    parser.add_argument("--data_dir", type=str, default="./output/heatmap_data")
    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument("--mask_size", type=int, default=256)

    # 训练
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=4)

    # 模型
    parser.add_argument("--freeze_encoders", action="store_true",
                       help="冻结 SigLIP/CLIP 编码器，只训练融合层和热力图头")

    # 输出
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--eval_every", type=int, default=5)

    # 设备
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()


def freeze_encoder(model: PlacementHeatmap):
    """冻结编码器参数"""
    logger.info("冻结编码器参数...")

    # 冻结 SigLIP
    for param in model.siglip_encoder.parameters():
        param.requires_grad = False

    # 冻结 CLIP
    for param in model.clip_encoder.model.parameters():
        param.requires_grad = False

    # 只训练融合层、空间细化和热力图头
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"可训练参数: {trainable_params:,} / {total_params:,} "
                f"({trainable_params/total_params*100:.2f}%)")


def create_scheduler(optimizer, args):
    """创建学习率调度器 (warmup + cosine)"""
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=args.warmup_epochs,
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs - args.warmup_epochs,
        eta_min=1e-6,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[args.warmup_epochs],
    )
    return scheduler


def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(loader):
        room_images = batch["room_image"].to(device)
        object_images = batch["object_image"].to(device)
        masks = batch["mask"].to(device)
        descs = batch["object_desc"]

        # 前向传播
        heatmaps = model(room_images, descs, object_images)

        # 计算损失
        loss = criterion(heatmaps, masks)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if batch_idx % 10 == 0:
            logger.info(f"Epoch {epoch} [{batch_idx}/{len(loader)}] "
                       f"Loss: {loss.item():.4f}")

    avg_loss = total_loss / num_batches
    return avg_loss


@torch.no_grad()
def validate(model, loader, criterion, device):
    """验证"""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        room_images = batch["room_image"].to(device)
        object_images = batch["object_image"].to(device)
        masks = batch["mask"].to(device)
        descs = batch["object_desc"]

        heatmaps = model(room_images, descs, object_images)
        loss = criterion(heatmaps, masks)

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    return avg_loss


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("热力图放置模型训练")
    logger.info("=" * 60)
    logger.info(f"数据目录: {args.data_dir}")
    logger.info(f"输出目录: {args.output_dir}")
    logger.info(f"设备: {args.device}")
    logger.info(f"批大小: {args.batch_size}")
    logger.info(f"学习率: {args.lr}")
    logger.info(f"Epochs: {args.epochs}")

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存训练配置
    config_path = output_dir / "train_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)
    logger.info(f"训练配置已保存: {config_path}")

    # 数据集
    logger.info("\n加载数据集...")
    train_dataset = HeatmapPlacementDataset(
        data_dir=Path(args.data_dir),
        split="train",
        image_size=args.image_size,
        mask_size=args.mask_size,
    )
    val_dataset = HeatmapPlacementDataset(
        data_dir=Path(args.data_dir),
        split="val",
        image_size=args.image_size,
        mask_size=args.mask_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    logger.info(f"训练集: {len(train_dataset)} 样本")
    logger.info(f"验证集: {len(val_dataset)} 样本")

    # 模型
    logger.info("\n初始化模型...")
    model = PlacementHeatmap(device=args.device)

    if args.freeze_encoders:
        freeze_encoder(model)

    # 优化器
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # 学习率调度器
    scheduler = create_scheduler(optimizer, args)

    # 损失函数
    criterion = nn.MSELoss()

    # 训练循环
    logger.info("\n开始训练...")
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch}/{args.epochs}")
        logger.info(f"{'='*60}")

        # 训练
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, args.device, epoch
        )
        logger.info(f"训练损失: {train_loss:.4f}")

        # 更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"学习率: {current_lr:.6f}")

        # 验证
        if epoch % args.eval_every == 0:
            val_loss = validate(model, val_loader, criterion, args.device)
            logger.info(f"验证损失: {val_loss:.4f}")

            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = output_dir / "best_model.pth"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "args": vars(args),
                }, best_model_path)
                logger.info(f"✓ 最佳模型已保存: {best_model_path} (val_loss={val_loss:.4f})")

        # 定期保存 checkpoint
        if epoch % args.save_every == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch}.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_loss": train_loss,
                "args": vars(args),
            }, checkpoint_path)
            logger.info(f"Checkpoint 已保存: {checkpoint_path}")

    # 保存最终模型
    final_model_path = output_dir / "final_model.pth"
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "args": vars(args),
    }, final_model_path)
    logger.info(f"\n最终模型已保存: {final_model_path}")

    logger.info("\n" + "=" * 60)
    logger.info("训练完成!")
    logger.info(f"最佳验证损失: {best_val_loss:.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
