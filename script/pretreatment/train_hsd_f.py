"""
Train the HSD-F placement baseline.

HSD-F shares the same DINOv2/SigLIP encoders and two-way fusion decoder as
HAP-Place, but replaces the heatmap upsampling head with global pooling plus
an MLP coordinate head.
"""
import argparse
import logging
import math
import sys
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from train_heatmap import (  # noqa: E402
    HeatmapDataset,
    build_step_warmup_cosine_scheduler,
    heatmap_collate_fn,
)
from utils.placement_heatmap import (  # noqa: E402
    DINO_IMAGE_SIZE,
    PlacementHSDF,
    load_trainable_heatmap_state_dict,
    trainable_heatmap_state_dict,
)


def _gt_pixel_from_mask(mask: torch.Tensor) -> Tuple[float, float]:
    """Return (x, y) peak coordinate from a [1, H, W] mask tensor."""
    peak = torch.argmax(mask.flatten()).item()
    height, width = mask.shape[-2:]
    y, x = divmod(peak, width)
    return float(x), float(y)


def gt_pixel_center(
    batch: Dict,
    index: int,
    mask: torch.Tensor,
) -> Tuple[float, float]:
    """Return GT pixel center from the GT heatmap peak."""
    return _gt_pixel_from_mask(mask)


def normalize_xy(
    xy: Tuple[float, float],
    height: int,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    """Convert pixel-space (x, y) to normalized [0, 1] coordinates."""
    x, y = xy
    denom_x = max(width - 1, 1)
    denom_y = max(height - 1, 1)
    return torch.tensor(
        [x / denom_x, y / denom_y],
        dtype=torch.float32,
        device=device,
    ).clamp(0.0, 1.0)


def coord_distance_pixels(
    pred_xy_norm: torch.Tensor,
    target_xy_norm: torch.Tensor,
    eval_size: int,
) -> float:
    pred = pred_xy_norm.detach()
    target = target_xy_norm.detach()
    scale = torch.tensor(
        [max(eval_size - 1, 1), max(eval_size - 1, 1)],
        dtype=pred.dtype,
        device=pred.device,
    )
    return torch.linalg.vector_norm((pred - target) * scale).item()


def log_gt_sanity_check(dataset: HeatmapDataset, count: int = 5):
    """Log mask-peak targets for a few samples."""
    checks = []
    for idx in range(min(count, len(dataset))):
        item = dataset[idx]
        mask = item["mask"]
        mask_h, mask_w = mask.shape[-2:]
        peak = _gt_pixel_from_mask(mask)
        checks.append(
            f"{item['sample_id']}: mask={mask_w}x{mask_h}, "
            f"target_peak=({peak[0]:.1f},{peak[1]:.1f})"
        )
    logging.info("GT coordinate sanity check (training target = GT heatmap argmax):\n%s", "\n".join(checks))


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    batch_scheduler=None,
    peak_tolerance: float = 32.0,
    peak_window: int = 200,
    eval_size: int = 256,
) -> tuple[float, float, float]:
    model.train()
    total_loss = 0.0
    total_dist = 0.0
    num_batches = 0
    peak_correct = 0
    peak_total = 0
    recent_peak = deque(maxlen=max(1, peak_window))
    recent_dist = deque(maxlen=max(1, peak_window))

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]", leave=False)

    for batch in progress_bar:
        current_lr = optimizer.param_groups[0]["lr"]
        room_images = batch["room_image"].to(device)
        object_images = batch["object_image"].to(device)
        masks = batch["mask"].to(device)
        object_descs = batch["object_desc"]

        optimizer.zero_grad()
        batch_size = room_images.size(0)
        batch_loss = 0.0
        batch_dist = 0.0

        for i in range(batch_size):
            room_img = room_images[i:i + 1]
            obj_img = object_images[i:i + 1]
            mask = masks[i]
            desc = object_descs[i]
            height, width = mask.shape[-2:]

            pred_xy = model.forward_tensor(
                room_image=room_img,
                object_desc=desc,
                object_image=obj_img,
            )[0]
            target_xy = normalize_xy(
                gt_pixel_center(batch, i, mask),
                height=height,
                width=width,
                device=device,
            )

            loss = F.smooth_l1_loss(pred_xy, target_xy)
            batch_loss += loss

            with torch.no_grad():
                dist = coord_distance_pixels(pred_xy, target_xy, eval_size)
                is_correct = dist < peak_tolerance
                peak_correct += int(is_correct)
                peak_total += 1
                batch_dist += dist
                recent_peak.append(float(is_correct))
                recent_dist.append(dist)

        batch_loss = batch_loss / batch_size
        batch_loss.backward()
        optimizer.step()

        if batch_scheduler is not None:
            batch_scheduler.step()

        total_loss += batch_loss.item()
        total_dist += batch_dist / batch_size
        num_batches += 1

        avg_loss = total_loss / num_batches
        avg_dist = total_dist / num_batches
        peak_acc = peak_correct / peak_total if peak_total else 0.0
        recent_peak_acc = sum(recent_peak) / len(recent_peak) if recent_peak else 0.0
        recent_dist_avg = sum(recent_dist) / len(recent_dist) if recent_dist else 0.0

        progress_bar.set_postfix({
            "loss": f"{batch_loss.item():.4f}",
            "avg": f"{avg_loss:.4f}",
            "peak": f"{peak_acc:.2%}",
            f"p{peak_window}": f"{recent_peak_acc:.2%}",
            f"d{peak_window}": f"{recent_dist_avg:.1f}",
            "dist": f"{avg_dist:.1f}",
            "lr": f"{current_lr:.1e}",
        })

    avg_loss = total_loss / num_batches
    avg_dist = total_dist / num_batches
    peak_acc = peak_correct / peak_total if peak_total else 0.0
    return avg_loss, peak_acc, avg_dist


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    epoch: int,
    peak_tolerance: float = 32.0,
    eval_size: int = 256,
) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total_dist = 0.0
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
        batch_dist = 0.0

        for i in range(batch_size):
            room_img = room_images[i:i + 1]
            obj_img = object_images[i:i + 1]
            mask = masks[i]
            desc = object_descs[i]
            height, width = mask.shape[-2:]

            pred_xy = model.forward_tensor(
                room_image=room_img,
                object_desc=desc,
                object_image=obj_img,
            )[0]
            target_xy = normalize_xy(
                gt_pixel_center(batch, i, mask),
                height=height,
                width=width,
                device=device,
            )

            loss = F.smooth_l1_loss(pred_xy, target_xy)
            batch_loss += loss
            dist = coord_distance_pixels(pred_xy, target_xy, eval_size)
            batch_dist += dist
            peak_correct += int(dist < peak_tolerance)
            peak_total += 1

        batch_loss = batch_loss / batch_size
        total_loss += batch_loss.item()
        total_dist += batch_dist / batch_size
        num_batches += 1
        progress_bar.set_postfix({"loss": f"{batch_loss.item():.4f}"})

    avg_loss = total_loss / num_batches
    avg_dist = total_dist / num_batches
    peak_acc = peak_correct / peak_total if peak_total else 0.0
    return avg_loss, peak_acc, avg_dist


def main():
    parser = argparse.ArgumentParser(description="Train HSD-F placement baseline")
    parser.add_argument("--data_dir", type=str, required=True, help="Data directory")
    parser.add_argument("--output_dir", type=str, default="checkpoints/hsd_f", help="Output directory")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader worker count")
    parser.add_argument("--image_size", type=int, default=384, help="Image resolution for SigLIP inputs")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from")
    parser.add_argument("--peak_tolerance", type=float, default=32.0,
                        help="Peak accuracy tolerance in eval pixels")
    parser.add_argument("--eval_size", type=int, default=256,
                        help="Pixel scale for distance/peak metrics; HAP-Place uses 256")
    parser.add_argument("--peak_window", type=int, default=200,
                        help="Recent sample window for train progress diagnostics")
    parser.add_argument("--lr_scheduler", type=str, default="step_cosine",
                        choices=["step_cosine", "epoch_cosine"],
                        help="Learning-rate schedule")
    parser.add_argument("--warmup_steps", type=int, default=1000,
                        help="Warmup optimizer steps for step_cosine")
    parser.add_argument("--min_lr", type=float, default=1e-6,
                        help="Minimum learning rate for cosine schedules")
    parser.add_argument("--early_stop_patience", type=int, default=0,
                        help="Stop after N epochs without val-loss improvement; 0 disables")
    parser.add_argument("--room_encoder", type=str, default="siglip", choices=["siglip", "dinov2"],
                        help="Room/top-view encoder")
    parser.add_argument("--dino_model", type=str, default=None,
                        help="DINOv2 model path or HF id for room_encoder=dinov2")
    parser.add_argument("--room_image_size", type=int, default=None,
                        help="Room image input size; defaults to 518 for DINOv2 and image_size for SigLIP")
    parser.add_argument("--object_image_size", type=int, default=None,
                        help="Object image input size; defaults to image_size")
    parser.add_argument("--hidden_dim", type=int, default=256,
                        help="Trainable attention hidden dimension")
    parser.add_argument("--decoder_layers", type=int, default=3,
                        help="Number of two-way fusion decoder layers")
    parser.add_argument("--num_heads", type=int, default=8,
                        help="Number of attention heads in trainable decoder")
    parser.add_argument("--mlp_ratio", type=float, default=4.0,
                        help="MLP expansion ratio in trainable decoder blocks")
    parser.add_argument("--decoder_dropout", type=float, default=0.0,
                        help="Dropout used inside trainable decoder blocks")
    args = parser.parse_args()

    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"train_hsd_f_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    logging.info(f"Training arguments: {args}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
    log_gt_sanity_check(train_dataset)
    logging.info(
        "HSD-F metrics use eval_size=%d; peak_tolerance=%.1f is measured on this scale",
        args.eval_size,
        args.peak_tolerance,
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

    model = PlacementHSDF(
        room_encoder=args.room_encoder,
        dino_model=args.dino_model,
        hidden_dim=args.hidden_dim,
        room_image_size=room_image_size,
        object_image_size=object_image_size,
        decoder_layers=args.decoder_layers,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        decoder_dropout=args.decoder_dropout,
    ).to(device)
    logging.info(f"Model initialized: {sum(p.numel() for p in model.parameters()):,} parameters")

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    if args.lr_scheduler == "step_cosine":
        total_steps = len(train_loader) * args.epochs
        scheduler = build_step_warmup_cosine_scheduler(
            optimizer=optimizer,
            total_steps=total_steps,
            warmup_steps=args.warmup_steps,
            min_lr=args.min_lr,
        )
        scheduler_step_unit = "batch"
        logging.info(
            "LR scheduler: step_cosine, total_steps=%d, warmup_steps=%d, lr=%s -> %s",
            total_steps,
            min(args.warmup_steps, max(0, total_steps - 1)),
            args.lr,
            args.min_lr,
        )
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
        scheduler_step_unit = "epoch"
        logging.info(
            "LR scheduler: epoch_cosine, epochs=%d, lr=%s -> %s",
            args.epochs,
            args.lr,
            args.min_lr,
        )

    start_epoch = 0
    best_val_loss = math.inf
    best_peak_acc = 0.0
    best_val_dist = math.inf
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
        best_val_loss = checkpoint.get("best_val_loss", math.inf)
        best_peak_acc = checkpoint.get("best_peak_acc", 0.0)
        best_val_dist = checkpoint.get("best_val_dist", math.inf)
        logging.info(
            "Resumed from epoch %d, best_val_loss=%.4f, best_peak_acc=%.2%%",
            start_epoch,
            best_val_loss,
            best_peak_acc * 100,
        )

    for epoch in range(start_epoch, args.epochs):
        logging.info(f"\n{'=' * 60}")
        logging.info(f"Epoch {epoch + 1}/{args.epochs}")
        logging.info(f"{'=' * 60}")

        batch_scheduler = scheduler if scheduler_step_unit == "batch" else None
        train_loss, train_peak_acc, train_dist = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch + 1,
            batch_scheduler=batch_scheduler,
            peak_tolerance=args.peak_tolerance,
            peak_window=args.peak_window,
            eval_size=args.eval_size,
        )
        logging.info(
            "Train Loss: %.4f, Peak Acc: %.2f%%, Mean Dist: %.2f px",
            train_loss,
            train_peak_acc * 100,
            train_dist,
        )

        val_loss, peak_acc, val_dist = validate(
            model,
            val_loader,
            device,
            epoch + 1,
            peak_tolerance=args.peak_tolerance,
            eval_size=args.eval_size,
        )
        logging.info(
            "Val Loss: %.4f, Peak Acc: %.2f%%, Mean Dist: %.2f px",
            val_loss,
            peak_acc * 100,
            val_dist,
        )

        if scheduler_step_unit == "epoch":
            scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        logging.info(f"Learning Rate: {current_lr:.6f}")

        checkpoint = {
            "epoch": epoch,
            "model_type": "hsd_f",
            "model_state_dict": trainable_heatmap_state_dict(model),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "best_peak_acc": best_peak_acc,
            "best_val_dist": best_val_dist,
            "args": vars(args),
        }

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_dist = val_dist
            epochs_without_improvement = 0
            checkpoint["best_val_loss"] = best_val_loss
            checkpoint["best_val_dist"] = best_val_dist
            torch.save(checkpoint, output_dir / "best.pth")
            logging.info(f"New best model saved (val_loss={val_loss:.4f})")
        else:
            epochs_without_improvement += 1

        if peak_acc > best_peak_acc:
            best_peak_acc = peak_acc
            checkpoint["best_peak_acc"] = best_peak_acc
            torch.save(checkpoint, output_dir / "best_peak.pth")
            logging.info(f"New best peak model saved (peak_acc={peak_acc:.2%})")

        checkpoint["best_val_loss"] = best_val_loss
        checkpoint["best_peak_acc"] = best_peak_acc
        checkpoint["best_val_dist"] = best_val_dist
        torch.save(checkpoint, output_dir / "latest.pth")

        if (epoch + 1) % 10 == 0:
            torch.save(checkpoint, output_dir / f"epoch_{epoch + 1}.pth")

        if args.early_stop_patience > 0 and epochs_without_improvement >= args.early_stop_patience:
            logging.info(
                "Early stopping: no val-loss improvement for %d epochs",
                epochs_without_improvement,
            )
            break

    logging.info(f"\n{'=' * 60}")
    logging.info(f"Training completed! Best val loss: {best_val_loss:.4f}")
    logging.info(f"Best peak accuracy: {best_peak_acc:.2%}")
    logging.info(f"Checkpoints saved to: {output_dir}")
    logging.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()
