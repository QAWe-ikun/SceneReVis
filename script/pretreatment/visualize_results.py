"""
Visualize training results: load checkpoint, compare predicted heatmap vs GT, show placement

Usage:
  python visualize_results.py \
    --data_dir /path/to/heatmap_data \
    --checkpoint checkpoints/test_lr_1e-4/latest.pth \
    --num_samples 10 \
    --output_dir visualizations
"""
import sys
import json
import argparse
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.placement_heatmap import PlacementHeatmap


def load_checkpoint(model, checkpoint_path, device):
    """Load checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    epoch = checkpoint["epoch"]
    best_val_loss = checkpoint.get("best_val_loss", float('inf'))
    logging.info(f"Loaded checkpoint: epoch={epoch}, best_val_loss={best_val_loss:.4f}")
    return model


def visualize_sample(model, sample, data_dir, device, output_path, image_size=384):
    """Visualize single sample: 2x4 layout

    [1,1] Original scene (all objects, only when VLM enabled)
    [1,2] Room top view (target object removed)
    [1,3] Object reference
    [1,4] GT heatmap
    [2,1] Predicted heatmap
    [2,2] GT placement overlay
    [2,3] Predicted placement overlay
    [2,4] Pred vs GT comparison (line shows distance)
    """
    scene_dir = data_dir / sample["scene_dir"]

    # Load images
    room_path = scene_dir / sample["plane_image_path"]
    object_path = scene_dir / sample["object_image_path"]
    mask_path = scene_dir / sample["mask_path"]

    room_img = Image.open(room_path).convert("RGB")
    object_img = Image.open(object_path).convert("RGB")
    mask_img = Image.open(mask_path).convert("L")

    # Load original scene image if available
    original_img = None
    if "original_image_path" in sample:
        original_path = scene_dir / sample["original_image_path"]
        if original_path.exists():
            original_img = Image.open(original_path).convert("RGB")

    # Preprocess
    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    room_tensor = transform(room_img).unsqueeze(0).to(device)
    object_tensor = transform(object_img).unsqueeze(0).to(device)
    mask_tensor = T.functional.to_tensor(mask_img).unsqueeze(0).to(device)

    # Inference
    model.eval()
    with torch.no_grad():
        pred_heatmap = model.forward_tensor(
            room_image=room_tensor,
            object_desc=sample["object_desc"],
            object_image=object_tensor,
        )

    # Resize GT to match prediction
    gt_heatmap = F.interpolate(
        mask_tensor,
        size=pred_heatmap.shape[-2:],
        mode='bilinear',
        align_corners=False
    ).squeeze()

    pred_heatmap = pred_heatmap.squeeze().cpu().numpy()
    gt_heatmap = gt_heatmap.cpu().numpy()

    # Compute peak position (col, row) = (x, y)
    pred_peak_idx = np.unravel_index(np.argmax(pred_heatmap), pred_heatmap.shape)
    gt_peak_idx = np.unravel_index(np.argmax(gt_heatmap), gt_heatmap.shape)
    pred_peak = (pred_peak_idx[1], pred_peak_idx[0])  # (col, row)
    gt_peak = (gt_peak_idx[1], gt_peak_idx[0])
    dist = np.sqrt((pred_peak[0] - gt_peak[0])**2 + (pred_peak[1] - gt_peak[1])**2)

    # Build title: scene name, text source, object description, peak distance
    scene_name = sample.get("scene_name", scene_dir.name)
    text_source = sample.get("text_source", "unknown")
    title_line1 = f'[{scene_name}] ({text_source}) {sample["object_desc"]}'
    title_line2 = f'Peak Distance: {dist:.1f}px | Pred: {pred_peak} | GT: {gt_peak}'

    # Visualization: 2x4 layout
    fig = plt.figure(figsize=(26, 13))
    fig.suptitle(f'{title_line1}\n{title_line2}',
                 fontsize=16, fontweight='bold', y=0.98)

    # [1,1] Original scene (all objects)
    ax0 = plt.subplot(2, 4, 1)
    if original_img is not None:
        ax0.imshow(original_img)
        ax0.set_title('Original Scene (All Objects)', fontsize=12, fontweight='bold')
    else:
        ax0.imshow(room_img)
        ax0.set_title('Original Scene (N/A)', fontsize=12, fontweight='bold', color='gray')
    ax0.axis('off')

    # [1,2] Room top view (target object removed)
    ax1 = plt.subplot(2, 4, 2)
    ax1.imshow(room_img)
    ax1.set_title('Room Top View (Object Removed)', fontsize=12, fontweight='bold')
    ax1.axis('off')

    # [1,3] Object reference
    ax2 = plt.subplot(2, 4, 3)
    ax2.imshow(object_img)
    ax2.set_title('Object Reference', fontsize=12, fontweight='bold')
    ax2.axis('off')

    # [1,4] GT heatmap
    ax3 = plt.subplot(2, 4, 4)
    im3 = ax3.imshow(gt_heatmap, cmap='jet', vmin=0, vmax=1)
    ax3.plot(gt_peak[0], gt_peak[1], 'r+', markersize=25, markeredgewidth=4, label='GT Peak')
    ax3.set_title('GT Heatmap', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=10)
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04, label='Probability')

    # [2,1] Predicted heatmap
    ax4 = plt.subplot(2, 4, 5)
    im4 = ax4.imshow(pred_heatmap, cmap='jet', vmin=0, vmax=1)
    ax4.plot(pred_peak[0], pred_peak[1], 'b+', markersize=25, markeredgewidth=4, label='Pred Peak')
    ax4.set_title('Predicted Heatmap', fontsize=12, fontweight='bold')
    ax4.legend(loc='upper right', fontsize=10)
    ax4.axis('off')
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04, label='Probability')

    # [2,2] GT placement
    ax5 = plt.subplot(2, 4, 6)
    ax5.imshow(room_img)
    ax5.imshow(gt_heatmap, cmap='jet', alpha=0.6, vmin=0, vmax=1)
    ax5.plot(gt_peak[0], gt_peak[1], 'r+', markersize=30, markeredgewidth=5, label='GT Peak')
    circle_gt = plt.Circle(gt_peak, 15, color='red', fill=False, linewidth=3, linestyle='--')
    ax5.add_patch(circle_gt)
    ax5.set_title('GT Placement (Top View)', fontsize=12, fontweight='bold')
    ax5.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax5.axis('off')

    # [2,3] Predicted placement
    ax6 = plt.subplot(2, 4, 7)
    ax6.imshow(room_img)
    ax6.imshow(pred_heatmap, cmap='jet', alpha=0.6, vmin=0, vmax=1)
    ax6.plot(pred_peak[0], pred_peak[1], 'b+', markersize=30, markeredgewidth=5, label='Pred Peak')
    circle_pred = plt.Circle(pred_peak, 15, color='blue', fill=False, linewidth=3, linestyle='-')
    ax6.add_patch(circle_pred)
    ax6.set_title('Predicted Placement (Top View)', fontsize=12, fontweight='bold')
    ax6.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax6.axis('off')

    # [2,4] Pred vs GT comparison
    ax7 = plt.subplot(2, 4, 8)
    ax7.imshow(room_img)
    ax7.plot(pred_peak[0], pred_peak[1], 'b+', markersize=30, markeredgewidth=5, label='Pred Peak')
    ax7.plot(gt_peak[0], gt_peak[1], 'rx', markersize=25, markeredgewidth=4, label='GT Peak')
    circle_pred2 = plt.Circle(pred_peak, 15, color='blue', fill=False, linewidth=3, linestyle='-')
    circle_gt2 = plt.Circle(gt_peak, 15, color='red', fill=False, linewidth=3, linestyle='--')
    ax7.add_patch(circle_pred2)
    ax7.add_patch(circle_gt2)
    ax7.plot([gt_peak[0], pred_peak[0]], [gt_peak[1], pred_peak[1]],
             'y-', linewidth=3, label=f'Dist: {dist:.1f}px')
    ax7.set_title('Pred vs GT Comparison', fontsize=12, fontweight='bold')
    ax7.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax7.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return dist


def main():
    parser = argparse.ArgumentParser(description="Visualize heatmap training results")
    parser.add_argument("--data_dir", type=str, required=True, help="Data directory")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to visualize")
    parser.add_argument("--output_dir", type=str, default="visualizations", help="Output directory")
    parser.add_argument("--image_size", type=int, default=384, help="Image resolution")
    parser.add_argument("--split", type=str, default="val", help="Dataset split (train/val/test)")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    data_dir = Path(args.data_dir)
    json_path = data_dir / args.split / f"{args.split}.json"
    with open(json_path, 'r', encoding='utf-8') as f:
        samples = json.load(f)
    logging.info(f"Loaded {len(samples)} samples from {json_path}")

    # Select samples (uniform sampling)
    if args.num_samples < len(samples):
        step = len(samples) // args.num_samples
        selected_samples = samples[::step][:args.num_samples]
    else:
        selected_samples = samples[:args.num_samples]

    # Load model
    model = PlacementHeatmap(heatmap_res=256).to(device)
    model = load_checkpoint(model, args.checkpoint, device)

    # Visualize
    distances = []
    for i, sample in enumerate(selected_samples):
        output_path = output_dir / f"sample_{i:03d}_{sample['sample_id']}.png"
        logging.info(f"[{i+1}/{len(selected_samples)}] {sample['object_desc']}")
        dist = visualize_sample(model, sample, data_dir, device, output_path, args.image_size)
        distances.append(dist)
        logging.info(f"  Peak distance: {dist:.1f}px -> {output_path}")

    # Statistics
    avg_dist = np.mean(distances)
    median_dist = np.median(distances)
    acc_32 = sum(1 for d in distances if d < 32) / len(distances)

    logging.info(f"\n{'='*60}")
    logging.info(f"Results:")
    logging.info(f"  Samples: {len(distances)}")
    logging.info(f"  Avg peak distance: {avg_dist:.1f}px")
    logging.info(f"  Median peak distance: {median_dist:.1f}px")
    logging.info(f"  Peak accuracy (<32px): {acc_32:.1%}")
    logging.info(f"  Visualizations saved to: {output_dir}")
    logging.info(f"{'='*60}")


if __name__ == "__main__":
    main()
