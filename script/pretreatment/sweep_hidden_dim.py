"""Run heatmap training for multiple attention hidden dimensions.

Example:
  python script/pretreatment/sweep_hidden_dim.py \
    --data_dir /mnt/f/scenerevis/output/heatmap_data \
    --output_root checkpoints/heatmap_dinov2_b_hs_sweep \
    --hidden_dims 256,384,512 \
    --room_encoder dinov2 \
    --dino_model "$SCENEREVIS_DINOV2_MODEL" \
    --epochs 20 \
    --batch_size 2
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = PROJECT_ROOT / "script" / "pretreatment" / "train_heatmap.py"


def parse_hidden_dims(value: str) -> list[int]:
    dims = []
    for item in value.replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        dim = int(item)
        if dim <= 0:
            raise argparse.ArgumentTypeError("hidden_dim values must be positive")
        dims.append(dim)
    if not dims:
        raise argparse.ArgumentTypeError("at least one hidden_dim is required")
    return dims


def add_if_present(cmd: list[str], flag: str, value) -> None:
    if value is not None:
        cmd.extend([flag, str(value)])


def build_train_command(args, hidden_dim: int, output_dir: Path) -> list[str]:
    cmd = [
        args.python,
        str(TRAIN_SCRIPT),
        "--data_dir",
        args.data_dir,
        "--output_dir",
        str(output_dir),
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--weight_decay",
        str(args.weight_decay),
        "--num_workers",
        str(args.num_workers),
        "--image_size",
        str(args.image_size),
        "--pos_weight",
        str(args.pos_weight),
        "--peak_tolerance",
        str(args.peak_tolerance),
        "--room_encoder",
        args.room_encoder,
        "--hidden_dim",
        str(hidden_dim),
    ]

    add_if_present(cmd, "--dino_model", args.dino_model)
    add_if_present(cmd, "--room_image_size", args.room_image_size)
    add_if_present(cmd, "--object_image_size", args.object_image_size)
    add_if_present(cmd, "--early_stop_patience", args.early_stop_patience)

    latest = output_dir / "latest.pth"
    if args.resume_existing and latest.exists():
        cmd.extend(["--resume", str(latest)])

    if args.test_lr:
        cmd.append("--test_lr")

    return cmd


def checkpoint_candidates(output_dir: Path) -> Iterable[Path]:
    yield output_dir / "best_peak.pth"
    yield output_dir / "best.pth"
    yield output_dir / "latest.pth"


def summarize_run(hidden_dim: int, output_dir: Path) -> dict:
    row = {
        "hidden_dim": hidden_dim,
        "output_dir": str(output_dir),
        "checkpoint": None,
        "epoch": None,
        "best_val_loss": None,
        "best_peak_acc": None,
        "status": "missing",
    }

    checkpoint_path = next((path for path in checkpoint_candidates(output_dir) if path.exists()), None)
    if checkpoint_path is None:
        return row

    row["checkpoint"] = str(checkpoint_path)
    try:
        import torch

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        row["epoch"] = checkpoint.get("epoch")
        row["best_val_loss"] = checkpoint.get("best_val_loss")
        row["best_peak_acc"] = checkpoint.get("best_peak_acc")
        row["status"] = "ok"
    except Exception as exc:
        row["status"] = f"summary_failed: {exc}"

    return row


def write_summary(rows: list[dict], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)

    json_path = output_root / "summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    csv_path = output_root / "summary.csv"
    fieldnames = ["hidden_dim", "status", "best_peak_acc", "best_val_loss", "epoch", "checkpoint", "output_dir"]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})

    print(f"\nSummary written to:\n  {json_path}\n  {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep PlacementHeatmap hidden_dim values")
    parser.add_argument("--data_dir", required=True, help="Heatmap data root containing train/ and val/")
    parser.add_argument("--output_root", default="checkpoints/heatmap_hidden_sweep",
                        help="Root directory for per-hidden-dim checkpoints")
    parser.add_argument("--hidden_dims", type=parse_hidden_dims, default=parse_hidden_dims("256,384,512"),
                        help="Comma-separated hidden_dim values, e.g. 256,384,512")
    parser.add_argument("--room_encoder", default="dinov2", choices=["siglip", "dinov2"])
    parser.add_argument("--dino_model", default=os.environ.get("SCENEREVIS_DINOV2_MODEL"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=384)
    parser.add_argument("--room_image_size", type=int, default=None)
    parser.add_argument("--object_image_size", type=int, default=None)
    parser.add_argument("--pos_weight", type=float, default=10.0)
    parser.add_argument("--peak_tolerance", type=float, default=32.0)
    parser.add_argument("--early_stop_patience", type=int, default=0)
    parser.add_argument("--test_lr", action="store_true")
    parser.add_argument("--resume_existing", action="store_true",
                        help="Resume from each output_dir/latest.pth when present")
    parser.add_argument("--force", action="store_true",
                        help="Run even when best_peak.pth already exists")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without running them")
    parser.add_argument("--continue_on_error", action="store_true",
                        help="Continue sweeping if one hidden_dim fails")
    parser.add_argument("--python", default=sys.executable, help="Python executable used to launch training")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for hidden_dim in args.hidden_dims:
        output_dir = output_root / f"{args.room_encoder}_hd{hidden_dim}"
        done_checkpoint = output_dir / "best_peak.pth"

        print("\n" + "=" * 80)
        print(f"hidden_dim={hidden_dim}")
        print(f"output_dir={output_dir}")

        if done_checkpoint.exists() and not args.force and not args.resume_existing:
            print(f"Skip existing run: {done_checkpoint}")
            rows.append(summarize_run(hidden_dim, output_dir))
            continue

        cmd = build_train_command(args, hidden_dim, output_dir)
        print("Command:")
        print(" ".join(f'"{part}"' if " " in part else part for part in cmd))

        if args.dry_run:
            rows.append({
                "hidden_dim": hidden_dim,
                "output_dir": str(output_dir),
                "checkpoint": None,
                "epoch": None,
                "best_val_loss": None,
                "best_peak_acc": None,
                "status": "dry_run",
            })
            continue

        try:
            subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
        except subprocess.CalledProcessError as exc:
            rows.append({
                "hidden_dim": hidden_dim,
                "output_dir": str(output_dir),
                "checkpoint": None,
                "epoch": None,
                "best_val_loss": None,
                "best_peak_acc": None,
                "status": f"failed: exit {exc.returncode}",
            })
            write_summary(rows, output_root)
            if args.continue_on_error:
                continue
            raise

        rows.append(summarize_run(hidden_dim, output_dir))
        write_summary(rows, output_root)

    write_summary(rows, output_root)


if __name__ == "__main__":
    main()
