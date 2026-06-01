#!/usr/bin/env python3
"""
SceneReVis 热力图训练数据生成 CLI

使用流程:
  # Phase 1: 渲染所有场景 (CPU, 不需要 GPU)
  python generate_data.py --config config.yaml --phase 1

  # Phase 2: 批量 VLM 描述生成 (需要 GPU)
  python generate_data.py --config config.yaml --phase 2

  # 或一次性跑完两阶段
  python generate_data.py --config config.yaml --phase all
"""
import sys
import argparse
from pathlib import Path

# 将 script/ 目录加入路径，使 pretreatment 模块可导入
sys.path.insert(0, str(Path(__file__).parent.parent))

from pretreatment.data_generator import HeatmapDataGenerator


def load_config(config_path: str) -> dict:
    """加载 YAML 配置"""
    import yaml
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="SceneReVis 热力图训练数据生成"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="配置文件路径 (YAML)"
    )
    parser.add_argument(
        "--phase", type=str, default="all",
        choices=["1", "2", "all"],
        help="执行阶段: 1=渲染, 2=VLM描述, all=两阶段 (默认 all)"
    )
    parser.add_argument(
        "--scene-dir", type=str, default=None,
        help="覆盖配置中的 scene_dir"
    )
    parser.add_argument(
        "--model-dir", type=str, default=None,
        help="覆盖配置中的 model_dir"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="覆盖配置中的 output_dir"
    )
    parser.add_argument(
        "--max-objects", type=int, default=None,
        help="覆盖配置中的 max_object_nums"
    )
    parser.add_argument(
        "--vlm-batch-size", type=int, default=256,
        help="Phase 2 VLM 批量推理批次大小（默认 256，减小可避免 OOM）"
    )

    args = parser.parse_args()

    config = load_config(args.config)

    # 命令行参数覆盖配置
    if args.scene_dir:
        config.setdefault("data", {})["scene_dir"] = args.scene_dir
    if args.model_dir:
        config.setdefault("data", {})["model_dir"] = args.model_dir
    if args.output_dir:
        config.setdefault("data", {})["output_dir"] = args.output_dir
    if args.max_objects:
        config.setdefault("generation", {})["max_object_nums"] = args.max_objects

    # 验证路径
    data_config = config.get("data", {})
    scene_dir = Path(data_config.get("scene_dir", ""))
    model_dir = Path(data_config.get("model_dir", ""))

    if not scene_dir.exists():
        print(f"错误: 场景目录不存在: {scene_dir}")
        sys.exit(1)
    if not model_dir.exists():
        print(f"错误: 模型目录不存在: {model_dir}")
        sys.exit(1)

    generator = HeatmapDataGenerator(config)

    if args.phase == "1":
        generator.run_phase1()
    elif args.phase == "2":
        generator.run_phase2(vlm_batch_size=args.vlm_batch_size)
    else:  # all
        generator.run_phase1()
        generator.run_phase2(vlm_batch_size=args.vlm_batch_size)


if __name__ == "__main__":
    main()
