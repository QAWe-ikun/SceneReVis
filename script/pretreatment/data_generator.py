"""
SceneReVis 热力图训练数据生成器

从 3D-FUTURE 场景 JSON 生成训练数据:
  - 剔除目标物体后的房间俯视图 (plane_image)
  - 物体位置的高斯 GT 热力图 (mask)
  - 物体描述文本 (object_desc)

使用流程:
  1. python generate_data.py --config config.yaml
  2. 输出: {output_dir}/{split}/{scene_name}/plane_images/ + masks/ + {split}.json
"""
import os
import json
import tqdm
import random
import logging
import warnings
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List

if not os.environ.get("PYOPENGL_PLATFORM"):
    os.environ["PYOPENGL_PLATFORM"] = "egl"

warnings.filterwarnings("ignore", category=UserWarning, module="trimesh")
warnings.filterwarnings("ignore", category=UserWarning, module="pyrender")

# 日志配置
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
log_file = LOG_DIR / f"generate_heatmap_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logger = logging.getLogger("HeatmapDataGenerator")
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(log_file, encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
))
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logger.addHandler(console_handler)

from .components.scene_builder import SceneBuilder
from .components.renderer import OrthoRenderer
from .components.heatmap_generator import HeatmapGenerator
from .components.sample_saver import SampleSaver
from .components.text_processor import TextProcessor


class HeatmapDataGenerator:
    """热力图训练数据生成器"""

    def __init__(self, config: Dict[str, Any]):
        data_config = config.get("data", {})
        gen_config = config.get("generation", {})

        self.scene_dir = Path(data_config.get("scene_dir", ""))
        self.model_dir = Path(data_config.get("model_dir", ""))
        self.output_dir = Path(data_config.get("output_dir", ""))

        self.output_dir.mkdir(parents=True, exist_ok=True)
        for split in ["train", "val", "test"]:
            (self.output_dir / split).mkdir(parents=True, exist_ok=True)

        image_size = gen_config.get("image_size", 1024)

        self.scene_builder = SceneBuilder(self.model_dir)

        self.renderer = OrthoRenderer(image_size=image_size)

        self.heatmap_generator = HeatmapGenerator(
            image_size=image_size,
            sigma=gen_config.get("heatmap_sigma", 15.0),
        )

        self.sample_saver = SampleSaver(self.output_dir)

        # 文本处理器：从 metadata 加载丰富的物体描述
        metadata_dir = data_config.get("metadata_dir", self.model_dir.parent / "metadata")
        text_config = gen_config.get("text", {})
        self.text_processor = TextProcessor(
            metadata_dir=Path(metadata_dir),
            use_summary=text_config.get("use_summary", True),
            use_prompts=text_config.get("use_prompts", True),
            use_simple_descs=text_config.get("use_simple_descs", True),
            augmentation_prob=text_config.get("augmentation_prob", 0.5),
        )

        # 数据集划分比例
        split_ratio = gen_config.get("split_ratio", {"train": 0.8, "val": 0.1, "test": 0.1})
        self.train_ratio = split_ratio.get("train", 0.8)
        self.val_ratio = split_ratio.get("val", 0.1)

        # 每个场景最多处理的物体数
        self.max_object_nums = gen_config.get("max_object_nums", 5)

        # 是否使用物体尺寸自适应 sigma
        self.adaptive_sigma = gen_config.get("adaptive_sigma", False)

        # 随机种子
        seed = gen_config.get("seed", 42)
        random.seed(seed)
        np.random.seed(seed)

    @staticmethod
    def _remove_object_from_scene(scene, obj):
        """从场景副本中删除指定物体"""
        new_scene = scene.copy()
        geom_name = f"obj_{obj.jid}"
        if geom_name in new_scene.geometry:
            del new_scene.geometry[geom_name]
        return new_scene

    def _get_splits(self, n: int) -> List[str]:
        """根据比例生成划分列表"""
        n_train = int(n * self.train_ratio)
        n_val = int(n * self.val_ratio)
        return ["train"] * n_train + ["val"] * n_val + ["test"] * (n - n_train - n_val)

    def _render_and_save_scene(self, json_path: Path, split: str):
        """渲染单个场景并保存训练样本"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                scene_data = json.load(f)
        except Exception as e:
            logger.debug(f"无法加载 {json_path}: {e}")
            return

        scene, objects = self.scene_builder.build_scene(scene_data)
        if not objects:
            logger.debug(f"场景 {json_path.stem} 无有效物体，跳过")
            return

        bounds_bottom = self.scene_builder.extract_bounds_bottom(scene_data)
        if not bounds_bottom:
            logger.debug(f"场景 {json_path.stem} 无 bounds_bottom，跳过")
            return

        scene_name = json_path.stem
        scene_dir = self.output_dir / split / scene_name

        # 筛选地面物体
        floor_objects = [obj for obj in objects if obj.is_on_floor]
        if len(floor_objects) <= 1:
            logger.debug(f"场景 {scene_name} 地面物体 <= 1，跳过")
            return

        random.shuffle(floor_objects)
        saved_count = 0

        for target_obj in floor_objects:
            if saved_count >= self.max_object_nums:
                break

            # 生成 GT 热力图
            if self.adaptive_sigma:
                heatmap = self.heatmap_generator.generate_with_object_sigma(
                    target_obj.pos, target_obj.size, bounds_bottom
                )
            else:
                heatmap = self.heatmap_generator.generate(
                    target_obj.pos, bounds_bottom
                )

            if heatmap is None or heatmap.max() == 0:
                continue

            # 从场景中删除目标物体
            scene_without_obj = self._remove_object_from_scene(scene, target_obj)

            # 渲染剔除后的房间俯视图
            plane_image = self.renderer.render_top_view(scene_without_obj, bounds_bottom)
            if plane_image is None:
                continue

            # 渲染物体参考图
            object_image = self.renderer.render_object_reference(
                target_obj.mesh, bounds_bottom
            )
            if object_image is None:
                logger.debug(f"物体 {target_obj.jid} 参考图渲染失败，跳过")
                continue

            # 使用文本处理器获取丰富的描述
            enriched_desc = self.text_processor.get_description(
                target_obj.model_id,
                target_obj.desc
            )

            # 保存样本
            self.sample_saver.save_sample(
                scene_dir=scene_dir,
                obj_id=target_obj.jid,
                plane_image=plane_image,
                heatmap=heatmap,
                object_desc=enriched_desc,
                object_image=object_image,
                split=split,
            )
            saved_count += 1

        if saved_count > 0:
            logger.debug(f"场景 {scene_name}: 保存 {saved_count} 个样本 ({split})")

    def run(self):
        """执行数据生成"""
        logger.info("=" * 60)
        logger.info("SceneReVis 热力图训练数据生成")
        logger.info(f"场景目录: {self.scene_dir}")
        logger.info(f"模型目录: {self.model_dir}")
        logger.info(f"输出目录: {self.output_dir}")
        logger.info("=" * 60)

        json_files = sorted(self.scene_dir.glob("*.json"))
        if not json_files:
            logger.error(f"在 {self.scene_dir} 中未找到 JSON 文件")
            return

        n = len(json_files)
        splits = self._get_splits(n)
        logger.info(f"共 {n} 个场景: "
                     f"train={splits.count('train')}, "
                     f"val={splits.count('val')}, "
                     f"test={splits.count('test')}")

        for idx, json_path in enumerate(tqdm.tqdm(json_files, desc="生成数据")):
            split = splits[idx]
            self._render_and_save_scene(json_path, split)

        # 保存元数据 JSON
        for split_name in ["train", "val", "test"]:
            self.sample_saver.save_split_json(split_name)

        logger.info("=" * 60)
        logger.info(f"完成! 共生成 {self.sample_saver.sample_counter} 个训练样本")
        logger.info("=" * 60)
