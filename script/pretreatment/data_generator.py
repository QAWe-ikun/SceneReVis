"""
SceneReVis 热力图训练数据生成器

从 3D-FUTURE 场景 JSON 生成训练数据:
  - 剔除目标物体后的房间俯视图 (plane_image)
  - 物体位置的高斯 GT 热力图 (mask)
  - 物体描述文本 (object_desc)

使用流程:
  Phase 1 (渲染): python generate_data.py --config config.yaml --phase 1
  Phase 2 (VLM):  python generate_data.py --config config.yaml --phase 2
  输出: {output_dir}/{split}/{scene_name}/plane_images/ + masks/ + {split}.json
"""
import os
import json
import tqdm
import random
import logging
import warnings
import numpy as np
from scipy import ndimage
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional

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

# 避免重复添加 handler
if not logger.handlers:
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

        # VLM 描述生成配置
        vlm_config = gen_config.get("vlm", {})
        self.use_vlm = vlm_config.get("enabled", False)
        self.vlm_client = None
        if self.use_vlm:
            vlm_model_path = vlm_config.get("model_path", "")
            if vlm_model_path and Path(vlm_model_path).exists():
                from .components.vlm_client import VLMClient
                try:
                    self.vlm_client = VLMClient(
                        model_path=vlm_model_path,
                        backend=vlm_config.get("backend", "vllm"),
                        max_tokens=vlm_config.get("max_tokens", 256),
                        temperature=vlm_config.get("temperature", 0.7),
                        cache_enabled=vlm_config.get("cache_enabled", True),
                        cache_path=self.output_dir / "vlm_desc_cache.json",
                    )
                    logger.info(f"VLM 描述生成已配置: {vlm_model_path} (backend={vlm_config.get('backend', 'vllm')})")
                except Exception as e:
                    logger.warning(f"VLM 客户端初始化失败: {e}")
                    self.use_vlm = False
            else:
                logger.warning(f"VLM 模型路径无效: {vlm_model_path}")
                self.use_vlm = False

        # 随机种子
        self.seed = gen_config.get("seed", 42)
        self.append_metadata = gen_config.get("append_metadata", False)
        random.seed(self.seed)
        np.random.seed(self.seed)

    @staticmethod
    def _remove_object_from_scene(scene, obj):
        """从场景副本中删除指定物体"""
        new_scene = scene.copy()
        geom_name = getattr(obj, "geom_name", f"obj_{obj.jid}")
        if geom_name in new_scene.geometry:
            del new_scene.geometry[geom_name]
        return new_scene

    @staticmethod
    def _removed_object_pixel_center(original_image, plane_image):
        """Estimate removed object center from rendered image difference."""
        if original_image is None or plane_image is None:
            return None
        if original_image.shape != plane_image.shape:
            return None

        diff = np.abs(
            original_image.astype(np.int16) - plane_image.astype(np.int16)
        ).sum(axis=2)
        if diff.max() <= 0:
            return None

        threshold = max(30.0, float(np.percentile(diff, 99) * 0.35))
        labels, count = ndimage.label(diff > threshold)
        if count == 0:
            return None

        best_score = 0.0
        best_center = None
        for label_idx in range(1, count + 1):
            ys, xs = np.nonzero(labels == label_idx)
            area = len(xs)
            if area < 50:
                continue
            weights = diff[ys, xs].astype(np.float64)
            weight_sum = weights.sum()
            if weight_sum <= 0:
                continue
            score = area * float(weights.mean())
            if score <= best_score:
                continue
            best_score = score
            peak_row = float((ys * weights).sum() / weight_sum)
            peak_col = float((xs * weights).sum() / weight_sum)
            best_center = (peak_row, peak_col)

        return best_center

    def _get_splits(self, n: int) -> List[str]:
        """根据比例生成划分列表"""
        n_train = int(n * self.train_ratio)
        n_val = int(n * self.val_ratio)
        return ["train"] * n_train + ["val"] * n_val + ["test"] * (n - n_train - n_val)

    def _render_scene(self, json_path: Path, split: str) -> List[Dict]:
        """渲染单个场景，收集所有物体结果（不调用 VLM，不保存到磁盘）

        Returns:
            结果列表，每项包含渲染图像和元数据。空列表表示跳过该场景。
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                scene_data = json.load(f)
        except Exception as e:
            logger.debug(f"无法加载 {json_path}: {e}")
            return []

        scene, objects = self.scene_builder.build_scene(scene_data)
        if not objects:
            return []

        bounds_bottom = self.scene_builder.extract_bounds_bottom(scene_data)
        if not bounds_bottom:
            return []

        scene_name = json_path.stem

        # 筛选地面物体
        floor_objects = [obj for obj in objects if obj.is_on_floor]
        if len(floor_objects) <= 1:
            return []

        # 渲染完整场景俯视图（VLM Phase 2 需要）
        original_image = self.renderer.render_top_view(scene, bounds_bottom)

        random.shuffle(floor_objects)
        results = []

        for target_obj in floor_objects:
            if len(results) >= self.max_object_nums:
                break

            # 从场景中删除目标物体
            scene_without_obj = self._remove_object_from_scene(scene, target_obj)

            # 渲染剔除后的房间俯视图
            plane_image = self.renderer.render_top_view(scene_without_obj, bounds_bottom)
            if plane_image is None:
                continue

            gt_pixel_center = self._removed_object_pixel_center(original_image, plane_image)
            gt_center_source = "render_diff" if gt_pixel_center is not None else "world_pos"
            sigma_pixels = (
                self.heatmap_generator.object_sigma_pixels(target_obj.size, bounds_bottom)
                if self.adaptive_sigma
                else self.heatmap_generator.sigma
            )
            if gt_pixel_center is not None:
                peak_row, peak_col = gt_pixel_center
                heatmap = self.heatmap_generator.generate_from_pixel(
                    peak_row, peak_col, sigma=sigma_pixels
                )
            elif self.adaptive_sigma:
                heatmap = self.heatmap_generator.generate_with_object_sigma(
                    target_obj.pos, target_obj.size, bounds_bottom
                )
            else:
                heatmap = self.heatmap_generator.generate(
                    target_obj.pos, bounds_bottom
                )

            if heatmap is None or heatmap.max() == 0:
                continue

            # 渲染物体参考图
            object_image = self.renderer.render_object_reference(
                target_obj.mesh, bounds_bottom
            )
            if object_image is None:
                continue

            # TextProcessor 描述（默认）
            tp_desc = self.text_processor.get_description(
                target_obj.model_id, target_obj.desc
            )

            # 被移除物体的 3D 元数据
            removed_object = {
                "instance_id": target_obj.instance_id,
                "geom_name": target_obj.geom_name,
                "jid": target_obj.jid,
                "model_id": target_obj.model_id,
                "desc": target_obj.desc,
                "pos": target_obj.pos,
                "rot": target_obj.rot,
                "size": target_obj.size,
                "gt_center_source": gt_center_source,
            }
            if gt_pixel_center is not None:
                peak_row, peak_col = gt_pixel_center
                removed_object["gt_pixel_center"] = [float(peak_col), float(peak_row)]

            results.append({
                "scene_name": scene_name,
                "split": split,
                "obj_id": f"{target_obj.instance_id:04d}_{target_obj.jid}",
                "plane_image": plane_image,
                "heatmap": heatmap,
                "object_image": object_image,
                "original_image": original_image,
                "removed_object": removed_object,
                "object_desc": tp_desc,
                "text_source": "text_processor",
            })

        return results

    # ================================================================
    # Phase 1: 渲染
    # ================================================================

    def run_phase1(self):
        """Phase 1: 渲染所有场景，保存图像和 TextProcessor 描述

        输出:
          {output_dir}/{split}/{scene_name}/
            plane_images/  masks/  object_images/  original_images/
          {output_dir}/{split}/{split}.json
        """
        logger.info("=" * 60)
        logger.info("Phase 1: 渲染所有场景")
        logger.info(f"场景目录: {self.scene_dir}")
        logger.info(f"模型目录: {self.model_dir}")
        logger.info(f"输出目录: {self.output_dir}")
        logger.info("=" * 60)

        json_files = sorted(self.scene_dir.glob("*.json"))
        if not json_files:
            logger.error(f"在 {self.scene_dir} 中未找到 JSON 文件")
            return

        random.Random(self.seed).shuffle(json_files)

        n = len(json_files)
        splits = self._get_splits(n)
        logger.info(f"共 {n} 个场景: "
                     f"train={splits.count('train')}, "
                     f"val={splits.count('val')}, "
                     f"test={splits.count('test')}")

        for idx, json_path in enumerate(tqdm.tqdm(json_files, desc="Phase 1 渲染")):
            split = splits[idx]
            results = self._render_scene(json_path, split)

            for r in results:
                scene_dir = self.output_dir / r["split"] / r["scene_name"]
                self.sample_saver.save_sample(
                    scene_dir=scene_dir,
                    obj_id=r["obj_id"],
                    plane_image=r["plane_image"],
                    heatmap=r["heatmap"],
                    object_desc=r["object_desc"],
                    object_image=r["object_image"],
                    split=r["split"],
                    scene_name=r["scene_name"],
                    original_image=r["original_image"],
                    removed_object=r["removed_object"],
                    text_source=r["text_source"],
                )

            if (idx + 1) % 500 == 0:
                logger.info(
                    f"  已处理 {idx + 1}/{n} 场景，"
                    f"已收集 {self.sample_saver.sample_counter} 个样本"
                )

        # 保存元数据 JSON
        for split_name in ["train", "val", "test"]:
            self.sample_saver.save_split_json(split_name, append=self.append_metadata)

        logger.info("=" * 60)
        logger.info(f"Phase 1 完成! 共生成 {self.sample_saver.sample_counter} 个训练样本")
        logger.info(f"输出: {self.output_dir}")
        if self.use_vlm:
            logger.info(f"下一步: python generate_data.py --config config.yaml --phase 2")
        logger.info("=" * 60)

    # ================================================================
    # Phase 2: VLM 批量描述
    # ================================================================

    def run_phase2(self, vlm_batch_size: int = 256):
        """Phase 2: 加载 Phase 1 数据，批量 VLM 描述生成

        读取 {split}.json，对 text_source=="text_processor" 的样本
        批量调用 VLM，更新 object_desc 和 text_source。

        Args:
            vlm_batch_size: VLM 批量推理的批次大小
        """
        if not self.use_vlm or self.vlm_client is None:
            logger.error("VLM 未启用，无法执行 Phase 2")
            return

        logger.info("=" * 60)
        logger.info("Phase 2: 批量 VLM 描述生成")
        logger.info(f"数据目录: {self.output_dir}")
        logger.info(f"VLM batch_size: {vlm_batch_size}")
        logger.info("=" * 60)

        # 预加载 VLM 模型
        logger.info("加载 VLM 模型...")
        self.vlm_client._load_model()
        logger.info("VLM 模型加载完成")

        vlm_count = 0
        tp_count = 0
        total_updated = 0

        # for split_name in ["train", "val", "test"]:
        for split_name in ["val", "test"]:
            json_path = self.output_dir / split_name / f"{split_name}.json"
            if not json_path.exists():
                logger.warning(f"跳过 {split_name}: {json_path} 不存在")
                continue

            with open(json_path, 'r', encoding='utf-8') as f:
                samples = json.load(f)

            # 筛选需要 VLM 处理的样本
            pending = [
                (i, s) for i, s in enumerate(samples)
                if s.get("text_source", "text_processor") == "text_processor"
            ]

            if not pending:
                logger.info(f"{split_name}: 所有样本已有 VLM 描述，跳过")
                continue

            logger.info(
                f"{split_name}: {len(pending)}/{len(samples)} 个样本待 VLM 处理"
            )

            # 分批处理
            from PIL import Image

            for batch_start in tqdm.tqdm(
                range(0, len(pending), vlm_batch_size),
                desc=f"Phase 2 {split_name}",
            ):
                batch = pending[batch_start:batch_start + vlm_batch_size]
                vlm_items = []

                for _, sample in batch:
                    scene_dir = self.output_dir / sample["scene_dir"]

                    # 加载图像
                    original_path = scene_dir / sample.get("original_image_path", "")
                    plane_path = scene_dir / sample["plane_image_path"]
                    object_path = scene_dir / sample["object_image_path"]

                    original_img = None
                    if original_path.exists():
                        original_img = np.array(Image.open(original_path).convert("RGB"))

                    plane_img = np.array(Image.open(plane_path).convert("RGB"))
                    object_img = np.array(Image.open(object_path).convert("RGB"))

                    cache_key = f"{sample.get('scene_name', '')}_{sample['sample_id'].replace('obj_', '')}"

                    vlm_items.append({
                        "cache_key": cache_key,
                        "original_image": original_img,
                        "plane_image": plane_img,
                        "object_image": object_img,
                        "desc": sample.get("removed_object", {}).get("desc", ""),
                    })

                # 批量 VLM 推理
                vlm_descs = self.vlm_client.get_cached_or_generate_batch(vlm_items)

                # 更新样本描述
                for (idx, sample), vlm_desc in zip(batch, vlm_descs):
                    if vlm_desc:
                        samples[idx]["object_desc"] = vlm_desc
                        samples[idx]["text_source"] = "vlm"
                        vlm_count += 1
                    else:
                        tp_count += 1

                total_updated += len(batch)
                logger.info(
                    f"  {split_name} 进度: {total_updated}/{len(pending)} | "
                    f"VLM: {vlm_count}, TextProcessor: {tp_count}"
                )

            # 写回 JSON
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(samples, f, ensure_ascii=False, indent=2)

            logger.info(f"{split_name}: 已更新 {json_path}")

        logger.info("=" * 60)
        logger.info(
            f"Phase 2 完成! VLM={vlm_count}, TextProcessor={tp_count}"
        )
        logger.info("=" * 60)
