"""
VLM 客户端模块：使用 Qwen3-VL 生成摆放位置描述

通过对比三张图像（原始场景、移除物体后的场景、物体参考图）生成自然语言描述，
描述物体原来放在什么位置以及与周围参照物的关系。

支持 vLLM（批量快速）和 transformers（回退）两种后端。
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class VLMClient:
    """Qwen3-VL 客户端，用于生成摆放位置描述"""

    def __init__(
        self,
        model_path: str,
        backend: str = "vllm",
        max_tokens: int = 256,
        temperature: float = 0.7,
        cache_enabled: bool = True,
        cache_path: Optional[Path] = None,
    ):
        """
        Args:
            model_path: Qwen3-VL 模型路径
            backend: 推理后端 "vllm" 或 "transformers"
            max_tokens: 最大生成 token 数
            temperature: 采样温度
            cache_enabled: 是否启用磁盘缓存
            cache_path: 缓存文件路径
        """
        self.model_path = Path(model_path)
        self.backend = backend
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.cache_enabled = cache_enabled
        self.cache_path = Path(cache_path) if cache_path else None

        # 懒加载模型
        self._model = None
        self._processor = None
        self._vllm_engine = None

        # 加载缓存
        self._cache: Dict[str, str] = {}
        if self.cache_enabled and self.cache_path and self.cache_path.exists():
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    self._cache = json.load(f)
                logger.info(f"加载 VLM 缓存: {len(self._cache)} 条记录")
            except Exception as e:
                logger.warning(f"加载 VLM 缓存失败: {e}")

    def _load_model(self):
        """懒加载模型"""
        if self._model is not None or self._vllm_engine is not None:
            return

        if not self.model_path.exists():
            raise RuntimeError(f"模型路径不存在: {self.model_path}")

        if self.backend == "vllm":
            self._load_vllm_backend()
        else:
            self._load_transformers_backend()

    def _load_vllm_backend(self):
        """加载 vLLM 后端"""
        try:
            from vllm import LLM
            from transformers import AutoProcessor

            logger.info(f"使用 vLLM 加载模型: {self.model_path}")

            self._vllm_engine = LLM(
                model=str(self.model_path),
                limit_mm_per_prompt={"image": 3},
                gpu_memory_utilization=0.9,
                max_model_len=4096,
                swap_space=8,
                trust_remote_code=True,
                disable_log_stats=True,
                enforce_eager=True,
                disable_custom_all_reduce=True,
            )

            self._processor = AutoProcessor.from_pretrained(
                str(self.model_path),
                use_fast=True,
            )

            logger.info("vLLM 后端加载成功")
        except ImportError:
            logger.warning("vLLM 未安装，回退到 transformers")
            self.backend = "transformers"
            self._load_transformers_backend()
        except Exception as e:
            logger.error(f"vLLM 加载失败: {e}，回退到 transformers")
            self.backend = "transformers"
            self._load_transformers_backend()

    def _load_transformers_backend(self):
        """加载 transformers 后端"""
        import torch
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

        logger.info(f"使用 transformers 加载模型: {self.model_path}")

        self._processor = AutoProcessor.from_pretrained(
            str(self.model_path),
            use_cache=True,
        )

        attn_impl = "eager"
        if torch.cuda.is_available():
            try:
                import flash_attn
                attn_impl = "flash_attention_2"
            except ImportError:
                attn_impl = "sdpa"

        self._model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(self.model_path),
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation=attn_impl,
        )
        self._model.config.use_cache = True
        self._model.eval()

        logger.info(f"transformers 后端加载成功: {self._model.device}")

    def _build_prompt(self, desc: str) -> str:
        """Build the prompt used to generate the placement request text."""
        return (
            "Image 1 is the original top-down room view containing all objects. "
            "Image 2 is the same room after one target object has been removed. "
            "Image 3 is a reference image of the removed target object. "
            f"Compare the three images and write one short English placement request for the removed object: {desc}. "
            "Describe where the object was originally placed and its spatial relationship to nearby reference objects. "
            "Start exactly with: 'Please place [object name]'. "
            "Do not use Chinese. Do not include coordinates. Output only the request sentence."
        )

    def _save_cache(self):
        """保存缓存到磁盘"""
        if not self.cache_enabled or not self.cache_path:
            return

        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"保存 VLM 缓存失败: {e}")

    def get_cached_or_generate(
        self,
        cache_key: str,
        original_image: np.ndarray,
        plane_image: np.ndarray,
        object_image: np.ndarray,
        desc: str,
    ) -> Optional[str]:
        """获取缓存的描述或生成新描述

        Args:
            cache_key: 缓存键 (如 "scene_name_jid")
            original_image: 原始场景图 (H, W, 3)
            plane_image: 移除物体后的场景图 (H, W, 3)
            object_image: 物体参考图 (H, W, 3)
            desc: 物体原始描述

        Returns:
            生成的描述字符串，失败返回 None
        """
        # 检查缓存
        if self.cache_enabled and cache_key in self._cache:
            return self._cache[cache_key]

        # 生成描述
        try:
            result = self.generate_description(
                original_image, plane_image, object_image, desc
            )

            if result:
                # 保存到缓存
                if self.cache_enabled:
                    self._cache[cache_key] = result
                    self._save_cache()

            return result
        except Exception as e:
            logger.warning(f"VLM 生成失败: {e}")
            return None

    def get_cached_or_generate_batch(
        self,
        items: List[Dict],
    ) -> List[Optional[str]]:
        """批量获取缓存描述或生成新描述

        Args:
            items: 列表，每项包含:
                - cache_key: str
                - original_image: np.ndarray (H, W, 3)，可为 None
                - plane_image: np.ndarray (H, W, 3)
                - object_image: np.ndarray (H, W, 3)
                - desc: str

        Returns:
            描述列表，失败项为 None
        """
        self._load_model()

        cached_results = {}   # cache_key -> description
        pending_items = []    # (index, item)

        # 1. 检查缓存
        for i, item in enumerate(items):
            key = item["cache_key"]
            if self.cache_enabled and key in self._cache:
                cached_results[i] = self._cache[key]
            elif item.get("original_image") is None:
                # 无完整场景图（渲染失败），跳过 VLM，由调用方回退到 TextProcessor
                cached_results[i] = None
            else:
                pending_items.append((i, item))

        # 2. 批量生成未缓存的
        results: Dict[int, Optional[str]] = dict(cached_results)

        if pending_items:
            try:
                descs = self._generate_batch([it for _, it in pending_items])
                for (idx, item), desc in zip(pending_items, descs):
                    if desc and self.cache_enabled:
                        self._cache[item["cache_key"]] = desc
                    results[idx] = desc
            except Exception as e:
                logger.error(f"VLM 批量生成失败: {e}")
                for idx, _ in pending_items:
                    results[idx] = None

            if self.cache_enabled:
                self._save_cache()

        return [results.get(i) for i in range(len(items))]

    def _generate_batch(self, items: List[Dict]) -> List[Optional[str]]:
        """批量生成描述 (内部方法)

        Args:
            items: 每项包含 original_image, plane_image, object_image, desc
        """
        if self.backend == "vllm":
            return self._generate_vllm_batch(
                [it["original_image"] for it in items],
                [it["plane_image"] for it in items],
                [it["object_image"] for it in items],
                [it["desc"] for it in items],
            )
        else:
            results = []
            for it in items:
                r = self._generate_transformers(
                    it["original_image"], it["plane_image"],
                    it["object_image"], it["desc"],
                )
                results.append(r)
            return results

    def generate_description(
        self,
        original_image: np.ndarray,
        plane_image: np.ndarray,
        object_image: np.ndarray,
        desc: str,
    ) -> Optional[str]:
        """生成单个物体的摆放位置描述

        Args:
            original_image: 原始场景图 (H, W, 3)
            plane_image: 移除物体后的场景图 (H, W, 3)
            object_image: 物体参考图 (H, W, 3)
            desc: 物体原始描述

        Returns:
            生成的描述字符串，失败返回 None
        """
        self._load_model()

        prompt = self._build_prompt(desc)

        if self.backend == "vllm":
            return self._generate_vllm(original_image, plane_image, object_image, prompt)
        else:
            return self._generate_transformers(
                original_image, plane_image, object_image, prompt
            )

    def generate_descriptions_batch(
        self,
        original_images: List[np.ndarray],
        plane_images: List[np.ndarray],
        object_images: List[np.ndarray],
        descs: List[str],
    ) -> List[Optional[str]]:
        """批量生成描述

        Args:
            original_images: 原始场景图列表
            plane_images: 移除物体后的场景图列表
            object_images: 物体参考图列表
            descs: 物体原始描述列表

        Returns:
            生成的描述列表，失败返回 None
        """
        self._load_model()

        if self.backend == "vllm":
            return self._generate_vllm_batch(
                original_images, plane_images, object_images, descs
            )
        else:
            # transformers 不支持批量，逐个生成
            results = []
            for i in range(len(descs)):
                result = self.generate_description(
                    original_images[i], plane_images[i], object_images[i], descs[i]
                )
                results.append(result)
            return results

    def _generate_vllm(
        self,
        original_image: np.ndarray,
        plane_image: np.ndarray,
        object_image: np.ndarray,
        prompt: str,
    ) -> Optional[str]:
        """vLLM 生成单个描述"""
        from vllm import SamplingParams

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": original_image},
                    {"type": "image", "image": plane_image},
                    {"type": "image", "image": object_image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text_prompt = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        vllm_input = {
            "prompt": text_prompt,
            "multi_modal_data": {
                "image": [
                    Image.fromarray(original_image),
                    Image.fromarray(plane_image),
                    Image.fromarray(object_image),
                ]
            },
        }

        sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=0.9,
            max_tokens=self.max_tokens,
        )

        outputs = self._vllm_engine.generate([vllm_input], sampling_params, use_tqdm=False)
        result = outputs[0].outputs[0].text.strip()

        return result if result else None

    def _generate_vllm_batch(
        self,
        original_images: List[np.ndarray],
        plane_images: List[np.ndarray],
        object_images: List[np.ndarray],
        descs: List[str],
    ) -> List[Optional[str]]:
        """vLLM 批量生成描述"""
        from vllm import SamplingParams

        vllm_inputs = []
        for i in range(len(descs)):
            prompt = self._build_prompt(descs[i])

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": original_images[i]},
                        {"type": "image", "image": plane_images[i]},
                        {"type": "image", "image": object_images[i]},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            text_prompt = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            vllm_input = {
                "prompt": text_prompt,
                "multi_modal_data": {
                    "image": [
                        Image.fromarray(original_images[i]),
                        Image.fromarray(plane_images[i]),
                        Image.fromarray(object_images[i]),
                    ]
                },
            }
            vllm_inputs.append(vllm_input)

        sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=0.9,
            max_tokens=self.max_tokens,
        )

        outputs = self._vllm_engine.generate(vllm_inputs, sampling_params, use_tqdm=False)
        results = []
        for out in outputs:
            text = out.outputs[0].text.strip()
            results.append(text if text else None)

        return results

    def _generate_transformers(
        self,
        original_image: np.ndarray,
        plane_image: np.ndarray,
        object_image: np.ndarray,
        prompt: str,
    ) -> Optional[str]:
        """transformers 生成单个描述"""
        import torch
        from transformers import GenerationConfig

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": Image.fromarray(original_image)},
                    {"type": "image", "image": Image.fromarray(plane_image)},
                    {"type": "image", "image": Image.fromarray(object_image)},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self._processor(
            text=[text],
            images=[
                Image.fromarray(original_image),
                Image.fromarray(plane_image),
                Image.fromarray(object_image),
            ],
            return_tensors="pt",
        ).to(self._model.device)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                generation_config=GenerationConfig(
                    max_new_tokens=self.max_tokens,
                    do_sample=True,
                    temperature=self.temperature,
                ),
            )

        input_len = inputs["input_ids"].shape[1]
        result = self._processor.decode(
            output_ids[0, input_len:], skip_special_tokens=True
        ).strip()

        return result if result else None
