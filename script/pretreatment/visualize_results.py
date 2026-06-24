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
import re
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib as mpl
import matplotlib.font_manager as fm
from matplotlib.ft2font import FT2Font
import matplotlib.pyplot as plt
import torchvision.transforms as T

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.placement_heatmap import PlacementHeatmap, load_trainable_heatmap_state_dict


CJK_FONT_NAMES = [
    "Noto Sans CJK SC",
    "Noto Sans CJK JP",
    "Source Han Sans SC",
    "Source Han Sans CN",
    "WenQuanYi Zen Hei",
    "Microsoft YaHei",
    "SimHei",
    "PingFang SC",
    "Arial Unicode MS",
]

CJK_FONT_PATHS = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
    "/mnt/c/Windows/Fonts/msyh.ttc",
    "/mnt/c/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
]

WSL_WINDOWS_FONT_DIR = Path("/mnt/c/Windows/Fonts")
CJK_TEST_CODEPOINT = ord("请")
CJK_RE = re.compile(r"[\u3000-\u303f\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff00-\uffef]")
HAS_CJK_FONT = False
CJK_FONT_PROP = None


def font_supports_cjk(font_path):
    """Return True if a font file contains common Chinese glyphs."""
    try:
        return CJK_TEST_CODEPOINT in FT2Font(str(font_path)).get_charmap()
    except Exception:
        return False


def iter_cjk_font_paths():
    """Yield likely CJK font paths, including Windows fonts mounted in WSL."""
    seen = set()
    for font_path in CJK_FONT_PATHS:
        path = Path(font_path)
        if path not in seen:
            seen.add(path)
            yield path

    if WSL_WINDOWS_FONT_DIR.exists():
        preferred = [
            "msyh.ttc",
            "msyhbd.ttc",
            "simhei.ttf",
            "simsun.ttc",
            "Deng.ttf",
            "Dengb.ttf",
            "msjh.ttc",
        ]
        for name in preferred:
            path = WSL_WINDOWS_FONT_DIR / name
            if path not in seen:
                seen.add(path)
                yield path
        for path in WSL_WINDOWS_FONT_DIR.glob("*"):
            if path.suffix.lower() in {".ttf", ".ttc", ".otf"} and path not in seen:
                seen.add(path)
                yield path


def configure_matplotlib_fonts():
    """Configure a CJK-capable font if one is available."""
    global CJK_FONT_PROP
    installed_fonts = {font.name for font in fm.fontManager.ttflist}

    selected_font = next((name for name in CJK_FONT_NAMES if name in installed_fonts), None)
    selected_path = None
    for path in iter_cjk_font_paths():
        if path.exists() and font_supports_cjk(path):
            fm.fontManager.addfont(str(path))
            selected_font = fm.FontProperties(fname=str(path)).get_name()
            selected_path = str(path)
            break

    if selected_font:
        CJK_FONT_PROP = fm.FontProperties(fname=selected_path) if selected_path else fm.FontProperties(family=selected_font)
        mpl.rcParams["font.sans-serif"] = [selected_font, *mpl.rcParams.get("font.sans-serif", [])]
        mpl.rcParams["font.family"] = "sans-serif"
        mpl.rcParams["axes.unicode_minus"] = False
        return True

    return False


def safe_plot_text(text):
    """Avoid missing-glyph warnings when no CJK font exists in the runtime."""
    if HAS_CJK_FONT:
        return text
    return CJK_RE.sub("?", text)


HAS_CJK_FONT = configure_matplotlib_fonts()


def load_checkpoint(model, checkpoint_path, device):
    """Load checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
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
    epoch = checkpoint["epoch"]
    best_val_loss = checkpoint.get("best_val_loss", float('inf'))
    logging.info(f"Loaded checkpoint: epoch={epoch}, best_val_loss={best_val_loss:.4f}")
    return model


class Qwen3VLCoordinateBaseline:
    """Qwen3-VL baseline that predicts a 2D placement coordinate."""

    def __init__(
        self,
        model_path: str,
        scene_json_dir: str,
        backend: str = "vllm",
        max_tokens: int = 768,
        temperature: float = 0.0,
        cache_path: Optional[Path] = None,
        refresh_cache: bool = False,
    ):
        self.model_path = Path(model_path)
        self.scene_json_dir = Path(scene_json_dir)
        self.backend = backend
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.cache_path = Path(cache_path) if cache_path else None
        self.refresh_cache = refresh_cache
        self._model = None
        self._processor = None
        self._vllm_engine = None
        self._cache = {}
        self._scene_cache = {}

        if self.cache_path and self.cache_path.exists() and not self.refresh_cache:
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    self._cache = json.load(f)
            except Exception as e:
                logging.warning(f"Failed to load Qwen3-VL coordinate cache: {e}")

    def _load_model(self):
        if self._model is not None or self._vllm_engine is not None:
            return
        if not self.model_path.exists():
            raise RuntimeError(f"Qwen3-VL model path does not exist: {self.model_path}")
        if not self.scene_json_dir.exists():
            raise RuntimeError(f"Scene JSON directory does not exist: {self.scene_json_dir}")
        if self.backend == "vllm":
            self._load_vllm_backend()
        else:
            self._load_transformers_backend()

    def _load_vllm_backend(self):
        try:
            from transformers import AutoProcessor
            from vllm import LLM

            logging.info(f"Loading Qwen3-VL baseline with vLLM: {self.model_path}")
            self._vllm_engine = LLM(
                model=str(self.model_path),
                limit_mm_per_prompt={"image": 2},
                gpu_memory_utilization=0.9,
                max_model_len=8192,
                swap_space=8,
                trust_remote_code=True,
                disable_log_stats=True,
                enforce_eager=True,
                disable_custom_all_reduce=True,
            )
            self._processor = AutoProcessor.from_pretrained(str(self.model_path), use_fast=True)
        except ImportError:
            logging.warning("vLLM is not installed; falling back to transformers backend")
            self.backend = "transformers"
            self._load_transformers_backend()
        except Exception as e:
            logging.warning(f"Failed to load vLLM backend ({e}); falling back to transformers")
            self.backend = "transformers"
            self._load_transformers_backend()

    def _load_transformers_backend(self):
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        logging.info(f"Loading Qwen3-VL baseline with transformers: {self.model_path}")
        self._processor = AutoProcessor.from_pretrained(str(self.model_path), use_cache=True)
        attn_impl = "sdpa" if torch.cuda.is_available() else "eager"
        self._model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(self.model_path),
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            attn_implementation=attn_impl,
        )
        self._model.eval()

    @staticmethod
    def _clean_scene_object(obj: dict, pixel_center: Optional[Tuple[float, float]] = None) -> dict:
        cleaned = {}
        for key in ("desc", "size", "pos", "rot", "jid"):
            if key in obj:
                cleaned[key] = obj[key]
        if pixel_center is not None:
            cleaned["pixel_center_2d"] = [round(float(pixel_center[0]), 1), round(float(pixel_center[1]), 1)]
        return cleaned

    @staticmethod
    def _world_to_pixel(pos: list, bounds_bottom: list, image_size: Tuple[int, int]) -> Optional[Tuple[float, float]]:
        if not pos or len(pos) < 3 or not bounds_bottom:
            return None
        try:
            xs = [float(p[0]) for p in bounds_bottom]
            zs = [float(p[2]) for p in bounds_bottom]
            cx = (min(xs) + max(xs)) / 2.0
            cz = (min(zs) + max(zs)) / 2.0
            ortho_scale = max(max(xs) - min(xs), max(zs) - min(zs))
            if ortho_scale <= 0:
                return None
            image_w, image_h = image_size
            x = ((float(pos[0]) - cx) / ortho_scale + 0.5) * image_w
            y = ((float(pos[2]) - cz) / ortho_scale + 0.5) * image_h
            return float(np.clip(x, 0, image_w - 1)), float(np.clip(y, 0, image_h - 1))
        except Exception:
            return None

    def _load_scene(self, scene_name: str) -> dict:
        if scene_name in self._scene_cache:
            return self._scene_cache[scene_name]
        scene_path = self.scene_json_dir / f"{scene_name}.json"
        if not scene_path.exists():
            raise FileNotFoundError(f"Scene JSON not found: {scene_path}")
        with open(scene_path, "r", encoding="utf-8") as f:
            scene = json.load(f)
        self._scene_cache[scene_name] = scene
        return scene

    def _build_current_scene(self, sample: dict, image_size: Tuple[int, int]) -> dict:
        scene_name = sample.get("scene_name")
        if not scene_name:
            raise ValueError("Sample has no scene_name; cannot build SceneReVis prompt")

        source_scene = self._load_scene(scene_name)
        removed = sample.get("removed_object", {})
        removed_idx = removed.get("instance_id")
        removed_jid = removed.get("jid")
        bounds_bottom = source_scene.get("bounds_bottom", [])

        objects = source_scene.get("objects", [])
        kept_objects = []
        for idx, obj in enumerate(objects):
            is_removed = False
            if isinstance(removed_idx, int):
                is_removed = idx == removed_idx
            elif removed_jid is not None:
                is_removed = obj.get("jid") == removed_jid
            if is_removed:
                continue
            kept_objects.append(
                self._clean_scene_object(
                    obj,
                    pixel_center=self._world_to_pixel(obj.get("pos"), bounds_bottom, image_size),
                )
            )

        target_object = self._clean_scene_object(removed)
        target_object.pop("pos", None)
        target_object.pop("gt_pixel_center", None)
        target_object.pop("gt_center_source", None)

        return {
            "room_type": source_scene.get("room_type"),
            "room_id": source_scene.get("room_id"),
            "bounds_top": source_scene.get("bounds_top"),
            "bounds_bottom": bounds_bottom,
            "objects": kept_objects,
            "target_object_to_add": target_object,
        }

    @staticmethod
    def _build_prompt(desc: str, image_size: Tuple[int, int], current_scene: dict) -> str:
        width, height = image_size
        current_scene_json = json.dumps(current_scene, ensure_ascii=False, indent=2)
        return (
            "### Role and Core Directive\n\n"
            "You are an AI spatial layout planner. Your core task is to analyze and optimize "
            "indoor scenes, ensuring they are physically valid and functionally efficient.\n\n"
            "### Core Capabilities\n\n"
            "Your primary responsibility is to diagnose the current scene and produce a SceneReVis-style "
            "structured edit that adds the requested object back into the layout.\n\n"
            "### Scene Analysis and Spatial Rules\n\n"
            "Your input includes two rendered images:\n"
            "* Image 1: a top-down room view after the target object has been removed. Use this as "
            "the primary basis for judging relative positions, free space, object spacing, and room boundaries.\n"
            "* Image 2: a reference image of the target object to add back into the room.\n\n"
            "You are also given the SceneReVis `<current_scene>` JSON after the target object has been removed. "
            "Existing objects include 3D world coordinates in `pos` and projected top-down image coordinates in "
            "`pixel_center_2d`. Use these structured coordinates to resolve anchor objects, left/right/front/back "
            "relations, and symmetric or repeated furniture.\n\n"
            "Mandatory requirements:\n"
            "1. Place the object in a physically valid location without obvious overlap with existing objects.\n"
            "2. Prefer a functionally reasonable placement that matches the surrounding furniture layout.\n"
            "3. Use the top-down view coordinate system for the final placement point. The origin is the top-left "
            f"corner of Image 1, x increases to the right, y increases downward, and Image 1 is {width}x{height} pixels.\n\n"
            "<current_scene>\n"
            "```json\n"
            f"{current_scene_json}\n"
            "```\n"
            "</current_scene>\n\n"
            "### Available Tools\n\n"
            "**add_object**: Add a new furniture piece. This visualization baseline extends the SceneReVis "
            "tool arguments with `position_2d_pixel` so the result can be compared with heatmap peaks.\n"
            "* `position_2d_pixel` (array): [x, y] pixel coordinate in Image 1 for the object center.\n"
            "* `placement_plane` (string): use \"floor\" unless the object should be placed on top of another object.\n\n"
            "### Current User Request\n\n"
            f"{desc}\n\n"
            "### Output Format Requirements\n\n"
            "You must follow the SceneReVis editing format exactly: first `<think>`, then `<tool_calls>`. "
            "Use English only in the response. "
            "Do not output markdown fences. Do not output a standalone JSON block. "
            "The final answer must contain the `<tool_calls>` XML tag. "
            "Keep `<think>` to few sentences so the complete `<tool_calls>` block is always emitted. "
            "Return exactly one `add_object` call.\n\n"
            "<think>\n"
            "[Few sentences thinking how to place.]\n"
            "</think>\n\n"
            "<tool_calls>\n"
            "[\n"
            "  {\n"
            "    \"id\": \"tool_1\",\n"
            "    \"name\": \"add_object\",\n"
            "    \"arguments\": {\n"
            "      \"position_2d_pixel\": [x, y],\n"
            "      \"placement_plane\": \"floor\"\n"
            "    }\n"
            "  }\n"
            "]\n"
            "</tool_calls>"
        )

    @staticmethod
    def _parse_coordinate(text: str, image_size: Tuple[int, int]) -> Optional[Tuple[float, float]]:
        width, height = image_size

        def normalize_coord(value) -> Optional[Tuple[float, float]]:
            if not isinstance(value, list) or len(value) < 2:
                return None
            try:
                x = float(value[0])
                y = float(value[1])
            except (TypeError, ValueError):
                return None
            return float(np.clip(x, 0, width - 1)), float(np.clip(y, 0, height - 1))

        def extract_from_data(data) -> Optional[Tuple[float, float]]:
            if isinstance(data, list):
                for item in data:
                    parsed = extract_from_data(item)
                    if parsed is not None:
                        return parsed
                return None
            if not isinstance(data, dict):
                return None

            args = data.get("arguments")
            if isinstance(args, dict):
                parsed = extract_from_data(args)
                if parsed is not None:
                    return parsed

            for key in ("position_2d_pixel", "pixel_position", "image_position", "position"):
                parsed = normalize_coord(data.get(key))
                if parsed is not None:
                    return parsed
            if "x" in data and "y" in data:
                try:
                    return normalize_coord([data["x"], data["y"]])
                except Exception:
                    return None
            return None

        def iter_json_blocks(raw_text: str):
            code_blocks = re.findall(r"```(?:json)?\s*(.*?)\s*```", raw_text, flags=re.DOTALL | re.IGNORECASE)
            for block in code_blocks:
                yield block.strip()

            for start_char, end_char in (("[", "]"), ("{", "}")):
                starts = [m.start() for m in re.finditer(re.escape(start_char), raw_text)]
                for start in starts:
                    depth = 0
                    in_string = False
                    escape = False
                    for idx in range(start, len(raw_text)):
                        ch = raw_text[idx]
                        if in_string:
                            if escape:
                                escape = False
                            elif ch == "\\":
                                escape = True
                            elif ch == '"':
                                in_string = False
                            continue
                        if ch == '"':
                            in_string = True
                        elif ch == start_char:
                            depth += 1
                        elif ch == end_char:
                            depth -= 1
                            if depth == 0:
                                yield raw_text[start:idx + 1]
                                break

        tool_match = re.search(r"<tool_calls>\s*(.*?)\s*</tool_calls>", text, flags=re.DOTALL)
        if tool_match:
            try:
                parsed = extract_from_data(json.loads(tool_match.group(1).strip()))
                if parsed is not None:
                    return parsed
            except Exception:
                pass

        for block in iter_json_blocks(text):
            try:
                parsed = extract_from_data(json.loads(block))
                if parsed is not None:
                    return parsed
            except Exception:
                pass

        key_match = re.search(
            r"\"?(?:position_2d_pixel|pixel_position|image_position|position)\"?\s*[:=]\s*\[([^\]]+)\]",
            text,
            flags=re.DOTALL,
        )
        if key_match:
            nums = re.findall(r"-?\d+(?:\.\d+)?", key_match.group(1))
            if len(nums) >= 2:
                return normalize_coord([nums[0], nums[1]])

        return None

    def _save_cache(self):
        if not self.cache_path:
            return
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.warning(f"Failed to save Qwen3-VL coordinate cache: {e}")

    def predict(
        self,
        room_img: Image.Image,
        object_img: Image.Image,
        sample: dict,
        cache_key: str,
    ) -> Tuple[Optional[Tuple[float, float]], Optional[str], Optional[str]]:
        if not self.refresh_cache and cache_key in self._cache:
            cached = self._cache[cache_key]
            coord = cached.get("coord")
            if coord is None and cached.get("response"):
                coord = self._parse_coordinate(cached["response"], room_img.size)
                if coord is not None:
                    cached["coord"] = list(coord)
                    self._save_cache()
            parsed_coord = (float(coord[0]), float(coord[1])) if coord is not None else None
            return parsed_coord, cached.get("response"), cached.get("prompt")

        self._load_model()
        current_scene = self._build_current_scene(sample, room_img.size)
        prompt = self._build_prompt(sample["object_desc"], room_img.size, current_scene)
        response = self._generate(room_img, object_img, prompt)
        coord = self._parse_coordinate(response or "", room_img.size) if response else None

        self._cache[cache_key] = {
            "coord": list(coord) if coord is not None else None,
            "response": response,
            "prompt": prompt,
            "current_scene": current_scene,
            "prompt_mode": "scenerevis_current_scene_v1",
        }
        self._save_cache()
        return coord, response, prompt

    def _generate(self, room_img: Image.Image, object_img: Image.Image, prompt: str) -> Optional[str]:
        if self.backend == "vllm":
            return self._generate_vllm(room_img, object_img, prompt)
        return self._generate_transformers(room_img, object_img, prompt)

    def _generate_vllm(self, room_img: Image.Image, object_img: Image.Image, prompt: str) -> Optional[str]:
        from vllm import SamplingParams

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": np.array(room_img)},
                {"type": "image", "image": np.array(object_img)},
                {"type": "text", "text": prompt},
            ],
        }]
        text_prompt = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        vllm_input = {
            "prompt": text_prompt,
            "multi_modal_data": {"image": [room_img, object_img]},
        }
        sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=1.0,
            max_tokens=self.max_tokens,
        )
        outputs = self._vllm_engine.generate([vllm_input], sampling_params, use_tqdm=False)
        text = outputs[0].outputs[0].text.strip()
        return text if text else None

    def _generate_transformers(self, room_img: Image.Image, object_img: Image.Image, prompt: str) -> Optional[str]:
        from transformers import GenerationConfig

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": room_img},
                {"type": "image", "image": object_img},
                {"type": "text", "text": prompt},
            ],
        }]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[text],
            images=[room_img, object_img],
            return_tensors="pt",
        ).to(self._model.device)

        with torch.no_grad():
            generation_kwargs = {
                "max_new_tokens": self.max_tokens,
                "do_sample": self.temperature > 0,
            }
            if self.temperature > 0:
                generation_kwargs["temperature"] = self.temperature
            output_ids = self._model.generate(
                **inputs,
                generation_config=GenerationConfig(**generation_kwargs),
            )
        input_len = inputs["input_ids"].shape[1]
        result = self._processor.decode(
            output_ids[0, input_len:], skip_special_tokens=True
        ).strip()
        return result if result else None


def resize_heatmap_for_overlay(heatmap, image_size):
    """Resize a heatmap to match the displayed room image size."""
    image_w, image_h = image_size
    heatmap_uint8 = np.clip(heatmap * 255, 0, 255).astype(np.uint8)
    resized = Image.fromarray(heatmap_uint8).resize((image_w, image_h), Image.Resampling.BILINEAR)
    return np.array(resized).astype(np.float32) / 255.0


def scale_peak_to_image(peak, heatmap_shape, image_size):
    """Scale a peak from heatmap coordinates to image pixel coordinates."""
    image_w, image_h = image_size
    heatmap_h, heatmap_w = heatmap_shape
    x = (peak[0] + 0.5) * image_w / heatmap_w - 0.5
    y = (peak[1] + 0.5) * image_h / heatmap_h - 0.5
    return x, y


def image_point_to_heatmap(point, image_size, heatmap_shape):
    """Scale a point from image coordinates to heatmap coordinates."""
    image_w, image_h = image_size
    heatmap_h, heatmap_w = heatmap_shape
    x = (point[0] + 0.5) * heatmap_w / image_w - 0.5
    y = (point[1] + 0.5) * heatmap_h / image_h - 0.5
    return x, y


def point_distance(a, b):
    """Euclidean distance between two 2D points."""
    return float(np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2))


def save_three_way_comparison(
    room_img,
    gt_peak_overlay,
    pred_peak_overlay,
    qwen_peak_overlay,
    output_path,
    pred_dist,
    qwen_dist,
):
    """Save a standalone GT vs heatmap model vs Qwen3-VL comparison figure."""
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(room_img)
    ax.plot(gt_peak_overlay[0], gt_peak_overlay[1], 'rx', markersize=24, markeredgewidth=4, label='GT')
    ax.plot(pred_peak_overlay[0], pred_peak_overlay[1], 'b+', markersize=28, markeredgewidth=5,
            label=f'Heatmap Pred ({pred_dist:.1f}px)')

    if qwen_peak_overlay is not None:
        ax.plot(qwen_peak_overlay[0], qwen_peak_overlay[1], marker='*', color='magenta',
                markersize=26, markeredgewidth=2, label=f'Qwen3-VL ({qwen_dist:.1f}px)')
        ax.plot([gt_peak_overlay[0], qwen_peak_overlay[0]], [gt_peak_overlay[1], qwen_peak_overlay[1]],
                color='magenta', linewidth=2, linestyle='--')

    ax.plot([gt_peak_overlay[0], pred_peak_overlay[0]], [gt_peak_overlay[1], pred_peak_overlay[1]],
            color='yellow', linewidth=3)
    ax.set_title('GT vs Heatmap Pred vs Qwen3-VL', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_sample(
    model,
    sample,
    data_dir,
    device,
    output_path,
    image_size=384,
    qwen3_baseline: Optional[Qwen3VLCoordinateBaseline] = None,
    comparison_output_path: Optional[Path] = None,
    legacy_flip_gt_z: bool = False,
):
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
    if legacy_flip_gt_z:
        mask_img = mask_img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

    # Load original scene image if available
    original_img = None
    if "original_image_path" in sample:
        original_path = scene_dir / sample["original_image_path"]
        if original_path.exists():
            original_img = Image.open(original_path).convert("RGB")

    # Preprocess room/object separately so DINOv2 room encoders use their own normalization.
    room_tensor = model.preprocess_room_image(room_img).unsqueeze(0).to(device)
    object_tensor = model.preprocess_object_image(object_img).unsqueeze(0).to(device)
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
    dist = point_distance(pred_peak, gt_peak)

    # Overlay panels use the room image as the coordinate system. The heatmaps are
    # 256x256, while the rendered top-view image is usually 1024x1024.
    gt_heatmap_overlay = resize_heatmap_for_overlay(gt_heatmap, room_img.size)
    pred_heatmap_overlay = resize_heatmap_for_overlay(pred_heatmap, room_img.size)
    gt_peak_overlay = scale_peak_to_image(gt_peak, gt_heatmap.shape, room_img.size)
    pred_peak_overlay = scale_peak_to_image(pred_peak, pred_heatmap.shape, room_img.size)
    radius = max(room_img.size) * 15 / max(gt_heatmap.shape)

    qwen_peak_overlay = None
    qwen_dist = None
    qwen_response = None
    if qwen3_baseline is not None:
        try:
            cache_key = (
                f"scenerevis_current_scene_v1_mt{qwen3_baseline.max_tokens}_"
                f"{sample.get('sample_id', '')}_{sample.get('text_source', '')}_"
                f"{sample.get('object_desc', '')[:80]}"
            )
            qwen_peak_overlay, qwen_response, _ = qwen3_baseline.predict(
                room_img=room_img,
                object_img=object_img,
                sample=sample,
                cache_key=cache_key,
            )
            if qwen_peak_overlay is not None:
                qwen_peak_heatmap = image_point_to_heatmap(qwen_peak_overlay, room_img.size, gt_heatmap.shape)
                qwen_dist = point_distance(qwen_peak_heatmap, gt_peak)
        except Exception as e:
            logging.warning(f"Qwen3-VL baseline failed for {sample.get('sample_id', 'unknown')}: {e}")

    # Build title: text source, object description, peak distance.
    # Keep the long scene/sample id out of the figure title for readability.
    text_source = sample.get("text_source", "unknown")
    title_line1 = safe_plot_text(f'({text_source}) {sample["object_desc"]}')
    qwen_title = f" | Qwen3-VL: {qwen_dist:.1f}px" if qwen_dist is not None else ""
    title_line2 = f'Peak Distance: {dist:.1f}px{qwen_title} | Pred: {pred_peak} | GT: {gt_peak}'

    # Visualization: 2x4 layout
    fig = plt.figure(figsize=(26, 13))
    suptitle_kwargs = {"fontsize": 16, "fontweight": "bold", "y": 0.98}
    if CJK_FONT_PROP is not None:
        suptitle_kwargs["fontproperties"] = CJK_FONT_PROP
    fig.suptitle(f'{title_line1}\n{title_line2}', **suptitle_kwargs)

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
    ax5.imshow(gt_heatmap_overlay, cmap='jet', alpha=0.6, vmin=0, vmax=1)
    ax5.plot(gt_peak_overlay[0], gt_peak_overlay[1], 'r+', markersize=30, markeredgewidth=5, label='GT Peak')
    circle_gt = plt.Circle(gt_peak_overlay, radius, color='red', fill=False, linewidth=3, linestyle='--')
    ax5.add_patch(circle_gt)
    ax5.set_title('GT Placement (Top View)', fontsize=12, fontweight='bold')
    ax5.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax5.axis('off')

    # [2,3] Predicted placement
    ax6 = plt.subplot(2, 4, 7)
    ax6.imshow(room_img)
    ax6.imshow(pred_heatmap_overlay, cmap='jet', alpha=0.6, vmin=0, vmax=1)
    ax6.plot(pred_peak_overlay[0], pred_peak_overlay[1], 'b+', markersize=30, markeredgewidth=5, label='Pred Peak')
    circle_pred = plt.Circle(pred_peak_overlay, radius, color='blue', fill=False, linewidth=3, linestyle='-')
    ax6.add_patch(circle_pred)
    ax6.set_title('Predicted Placement (Top View)', fontsize=12, fontweight='bold')
    ax6.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax6.axis('off')

    # [2,4] Qwen3-VL coordinate baseline
    ax7 = plt.subplot(2, 4, 8)
    ax7.imshow(room_img)
    ax7.plot(gt_peak_overlay[0], gt_peak_overlay[1], 'rx', markersize=25, markeredgewidth=4, label='GT Peak')
    circle_gt2 = plt.Circle(gt_peak_overlay, radius, color='red', fill=False, linewidth=3, linestyle='--')
    ax7.add_patch(circle_gt2)
    if qwen_peak_overlay is not None:
        ax7.plot(qwen_peak_overlay[0], qwen_peak_overlay[1], marker='*', color='magenta',
                 markersize=28, markeredgewidth=2, label='Qwen3-VL')
        circle_qwen = plt.Circle(qwen_peak_overlay, radius, color='magenta', fill=False, linewidth=3, linestyle='-')
        ax7.add_patch(circle_qwen)
        ax7.plot([gt_peak_overlay[0], qwen_peak_overlay[0]], [gt_peak_overlay[1], qwen_peak_overlay[1]],
                 color='magenta', linewidth=3, label=f'Dist: {qwen_dist:.1f}px')
    else:
        ax7.text(0.5, 0.5, 'Qwen3-VL unavailable', ha='center', va='center',
                 transform=ax7.transAxes, fontsize=14, color='magenta')
        if qwen_response:
            ax7.text(0.5, 0.42, safe_plot_text(qwen_response[:80]), ha='center', va='center',
                     transform=ax7.transAxes, fontsize=8, color='magenta')
    ax7.set_title('Qwen3-VL Placement (Top View)', fontsize=12, fontweight='bold')
    ax7.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax7.axis('off')

    if comparison_output_path is not None:
        save_three_way_comparison(
            room_img=room_img,
            gt_peak_overlay=gt_peak_overlay,
            pred_peak_overlay=pred_peak_overlay,
            qwen_peak_overlay=qwen_peak_overlay,
            output_path=comparison_output_path,
            pred_dist=dist,
            qwen_dist=qwen_dist if qwen_dist is not None else float("nan"),
        )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return dist, qwen_dist


def main():
    parser = argparse.ArgumentParser(description="Visualize heatmap training results")
    parser.add_argument("--data_dir", type=str, required=True, help="Data directory")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to visualize")
    parser.add_argument("--output_dir", type=str, default="visualizations", help="Output directory")
    parser.add_argument("--image_size", type=int, default=384, help="Image resolution")
    parser.add_argument("--room_encoder", type=str, default=None, choices=["siglip", "dinov2"],
                        help="Room/top-view encoder used by the checkpoint")
    parser.add_argument("--dino_model", type=str, default=None,
                        help="DINOv2 model path or HF id for room_encoder=dinov2")
    parser.add_argument("--room_image_size", type=int, default=None,
                        help="Room image input size; defaults to 518 for DINOv2 and image_size for SigLIP")
    parser.add_argument("--object_image_size", type=int, default=None,
                        help="Object image input size; defaults to image_size")
    parser.add_argument("--hidden_dim", type=int, default=None,
                        help="Trainable attention hidden dimension used by the checkpoint")
    parser.add_argument("--decoder_layers", type=int, default=None,
                        help="Number of SAM-style two-way decoder layers used by the checkpoint")
    parser.add_argument("--num_heads", type=int, default=None,
                        help="Number of attention heads used by the checkpoint")
    parser.add_argument("--mlp_ratio", type=float, default=None,
                        help="MLP expansion ratio used by the checkpoint")
    parser.add_argument("--decoder_dropout", type=float, default=None,
                        help="Decoder dropout used by the checkpoint")
    parser.add_argument("--split", type=str, default="val", help="Dataset split (train/val/test)")
    parser.add_argument("--qwen3-vl-model", type=str, default="/mnt/f/models/qwen3_vl",
                        help="Qwen3-VL model path for coordinate baseline")
    parser.add_argument("--scene_json_dir", type=str, default=None,
                        help="Original SceneReVis/3D-FRONT scene JSON directory for Qwen3-VL current_scene prompts")
    parser.add_argument("--qwen3-vl-backend", type=str, default="vllm",
                        choices=["vllm", "transformers"], help="Qwen3-VL inference backend")
    parser.add_argument("--qwen3-vl-max-tokens", type=int, default=768,
                        help="Maximum new tokens for Qwen3-VL coordinate baseline")
    parser.add_argument("--disable-qwen3-vl-baseline", action="store_true",
                        help="Disable Qwen3-VL coordinate baseline")
    parser.add_argument("--qwen3-vl-cache", type=str, default=None,
                        help="Cache file for Qwen3-VL coordinate predictions")
    parser.add_argument("--refresh-qwen3-vl-cache", action="store_true",
                        help="Ignore existing Qwen3-VL coordinate cache and regenerate predictions")
    parser.add_argument("--legacy-flip-gt-z", action="store_true",
                        help="Flip old GT heatmaps vertically for datasets generated before the Z-axis fix")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    if not HAS_CJK_FONT:
        logging.warning(
            "No CJK-capable font found. Chinese characters in plot titles will be replaced "
            "to avoid Matplotlib missing-glyph warnings."
        )

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

    checkpoint_meta = torch.load(args.checkpoint, map_location="cpu")
    checkpoint_args = checkpoint_meta.get("args", {})
    room_encoder = args.room_encoder or checkpoint_args.get("room_encoder", "siglip")
    dino_model = args.dino_model or checkpoint_args.get("dino_model")
    hidden_dim = args.hidden_dim or checkpoint_args.get("hidden_dim", 256)
    decoder_layers = args.decoder_layers or checkpoint_args.get("decoder_layers", 3)
    num_heads = args.num_heads or checkpoint_args.get("num_heads", 8)
    mlp_ratio = args.mlp_ratio or checkpoint_args.get("mlp_ratio", 4.0)
    decoder_dropout = args.decoder_dropout
    if decoder_dropout is None:
        decoder_dropout = checkpoint_args.get("decoder_dropout", 0.0)
    room_image_size = args.room_image_size or checkpoint_args.get("room_image_size")
    object_image_size = (
        args.object_image_size
        or checkpoint_args.get("object_image_size")
        or checkpoint_args.get("image_size")
        or args.image_size
    )
    logging.info(
        "Heatmap model config: room_encoder=%s, dino_model=%s, hidden_dim=%s, "
        "decoder_layers=%s, num_heads=%s, mlp_ratio=%s, decoder_dropout=%s, "
        "room_image_size=%s, object_image_size=%s",
        room_encoder,
        dino_model,
        hidden_dim,
        decoder_layers,
        num_heads,
        mlp_ratio,
        decoder_dropout,
        room_image_size,
        object_image_size,
    )

    # Load model
    model = PlacementHeatmap(
        heatmap_res=256,
        room_encoder=room_encoder,
        dino_model=dino_model,
        hidden_dim=hidden_dim,
        room_image_size=room_image_size,
        object_image_size=object_image_size,
        decoder_layers=decoder_layers,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        decoder_dropout=decoder_dropout,
    ).to(device)
    model = load_checkpoint(model, args.checkpoint, device)

    qwen3_baseline = None
    if not args.disable_qwen3_vl_baseline and args.qwen3_vl_model:
        if not args.scene_json_dir:
            raise ValueError("--scene_json_dir is required when Qwen3-VL baseline is enabled")
        qwen3_model_path = Path(args.qwen3_vl_model)
        scene_json_dir = Path(args.scene_json_dir)
        if qwen3_model_path.exists():
            if not scene_json_dir.exists():
                raise FileNotFoundError(f"Scene JSON directory not found: {scene_json_dir}")
            cache_path = Path(args.qwen3_vl_cache) if args.qwen3_vl_cache else output_dir / "qwen3_vl_scenerevis_results.json"
            qwen3_baseline = Qwen3VLCoordinateBaseline(
                model_path=str(qwen3_model_path),
                scene_json_dir=str(scene_json_dir),
                backend=args.qwen3_vl_backend,
                max_tokens=args.qwen3_vl_max_tokens,
                temperature=0.0,
                cache_path=cache_path,
                refresh_cache=args.refresh_qwen3_vl_cache,
            )
            logging.info(f"Qwen3-VL SceneReVis coordinate baseline enabled: {qwen3_model_path}")
            logging.info(f"Qwen3-VL SceneReVis scene JSON dir: {scene_json_dir}")
            logging.info(f"Qwen3-VL coordinate cache: {cache_path}")
            logging.info(f"Qwen3-VL max new tokens: {args.qwen3_vl_max_tokens}")
            logging.info(f"Qwen3-VL refresh cache: {args.refresh_qwen3_vl_cache}")
        else:
            logging.warning(f"Qwen3-VL model path not found, baseline disabled: {qwen3_model_path}")

    # Visualize
    distances = []
    qwen_distances = []
    for i, sample in enumerate(selected_samples):
        output_path = output_dir / f"sample_{i:03d}_{sample['sample_id']}.png"
        comparison_output_path = output_dir / f"sample_{i:03d}_{sample['sample_id']}_three_way.png"
        logging.info(f"[{i+1}/{len(selected_samples)}] {sample['object_desc']}")
        dist, qwen_dist = visualize_sample(
            model,
            sample,
            data_dir,
            device,
            output_path,
            args.image_size,
            qwen3_baseline=qwen3_baseline,
            comparison_output_path=comparison_output_path if qwen3_baseline is not None else None,
            legacy_flip_gt_z=args.legacy_flip_gt_z,
        )
        distances.append(dist)
        if qwen_dist is not None:
            qwen_distances.append(qwen_dist)
        logging.info(f"  Peak distance: {dist:.1f}px -> {output_path}")
        if qwen_dist is not None:
            logging.info(f"  Qwen3-VL distance: {qwen_dist:.1f}px -> {comparison_output_path}")

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
    if qwen_distances:
        qwen_avg = np.mean(qwen_distances)
        qwen_median = np.median(qwen_distances)
        qwen_acc_32 = sum(1 for d in qwen_distances if d < 32) / len(qwen_distances)
        logging.info(f"  Qwen3-VL samples: {len(qwen_distances)}")
        logging.info(f"  Qwen3-VL avg distance: {qwen_avg:.1f}px")
        logging.info(f"  Qwen3-VL median distance: {qwen_median:.1f}px")
        logging.info(f"  Qwen3-VL accuracy (<32px): {qwen_acc_32:.1%}")
    logging.info(f"  Visualizations saved to: {output_dir}")
    logging.info(f"{'='*60}")


if __name__ == "__main__":
    main()
