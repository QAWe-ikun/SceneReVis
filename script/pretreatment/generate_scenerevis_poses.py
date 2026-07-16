"""Generate SceneReVis rotation and scale/size proposals for HAP-Place.

The output is one resumable JSON containing prompts, complete model responses,
parsed tool calls, and validity flags. HAP-Place deliberately ignores any
position emitted by SceneReVis and consumes only rotation plus size/scale.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.hap_place import parse_scenerevis_pose


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SceneReVis pose proposals")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--scene_json_dir", required=True)
    parser.add_argument("--model", default=str(PROJECT_ROOT / "ckpt" / "SceneReVis-7B"))
    parser.add_argument("--backend", default="vllm", choices=["vllm", "transformers"])
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--sample_id", action="append", default=None)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--refresh", action="store_true", help="Regenerate existing sample records")
    return parser.parse_args()


def _extract_objects(scene: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if isinstance(scene.get("objects"), list):
        return list(scene["objects"])
    objects = []
    for group in scene.get("groups", []):
        objects.extend(group.get("objects", []))
    return objects


def _clean_object(obj: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: obj[key]
        for key in ("desc", "size", "pos", "rot", "jid")
        if key in obj
    }


def _load_current_scene(
    scene_json_dir: Path,
    sample: Mapping[str, Any],
) -> Dict[str, Any]:
    scene_name = sample.get("scene_name")
    if not scene_name:
        raise ValueError("Sample has no scene_name")
    scene_path = scene_json_dir / f"{scene_name}.json"
    with open(scene_path, "r", encoding="utf-8") as handle:
        source = json.load(handle)

    removed = sample.get("removed_object", {})
    removed_index = removed.get("instance_id")
    removed_jid = removed.get("jid")
    kept = []
    for index, obj in enumerate(_extract_objects(source)):
        is_target = index == removed_index if isinstance(removed_index, int) else obj.get("jid") == removed_jid
        if not is_target:
            kept.append(_clean_object(obj))
    envelope = source.get("room_envelope", {})
    return {
        "room_type": source.get("room_type"),
        "room_id": source.get("room_id"),
        "bounds_bottom": source.get("bounds_bottom", envelope.get("bounds_bottom")),
        "bounds_top": source.get("bounds_top", envelope.get("bounds_top")),
        "objects": kept,
    }


def _build_prompt(sample: Mapping[str, Any], current_scene: Mapping[str, Any]) -> str:
    scene_json = json.dumps(current_scene, ensure_ascii=False, indent=2)
    description = str(sample.get("object_desc", ""))
    return (
        "### Role and Core Directive\n\n"
        "You are the SceneReVis indoor-scene editing model. Analyze the removed-scene image, "
        "the target-object reference image, the language request, and the structured scene. "
        "Return one SceneReVis add_object operation.\n\n"
        "### Inputs\n\n"
        "Image 1 is the room after the target object was removed. Image 2 is the target object.\n\n"
        "<current_scene>\n"
        f"{scene_json}\n"
        "</current_scene>\n\n"
        "<placement_request>\n"
        f"{description}\n"
        "</placement_request>\n\n"
        "### Requirements\n\n"
        "1. Infer a functionally and aesthetically appropriate orientation and metric size.\n"
        "2. Keep the object inside the room and avoid obvious overlap.\n"
        "3. Use a quaternion [x, y, z, w] for rotation.\n"
        "4. Use [width, height, depth] in meters for size.\n"
        "5. Return exactly one add_object call.\n"
        "6. A position may be included for native SceneReVis behavior, but HAP-Place will discard it.\n"
        "7. Use English only and always finish the complete <tool_calls> block.\n\n"
        "### Output Format\n\n"
        "<think>\n"
        "Briefly reason about the target orientation and dimensions.\n"
        "</think>\n"
        "<tool_calls>\n"
        "[\n"
        "  {\n"
        "    \"id\": \"tool_1\",\n"
        "    \"name\": \"add_object\",\n"
        "    \"arguments\": {\n"
        "      \"object_description\": \"...\",\n"
        "      \"size\": [width, height, depth],\n"
        "      \"rotation\": [x, y, z, w],\n"
        "      \"placement_plane\": \"floor\"\n"
        "    }\n"
        "  }\n"
        "]\n"
        "</tool_calls>"
    )


class SceneReVisGenerator:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.model_path = Path(args.model)
        if not self.model_path.exists():
            raise FileNotFoundError(f"SceneReVis model does not exist: {self.model_path}")
        self.processor = None
        self.engine = None
        self.model = None

    def load(self) -> None:
        from transformers import AutoProcessor

        self.processor = AutoProcessor.from_pretrained(
            str(self.model_path),
            local_files_only=True,
            use_fast=True,
        )
        if self.args.backend == "vllm":
            from vllm import LLM

            self.engine = LLM(
                model=str(self.model_path),
                limit_mm_per_prompt={"image": 2},
                gpu_memory_utilization=self.args.gpu_memory_utilization,
                max_model_len=self.args.max_model_len,
                trust_remote_code=True,
                disable_log_stats=True,
                enforce_eager=True,
                disable_custom_all_reduce=True,
            )
        else:
            from transformers import Qwen2_5_VLForConditionalGeneration

            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                str(self.model_path),
                dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                attn_implementation="sdpa" if torch.cuda.is_available() else "eager",
                local_files_only=True,
            )
            self.model.eval()

    @staticmethod
    def _messages(room_image: Image.Image, object_image: Image.Image, prompt: str) -> list[dict]:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": room_image},
                    {"type": "image", "image": object_image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

    def generate_batch(self, items: Sequence[Dict[str, Any]]) -> list[str]:
        if self.args.backend == "vllm":
            return self._generate_vllm(items)
        return [self._generate_transformers(item) for item in items]

    def _generate_vllm(self, items: Sequence[Dict[str, Any]]) -> list[str]:
        from vllm import SamplingParams

        inputs = []
        for item in items:
            messages = self._messages(item["room_image"], item["object_image"], item["prompt"])
            text_prompt = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs.append(
                {
                    "prompt": text_prompt,
                    "multi_modal_data": {
                        "image": [item["room_image"], item["object_image"]],
                    },
                }
            )
        sampling = SamplingParams(
            temperature=self.args.temperature,
            top_p=1.0,
            max_tokens=self.args.max_tokens,
        )
        outputs = self.engine.generate(inputs, sampling, use_tqdm=False)
        return [output.outputs[0].text.strip() for output in outputs]

    def _generate_transformers(self, item: Dict[str, Any]) -> str:
        messages = self._messages(item["room_image"], item["object_image"], item["prompt"])
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(
            text=[text],
            images=[item["room_image"], item["object_image"]],
            return_tensors="pt",
        ).to(self.model.device)
        kwargs = {
            "max_new_tokens": self.args.max_tokens,
            "do_sample": self.args.temperature > 0,
        }
        if self.args.temperature > 0:
            kwargs["temperature"] = self.args.temperature
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **kwargs)
        input_length = inputs["input_ids"].shape[1]
        return self.processor.decode(
            output_ids[0, input_length:],
            skip_special_tokens=True,
        ).strip()


def _load_existing(path: Path, refresh: bool) -> Dict[str, Dict[str, Any]]:
    if refresh or not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    records = data.get("results", data)
    if isinstance(records, list):
        return {str(item["sample_id"]): item for item in records if "sample_id" in item}
    if isinstance(records, Mapping):
        return {str(key): dict(value) for key, value in records.items()}
    return {}


def _write_output(path: Path, args: argparse.Namespace, records: Mapping[str, Dict[str, Any]]) -> None:
    results = list(records.values())
    payload = {
        "schema": "scenerevis_pose_results_v1",
        "config": {
            "data_dir": args.data_dir,
            "split": args.split,
            "scene_json_dir": args.scene_json_dir,
            "model": args.model,
            "backend": args.backend,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
        },
        "summary": {
            "count": len(results),
            "valid": sum(bool(item.get("valid")) for item in results),
            "invalid": sum(not bool(item.get("valid")) for item in results),
        },
        "results": results,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")
    data_dir = Path(args.data_dir)
    scene_json_dir = Path(args.scene_json_dir)
    split_json = data_dir / args.split / f"{args.split}.json"
    with open(split_json, "r", encoding="utf-8") as handle:
        samples = json.load(handle)
    if args.sample_id:
        selected_ids = set(args.sample_id)
        samples = [sample for sample in samples if sample.get("sample_id") in selected_ids]
    else:
        samples = samples[max(0, args.start_index):]
        if args.num_samples > 0:
            samples = samples[:args.num_samples]

    output_path = Path(args.output_json)
    records = _load_existing(output_path, refresh=args.refresh)
    pending = [
        sample
        for sample in samples
        if str(sample["sample_id"]) not in records
        or not bool(records[str(sample["sample_id"])].get("valid"))
    ]
    if not pending:
        logging.info("No pending samples; use --refresh to regenerate existing records")
        _write_output(output_path, args, records)
        return

    generator = SceneReVisGenerator(args)
    generator.load()
    for start in range(0, len(pending), args.batch_size):
        batch_samples = pending[start:start + args.batch_size]
        batch_items = []
        for sample in batch_samples:
            sample_dir = data_dir / sample["scene_dir"]
            current_scene = _load_current_scene(scene_json_dir, sample)
            prompt = _build_prompt(sample, current_scene)
            batch_items.append(
                {
                    "sample": sample,
                    "current_scene": current_scene,
                    "prompt": prompt,
                    "room_image": Image.open(sample_dir / sample["plane_image_path"]).convert("RGB"),
                    "object_image": Image.open(sample_dir / sample["object_image_path"]).convert("RGB"),
                }
            )
        responses = generator.generate_batch(batch_items)
        for item, response in zip(batch_items, responses):
            sample_id = str(item["sample"]["sample_id"])
            pose = parse_scenerevis_pose(response, source=f"scenerevis:{sample_id}")
            records[sample_id] = {
                "sample_id": sample_id,
                "scene_name": item["sample"].get("scene_name"),
                "valid": pose is not None,
                "pose": pose.to_dict() if pose is not None else None,
                "response": response,
                "prompt": item["prompt"],
                "current_scene": item["current_scene"],
            }
            logging.info("[%d/%d] %s valid=%s", len(records), len(samples), sample_id, pose is not None)
        _write_output(output_path, args, records)

    logging.info("Wrote %d SceneReVis pose records to %s", len(records), output_path)


if __name__ == "__main__":
    main()
