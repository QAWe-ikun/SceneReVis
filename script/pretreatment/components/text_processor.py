"""
文本处理器：为热力图训练提供丰富的物体描述
支持多种文本增强策略
"""
import json
import random
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class TextProcessor:
    """处理物体描述的文本增强"""

    metadata_dir: Path
    use_summary: bool = True
    use_prompts: bool = True
    use_simple_descs: bool = True
    augmentation_prob: float = 0.5

    def __post_init__(self):
        self.metadata_dir = Path(self.metadata_dir)

        # 加载各种描述数据
        self.summaries = {}
        self.simple_descs = {}
        self.prompts = {}

        if self.use_summary:
            summary_file = self.metadata_dir / "model_info_3dfuture_assets.json"
            if summary_file.exists():
                with open(summary_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.summaries = {k: v.get('summary', '') for k, v in data.items()}

        if self.use_simple_descs:
            simple_file = self.metadata_dir / "model_info_3dfuture_assets_simple_descs.json"
            if simple_file.exists():
                with open(simple_file, 'r', encoding='utf-8') as f:
                    self.simple_descs = json.load(f)

        if self.use_prompts:
            prompts_file = self.metadata_dir / "model_info_3dfuture_assets_prompts.json"
            if prompts_file.exists():
                with open(prompts_file, 'r', encoding='utf-8') as f:
                    self.prompts = json.load(f)

    def get_description(self, model_id: str, scene_desc: Optional[str] = None) -> str:
        """获取物体描述，支持多种策略和增强"""
        candidates = []

        # 1. 场景中的原始描述（如果有）
        if scene_desc:
            candidates.append(scene_desc)

        # 2. 从 prompts 中获取多个变体
        if model_id in self.prompts and self.prompts[model_id]:
            candidates.extend(self.prompts[model_id])

        # 3. 从 summaries 中获取
        if model_id in self.summaries and self.summaries[model_id]:
            candidates.append(self.summaries[model_id])

        # 4. 从 simple_descs 中获取（这是反向映射：summary -> category）
        # simple_descs 是 {summary: category}，我们需要找到匹配的 category
        if self.use_simple_descs:
            for summary, category in self.simple_descs.items():
                if summary in candidates or model_id in summary:
                    candidates.append(category)
                    break

        if not candidates:
            # 回退：使用 model_id 作为描述
            return f"object {model_id[:8]}"

        # 文本增强：随机选择一个描述
        if random.random() < self.augmentation_prob and len(candidates) > 1:
            return random.choice(candidates)
        else:
            # 优先使用场景描述，否则使用第一个候选
            return scene_desc if scene_desc else candidates[0]

    def get_descriptions_batch(self, model_ids: List[str],
                              scene_descs: Optional[List[str]] = None) -> List[str]:
        """批量获取描述"""
        if scene_descs is None:
            scene_descs = [None] * len(model_ids)

        return [
            self.get_description(model_id, scene_desc)
            for model_id, scene_desc in zip(model_ids, scene_descs)
        ]
