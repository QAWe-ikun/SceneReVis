# 热力图放置模型训练指南

本目录包含 SceneReVis 热力图放置模型的完整数据准备、训练和可视化流程。

## 架构概述

模型使用 **纯 SigLIP** 单编码器架构，同时处理房间空间、物体视觉和文本语义：

- **SigLIP ViT** (`google/siglip-so400m-patch14-384`): 统一编码器
  - 房间俯视图 → 27×27 空间特征网格 (1152 维)
  - 物体参考图 → 全局物体特征 (1152 维)
  - 文本描述 → 文本特征 (1152 维)
- **特征融合**: 物体特征 + 文本特征拼接后投影到 1152 维
- **全局自注意力 (SpatialRefinement)**: 27×27 = 729 个 token 做全局自注意力，捕捉长距离空间依赖
- **交叉注意力**: 房间空间特征为 Query，物体+文本融合特征为 Key，每个空间位置评估 "我这里适合放置该物体吗"
- **热力图头**: 输出 256×256 的放置概率热力图 (sigmoid + 可学习 logit_bias)

## 数据准备

### 1. 生成训练数据

从 3D-FUTURE 场景 JSON 生成训练样本：

```bash
cd script/pretreatment

# 使用默认配置
python generate_data.py --config config.yaml

# 命令行覆盖路径
python generate_data.py \
  --config config.yaml \
  --scene_dir /path/to/scene_jsons \
  --model_dir /path/to/3D-FUTURE-model \
  --output_dir ./output/heatmap_data
```

**配置项** (`config.yaml`):
- `data.scene_dir`: 场景 JSON 目录
- `data.model_dir`: 3D-FUTURE 模型文件目录
- `data.metadata_dir`: 元数据 JSON 目录 (默认 `model_dir/../metadata`)
- `data.output_dir`: 输出目录
- `generation.image_size`: 图像分辨率 (默认 1024)
- `generation.heatmap_sigma`: 高斯热力图 sigma (默认 15 像素)
- `generation.adaptive_sigma`: 是否使用物体尺寸自适应 sigma (默认 false)
- `generation.max_object_nums`: 每个场景最多处理物体数 (默认 5)
- `generation.text.*`: 文本增强配置
- `generation.vlm.*`: VLM 描述生成配置 (见下方 [VLM 描述生成](#vlm-描述生成))

**输出结构**:
```
output/heatmap_data/
├── train/
│   ├── train.json                 # 训练集元数据
│   └── scene_001/
│       ├── plane_images/          # 房间俯视图 (剔除目标物体)
│       │   └── obj_xxx.png
│       ├── object_images/         # 物体参考图 (居中)
│       │   └── obj_xxx_object.png
│       ├── masks/                 # GT 热力图
│       │   └── obj_xxx_mask.png
│       └── original_images/       # 完整场景俯视图 (仅 VLM 启用时)
│           └── obj_xxx_original.png
├── val/
│   ├── val.json
│   └── ...
└── test/
    ├── test.json
    └── ...
```

**JSON 元数据格式**:
```json
[
  {
    "sample_id": "obj_xxx",
    "scene_dir": "train/scene_001",
    "plane_image_path": "plane_images/obj_xxx.png",
    "object_image_path": "object_images/obj_xxx_object.png",
    "mask_path": "masks/obj_xxx_mask.png",
    "object_desc": "a wooden chair with armrests",
    "split": "train",
    "scene_name": "scene_001",
    "text_source": "text_processor",
    "original_image_path": "original_images/obj_xxx_original.png",
    "removed_object": {
      "jid": "abc123",
      "model_id": "abc123",
      "desc": "椅子",
      "pos": [1.5, 0.0, 3.2],
      "rot": [0, 0.707, 0, 0.707],
      "size": [0.5, 0.8, 0.9]
    }
  }
]
```

**字段说明**:
| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `sample_id` | str | 是 | 样本唯一标识 (`obj_{jid}`) |
| `scene_dir` | str | 是 | 场景相对路径 (`{split}/{scene_name}`) |
| `plane_image_path` | str | 是 | 房间俯视图 (剔除目标物体) |
| `object_image_path` | str | 是 | 物体参考图 (居中渲染) |
| `mask_path` | str | 是 | GT 高斯热力图 (灰度) |
| `object_desc` | str | 是 | 物体描述 (TextProcessor 或 VLM 生成) |
| `split` | str | 是 | 数据集划分 (train/val/test) |
| `scene_name` | str | 否 | 场景名称 |
| `text_source` | str | 是 | 描述来源: `"text_processor"` 或 `"vlm"` |
| `original_image_path` | str | 否 | 完整场景俯视图 (仅 VLM 启用时存在) |
| `removed_object` | dict | 否 | 被移除物体的 3D 元数据 (见下) |

**removed_object 字段**:
| 字段 | 说明 |
|------|------|
| `jid` | 物体实例 ID |
| `model_id` | 3D-FUTURE 模型 ID |
| `desc` | 原始简短描述 |
| `pos` | 世界坐标位置 `[x, y, z]` |
| `rot` | 四元数旋转 `[x, y, z, w]` |
| `size` | 包围盒尺寸 `[w, h, d]` |

### 2. 数据组件

- **SceneBuilder**: 加载场景 JSON + 3D-FUTURE `.glb` 模型
- **OrthoRenderer**: 正交投影渲染 (与 SceneReVis Blender 相机对齐)
  - 房间俯视图: `ortho_scale = span * 1.2`
  - 物体参考图: 居中于原点，使用相同 ortho_scale
- **HeatmapGenerator**: 生成高斯 GT 热力图
  - mask 网格约定: `heatmap[gi=X, gj=Z]`，其中 `gj=0 → z_min`
  - 与 `utils/placement_mask.py` 直接对齐，无需翻转
- **TextProcessor**: 从多个元数据源加载丰富的物体描述
  - `model_info_3dfuture_assets.json`: summary
  - `model_info_3dfuture_assets_prompts.json`: 多变体描述
  - `model_info_3dfuture_assets_simple_descs.json`: 简单分类
- **VLMClient**: 使用 Qwen3-VL 生成摆放位置描述 (可选)
- **SampleSaver**: 保存图片和元数据 JSON

### 3. 测试数据生成

```bash
# 检查生成的数据
python -c "
import json
from pathlib import Path
data_dir = Path('./output/heatmap_data/train')
with open(data_dir / 'train.json') as f:
    samples = json.load(f)
print(f'训练样本数: {len(samples)}')
print(f'示例: {json.dumps(samples[0], indent=2, ensure_ascii=False)}')
"
```

## VLM 描述生成

启用 VLM 后，使用 Qwen3-VL 对比 3 张图生成英文摆放请求，比 TextProcessor 的静态描述信息更丰富，并避免训练/可视化 prompt 中英文混用。

### 工作流程

1. 渲染完整场景俯视图 (每个场景只渲染一次)
2. 对每个目标物体: 渲染剔除后房间 + 物体参考图
3. 将 3 张图送入 Qwen3-VL，生成英文描述 (如 "Please place the bed against the left wall of the room...")
4. VLM 失败时自动回退到 TextProcessor

### 启用配置

```yaml
generation:
  vlm:
    enabled: true
    model_path: "/path/to/Qwen3-VL"
    backend: "vllm"        # "vllm" (快) 或 "transformers" (回退)
    max_tokens: 256
    temperature: 0.7
    cache_enabled: true    # 磁盘缓存，避免重复生成
```

### 向后兼容

- `vlm.enabled: false` (默认): 输出与不使用 VLM 时完全一致
- 旧 JSON + 新代码: 新字段不存在时自动跳过 (`if key in sample`)
- 新 JSON + 旧代码: 多余字段被忽略
- **模型代码 (`placement_heatmap.py`) 零改动**: SigLIP text encoder 不关心文本来源

## 训练

### 使用训练脚本

```bash
cd script/pretreatment

# 基础训练
python train_heatmap.py \
  --data_dir /path/to/heatmap_data \
  --output_dir checkpoints/heatmap \
  --epochs 100 \
  --batch_size 4 \
  --lr 1e-4

# 快速测试学习率 (1 epoch 内 cosine 衰减)
python train_heatmap.py \
  --data_dir /path/to/heatmap_data \
  --output_dir checkpoints/test_lr_1e-4 \
  --lr 1e-4 \
  --test_lr

# 恢复训练
python train_heatmap.py \
  --data_dir /path/to/heatmap_data \
  --output_dir checkpoints/heatmap \
  --epochs 100 \
  --lr 1e-4 \
  --resume checkpoints/heatmap/latest.pth

# HSD-F baseline: same DINOv2/SigLIP two-way fusion, learned query + MLP coordinate head
python train_hsd_f.py \
  --data_dir /path/to/heatmap_data \
  --output_dir checkpoints/hsd_f_dinov2_twoway_hd256 \
  --epochs 20 \
  --batch_size 2 \
  --lr 1e-4 \
  --min_lr 1e-6 \
  --warmup_steps 1000 \
  --lr_scheduler step_cosine \
  --num_workers 4 \
  --room_encoder dinov2 \
  --dino_model "$SCENEREVIS_DINOV2_MODEL" \
  --hidden_dim 256 \
  --decoder_layers 3 \
  --num_heads 8 \
  --mlp_ratio 4.0 \
  --decoder_dropout 0.0
```

**训练参数**:
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_dir` | (必填) | 数据目录 |
| `--output_dir` | `checkpoints/heatmap` | 检查点输出目录 |
| `--epochs` | 100 | 训练轮数 |
| `--batch_size` | 4 | 批量大小 |
| `--lr` | 1e-4 | 初始学习率 |
| `--weight_decay` | 1e-4 | 权重衰减 |
| `--image_size` | 384 | 图像分辨率 (匹配 SigLIP 输入) |
| `--num_workers` | 4 | 数据加载线程数 |
| `--resume` | None | 恢复训练的检查点路径 |
| `--test_lr` | false | LR 测试模式: 1 epoch 内 cosine 衰减 |

**学习率调度**: CosineAnnealingLR (`eta_min=1e-6`)
- 正常模式: 按 epoch 衰减
- `--test_lr` 模式: 按 batch 衰减，1 epoch 内完成扫描

**损失函数**: 加权 Binary Cross Entropy
```python
loss = F.binary_cross_entropy(
    pred_heatmap, mask_resized,
    weight=torch.where(
        mask_resized > 0.1,
        torch.tensor(10.0),  # 峰值区域 10 倍权重
        torch.tensor(1.0),
    ),
)
```

**进度条指标**:
```
Epoch 1 [Train]: 39%|...| loss=0.0495, avg=0.0779, peak=25%, hm=[0.00,0.59], lr=1.0e-04
```
- `loss`: 当前 batch 损失
- `avg`: 累计平均损失
- `peak`: 峰值准确率 (预测峰值与 GT 峰值距离 < 32px)
- `hm`: 预测热力图值域 `[min, max]`
- `lr`: 当前学习率

### 使用 PyTorch Dataset

```python
from script.pretreatment.dataset import HeatmapPlacementDataset, collate_fn
from torch.utils.data import DataLoader
from pathlib import Path

dataset = HeatmapPlacementDataset(
    data_dir=Path("./output/heatmap_data"),
    split="train",
    image_size=1024,
    mask_size=256,
    normalize=True,
)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=4,
)

for batch in loader:
    room_images = batch["room_image"]       # (B, 3, 1024, 1024)
    object_images = batch["object_image"]   # (B, 3, 1024, 1024)
    masks = batch["mask"]                   # (B, 1, 256, 256)
    descs = batch["object_desc"]            # list of str
    # 可选新字段:
    # batch["scene_name"]                   # list of str
    # batch["text_source"]                  # list of str
    # batch["removed_object"]               # list of dict
```

## 可视化

### 可视化训练结果

```bash
python visualize_results.py \
  --data_dir /path/to/heatmap_data \
  --checkpoint checkpoints/heatmap/latest.pth \
  --num_samples 10 \
  --output_dir visualizations \
  --split val
```

输出 **2×4 布局** 的对比图:

| 位置 | 内容 |
|------|------|
| [1,1] | 原始完整场景 (含所有物体，VLM 未启用时显示 N/A) |
| [1,2] | 房间俯视图 (剔除目标物体后) |
| [1,3] | 物体参考图 |
| [1,4] | GT 热力图 + 峰值标记 |
| [2,1] | 预测热力图 + 峰值标记 |
| [2,2] | GT 摆放位置叠加 (房间 + 热力图 + 圆圈) |
| [2,3] | 预测摆放位置叠加 |
| [2,4] | 预测 vs GT 对比 (蓝/红圆圈 + 黄色连接线) |

标题格式: `[scene_name] (text_source) object_desc`

## 关键设计

### 正交投影对齐

渲染器与 SceneReVis 推理流程保持一致：
- `ortho_scale = span * 1.2`
- `span = max(x_max - x_min, z_max - z_min)`
- 相机位姿: 右=+X, 上=-Z, 后=+Y

### Mask 网格约定

热力图网格直接对应 `placement_mask.py`：
```python
# 世界坐标 → 网格坐标
gi = (x - x_min) / cell_size - 0.5
gj = (z - z_min) / cell_size - 0.5

# 热力图索引
heatmap[gi=X, gj=Z]
```

无需在推理时翻转轴。

### SigLIP 特征统一

所有输入使用同一 SigLIP 编码器：
- 房间图像: `siglip.encode(room_image)` → 27×27×1152 空间特征
- 物体图像: `siglip.encode(object_image)` → 1152 维全局特征
- 文本描述: `siglip.encode_text(description)` → 1152 维文本特征

## 故障排除

### 内存不足

- 减小 `batch_size`
- 使用梯度累积
- 冻结 SigLIP 编码器，只训练融合层和热力图头

### 训练不稳定

- 降低学习率 (从 1e-4 降到 1e-5)
- 使用 `--test_lr` 快速扫描合适的学习率
- 增加 `heatmap_sigma` 使 GT 更平滑

### 热力图值域异常

- `hm=[1.00,1.00]`: 模型输出均匀 → 检查是否误加了 max 归一化
- `hm=[0.00,0.00]`: logit_bias 过负 → 检查学习率是否过大

### 数据加载慢

- 增加 `num_workers`
- 使用 SSD 存储
- 预处理图像到更小分辨率 (如 384×384)

## 文件列表

- `generate_data.py`: 数据生成入口
- `data_generator.py`: 数据生成器主类
- `config.yaml`: 数据生成配置
- `train_heatmap.py`: 训练脚本 (含 `--test_lr` 模式)
- `dataset.py`: PyTorch Dataset 类
- `visualize_results.py`: 训练结果可视化
- `test_forward.py`: 模型前向传播测试
- `components/`:
  - `scene_builder.py`: 场景构建
  - `renderer.py`: 正交投影渲染
  - `heatmap_generator.py`: GT 热力图生成
  - `sample_saver.py`: 样本保存
  - `text_processor.py`: 文本增强
  - `vlm_client.py`: VLM 描述生成 (Qwen3-VL)
- `models.py`: 数据结构定义
