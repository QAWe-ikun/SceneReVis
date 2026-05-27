# 热力图放置模型训练指南

本目录包含 SceneReVis 热力图放置模型的完整数据准备和训练流程。

## 架构概述

模型使用 **SigLIP + CLIP** 双编码器架构：

- **SigLIP ViT** (`google/siglip-so400m-patch14-384`): 编码房间俯视图，提取空间特征 (27×27 网格, 1152 维)
- **CLIP ViT-L-14** (`openai/clip-vit-large-patch14`): 编码物体参考图和文本描述 (768 维对齐特征)
- **融合层**: 将 CLIP 特征拼接后投影到 SigLIP 维度
- **交叉注意力**: 以融合后的物体+文本特征为 query，房间空间特征为 key/value
- **热力图头**: 输出 256×256 的放置概率热力图

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
- `data.output_dir`: 输出目录
- `generation.image_size`: 图像分辨率 (默认 1024)
- `generation.heatmap_sigma`: 高斯热力图 sigma (默认 15 像素)
- `generation.max_object_nums`: 每个场景最多处理物体数 (默认 5)
- `generation.text.augmentation_prob`: 文本增强概率 (默认 0.5)

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
│       └── masks/                 # GT 热力图
│           └── obj_xxx_mask.png
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
    "split": "train"
  }
]
```

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
print(f'示例: {samples[0]}')
"
```

## 模型测试

### 验证前向传播

```bash
cd script/pretreatment
python test_forward.py
```

测试内容：
- 模型初始化
- 单样本前向传播
- 批量前向传播
- 输出验证 (形状、值域)

## 数据集加载

### 使用 PyTorch Dataset

```python
from script.pretreatment.dataset import HeatmapPlacementDataset, collate_fn
from torch.utils.data import DataLoader
from pathlib import Path

# 创建数据集
dataset = HeatmapPlacementDataset(
    data_dir=Path("./output/heatmap_data"),
    split="train",
    image_size=1024,
    mask_size=256,
    normalize=True,
)

# 创建 DataLoader
loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=4,
)

# 训练循环
for batch in loader:
    room_images = batch["room_image"]       # (B, 3, 1024, 1024)
    object_images = batch["object_image"]   # (B, 3, 1024, 1024)
    masks = batch["mask"]                   # (B, 1, 256, 256)
    descs = batch["object_desc"]            # list of str
    
    # 前向传播
    # heatmaps = model(room_images, descs, object_images)
    # loss = criterion(heatmaps, masks)
    # ...
```

### 测试数据集加载

```bash
python dataset.py --data_dir ./output/heatmap_data --split train
```

## 训练

### 基础训练脚本 (示例)

```python
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from pathlib import Path

from script.pretreatment.dataset import HeatmapPlacementDataset, collate_fn
from utils.placement_heatmap import PlacementHeatmap

# 数据
train_dataset = HeatmapPlacementDataset(
    data_dir=Path("./output/heatmap_data"),
    split="train",
)
train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=4,
)

# 模型
model = PlacementHeatmap(device="cuda")

# 优化器
optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

# 损失函数
criterion = torch.nn.MSELoss()

# 训练循环
for epoch in range(100):
    for batch in train_loader:
        room_images = batch["room_image"].to(model.device)
        object_images = batch["object_image"].to(model.device)
        masks = batch["mask"].to(model.device)
        descs = batch["object_desc"]
        
        # 前向传播
        heatmaps = model(room_images, descs, object_images)
        
        # 计算损失
        loss = criterion(heatmaps, masks)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

## 推理

### 使用 PlacementEngine

```python
from utils.placement_heatmap import PlacementEngine

# 加载训练好的模型
engine = PlacementEngine(
    checkpoint_path="checkpoints/heatmap_model.pth",
    device="cuda"
)

# 放置物体
result = engine.place_object(
    room_image_path="path/to/room.png",
    object_desc="a wooden chair",
    object_image_path="path/to/chair.png",
    bounds_bottom=[...],  # 房间边界
    top_k=5,  # 返回 top-k 个候选位置
)

# 结果
positions = result["positions"]  # list of (x, y, z) 世界坐标
scores = result["scores"]        # list of float 置信度
heatmap = result["heatmap"]      # numpy array (256, 256)
```

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

### CLIP 特征对齐

CLIP 的视觉和文本编码器已经在同一特征空间中预训练，因此：
- 物体图像特征: `clip.encode_image(object_image)` → 768 维
- 文本特征: `clip.encode_text(description)` → 768 维
- 拼接后投影到 SigLIP 维度 (1152)

## 故障排除

### 内存不足

- 减小 `batch_size`
- 使用梯度累积
- 冻结 SigLIP/CLIP 编码器，只训练融合层和热力图头

### 训练不稳定

- 降低学习率 (从 1e-4 降到 1e-5)
- 使用 warmup 调度器
- 增加 `heatmap_sigma` 使 GT 更平滑

### 数据加载慢

- 增加 `num_workers`
- 使用 SSD 存储
- 预处理图像到更小分辨率 (如 512×512)

## 文件列表

- `generate_data.py`: 数据生成入口
- `data_generator.py`: 数据生成器主类
- `config.yaml`: 数据生成配置
- `dataset.py`: PyTorch Dataset 类
- `test_forward.py`: 模型前向传播测试
- `components/`:
  - `scene_builder.py`: 场景构建
  - `renderer.py`: 正交投影渲染
  - `heatmap_generator.py`: GT 热力图生成
  - `sample_saver.py`: 样本保存
  - `text_processor.py`: 文本增强
- `models.py`: 数据结构定义
