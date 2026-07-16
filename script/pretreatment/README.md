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

## HAP-Place P0 pipeline

The complete release pipeline is implemented by:

- `utils/hap_place.py`: calibrated rays, exact first-hit mesh intersections,
  packed 3D occupancy, target voxelization, and score-ordered release search.
- `run_hap_place.py`: heatmap inference, SceneReVis pose loading, scene
  reconstruction, release search, result JSON, and the Isaac Sim handoff.
- `run_isaac_settle.py`: Isaac Sim 6.x drop-and-settle worker.

Install the exact ray-intersection dependency in the WSL environment:

```bash
pip install rtree
```

SceneReVis results must be keyed by `sample_id` and contain an `add_object`
tool call with `rotation` plus either `scale` or `size`:

```json
{
  "obj_0001_example": {
    "tool_calls": [
      {
        "name": "add_object",
        "arguments": {
          "rotation": [0.0, 0.0, 0.0, 1.0],
          "size": [0.8, 1.1, 0.7]
        }
      }
    ]
  }
}
```

Generate the pose file with the SceneReVis-7B checkpoint. The command resumes
valid records in the existing JSON; `--refresh` explicitly regenerates all
selected samples, while invalid records are retried automatically.

```bash
python script/pretreatment/generate_scenerevis_poses.py \
  --data_dir /mnt/f/scenerevis/output/heatmap_data \
  --split test \
  --scene_json_dir /mnt/d/3D-Dataset/dataset-ssr3dfront/scenes \
  --model "$SCENEREVIS_MODEL" \
  --backend vllm \
  --output_json outputs/scenerevis_pose_results.json \
  --batch_size 8 \
  --max_tokens 768
```

Run release search without starting Isaac Sim:

```bash
python script/pretreatment/run_hap_place.py \
  --data_dir /mnt/f/scenerevis/output/heatmap_data \
  --split val \
  --checkpoint checkpoints/heatmap_dinov2_twoway_hd256/best_peak.pth \
  --scene_json_dir /mnt/d/3D-Dataset/dataset-ssr3dfront/scenes \
  --model_dir /mnt/d/3D-Dataset/3D-FUTURE-model \
  --scenerevis_results outputs/scenerevis_pose_results.json \
  --output_json outputs/hap_place_val.json \
  --num_samples 10 \
  --voxel_resolution 256 \
  --minimum_release_height_voxels 1 \
  --physics_backend none
```

`--minimum_release_height_voxels` adds a fixed empty layer below the target
voxel kernel. Each unique first-hit surface candidate is tested exactly once at
this height. A collision rejects the candidate immediately and advances the
search to the next-highest heatmap response; release height is never traversed.

For a one-sample geometry smoke test, `--allow_metadata_pose` explicitly uses
the removed-object metadata instead of SceneReVis rotation/scale. Results from
this debug mode must not be reported as the full HAP-Place method.

For a one-sample Isaac Sim smoke test, run the integrated backend. This starts
one Isaac process per sample and is intended for debugging rather than dataset
evaluation:

```bash
python script/pretreatment/run_hap_place.py \
  --data_dir /mnt/f/scenerevis/output/heatmap_data \
  --split test \
  --checkpoint checkpoints/heatmap_dinov2_twoway_hd256/best_peak.pth \
  --scene_json_dir /mnt/d/3D-Dataset/dataset-ssr3dfront/scenes \
  --model_dir /mnt/d/3D-Dataset/3D-FUTURE-model \
  --scenerevis_results outputs/scenerevis_pose_results.json \
  --output_json outputs/hap_place_isaac_smoke.json \
  --num_samples 1 \
  --physics_backend isaac \
  --isaac_python /path/to/isaac-sim/python.sh \
  --save_physics_usd
```

### Windows Isaac Sim producer-consumer execution

The recommended dataset-scale deployment keeps model inference in WSL and runs
one or more persistent Isaac Sim consumers on Windows. Download the Isaac Sim
6.x workstation archive, extract it to `C:\isaacsim`, and complete its one-time
setup from PowerShell:

```powershell
Set-Location C:\isaacsim
.\isaac-sim.compatibility_check.bat
.\post_install.bat
$env:OMNI_KIT_ACCEPT_EULA = "YES"
```

First run `run_hap_place.py --physics_backend none` in WSL. It writes portable
manifests whose internal geometry and result paths are relative to each
manifest. Publish all valid release poses to a queue on the shared Windows
drive:

```bash
python script/pretreatment/enqueue_isaac_jobs.py \
  --input_json /mnt/e/project/SceneReVis/outputs/hap_place_val.json \
  --shared_root /mnt/e/project/SceneReVis \
  --queue_root /mnt/e/project/SceneReVis/outputs/physics_queue \
  --max_retries 2
```

Start a persistent Windows consumer. One `SimulationApp` is created at startup
and reused for every claimed job:

```powershell
$env:OMNI_KIT_ACCEPT_EULA = "YES"
C:\isaacsim\python.bat `
  E:\project\SceneReVis\script\pretreatment\run_isaac_consumer.py `
  --shared_root E:\project\SceneReVis `
  --queue_root E:\project\SceneReVis\outputs\physics_queue `
  --worker_id isaac-gpu0 `
  --gpu 0
```

For multiple GPUs, start one command in a separate PowerShell terminal per GPU
and give every process a unique `--worker_id` and `--gpu`. Begin with one
consumer on a single GPU; multiple full Isaac applications on the same GPU can
reduce throughput through memory pressure and context contention.

The queue has `pending`, `running/<worker_id>`, `done`, and `failed` states.
Publishing and claiming use atomic same-volume renames. Consumers emit heartbeat
leases while physics is running. Inspect the queue or recover consumers that
have stopped heartbeating for ten minutes:

```bash
python script/pretreatment/manage_isaac_queue.py status \
  --queue_root /mnt/e/project/SceneReVis/outputs/physics_queue

python script/pretreatment/manage_isaac_queue.py recover \
  --queue_root /mnt/e/project/SceneReVis/outputs/physics_queue \
  --stale_seconds 600 \
  --watch_interval 30
```

An Isaac execution exception is retried up to `--max_retries`. A completed
simulation that returns `sim_ready=false` is a valid physical evaluation and
moves to `done`; it is not retried. After consumers drain the queue, merge all
available settled results into one JSON from WSL:

```bash
python script/pretreatment/manage_isaac_queue.py merge \
  --input_json /mnt/e/project/SceneReVis/outputs/hap_place_val.json \
  --output_json /mnt/e/project/SceneReVis/outputs/hap_place_val_simready.json \
  --shared_root /mnt/e/project/SceneReVis
```

Use `--refresh_terminal` when enqueueing only when intentionally rerunning jobs
already in `done` or `failed`. Existing `pending` or `running` jobs are never
duplicated.

For a local same-system batch evaluation, `run_isaac_batch.py` remains a simpler
alternative. It processes every emitted manifest in one Isaac Sim process and
merges the settled poses into a new single JSON:

```bash
export OMNI_KIT_ACCEPT_EULA=YES

python script/pretreatment/run_isaac_batch.py \
  --input_json outputs/hap_place_val.json \
  --output_json outputs/hap_place_val_simready.json \
  --isaac_python /path/to/isaac-sim/python.sh \
  --timeout 7200
```

Batch execution reruns all manifests by default. Pass `--skip_completed` only
when intentionally resuming valid `hap_place_isaac_result_v2` files.

The default camera exactly matches the current orthographic training renderer.
For an arbitrary calibrated perspective view, pass `--camera_json` with this
shape:

```json
{
  "projection": "perspective",
  "image_width": 1024,
  "image_height": 1024,
  "convention": "opencv",
  "intrinsics": [[800, 0, 512], [0, 800, 512], [0, 0, 1]],
  "camera_to_world": [
    [1, 0, 0, 0],
    [0, 1, 0, 1.5],
    [0, 0, 1, -3],
    [0, 0, 0, 1]
  ]
}
```

`run_hap_place.py` writes one crash-resilient JSON. Simulator geometry and
manifests are kept under a hidden `.<output_stem>_work` directory next to it.
Each successful record contains the score-ordered release candidate, exact
mesh first hit, packed scene/target voxel statistics, release transform, and a
`simulator_record`. With the Isaac backend, `simulator_record.pose_stage` is
`settled`; otherwise it is `release`. The 4x4 `original_to_world` transform is
the authoritative pose and avoids ambiguity from scale encoded in legacy
`jid` strings.

Both scene occupancy and the transformed target kernel are packed into
`uint64` rows. Candidate collision checks shift the target rows to the release
anchor and use bitwise intersection. Occupancy construction uses conservative
triangle-AABB voxelization: triangle bounds provide the broad phase, a 13-axis
SAT test marks every intersecting voxel, and watertight surfaces are filled.
The cubic grid extent is derived only from the room envelope with 2% padding;
target height never expands `max_y`. Within that finite grid, every voxel whose
center lies outside the room's extruded XZ polygon or below/above its
floor/ceiling interval is explicitly set to one. A target kernel extending
beyond the finite grid is also rejected as a collision.
Mesh-ray projection strictly requires `rtree`; there is no voxel-ray fallback.
A sample also fails if any scene mesh cannot be voxelized, so incomplete
occupancy is never accepted silently. Result JSON records the method as
`conservative_triangle_aabb_sat_v1`.

Run the focused WSL checks before a full evaluation:

```bash
source setup_env.sh
pip install rtree pytest
python -m pytest tests/test_hap_place.py -q
python -m pytest tests/test_isaac_settle.py -q

python script/pretreatment/generate_scenerevis_poses.py \
  --data_dir /mnt/f/scenerevis/output/heatmap_data \
  --split val \
  --scene_json_dir /mnt/d/3D-Dataset/dataset-ssr3dfront/scenes \
  --model "$SCENEREVIS_MODEL" \
  --backend vllm \
  --output_json outputs/scenerevis_pose_smoke.json \
  --num_samples 1 \
  --batch_size 1 \
  --refresh

python script/pretreatment/run_hap_place.py \
  --data_dir /mnt/f/scenerevis/output/heatmap_data \
  --split val \
  --checkpoint checkpoints/heatmap_dinov2_twoway_hd256/best_peak.pth \
  --scene_json_dir /mnt/d/3D-Dataset/dataset-ssr3dfront/scenes \
  --model_dir /mnt/d/3D-Dataset/3D-FUTURE-model \
  --scenerevis_results outputs/scenerevis_pose_smoke.json \
  --output_json outputs/hap_place_smoke.json \
  --num_samples 1 \
  --physics_backend none
```

The Isaac worker reads PhysX contact reports directly. It converts negative
contact separation into penetration depth and evaluates the maximum depth over
the settled low-velocity window. SceneReVis geometry, calibration, and result
JSON use a right-handed Y-up frame. At the simulator boundary, the worker maps
project coordinates $(x,y,z)$ to Isaac Sim Z-up coordinates $(x,-z,y)$ for the
scene, target, release pose, gravity, and contact tests. It maps the settled
transform back to project Y-up before merging results. Therefore debug
`settled.usda` files use Isaac's native Z-up convention, while final JSON poses
remain in SceneReVis Y-up. A result is simulator-ready only when all of the
following hold by default:

- linear speed below `0.01 m/s` and angular speed below `0.05 rad/s` for 60
  consecutive frames at 120 Hz;
- support must first be established by a bottom PhysX contact with
  `abs(dot(contact_normal, world_up)) >= 0.7`; after PhysX puts the rigid body
  to sleep, the support state is retained only while motion remains below the
  thresholds and bottom height stays within the contact-height tolerance;
- settled penetration at most `0.005 m`;
- tilt from the prepared SceneReVis orientation at most `15 degrees`;
- horizontal displacement from the semantic release anchor at most `0.1 m`.

The JSON also records peak transient penetration, raw contact/support frame
counts, sleep-latched support frames, final speeds, tilt, displacement, contact
force, and both prepared and original asset transforms. `--save_physics_usd`
exports a per-sample `settled.usda` for debugging.

Export any successful release or settled result as an interactive GLB:

```bash
python script/pretreatment/visualize_hap_place_3d.py \
  --result_json outputs/hap_place_smoke.json \
  --output_glb outputs/hap_place_smoke.glb
```

The scene is gray, the placed target is green, the exact first surface hit is
orange, the release anchor is blue, and the dataset GT anchor is magenta. Use
`--hide_gt` for a prediction-only visualization. The exporter automatically
uses the settled pose after an Isaac run and the release pose otherwise.
