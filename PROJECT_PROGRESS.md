# SceneReVis / HAP-Place 项目整理与进展

## 1. 项目定位

本项目的定位是：

> **HAP-Place: Fast Simulator-Ready Object Re-placement for Embodied AI**

目标是面向具身智能和规模化仿真数据生成，输入移除目标物体后的房间观察、目标物体图像、语言摆放请求、场景几何和相机标定，输出可直接进入模拟器的稳定物体位姿。

方法边界是：

- SceneReVis-7B 负责目标物体的旋转和尺度/绝对尺寸；
- HAP-Place 丢弃 SceneReVis 的粗坐标；
- 稠密热力图负责精细位置先验；
- 几何模块负责首个可见表面与无碰撞 release pose；
- Isaac Sim 负责自由落体、接触和稳定化。

整体目标：

$$
\text{语义和布局一致} + \text{初始 release 无碰撞}
+ \text{物理仿真后稳定、可支撑、低穿透}

$$

$$
= \text{Simulator-Ready placement data}
$$

## 2. 当前方法架构

### 2.1 完整流水线

```text
移除目标后的房间图 + 目标物体图 + 文本请求
        |
        +--> SceneReVis-7B --> rotation + size/scale
        |
        +--> HAP-Place heatmap model --> 256 x 256 dense heatmap
                                            |
                                            v
                              按 heatmap score 从高到低访问
                                            |
                                            v
                              calibrated camera exact first hit
                                            |
                                            v
                              conservative 3D voxel occupancy
                                            |
                                            v
                              shifted uint64 bitset overlap test
                                            |
                                            v
                              第一个可行候选，立即停止搜索
                                            |
                                            v
                              fixed-minimum-height release pose
                                            |
                                            v
                              Isaac Sim drop-and-settle
                                            |
                                            v
                              simulator-ready settled pose
```

第一阶段求解离散热力图候选集上的约束最大值：

$$
u_0 = \underset{u}{\operatorname{argmax}}\; H_{\theta}(u)
\quad \text{s.t.} \quad
M_{\mathrm{rel}}(u, R_{\mathrm{sr}}, s_{\mathrm{sr}}) = 1
$$

第二阶段执行物理投影：

$$
p_{\mathrm{final}}
= \Phi_{\mathrm{IsaacSim}}(p_{\mathrm{release}}, \mathcal{S})
$$

这里没有 top-K rerank。候选按完整热力图分数排序，找到第一个可行 release 后早停。默认 $256 \times 256$ 热力图和 $\texttt{max\_candidates}=65{,}536$ 会遍历完整离散候选集上界。

### 2.2 热力图网络

当前模型架构为：

- Room encoder：DINOv2-B/14，输出 $37 \times 37$ 空间 token，hidden dim 768；
- Object image encoder：SigLIP SO400M，输出 $27 \times 27$ token，hidden dim 1152；
- Text encoder：同一个 SigLIP text encoder；
- Fusion：3 层 two-way decoder；
- Decoder hidden dim：256；
- Attention heads：8；
- MLP ratio：4.0；
- Upsampling：两级 transpose convolution + heatmap head；
- 输出：$256 \times 256$ heatmap；
- DINOv2 和 SigLIP 均冻结，只训练投影、融合 decoder 和 heatmap head；
- checkpoint 只保存 trainable state，不重复保存冻结编码器。

### 2.3 2D 到 3D

重投影原理：

- 支持当前训练使用的正交俯视相机；
- 支持任意已标定 perspective camera；
- 世界坐标使用 XYZ，$+Y$ 为上；
- 每个 heatmap 像素转换为相机射线；
- 使用 `trimesh.ray.ray_triangle.RayMeshIntersector` 和 `rtree` 取得三角网格首交点；
- 不使用 voxel DDA 或射线回退；
- 遮挡后方不推断，当前只使用第一个可见表面。

### 2.4 保守体素化与位图 release test

当前体素化方法为：

```text
triangle AABB broad phase
        -> 13-axis triangle-box SAT
        -> 任意相交或边界接触即置 1
        -> watertight mesh 填充内部
```

场景和目标物体都按 X 方向打包为 `uint64` 行：

- 场景：packed occupancy bitset；
- 目标：按 `(y,z)` 行预打包的 target kernel；
- 碰撞：平移目标位图后，执行移位和按位与；
- 网格外区域默认视为碰撞；
- 目标底部默认保留 $\texttt{minimum\_release\_height\_voxels}=1$ 的固定空层；
- 固定空层是唯一释放高度，不再向上遍历高度。

当前 release 搜索：

```text
for pixel in argsort(heatmap, descending=True):
    hit = exact_first_visible_surface(pixel)
    skip duplicated hit voxel
    if shifted_target_bitset(hit, fixed_minimum_height) AND scene_bitset == 0:
        return first feasible release
    continue to the next-highest heatmap pixel
```

### 2.5 Isaac Sim 物理阶段

逻辑为：

- 场景：static triangle mesh collider；
- 目标：dynamic rigid body + convex decomposition collider；
- 重力方向：`-Y`；
- 初始线速度和角速度：0；
- 直接读取 PhysX contact report，用负 contact separation 计算真实接触穿透深度；
- 支撑必须是靠近物体底部、法向接近世界竖直方向的持续接触，侧壁卡住不算支撑；
- 默认在 120 Hz 下连续 60 帧满足线速度小于 `0.01 m/s`、角速度小于 `0.05 rad/s`；
- 默认要求连续 30 个低速帧存在有效底部支撑；
- 默认 SimReady 阈值为：穿透不超过 `0.005 m`、倾角不超过 `15 degrees`、水平漂移不超过 `0.1 m`；
- 输出 prepared mesh 和 original asset 的最终世界变换；
- 多样本通过 `run_isaac_batch.py` 在同一个 Isaac Sim 进程中顺序执行，避免逐样本启动模拟器。

## 3. 数据集现状

数据集实际统计：

| Split | Samples | Rooms  | Unique object models |
| ----- | -------:| ------:| --------------------:|
| Train | 47,256  | 11,391 | 7,340                |
| Val   | 5,895   | 1,428  | 2,445                |
| Test  | 5,909   | 1,422  | 2,503                |
| Total | 59,060  | 14,241 | -                    |

房间交集检查：

$$
\mathcal{R}_{\mathrm{train}} \cap \mathcal{R}_{\mathrm{val}} = \varnothing,
\qquad
\mathcal{R}_{\mathrm{train}} \cap \mathcal{R}_{\mathrm{test}} = \varnothing,
\qquad
\mathcal{R}_{\mathrm{val}} \cap \mathcal{R}_{\mathrm{test}} = \varnothing
$$

因此当前数据集满足 room-disjoint 划分。

每个样本包含：

- 移除目标物体后的房间俯视图；
- 目标物体参考图；
- 原始完整房间图；
- GT Gaussian heatmap；
- 语言摆放请求；
- `removed_object` 的 mesh identity、GT position、rotation、size；
- 由目标物体 world position 直接投影得到的 `gt_pixel_center`。

## 4. 训练进展

当前训练曲线：

| Epoch | Train Peak Acc | Val Peak Acc | Val Loss |
| -----:| --------------:| ------------:| --------:|
| 1     | 42.81%         | 52.71%       | 0.0410   |
| 4     | 63.07%         | 60.64%       | 0.0381   |
| 7     | 73.01%         | 63.41%       | 0.0387   |
| 9     | 80.61%         | 64.65%       | 0.0410   |
| 11    | 88.11%         | 64.95%       | 0.0473   |
| 12    | 91.14%         | **65.46%**   | 0.0484   |
| 13    | 94.26%         | 64.89%       | 0.0522   |

当前结论：

- 最佳日志验证 Peak@32 为 `65.46%`，出现在 epoch 12；
- `best_peak.pth` 应用于坐标精度评估；
- 最低 val loss 大约出现在 epoch 4；
- epoch 9 之后 train 继续快速上升，而 val 基本平台化，过拟合已经明显；
- 日志停在 epoch 14 训练中，不能声称 20 epochs 已完整结束；
- 论文/海报中的 `66.0%` 和 `14.6 px` 暂时视为先前评估的暂定数值，必须重新生成并保存原始 test JSON 后才能作为正式结果。

## 5. 已知技术问题

### 性能问题

1. 单样本 conservative voxelization 当前约 2.7s，是主要预处理开销；
2. release bitset search 约 5ms，不是瓶颈；
3. 当前每个样本重新构建场景 occupancy；
4. 后续应按房间预计算每个物体的 bitset，再对“移除目标后的场景”执行快速 OR 合并；
5. 应缓存 SceneBuilder mesh 和 per-object conservative voxel bitset。

### 方法边界

1. 单视角只使用 first visible hit，遮挡后方保持 unknown；
2. rotation/scale 质量受 SceneReVis 输出限制；
3. 非 watertight mesh 只能严格表示 triangle surface，无法无歧义定义实体内部；
4. conservative boundary-touch policy 会减少漏碰撞，但可能增加假碰撞和 release 高度。

# 6. Cross-platform Isaac Sim execution status

The P0 system now has a durable producer-consumer execution path:

- WSL producer: `enqueue_isaac_jobs.py` publishes release manifests.
- Shared queue: `pending -> running/<worker> -> done/failed`.
- Windows consumer: `run_isaac_consumer.py` keeps one Isaac Sim application alive.
- Lease recovery: `manage_isaac_queue.py recover` requeues stale jobs within a bounded retry budget.
- Result collection: `manage_isaac_queue.py merge` writes the final single simulator-ready JSON.
- Portable handoff: geometry, result, and debug USD paths inside each manifest are relative to the manifest directory.
- Correct failure semantics: simulation threshold failures are completed physical evaluations; only Isaac execution errors are retried.
- Queue, geometry, and physics-protocol tests: 35 focused WSL tests pass.
- The Isaac boundary maps $(x,y,z)_{\mathrm{Y-up}}$ to $(x,-z,y)_{\mathrm{Z-up}}$ and maps the settled transform back to project Y-up coordinates.
- The voxel grid is bounded only by the room envelope; the extruded room exterior is explicitly occupied and any target-kernel grid overflow is a collision.

Remaining P0 runtime validation:

1. Install Isaac Sim 6.x natively on Windows and run one real queue smoke job.
2. Confirm PhysX contact-report APIs and collider cooking with the installed Isaac build.
3. Measure persistent-consumer startup amortization, per-sample P50/P95 latency, GPU memory, and failure rates.
4. Run a small multi-consumer test only after one consumer is stable.
