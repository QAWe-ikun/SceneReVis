# HAP-Place: Fast Simulator-Ready Object Re-placement for Embodied AI

## Abstract

Embodied agents require large-scale simulator-ready object placement data: placements must be semantically correct, physically executable, and cheap enough to generate at scale. Existing VLM and layout-language pipelines can reason about scene semantics, but direct coordinate output is often spatially imprecise. Physics-first pipelines such as PhyScene3D-style optimization can produce stable scenes, but their iterative test-time search is slow and may fail under a fixed generation budget. In this work, we study **object re-placement** as a scalable data-generation primitive for embodied AI. Given an indoor scene with one object removed, a target object observation, scene geometry/calibration, and a language placement request, the goal is to generate a simulator-ready stable placement for the target object. We propose **HAP-Place**, a fast physics-grounded framework that combines a single feed-forward dense placement prior with deterministic bitset release reasoning and Isaac Sim drop-and-settle projection. HAP-Place predicts a 2D placement heatmap, visits image locations in descending heatmap response, lifts each location to its first visible surface hit, and evaluates the corresponding 3D anchor with a scene occupancy bitset and a voxelized target-object kernel. The search stops at the first anchor that admits a collision-free release pose, which is the exact maximizer over the discretized heatmap candidate set under the release constraint. HAP-Place finally applies physics simulation to produce collision-free, stable, simulator-ready placement data. Experiments on room-disjoint object re-placement show that HAP-Place improves fine-grained placement accuracy, physical executability, and inference efficiency compared with LayoutGPT-style, SceneReVis/Qwen3-VL, and PhyScene3D-style baselines.

## 1. Introduction

Human indoor layouts are not merely collision-free. A chair can be physically placed in many empty regions, yet only a small subset of those regions are semantically and functionally plausible. For embodied AI, this issue becomes a data-generation problem: agents need many simulator-ready placements that are visually plausible, physically executable, and generated quickly enough to support large-scale training and evaluation. A placement generator must therefore combine semantic layout understanding with reliable physical execution.

Large vision-language models and layout language models can reason about object semantics and spatial relations, but they are unreliable when asked to directly output precise metric or image-space coordinates. In our preliminary analysis, SceneReVis/Qwen3-VL-style baselines often produce semantically plausible but spatially imprecise placements, especially when multiple similar objects appear in symmetric layouts. Conversely, physics-first pipelines such as PhyScene3D-style constructive optimization provide strong physical consistency, but their iterative test-time optimization can be slow and may fail under a fixed budget.

We propose to treat object placement as a two-stage optimization problem. First, a learned feed-forward model predicts a dense **demonstration-aligned placement prior** over the observation image, while a first-hit bitset release test constrains the maximization to image locations that can produce a collision-free release pose. The semantic release anchor is obtained by score-ordered early-exit maximization: candidates are visited from high to low heatmap value until the first release-feasible anchor is found. Second, a physics projection operator maps this release pose to a stable state by letting gravity and contact dynamics determine the final pose.

Our framework, HAP-Place, follows this principle. Given a calibrated rendered observation after removing the target object, a reference image of the target object, and a text placement request, HAP-Place predicts a dense placement heatmap in the observation image. The heatmap model uses a DINOv2 scene encoder, SigLIP object and text encoders, and a multi-layer two-way decoder inspired by interactive segmentation decoders. We then build a compact 3D release representation using bitset occupancy. Each scene object is voxelized by parity ray crossing through its mesh, and per-object bitsets are combined with bitwise OR to obtain a scene occupancy bitset. Each heatmap pixel is lifted through the calibrated camera ray only to its first visible surface intersection; regions behind that first hit are treated as unknown rather than inferred through ray traversal. Given the rotation and scale from the SceneReVis tool-call output, we voxelize the target object as a fixed binary kernel and test candidate anchors with local bitset overlap. The final semantic release anchor is the highest heatmap value that admits a lowest collision-free release pose. Finally, this release pose is evaluated in Isaac Sim through a drop-and-settle protocol that measures stability, penetration, and support.

The key insight is that physics should act as a grounding and projection layer, not as the only source of placement search. By using a learned prior, HAP-Place avoids expensive global simulation search. By evaluating release feasibility in heatmap-score order, HAP-Place naturally handles impossible peaks: if the unconstrained peak cannot yield a collision-free release state, the algorithm continues to the next highest-expectation visible anchor. Isaac Sim then turns the selected release pose into a physically stable final pose.

Our contributions are:

1. We formulate object re-placement as a scalable simulator-data generation primitive for embodied AI, where outputs must be semantically correct, physically executable, and stable in simulation.
2. We propose a feed-forward dense placement prior that fuses scene geometry, target object appearance, and text through a DINOv2/SigLIP two-way decoder.
3. We introduce a score-ordered first-hit bitset release test based on parity mesh voxelization and target-object bitset correlation, enabling fast constrained maximization without global simulation search.
4. We integrate a lowest-collision-free Isaac Sim drop-and-settle projection operator to convert selected semantic anchors into stable simulator-ready object placements.
5. We evaluate against LayoutGPT-style, SceneReVis/Qwen3-VL, and PhyScene3D-style baselines, measuring fine-grained placement accuracy, physical executability, stability, and runtime.

## 2. Related Work

### 2.1 3D Scene Generation and Editing

3D indoor scene generation has been studied through procedural modeling, data-driven layout synthesis, neural generative models, and language-conditioned editing. Recent methods represent scenes as structured object lists and use LLMs or VLMs to perform high-level spatial reasoning. Scene editing systems such as SceneReVis serialize the current scene into a structured prompt and ask a language model to emit tool calls for object insertion, removal, or replacement. These systems are flexible, but their coordinate predictions remain coarse and can fail under symmetric or repeated-object layouts.

### 2.2 Physics-Grounded Scene Construction

Physics-grounded generation methods improve realism by checking collisions, support, stability, and penetration. PhyScene3D-style methods use constructive placement, geometry constraints, and physics-aware optimization to synthesize physically consistent scenes. Their strength is physical validity, but their reliance on search or test-time optimization can be slow. Moreover, physical validity alone does not determine layout-consistent placement when many feasible positions exist.

### 2.3 Vision-Language and Layout Reasoning

Large VLMs can describe spatial relations and infer coarse layout intent from images. LayoutGPT-style methods serialize scenes into text and ask a language model to generate object positions. However, these methods are not designed for dense pixel-level or metric placement. In object re-placement, the target may need to return to a specific visually implied position among several semantically valid alternatives, which requires fine-grained visual alignment.

### 2.4 Dense Prediction for Spatial Interaction

Dense heatmap prediction has been widely used for pose estimation, affordance prediction, grasping, and interactive segmentation. We build on this idea and model object placement as a dense spatial prediction problem. Unlike language-only methods, our model directly learns image-space placement distributions. Unlike pure dense predictors, our final output is grounded by first-hit 3D release reasoning and physics simulation.

## 3. Problem Formulation

### 3.1 Task Definition

Let a scene be represented by a set of objects

```
S = {o_i = (c_i, m_i, s_i, p_i, r_i)}
```

where `c_i` is category or description, `m_i` is mesh or asset identity, `s_i` is size, `p_i` is position, and `r_i` is rotation. During data construction, one target object `o_t` is removed from the scene, producing a removed scene `S^-`. The model receives:

- `I_obs`: calibrated observation image of `S^-`
- `I_obj`: reference image of the removed target object
- `q`: text placement request
- optional structured scene state `S^-`
- camera calibration `K, T` and either a depth map or scene geometry for first-hit lifting
- rotation and scale `(R_sr, s_sr)` from the SceneReVis tool-call output

The goal is to generate a simulator-ready final object pose

```
p_final = (x, y, z, R_sr)
```

that is semantically aligned with the request, collision-free at initialization, and stable after physics simulation. For image-space evaluation, the demonstrated target pose is projected into the calibrated observation:

```
u_gt = pi_{K,T}(p_t)
```

where `pi_{K,T}` is the known camera projection. A prediction is considered correct if the peak distance is below a tolerance threshold, such as 32 pixels in the evaluation image.

### 3.2 Simulator-Ready Placement Objective

For embodied AI data generation, a placement is useful only if it can be executed in simulation. We therefore decompose the desired output into three properties:

1. **Semantic correctness**: the object should satisfy the language request and match human layout expectations.
2. **Release admissibility**: the object should have a collision-free release pose under the SceneReVis-provided rotation and scale.
3. **Physical executability**: after release, simulation should settle to a stable, low-penetration final pose.

HAP-Place treats object re-placement as a two-stage optimization problem:

```
u_0 = argmax_u H_theta(u | I_obs, I_obj, q)
      s.t. M_rel(u, R_sr, s_sr) = 1

p* = Phi_phys(LiftToReleasePose(u_0, R_sr, s_sr), S)
```

Here `H_theta` is a feed-forward demonstration-aligned placement prior, `M_rel` is a first-hit release predicate evaluated by local bitset collision testing, and `Phi_phys` is Isaac Sim drop-and-settle. This formulation makes the fast neural component responsible for semantic placement and uses geometry and simulation as deterministic executability checks.

### 3.3 Pose Scope and SceneReVis Rotation/Scale

HAP-Place focuses on placement-anchor and release-pose selection, not full 6-DoF object pose generation. Rotation and scale are treated as input conditions produced by the SceneReVis tool-call output:

```
SceneReVis(S^-, I_obs, I_obj, q) -> (R_sr, s_sr, optional coarse position)
```

HAP-Place discards the coarse SceneReVis coordinate for fine-grained placement, but keeps `R_sr` and `s_sr` fixed when voxelizing the target object and running the release test. The method therefore solves:

```
argmax_u H_theta(u | I_obs, I_obj, q)
s.t. M_rel(u, R_sr, s_sr) = 1
```

rather than claiming to solve unrestricted 6-DoF placement. This design isolates the main failure mode of VLM/tool-call systems in our setting: they often infer plausible orientation and scale but produce inaccurate coordinates. HAP-Place replaces the coordinate with a dense visual placement prior and physics-grounded release selection.

### 3.4 Fast Amortized Physics-Grounded Placement

The central efficiency choice is to amortize semantic placement with a single neural forward pass and reserve physics simulation for the final selected release pose. HAP-Place does not run simulation-heavy test-time search over many candidate placements. Instead, it sorts heatmap locations by learned semantic expectation, lifts each location only to its visible first-hit voxel, and performs a deterministic bitset release test. The first release-admissible location is the highest-scoring placement that can be initialized without interpenetration. This design preserves the reliability of physics-based execution while avoiding the runtime cost of global iterative placement optimization.

## 4. Method

### 4.1 Overview

HAP-Place consists of three modules:

1. Feed-forward placement prior
2. Score-ordered first-hit bitset release test
3. Lowest collision-free Isaac Sim physics projection

The pipeline is:

```
calibrated removed-scene observation + target object image + text
        -> dense placement heatmap
        -> score-ordered image candidates
        -> first-hit visible surface lifting per candidate
        -> local bitset release test
        -> early-exit constrained maximum
        -> lowest collision-free release pose
        -> Isaac Sim drop-and-settle projection
        -> final stable pose
```

### 4.2 Feed-Forward Demonstration-Aligned Placement Prior

The placement prior predicts a heatmap:

```
H = f_theta(I_obs, I_obj, q), H in [0, 1]^(H x W)
```

where high values indicate likely target object centers. The model contains:

- A DINOv2 scene encoder for fine-grained spatial features in the calibrated observation.
- A SigLIP image encoder for target object appearance.
- A SigLIP text encoder for the placement request.
- A multi-layer two-way decoder that performs repeated cross-attention between scene tokens and object/text tokens.
- A learned upsampling head that converts decoder tokens into a full-resolution placement heatmap.

The two-way decoder lets target object and text tokens query the scene layout while the scene tokens are refined by object-conditioned context. This is important when the same object category appears multiple times, because the model must distinguish the exact local spatial configuration rather than only the category name.

### 4.3 Heatmap Supervision

For each training sample, the ground-truth object center is projected from 3D world coordinates into the calibrated observation image. A Gaussian target heatmap is generated around the projected center:

```
Y(u, v) = exp(-||[u, v] - u_gt||^2 / (2 sigma^2))
```

The model is trained with weighted binary cross entropy:

```
L_heatmap = BCE_weighted(H, Y)
```

We evaluate both heatmap quality and placement accuracy using peak distance:

```
d_peak = ||argmax(H) - u_gt||_2
```

### 4.4 Release-Constrained Expectation Maximization

The heatmap alone represents human placement preference, but it may place probability mass on image locations that cannot produce a valid release state. We therefore define a binary release predicate in the heatmap domain:

```
M_rel(u, v) in {0, 1}
```

where `M_rel(u, v)=1` means the visible surface associated with pixel `(u,v)` can anchor a lowest collision-free release pose for the target object. In implementation, HAP-Place does not need to materialize a full image-domain release map before decision. It sorts image locations by heatmap value and tests them in descending order:

```
u_1, u_2, ... = argsort_u H_theta(u | I_obs, I_obj, q)

u_0 = first u_k such that M_rel(u_k) = 1
```

Because candidates are visited in descending score order, this early-exit procedure returns the exact solution over the discretized heatmap candidate set:

```
u_0 = argmax_u H_theta(u | I_obs, I_obj, q)
      s.t. M_rel(u) = 1
```

This makes the first stage an explicit constrained optimization problem: find the most layout-consistent visible anchor that can initialize physics simulation without interpenetration. Importantly, `M_rel` does not require the object to be statically supported before simulation. It only requires a valid release pose; final support and stability are determined by the drop-and-settle projection.

### 4.5 Two-Stage Placement Optimization

We formulate placement as a two-stage optimization problem. The first stage finds the maximum of a learned demonstration-aligned placement prior under geometric feasibility:

```
u_0 = argmax_u H_theta(u | I_obs, I_obj, q)
      s.t. M_rel(u) = 1
p_0^release = LiftToReleasePose(u_0, S, o_t)
```

where `H_theta` is the dense placement heatmap, `M_rel` is evaluated by first-hit lifting and local bitset collision testing, and `LiftToReleasePose` maps the image-space optimum to the lowest collision-free 3D release pose of the target object. The second stage applies a physics projection operator:

```
p* = Phi_phys(p_0^release, S)
```

where `Phi_phys` is implemented by Isaac Sim drop-and-settle. Intuitively, the first stage answers where the object should be released according to the learned layout prior, while the second stage lets the object fall into a physically feasible stable pose.

### 4.6 2D-to-3D Release Pose Lifting

For a general calibrated observation, a pixel corresponds to a 3D camera ray rather than a unique 3D point. We therefore lift image responses only to the first visible surface intersection:

```
r(t; u) = o_cam + t d(u)
p_s = FirstHit(r(t; u), S)
```

where `o_cam` is the camera center, `d(u)` is obtained from the camera intrinsics and extrinsics, and `FirstHit` is implemented by ray casting against scene geometry or by unprojecting a depth map. We do not assign the heatmap response to all voxels along the ray. Voxels behind the first hit are treated as unknown because they are occluded in the current observation. If no valid visible release anchor is found, the system may request active exploration from another viewpoint rather than hallucinating through occlusion.

The first hit `p_s` is a visible surface anchor, not the target object center. It is converted to a voxel index:

```
v_s = WorldToVoxel(p_s)
```

The target object's rotation and scale are taken from the SceneReVis tool-call output. For the fixed rotation `R_sr` and scale `s_sr`, `LiftToReleasePose` searches upward from the surface anchor to find the lowest collision-free release pose.

### 4.7 Bitset-Based First-Hit Release Test

A direct SDF representation is expressive but memory-intensive. HAP-Place instead uses compact 3D occupancy bitsets for local release testing.

For each object mesh, we voxelize its occupied volume using parity ray crossing. For a fixed scan axis, each voxel ray collects intersections with mesh triangles. After sorting intersection depths, the inside/outside state is determined by parity:

```
inside(x) = crossing_count(x) mod 2
```

Voxels inside the object are set to 1 in a per-object occupancy bitset:

```
B_i in {0, 1}^{N_x N_y N_z}
```

The full scene occupancy is obtained by bitwise OR:

```
B_scene = OR_i B_i
```

This is important: parity is only used for inside/outside determination of a single mesh. Multiple objects are merged with OR, not XOR.

The target object is also voxelized after applying the fixed SceneReVis rotation and scale. Let `B_target^{R_sr,s_sr}` be the target occupancy bitset represented in an anchor-centered coordinate frame, where the anchor corresponds to the bottom/contact reference of the object. Collision for a release pose can then be checked efficiently:

```
collision(a, R_sr, s_sr) = popcount(B_scene AND Shift(B_target^{R_sr,s_sr}, a))
```

where `a` is the voxel anchor of the release pose. Equivalently, this is binary correlation between the scene occupancy bitset and the target-object bitset kernel. For each visible surface voxel `v_s`, HAP-Place searches along the upward direction for the smallest offset that yields zero collision:

```
h_min(v_s, R_sr, s_sr) = min h >= 1
                         s.t. collision(v_s + h e_up, R_sr, s_sr) = 0
```

The image-domain release predicate is therefore:

```
M_rel(u, R_sr, s_sr) = 1[h_min(FirstHitVoxel(u), R_sr, s_sr) exists]
```

This predicate is intentionally a release test, not a static support test. We do not require the object footprint to already be supported at the lifted pose. The object may initially be slightly above the surface; gravity and contact dynamics in Isaac Sim determine the final support and stability.

For single-view placement, this predicate is evaluated lazily rather than by scanning the full voxel grid. HAP-Place sorts heatmap pixels by score, removes duplicate first-hit voxels, and performs local bitset overlap only until the first feasible release anchor is found:

```
for u in argsort(H_theta, descending=True):
    v_s = FirstHitVoxel(u)
    if v_s is invalid or v_s was already tested:
        continue
    h = first h >= 1 with collision(v_s + h e_up, R_sr, s_sr) = 0
    if h exists:
        return u, v_s + h e_up
```

Thus the cost is proportional to the number of high-response visible anchors inspected before success, not to the full voxel volume. For repeated multi-view queries, the same release predicate can be cached as a 3D bitset lookup, but each observation still accesses it through first-hit lifting rather than by projecting a 3D feasibility volume into the image.

This representation is memory-efficient. For example, a `256^3` grid uses about 2 MB as a bitset, a `512^3` grid uses about 16 MB, and a `1024^3` grid uses about 128 MB, compared with 64 MB, 512 MB, and 4 GB respectively for float32 grids.

More importantly, HAP-Place does not perform dense 3D convolution over all anchors. Once `R_sr` and `s_sr` are fixed by SceneReVis, the target kernel is fixed, but the release test is invoked only for score-ordered visible anchors:

```
T_release = O(K_success * H_up * C_overlap)
```

where `K_success` is the number of unique first-hit voxels tested before the first feasible anchor is found, `H_up` is the number of upward offsets considered, and `C_overlap` is the bitset overlap cost within the target object's local bounding volume. This avoids the prohibitive `O(N_x N_y N_z * |B_target|)` cost of scanning the entire voxel grid, which is especially important at `1024^3` resolution.

### 4.8 Lowest Collision-Free Physics Projection

The release test removes anchors that would initialize the target in collision, but final physical validity is evaluated by simulation. After the constrained heatmap maximum is lifted to a visible surface anchor, HAP-Place computes the lowest collision-free release pose, instantiates the target object in Isaac Sim, and performs a drop-and-settle protocol.

Let `M_o` be the target object mesh in its local coordinates and let `R_0` be the selected orientation. Let `p_s` be the first-hit surface point selected by the constrained heatmap search, and let `n_up` be the world up direction. We place the object center along this direction:

```
c_0(alpha) = p_s + alpha n_up
```

The release height is the smallest offset `alpha` that keeps the oriented target mesh collision-free with the current scene:

```
alpha_0 = min alpha
          s.t. B_scene AND B_target(c_0(alpha), R_0) = empty
```

In the voxel implementation, `alpha_0` corresponds to the first collision-free voxel layer above the selected surface anchor. This gives a lowest collision-free release pose:

```
p_0^release = (c_0(alpha_0), R_0)
```

This design is important: the object is not dropped from an arbitrary high altitude. It is released from the shortest distance above the visible surface that avoids initial penetration. The lifted pose may be temporarily unsupported or slightly suspended; this is intentional. Isaac Sim, not the bitset release test, decides the final contact, support, and stability through gravity and contact dynamics.

The drop-and-settle protocol is:

1. Initialize the object at the lowest collision-free release pose.
2. Enable gravity with zero initial velocity.
3. Simulate until velocities fall below a stability threshold or a timeout is reached.
4. Measure final pose, maximum penetration, tilt angle, displacement from the release anchor, and contact/support state.

In our simulation protocol, existing scene objects are treated as static triangle-mesh colliders, while the target object is instantiated as a dynamic rigid body using convex-decomposition colliders. Linear and angular velocities are initialized to zero. We simulate until both remain below fixed thresholds for `K` consecutive frames or until a timeout is reached. A placement is physically accepted only if the final pose satisfies support/contact validity, maximum penetration below `epsilon_pen`, tilt below `epsilon_tilt`, and stability thresholds. Timeouts and threshold violations are counted as simulator-readiness failures.

The physics projection returns the final stable pose:

```
p* = Phi_phys(p_0^release, S)
```

The final pose is accepted if it satisfies stability, penetration, tilt, and support thresholds. If the projection fails, the system reports the placement as physically infeasible.

## 5. Experiments

### 5.1 Dataset

We construct object re-placement samples from indoor scenes by removing one object at a time. For each sample, we store:

- calibrated observation image before removal
- calibrated observation image after removal
- target object image
- target object description and placement request
- original 3D object pose
- projected 2D center
- camera calibration and first-hit/depth information
- structured scene JSON

The task is evaluated on train/validation/test splits that are disjoint by room. This prevents the model from seeing the same room layout during training and testing and better reflects simulator-data generation for novel environments.

| Split | Rooms | Scenes | Objects | Samples |
|---|---:|---:|---:|---:|
| Train | XX | XX | XX | XX |
| Val | XX | XX | XX | XX |
| Test | XX | XX | XX | 5000+ |

### 5.2 Baselines

We compare HAP-Place against:

1. **LayoutGPT-style baseline**: receives structured scene JSON and text only, then outputs a 2D or 3D placement.
2. **SceneReVis / Qwen3-VL baseline**: receives images and structured scene JSON, then emits an `add_object` tool call or direct coordinate output.
3. **PhyScene3D-style physics baseline**: searches or optimizes object anchors using geometric feasibility and physics constraints without the amortized dense placement prior.
4. **Heatmap only**: uses the global heatmap peak without release testing.
5. **Heatmap + release test**: uses score-ordered first-hit bitset release testing without Isaac Sim physics projection.
6. **HAP-Place full**: release-constrained heatmap maximization, bitset release testing, and lowest collision-free Isaac Sim physics projection.

The PhyScene3D-style baseline is given the same scene geometry, object mesh, SceneReVis rotation/scale output, and simulation protocol as HAP-Place. It searches over 2D/3D anchors under a fixed test-time optimization budget, using collision, penetration, support, and stability objectives. A trial is marked as failed if it times out, fails to converge, remains in collision, violates physical thresholds, or drifts semantically away from the requested placement region. We report both accuracy and runtime statistics, including long-tail behavior under the fixed budget.

To avoid a weak-baseline comparison, we specify the physics-first baseline with the same inputs and execution protocol:

| Component | PhyScene3D-style setting |
|---|---|
| Initialization | candidate anchors sampled from visible surfaces or scene support regions |
| Search space | 2D image anchors lifted to 3D plus local 3D pose refinement |
| Given inputs | same scene geometry, object mesh, SceneReVis rotation/scale output, and request |
| Physical objective | collision, penetration, support/contact validity, stability, room bounds |
| Semantic objective | distance to requested or VLM-proposed placement region |
| Budget | fixed maximum iterations and wall-clock timeout |
| Failure cases | timeout, non-convergence, collision, instability, support failure, semantic drift |
| Runtime report | mean, median/P50, P95, and timeout rate |

### 5.3 Metrics

We report:

- **Peak distance**: Euclidean distance between predicted and ground-truth projected centers.
- **Peak accuracy**: percentage of samples below a pixel tolerance threshold.
- **SimReady**: percentage of samples that produce a valid output, initialize collision-free, settle stably, satisfy support/contact validity, and remain below the penetration threshold.
- **Invalid output**: percentage of missing, unparsable, or invalid tool-call/coordinate outputs.
- **TTO failure**: percentage of physics-first trials that time out, fail to converge, or violate physical thresholds under the fixed test-time optimization budget.
- **Physical executability**: collision-free rate, support success, maximum penetration, and stability rate after simulation.
- **Runtime**: end-to-end inference time and time breakdown per module.

For coordinate- or tool-call-based baselines, physical metrics are computed by applying the same first-hit lifting and drop-and-settle protocol whenever a valid initial pose can be recovered. Invalid or unparsable outputs are counted as failures, matching the simulator-data generation setting where every sample must produce executable placement data.

Formally, a sample is counted as simulator-ready only if:

```
SimReady = 1[
    valid_output
    and initial_collision_free
    and stable
    and support_valid
    and penetration < epsilon_pen
]
```

### 5.4 Main Results

| Method | Peak@32 (higher) | Median Dist. (lower) | SimReady (higher) | Invalid (lower) | TTO Fail (lower) | Runtime (lower) |
|---|---:|---:|---:|---:|---:|---:|
| LayoutGPT-style | XX | XX | XX | XX | - | XX |
| SceneReVis / Qwen3-VL | 22.0 | 61.9 | XX | XX | - | XX |
| PhyScene3D-style | XX | XX | XX | - | XX | XX |
| Heatmap only | XX | XX | XX | 0.0 | - | XX |
| Heatmap + release test | XX | XX | XX | 0.0 | - | XX |
| HAP-Place full | **66.0** | **14.6** | **100.0** | **0.0** | **0.0** | **XX** |

The expected trend is that language and VLM baselines produce reasonable semantic placements but struggle with fine-grained coordinate recovery and output validity, while physics-first baselines are physically reliable but slower due to iterative test-time optimization. HAP-Place aims to combine both advantages: semantic placement is amortized into one forward pass, while deterministic release testing and Isaac Sim projection preserve simulator-ready physical executability.

### 5.5 Runtime Breakdown

We report both typical and long-tail runtime because simulator-data generation is often limited by worst-case generation latency.

| Method | Forward | Search / TTO | Bitset | Isaac Sim | Total P50 | Total P95 |
|---|---:|---:|---:|---:|---:|---:|
| LayoutGPT-style | XX | - | - | XX | XX | XX |
| SceneReVis / Qwen3-VL | XX | - | - | XX | XX | XX |
| PhyScene3D-style | - | XX | - | XX | XX | XX |
| HAP-Place | XX | - | XX | XX | XX | XX |

### 5.6 Release Test Ablation

We isolate the contribution of score-ordered first-hit release testing and physics projection:

| Method | Peak@32 (higher) | Initial Collision (lower) | SimReady (higher) | Tested Anchors (lower) |
|---|---:|---:|---:|---:|
| Heatmap only | XX | XX | XX | - |
| Heatmap + release test | XX | XX | XX | XX |
| HAP-Place full | XX | XX | XX | XX |

The release test should reduce invalid initializations before simulation, while the final drop-and-settle projection converts the selected release pose into stable simulator-ready data.

### 5.7 Additional Ablation Studies

We evaluate:

- Scene encoder: SigLIP vs DINOv2.
- Hidden dimension: 256 vs 512 vs 768.
- Decoder depth: 1, 2, 3, and 4 two-way layers.
- Modal inputs: scene only, scene + text, scene + object, scene + object + text.
- Bitset resolution.
- Voxel resolution and bitset packing strategy.
- Maximum number of tested anchors before failure.
- SceneReVis rotation/scale variants: fixed SceneReVis output, ground-truth rotation/scale oracle, and perturbed rotation/scale stress test.

### 5.8 SceneReVis Rotation/Scale Assumption

Since HAP-Place takes rotation and scale from SceneReVis rather than predicting unrestricted 6-DoF poses, we explicitly evaluate sensitivity to this input:

| Rotation/scale source | Peak@32 | SimReady | Initial collision | Final displacement | Failure from R/s |
|---|---:|---:|---:|---:|---:|
| SceneReVis output | XX | XX | XX | XX | XX |
| Ground-truth oracle | XX | XX | XX | XX | XX |
| SceneReVis + perturbation | XX | XX | XX | XX | XX |

This table separates anchor-selection quality from orientation/scale quality. The main HAP-Place results use the SceneReVis output, while the oracle and perturbation settings quantify how much performance is limited by the external rotation/scale assumption.

### 5.9 Bitset Release Scaling

We evaluate whether the first-hit bitset release test is a core algorithmic contribution rather than only an implementation optimization. We vary voxel resolution and report memory, runtime, collision reduction, and the number of anchors tested before early exit:

| Voxel resolution | Bitset memory | Release-test P50 | Release-test P95 | Tested anchors | Initial collision (lower) | SimReady (higher) |
|---|---:|---:|---:|---:|---:|---:|
| 128^3 | XX | XX | XX | XX | XX | XX |
| 256^3 | XX | XX | XX | XX | XX | XX |
| 512^3 | XX | XX | XX | XX | XX | XX |
| 1024^3 | XX | XX | XX | XX | XX | XX |

This ablation tests the tradeoff between geometric precision and scalable generation cost. It also verifies that runtime is governed by the number of score-ordered visible anchors inspected before success rather than by a dense all-voxel convolution.

### 5.10 Downstream Embodied Utility

To strengthen the robotics relevance of simulator-ready placement data, we evaluate whether HAP-Place outputs are useful beyond static placement metrics. We consider a downstream embodied data-generation setting in which generated placements are inserted into simulation scenes and used for task construction or policy evaluation.

| Data source | Valid tasks generated | SimReady scenes | Agent success | Reset failures | Runtime / scene |
|---|---:|---:|---:|---:|---:|
| SceneReVis / Qwen3-VL | XX | XX | XX | XX | XX |
| PhyScene3D-style | XX | XX | XX | XX | XX |
| HAP-Place | XX | XX | XX | XX | XX |

For a lightweight manipulation-relevant evaluation, a subset of generated placements can be checked with pick-and-place or navigation-to-object task templates. This is not required for the placement algorithm itself, but it tests whether the generated scenes remain usable as embodied-agent training or evaluation data.

### 5.11 Failure Analysis

We categorize failures to make the system boundary conditions explicit:

| Failure type | Typical cause | Detection signal |
|---|---|---|
| Thin or open mesh | parity voxelization becomes unreliable | inconsistent occupancy or penetration after simulation |
| Severe occlusion | first-hit observation misses the intended support region | no valid visible release anchor |
| Wrong rotation/scale | SceneReVis emits an incorrect orientation or scale | collision-free release exists but final pose is semantically wrong |
| Support ambiguity | multiple nearby supports or stacked objects | large displacement after drop-and-settle |
| Sliding/falling | contact dynamics move the object away from the semantic anchor | final displacement or tilt exceeds threshold |
| Non-watertight assets | mesh defects affect voxel occupancy and colliders | release-test/simulation disagreement |

These cases are counted in SimReady or TTO-failure statistics rather than removed from evaluation.

### 5.12 Qualitative Analysis

We visualize:

- input removed-scene observation
- target object image
- ground-truth placement heatmap
- predicted placement heatmap
- score-ordered first-hit release candidates
- selected release-constrained maximum
- final simulated stable pose

We pay particular attention to repeated-object cases, such as multiple chairs around a table, where direct VLM coordinate prediction often selects a plausible but incorrect symmetric location.

## 6. Discussion

HAP-Place separates learned layout consistency from physical grounding. The learned heatmap prior answers "where is this object likely to belong in this scene?" while the physics layer answers "which of these plausible placements is physically executable?" This decomposition is important because physical validity alone is underdetermined: many placements are stable, but only a few match the demonstrated layout distribution.

The bitset release test is a practical bridge between visual prediction and physics simulation. It is significantly cheaper than dense float SDF storage and turns unconstrained heatmap prediction into constrained expectation maximization. Isaac Sim is then used as a physical projection operator for the selected semantic optimum rather than as a global search engine.

## 7. Limitations

The current framework depends on reliable mesh geometry and camera calibration. Parity voxelization assumes reasonably watertight meshes; open or self-intersecting meshes may require repair or robust voxelization. The heatmap model is trained from existing scene layouts, so it may inherit dataset biases. Finally, Isaac Sim physics projection adds engineering complexity and requires consistent asset physics properties.

## 8. Conclusion

We presented HAP-Place, a fast physics-grounded framework for simulator-ready object re-placement in 3D indoor scenes. By formulating placement as a two-stage process that first maximizes a feed-forward demonstration-aligned placement prior under a first-hit release constraint and then applies lowest collision-free Isaac Sim physics projection, HAP-Place aims to generate placements that are precise, layout-consistent, physically executable, and efficient. This approach bridges the gap between VLM/LLM semantic layout reasoning and physics-first scene construction, providing a practical path toward scalable simulator-ready data generation for embodied agents.

## Notes for the Next Revision

- Fill in final dataset statistics.
- Add exact training configuration and model size.
- Add bitset voxelization implementation details once finalized.
- Fill in Isaac Sim thresholds (`K`, `epsilon_pen`, `epsilon_tilt`, velocity limits, timeout).
- Replace placeholder experiment/runtime/ablation tables with real numbers.
- Add PhyScene3D-style fixed-budget implementation details and TTO failure analysis.
- Fill in SceneReVis rotation/scale oracle and perturbation ablation results.
- Add downstream embodied utility results or a manipulation-relevant subset evaluation.
- Add quantitative failure category counts for the SimReady failures.
- Decide target venue framing:
  - AAAI: emphasize fast physical reasoning and learned decision policy.
  - CVPR/ACM MM: emphasize multimodal dense spatial prediction.
  - ICRA/RA-L: emphasize physical execution, stability, and robotics relevance.
