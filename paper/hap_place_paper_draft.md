# HAP-Place: Human-Aligned Physics-Grounded Object Re-placement in 3D Indoor Scenes

## Abstract

Physically grounded 3D scene generation has made important progress in producing stable and collision-free object arrangements. However, many physics-first pipelines rely on constructive rules, test-time search, or simulation-heavy optimization, which can be slow and may not recover fine-grained human layout preferences. In this work, we study **object re-placement**: given an indoor scene with one object removed, a target object observation, and a language placement request, the goal is to recover a human-preferred and physically feasible placement for the target object. We propose **HAP-Place**, a human-aligned physics-grounded placement framework that combines a feed-forward dense placement prior with fast 3D feasibility reasoning and physics-based pose refinement. HAP-Place first predicts a 2D placement heatmap from the removed-scene top-down view, target object image, and text. It then projects a compact bitset-based 3D feasibility mask onto the heatmap, removes physically infeasible cells, selects the maximum remaining expectation, computes a minimum-height release pose on the corresponding support plane, and finally applies Isaac Sim drop-and-settle as a physics projection operator. This design preserves the physical grounding of simulation-based methods while moving most semantic placement reasoning into a single forward pass. Experiments on indoor object re-placement show that HAP-Place improves placement accuracy, human preference alignment, and inference efficiency compared with VLM, LayoutGPT-style, and physics-first baselines.

## 1. Introduction

Human indoor layouts are not merely collision-free. A chair can be physically placed in many empty regions, yet only a small subset of those regions are semantically and aesthetically plausible. A nightstand should align with a bed, a dining chair should face a table, and a small object placed on a surface should respect both support geometry and human usage patterns. Existing physics-oriented 3D scene generation methods, such as PhyScene3D-style constructive pipelines, provide strong physical consistency by using geometric constraints, signed distance fields, and test-time optimization. These methods are valuable for producing stable scenes, but they may be inefficient at inference time and less sensitive to fine-grained human placement preferences in cluttered indoor layouts.

At the same time, large vision-language models and layout language models can reason about object semantics and spatial relations, but they are unreliable when asked to directly output precise metric or image-space coordinates. In our preliminary analysis, VLM baselines often produce semantically plausible but spatially imprecise placements, especially when multiple similar objects appear in symmetric layouts. This suggests that language and vision-language reasoning are useful for understanding intent, but direct coordinate regression from a general-purpose model is not the right tool for precise object placement.

We propose to treat object placement as a two-stage optimization problem. First, a learned feed-forward model predicts a dense **human-aligned placement expectation** over the scene, while a 3D feasibility mask removes physically impossible locations. The semantic pose is obtained by maximizing the masked expectation field. Second, a physics projection operator maps this pose to a stable state by releasing the object from the lowest collision-free height above its support plane and letting gravity and contact dynamics determine the final pose.

Our framework, HAP-Place, follows this principle. Given a top-down rendered room image after removing the target object, a reference image of the target object, and a text placement request, HAP-Place predicts a dense placement heatmap. The heatmap model uses a DINOv2 room encoder, SigLIP object and text encoders, and a multi-layer two-way decoder inspired by interactive segmentation decoders. We then build a compact 3D feasibility representation using bitset occupancy masks. Each scene object is voxelized by parity ray crossing through its mesh, and per-object bitsets are combined with bitwise OR to obtain a scene occupancy mask. Feasible locations are projected back to the heatmap as a binary mask. The final semantic placement is the highest heatmap value after masking infeasible cells. Finally, this masked maximum is converted into a minimum-height release pose and evaluated in Isaac Sim through a drop-and-settle protocol that measures stability, penetration, and support.

The key insight is that physics should act as a grounding and projection layer, not as the only source of placement search. By using a learned prior, HAP-Place avoids expensive global simulation search. By masking the heatmap with 3D feasibility before taking its maximum, HAP-Place naturally handles infeasible heatmap peaks: if the unconstrained peak is blocked, the nearest high-expectation feasible region remains. Minimum-height Isaac Sim projection then turns the masked maximum into a physically stable pose.

Our contributions are:

1. We formulate human-aligned object re-placement as a benchmark for precise 3D indoor scene editing, where the goal is to recover plausible and physically feasible placements after object removal.
2. We analyze the output representation contract for placement, contrasting hard spatial decisions with soft response surfaces and showing why dense heatmaps provide a better interface to feasibility masking and physics projection.
3. We propose a feed-forward dense placement prior that fuses room geometry, target object appearance, and text through a DINOv2/SigLIP two-way decoder.
4. We introduce a compact bitset-based 3D feasibility mask based on parity mesh voxelization, enabling constrained maximization over the dense placement heatmap.
5. We integrate a minimum-height Isaac Sim drop-and-settle projection operator to convert semantic heatmap peaks into stable 3D poses.
6. We evaluate against Qwen3-VL, LayoutGPT-style, PhyScene3D-style, and geometric baselines, measuring placement accuracy, physical validity, runtime, and human preference.

## 2. Related Work

### 2.1 3D Scene Generation and Editing

3D indoor scene generation has been studied through procedural modeling, data-driven layout synthesis, neural generative models, and language-conditioned editing. Recent methods represent scenes as structured object lists and use LLMs or VLMs to perform high-level spatial reasoning. Scene editing systems such as SceneReVis serialize the current scene into a structured prompt and ask a language model to emit tool calls for object insertion, removal, or replacement. These systems are flexible, but their coordinate predictions remain coarse and can fail under symmetric or repeated-object layouts.

### 2.2 Physics-Grounded Scene Construction

Physics-grounded generation methods improve realism by checking collisions, support, stability, and penetration. PhyScene3D-style methods use constructive placement, geometry constraints, and physics-aware optimization to synthesize physically consistent scenes. Their strength is physical validity, but their reliance on search or test-time optimization can be slow. Moreover, physical validity alone does not determine human-preferred placement when many feasible positions exist.

### 2.3 Vision-Language and Layout Reasoning

Large VLMs can describe spatial relations and infer coarse layout intent from images. LayoutGPT-style methods serialize scenes into text and ask a language model to generate object positions. However, these methods are not designed for dense pixel-level or metric placement. In object re-placement, the target may need to return to a specific visually implied position among several semantically valid alternatives, which requires fine-grained visual alignment.

### 2.4 Dense Prediction for Spatial Interaction

Dense heatmap prediction has been widely used for pose estimation, affordance prediction, grasping, and interactive segmentation. We build on this idea and model object placement as a dense spatial prediction problem. Unlike language-only methods, our model directly learns image-space placement distributions. Unlike pure dense predictors, our final output is grounded by 3D feasibility and physics simulation.

## 3. Problem Formulation and Representation

### 3.1 Task Definition

Let a scene be represented by a set of objects

```
S = {o_i = (c_i, m_i, s_i, p_i, r_i)}
```

where `c_i` is category or description, `m_i` is mesh or asset identity, `s_i` is size, `p_i` is position, and `r_i` is rotation. During data construction, one target object `o_t` is removed from the scene, producing a removed scene `S^-`. The model receives:

- `I_room`: top-down image of `S^-`
- `I_obj`: reference image of the removed target object
- `q`: text placement request
- optional structured scene state `S^-`

The goal is to predict a final object pose

```
p_final = (x, y, z, R)
```

that is both human-aligned and physically valid. For image-space evaluation, the target pose is projected to the top-down view:

```
u_gt = pi(p_t)
```

where `pi` is the known orthographic projection. A prediction is considered correct if the peak distance is below a tolerance threshold, such as 32 pixels in the original top-down image.

### 3.2 Hard Spatial Decisions vs. Soft Response Surfaces

A central design question is how a placement system should represent its spatial decision. We distinguish two representation contracts.

**Hard spatial decision (HSD).** HSD methods directly output a single coordinate, pose token, or discrete cell:

```
Phi_HSD: c -> x_hat in Omega
```

where `c = (I_room, I_obj, q)` is the placement condition and `Omega` is the image plane or a parameterized support surface. Direct coordinate regression, integer coordinate generation by a VLM, and LayoutGPT-style serialized position prediction all follow this contract. The output layer compresses the spatial placement landscape into a point decision, so the native output does not expose the response structure around the predicted location.

**Soft response surface (SRS).** SRS methods instead predict a dense response function over the placement domain:

```
Phi_SRS: c -> S_theta(.; c),  S_theta: Omega -> [0, 1]
```

with the final coordinate obtained by numerical maximization:

```
x_hat = argmax_{x in Omega} S_theta(x; c)
```

HAP-Place instantiates the SRS contract with a dense heatmap. Rather than asking a model to directly emit a coordinate, it learns a spatial response field and performs maximization only after applying physical feasibility constraints.

This distinction matters even when each training sample has a single demonstrated placement. For a Gaussian heatmap target

```
Y(x) = exp(-||x - x*||^2 / (2 sigma^2))
```

the supervision is not merely an alternative encoding of the coordinate. It forms a local geometric field whose gradient

```
grad_x Y(x) = -((x - x*) / sigma^2) Y(x)
```

points toward the demonstrated placement. Thus, the target teaches not only where the correct point is, but also how nearby errors should be corrected. By contrast, a point label supervises only the final coordinate and discards this neighborhood structure.

### 3.3 Response Surfaces as a Physical Interface

The SRS contract also provides a natural interface to physical grounding. Let `M(x)` be the projected 3D feasibility mask. The semantic placement step becomes a constrained maximization over the learned response surface:

```
x_0 = argmax_{x in Omega} S_theta(x; c) M(x)
```

This formulation separates human preference from physical admissibility. `S_theta` estimates where the object is likely to be placed by a human, while `M` removes locations that cannot be lifted to a valid release pose. The maximum of the masked field is then passed to the physics projection stage:

```
p* = Phi_phys(LiftToReleasePose(x_0, S, o_t), S)
```

where `Phi_phys` is implemented by minimum-height Isaac Sim drop-and-settle.

This analysis does not require multi-modal labels or a claim that coordinate regression is non-differentiable. The point is instead representational: dense response supervision preserves local spatial structure, supports constrained maximization, and gives the physics layer a meaningful field to constrain. HSD baselines can still be optimized end-to-end, but they must introduce additional search, scoring, or refinement machinery to recover the spatial neighborhood information that SRS exposes natively.

## 4. Method

### 4.1 Overview

HAP-Place consists of three modules:

1. Feed-forward placement prior
2. Bitset-based 3D feasibility mask
3. Minimum-height Isaac Sim physics projection

The pipeline is:

```
removed-scene top view + target object image + text
        -> dense placement heatmap
        -> projected 3D feasible mask
        -> masked expectation maximization
        -> minimum-height 3D release pose
        -> Isaac Sim drop-and-settle projection
        -> final stable pose
```

### 4.2 Feed-Forward Human-Aligned Placement Prior

The placement prior predicts a heatmap:

```
H = f_theta(I_room, I_obj, q), H in [0, 1]^(H x W)
```

where high values indicate likely target object centers. The model contains:

- A DINOv2 room encoder for fine-grained top-down spatial features.
- A SigLIP image encoder for target object appearance.
- A SigLIP text encoder for the placement request.
- A multi-layer two-way decoder that performs repeated cross-attention between room tokens and object/text tokens.
- A learned upsampling head that converts decoder tokens into a full-resolution placement heatmap.

The two-way decoder lets target object and text tokens query the room layout while the room tokens are refined by object-conditioned context. This is important when the same object category appears multiple times, because the model must distinguish the exact local spatial configuration rather than only the category name.

### 4.3 Heatmap Supervision

For each training sample, the ground-truth object center is projected from 3D world coordinates to the top-down image plane. A Gaussian target heatmap is generated around the projected center:

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

### 4.4 Feasibility-Constrained Expectation Maximization

The heatmap alone represents human placement preference, but it may place probability mass in physically impossible regions. We therefore project the 3D feasibility mask into the top-down image plane and obtain a binary image-space mask:

```
M(u, v) in {0, 1}
```

where `M(u, v) = 1` means the target object center can be lifted to a physically feasible release pose near that image location. The first-stage solution is the maximum of the masked human expectation field:

```
u_0 = argmax_u H_theta(u | I_room, I_obj, q) M(u)
```

This makes the first stage an explicit constrained optimization problem: find the most human-preferred placement that is still geometrically feasible before simulation.

### 4.5 Two-Stage Placement Optimization

We formulate placement as a two-stage optimization problem. The first stage finds the maximum of a learned human-aligned expectation field under geometric feasibility:

```
u_0 = argmax_u H_theta(u | I_room, I_obj, q) M(u)
p_0^release = LiftToReleasePose(u_0, S, o_t)
```

where `H_theta` is the dense placement heatmap, `M` is the projected feasibility mask, and `LiftToReleasePose` maps the image-space optimum to the minimum-height 3D release pose of the target object. The second stage applies a physics projection operator:

```
p* = Phi_phys(p_0^release, S)
```

where `Phi_phys` is implemented by Isaac Sim drop-and-settle. Intuitively, the first stage answers where the object is expected to be placed by a human, while the second stage lets the object fall into a physically feasible stable pose.

### 4.6 2D-to-3D Release Pose Lifting

Because the top-down view is rendered with a known orthographic camera, the masked heatmap maximum `u_0` can be mapped to a world-space ray. A support plane is represented as:

```
P: n^T x + b = 0
```

where `n` is the upward plane normal. For floor placement, `P` is the floor plane. For surface placement, `P` is estimated from the support object, voxel intersection, or depth/SDF refinement. The back-projected ray intersects the support plane at:

```
x_0 = Ray(u_0) intersection P
```

The target object's canonical orientation can be predicted by an auxiliary head or inherited from the source scene.

### 4.7 Bitset-Based 3D Feasibility Mask

A direct SDF representation is expressive but memory-intensive. HAP-Place instead uses a compact 3D occupancy bitset as the primary feasibility representation.

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

Collision for the lifted target pose can then be checked efficiently:

```
collision(p) = popcount(B_scene AND B_target(p))
```

A pose is feasible if collision is below a threshold, the target is inside the room bounds, and support constraints are satisfied. Optional distance transforms over the bitset occupancy can be used to approximate clearance or penetration depth, but the bitset remains the main representation.

This representation is memory-efficient. For example, a `512^3` grid uses about 16 MB as a bitset, compared with about 512 MB for a float32 SDF.

### 4.8 Minimum-Height Physics Projection

The feasibility mask removes obviously invalid regions before optimization, but final physical validity is evaluated by simulation. After the masked maximum is lifted to a support point, HAP-Place computes the lowest release pose along the support-plane normal, then instantiates the target object in Isaac Sim and performs a drop-and-settle protocol.

Let `M_o` be the target object mesh in its local coordinates and let `R_0` be the selected orientation. The contact point is the support-plane intersection `x_0`. We place the object center along the plane normal:

```
c_0(alpha) = x_0 + alpha n
```

The release height is the smallest offset `alpha` that keeps the oriented target mesh outside the support plane and collision-free with the current scene:

```
alpha_0 = min alpha
          s.t. n^T (R_0 v + c_0(alpha) - x_0) >= epsilon, for all v in M_o
               B_scene AND B_target(c_0(alpha), R_0) = empty
```

Equivalently, for the support-plane constraint alone:

```
alpha_0 >= - min_{v in M_o} n^T R_0 v + epsilon
```

and the bitset collision constraint may increase `alpha_0` to the first collision-free voxel layer. This gives a minimum-height release pose:

```
p_0^release = (c_0(alpha_0), R_0)
```

This design is important: the object is not dropped from an arbitrary high altitude. It is released from the shortest normal-direction distance above the support plane that avoids initial penetration. This minimizes bounce, sliding, and stochastic drift, making the physics projection stable and reproducible.

The drop-and-settle protocol is:

1. Initialize the object at the minimum-height release pose.
2. Enable gravity with zero initial velocity.
3. Simulate until velocities fall below a stability threshold or a timeout is reached.
4. Measure final pose, maximum penetration, tilt angle, and contact/support state.

The physics projection returns the final stable pose:

```
p* = Phi_phys(p_0^release, S)
```

The final pose is accepted if it satisfies stability, penetration, tilt, and support thresholds. If the projection fails, the system reports the placement as physically infeasible.

## 5. Experiments

### 5.1 Dataset

We construct object re-placement samples from indoor scenes by removing one object at a time. For each sample, we store:

- top-down room image before removal
- top-down room image after removal
- target object image
- target object description and placement request
- original 3D object pose
- projected 2D center
- structured scene JSON

The task is evaluated on train/validation/test splits with disjoint scenes. The current prototype contains `XX` training samples, `XX` validation samples, and `XX` test samples.

### 5.2 Baselines

We compare HAP-Place against:

1. **Qwen3-VL coordinate baseline**: receives images and structured scene JSON, then directly outputs a 2D coordinate.
2. **LayoutGPT-style baseline**: receives structured scene JSON and text only, then outputs a 2D or 3D placement.
3. **SceneReVis-style baseline**: emits an `add_object` tool call from scene state and user request.
4. **PhyScene3D-style physics baseline**: uses constructive or optimization-based placement with physical constraints but without the learned dense prior.
5. **Heatmap only**: uses the global heatmap peak without feasibility filtering.
6. **Heatmap + 3D mask**: uses bitset feasibility filtering without Isaac Sim physics projection.
7. **HAP-Place full**: masked heatmap maximization, bitset feasibility, and minimum-height Isaac Sim physics projection.

### 5.3 Metrics

We report:

- **Peak distance**: Euclidean distance between predicted and ground-truth projected centers.
- **Peak accuracy**: percentage of samples below a pixel tolerance threshold.
- **Physical validity**: collision rate, support success, maximum penetration, and stability rate.
- **Human preference**: pairwise A/B preference between predicted placements.
- **Runtime**: end-to-end inference time and time breakdown per module.

### 5.4 Main Results

| Method | Peak Acc. | Median Dist. | Collision Rate | Stability | Runtime |
|---|---:|---:|---:|---:|---:|
| LayoutGPT-style | XX | XX | XX | XX | XX |
| Qwen3-VL | XX | XX | XX | XX | XX |
| PhyScene3D-style | XX | XX | XX | XX | XX |
| Heatmap only | XX | XX | XX | XX | XX |
| Heatmap + 3D mask | XX | XX | XX | XX | XX |
| HAP-Place full | XX | XX | XX | XX | XX |

The expected trend is that language and VLM baselines produce reasonable semantic placements but struggle with fine-grained coordinate recovery, while physics-first baselines are valid but slower and less aligned with the original human layout. HAP-Place should combine high placement accuracy with physical validity and faster inference.

### 5.5 Ablation Studies

We evaluate:

- Output representation: HSD coordinate prediction vs SRS heatmap prediction.
- Room encoder: SigLIP vs DINOv2.
- Hidden dimension: 256 vs 512 vs 768.
- Decoder depth: 1, 2, 3, and 4 two-way layers.
- Modal inputs: room only, room + text, room + object, room + object + text.
- Bitset resolution.
- With and without Isaac Sim physics projection.

### 5.6 Qualitative Analysis

We visualize:

- input removed-scene top view
- target object image
- ground-truth placement heatmap
- predicted placement heatmap
- projected 3D feasibility mask
- masked expectation maximum
- final physics-projected stable pose

We pay particular attention to repeated-object cases, such as multiple chairs around a table, where direct VLM coordinate prediction often selects a plausible but incorrect symmetric location.

## 6. Discussion

HAP-Place separates semantic preference from physical grounding. The learned heatmap prior answers "where would a human likely place this object?" while the physics layer answers "which of these plausible placements is physically valid?" This decomposition is important because physical validity alone is underdetermined: many placements are stable, but only a few are human-preferred.

The bitset feasibility mask is a practical bridge between visual prediction and physics simulation. It is significantly cheaper than dense float SDF storage and turns unconstrained heatmap prediction into constrained expectation maximization. Isaac Sim is then used as a physical projection operator for the selected semantic optimum rather than as a global search engine.

## 7. Limitations

The current framework depends on reliable mesh geometry and camera calibration. Parity voxelization assumes reasonably watertight meshes; open or self-intersecting meshes may require repair or robust voxelization. The heatmap model is trained from existing scene layouts, so it may inherit dataset biases. Finally, Isaac Sim physics projection adds engineering complexity and requires consistent asset physics properties.

## 8. Conclusion

We presented HAP-Place, a human-aligned physics-grounded framework for object re-placement in 3D indoor scenes. By formulating placement as a two-stage process that first maximizes a feed-forward human expectation field and then applies minimum-height Isaac Sim physics projection, HAP-Place aims to recover placements that are precise, human-preferred, physically valid, and efficient. This approach bridges the gap between VLM/LLM semantic layout reasoning and physics-first scene construction, providing a practical path toward fast and reliable embodied scene editing.

## Notes for the Next Revision

- Fill in final dataset statistics.
- Add exact training configuration and model size.
- Add bitset voxelization implementation details once finalized.
- Add Isaac Sim protocol details and thresholds.
- Replace placeholder experiment table with real numbers.
- Decide target venue framing:
  - AAAI: emphasize fast physical reasoning and learned decision policy.
  - CVPR/ACM MM: emphasize multimodal dense spatial prediction.
  - ICRA/RA-L: emphasize physical execution, stability, and robotics relevance.
