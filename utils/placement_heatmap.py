"""
Learning-based placement heatmap module (pure SigLIP architecture).

Uses SigLIP for all encoding: room spatial features, object reference, and text.
All features share the same embedding space, making fusion straightforward.

Architecture:
    Room Image   → SigLIP ViT → Spatial Features (27x27) → SpatialRefinement → Key/Value
    Object Image → SigLIP ViT → Pooled Global Feature ─┐
                                                        ├→ ObjTextFusion → Query
    Text         → SigLIP Text Encoder ────────────────┘
                                                              ↓
                                                        CrossAttention → Heatmap
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from PIL import Image
import torchvision.transforms as T


# ============================================================================
# SigLIP Vision Encoder (shared for room spatial + object global features)
# ============================================================================

class SiglipVisionEncoder(nn.Module):
    """SigLIP vision encoder for both spatial and global features.

    Args:
        model_name: HuggingFace model name, e.g. "google/siglip-so400m-patch14-384"
    """

    def __init__(self, model_name: str = "google/siglip-so400m-patch14-384"):
        super().__init__()
        from transformers import SiglipVisionModel

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SiglipVisionModel.from_pretrained(model_name).to(device)

        config = self.model.config
        self.feature_dim = config.hidden_size
        self.patch_size = config.patch_size
        self.image_size = config.image_size

        # Freeze all weights
        for param in self.model.parameters():
            param.requires_grad = False

        # Preprocess: resize + normalize (SigLIP uses mean=0.5, std=0.5)
        self.preprocess = T.Compose([
            T.Resize((self.image_size, self.image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        # Infer actual grid_size from a single forward pass
        dummy = torch.zeros(1, 3, self.image_size, self.image_size, device=device)
        with torch.no_grad():
            outputs = self.model(pixel_values=dummy, output_hidden_states=True)
        hidden = outputs.last_hidden_state
        seq_len = hidden.shape[1]
        # SigLIP so400m: seq_len=729=27x27 (no separate CLS token)
        if int(round(seq_len ** 0.5)) ** 2 == seq_len:
            num_patches = seq_len
            self.drop_cls = False
        else:
            num_patches = seq_len - 1
            self.drop_cls = True
        self.grid_size = int(round(num_patches ** 0.5))
        print(f"[SiglipVisionEncoder] seq_len={seq_len}, grid={self.grid_size}x{self.grid_size}, drop_cls={self.drop_cls}")

        self.eval()

    def encode_spatial(self, image: torch.Tensor) -> torch.Tensor:
        """Extract spatial features (2D grid) from image tensor.

        Args:
            image: [B, 3, H, W] preprocessed image tensor

        Returns:
            [B, grid_size, grid_size, feature_dim] spatial feature map
        """
        outputs = self.model(pixel_values=image, output_hidden_states=True)
        hidden = outputs.last_hidden_state  # [B, seq_len, C]
        patch_tokens = hidden[:, 1:, :] if self.drop_cls else hidden

        B = patch_tokens.shape[0]
        features = patch_tokens.reshape(B, self.grid_size, self.grid_size, self.feature_dim)
        return features

    def encode_global(self, image: torch.Tensor) -> torch.Tensor:
        """Extract global (pooled) feature from image tensor.

        Args:
            image: [B, 3, H, W] preprocessed image tensor

        Returns:
            [B, feature_dim] global feature vector
        """
        outputs = self.model(pixel_values=image)
        # pooler_output is the mean-pooled representation
        pooled = outputs.pooler_output  # [B, feature_dim]
        return pooled


# ============================================================================
# SigLIP Text Encoder
# ============================================================================

class SiglipTextEncoder(nn.Module):
    """SigLIP text encoder for object descriptions.

    Args:
        model_name: HuggingFace model name, e.g. "google/siglip-so400m-patch14-384"
    """

    def __init__(self, model_name: str = "google/siglip-so400m-patch14-384"):
        super().__init__()
        from transformers import SiglipTextModel, AutoTokenizer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SiglipTextModel.from_pretrained(model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.feature_dim = self.model.config.hidden_size

        # Freeze all weights
        for param in self.model.parameters():
            param.requires_grad = False

        self.eval()

    def encode(self, text: str) -> torch.Tensor:
        """Encode text description.

        Args:
            text: object description string

        Returns:
            [1, feature_dim] text feature vector
        """
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            # pooler_output is the [CLS] token representation
            text_features = outputs.pooler_output  # [1, feature_dim]

        return text_features


# ============================================================================
# Spatial Attention Module
# ============================================================================

class SpatialRefinement(nn.Module):
    """全局自注意力细化空间特征

    在 27×27 = 729 个 token 上做全局自注意力，
    让每个空间位置都能感知整个房间的上下文。
    """

    def __init__(self, in_channels: int, hidden_dim: int = 256,
                 num_heads: int = 8):
        super().__init__()

        self.proj_in = nn.Linear(in_channels, hidden_dim)

        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )

        self.proj_out = nn.Linear(hidden_dim, in_channels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """全局自注意力

        Args:
            features: [B, H, W, C] spatial feature map

        Returns:
            [B, H, W, C] refined spatial feature map
        """
        B, H, W, C = features.shape
        x = self.proj_in(features)

        # [B, H, W, D] -> [B, H*W, D]
        x_flat = x.reshape(B, H * W, -1)

        # 全局自注意力 (729 tokens, 完全可行)
        attn_out, _ = self.self_attn(x_flat, x_flat, x_flat)

        # [B, H*W, D] -> [B, H, W, D]
        attn_out = attn_out.reshape(B, H, W, -1)

        return self.proj_out(attn_out) + features


# ============================================================================
# Cross-Attention Heatmap Head
# ============================================================================

class CrossAttentionHeatmap(nn.Module):
    """Computes a 2D heatmap via cross-attention between query and spatial features.

    The fused object+text embedding serves as the query, and each spatial
    feature vector serves as a key. The resulting attention weights form the heatmap.
    """

    def __init__(self, feature_dim: int, query_dim: int = None):
        super().__init__()
        query_dim = query_dim or feature_dim
        self.visual_proj = nn.Linear(feature_dim, query_dim)
        self.text_proj = nn.Linear(query_dim, query_dim)
        self.temperature = nn.Parameter(torch.tensor(1.0))
        self.logit_bias = nn.Parameter(torch.tensor(0.0))  # 可学习偏置，提升峰值置信度

    def forward(self, spatial_features: torch.Tensor,
                text_query: torch.Tensor) -> torch.Tensor:
        """Compute placement heatmap.

        Args:
            spatial_features: [B, H, W, C_v] from spatial refinement
            text_query: [B, C_q] fused object+text embedding

        Returns:
            [B, H, W] heatmap with values in [0, 1]
        """
        B, H, W, C_v = spatial_features.shape

        visual_keys = self.visual_proj(spatial_features)
        text_q = self.text_proj(text_query)

        visual_keys_flat = visual_keys.reshape(B, H * W, -1)

        logits = torch.einsum("bd,bnd->bn", text_q, visual_keys_flat)
        logits = logits / self.temperature.clamp(min=0.01) + self.logit_bias

        # 使用 sigmoid (每个像素独立预测 [0,1])，不用 softmax (互斥概率分布)
        # sigmoid + weighted BCE 自然驱动: 峰值区域→1.0, 背景→0.0
        # logit_bias 让模型自己学习将 logit 上移(提高峰值置信度)
        # 训练时不做 max 归一化——初始均匀输出除以 max 后全变 1.0 会导致梯度崩溃
        heatmap_flat = torch.sigmoid(logits)
        heatmap = heatmap_flat.reshape(B, H, W)

        return heatmap


# ============================================================================
# Object-Text Fusion Module (SigLIP aligned space)
# ============================================================================

class ObjTextFusion(nn.Module):
    """Fuse SigLIP object visual features with SigLIP text embedding.

    Since both are in the same SigLIP embedding space, fusion is simple:
    concatenate and project.

    Args:
        siglip_dim: SigLIP feature dimension (e.g. 1152)
        output_dim: output dimension for query vector
    """

    def __init__(self, siglip_dim: int, output_dim: int = None):
        super().__init__()
        output_dim = output_dim or siglip_dim
        self.fusion = nn.Sequential(
            nn.Linear(siglip_dim * 2, siglip_dim),
            nn.GELU(),
            nn.Linear(siglip_dim, output_dim),
        )

    def forward(self, obj_features: torch.Tensor,
                text_features: torch.Tensor) -> torch.Tensor:
        """Fuse object visual and text features.

        Args:
            obj_features: [B, siglip_dim] SigLIP object visual embedding
            text_features: [B, siglip_dim] SigLIP text embedding

        Returns:
            [B, output_dim] fused query vector
        """
        combined = torch.cat([obj_features, text_features], dim=-1)
        return self.fusion(combined)


# ============================================================================
# Full PlacementHeatmap Module (pure SigLIP)
# ============================================================================

class PlacementHeatmap(nn.Module):
    """Complete heatmap generation pipeline with pure SigLIP.

    Architecture:
        Room Image   -> SigLIP ViT (spatial) -> SpatialRefinement -> Key/Value
        Object Image -> SigLIP ViT (global) ─┐
                                             +-> ObjTextFusion -> Query
        Text         -> SigLIP Text Encoder ─┘
                                                   |
                                             CrossAttention -> Heatmap
    """

    SIGLIP_MODEL = "google/siglip-so400m-patch14-384"

    def __init__(self, siglip_model: str = None, heatmap_res: int = 256):
        super().__init__()
        self.heatmap_res = heatmap_res

        siglip_model = siglip_model or self.SIGLIP_MODEL

        # SigLIP vision encoder (shared for room spatial + object global)
        self.vision_encoder = SiglipVisionEncoder(siglip_model)
        siglip_dim = self.vision_encoder.feature_dim

        # SigLIP text encoder
        self.text_encoder = SiglipTextEncoder(siglip_model)

        # Object-Text fusion (both already in SigLIP space)
        self.obj_text_fusion = ObjTextFusion(siglip_dim, siglip_dim)

        # Spatial refinement (windowed self-attention on room features)
        self.spatial_refinement = SpatialRefinement(
            in_channels=siglip_dim,
            hidden_dim=256,
            num_heads=8,
        )

        # Cross-attention heatmap head
        self.heatmap_head = CrossAttentionHeatmap(
            feature_dim=siglip_dim,
            query_dim=siglip_dim,
        )

    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess image for SigLIP.

        Args:
            image_path: path to image PNG

        Returns:
            [1, 3, H, W] preprocessed tensor
        """
        device = next(self.vision_encoder.model.parameters()).device
        image = Image.open(image_path).convert("RGB")
        preprocessed = self.vision_encoder.preprocess(image).unsqueeze(0).to(device)
        return preprocessed

    def forward(self, room_image_path: str, object_desc: str,
                object_image_path: Optional[str] = None) -> torch.Tensor:
        """Generate placement heatmap for an object in a room.

        Args:
            room_image_path: path to the room top-view PNG (without target object)
            object_desc: text description of the object to place
            object_image_path: path to the object reference PNG

        Returns:
            [H, W] heatmap tensor with values in [0, 1], upscaled to heatmap_res
        """
        # Stage 1: SigLIP encode room top view -> spatial features
        room_tensor = self.preprocess_image(room_image_path)
        spatial_features = self.vision_encoder.encode_spatial(room_tensor)  # [1, g, g, C]

        # Stage 2: Spatial refinement (self-attention)
        spatial_features = self.spatial_refinement(spatial_features)

        # Stage 3: SigLIP encode object image + text
        if object_image_path:
            obj_tensor = self.preprocess_image(object_image_path)
            obj_features = self.vision_encoder.encode_global(obj_tensor)  # [1, C]
        else:
            # If no object image, use zeros
            device = next(self.vision_encoder.model.parameters()).device
            obj_features = torch.zeros(1, self.vision_encoder.feature_dim, device=device)

        text_features = self.text_encoder.encode(object_desc)  # [1, C]

        # Stage 4: Object-Text fusion -> query
        query = self.obj_text_fusion(obj_features, text_features)  # [1, C]

        # Stage 5: Cross-attention -> heatmap at ViT patch resolution
        heatmap_low = self.heatmap_head(spatial_features, query)  # [1, g, g]

        # Stage 6: Upsample to target heatmap resolution
        heatmap = F.interpolate(
            heatmap_low.unsqueeze(0),
            size=(self.heatmap_res, self.heatmap_res),
            mode='bilinear',
            align_corners=False,
        ).squeeze(0).squeeze(0)

        # Normalize to [0, 1]
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        return heatmap

    def forward_tensor(
        self,
        room_image: torch.Tensor,
        object_desc: str,
        object_image: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate placement heatmap from tensor inputs (for training).

        Args:
            room_image: [B, 3, H, W] preprocessed room image tensor
            object_desc: text description of the object to place
            object_image: [B, 3, H, W] preprocessed object image tensor (optional)

        Returns:
            [B, heatmap_res, heatmap_res] heatmap tensor with values in [0, 1]
        """
        # Stage 1: SigLIP encode room top view -> spatial features
        spatial_features = self.vision_encoder.encode_spatial(room_image)  # [B, g, g, C]

        # Stage 2: Spatial refinement (self-attention)
        spatial_features = self.spatial_refinement(spatial_features)

        # Stage 3: SigLIP encode object image + text
        if object_image is not None:
            obj_features = self.vision_encoder.encode_global(object_image)  # [B, C]
        else:
            # If no object image, use zeros
            device = next(self.vision_encoder.model.parameters()).device
            B = room_image.size(0)
            obj_features = torch.zeros(B, self.vision_encoder.feature_dim, device=device)

        text_features = self.text_encoder.encode(object_desc)  # [1, C]
        # Broadcast text features to match batch size
        if text_features.size(0) != room_image.size(0):
            text_features = text_features.expand(room_image.size(0), -1)

        # Stage 4: Object-Text fusion -> query
        query = self.obj_text_fusion(obj_features, text_features)  # [B, C]

        # Stage 5: Cross-attention -> heatmap at ViT patch resolution
        heatmap_low = self.heatmap_head(spatial_features, query)  # [B, g, g]

        # Stage 6: Upsample to target heatmap resolution
        heatmap = F.interpolate(
            heatmap_low.unsqueeze(1),  # [B, 1, g, g]
            size=(self.heatmap_res, self.heatmap_res),
            mode='bilinear',
            align_corners=False,
        ).squeeze(1)  # [B, H, W]

        return heatmap


# ============================================================================
# Unified inference interface (combines heatmap + mask)
# ============================================================================

class PlacementEngine:
    """High-level placement engine that combines heatmap and feasibility mask.

    Usage:
        engine = PlacementEngine(heatmap_model=model)
        position = engine.place_object(
            scene=scene_json,
            top_view_path="path/to/top_view.png",
            object_image_path="path/to/object_ref.png",
            object_desc="bed",
            size=[1.8, 0.8, 2.1],
            rotation=[0, 0, 0, 1],
            placement_plane="floor",
        )
    """

    def __init__(self, heatmap_res: int = 256,
                 heatmap_model: Optional[PlacementHeatmap] = None,
                 enable_heatmap: bool = True):
        self.heatmap_res = heatmap_res
        self.enable_heatmap = enable_heatmap
        self.heatmap_model = heatmap_model

    def compute_heatmap(self, room_image_path: str, object_desc: str,
                        object_image_path: str) -> torch.Tensor:
        """Compute the learning-based heatmap for a given object.

        Args:
            room_image_path: path to the room top-view PNG
            object_desc: text description of the object
            object_image_path: path to the object reference PNG

        Returns:
            [heatmap_res, heatmap_res] heatmap tensor with values in [0, 1].
        """
        if not self.enable_heatmap or self.heatmap_model is None:
            return torch.ones(self.heatmap_res, self.heatmap_res)

        self.heatmap_model.eval()
        with torch.no_grad():
            heatmap = self.heatmap_model(room_image_path, object_desc, object_image_path)
        return heatmap

    def place_object(self, scene: dict, top_view_path: str,
                     object_desc: str, size: list,
                     rotation: list = None,
                     placement_plane: str = "floor",
                     clearance: float = 0.5,
                     object_image_path: Optional[str] = None) -> Optional[list]:
        """Compute the optimal placement position for an object.

        Args:
            scene: scene JSON data
            top_view_path: path to room top-view PNG
            object_desc: text description of the object
            size: [width, height, depth]
            rotation: [x, y, z, w] quaternion
            placement_plane: "floor" or target object jid
            clearance: minimum clearance around existing objects
            object_image_path: path to object reference PNG (optional)

        Returns:
            [x, y, z] placement position, or None if no feasible position found.
        """
        from utils.placement_mask import compute_mask, find_best_position

        mask, ortho_scale, cx, cz = compute_mask(
            scene, self.heatmap_res, clearance, placement_plane
        )

        if not torch.any(mask):
            print(f"[placement] No feasible position for '{object_desc}'")
            return None

        if object_image_path is not None:
            heatmap = self.compute_heatmap(top_view_path, object_desc, object_image_path)
        else:
            heatmap = torch.ones(self.heatmap_res, self.heatmap_res)

        score = heatmap * mask.float()

        if not torch.any(score > 0):
            score = mask.float()

        positions = find_best_position_from_score(
            score, ortho_scale, cx, cz, self.heatmap_res,
            placement_plane, scene, top_k=1
        )

        if not positions:
            print(f"[placement] Failed to find position for '{object_desc}'")
            return None

        print(f"[placement] Optimal position for '{object_desc}': {positions[0]}")
        return positions[0]


def find_best_position_from_score(score: torch.Tensor, ortho_scale: float,
                                  cx: float, cz: float, heatmap_res: int,
                                  placement_plane: str = "floor",
                                  scene: Optional[dict] = None,
                                  top_k: int = 1,
                                  min_distance: float = 0.5) -> list:
    """Extract top-k positions from a fused score field (heatmap x mask).

    Args:
        score: 2D float tensor [heatmap_res, heatmap_res]
        ortho_scale: orthographic scale in meters
        cx, cz: room center
        heatmap_res: grid resolution
        placement_plane: "floor" or object jid
        scene: optional scene data
        top_k: number of candidates
        min_distance: minimum distance between candidates

    Returns:
        List of [x, y, z] positions.
    """
    from utils.placement_mask import grid_to_world, _extract_objects

    if not torch.any(score > 0):
        return []

    positions = []
    working_score = score.clone()

    for _ in range(top_k):
        idx = torch.argmax(working_score.flatten()).item()
        gi, gj = divmod(idx, heatmap_res)
        x, z = grid_to_world(gi, gj, cx, cz, ortho_scale, heatmap_res)

        if placement_plane == "floor":
            y = 0.0
        else:
            y = 0.0
            if scene:
                objects = _extract_objects(scene)
                for obj in objects:
                    obj_id = obj.get('jid', '') or obj.get('uid', '')
                    if obj_id == placement_plane:
                        pos = obj.get('pos', [0, 0, 0])
                        sz = obj.get('size', [1, 1, 1])
                        y = pos[1] + sz[1] / 2
                        break

        positions.append([float(x), float(y), float(z)])

        cell_size = ortho_scale / heatmap_res
        radius_cells = max(1, int(min_distance / cell_size))
        y1 = max(0, gi - radius_cells)
        y2 = min(heatmap_res, gi + radius_cells + 1)
        x1 = max(0, gj - radius_cells)
        x2 = min(heatmap_res, gj + radius_cells + 1)
        working_score[y1:y2, x1:x2] = 0

    return positions


def visualize_placement(score: torch.Tensor, top_view_path: str,
                        positions: list, object_desc: str,
                        ortho_scale: float, cx: float, cz: float,
                        heatmap_res: int, save_path: str) -> None:
    """Create a debug visualization overlaying the score field on the top view."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    import numpy as np

    top_img = Image.open(top_view_path).convert('RGB')

    score_np = score.cpu().numpy()
    score_resized = np.array(Image.fromarray(
        (score_np * 255).astype(np.uint8)
    ).resize((top_img.width, top_img.height), Image.BILINEAR)) / 255.0

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(top_img)

    cmap = plt.cm.jet
    heatmap_rgba = cmap(score_resized)
    heatmap_rgba[..., 3] = 0.4
    ax.imshow(heatmap_rgba)

    for i, pos in enumerate(positions):
        px, pz = pos[0], pos[2]
        u = ((px - cx) / ortho_scale + 0.5) * top_img.width
        v = ((pz - cz) / ortho_scale + 0.5) * top_img.height
        ax.plot(u, v, 'rX', markersize=15, linewidth=2,
                label=f'#{i+1} ({px:.2f}, {pz:.2f})' if i == 0 else f'#{i+1}')

    ax.set_title(f'Placement: "{object_desc}"\nScore = Heatmap x FeasibleMask')
    ax.legend(loc='upper right', fontsize=8)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[placement] Visualization saved to {save_path}")
