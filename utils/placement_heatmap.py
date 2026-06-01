"""
Learning-based placement heatmap module (pure SigLIP architecture).

Uses SigLIP for all encoding: room spatial features, object reference, and text.
All features share the same embedding space, making fusion straightforward.

Architecture:
    Room Image   → SigLIP ViT → Spatial Features (27×27=729 tokens) → SpatialRefinement → Query
    Object Image → SigLIP ViT → Patch Features (729 tokens) ─┐
                                                             ├→ 拼接 → 全局自注意力 → Key sequence
    Text         → SigLIP Text Encoder → Token Features ────┘
                                                                     ↓
                                                     CrossAttention → Heatmap

    所有编码保留原始序列维度，不压缩为单向量。
    物体 729 tokens + 文本 ~64 tokens 拼接为整体序列，全局自注意力后作为 key。
    房间 729 个空间 query 交叉注意力到 ~793 个 key，输出热力图。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from PIL import Image
import torchvision.transforms as T


# ============================================================================
# SigLIP Vision Encoder (shared for room spatial + object patch features)
# ============================================================================

class SiglipVisionEncoder(nn.Module):
    """SigLIP vision encoder for both room spatial and object patch features.

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
        self.num_patches = num_patches
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

    def encode_patches(self, image: torch.Tensor) -> torch.Tensor:
        """Extract patch token sequence from image tensor (不压缩).

        Args:
            image: [B, 3, H, W] preprocessed image tensor

        Returns:
            [B, num_patches, feature_dim] patch token sequence
        """
        outputs = self.model(pixel_values=image, output_hidden_states=True)
        hidden = outputs.last_hidden_state  # [B, seq_len, C]
        patch_tokens = hidden[:, 1:, :] if self.drop_cls else hidden
        return patch_tokens  # [B, 729, C]


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

    def encode(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode text description, 保留完整 token 序列.

        Args:
            text: object description string

        Returns:
            tuple:
                - text_features: [1, T, feature_dim] token-level features
                - attention_mask: [1, T] bool mask (True = valid token)
        """
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt",
            return_attention_mask=True,
        ).to(device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            text_features = outputs.last_hidden_state  # [1, T, feature_dim]

        # Build attention mask: True = valid token
        if "attention_mask" in inputs:
            attention_mask = inputs["attention_mask"].bool()  # [1, T]
        else:
            # Fallback: non-padding tokens are valid (SigLIP pad_token_id=1)
            pad_id = self.tokenizer.pad_token_id or 1
            attention_mask = inputs["input_ids"] != pad_id  # [1, T]
        return text_features, attention_mask


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

        # 全局自注意力 (729 tokens)
        attn_out, _ = self.self_attn(x_flat, x_flat, x_flat)

        # [B, H*W, D] -> [B, H, W, D]
        attn_out = attn_out.reshape(B, H, W, -1)

        return self.proj_out(attn_out) + features


# ============================================================================
# Cross-Attention Heatmap Head
# ============================================================================

class CrossAttentionHeatmap(nn.Module):
    """多头交叉注意力热力图头

    Room spatial features (729 tokens) 为 Query,
    物体+文本序列 (~793 tokens) 为 Key/Value.
    每个空间位置交叉注意力后得到一个标量分数，形成热力图。

    Args:
        feature_dim: spatial feature dim (siglip_dim, 如 1152)
        kv_dim: KV sequence dim (ObjTextFusion 输出，如 256)
        hidden_dim: attention hidden dim
        num_heads: attention heads
    """

    def __init__(self, feature_dim: int, kv_dim: int,
                 hidden_dim: int = 256, num_heads: int = 8):
        super().__init__()

        # Query 从 siglip_dim 投影到 hidden_dim
        self.query_proj = nn.Linear(feature_dim, hidden_dim)
        # KV 已经在 ObjTextFusion 投影到 kv_dim，只需投影到 hidden_dim
        self.kv_proj = nn.Linear(kv_dim, hidden_dim) if kv_dim != hidden_dim else nn.Identity()

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        # 将注意力输出投影为标量分数
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
        )

        self.logit_bias = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        spatial_features: torch.Tensor,
        kv_features: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """多头交叉注意力 → 热力图

        Args:
            spatial_features: [B, H, W, C] room spatial features (Query)
            kv_features: [B, N, C] object+text sequence (Key/Value)
            key_padding_mask: [B, N] bool, True = 需要屏蔽的 padding token

        Returns:
            [B, H, W] heatmap with values in [0, 1]
        """
        B, H, W, C = spatial_features.shape

        # Query: 空间特征 → [B, H*W, D]
        spatial_q = self.query_proj(spatial_features)
        spatial_q = spatial_q.reshape(B, H * W, -1)

        # Key/Value: 物体+文本序列 → [B, N, D]
        kv = self.kv_proj(kv_features)

        # 交叉注意力: Q=spatial, K=V=kv
        attn_out, _ = self.cross_attn(
            query=spatial_q,
            key=kv,
            value=kv,
            key_padding_mask=key_padding_mask,
        )  # [B, H*W, D]

        # 投影为标量分数
        logits = self.score_head(attn_out).squeeze(-1)  # [B, H*W]
        logits = logits + self.logit_bias

        # sigmoid → [0, 1]
        heatmap_flat = torch.sigmoid(logits)
        heatmap = heatmap_flat.reshape(B, H, W)

        return heatmap


# ============================================================================
# Object-Text Fusion Module (序列拼接 + 全局自注意力)
# ============================================================================

class ObjTextFusion(nn.Module):
    """Fuse object patch tokens with text tokens by sequence concatenation + self-attention.

    物体图像 729 tokens + 文本 T tokens → 拼接为 (729+T) tokens 的整体序列，
    通过全局自注意力让所有 token 相互交互，形成统一的上下文表示。

    Args:
        siglip_dim: SigLIP feature dimension (e.g. 1152)
        output_dim: output dimension per token
        num_heads: number of attention heads
        num_layers: number of self-attention layers
    """

    def __init__(self, siglip_dim: int, output_dim: int = 256,
                 num_heads: int = 8, num_layers: int = 2):
        super().__init__()

        self.proj = nn.Linear(siglip_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

        # Self-attention layers for cross-modal interaction
        self.self_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=output_dim,
                num_heads=num_heads,
                batch_first=True,
            )
            for _ in range(num_layers)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(output_dim)
            for _ in range(num_layers)
        ])

        self.num_layers = num_layers

    def forward(
        self,
        obj_patches: torch.Tensor,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fuse object and text token sequences with global self-attention.

        Args:
            obj_patches: [B, N_obj, siglip_dim] object patch tokens (729)
            text_tokens: [B, T, siglip_dim] text tokens
            text_mask: [B, T] bool, True = valid token

        Returns:
            tuple:
                - fused: [B, N_obj+T, output_dim] fused sequence after global self-attention
                - padding_mask: [B, N_obj+T] bool, True = 需要屏蔽的 padding
        """
        B = obj_patches.size(0)
        N_obj = obj_patches.size(1)
        T_len = text_tokens.size(1)

        # Expand text to match batch size if needed
        if text_tokens.size(0) == 1 and B > 1:
            text_tokens = text_tokens.expand(B, -1, -1)
            text_mask = text_mask.expand(B, -1)

        # Concatenate as a whole sequence: [obj_patches, text_tokens]
        combined = torch.cat([obj_patches, text_tokens], dim=1)  # [B, N_obj+T, siglip_dim]

        # Project to output_dim
        fused = self.proj(combined)
        fused = self.norm(fused)

        # Build padding mask: object tokens are all valid, text uses original mask
        obj_mask = torch.ones(B, N_obj, dtype=torch.bool, device=obj_patches.device)
        # key_padding_mask: True = IGNORE (padding)
        padding_mask = ~torch.cat([obj_mask, text_mask], dim=1)  # [B, N_obj+T]

        # Self-attention layers for global token interaction
        for i in range(self.num_layers):
            # Global self-attention with padding mask
            attn_out, _ = self.self_attn_layers[i](
                query=fused,
                key=fused,
                value=fused,
                key_padding_mask=padding_mask,
            )
            # Residual connection + layer norm
            fused = self.layer_norms[i](fused + attn_out)

        return fused, padding_mask


# ============================================================================
# Full PlacementHeatmap Module (pure SigLIP)
# ============================================================================

class PlacementHeatmap(nn.Module):
    """Complete heatmap generation pipeline with pure SigLIP.

    Architecture:
        Room Image   -> SigLIP ViT -> 729 spatial tokens -> SpatialRefinement -> Query
        Object Image -> SigLIP ViT -> 729 patch tokens ──┐
                                                         +-> 拼接 -> 全局自注意力 -> KV sequence
        Text         -> SigLIP Text Encoder -> T tokens ─┘
                                                                         |
                                                               CrossAttention -> Heatmap

        所有编码保留原始序列维度。物体+文本作为整体序列全局自注意力。
    """

    SIGLIP_MODEL = "google/siglip-so400m-patch14-384"

    def __init__(self, siglip_model: str = None, heatmap_res: int = 256):
        super().__init__()
        self.heatmap_res = heatmap_res

        siglip_model = siglip_model or self.SIGLIP_MODEL

        # SigLIP vision encoder (shared for room spatial + object patches)
        self.vision_encoder = SiglipVisionEncoder(siglip_model)
        siglip_dim = self.vision_encoder.feature_dim

        # SigLIP text encoder
        self.text_encoder = SiglipTextEncoder(siglip_model)

        # Hidden dim for attention modules
        hidden_dim = 256

        # Object-Text fusion (sequence concat, 不压缩)
        self.obj_text_fusion = ObjTextFusion(siglip_dim, output_dim=hidden_dim)

        # Spatial refinement (self-attention on room features)
        self.spatial_refinement = SpatialRefinement(
            in_channels=siglip_dim,
            hidden_dim=hidden_dim,
            num_heads=8,
        )

        # Cross-attention heatmap head
        # spatial_features 来自 SpatialRefinement (siglip_dim), KV 来自 ObjTextFusion (hidden_dim)
        self.heatmap_head = CrossAttentionHeatmap(
            feature_dim=siglip_dim,
            kv_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_heads=8,
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

        # Stage 3: SigLIP encode object image (patch sequence) + text (token sequence)
        if object_image_path:
            obj_tensor = self.preprocess_image(object_image_path)
            obj_patches = self.vision_encoder.encode_patches(obj_tensor)  # [1, 729, C]
        else:
            device = next(self.vision_encoder.model.parameters()).device
            obj_patches = torch.zeros(1, self.vision_encoder.num_patches,
                                      self.vision_encoder.feature_dim, device=device)

        text_tokens, text_mask = self.text_encoder.encode(object_desc)  # [1, T, C], [1, T]

        # Stage 4: Object-Text fusion -> KV sequence
        kv_seq, kv_padding_mask = self.obj_text_fusion(obj_patches, text_tokens, text_mask)

        # Stage 5: Cross-attention -> heatmap at ViT patch resolution
        heatmap_low = self.heatmap_head(spatial_features, kv_seq, kv_padding_mask)  # [1, g, g]

        # Stage 6: Upsample to target heatmap resolution
        heatmap = F.interpolate(
            heatmap_low.unsqueeze(0),
            size=(self.heatmap_res, self.heatmap_res),
            mode='bilinear',
            align_corners=False,
        ).squeeze(0).squeeze(0)

        # Normalize to [0, 1] (inference only)
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
        B = room_image.size(0)

        # Stage 1: SigLIP encode room top view -> spatial features
        spatial_features = self.vision_encoder.encode_spatial(room_image)  # [B, g, g, C]

        # Stage 2: Spatial refinement (self-attention)
        spatial_features = self.spatial_refinement(spatial_features)

        # Stage 3: SigLIP encode object image (patch sequence) + text (token sequence)
        if object_image is not None:
            obj_patches = self.vision_encoder.encode_patches(object_image)  # [B, 729, C]
        else:
            device = next(self.vision_encoder.model.parameters()).device
            obj_patches = torch.zeros(B, self.vision_encoder.num_patches,
                                      self.vision_encoder.feature_dim, device=device)

        text_tokens, text_mask = self.text_encoder.encode(object_desc)  # [1, T, C], [1, T]

        # Stage 4: Object-Text fusion -> KV sequence
        kv_seq, kv_padding_mask = self.obj_text_fusion(obj_patches, text_tokens, text_mask)

        # Stage 5: Cross-attention -> heatmap at ViT patch resolution
        heatmap_low = self.heatmap_head(spatial_features, kv_seq, kv_padding_mask)  # [B, g, g]

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
