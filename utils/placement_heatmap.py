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

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import torchvision.transforms as T


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REMOTE_SIGLIP_MODEL = "google/siglip-so400m-patch14-384"
REMOTE_DINOV2_MODEL = "facebook/dinov2-base"
DEFAULT_LOCAL_SIGLIP_MODEL = PROJECT_ROOT / "ckpt" / "google" / "siglip-so400m-patch14-384"
DEFAULT_LOCAL_DINOV2_MODEL = PROJECT_ROOT / "ckpt" / "facebook" / "dinov2-base"

SIGLIP_IMAGE_SIZE = 384
DINO_IMAGE_SIZE = 518
SIGLIP_MEAN = [0.5, 0.5, 0.5]
SIGLIP_STD = [0.5, 0.5, 0.5]
DINO_MEAN = [0.485, 0.456, 0.406]
DINO_STD = [0.229, 0.224, 0.225]

FROZEN_ENCODER_STATE_PREFIXES = (
    "room_encoder.model.",
    "vision_encoder.model.",
    "text_encoder.model.",
)


def resolve_model_path(
    explicit_model: Optional[str],
    env_var: str,
    default_local: Path,
    default_remote: str,
) -> str:
    """Resolve a model id/path, preferring explicit args, env vars, then local ckpt."""
    requested = explicit_model or os.environ.get(env_var)
    if requested:
        requested_path = Path(requested).expanduser()
        if requested_path.exists():
            return str(requested_path)
        if requested_path.is_absolute() or requested.startswith(("~", ".")) or "\\" in requested:
            raise FileNotFoundError(
                f"{env_var if not explicit_model else 'model path'} points to a missing path: "
                f"{requested_path}"
            )
        return requested
    if default_local.exists():
        return str(default_local)
    return default_remote


def is_local_model_path(model_name: str) -> bool:
    return Path(model_name).expanduser().exists()


def resolve_siglip_model(model_name: Optional[str] = None) -> str:
    return resolve_model_path(
        model_name,
        "SCENEREVIS_SIGLIP_MODEL",
        DEFAULT_LOCAL_SIGLIP_MODEL,
        REMOTE_SIGLIP_MODEL,
    )


def resolve_dinov2_model(model_name: Optional[str] = None) -> str:
    return resolve_model_path(
        model_name,
        "SCENEREVIS_DINOV2_MODEL",
        DEFAULT_LOCAL_DINOV2_MODEL,
        REMOTE_DINOV2_MODEL,
    )


def build_siglip_image_transform(image_size: int = SIGLIP_IMAGE_SIZE):
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=SIGLIP_MEAN, std=SIGLIP_STD),
    ])


def build_dino_image_transform(image_size: int = DINO_IMAGE_SIZE):
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=DINO_MEAN, std=DINO_STD),
    ])


def is_frozen_encoder_state_key(key: str) -> bool:
    """Return True for frozen SigLIP weights that should not live in checkpoints."""
    return key.startswith(FROZEN_ENCODER_STATE_PREFIXES)


def trainable_heatmap_state_dict(model: nn.Module) -> dict:
    """State dict excluding frozen SigLIP encoder weights."""
    return {
        key: value
        for key, value in model.state_dict().items()
        if not is_frozen_encoder_state_key(key)
    }


def load_trainable_heatmap_state_dict(model: nn.Module, state_dict: dict):
    """Load trainable/head weights from a trainable-only checkpoint."""
    frozen_keys = [key for key in state_dict if is_frozen_encoder_state_key(key)]
    if frozen_keys:
        raise ValueError(
            "Trainable-only checkpoint unexpectedly contains frozen SigLIP keys, "
            f"for example: {frozen_keys[:3]}"
        )

    incompatible = model.load_state_dict(state_dict, strict=False)
    missing_keys = [
        key for key in incompatible.missing_keys
        if not is_frozen_encoder_state_key(key)
    ]
    return missing_keys, incompatible.unexpected_keys


# ============================================================================
# SigLIP Vision Encoder (shared for room spatial + object patch features)
# ============================================================================

class SiglipVisionEncoder(nn.Module):
    """SigLIP vision encoder for both room spatial and object patch features.

    Args:
        model_name: HuggingFace model name, e.g. "google/siglip-so400m-patch14-384"
    """

    def __init__(self, model_name: Optional[str] = None, image_size: Optional[int] = None):
        super().__init__()
        from transformers import SiglipVisionModel

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = resolve_siglip_model(model_name)
        self.model = SiglipVisionModel.from_pretrained(
            model_name,
            local_files_only=is_local_model_path(model_name),
        ).to(device)

        config = self.model.config
        self.feature_dim = config.hidden_size
        self.patch_size = config.patch_size
        self.image_size = image_size or config.image_size

        # Freeze all weights
        for param in self.model.parameters():
            param.requires_grad = False

        # Preprocess: resize + normalize (SigLIP uses mean=0.5, std=0.5)
        self.preprocess = build_siglip_image_transform(self.image_size)

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
        with torch.no_grad():
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
        with torch.no_grad():
            outputs = self.model(pixel_values=image, output_hidden_states=True)
        hidden = outputs.last_hidden_state  # [B, seq_len, C]
        patch_tokens = hidden[:, 1:, :] if self.drop_cls else hidden
        return patch_tokens  # [B, 729, C]


class DinoVisionEncoder(nn.Module):
    """DINOv2 vision encoder for room top-view spatial features."""

    def __init__(self, model_name: Optional[str] = None, image_size: int = DINO_IMAGE_SIZE):
        super().__init__()
        from transformers import AutoModel

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = resolve_dinov2_model(model_name)
        self.model = AutoModel.from_pretrained(
            model_name,
            local_files_only=is_local_model_path(model_name),
        ).to(device)

        config = self.model.config
        self.feature_dim = config.hidden_size
        self.patch_size = getattr(config, "patch_size", 14)
        self.image_size = image_size

        for param in self.model.parameters():
            param.requires_grad = False

        self.preprocess = build_dino_image_transform(self.image_size)

        dummy = torch.zeros(1, 3, self.image_size, self.image_size, device=device)
        with torch.no_grad():
            outputs = self._forward_model(dummy)
        patch_tokens = self._extract_patch_tokens(outputs.last_hidden_state)
        self.num_patches = patch_tokens.shape[1]
        self.grid_size = int(round(self.num_patches ** 0.5))
        if self.grid_size * self.grid_size != self.num_patches:
            raise ValueError(
                f"DINOv2 patch token count is not square: {self.num_patches}"
            )
        print(
            f"[DinoVisionEncoder] patches={self.num_patches}, "
            f"grid={self.grid_size}x{self.grid_size}, dim={self.feature_dim}"
        )

        self.eval()

    def _forward_model(self, pixel_values: torch.Tensor):
        try:
            return self.model(pixel_values=pixel_values, interpolate_pos_encoding=True)
        except TypeError:
            return self.model(pixel_values=pixel_values)

    def _extract_patch_tokens(self, hidden: torch.Tensor) -> torch.Tensor:
        seq_len = hidden.shape[1]
        if int(round(seq_len ** 0.5)) ** 2 == seq_len:
            return hidden

        candidates = [1 + int(getattr(self.model.config, "num_register_tokens", 0)), 1]
        for offset in candidates:
            num_patches = seq_len - offset
            if num_patches > 0 and int(round(num_patches ** 0.5)) ** 2 == num_patches:
                return hidden[:, offset:, :]

        raise ValueError(f"Cannot infer DINOv2 patch tokens from seq_len={seq_len}")

    def encode_spatial(self, image: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            outputs = self._forward_model(image)
        patch_tokens = self._extract_patch_tokens(outputs.last_hidden_state)
        B = patch_tokens.shape[0]
        return patch_tokens.reshape(B, self.grid_size, self.grid_size, self.feature_dim)


# ============================================================================
# SigLIP Text Encoder
# ============================================================================

class SiglipTextEncoder(nn.Module):
    """SigLIP text encoder for object descriptions.

    Args:
        model_name: HuggingFace model name, e.g. "google/siglip-so400m-patch14-384"
    """

    def __init__(self, model_name: Optional[str] = None):
        super().__init__()
        from transformers import SiglipTextModel, AutoTokenizer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = resolve_siglip_model(model_name)
        local_files_only = is_local_model_path(model_name)
        self.model = SiglipTextModel.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        ).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )

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

    SIGLIP_MODEL = REMOTE_SIGLIP_MODEL
    DINOV2_MODEL = REMOTE_DINOV2_MODEL

    def __init__(
        self,
        siglip_model: str = None,
        heatmap_res: int = 256,
        room_encoder: str = "siglip",
        dino_model: str = None,
        hidden_dim: int = 256,
        room_image_size: Optional[int] = None,
        object_image_size: Optional[int] = None,
    ):
        super().__init__()
        self.heatmap_res = heatmap_res
        self.room_encoder_type = room_encoder.lower()

        if self.room_encoder_type not in {"siglip", "dinov2"}:
            raise ValueError(f"Unsupported room_encoder: {room_encoder}")

        siglip_model = resolve_siglip_model(siglip_model)
        print(f"[PlacementHeatmap] Using SigLIP object/text model: {siglip_model}")

        # SigLIP vision encoder (shared for room spatial + object patches)
        self.vision_encoder = SiglipVisionEncoder(
            siglip_model,
            image_size=object_image_size,
        )
        siglip_dim = self.vision_encoder.feature_dim

        # SigLIP text encoder
        self.text_encoder = SiglipTextEncoder(siglip_model)

        if self.room_encoder_type == "dinov2":
            dino_model = resolve_dinov2_model(dino_model)
            print(f"[PlacementHeatmap] Using DINOv2 room model: {dino_model}")
            self.room_encoder = DinoVisionEncoder(
                dino_model,
                image_size=room_image_size or DINO_IMAGE_SIZE,
            )
            room_dim = self.room_encoder.feature_dim
        else:
            self.room_encoder = None
            room_dim = siglip_dim

        # Object-Text fusion (sequence concat, 不压缩)
        self.obj_text_fusion = ObjTextFusion(siglip_dim, output_dim=hidden_dim)

        # Spatial refinement (self-attention on room features)
        self.spatial_refinement = SpatialRefinement(
            in_channels=room_dim,
            hidden_dim=hidden_dim,
            num_heads=8,
        )

        # Cross-attention heatmap head
        # spatial_features 来自 SpatialRefinement (siglip_dim), KV 来自 ObjTextFusion (hidden_dim)
        self.heatmap_head = CrossAttentionHeatmap(
            feature_dim=room_dim,
            kv_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_heads=8,
        )

    def preprocess_room_image(self, image):
        if self.room_encoder_type == "dinov2":
            return self.room_encoder.preprocess(image)
        return self.vision_encoder.preprocess(image)

    def preprocess_object_image(self, image):
        return self.vision_encoder.preprocess(image)

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
        return self.forward(room_image, object_desc, object_image)

    def forward(
        self,
        room_image: torch.Tensor,
        object_desc: str,
        object_image: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate placement heatmap from preprocessed tensor inputs."""
        B = room_image.size(0)

        # Stage 1: encode room top view -> spatial features
        if self.room_encoder_type == "dinov2":
            spatial_features = self.room_encoder.encode_spatial(room_image)  # [B, g, g, C]
        else:
            spatial_features = self.vision_encoder.encode_spatial(room_image)  # [B, g, g, C]

        # Stage 2: Spatial refinement (self-attention)
        spatial_features = self.spatial_refinement(spatial_features)

        # Stage 3: SigLIP encode object image (patch sequence) + text (token sequence)
        if object_image is not None:
            obj_patches = self.vision_encoder.encode_patches(object_image)  # [B, 729, C]
        else:
            device = spatial_features.device
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
