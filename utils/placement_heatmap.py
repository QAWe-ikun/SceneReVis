"""
Learning-based placement heatmap module.

Uses a ViT to encode the top-down room view, then cross-attends
with the object description to produce a 2D placement probability field.

Architecture:
    Top View → ViT Encoder → Self-Attention → Cross-Attention(text query) → Heatmap
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
from PIL import Image


# ============================================================================
# CLIP ViT Encoder (spatial feature extraction)
# ============================================================================

class ClipViTEncoder(nn.Module):
    """Wraps an OpenCLIP visual encoder to extract spatial patch features.

    Unlike standard CLIP which uses only the pooled [CLS] token,
    this encoder returns the full sequence of patch tokens reshaped
    to a 2D spatial grid.

    Args:
        model_name: OpenCLIP model name, e.g. "ViT-L-14"
        pretrained: OpenCLIP pretrained weight name
    """

    def __init__(self, model_name: str = "ViT-L-14",
                 pretrained: str = "laion2b_s32b_b82k"):
        super().__init__()
        import open_clip

        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device
        )

        # Extract the visual trunk (ViT encoder without projection head)
        self.visual = clip_model.visual
        self.feature_dim = self.visual.ln_post.normalized_shape[0]

        # CLIP ViT patch grid size
        # ViT-L-14 with 224x224 input → 224/14 = 16 patches per side
        # But we need to handle different input resolutions
        self.patch_size = 14  # ViT-L-14 uses 14x14 patches
        self.input_size = 224
        self.num_patches = (self.input_size // self.patch_size) ** 2  # 256
        self.grid_size = self.input_size // self.patch_size  # 16

        # Freeze all CLIP weights
        for param in self.visual.parameters():
            param.requires_grad = False

        self.eval()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Extract spatial features from image tensor.

        Args:
            image: [B, 3, H, W] preprocessed image tensor

        Returns:
            [B, grid_size, grid_size, feature_dim] spatial feature map
        """
        # Use the visual trunk to get patch tokens
        x = self.visual.conv1(image)  # [B, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # [B, width, grid*grid]
        x = x.permute(0, 2, 1)  # [B, grid*grid, width]

        # Prepend class token
        class_embedding = self.visual.class_embedding
        if class_embedding.ndim == 1:
            class_embedding = class_embedding.unsqueeze(0).unsqueeze(0)
        class_tokens = class_embedding.expand(x.shape[0], -1, -1)
        x = torch.cat([class_tokens, x], dim=1)  # [B, 1+grid*grid, width]

        # Add positional embeddings
        x = x + self.visual.positional_embedding.to(x.dtype)

        # Transformer layers
        x = self.visual.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD → LND

        for block in self.visual.transformer.resblocks:
            x = block(x)

        x = x.permute(1, 0, 2)  # LND → NLD

        # Remove CLS token, keep patch tokens only
        patch_tokens = x[:, 1:, :]  # [B, grid*grid, width]

        # Reshape to 2D spatial grid
        B = patch_tokens.shape[0]
        features = patch_tokens.reshape(B, self.grid_size, self.grid_size, self.feature_dim)

        return features


# ============================================================================
# Spatial Attention Module
# ============================================================================

class SpatialRefinement(nn.Module):
    """Refines spatial features using windowed self-attention.

    To avoid the O(n^2) cost on 256x256 grid, we first downsample
    via strided conv, apply self-attention at lower resolution,
    then upsample back.
    """

    def __init__(self, in_channels: int, hidden_dim: int = 256,
                 num_heads: int = 8, window_size: int = 8):
        super().__init__()
        self.window_size = window_size

        # Channel reduction for efficiency
        self.proj_in = nn.Linear(in_channels, hidden_dim)

        # Windowed self-attention: process the grid in non-overlapping windows
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )

        # Output projection
        self.proj_out = nn.Linear(hidden_dim, in_channels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Apply windowed self-attention to spatial features.

        Args:
            features: [B, H, W, C] spatial feature map

        Returns:
            [B, H, W, C] refined spatial feature map
        """
        B, H, W, C = features.shape
        x = self.proj_in(features)  # [B, H, W, hidden]

        # Pad to multiple of window_size
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, pad_h, 0))

        pH, pW = x.shape[1], x.shape[2]
        nH, nW = pH // self.window_size, pW // self.window_size

        # Reshape into windows: [B * nH * nW, window_size^2, hidden]
        x_windows = x.reshape(B, nH, self.window_size, nW, self.window_size, -1)
        x_windows = x_windows.permute(0, 1, 3, 2, 4, 5)  # [B, nH, nW, ws, ws, C]
        x_windows = x_windows.reshape(B * nH * nW, self.window_size * self.window_size, -1)

        # Self-attention within each window
        attn_out, _ = self.self_attn(x_windows, x_windows, x_windows)

        # Reshape back to spatial grid
        attn_out = attn_out.reshape(B, nH, nW, self.window_size, self.window_size, -1)
        attn_out = attn_out.permute(0, 1, 3, 2, 4, 5)  # [B, nH, ws, nW, ws, C]
        attn_out = attn_out.reshape(B, pH, pW, -1)

        # Remove padding
        if pad_h > 0 or pad_w > 0:
            attn_out = attn_out[:, :H, :W, :]

        return self.proj_out(attn_out) + features  # residual


# ============================================================================
# Cross-Attention Heatmap Head
# ============================================================================

class CrossAttentionHeatmap(nn.Module):
    """Computes a 2D heatmap via cross-attention between text query and spatial features.

    The text embedding serves as the query, and each spatial feature vector
    serves as a key. The resulting attention weights form the heatmap.
    """

    def __init__(self, feature_dim: int, text_dim: int = 768):
        super().__init__()
        # Projection layers to align text and visual dimensions
        self.visual_proj = nn.Linear(feature_dim, text_dim)
        self.text_proj = nn.Linear(text_dim, text_dim)

        # Temperature for scaling attention logits
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, spatial_features: torch.Tensor,
                text_query: torch.Tensor) -> torch.Tensor:
        """Compute placement heatmap.

        Args:
            spatial_features: [B, H, W, C_v] from spatial refinement
            text_query: [B, C_t] text embedding

        Returns:
            [B, H, W] heatmap with values in [0, 1]
        """
        B, H, W, C_v = spatial_features.shape

        # Project to shared dimension
        visual_keys = self.visual_proj(spatial_features)  # [B, H, W, D]
        text_q = self.text_proj(text_query)  # [B, D]

        # Flatten spatial dimensions
        visual_keys_flat = visual_keys.reshape(B, H * W, -1)  # [B, HW, D]

        # Compute attention logits
        logits = torch.einsum("bd,bnd->bn", text_q, visual_keys_flat)  # [B, HW]
        logits = logits / self.temperature.clamp(min=0.01)

        # Softmax over spatial positions → probability distribution
        heatmap_flat = F.softmax(logits, dim=-1)  # [B, HW]

        # Reshape to 2D and upscale to heatmap_res
        heatmap = heatmap_flat.reshape(B, H, W)

        return heatmap


# ============================================================================
# Full PlacementHeatmap Module
# ============================================================================

class PlacementHeatmap(nn.Module):
    """Complete heatmap generation pipeline.

    Top View PNG → ViT Encoder → Spatial Refinement → Cross-Attention(text) → Heatmap
    """

    def __init__(self, clip_model_name: str = "ViT-L-14",
                 clip_pretrained: str = "laion2b_s32b_b82k",
                 heatmap_res: int = 256):
        super().__init__()
        self.heatmap_res = heatmap_res

        # CLIP ViT encoder for spatial features
        self.vit_encoder = ClipViTEncoder(clip_model_name, clip_pretrained)
        feature_dim = self.vit_encoder.feature_dim

        # Spatial refinement (windowed self-attention)
        self.spatial_refinement = SpatialRefinement(
            in_channels=feature_dim,
            hidden_dim=256,
            num_heads=8,
            window_size=8,
        )

        # Cross-attention heatmap head
        self.heatmap_head = CrossAttentionHeatmap(
            feature_dim=feature_dim,
            text_dim=768,
        )

        # Text encoder: simple projection from CLIP text embedding
        # We use a lightweight text encoder that takes CLIP text tokens
        self.text_dim = 768

    @torch.no_grad()
    def encode_image(self, image_path: str, device: Optional[str] = None) -> torch.Tensor:
        """Load and preprocess a top-view image, return spatial features.

        Args:
            image_path: path to the top-view PNG
            device: override device (default: auto-detect)

        Returns:
            [1, grid_size, grid_size, feature_dim] spatial features
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        image = Image.open(image_path).convert("RGB")
        preprocessed = self.vit_encoder.preprocess(image).unsqueeze(0).to(device)

        return self.vit_encoder(preprocessed)

    def encode_text(self, text: str, device: Optional[str] = None) -> torch.Tensor:
        """Encode object description into a text embedding.

        Uses OpenCLIP's text encoder for consistent alignment with visual features.

        Args:
            text: object description (e.g. "bed", "desk lamp")

        Returns:
            [1, text_dim] text embedding
        """
        import open_clip
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        tokenizer = open_clip.get_tokenizer("ViT-L-14")
        text_tokens = tokenizer([text]).to(device)

        # Get text features from CLIP
        # We need access to the text encoder, which is separate from visual
        clip_model = self.vit_encoder.visual
        # The text encoder is on the parent CLIP model, not just visual
        # For now, use a simpler approach: encode text via the same mechanism

        # Actually, we need the full CLIP model for text encoding.
        # Let's create a text encoder that doesn't depend on the full CLIP model.
        return self._encode_text_simple(text, text_tokens, device)

    def _encode_text_simple(self, text: str, text_tokens: torch.Tensor,
                            device: str) -> torch.Tensor:
        """Simple text encoding using the CLIP text encoder."""
        import open_clip

        # We need the full CLIP model, not just visual
        # Load it if not already available
        if not hasattr(self, '_clip_full'):
            self._clip_full, _, _ = open_clip.create_model_and_transforms(
                "ViT-L-14", pretrained="laion2b_s32b_b82k", device=device
            )
            for param in self._clip_full.parameters():
                param.requires_grad = False

        text_features = self._clip_full.encode_text(text_tokens)
        text_features = F.normalize(text_features.float(), p=2, dim=-1)
        return text_features  # [1, 768]

    def forward(self, image_path: str, object_desc: str,
                device: Optional[str] = None) -> torch.Tensor:
        """Generate placement heatmap for an object in a room.

        Args:
            image_path: path to the top-view PNG of the room
            object_desc: text description of the object to place

        Returns:
            [H, W] heatmap tensor with values in [0, 1], upscaled to heatmap_res
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Stage 1: ViT encode top view → spatial features
        spatial_features = self.encode_image(image_path, device)  # [1, g, g, C]

        # Stage 2: Spatial refinement (self-attention)
        spatial_features = self.spatial_refinement(spatial_features)

        # Stage 3: Text encoding
        text_features = self.encode_text(object_desc, device)  # [1, text_dim]

        # Stage 4: Cross-attention → heatmap at CLIP patch resolution
        heatmap_low = self.heatmap_head(spatial_features, text_features)  # [1, g, g]

        # Stage 5: Upsample to target heatmap resolution
        heatmap = F.interpolate(
            heatmap_low.unsqueeze(0),  # [1, 1, g, g]
            size=(self.heatmap_res, self.heatmap_res),
            mode='bilinear',
            align_corners=False,
        ).squeeze(0).squeeze(0)  # [H, W]

        # Normalize to [0, 1] (softmax already ensures sum=1, but scale for visibility)
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        return heatmap


# ============================================================================
# Unified inference interface (combines heatmap + mask)
# ============================================================================

class PlacementEngine:
    """High-level placement engine that combines heatmap and feasibility mask.

    Usage:
        engine = PlacementEngine()
        position = engine.place_object(
            scene=scene_json,
            top_view_path="path/to/top_view.png",
            object_desc="bed",
            size=[1.8, 0.8, 2.1],
            rotation=[0, 0, 0, 1],
            placement_plane="floor",
        )
    """

    def __init__(self, heatmap_res: int = 256,
                 heatmap_model: Optional[PlacementHeatmap] = None,
                 enable_heatmap: bool = True):
        """
        Args:
            heatmap_res: grid resolution
            heatmap_model: pre-loaded PlacementHeatmap module, or None for
                          uniform heatmap (no learning)
            enable_heatmap: if False, skip heatmap and use uniform scores
        """
        self.heatmap_res = heatmap_res
        self.enable_heatmap = enable_heatmap
        self.heatmap_model = heatmap_model

    def compute_heatmap(self, top_view_path: str,
                        object_desc: str) -> np.ndarray:
        """Compute the learning-based heatmap for a given object.

        Returns:
            numpy array [heatmap_res, heatmap_res] with values in [0, 1].
        """
        if not self.enable_heatmap or self.heatmap_model is None:
            # Fallback: uniform heatmap (position decided purely by mask)
            return np.ones((self.heatmap_res, self.heatmap_res), dtype=np.float32)

        self.heatmap_model.eval()
        with torch.no_grad():
            heatmap = self.heatmap_model(top_view_path, object_desc)
        return heatmap.cpu().numpy()

    def place_object(self, scene: dict, top_view_path: str,
                     object_desc: str, size: list,
                     rotation: list = None,
                     placement_plane: str = "floor",
                     clearance: float = 0.5) -> Optional[list]:
        """Compute the optimal placement position for an object.

        Args:
            scene: scene JSON data
            top_view_path: path to top-view PNG
            object_desc: text description of the object
            size: [width, height, depth]
            rotation: [x, y, z, w] quaternion (used for collision mask)
            placement_plane: "floor" or target object jid
            clearance: minimum clearance around existing objects

        Returns:
            [x, y, z] placement position, or None if no feasible position found.
        """
        from utils.placement_mask import compute_mask, find_best_position

        # Step 1: Compute feasibility mask
        mask, ortho_scale, cx, cz = compute_mask(
            scene, self.heatmap_res, clearance, placement_plane
        )

        if not np.any(mask):
            print(f"[placement] No feasible position for '{object_desc}'")
            return None

        # Step 2: Compute learning-based heatmap
        heatmap = self.compute_heatmap(top_view_path, object_desc)

        # Step 3: Fuse heatmap × mask
        score = heatmap * mask.astype(np.float32)

        if not np.any(score > 0):
            # Heatmap disagrees with all feasible positions, fallback to mask
            score = mask.astype(np.float32)

        # Step 4: Extract best position
        positions = find_best_position_from_score(
            score, ortho_scale, cx, cz, self.heatmap_res,
            placement_plane, scene, top_k=1
        )

        if not positions:
            print(f"[placement] Failed to find position for '{object_desc}'")
            return None

        print(f"[placement] Optimal position for '{object_desc}': {positions[0]}")
        return positions[0]


def find_best_position_from_score(score: np.ndarray, ortho_scale: float,
                                  cx: float, cz: float, heatmap_res: int,
                                  placement_plane: str = "floor",
                                  scene: Optional[dict] = None,
                                  top_k: int = 1,
                                  min_distance: float = 0.5) -> list:
    """Extract top-k positions from a fused score field (heatmap × mask).

    Args:
        score: 2D float array [heatmap_res, heatmap_res]
        ortho_scale: orthographic scale in meters
        cx, cz: room center
        heatmap_res: grid resolution
        placement_plane: "floor" or object jid
        scene: optional scene data (for Y height when placing on object)
        top_k: number of candidates
        min_distance: minimum distance between candidates

    Returns:
        List of [x, y, z] positions.
    """
    from utils.placement_mask import grid_to_world, _extract_objects

    if not np.any(score > 0):
        return []

    positions = []
    working_score = score.copy()

    for _ in range(top_k):
        idx = np.argmax(working_score.ravel())
        gi, gj = divmod(idx, heatmap_res)
        x, z = grid_to_world(gi, gj, cx, cz, heatmap_res)

        # Y coordinate
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

        # Suppress neighborhood
        cell_size = ortho_scale / heatmap_res
        radius_cells = max(1, int(min_distance / cell_size))
        y1 = max(0, gi - radius_cells)
        y2 = min(heatmap_res, gi + radius_cells + 1)
        x1 = max(0, gj - radius_cells)
        x2 = min(heatmap_res, gj + radius_cells + 1)
        working_score[y1:y2, x1:x2] = 0

    return positions


def visualize_placement(score: np.ndarray, top_view_path: str,
                        positions: list, object_desc: str,
                        ortho_scale: float, cx: float, cz: float,
                        heatmap_res: int, save_path: str) -> None:
    """Create a debug visualization overlaying the score field on the top view.

    Args:
        score: 2D float array [heatmap_res, heatmap_res] — fused score
        top_view_path: path to the top-view PNG
        positions: list of [x, y, z] positions (argmax candidates)
        object_desc: object description
        ortho_scale: Blender orthographic scale
        cx, cz: room center
        heatmap_res: grid resolution
        save_path: output PNG path
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image

    # Load top view
    top_img = Image.open(top_view_path).convert('RGB')

    # Resize score to match top view resolution
    score_resized = np.array(Image.fromarray(
        (score * 255).astype(np.uint8)
    ).resize((top_img.width, top_img.height), Image.BILINEAR)) / 255.0

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(top_img)

    # Overlay score heatmap as semi-transparent colormap
    cmap = plt.cm.jet
    heatmap_rgba = cmap(score_resized)
    heatmap_rgba[..., 3] = 0.4  # 40% opacity
    ax.imshow(heatmap_rgba)

    # Mark best positions with red dots
    for i, pos in enumerate(positions):
        px, pz = pos[0], pos[2]
        # Convert world coords to pixel coords (linear ortho mapping)
        u = ((px - cx) / ortho_scale + 0.5) * top_img.width
        v = ((cz - pz) / ortho_scale + 0.5) * top_img.height
        ax.plot(u, v, 'rX', markersize=15, linewidth=2,
                label=f'#{i+1} ({px:.2f}, {pz:.2f})' if i == 0 else f'#{i+1}')

    ax.set_title(f'Placement: "{object_desc}"\nScore = Heatmap × FeasibleMask')
    ax.legend(loc='upper right', fontsize=8)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[placement] Visualization saved to {save_path}")
