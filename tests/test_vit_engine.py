"""
Test ViT heatmap engine end-to-end (SigLIP-based).
Loads PlacementHeatmap model, runs a fake top-view image through the pipeline.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
from PIL import Image


def create_fake_top_view(path, size=(512, 512)):
    """Create a simple fake top-view image."""
    img = Image.new('RGB', size, color=(200, 200, 200))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    w, h = size
    margin = 50
    draw.rectangle([margin, margin, w-margin, h-margin], outline=(50, 50, 50), width=3)
    draw.rectangle([w//3, h//3, w//3+80, h//3+60], fill=(100, 80, 60))
    img.save(path)
    print(f"Created fake top-view at {path}")


def test_vit_pipeline():
    from utils.placement_heatmap import PlacementHeatmap, PlacementEngine

    # Stage 1: Load model
    print("=== Stage 1: Load PlacementHeatmap (SigLIP) ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = PlacementHeatmap(heatmap_res=256)
    model.to(device)
    model.eval()
    print(f"Model loaded: {type(model).__name__}")
    print(f"  ViT grid_size: {model.vit_encoder.grid_size}x{model.vit_encoder.grid_size}")
    print(f"  Feature dim: {model.vit_encoder.feature_dim}")

    # Stage 2: Encode image
    print("\n=== Stage 2: Encode top-view → spatial features ===")
    fake_img = os.path.join(os.path.dirname(__file__), 'fake_top_view.png')
    create_fake_top_view(fake_img)

    spatial_feats = model.encode_image(fake_img, device)
    print(f"Spatial features shape: {spatial_feats.shape}")
    assert spatial_feats.dim() == 4, "Expected [B, H, W, C]"
    assert spatial_feats.shape[0] == 1, "Batch size should be 1"

    # Stage 3: Encode text
    print("\n=== Stage 3: Encode text description ===")
    text_feats = model.encode_text("nightstand", device)
    print(f"Text features shape: {text_feats.shape}")
    assert text_feats.dim() == 2, "Expected [B, text_dim]"

    # Stage 4: Spatial refinement
    print("\n=== Stage 4: Spatial refinement (self-attention) ===")
    refined = model.spatial_refinement(spatial_feats)
    print(f"Refined features shape: {refined.shape}")

    # Stage 5: Cross-attention → heatmap
    print("\n=== Stage 5: Cross-attention → heatmap ===")
    heatmap = model.heatmap_head(refined, text_feats)
    print(f"Heatmap shape: {heatmap.shape}")
    print(f"Heatmap range: [{heatmap.min():.4f}, {heatmap.max():.4f}]")
    assert heatmap.min() >= 0.0 and heatmap.max() <= 1.0, "Heatmap should be in [0,1]"
    print(f"Heatmap sum: {heatmap.sum():.4f}")

    # Stage 6: Full forward pass
    print("\n=== Stage 6: Full PlacementHeatmap forward ===")
    with torch.no_grad():
        full_heatmap = model(fake_img, "nightstand", device)
    print(f"Full heatmap shape: {full_heatmap.shape}")
    print(f"Full heatmap range: [{full_heatmap.min():.4f}, {full_heatmap.max():.4f}]")

    # Stage 7: PlacementEngine integration
    print("\n=== Stage 7: PlacementEngine with ViT heatmap ===")
    engine = PlacementEngine(heatmap_res=256, heatmap_model=model, enable_heatmap=True)
    scene = {
        "room_type": "bedroom",
        "bounds_bottom": [[-2, 0, 3], [2, 0, 3], [2, 0, -3], [-2, 0, -3]],
        "objects": []
    }
    pos = engine.place_object(
        scene=scene,
        top_view_path=fake_img,
        object_desc="nightstand",
        size=[0.5, 0.6, 0.4],
        rotation=[0, 0, 0, 1],
        placement_plane="floor",
        clearance=0.3,
    )
    print(f"Placed nightstand at: {pos}")
    assert pos is not None, "Should return a position"
    assert len(pos) == 3, "Position should be [x, y, z]"

    os.remove(fake_img)
    print("\n*** All ViT pipeline tests PASSED! ***")


if __name__ == '__main__':
    test_vit_pipeline()
