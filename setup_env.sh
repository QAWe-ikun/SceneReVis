#!/bin/bash
# Set environment variables required for asset retrieval

# Project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 3D资产目录
export PTH_3DFUTURE_ASSETS="/mnt/d/3D-Dataset/3D-FUTURE-model"

# Metadata files
export PTH_ASSETS_METADATA="${PROJECT_ROOT}/metadata/model_info_3dfuture_assets.json"
export PTH_ASSETS_METADATA_SCALED="${PROJECT_ROOT}/metadata/model_info_3dfuture_assets_scaled.json"
export PTH_ASSETS_EMBED="${PROJECT_ROOT}/metadata/model_info_3dfuture_assets_embeds.pickle"
export SCENEREVIS_SIGLIP_MODEL="${PROJECT_ROOT}/ckpt/google/siglip-so400m-patch14-384"
export SCENEREVIS_DINOV2_MODEL="${PROJECT_ROOT}/ckpt/facebook/dinov2-base"
export SCENEREVIS_MODEL="${PROJECT_ROOT}/ckpt/SceneReVis-7B"

echo "✓ Environment variables set:"
echo "  PTH_3DFUTURE_ASSETS: $PTH_3DFUTURE_ASSETS"
echo "  PTH_ASSETS_METADATA: $PTH_ASSETS_METADATA"
echo "  PTH_ASSETS_METADATA_SCALED: $PTH_ASSETS_METADATA_SCALED"
echo "  PTH_ASSETS_EMBED: $PTH_ASSETS_EMBED"
echo "  SCENEREVIS_SIGLIP_MODEL: $SCENEREVIS_SIGLIP_MODEL"
echo "  SCENEREVIS_DINOV2_MODEL: $SCENEREVIS_DINOV2_MODEL"
echo "  SCENEREVIS_MODEL: $SCENEREVIS_MODEL"
echo ""
echo "Verifying file existence..."

if [ -d "$PTH_3DFUTURE_ASSETS" ]; then
    echo "  ✓ 3D asset directory exists"
else
    echo "  ✗ 3D asset directory does not exist: $PTH_3DFUTURE_ASSETS"
fi

if [ -f "$PTH_ASSETS_METADATA" ]; then
    echo "  ✓ Asset metadata file exists"
else
    echo "  ✗ Asset metadata file does not exist: $PTH_ASSETS_METADATA"
fi

if [ -f "$PTH_ASSETS_METADATA_SCALED" ]; then
    echo "  ✓ Scaled asset metadata file exists"
else
    echo "  ✗ Scaled asset metadata file does not exist: $PTH_ASSETS_METADATA_SCALED"
fi

if [ -f "$PTH_ASSETS_EMBED" ]; then
    echo "  ✓ Asset embedding file exists"
else
    echo "  ✗ Asset embedding file does not exist: $PTH_ASSETS_EMBED"
fi

if [ -d "$SCENEREVIS_SIGLIP_MODEL" ]; then
    echo "  ✓ SigLIP model directory exists"
else
    echo "  ✗ SigLIP model directory does not exist: $SCENEREVIS_SIGLIP_MODEL"
fi

if [ -d "$SCENEREVIS_DINOV2_MODEL" ]; then
    echo "  DINOv2 model directory exists"
else
    echo "  DINOv2 model directory does not exist: $SCENEREVIS_DINOV2_MODEL"
fi

if [ -d "$SCENEREVIS_MODEL" ]; then
    echo "  SceneReVis model directory exists"
else
    echo "  SceneReVis model directory does not exist: $SCENEREVIS_MODEL"
fi

echo ""
echo "Environment setup complete! You can now run tests."
