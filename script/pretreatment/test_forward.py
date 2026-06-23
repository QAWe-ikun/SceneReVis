"""
测试 PlacementHeatmap 模型前向传播
验证 SigLIP + CLIP 架构是否正常工作
"""
import sys
from pathlib import Path

# 添加项目根目录到 path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
from utils.placement_heatmap import PlacementHeatmap


def test_model_forward():
    """测试模型前向传播"""
    print("=" * 60)
    print("测试 PlacementHeatmap 模型")
    print("=" * 60)

    # 初始化模型
    print("\n[1/4] 初始化模型...")
    model = PlacementHeatmap()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"  ✓ 模型初始化成功 (device={device})")

    # 创建测试输入
    print("\n[2/4] 创建测试输入...")

    # 创建临时测试图像
    import tempfile
    from PIL import Image
    import numpy as np

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # 房间俯视图 (1024x1024 RGB)
        room_image_path = tmpdir / "room.png"
        room_img = Image.fromarray(np.random.randint(0, 256, (1024, 1024, 3), dtype=np.uint8))
        room_img.save(room_image_path)
        print(f"  ✓ 房间图像: {room_image_path}")

        # 物体参考图 (1024x1024 RGB)
        object_image_path = tmpdir / "object.png"
        object_img = Image.fromarray(np.random.randint(0, 256, (1024, 1024, 3), dtype=np.uint8))
        object_img.save(object_image_path)
        print(f"  ✓ 物体图像: {object_image_path}")

        # 文本描述
        object_desc = "a wooden chair with armrests"
        print(f"  ✓ 文本描述: '{object_desc}'")

        # 前向传播
        print("\n[3/4] 执行前向传播...")
        model.eval()
        room_tensor = model.preprocess_room_image(
            Image.open(room_image_path).convert("RGB")
        ).unsqueeze(0).to(device)
        object_tensor = model.preprocess_object_image(
            Image.open(object_image_path).convert("RGB")
        ).unsqueeze(0).to(device)
        with torch.no_grad():
            heatmap = model.forward_tensor(room_tensor, object_desc, object_tensor)[0]
        print(f"  ✓ 前向传播成功")

        # 验证输出
        print("\n[4/4] 验证输出...")
        print(f"  输出形状: {heatmap.shape}")
        print(f"  输出范围: [{heatmap.min():.4f}, {heatmap.max():.4f}]")
        print(f"  输出类型: {heatmap.dtype}")

        assert heatmap.shape == (256, 256), f"期望形状 (256, 256), 实际 {heatmap.shape}"
        assert heatmap.min() >= 0, f"热力图最小值 < 0: {heatmap.min()}"
        assert heatmap.max() <= 1, f"热力图最大值 > 1: {heatmap.max()}"
        print(f"  ✓ 输出验证通过")

    print("\n" + "=" * 60)
    print("✓ 模型测试通过!")
    print("=" * 60)


def test_batch_forward():
    """测试批量前向传播"""
    print("\n" + "=" * 60)
    print("测试批量前向传播")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PlacementHeatmap().to(device)

    import tempfile
    from PIL import Image
    import numpy as np

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        batch_size = 2

        room_paths = []
        object_paths = []
        descs = []

        for i in range(batch_size):
            room_path = tmpdir / f"room_{i}.png"
            Image.fromarray(np.random.randint(0, 256, (1024, 1024, 3), dtype=np.uint8)).save(room_path)
            room_paths.append(str(room_path))

            object_path = tmpdir / f"object_{i}.png"
            Image.fromarray(np.random.randint(0, 256, (1024, 1024, 3), dtype=np.uint8)).save(object_path)
            object_paths.append(str(object_path))

            descs.append(f"object description {i}")

        print(f"\n批量大小: {batch_size}")

        # 逐个处理 (当前实现)
        print("逐个处理样本...")
        heatmaps = []
        for room_path, object_path, desc in zip(room_paths, object_paths, descs):
            room_tensor = model.preprocess_room_image(
                Image.open(room_path).convert("RGB")
            ).unsqueeze(0).to(device)
            object_tensor = model.preprocess_object_image(
                Image.open(object_path).convert("RGB")
            ).unsqueeze(0).to(device)
            with torch.no_grad():
                heatmap = model.forward_tensor(room_tensor, desc, object_tensor)
                heatmaps.append(heatmap)

        heatmaps = torch.cat(heatmaps, dim=0)
        print(f"  输出形状: {heatmaps.shape}")
        print(f"  ✓ 批量处理成功")

    print("=" * 60)


if __name__ == "__main__":
    try:
        test_model_forward()
        test_batch_forward()
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
