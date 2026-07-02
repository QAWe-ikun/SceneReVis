"""Placement engine that combines a heatmap model with feasibility masks."""

from __future__ import annotations

from typing import Optional

import torch
from PIL import Image


class PlacementEngine:
    """High-level placement engine: score = heatmap * feasibility mask."""

    def __init__(
        self,
        heatmap_res: int = 256,
        heatmap_model: Optional[torch.nn.Module] = None,
        enable_heatmap: bool = True,
    ):
        self.heatmap_res = heatmap_res
        self.enable_heatmap = enable_heatmap
        self.heatmap_model = heatmap_model

    def _preprocess_image(self, image_path: str, kind: str = "room") -> torch.Tensor:
        if self.heatmap_model is None:
            raise ValueError("heatmap_model is required for image preprocessing")
        device = next(self.heatmap_model.parameters()).device
        image = Image.open(image_path).convert("RGB")
        if kind == "room" and hasattr(self.heatmap_model, "preprocess_room_image"):
            tensor = self.heatmap_model.preprocess_room_image(image)
        elif kind == "object" and hasattr(self.heatmap_model, "preprocess_object_image"):
            tensor = self.heatmap_model.preprocess_object_image(image)
        else:
            tensor = self.heatmap_model.vision_encoder.preprocess(image)
        return tensor.unsqueeze(0).to(device)

    def compute_heatmap(
        self,
        room_image_path: str,
        object_desc: str,
        object_image_path: str,
    ) -> torch.Tensor:
        if not self.enable_heatmap or self.heatmap_model is None:
            return torch.ones(self.heatmap_res, self.heatmap_res)

        room_tensor = self._preprocess_image(room_image_path, kind="room")
        object_tensor = self._preprocess_image(object_image_path, kind="object")

        self.heatmap_model.eval()
        with torch.no_grad():
            heatmap = self.heatmap_model.forward_tensor(
                room_image=room_tensor,
                object_desc=object_desc,
                object_image=object_tensor,
            )[0]

        max_val = heatmap.max()
        if max_val > 0:
            heatmap = heatmap / max_val
        return heatmap

    def place_object(
        self,
        scene: dict,
        top_view_path: str,
        object_desc: str,
        size: list,
        rotation: list = None,
        placement_plane: str = "floor",
        clearance: float = 0.5,
        object_image_path: Optional[str] = None,
    ) -> Optional[list]:
        from utils.placement_mask import compute_mask

        mask, ortho_scale, cx, cz = compute_mask(
            scene, self.heatmap_res, clearance, placement_plane
        )

        if not mask.any():
            print(f"[placement] No feasible position for '{object_desc}'")
            return None

        if object_image_path is not None:
            heatmap = self.compute_heatmap(top_view_path, object_desc, object_image_path)
        else:
            heatmap = torch.ones(self.heatmap_res, self.heatmap_res)

        mask_tensor = torch.from_numpy(mask).to(device=heatmap.device, dtype=heatmap.dtype)
        score = heatmap * mask_tensor

        if not torch.any(score > 0):
            score = mask_tensor

        positions = find_best_position_from_score(
            score=score,
            ortho_scale=ortho_scale,
            cx=cx,
            cz=cz,
            heatmap_res=self.heatmap_res,
            placement_plane=placement_plane,
            scene=scene,
            top_k=1,
        )

        if not positions:
            print(f"[placement] Failed to find position for '{object_desc}'")
            return None

        # Store heatmap data for external visualization
        self.last_heatmap = heatmap
        self.last_score = score
        self.last_mask = mask
        self.last_ortho_scale = ortho_scale
        self.last_cx = cx
        self.last_cz = cz
        self.last_top_view_path = top_view_path
        self.last_object_desc = object_desc

        print(f"[placement] Optimal position for '{object_desc}': {positions[0]}")
        return positions[0]


def find_best_position_from_score(
    score: torch.Tensor,
    ortho_scale: float,
    cx: float,
    cz: float,
    heatmap_res: int,
    placement_plane: str = "floor",
    scene: Optional[dict] = None,
    top_k: int = 1,
    min_distance: float = 0.5,
) -> list:
    """Extract top-k world positions from a fused score field."""
    from utils.placement_mask import _extract_objects, grid_to_world

    if not torch.any(score > 0):
        return []

    positions = []
    working_score = score.clone()

    for _ in range(top_k):
        idx = torch.argmax(working_score.flatten()).item()
        gi, gj = divmod(idx, heatmap_res)
        x, z = grid_to_world(gi, gj, cx, cz, ortho_scale, heatmap_res)

        y = 0.0
        if placement_plane != "floor" and scene:
            for obj in _extract_objects(scene):
                obj_id = obj.get("jid", "") or obj.get("uid", "")
                if obj_id == placement_plane:
                    pos = obj.get("pos", [0, 0, 0])
                    size = obj.get("size", [1, 1, 1])
                    y = pos[1] + size[1] / 2
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


def visualize_placement(
    score: torch.Tensor,
    top_view_path: str,
    positions: list,
    object_desc: str,
    ortho_scale: float,
    cx: float,
    cz: float,
    heatmap_res: int,
    save_path: str,
) -> None:
    """Create a debug visualization overlaying the score field on the top view."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    top_img = Image.open(top_view_path).convert("RGB")

    score_np = score.detach().cpu().numpy()
    score_resized = np.array(
        Image.fromarray((score_np * 255).astype(np.uint8)).resize(
            (top_img.width, top_img.height),
            Image.BILINEAR,
        )
    ) / 255.0

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(top_img)

    heatmap_rgba = plt.cm.jet(score_resized)
    heatmap_rgba[..., 3] = 0.4
    ax.imshow(heatmap_rgba)

    for i, pos in enumerate(positions):
        px, pz = pos[0], pos[2]
        u = ((px - cx) / ortho_scale + 0.5) * top_img.width
        v = ((pz - cz) / ortho_scale + 0.5) * top_img.height
        ax.plot(
            u,
            v,
            "rX",
            markersize=15,
            linewidth=2,
            label=f"#{i + 1} ({px:.2f}, {pz:.2f})" if i == 0 else f"#{i + 1}",
        )

    ax.set_title(f'Placement: "{object_desc}"\nScore = Heatmap x FeasibleMask')
    ax.legend(loc="upper right", fontsize=8)
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[placement] Visualization saved to {save_path}")
