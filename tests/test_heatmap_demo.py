"""
Heatmap placement demo script.
Loads a real scene, computes feasibility mask, and visualizes results.
"""

import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.placement_mask import compute_mask, find_best_position, grid_to_world, world_to_grid
from utils.placement_engine import PlacementEngine


def demo_mask(scene_path, res=256, clearance=0.5):
    """Compute and print feasibility mask stats for a real scene."""
    with open(scene_path) as f:
        scene = json.load(f)

    print(f"=== Scene: {os.path.basename(scene_path)} ===")
    print(f"  Room bounds: X=[{min(p[0] for p in scene['bounds_bottom'])}, "
          f"{max(p[0] for p in scene['bounds_bottom'])}] "
          f"Z=[{min(p[2] for p in scene['bounds_bottom'])}, "
          f"{max(p[2] for p in scene['bounds_bottom'])}]")
    print(f"  Objects: {len(scene['objects'])}")
    for i, obj in enumerate(scene['objects']):
        desc = obj.get('desc', '?')[:50]
        print(f"    [{i}] {desc} pos={obj['pos']} size={obj['size']}")

    # Compute mask
    mask, ortho_scale, cx, cz = compute_mask(scene, heatmap_res=res, clearance=clearance)
    cell_size = ortho_scale / res
    total = mask.size
    feasible = mask.sum()
    infeasible = total - feasible

    print(f"\n=== Feasibility Mask ({res}x{res}) ===")
    print(f"  ortho_scale = {ortho_scale:.2f}m, cell_size = {cell_size*100:.1f}cm")
    print(f"  center (world) = ({cx:.2f}, {cz:.2f})")
    print(f"  Feasible: {feasible}/{total} ({100*feasible/total:.1f}%)")
    print(f"  Infeasible (collision+clearance): {infeasible}/{total} ({100*infeasible/total:.1f}%)")

    # Find best positions (mask only, uniform heatmap)
    positions = find_best_position(mask, ortho_scale, cx, cz, res, top_k=5, min_distance=0.5)
    print(f"\n=== Top-5 Best Positions (mask only) ===")
    for i, pos in enumerate(positions):
        print(f"  [{i}] x={pos[0]:.2f}, y={pos[1]:.2f}, z={pos[2]:.2f}")

    return mask, ortho_scale, cx, cz


def demo_engine(scene_path, res=256):
    """Run the full placement engine (uniform heatmap + mask) on a scene."""
    with open(scene_path) as f:
        scene = json.load(f)

    print(f"\n=== Placement Engine Demo (uniform heatmap) ===")

    engine = PlacementEngine(heatmap_res=res, enable_heatmap=False)

    # Test placing a nightstand on floor
    pos = engine.place_object(
        scene=scene,
        top_view_path="",  # not used when enable_heatmap=False
        object_desc="nightstand",
        size=[0.5, 0.6, 0.4],
        rotation=[0, 0, 0, 1],
        placement_plane="floor",
        clearance=0.3,
    )
    print(f"  Nightstand [0.5, 0.6, 0.4] on floor:")
    print(f"    => {pos}")

    # Test placing on top of wardrobe (placement_plane = jid)
    if scene['objects']:
        target = scene['objects'][0]
        jid = target.get('jid', '')
        pos = engine.place_object(
            scene=scene,
            top_view_path="",
            object_desc="small lamp",
            size=[0.3, 0.2, 0.3],
            rotation=[0, 0, 0, 1],
            placement_plane=jid,
            clearance=0.05,
        )
        print(f"  Small lamp [0.3, 0.2, 0.3] on wardrobe ({jid}):")
        print(f"    => {pos}")

    # Add objects one by one to show mask updating
    print(f"\n=== Incremental Placement Demo ===")
    sim_scene = json.loads(json.dumps(scene))  # deep copy

    for i, item in enumerate([
        {"desc": "queen bed", "size": [1.8, 0.8, 2.1]},
        {"desc": "nightstand", "size": [0.5, 0.6, 0.4]},
        {"desc": "desk chair", "size": [0.6, 0.8, 0.6]},
    ]):
        pos = engine.place_object(
            scene=sim_scene,
            top_view_path="",
            object_desc=item["desc"],
            size=item["size"],
            rotation=[0, 0, 0, 1],
            placement_plane="floor",
            clearance=0.3,
        )
        if pos:
            print(f"  Placed '{item['desc']}' {item['size']} => {pos}")
            sim_scene['objects'].append({
                "desc": item["desc"],
                "size": item["size"],
                "pos": pos,
                "rot": [0, 0, 0, 1],
            })
        else:
            print(f"  Could not place '{item['desc']}' — no feasible position")


def demo_visualization(scene_path, output_path=None):
    """Save a visualization PNG of the feasibility mask (no top view needed)."""
    with open(scene_path) as f:
        scene = json.load(f)

    mask, ortho_scale, cx, cz = compute_mask(scene, heatmap_res=256, clearance=0.5)
    positions = find_best_position(mask, ortho_scale, cx, cz, 256, top_k=3, min_distance=0.5)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    # mask[gi, gj] = xs[gi], zs[gj] (indexing='ij')
    # imshow: row=axis0, col=axis1 -> need to transpose to get X on horizontal
    ax.imshow(mask.T.astype(int), cmap='RdYlGn', origin='lower',
              extent=[cx - ortho_scale/2, cx + ortho_scale/2,
                      cz - ortho_scale/2, cz + ortho_scale/2])
    ax.set_title(f"Feasibility Mask (green=feasible, red=blocked)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")

    # Draw room bounds
    bounds = scene.get('bounds_bottom', [])
    if bounds:
        xs = [p[0] for p in bounds] + [bounds[0][0]]
        zs = [p[2] for p in bounds] + [bounds[0][2]]
        ax.plot(xs, zs, 'b-', linewidth=2, label='Room boundary')

    # Draw existing objects
    for obj in scene.get('objects', []):
        ax.plot(obj['pos'][0], obj['pos'][2], 'bs', markersize=8, label='Existing object')

    # Draw best positions
    for i, pos in enumerate(positions):
        ax.plot(pos[0], pos[2], 'r*', markersize=15,
               label=f'Best pos {i}' if i == 0 else '')

    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"\n=== Visualization saved to {output_path} ===")


if __name__ == '__main__':
    scene_file = os.path.join(os.path.dirname(__file__), '..', 'output', 'sft_50k_e3', 'scene_iter_1.json')

    if not os.path.exists(scene_file):
        print(f"Scene file not found: {scene_file}")
        sys.exit(1)

    demo_mask(scene_file, res=256, clearance=0.5)
    demo_engine(scene_file, res=256)

    vis_path = os.path.join(os.path.dirname(__file__), 'heatmap_demo.png')
    demo_visualization(scene_file, vis_path)

    print("\nDemo complete.")
