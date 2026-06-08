"""Verify coordinate consistency: renderer <-> heatmap_generator <-> placement_mask"""
import sys
import os
sys.path.insert(0, os.getcwd())

import numpy as np

# ---- From renderer.py: actual pixel mapping (camera right=[1,0,0], up=[0,0,-1]) ----
def renderer_pixel_mapping(world_x, world_z, cx, cz, ortho_scale, image_size):
    """Actual pixel mapping from camera pose (world X -> col, world Z -> row)"""
    col = ((world_x - cx) / ortho_scale + 0.5) * image_size
    row = ((cz - world_z) / ortho_scale + 0.5) * image_size  # row 0 = max Z
    return row, col

# ---- From heatmap_generator.py ----
def heatmap_world_to_pixel(world_x, world_z, cx, cz, ortho_scale, image_size):
    pixel_col = ((world_x - cx) / ortho_scale + 0.5) * image_size
    pixel_row = ((cz - world_z) / ortho_scale + 0.5) * image_size  # row 0 = max Z
    return pixel_row, pixel_col

# ---- From placement_mask.py ----
def mask_grid_to_world(gi, gj, cx, cz, ortho_scale, heatmap_res):
    x = cx + ((gj + 0.5) / heatmap_res - 0.5) * ortho_scale
    z = cz - ((gi + 0.5) / heatmap_res - 0.5) * ortho_scale  # gi 0 = max Z
    return x, z

def mask_world_to_grid(x, z, cx, cz, ortho_scale, heatmap_res):
    gj_cont = (x - cx) / ortho_scale * heatmap_res + heatmap_res / 2 - 0.5
    gi_cont = (cz - z) / ortho_scale * heatmap_res + heatmap_res / 2 - 0.5  # z -> gi
    return int(round(gi_cont)), int(round(gj_cont))


# ---- Test setup ----
bounds_bottom = [[-3, 0, -2], [3, 0, -2], [3, 0, 4], [-3, 0, 4]]
bounds = np.array(bounds_bottom)
xs, zs = bounds[:, 0], bounds[:, 2]
cx = (xs.min() + xs.max()) / 2
cz = (zs.min() + zs.max()) / 2
span = max(xs.max() - xs.min(), zs.max() - zs.min(), 1.0)
ortho_scale = span * 1.2

print(f"Room params: cx={cx}, cz={cz}, ortho_scale={ortho_scale}")

image_size = 1024
heatmap_res = 256
all_pass = True

def check(name, condition):
    global all_pass
    status = "OK" if condition else "FAIL"
    if not condition:
        all_pass = False
    print(f"  [{status}] {name}")

print("\n" + "=" * 60)
print("TEST 1: Renderer vs HeatmapGenerator pixel mapping")
print("=" * 60)

for tx, tz in [(1.5, 2.0), (-2.0, -1.0), (0.0, 0.0), (3.0, 4.0), (-3.0, -2.0)]:
    r_row, r_col = renderer_pixel_mapping(tx, tz, cx, cz, ortho_scale, image_size)
    h_row, h_col = heatmap_world_to_pixel(tx, tz, cx, cz, ortho_scale, image_size)
    match = abs(r_row - h_row) < 0.01 and abs(r_col - h_col) < 0.01
    check(f"({tx:+.1f},{tz:+.1f}) r=({r_row:.1f},{r_col:.1f}) h=({h_row:.1f},{h_col:.1f})", match)

print("\n" + "=" * 60)
print("TEST 2: Image convention: row 0=max_Z (top), col 0=min_X (left)")
print("=" * 60)

x_max = cx + ortho_scale / 2
x_min = cx - ortho_scale / 2
z_max = cz + ortho_scale / 2
z_min = cz - ortho_scale / 2
cell = ortho_scale / heatmap_res

x_r0c0, z_r0c0 = mask_grid_to_world(0, 0, cx, cz, ortho_scale, heatmap_res)
x_rNcN, z_rNcN = mask_grid_to_world(heatmap_res-1, heatmap_res-1, cx, cz, ortho_scale, heatmap_res)

check(f"row=0 -> z={z_r0c0:.4f} (expect ~{z_max-cell/2:.4f})", z_r0c0 > cz)
check(f"row=N -> z={z_rNcN:.4f} (expect ~{z_min+cell/2:.4f})", z_rNcN < cz)
check(f"col=0 -> x={x_r0c0:.4f} (expect ~{x_min+cell/2:.4f})", x_r0c0 < cx)
check(f"col=N -> x={x_rNcN:.4f} (expect ~{x_max-cell/2:.4f})", x_rNcN > cx)

print("\n" + "=" * 60)
print("TEST 3: grid_to_world <-> world_to_grid roundtrip")
print("=" * 60)

for gi, gj in [(0, 0), (128, 128), (255, 0), (0, 255), (100, 200)]:
    x, z = mask_grid_to_world(gi, gj, cx, cz, ortho_scale, heatmap_res)
    gi2, gj2 = mask_world_to_grid(x, z, cx, cz, ortho_scale, heatmap_res)
    check(f"({gi},{gj}) -> ({x:.4f},{z:.4f}) -> ({gi2},{gj2})", gi == gi2 and gj == gj2)

print("\n" + "=" * 60)
print("TEST 4: HeatmapGen (scaled) matches PlacementMask grid")
print("=" * 60)

scale = heatmap_res / image_size
for tx, tz in [(1.5, 2.0), (-2.0, -1.0), (0.0, 0.0)]:
    h_row, h_col = heatmap_world_to_pixel(tx, tz, cx, cz, ortho_scale, image_size)
    m_gi, m_gj = mask_world_to_grid(tx, tz, cx, cz, ortho_scale, heatmap_res)
    s_row = h_row * scale
    s_col = h_col * scale
    err_r = abs(s_row - m_gi)
    err_c = abs(s_col - m_gj)
    check(f"({tx:+.1f},{tz:+.1f}) hm_scaled=({s_row:.2f},{s_col:.2f}) mask=({m_gi},{m_gj}) err=({err_r:.2f},{err_c:.2f})",
          err_r < 1.0 and err_c < 1.0)

print("\n" + "=" * 60)
print("TEST 5: Heatmap peak aligns with renderer pixel")
print("=" * 60)

def gen_heatmap(tx, tz, bounds_bottom, image_size, sigma=15.0):
    bounds = np.array(bounds_bottom)
    xs, zs = bounds[:, 0], bounds[:, 2]
    cx_ = (xs.min() + xs.max()) / 2
    cz_ = (zs.min() + zs.max()) / 2
    span_ = max(xs.max() - xs.min(), zs.max() - zs.min(), 1.0)
    os_ = span_ * 1.2
    peak_col = ((tx - cx_) / os_ + 0.5) * image_size
    peak_row = ((cz_ - tz) / os_ + 0.5) * image_size  # row 0 = max Z
    rc, cc = np.ogrid[:image_size, :image_size]
    hm = np.exp(-((rc - peak_row)**2 + (cc - peak_col)**2) / (2 * sigma**2))
    return hm / hm.max()

for tx, tz in [(1.5, 2.0), (-2.0, -1.0), (0.0, 0.0)]:
    hm = gen_heatmap(tx, tz, bounds_bottom, image_size)
    peak = np.unravel_index(np.argmax(hm), hm.shape)
    r_row, r_col = renderer_pixel_mapping(tx, tz, cx, cz, ortho_scale, image_size)
    err_r = abs(peak[0] - r_row)
    err_c = abs(peak[1] - r_col)
    check(f"({tx:+.1f},{tz:+.1f}) peak=({peak[0]},{peak[1]}) renderer=({r_row:.1f},{r_col:.1f}) err=({err_r:.2f},{err_c:.2f})",
          err_r < 1 and err_c < 1)

print("\n" + "=" * 60)
if all_pass:
    print("ALL TESTS PASSED!")
else:
    print("SOME TESTS FAILED!")
print("=" * 60)
