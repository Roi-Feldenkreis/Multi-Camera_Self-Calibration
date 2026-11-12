"""
generate_flipped_caldata.py  (v2 — clean rewrite)
==================================================

STEP 2 of the Option-3 validation chain.

PURPOSE
-------
Take the CAMERA POSITIONS and the POINT CLOUD SHAPE from a successful
caldata20130726 calibration run, reflect cameras with Z<0 through the XY
plane (so all four cameras end up above z=0), reflect all 3-D points
through the same plane (so they end up below z=0), scale the whole
configuration to physical millimetres, and re-synthesize the 2-D pixel
measurements with cleanly defined intrinsics and orientations.

The output is a self-contained calibration dataset whose geometry is
guaranteed to satisfy the multimedia BA's pre-conditions (all cameras
above the glass plane, all points below it).

This step applies NO refraction; the dataset is meant to validate that
the geometry itself self-calibrates correctly before refraction is
added in Step 3.

DESIGN CHOICES
--------------
Rather than inheriting the intrinsics and rotations recovered from the
original run by DLT (which are noisy and may have improper rotations),
we:

  * Inherit only the camera CENTRES (clean values from Ce.dat).
  * Construct each rotation matrix fresh by pointing each camera at the
    centroid of the point cloud.
  * Use a fixed K per camera (with slight per-camera variation to break
    degeneracy in the metric upgrade).

USAGE
-----
::

    python generate_flipped_caldata.py <input_run_folder> <output_dataset>
"""

from __future__ import annotations
import sys
import json
import argparse
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
SCALE_FACTOR = 200.0
IMG_W, IMG_H = 752, 480
Z_GLASS_TOP = 0.0
GLASS_THICKNESS_MM = 5.0
SAFETY_SHIFT_MM = 20.0       # ensure cameras above z=0, points below z=-5

NOMINAL_FX = 600.0
PER_CAM_FX_VARIATION_PCT = 5.0    # ±5%
PER_CAM_PP_VARIATION_PX  = 8.0    # ±8 px
NOISE_STD_PIXELS         = 0.5    # measurement noise, similar to real PTV
RADIAL_K1                = -0.05
RADIAL_K2                =  0.01


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def look_at_rotation(C: np.ndarray, target: np.ndarray,
                     up_hint: np.ndarray = None) -> np.ndarray:
    """Build a world-to-camera rotation matrix so that the camera at
    position C looks at `target`. OpenCV convention: +z forward, +y down."""
    if up_hint is None:
        up_hint = np.array([0.0, 0.0, 1.0])
    z_cam = target - C
    z_cam /= np.linalg.norm(z_cam)
    x_cam = np.cross(up_hint, z_cam)
    if np.linalg.norm(x_cam) < 1e-6:
        x_cam = np.cross(np.array([0.0, 1.0, 0.0]), z_cam)
    x_cam /= np.linalg.norm(x_cam)
    y_cam = np.cross(z_cam, x_cam)
    R = np.array([x_cam, y_cam, z_cam])
    assert abs(np.linalg.det(R) - 1.0) < 1e-6
    return R


def apply_radial_distortion(u_norm, v_norm, k1, k2):
    r2 = u_norm * u_norm + v_norm * v_norm
    factor = 1.0 + k1 * r2 + k2 * r2 * r2
    return u_norm * factor, v_norm * factor


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('input', help='caldata20130726 output folder (with Ce.dat and points4cal*.dat)')
    ap.add_argument('output', help='destination folder')
    args = ap.parse_args()

    in_dir  = Path(args.input).resolve()
    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    n_cams = 4
    print(f"\nStep 2 generator (v2) — flipped + scaled, NO refraction")
    print(f"  Input : {in_dir}")
    print(f"  Output: {out_dir}\n")

    # --- Load camera centres ----------------------------------------
    Ce_path = in_dir / 'Ce.dat'
    if not Ce_path.exists():
        sys.exit(f"ERROR: {Ce_path} not found.")
    Ce_raw = np.loadtxt(Ce_path)
    if Ce_raw.shape[0] == 4:
        C_orig = (Ce_raw[:3, :] / Ce_raw[3:4, :]).T
    else:
        C_orig = Ce_raw[:3, :].T
    print(f"Original camera centres:")
    for k in range(n_cams):
        print(f"  Cam {k+1}: ({C_orig[k,0]:+7.3f}, {C_orig[k,1]:+7.3f}, {C_orig[k,2]:+7.3f})")

    # --- Load 3D points ----------------------------------------------
    all_pts = set()
    for cam in range(1, n_cams + 1):
        pts_path = in_dir / f'points4cal{cam}.dat'
        if not pts_path.exists():
            sys.exit(f"ERROR: {pts_path} not found.")
        data = np.loadtxt(pts_path)
        for row in data:
            all_pts.add(tuple(np.round(row[:3], 6)))
    Xw_global = np.array(sorted(all_pts))
    M = Xw_global.shape[0]
    print(f"\nTotal unique 3-D points: {M}")
    print(f"  Z range: [{Xw_global[:,2].min():+.3f}, {Xw_global[:,2].max():+.3f}]")

    # === 1. Flip cameras with Z<0 ====================================
    print(f"\nFlipping cameras with Z<0...")
    C_flipped = C_orig.copy()
    flipped_cams = []
    for cam in range(n_cams):
        if C_orig[cam, 2] < 0:
            C_flipped[cam, 2] *= -1.0
            flipped_cams.append(cam + 1)
            print(f"  Cam {cam+1}: Z {C_orig[cam,2]:+.3f} -> {C_flipped[cam,2]:+.3f}")

    # === 2. Reflect points with Z>0 ==================================
    print(f"\nReflecting points with Z>0...")
    Xw_flipped = Xw_global.copy()
    above = Xw_flipped[:, 2] > 0
    Xw_flipped[above, 2] *= -1.0
    print(f"  Points reflected (Z was > 0): {int(above.sum())}/{M}")

    # === 3. Scale to mm + shift =====================================
    print(f"\nScaling by {SCALE_FACTOR}x and shifting by {SAFETY_SHIFT_MM} mm...")
    C_mm  = C_flipped * SCALE_FACTOR
    Xw_mm = Xw_flipped * SCALE_FACTOR
    C_mm[:, 2]  += SAFETY_SHIFT_MM
    Xw_mm[:, 2] -= SAFETY_SHIFT_MM
    print(f"  Camera Z range (after shift): "
          f"[{C_mm[:,2].min():.1f}, {C_mm[:,2].max():.1f}] mm")
    print(f"  Point  Z range (after shift): "
          f"[{Xw_mm[:,2].min():.1f}, {Xw_mm[:,2].max():.1f}] mm")

    # === 4. Build clean R per camera (look-at centroid) ============
    centroid = Xw_mm.mean(axis=0)
    print(f"\nPoint cloud centroid: ({centroid[0]:+.1f}, {centroid[1]:+.1f}, {centroid[2]:+.1f}) mm")
    R_new = []
    for cam in range(n_cams):
        R = look_at_rotation(C_mm[cam], centroid)
        R_new.append(R)

    # === 5. Build per-camera K ======================================
    print(f"\nBuilding per-camera K (fx +/-{PER_CAM_FX_VARIATION_PCT}%, "
          f"pp +/-{PER_CAM_PP_VARIATION_PX} px):")
    rng = np.random.default_rng(42)
    K_list = []
    for cam in range(n_cams):
        fx = NOMINAL_FX * (1 + rng.uniform(-PER_CAM_FX_VARIATION_PCT/100,
                                             PER_CAM_FX_VARIATION_PCT/100))
        cx = IMG_W / 2.0 + rng.uniform(-PER_CAM_PP_VARIATION_PX, PER_CAM_PP_VARIATION_PX)
        cy = IMG_H / 2.0 + rng.uniform(-PER_CAM_PP_VARIATION_PX, PER_CAM_PP_VARIATION_PX)
        K = np.array([[fx, 0,  cx],
                      [0,  fx, cy],
                      [0,  0,  1.0]])
        K_list.append(K)
        print(f"  Cam {cam+1}: fx={fx:7.2f}, (cx,cy)=({cx:6.2f},{cy:6.2f})")

    # === 6. Forward-project all points ===============================
    print(f"\nProjecting {M} points through {n_cams} cameras...")
    points = np.full((3 * n_cams, M), np.nan)
    vis = np.zeros((n_cams, M), dtype=int)

    for cam in range(n_cams):
        K = K_list[cam]
        R = R_new[cam]
        C = C_mm[cam]

        on_image = 0
        for j in range(M):
            X = Xw_mm[j]
            P_cam = R @ (X - C)
            if P_cam[2] <= 0:
                continue
            x_n = P_cam[0] / P_cam[2]
            y_n = P_cam[1] / P_cam[2]
            x_n, y_n = apply_radial_distortion(x_n, y_n, RADIAL_K1, RADIAL_K2)
            u = K[0, 0] * x_n + K[0, 2]
            v = K[1, 1] * y_n + K[1, 2]
            u += rng.normal(0, NOISE_STD_PIXELS)
            v += rng.normal(0, NOISE_STD_PIXELS)
            if not (0 <= u <= IMG_W and 0 <= v <= IMG_H):
                continue
            points[3*cam,     j] = u
            points[3*cam + 1, j] = v
            points[3*cam + 2, j] = 1.0
            vis[cam, j] = 1
            on_image += 1
        print(f"  Cam {cam+1}: {on_image} points on-image")

    vis_in_all = int((vis.sum(axis=0) == n_cams).sum())
    vis_in_3   = int((vis.sum(axis=0) >= 3).sum())
    print(f"\nVisibility summary:")
    print(f"  Visible in all 4 cameras: {vis_in_all}")
    print(f"  Visible in >= 3 cameras : {vis_in_3}")

    # === 7. Write output files =======================================
    print(f"\nWriting output files...")
    np.savetxt(out_dir / 'points.dat', points, fmt='%.6f')
    np.savetxt(out_dir / 'IdMat.dat', vis, fmt='%d')
    np.savetxt(out_dir / 'Res.dat', np.tile([IMG_W, IMG_H], (n_cams, 1)), fmt='%d')

    for cam in range(1, n_cams + 1):
        K = K_list[cam-1]
        with open(out_dir / f'cam{cam}.rad', 'w') as fh:
            fh.write(f"K11 = {float(K[0,0]):.16f}\n")
            fh.write(f"K12 = 0.0\n")
            fh.write(f"K13 = {float(K[0,2]):.16f}\n")
            fh.write(f"K21 = 0.0\n")
            fh.write(f"K22 = {float(K[1,1]):.16f}\n")
            fh.write(f"K23 = {float(K[1,2]):.16f}\n")
            fh.write(f"K31 = 0.0\n")
            fh.write(f"K32 = 0.0\n")
            fh.write(f"K33 = 1.0\n\n")
            fh.write(f"kc1 = 0.0\n")
            fh.write(f"kc2 = 0.0\n")
            fh.write(f"kc3 = 0.0\n")
            fh.write(f"kc4 = 0.0\n")

    with open(out_dir / 'multicamselfcal.cfg', 'w') as fh:
        fh.write("[Files]\nBasename: cam\nImage-Extension: tif\n\n")
        fh.write("[Images]\nSubpix: 0.5\n\n")
        fh.write("[Calibration]\n")
        fh.write(f"Num-Cameras: {n_cams}\n")
        fh.write("Num-Projectors: 0\n")
        fh.write("Nonlinear-Parameters: 50    0    1    0    0    0\n")
        fh.write("Nonlinear-Update: 1   0   1   0   0   0\n")
        fh.write("Initial-Tolerance: 10\n")
        fh.write("Do-Global-Iterations: 1\n")
        fh.write("Global-Iteration-Threshold: 0.5\n")
        fh.write("Global-Iteration-Max: 5\n")
        fh.write("Num-Cameras-Fill: 2\n")
        fh.write("Do-Bundle-Adjustment: 1\n")
        fh.write("Undo-Radial: 0\n")
        fh.write("Min-Points-Value: 30\n")
        fh.write("N-Tuples: 3\n")
        fh.write("Square-Pixels: 1\n")
        fh.write("Use-Nth-Frame: 1\n")
        fh.write("Align-Existing: ground_truth.json\n")

    gt = {
        'unit':    'mm',
        'source':  'caldata20130726 - cameras 3,4 flipped, points reflected, '
                   'scaled to mm, no refraction',
        'cameras': [
            {
                'id':              cam + 1,
                'position_mm':     list(map(float, C_mm[cam])),
                'rotation_matrix': R_new[cam].tolist(),
                'intrinsics': {
                    'fx': float(K_list[cam][0, 0]),
                    'fy': float(K_list[cam][1, 1]),
                    'cx': float(K_list[cam][0, 2]),
                    'cy': float(K_list[cam][1, 2]),
                },
                'was_flipped': (cam + 1) in flipped_cams,
            }
            for cam in range(n_cams)
        ],
        'particles_mm':       Xw_mm.tolist(),
        'glass':              {'z_top': Z_GLASS_TOP, 'thickness_mm': GLASS_THICKNESS_MM},
        'noise_std_pixels':   NOISE_STD_PIXELS,
        'scale_factor':       SCALE_FACTOR,
    }
    with open(out_dir / 'ground_truth.json', 'w') as fh:
        json.dump(gt, fh, indent=2)

    print(f"\n+ Output written to: {out_dir}")
    print(f"\nNext step:")
    print(f"  Point CalibrationSystem at {out_dir}")
    print(f"  Run GoCal.py (no multimedia for Step 2)")
    print(f"  Expected: 2D error sub-pixel, alignment error in single-digit mm")


if __name__ == '__main__':
    main()
