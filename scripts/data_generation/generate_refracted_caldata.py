"""
generate_refracted_caldata.py
==============================

STEP 3 of the Option-3 validation chain.

PURPOSE
-------
Take the ground-truth geometry from Step 2 (caldata20130726 with cameras
flipped to be above the glass plane, points reflected below it, scaled to
millimetres) and synthesise 2-D pixel measurements that are CONSISTENT
WITH REFRACTION at an air–glass–water interface. The output is a complete
dataset that can be fed to the multimedia pipeline.

If the multimedia BA is implemented correctly, it should recover the
original (Step-2) camera positions and intrinsics to within a fraction of
a millimetre, because the synthesis uses the same Maas (1996) radial
shift that the BA's `observation_equation_221` uses internally.

The only differences between Step 2 and Step 3 are:
    1. The 2-D measurements are now refracted (we project the radially-
       shifted point P̄ instead of the original P).
    2. The cfg file enables the multimedia path (adds [Multimedia] section).

The geometry, intrinsics, distortion, and measurement noise are identical.

DESIGN NOTE
-----------
To avoid sign-convention issues between the OpenCV-style look-at
rotations used in Step 2 and the photogrammetric convention of
collinearity_condition, we use multimedia_radial_shift to compute
P̄ (which is orientation-independent) and then project P̄ using the
SAME pinhole formula as Step 2. The forward synthesis is therefore
fully consistent with Step 2's projection, and the only "extra"
ingredient is the radial shift.

USAGE
-----
The script reads the ground_truth.json written by generate_flipped_caldata.py
to recover the true camera parameters and 3-D point positions::

    python generate_refracted_caldata.py <step2_folder> <step3_folder>

Example::

    python generate_refracted_caldata.py \
        C:\\...\\TestSystem\\caldata_flipped_step2 \
        C:\\...\\TestSystem\\caldata_refracted_step3
"""

from __future__ import annotations
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as Rot

# Multimedia model imports — we re-use the EXACT functions the pipeline uses,
# so the synthesis is consistent with the inversion.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
try:
    from Multimedia.Multimedia_params import MultimediaParams, CameraParams
    from Multimedia.Iterative_radial_shift import multimedia_radial_shift
except ImportError:
    print("ERROR: Could not import Multimedia module.")
    print("       Run this script from the Muticameracalb root directory")
    print("       or add it to PYTHONPATH.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Multimedia parameters (matching what we will write into the cfg)
# ---------------------------------------------------------------------------

N1                  = 1.0      # air
N2                  = 1.46     # glass
N3                  = 1.33     # water
GLASS_THICKNESS_MM  = 5.0      # glass plane sits at z=0 down to z=-5
Z_GLASS_TOP         = 0.0
NOISE_STD_PIXELS    = 0.5      # same as Step 2

# Same distortion params as Step 2 (must match exactly so Step 2 vs Step 3
# differ ONLY by refraction)
RADIAL_K1           = 0.0
RADIAL_K2           = 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rotation_matrix_to_xyz_euler(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to (omega, phi, kappa) Euler angles
    in the 'xyz' (extrinsic) convention used by the pipeline."""
    return Rot.from_matrix(R).as_euler('xyz')


def build_camera_params_for_radial_shift(cam_id: int,
                                          position_mm: np.ndarray,
                                          mm_params: MultimediaParams) -> CameraParams:
    """Build a minimal CameraParams just for use with multimedia_radial_shift.

    The radial shift only depends on the camera position relative to the
    glass plane and the particle position — orientation and intrinsics
    are NOT used.  So we leave them at defaults (zero angles, c=1, etc.).
    """
    cam = CameraParams(
        cam_id   = cam_id,
        position = position_mm.copy(),
        angles   = np.zeros(3),
        c        = 1.0,
        xh       = 0.0,
        yh       = 0.0,
        mm       = mm_params,
    )
    return cam


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('step2_folder',
                    help='Step 2 output folder (must contain ground_truth.json)')
    ap.add_argument('output',
                    help='Step 3 output folder (will be created)')
    ap.add_argument('--noise-std', type=float, default=NOISE_STD_PIXELS,
                    help=f'image noise in px (default {NOISE_STD_PIXELS}); 0 = noise-free')
    ap.add_argument('--noise-seed', type=int, default=123,
                    help='seed of the 0.5 px image-noise draw (default 123 = the original datasets); '
                         'use different seeds to build datasets for an accuracy-spread study')
    args = ap.parse_args()

    in_dir  = Path(args.step2_folder).resolve()
    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nStep 3 generator — flipped + scaled + REFRACTED")
    print(f"  Input : {in_dir}")
    print(f"  Output: {out_dir}\n")

    # --- Load the Step-2 ground truth ---------------------------------
    gt_path = in_dir / 'ground_truth.json'
    if not gt_path.exists():
        sys.exit(f"ERROR: {gt_path} not found.")
    gt = json.load(open(gt_path))
    n_cams = len(gt['cameras'])
    print(f"Loaded ground truth: {n_cams} cameras, "
          f"{len(gt['particles_mm'])} particles")

    # --- Reconstruct camera data from the JSON ---------------------------
    # We need:
    #   - K, R, C per camera for the standard pinhole projection
    #     (uses the SAME formula as the Step 2 generator, ensuring
    #     pixel values are consistent between Steps 2 and 3 modulo
    #     the refractive shift on the world point).
    #   - A minimal CameraParams per camera just for multimedia_radial_shift
    #     (which only uses cam.position and cam.mm, not orientation).
    mm_params = MultimediaParams(n1=N1, n2=[N2], d=[GLASS_THICKNESS_MM], n3=N3)
    K_list, R_list, C_list = [], [], []
    cam_radial_objs = []
    for cam_dict in gt['cameras']:
        position = np.array(cam_dict['position_mm'])
        R        = np.array(cam_dict['rotation_matrix'])
        intr     = cam_dict['intrinsics']
        K = np.array([[intr['fx'], 0,           intr['cx']],
                      [0,          intr['fy'],  intr['cy']],
                      [0,          0,           1.0]])
        K_list.append(K)
        R_list.append(R)
        C_list.append(position)
        cam_radial = build_camera_params_for_radial_shift(
            cam_dict['id'] - 1, position, mm_params)
        cam_radial_objs.append(cam_radial)
        print(f"  Cam {cam_dict['id']}: "
              f"pos=({position[0]:+5.1f},{position[1]:+5.1f},{position[2]:+5.1f}), "
              f"fx={K[0,0]:.1f}")

    # --- Load image resolution ----------------------------------------
    res_path = in_dir / 'Res.dat'
    if res_path.exists():
        res = np.loadtxt(res_path, dtype=int)
        if res.ndim == 1:
            res = res[None, :]
        img_w, img_h = int(res[0, 0]), int(res[0, 1])
    else:
        img_w, img_h = 752, 480
    print(f"  Image size: {img_w} x {img_h}")

    # --- Particles ----------------------------------------------------
    Xw = np.array(gt['particles_mm'])
    M = Xw.shape[0]
    print(f"\nParticles: {M}")
    print(f"  X range: [{Xw[:,0].min():+.1f}, {Xw[:,0].max():+.1f}] mm")
    print(f"  Y range: [{Xw[:,1].min():+.1f}, {Xw[:,1].max():+.1f}] mm")
    print(f"  Z range: [{Xw[:,2].min():+.1f}, {Xw[:,2].max():+.1f}] mm")

    # --- Geometric pre-checks -----------------------------------------
    print(f"\nVerifying geometric preconditions for Maas model:")
    glass_bottom = Z_GLASS_TOP - GLASS_THICKNESS_MM
    cams_above = sum(1 for c in cam_radial_objs if c.position[2] > Z_GLASS_TOP)
    pts_below = int((Xw[:, 2] < glass_bottom).sum())
    print(f"  Cameras above glass top (z>{Z_GLASS_TOP}): {cams_above}/{n_cams}")
    print(f"  Particles below glass bottom (z<{glass_bottom}): {pts_below}/{M}")
    if cams_above < n_cams:
        sys.exit("ERROR: not all cameras above glass — cannot apply refraction.")
    if pts_below < M:
        print(f"  WARNING: {M-pts_below} particles are not below glass bottom; "
              f"these will be skipped.")

    # --- Forward refractive projection ---------------------------------
    print(f"\nProjecting through refractive interface "
          f"(n1={N1}, n2={N2}, d={GLASS_THICKNESS_MM} mm, n3={N3})...")
    points = np.full((3 * n_cams, M), np.nan)
    vis = np.zeros((n_cams, M), dtype=int)
    rng = np.random.default_rng(args.noise_seed)
    print(f"  image noise: {args.noise_std} px, noise seed {args.noise_seed}")
    failed = 0
    skipped_geometry = 0

    for cam_idx in range(n_cams):
        cam_radial = cam_radial_objs[cam_idx]
        K = K_list[cam_idx]
        R = R_list[cam_idx]
        C = C_list[cam_idx]

        on_image = 0
        for j in range(M):
            P = Xw[j]
            # Skip if not below glass bottom (geometric precondition)
            if P[2] >= glass_bottom:
                skipped_geometry += 1
                continue

            # 1. Compute the radially-shifted point P_bar using the Maas model.
            #    The shift only depends on camera position relative to glass
            #    and the particle position — it is independent of orientation.
            try:
                P_bar, n_iter, dR = multimedia_radial_shift(
                    P, cam_radial, eps=1e-6, max_iter=100, ocf=1.3)
            except Exception:
                failed += 1
                continue

            # 2. Project P_bar through the STANDARD PINHOLE camera (same
            #    convention as Step 2). The only difference between Step 2
            #    and Step 3 is that here we project the SHIFTED point.
            P_cam = R @ (P_bar - C)
            if P_cam[2] <= 0:
                failed += 1
                continue
            x_n = P_cam[0] / P_cam[2]
            y_n = P_cam[1] / P_cam[2]
            # Same mild radial distortion as Step 2
            r2 = x_n*x_n + y_n*y_n
            factor = 1.0 + RADIAL_K1 * r2 + RADIAL_K2 * r2 * r2
            x_n *= factor
            y_n *= factor
            u = K[0, 0] * x_n + K[0, 2]
            v = K[1, 1] * y_n + K[1, 2]

            # Add measurement noise (same as Step 2)
            u += rng.normal(0, args.noise_std)
            v += rng.normal(0, args.noise_std)

            # Check on-image
            if not (0 <= u <= img_w and 0 <= v <= img_h):
                continue

            points[3*cam_idx,     j] = u
            points[3*cam_idx + 1, j] = v
            points[3*cam_idx + 2, j] = 1.0
            vis[cam_idx, j] = 1
            on_image += 1

        print(f"  Cam {cam_idx+1}: {on_image} points on-image")

    if failed > 0:
        print(f"\n  WARNING: {failed} (camera, point) pairs failed refraction "
              f"(degenerate rays / total internal reflection).")
    if skipped_geometry > 0:
        print(f"  Skipped {skipped_geometry} (cam, pt) pairs above glass bottom.")

    vis_in_all = int((vis.sum(axis=0) == n_cams).sum())
    vis_in_3   = int((vis.sum(axis=0) >= 3).sum())
    print(f"\nVisibility summary:")
    print(f"  Visible in all {n_cams} cameras: {vis_in_all}")
    print(f"  Visible in ≥ 3 cameras       : {vis_in_3}")

    # --- Sanity check: compare Step 2 vs Step 3 measurements ---------
    # Read Step 2's points.dat and compute mean refractive shift
    s2_points_path = in_dir / 'points.dat'
    if s2_points_path.exists():
        s2_points = np.loadtxt(s2_points_path)
        shifts = []
        for cam_idx in range(n_cams):
            for j in range(M):
                u_s2 = s2_points[3*cam_idx, j]
                v_s2 = s2_points[3*cam_idx + 1, j]
                u_s3 = points[3*cam_idx, j]
                v_s3 = points[3*cam_idx + 1, j]
                if not np.isnan(u_s2) and not np.isnan(u_s3):
                    shifts.append(np.linalg.norm([u_s3-u_s2, v_s3-v_s2]))
        if shifts:
            shifts = np.array(shifts)
            print(f"\nRefractive shift (Step 3 pixel - Step 2 pixel):")
            print(f"  Mean:    {shifts.mean():.2f} px")
            print(f"  Median:  {np.median(shifts):.2f} px")
            print(f"  Max:     {shifts.max():.2f} px")
            print(f"  Min:     {shifts.min():.2f} px")
            print(f"  (Shift should be NON-ZERO; this confirms the refractive")
            print(f"   model is being applied. Magnitude depends on geometry.)")

    # --- Write output files -------------------------------------------
    print(f"\nWriting output files...")
    np.savetxt(out_dir / 'points.dat', points, fmt='%.6f')
    np.savetxt(out_dir / 'IdMat.dat',  vis,    fmt='%d')
    np.savetxt(out_dir / 'Res.dat',
                np.tile([img_w, img_h], (n_cams, 1)), fmt='%d')

    # Write fresh .rad files using the TRUE intrinsics from ground_truth.json,
    # not the Step 2 rad files (which may have been overwritten by GoCal's
    # goradf step at the end of the Step 2 run).
    for cam_idx in range(1, n_cams + 1):
        K = K_list[cam_idx - 1]
        rad_path = out_dir / f'cam{cam_idx}.rad'
        with open(rad_path, 'w') as fh:
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

    # Determine the camera-to-glass distance (mean camera Z above glass)
    cam_zs = np.array([C[2] for C in C_list])
    cgd = float(cam_zs.mean() - Z_GLASS_TOP)

    # cfg — WITH [Multimedia] section
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
        fh.write("Align-Existing: ground_truth.json\n\n")
        # The multimedia section — this is what makes Step 3 different
        fh.write("[Multimedia]\n")
        fh.write(f"N1: {N1}\n")
        fh.write(f"N2: {N2}\n")
        fh.write(f"D:  {GLASS_THICKNESS_MM}\n")
        fh.write(f"N3: {N3}\n")
        fh.write(f"Z-Glass-Top: {Z_GLASS_TOP}\n")
        fh.write(f"Camera-Glass-Distance: {cgd:.1f}\n")

    # ground_truth.json — copy from Step 2 + record multimedia params
    gt_out = dict(gt)
    gt_out['source'] = (gt.get('source', '') +
                          ' + refracted using Maas (1996) multimedia model')
    gt_out['image_noise'] = {'std_px': args.noise_std, 'seed': args.noise_seed}
    gt_out['multimedia'] = {
        'n1': N1, 'n2': N2, 'n3': N3,
        'glass_thickness_mm': GLASS_THICKNESS_MM,
        'z_glass_top':        Z_GLASS_TOP,
    }
    with open(out_dir / 'ground_truth.json', 'w') as fh:
        json.dump(gt_out, fh, indent=2)

    print(f"\n+ Output written to: {out_dir}")
    print(f"\nNext step:")
    print(f"  Point CalibrationSystem at this folder and run GoCal.py.")
    print(f"  This time the multimedia path WILL activate.")
    print(f"\n  Expected:")
    print(f"  - 2D reproj error: sub-pixel (~0.5 px from noise floor)")
    print(f"  - Multimedia BA pre-validation: PASSES")
    print(f"  - Alignment error: comparable to Step 2 (40-50 mm probably,")
    print(f"    limited by goradf rather than the multimedia BA)")


if __name__ == '__main__':
    main()
