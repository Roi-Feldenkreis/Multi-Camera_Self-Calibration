#!/usr/bin/env python3
"""
penalty_isolation_test.py
-------------------------
Decisive, front-end-free test of the inter-camera distance penalty.

It removes RANSAC, the projective factorisation, euclidize, and the de-refraction
bootstrap entirely, and asks one question: given a scale-ambiguous starting point
(cameras + cloud scaled away from truth), does the multimedia bundle adjustment —
WITH the six known inter-camera distances — recover the true scale, where WITHOUT
them it cannot?

How it stays convention-safe:
  * Camera orientations are found numerically (each camera is aimed at the cloud
    centroid), so it adapts to whatever Euler convention CameraParams uses.
  * Synthetic image observations are produced by the BA's OWN observation model
    (observation_equation_221 called with a zero observation returns the predicted
    pixel), so the observations are guaranteed consistent with the optimiser.

Run from the project root:
    python penalty_isolation_test.py
"""
import numpy as np
from scipy.optimize import least_squares

import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))  # repo root on path
from Multimedia.Multimedia_params import CameraParams, MultimediaParams
from Multimedia.Iterative_radial_shift import observation_equation_221
from Multimedia.multimedia_bundle import multimedia_bundle_adjustment

# ----------------------------------------------------------------------------
# scene constants (roughly the Step-3 rig)
# ----------------------------------------------------------------------------
Z_GLASS_TOP = 0.0
MM = MultimediaParams(n1=1.0, n2=[1.46], d=[5.0], n3=1.33)
FOCAL = 600.0

# true cameras: four above the glass, ring layout (positions in mm)
TRUE_CAM_POS = np.array([
    [-180.0, -160.0, 406.0],
    [ 175.0, -150.0, 402.0],
    [ 168.0,  170.0, 398.0],
    [-172.0,  165.0, 404.0],
], dtype=float)
K = len(TRUE_CAM_POS)

# true cloud: particles in the water, below the glass
rng = np.random.default_rng(0)
N = 60
CLOUD = np.column_stack([
    rng.uniform(-90, 90, N),
    rng.uniform(-90, 90, N),
    rng.uniform(-190, -40, N),   # below glass (z<0)
])


def make_cam(cam_id, position, angles):
    return CameraParams(
        cam_id=cam_id, position=np.asarray(position, float),
        angles=np.asarray(angles, float),
        c=FOCAL, xh=0.0, yh=0.0,
        k1=0.0, k2=0.0, k3=0.0, p1=0.0, p2=0.0, a1=0.0, a2=0.0, mm=MM,
    )


def predicted_pixel(cam, P):
    """Forward pixel from the BA's own model: residual at obs=0 IS the prediction
    (sign auto-detected by the caller)."""
    vx, vy, _ = observation_equation_221(np.asarray(P, float), cam, 0.0, 0.0)
    return np.array([vx, vy])


def aim_camera(cam_id, position, target):
    """Find angles that point the camera at `target` (cloud centroid), adapting to
    the CameraParams Euler convention by minimising the centroid's image offset."""
    def resid(ang):
        cam = make_cam(cam_id, position, ang)
        try:
            return predicted_pixel(cam, target)
        except Exception:
            return np.array([1e4, 1e4])
    best = None
    for guess in (np.zeros(3), np.array([np.pi, 0, 0]), np.array([0, np.pi, 0])):
        sol = least_squares(resid, guess, max_nfev=400)
        if best is None or np.sum(sol.fun**2) < np.sum(best.fun**2):
            best = sol
    return best.x


def build_observations(cams, cloud, sign):
    """cam-structure list with xgt / recandvis / visandrec, valid points only."""
    cam = []
    ok_counts = []
    for k in range(K):
        us, cols_xe, cols_xg = [], [], []
        c = 0
        for n in range(cloud.shape[0]):
            try:
                px = sign * predicted_pixel(cams[k], cloud[n])
                if np.all(np.isfinite(px)) and np.abs(px).max() < 5e4:
                    us.append(px); cols_xe.append(n); cols_xg.append(c); c += 1
            except Exception:
                pass
        xgt = np.array(us).T if us else np.zeros((2, 0))
        cam.append({'cam_id': k, 'xgt': xgt,
                    'recandvis': np.array(cols_xe, int),
                    'visandrec': np.array(cols_xg, int)})
        ok_counts.append(c)
    return cam, ok_counts


def cam_pos_error(cams, true_pos, fit_scale):
    """RMS camera-position error after aligning by rotation+translation
    (and optionally scale). If fit_scale=False, a wrong overall scale shows up as
    large error — that is how we detect whether scale was recovered."""
    P = np.array([c.position for c in cams]); T = true_pos.copy()
    Pc = P - P.mean(0); Tc = T - T.mean(0)
    H = Pc.T @ Tc
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1; R = Vt.T @ U.T
    s = (S.sum() / (Pc**2).sum()) if fit_scale else 1.0
    aligned = (s * (Pc @ R.T))
    return float(np.sqrt(np.mean(np.sum((aligned - Tc)**2, 1))))


def main():
    centroid = CLOUD.mean(0)
    true_cams = [make_cam(k, TRUE_CAM_POS[k], aim_camera(k, TRUE_CAM_POS[k], centroid))
                 for k in range(K)]

    # detect the sign convention of observation_equation_221 (pred = +res or -res)
    p = predicted_pixel(true_cams[0], CLOUD[0])
    vx, vy, _ = observation_equation_221(CLOUD[0], true_cams[0], p[0], p[1])
    sign = 1.0 if abs(vx) + abs(vy) < abs(p[0]) + abs(p[1]) else -1.0

    cam, ok = build_observations(true_cams, CLOUD, sign)
    total_obs = sum(ok)
    print(f"scene: {K} cameras, {N} particles, {total_obs} valid observations "
          f"(per-cam {ok})")
    if total_obs < 3 * K:
        print("!! too few valid observations — camera aiming/convention issue; "
              "check CameraParams angle convention."); return

    # known inter-camera distances = true + hand-measurement noise (±4 mm)
    pairs = [(i, j) for i in range(K) for j in range(i + 1, K)]
    known = [(i, j, float(np.linalg.norm(TRUE_CAM_POS[i] - TRUE_CAM_POS[j])
                          + rng.normal(0, 4.0))) for (i, j) in pairs]

    # scale-ambiguous seed: scale cameras + cloud by 1.5 about the cloud centroid
    S = 1.5
    seed_pos = centroid + S * (TRUE_CAM_POS - centroid)
    seed_cloud = centroid + S * (CLOUD - centroid)
    seed_cams = [make_cam(k, seed_pos[k], true_cams[k].angles.copy()) for k in range(K)]
    Xe = np.vstack([seed_cloud.T, np.ones(N)])

    print(f"\nseed scaled by {S}x about the cloud centroid")
    print(f"  camera scale error at seed (no-scale align): "
          f"{cam_pos_error(seed_cams, TRUE_CAM_POS, fit_scale=False):7.1f} mm")

    common = dict(cam=cam, z_glass_top=Z_GLASS_TOP)
    base_opt = dict(fix_points=False, max_nfev=200, ftol=1e-4, xtol=1e-4, gtol=1e-4,
                    max_pos_step_mm=400.0, max_pt_step_mm=400.0,
                    allow_invalid_geometry=True, verbose=False)

    # --- WITHOUT penalty ---
    c_no, X_no, i_no = multimedia_bundle_adjustment(
        cameras_mm=[make_cam(k, seed_pos[k], seed_cams[k].angles.copy()) for k in range(K)],
        Xe=Xe.copy(), opt={**base_opt}, **common)
    # --- WITH penalty ---
    c_pen, X_pen, i_pen = multimedia_bundle_adjustment(
        cameras_mm=[make_cam(k, seed_pos[k], seed_cams[k].angles.copy()) for k in range(K)],
        Xe=Xe.copy(), opt={**base_opt, 'cam_dist': known, 'cam_dist_weight': 1.0}, **common)

    print("\n                         reproj(px)   cam-scale error (mm)   dist-misfit(mm)")
    print(f"  WITHOUT distance penalty  {i_no.get('rms_after', float('nan')):8.2f}      "
          f"{cam_pos_error(c_no, TRUE_CAM_POS, fit_scale=False):8.1f}            —")
    print(f"  WITH   distance penalty   {i_pen.get('rms_after', float('nan')):8.2f}      "
          f"{cam_pos_error(c_pen, TRUE_CAM_POS, fit_scale=False):8.1f}         "
          f"{i_pen.get('cam_dist_misfit', float('nan')):6.1f}")
    print("\nInterpretation:")
    print("  If WITH-penalty camera-scale error is much smaller than WITHOUT, the")
    print("  constraint recovers scale in the real BA and the mechanism is sound —")
    print("  the remaining problem is purely feeding it a below-glass seed.")


if __name__ == "__main__":
    main()
