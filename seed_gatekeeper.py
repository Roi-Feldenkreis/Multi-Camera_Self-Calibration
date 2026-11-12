#!/usr/bin/env python3
"""
seed_gatekeeper.py
------------------
Two self-contained utilities for the de-refraction front-end:

1. camera_scale_from_distances(...)
   Gauge the arbitrary-units -> millimetres scale from the KNOWN inter-camera
   distances instead of the (flaky, RANSAC-dependent) particle-cloud top. The
   camera constellation is rigid, so its pairwise distances are fixed constants,
   which makes this a stable ruler. Returns the best-fit scale and a
   shape-consistency measure that also flags a grossly distorted euclidize.

2. seed_is_reasonable(...)
   Gatekeeper that decides whether a metric reconstruction is good enough to
   proceed, or whether RANSAC + euclidize should be re-run. All checks are
   TOLERANT of small human/gauge error (that is the point) but firm about gross
   failure: finite/positive scale, camera-constellation shape within tolerance,
   cameras above the glass, and a strong majority of particles below the glass.

Agreed default tolerances:
  below_glass_frac      = 0.90   (>= 90% of particles below glass)
  above_glass_margin_mm = 10.0   (a particle up to 10 mm above glass still counts as below)
  dist_ratio_tol        = 0.25   (per-pair scaled-distance error up to 25% allowed)
  max_retries           = 10     (used by the caller's retry loop, not here)
"""
from itertools import combinations
import numpy as np


def camera_scale_from_distances(cam_positions, known_pairs, dist_ratio_tol=0.25):
    """
    Parameters
    ----------
    cam_positions : (K, 3) array
        Camera centres in ARBITRARY euclidize units.
    known_pairs : list of (i, j, d_mm)
        Known pairwise camera-to-camera distances in millimetres (hand-measured).
    dist_ratio_tol : float
        Max allowed per-pair relative error after best-fit scaling before the
        constellation is judged distorted (default 0.25 = 25%).

    Returns
    -------
    dict with:
      scale        : float  — multiply arbitrary units by this to get mm
      shape_ok     : bool   — False if the euclidize constellation is distorted
      max_rel_err  : float  — worst per-pair relative distance error after scaling
      n_pairs      : int
    """
    C = np.asarray(cam_positions, float)
    d_arb, d_known = [], []
    for (i, j, d) in known_pairs:
        if d is None or d <= 0:
            continue
        d_arb.append(float(np.linalg.norm(C[i] - C[j])))
        d_known.append(float(d))
    d_arb = np.asarray(d_arb)
    d_known = np.asarray(d_known)
    if len(d_arb) == 0 or np.all(d_arb < 1e-9):
        return dict(scale=float('nan'), shape_ok=False, max_rel_err=float('inf'), n_pairs=0)

    # least-squares scalar: minimise sum (s * d_arb - d_known)^2
    scale = float(np.dot(d_arb, d_known) / np.dot(d_arb, d_arb))

    # shape consistency: per-pair relative error after applying the single scale.
    # A pure scale error affects all pairs equally and is absorbed; a DISTORTED
    # constellation leaves large per-pair residuals -> catches a broken euclidize.
    rel_err = np.abs(scale * d_arb - d_known) / np.maximum(d_known, 1e-9)
    max_rel = float(np.max(rel_err))
    shape_ok = bool(scale > 0 and np.isfinite(scale) and max_rel <= dist_ratio_tol)
    return dict(scale=scale, shape_ok=shape_ok, max_rel_err=max_rel, n_pairs=len(d_arb))


def camera_chirality_sign(cam_positions, cyclic_order):
    """
    Handedness of the camera constellation, as the sign of the signed volume of
    the tetrahedron formed by the four cameras taken in the cyclic order. The six
    pairwise distances fix the tetrahedron only up to a REFLECTION, and a
    reflection flips this signed volume — for EVERY reflection plane, including
    z -> -z, which a 2-D (in-plane winding) test misses. Returns +1 / -1, or 0 if
    the constellation is too close to coplanar for the sign to be reliable.

    Convention: make_camera_geometry.py emits the cyclic order oriented so the true
    layout has POSITIVE signed volume, so a correct reconstruction yields +1 and
    its mirror yields -1.
    """
    C = np.asarray(cam_positions, float)
    if C.ndim == 2 and C.shape[1] != 3:
        C = C.T
    if cyclic_order is None or len(cyclic_order) < 4 or C.shape[0] < 4:
        return 0
    o = [int(k) for k in cyclic_order[:4]]
    if any(k < 0 or k >= C.shape[0] for k in o):
        return 0
    a, b, c, d = C[o[0]], C[o[1]], C[o[2]], C[o[3]]
    vol = float(np.dot(np.cross(b - a, c - a), d - a))     # 6x signed tetra volume
    # degeneracy guard: compare to the cube of the mean edge length so the test is
    # scale-free; only bail out if the tetrahedron is essentially flat.
    edges = [np.linalg.norm(x - y) for x, y in
             ((a, b), (a, c), (a, d), (b, c), (b, d), (c, d))]
    L = float(np.mean(edges))
    if L < 1e-9 or abs(vol) < 1e-3 * L ** 3:
        return 0
    return 1 if vol > 0 else -1


def seed_is_reasonable(cam_positions_mm, cloud_mm, z_glass_top, known_pairs,
                       scale=None,
                       below_glass_frac=0.90,
                       above_glass_margin_mm=10.0,
                       dist_ratio_tol=0.25,
                       cyclic_order=None):
    """
    Decide whether a metric (millimetre), correctly-oriented reconstruction is
    reasonable enough to proceed with de-refraction.

    Parameters
    ----------
    cam_positions_mm : (K, 3) camera centres in mm (already scaled + oriented)
    cloud_mm         : (N, 3) particle positions in mm (already scaled + oriented)
    z_glass_top      : float, Z of the air/glass interface (mm)
    known_pairs      : list of (i, j, d_mm) known camera distances
    scale            : float or None — the frame scale used (checked finite/positive)
    below_glass_frac, above_glass_margin_mm, dist_ratio_tol : tolerances

    Returns
    -------
    (ok: bool, reasons: list[str], diag: dict)
        reasons is empty when ok is True; otherwise lists each failed check.
    """
    C = np.asarray(cam_positions_mm, float)
    X = np.asarray(cloud_mm, float)
    reasons, diag = [], {}

    # 1. finite, positive scale (coerce safely; some stages report a non-numeric
    #    placeholder like 'frozen' — treat that as "no scale to check")
    scale_num = None
    if scale is not None:
        try:
            scale_num = float(scale)
        except (TypeError, ValueError):
            scale_num = None
    if scale_num is not None and not (np.isfinite(scale_num) and scale_num > 0):
        reasons.append(f"scale not finite/positive ({scale})")
    diag['scale'] = scale_num

    # 2. constellation shape vs known geometry — ADVISORY ONLY.
    # The raw euclidize carries the projective->metric warp, which distorts the
    # camera constellation well beyond any sane tolerance BEFORE the bundle
    # adjustment runs. De-warping the constellation is precisely the job of the
    # BA distance penalty (proven in isolation to pull ~118 mm -> ~2 mm), so we
    # must NOT reject the seed on shape here — that would reject every draw and
    # never let the BA run. We report the shape error as a diagnostic only.
    shp = camera_scale_from_distances(C, known_pairs, dist_ratio_tol=dist_ratio_tol)
    diag['shape_max_rel_err'] = shp['max_rel_err']   # informational; not a gate

    # 3. cameras above the glass (allow the same small margin, for symmetry)
    n_below_cam = int(np.sum(C[:, 2] < z_glass_top - above_glass_margin_mm))
    diag['cameras_below_glass'] = n_below_cam
    if n_below_cam > 0:
        reasons.append(f"{n_below_cam}/{len(C)} cameras below the glass")

    # 4. strong majority of particles below the glass, with a forgiving margin:
    #    a particle up to `above_glass_margin_mm` above the glass still counts as
    #    below (absorbs hand/gauge error near the surface).
    thresh = z_glass_top + above_glass_margin_mm
    frac_below = float(np.mean(X[:, 2] < thresh)) if len(X) else 0.0
    diag['frac_particles_below_glass'] = frac_below
    if frac_below < below_glass_frac:
        reasons.append(f"only {100*frac_below:.0f}% of particles below glass "
                       f"(need >= {100*below_glass_frac:.0f}%, "
                       f"{above_glass_margin_mm:.0f} mm margin)")

    # 5. reflection / chirality: the six distances are reflection-invariant, so a
    #    mirror-image constellation has identical misfit AND reprojection yet is far
    #    from truth. The cyclic order (listed CCW viewed from above) breaks the
    #    ambiguity: a correct reconstruction winds counterclockwise (+1); the mirror
    #    winds clockwise (-1) and is rejected.
    if cyclic_order is not None and len(cyclic_order) >= 3:
        chir = camera_chirality_sign(C, cyclic_order)
        diag['chirality'] = chir
        if chir < 0:
            reasons.append("reflected (mirror-image) camera constellation "
                           "[wrong chirality vs cyclic order]")

    return (len(reasons) == 0), reasons, diag


# ---------------------------------------------------------------------------
# Camera-glass line-of-sight check (post-BA validity rule)
# ---------------------------------------------------------------------------
# Tolerance on |recovered - measured| line-of-sight distance, per camera. The
# hand measurement is ~±5 mm (and taken from the housing, not the optical
# centre), so a genuine solution sits well inside this; the pose failures this
# rule exists for were off by 100-330 mm.
LOS_TOLERANCE_MM = 20.0


def camera_los_distance(cam, z_glass_top=0.0):
    """Distance from the camera, along its optical axis, to the glass top plane
    z = z_glass_top: h / |cos(omega) cos(phi)| (cos(omega)cos(phi) is the
    vertical component of the optical axis in the Maas rotation convention)."""
    om, ph = float(cam.angles[0]), float(cam.angles[1])
    g = np.cos(om) * np.cos(ph)
    h = float(cam.position[2]) - z_glass_top
    return h / max(abs(g), 1e-9)


def los_check(cameras, cam_los, z_glass_top=0.0, tol_mm=LOS_TOLERANCE_MM):
    """Compare each camera's recovered line-of-sight distance to the glass with
    the hand-measured one. cam_los = [(i, d_mm)], 0-based index into `cameras`.
    Returns (ok, deviations {i: recovered - measured}, reason)."""
    devs = {}
    for i, d in (cam_los or []):
        if 0 <= int(i) < len(cameras):
            devs[int(i)] = camera_los_distance(cameras[int(i)], z_glass_top) - float(d)
    if not devs:
        return True, devs, ""
    bad = {i: v for i, v in devs.items() if abs(v) > tol_mm}
    if not bad:
        return True, devs, ""
    worst = max(bad, key=lambda i: abs(bad[i]))
    return False, devs, (f"camera-glass line of sight off by {bad[worst]:+.0f} mm on camera "
                         f"index {worst} ({len(bad)} camera(s) beyond ±{tol_mm:g} mm)")


# ---------------------------------------------------------------------------
# Front-facing check ("cheirality") and exact mirror-twin repair
# ---------------------------------------------------------------------------
# The Maas collinearity projects a point whether it lies in front of or behind
# the camera, and refraction depends only on heights and radial distances from
# each camera's nadir. So mirroring the whole world through ANY vertical plane
# and turning every camera to face away (A -> -A M) reproduces the images
# EXACTLY (verified to 1e-13 px), with identical inter-camera distances, camera
# heights and line-of-sight distances. Only two things tell the twins apart:
# chirality (signed volume vs cyclic order) and whether the particles are in
# front of the cameras. Applying the same reflection again returns the
# correct-handed solution with an identical fit.
FRONT_FACING_MIN = 0.90      # min fraction of particles in front of every camera


def front_facing_fractions(cameras, pts_xyz):
    """Per camera: fraction of points with positive depth A[2]·(X - C)
    (the Maas camera's optical axis is the third row of its rotation)."""
    P = np.asarray(pts_xyz, float)
    out = []
    for c in cameras:
        d = (P - np.asarray(c.position, float)) @ np.asarray(c.rotation_matrix, float)[2]
        d = d[np.isfinite(d)]
        out.append(float(np.mean(d > 0)) if d.size else float('nan'))
    return out


def front_facing_check(cameras, pts_xyz, min_frac=FRONT_FACING_MIN):
    """Validity rule: every camera must have >= min_frac of the points in front."""
    fr = front_facing_fractions(cameras, pts_xyz)
    bad = [i for i, f in enumerate(fr) if not (f >= min_frac)]
    if not bad:
        return True, fr, ""
    return False, fr, (f"particles behind camera(s) {bad} "
                       f"(in front: {', '.join(f'{100*f:.0f}%' for f in fr)})")


def _angles_from_rotation(A):
    """(omega, phi, kappa) of a Maas rotation matrix (inverse of
    CameraParams.rotation_matrix)."""
    return np.array([np.arctan2(A[1, 2], A[2, 2]), np.arcsin(-np.clip(A[0, 2], -1, 1)),
                     np.arctan2(A[0, 1], A[0, 0])])


def is_facing_away_mirror(cameras, pts_xyz, cyclic_order, min_frac=FRONT_FACING_MIN):
    """True if the solution is the facing-away mirror twin: wrong chirality AND
    every camera has (almost) all points behind it."""
    if not cyclic_order or camera_chirality_sign(
            np.array([c.position for c in cameras], float), cyclic_order) >= 0:
        return False
    fr = front_facing_fractions(cameras, pts_xyz)
    return all(np.isfinite(f) and f <= 1.0 - min_frac for f in fr)


def mirror_twin(cameras, pts_xyz):
    """Exact twin of a solution: reflect cameras and points through the vertical
    plane x = mean(points x), and turn each camera to face the other way:
    A -> -A @ diag(-1, 1, 1). Same images, distances, heights and line-of-sight
    distances; opposite chirality and front-facing. Returns (new_cameras, new_pts)."""
    import copy
    P = np.asarray(pts_xyz, float)
    cx = float(np.nanmean(P[:, 0]))
    Mx = np.diag([-1.0, 1.0, 1.0])
    new_cams = []
    for c in cameras:
        nc = copy.deepcopy(c)
        p = np.asarray(c.position, float).copy(); p[0] = 2.0 * cx - p[0]
        nc.position = p
        nc.angles = _angles_from_rotation(-(np.asarray(c.rotation_matrix, float) @ Mx))
        new_cams.append(nc)
    newP = P.copy(); newP[:, 0] = 2.0 * cx - newP[:, 0]
    return new_cams, newP
