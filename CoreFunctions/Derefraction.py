"""
Derefraction.py
===============
Method B — iterative de-refraction bootstrap that bridges Svoboda's
projective self-calibration to the Maas multimedia model.

THE IDEA
--------
Svoboda's rank-4 factorisation + Euclidean stratification assume a PINHOLE
(straight-ray) model. Refraction makes that assumption invalid, so euclidize()'s
similarity upgrade is wrong by the irreducible non-similarity residual.

Instead of trying to bridge the two frames with a similarity transform AFTER
the fact (which provably cannot work), we correct the INPUT to Svoboda:

    1. Run Svoboda on the current image points  -> projective -> Euclidean.
    2. Anchor scale + orientation ONCE with the camera-glass distance
       (a rig constant -> still pure self-calibration).
    3. For each point j and camera k, compute the Maas radial shift P_bar in
       object space, then the pixel-space displacement it induces:
           delta_kj = proj(Pe_k, P_bar_j) - proj(Pe_k, Xe_j)
    4. Correct the RAW image points:  Ws_work = Ws_raw - delta.
    5. Re-run Svoboda on Ws_work and repeat until delta stops changing.

At the fixed point, Ws_work is a consistent pinhole observation set, so
euclidize() is finally valid and the cloud is metric. Hand the result to the
multimedia bundle adjustment (Method A) for final physical refinement.

WHY THE DELTA IS COMPUTED IN PIXEL SPACE
----------------------------------------
MultimediaLUT.shift_point is a pure OBJECT-SPACE operation (it only uses camera
position / nadir / refractive params, never focal length). Projecting the
shifted and unshifted 3D points through the same pinhole Pe that euclidize
produced keeps everything in the pixel units of Ws -- no pixel<->mm conversion,
no unit mismatch.

CONVENTION
----------
Ws / Xe / Pe are in the same pixel-space frame euclidize() outputs.
After the anchor, the cloud is metric (mm): glass top at z=0, cameras above at
z>0, particles below at z<-t. The LUT and the shift require this convention.

The Method-C virtual-camera pre-correction can be injected via `c_hook`; it is
applied to the very first pass only, to shrink the initial reprojection floor.
"""

from __future__ import annotations
import numpy as np
from typing import Callable, Optional
from Multimedia.Frame_convention import FrameConvention



# ---------------------------------------------------------------------------
# Pinhole projection (pixel space)
# ---------------------------------------------------------------------------

def project_pinhole(Pe_block: np.ndarray, X_h: np.ndarray) -> np.ndarray:
    """
    Project homogeneous 3-D points through one pixel-space pinhole camera.

    Parameters
    ----------
    Pe_block : (3, 4) projection matrix for one camera (euclidize output).
    X_h      : (4, N) homogeneous 3-D points.

    Returns
    -------
    (2, N) image points in pixels (homogeneous division applied).
    """
    x = Pe_block @ X_h            # (3, N)
    w = x[2, :]
    w = np.where(np.abs(w) < 1e-12, 1e-12, w)
    return x[:2, :] / w


# ---------------------------------------------------------------------------
# Vectorised Maas shift for one camera (uses the prebuilt LUT interpolator)
# ---------------------------------------------------------------------------

def maas_shift_camera(lut_cam, cam_position: np.ndarray, t_total: float,
                      Xe3: np.ndarray) -> np.ndarray:
    """
    Apply the Maas radial shift P -> P_bar to every point, for one camera.

    Vectorised equivalent of MultimediaLUT.shift_point over all columns.

    Parameters
    ----------
    lut_cam      : MultimediaLUT for this camera (already built).
    cam_position : (3,) camera centre in mm (anchored frame).
    t_total      : total glass thickness in mm (mm.t_total).
    Xe3          : (3, N) metric 3-D points (anchored frame).

    Returns
    -------
    Pbar3 : (3, N) radially shifted points (Z unchanged).
    """
    X0, Y0, _ = cam_position
    X, Y, Z = Xe3[0, :], Xe3[1, :], Xe3[2, :]

    dX, dY = X - X0, Y - Y0
    R = np.sqrt(dX * dX + dY * dY)                       # radial dist from nadir
    Zp = -(Z + t_total)                                  # depth below glass bottom
    Zp = np.clip(Zp, 0.0, None)                          # LUT convention (no raise)

    # Bilinear lookup of R_bar/R; RegularGridInterpolator takes (N, 2)
    pts = np.column_stack([R, Zp])
    ratio = lut_cam.interpolator(pts)                    # (N,)
    ratio = np.where(R < 1e-9, 1.0, ratio)               # no shift at nadir

    Xbar = X0 + dX * ratio
    Ybar = Y0 + dY * ratio
    return np.vstack([Xbar, Ybar, Z])


# ---------------------------------------------------------------------------
# Pixel-space de-refraction delta for the whole rig
# ---------------------------------------------------------------------------

def derefraction_delta(Pe: np.ndarray, Xe: np.ndarray, cameras_mm: list,
                       mc_lut, IdMat_cols: np.ndarray) -> np.ndarray:
    """
    Compute the pixel-space refraction displacement for every visible
    (camera, point) pair.

    delta_kj = proj(Pe_k, P_bar_j) - proj(Pe_k, Xe_j)

    Parameters
    ----------
    Pe         : (3*CAMS, 4) anchored pinhole projection matrices.
    Xe         : (4, N) anchored metric 3-D points (homogeneous).
    cameras_mm : list of CameraParams (anchored positions, mm).
    mc_lut     : MultiCameraLUT (built for the anchored cameras).
    IdMat_cols : (CAMS, N) visibility for these N columns (1 = visible).

    Returns
    -------
    delta : (3*CAMS, N) homogeneous image deltas; row 3k+2 is 0 so subtracting
            from a homogeneous Ws block leaves the weight component untouched.
            Entries for points not visible in a camera are 0.
    """
    CAMS = len(cameras_mm)
    N = Xe.shape[1]
    Xe3 = Xe[:3, :] / Xe[3:4, :]
    Xe_h = np.vstack([Xe3, np.ones(N)])

    delta = np.zeros((3 * CAMS, N))

    for k, cam in enumerate(cameras_mm):
        Pe_k = Pe[3 * k:3 * k + 3, :]
        lut_cam = mc_lut[cam.cam_id]

        x_pin = project_pinhole(Pe_k, Xe_h)                       # (2, N) no refraction
        Pbar3 = maas_shift_camera(lut_cam, cam.position,
                                  cam.mm.t_total, Xe3)            # (3, N)
        Pbar_h = np.vstack([Pbar3, np.ones(N)])
        x_ref = project_pinhole(Pe_k, Pbar_h)                     # (2, N) refracted

        d = x_ref - x_pin                                         # (2, N)
        vis = IdMat_cols[k, :] > 0
        delta[3 * k:3 * k + 2, :][:, vis] = d[:, vis]

    return delta


# ---------------------------------------------------------------------------
# Minimal one-time metric anchor (scale + orientation only)
# ---------------------------------------------------------------------------

def simple_metric_anchor(Pe, Xe, Ce, Re, camera_glass_distance: float,
                         z_glass_top: float = 0.0, verbose: bool = True):
    """
    Bring the euclidize() reconstruction to physical mm using ONE rig constant:
    the camera-to-glass distance. This is NOT the failed 'similarity bridge' --
    it only fixes the global scale/orientation gauge so the refraction physics
    have meaningful absolute lengths. The de-refraction loop, not the anchor,
    removes the non-similarity warp.

    ROUGH FALLBACK ONLY. Prefer wiring your validated Z-only camera anchor
    (Apply_anchor / Camera_anchor) into derefraction_bootstrap(anchor=...).
    Reason: a single scalar (camera-glass distance) fixes scale + orientation
    but UNDER-DETERMINES the Z-translation by one DOF -- the glass-plane
    location in the reconstruction is unknown without a second length or the
    camera Z heights. This fallback closes that gap with an explicit gauge
    assumption (glass at the top of the particle cloud), which is fine to
    bootstrap the loop but is not as tight as the Z-only anchor.

    Steps:
      1. Orientation by TWO-SIDED scoring (matches Metric_anchor mechanics):
         try both Z-orientations; the flip is a Z-REFLECTION -- negate Xe[2],
         Ce[2], and the third column Pe[:,2], leaving Re untouched (exactly
         reprojection-invariant). Pick the orientation whose camera centroid
         sits ABOVE the particle centroid (cameras above, particles below).
      2. Gauge the glass plane at the 95th percentile of particle Z (its top),
         then scale so mean camera height above that plane = camera_glass_distance.
      3. Translate that glass plane to z = z_glass_top.

    Returns (Pe, Xe, Ce, Re, info) in the anchored frame.
    """
    import numpy as np

    def _zreflect(Pe_, Xe_, Ce_):
        Pe2, Xe2, Ce2 = Pe_.copy(), Xe_.copy(), Ce_.copy()
        Pe2[:, 2] = -Pe2[:, 2]        # third COLUMN across all 3x4 blocks
        Xe2[2, :] = -Xe2[2, :]
        Ce2[2, :] = -Ce2[2, :]
        return Pe2, Xe2, Ce2          # Re unchanged; reprojection invariant

    # --- 1. two-sided orientation: cameras above particles ---
    Xe3_a = Xe[:3, :] / Xe[3:4, :]
    Cc_a = Ce[:3, :]
    sep_a = float(Cc_a[2, :].mean() - Xe3_a[2, :].mean())   # cam centroid above particles?

    Pe_f, Xe_f, Ce_f = _zreflect(Pe, Xe, Ce)
    Xe3_b = Xe_f[:3, :] / Xe_f[3:4, :]
    Cc_b = Ce_f[:3, :]
    sep_b = float(Cc_b[2, :].mean() - Xe3_b[2, :].mean())

    flipped = sep_b > sep_a
    if flipped:
        Pe, Xe, Ce = Pe_f, Xe_f, Ce_f
        Xe3, Cc = Xe3_b, Cc_b
    else:
        Xe3, Cc = Xe3_a, Cc_a

    Xe3 = Xe3.copy(); Cc = Cc.copy()
    N = Xe.shape[1]
    CAMS = Cc.shape[1]

    # --- 2. gauge glass plane (top of particle cloud) and scale ---
    glass_recon = float(np.percentile(Xe3[2, :], 95))   # particle nearest glass
    mean_h = Cc[2, :].mean() - glass_recon
    if abs(mean_h) < 1e-9:
        raise ValueError("Mean camera height above gauged glass ~0; cannot set scale.")
    s = camera_glass_distance / mean_h

    # --- 3. similarity: scale about the gauged glass plane, then set it to z_glass_top ---
    T = np.eye(4)
    T[:3, :3] *= s
    # after scaling, glass sits at s*glass_recon; shift it to z_glass_top
    T[2, 3] = z_glass_top - s * glass_recon

    Xe3 = s * Xe3 + np.array([[0.0], [0.0], [T[2, 3]]])
    Cc = s * Cc + np.array([[0.0], [0.0], [T[2, 3]]])

    Xe_out = np.vstack([Xe3, np.ones(N)])
    Ce_out = np.vstack([Cc, np.ones(CAMS)])

    # cameras: P_new = P_old @ T^{-1}  so proj(P_new, X_new) == proj(P_old, X_old)
    Tinv = np.linalg.inv(T)
    Pe_out = np.zeros_like(Pe)
    for k in range(CAMS):
        Pe_out[3 * k:3 * k + 3, :] = Pe[3 * k:3 * k + 3, :] @ Tinv

    info = {'scale': float(s), 'z_flipped': flipped,
            'glass_recon_z': glass_recon,
            'mean_cam_height_mm': float(Cc[2, :].mean() - z_glass_top),
            'frac_particles_below_glass': float(np.mean(Xe3[2, :] < z_glass_top))}
    if verbose:
        print(f"  [anchor] scale={s:.5f}  z_flipped={flipped}  "
              f"mean cam height={info['mean_cam_height_mm']:.2f} mm  "
              f"particles below glass={info['frac_particles_below_glass']*100:.0f}%")
    return Pe_out, Xe_out, Ce_out, Re, info


# ---------------------------------------------------------------------------
# Method C — flat-port virtual-camera pre-correction (first-pass seed)
# ---------------------------------------------------------------------------

def make_flatport_chook(focal_px, n1: float, n3: float,
                        camera_glass_distance: float, zp_rep: float,
                        eps: float = 1e-9):
    """
    Build a `c_hook` that pre-warps first-pass image points toward a pinhole.

    The observed (refracted) image radius corresponds to the air-side ray angle
    beta1 = atan(r/f). The straight-line angle to the object, beta_straight (what
    an unrefracted pinhole would image), is a DEPTH-BLEND of beta1 and the
    water-side angle beta3:

        tan(beta_straight) = [Z0 tan(beta1) + Zp tan(beta3)] / (Z0 + Zp)
        beta3 = asin((n1/n3) sin(beta1))            # Snell, glass-independent

    where Z0 is the camera height above the glass (camera_glass_distance) and Zp
    is a representative particle depth below the glass. Correcting r -> f*tan(beta_straight)
    removes the dominant refraction with NO 3-D required, giving Svoboda a
    near-pinhole problem on pass 1. The residual depth-spread (points away from
    Zp) is left to the Method-B de-refraction loop.

    NB: using beta3 directly (Zp >> Z0) over-corrects by ~n3/n1; the blend with the
    real camera height is essential.

    Parameters
    ----------
    focal_px : sequence of float, length CAMS — per-camera fx in pixels.
               Points must be principal-point-subtracted.
    n1, n3   : refractive indices (air, water).
    camera_glass_distance : Z0, camera height above glass in mm.
    zp_rep   : representative particle depth below glass in mm (rough is fine;
               the correction is weakly sensitive to it).
    """
    focal_px = [float(f) for f in focal_px]
    ratio = n1 / n3
    Z0 = float(camera_glass_distance)
    Zp = float(zp_rep)

    def c_hook(Ws: np.ndarray) -> np.ndarray:
        out = Ws.copy()
        CAMS = Ws.shape[0] // 3
        for k in range(CAMS):
            f = focal_px[k]
            x = Ws[3 * k, :]
            y = Ws[3 * k + 1, :]
            r = np.sqrt(x * x + y * y)
            beta1 = np.arctan2(r, f)
            s = np.clip(ratio * np.sin(beta1), -1.0, 1.0)
            beta3 = np.arcsin(s)
            tan_straight = (Z0 * np.tan(beta1) + Zp * np.tan(beta3)) / (Z0 + Zp)
            r_corr = f * tan_straight
            scale = np.where(r < eps, 1.0, r_corr / np.maximum(r, eps))
            out[3 * k, :] = x * scale
            out[3 * k + 1, :] = y * scale
        return out

    return c_hook


# ---------------------------------------------------------------------------
# Frame registration (lock the gauge across passes via the cameras)
# ---------------------------------------------------------------------------

def _umeyama_similarity(src: np.ndarray, dst: np.ndarray):
    """
    Fit s, R, t minimising || s R src + t - dst || with a PROPER rotation
    (no reflection). src, dst are (3, N). Returns (T 4x4, rms residual).

    Used to register each de-refraction pass to the pass-1 metric frame using
    the camera centres, which do not move during de-refraction. The residual
    RMS is a direct measure of Svoboda pose stability across passes.
    """
    src = np.asarray(src, float); dst = np.asarray(dst, float)
    N = src.shape[1]
    mu_s = src.mean(1, keepdims=True); mu_d = dst.mean(1, keepdims=True)
    sc = src - mu_s; dc = dst - mu_d
    Sigma = (dc @ sc.T) / N
    U, D, Vt = np.linalg.svd(Sigma)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1.0                      # force proper rotation, no flip-as-reflection
    R = U @ S @ Vt
    var_s = float((sc ** 2).sum() / N)
    s = float(np.trace(np.diag(D) @ S) / var_s) if var_s > 1e-12 else 1.0
    t = mu_d.ravel() - s * R @ mu_s.ravel()
    T = np.eye(4); T[:3, :3] = s * R; T[:3, 3] = t
    resid = s * R @ src + t[:, None] - dst
    rms = float(np.sqrt((resid ** 2).sum() / N))
    return T, rms


def _apply_T_homog(T: np.ndarray, Xh: np.ndarray) -> np.ndarray:
    """Apply 4x4 similarity to homogeneous (4,N) points; returns normalised (4,N)."""
    Y = T @ (Xh / Xh[3:4, :])
    return Y / Y[3:4, :]


def _triangulate_dlt(Pe: np.ndarray, Ws_cols: np.ndarray,
                     IdMat_cols: np.ndarray) -> np.ndarray:
    """
    Linear DLT triangulation of each point against FIXED cameras Pe.

    Pe         : (3*CAMS, 4) pixel-space projection matrices (frozen).
    Ws_cols    : (3*CAMS, N) image points (homogeneous rows per camera).
    IdMat_cols : (CAMS, N) visibility (1 = use this camera for this point).

    Returns (4, N) homogeneous points; columns with <2 views are NaN.
    """
    CAMS = IdMat_cols.shape[0]
    N = Ws_cols.shape[1]
    X = np.full((4, N), np.nan)
    for j in range(N):
        rows = []
        for k in range(CAMS):
            if IdMat_cols[k, j] > 0:
                u = Ws_cols[3 * k, j]; v = Ws_cols[3 * k + 1, j]
                if np.isnan(u) or np.isnan(v):
                    continue
                Pk = Pe[3 * k:3 * k + 3, :]
                rows.append(u * Pk[2] - Pk[0])
                rows.append(v * Pk[2] - Pk[1])
        if len(rows) >= 4:
            _, _, Vt = np.linalg.svd(np.array(rows))
            Xh = Vt[-1]
            if abs(Xh[3]) > 1e-12:
                X[:, j] = Xh / Xh[3]
    return X


# ---------------------------------------------------------------------------
# The bootstrap loop
# ---------------------------------------------------------------------------

def derefraction_bootstrap(
    Ws_raw: np.ndarray,
    inlier_idx: np.ndarray,
    IdMat: np.ndarray,
    svoboda_step: Callable[[np.ndarray], tuple],
    build_cams: Callable[[np.ndarray, np.ndarray], list],
    build_lut: Callable[[list], object],
    camera_glass_distance: float,
    *,
    anchor: Optional[Callable] = None,
    z_glass_top: float = 0.0,
    max_iter: int = 8,
    tol: float = 1e-2,
    c_hook: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    fixed_frame: bool = True,
    freeze_cameras: bool = True,
    camera_distances=None,
    camera_los=None,
    max_shift_px: float = 0.0,
    verbose: bool = True,
) -> dict:
    """
    Iterate Svoboda on progressively de-refracted image points until the
    correction field converges.

    Parameters
    ----------
    Ws_raw     : (3*CAMS, N_all) image matrix in euclidize pixel space
                 (your `linear.Ws`, radial-undistorted, pp-subtracted).
    inlier_idx : column indices of the reconstructed inlier points.
    IdMat      : (CAMS, N_all) visibility matrix aligned to Ws_raw columns.
    svoboda_step : callable(Ws_cols (3*CAMS, n_in)) -> (Pe, Xe, Ce, Re).
                   Wrap fill_mm_bundle + euclidize here (no multimedia BA).
    build_cams : callable(Ce, Re) -> cameras_mm list.
    build_lut  : callable(cameras_mm) -> built MultiCameraLUT.
    camera_glass_distance : known rig constant in mm (the only anchor).
    c_hook     : optional Method-C pre-correction applied to the FIRST pass
                 only: callable(Ws_cols) -> Ws_cols.
    max_iter   : cap on de-refraction passes.
    tol        : convergence when max |delta - delta_prev| (pixels) < tol.

    Returns
    -------
    dict with keys:
      Ws_corrected, Pe, Xe, Ce, Re, cameras_mm, lut, n_iter, history, anchor_info
    """
    cols = inlier_idx
    IdMat_cols = IdMat[:, cols]
    Ws_work = Ws_raw.copy()
    delta_prev = None
    history = []

    # anchor(Pe, Xe, Ce, Re) -> (Pe, Xe, Ce, Re, info). Default = scalar self-cal anchor.
    if anchor is None:
        # 'auto' camera-glass height (None): the frame derives it from the line-of-sight
        # distances; this fallback anchor (used only when no frame can be built) needs a
        # number, so it takes the mean line-of-sight distance - an overestimate of the
        # vertical height, on the side the seed tolerates (run evidence: -20..+50 mm).
        _cgd_anchor = camera_glass_distance
        if _cgd_anchor is None:
            if not camera_los:
                raise ValueError("camera_glass_distance is 'auto' but no line-of-sight "
                                 "distances were given")
            _cgd_anchor = float(np.mean([float(dd) for _, dd in camera_los]))

        def anchor(Pe_, Xe_, Ce_, Re_):
            return simple_metric_anchor(
                Pe_, Xe_, Ce_, Re_, _cgd_anchor,
                z_glass_top=z_glass_top, verbose=verbose)

    Ce_ref = None        # pass-1 anchored camera centres; the fixed gauge for later passes
    Pe_frozen = cams_frozen = lut_frozen = None   # set on pass 1 when freeze_cameras
    frame = None         # the single FrameConvention authority, built on pass 1
    geometry_ok = True   # set by frame.is_valid on pass 1; gates GoCal de-refraction

    _last_good = None    # state after the last pass whose correction stayed bounded (divergence guard)
    guard_stopped = False
    for it in range(1, max_iter + 1):
        Ws_in = Ws_work[:, cols]
        if it == 1 and c_hook is not None:
            Ws_in = c_hook(Ws_in)        # virtual-camera seed, first pass only

        if it == 1:
            # ---- pass 1: estimate, then establish the ONE frame (FrameConvention) ----
            Pe, Xe, Ce, Re = svoboda_step(Ws_in)
            cam_centers = (Ce[:3, :] / Ce[3:4, :]).T          # (CAMS,3) euclidize
            Xe3 = Xe[:3, :] / Xe[3:4, :]                       # (3,N)    euclidize
            cams_euc = build_cams(Ce, Re)
            t_glass = float(getattr(cams_euc[0].mm, 't_total', 0.0))

            # DIAGNOSTIC (log only): each particle's image radius (Ws is centred on the
            # principal point), averaged over the cameras that see it
            _img_r = None
            try:
                with np.errstate(all='ignore'):
                    _Kc = Ws_in.shape[0] // 3
                    _img_r = np.nanmean(np.vstack([np.hypot(Ws_in[3 * k], Ws_in[3 * k + 1])
                                                   for k in range(_Kc)]), axis=0)
            except Exception:
                _img_r = None
            frame = FrameConvention.from_euclidize(
                cam_centers, Xe3, camera_glass_distance, t_glass,
                z_glass_top=z_glass_top, camera_distances=camera_distances,
                verbose=verbose, img_radius=_img_r, camera_los=camera_los)

            if frame is None:
                # degenerate geometry — fall back to the old anchor, flag invalid
                Pe, Xe, Ce, Re, anchor_info = anchor(Pe, Xe, Ce, Re)
                geometry_ok = False
            else:
                # bring cameras + cloud + Pe into the single physical frame
                Pe = frame.to_frame_Pe(Pe)
                Ce3p = frame.to_frame_points(Ce[:3, :] / Ce[3:4, :])
                Xe3p = frame.to_frame_points(Xe3)
                Ce = np.vstack([Ce3p, np.ones(Ce3p.shape[1])])
                Xe = np.vstack([Xe3p, np.ones(Xe3p.shape[1])])
                geometry_ok, vinfo = frame.is_valid(Ce3p.T, Xe3p)
                anchor_info = {'scale': frame.scale, 'z_flipped': 'frame',
                               'reg_rms_mm': 0.0,
                               'frac_particles_below_glass':
                                   vinfo['frac_particles_below_glass']}
            Ce_ref = (Ce[:3, :] / Ce[3:4, :]).copy()
            reg_rms = 0.0
            cameras_mm = build_cams(Ce, Re)        # physical-frame cameras
            lut = build_lut(cameras_mm)
            if freeze_cameras:
                Pe_frozen, cams_frozen, lut_frozen = Pe, cameras_mm, lut

        elif freeze_cameras:
            # ---- later passes: cameras FIXED (physical Pe), triangulate only ----
            # Triangulating against the physical Pe yields physical points directly,
            # in the SAME frame the FrameConvention established — no re-derivation.
            Pe, cameras_mm, lut = Pe_frozen, cams_frozen, lut_frozen
            Xe = _triangulate_dlt(Pe, Ws_work[:, cols], IdMat_cols)
            reg_rms = 0.0
            anchor_info = {'scale': 'frozen', 'z_flipped': 'frozen', 'reg_rms_mm': 0.0,
                           'frac_particles_below_glass':
                               float(np.nanmean((Xe[2, :] / Xe[3, :]) < (
                                   frame.glass_bottom if frame is not None
                                   else z_glass_top - t_glass)))}   # no frame: the anchor fallback of pass 1

        else:
            # ---- later passes (re-estimate): register to pass-1 frame (fixed_frame) ----
            Pe, Xe, Ce, Re = svoboda_step(Ws_in)
            if not fixed_frame:
                Pe, Xe, Ce, Re, anchor_info = anchor(Pe, Xe, Ce, Re)
                reg_rms = 0.0
            else:
                Ce_now = Ce[:3, :] / Ce[3:4, :]
                T_reg, reg_rms = _umeyama_similarity(Ce_now, Ce_ref)
                Xe = _apply_T_homog(T_reg, Xe)
                Ce = _apply_T_homog(T_reg, Ce)
                Tinv = np.linalg.inv(T_reg)
                for _k in range(Ce_ref.shape[1]):
                    Pe[3 * _k:3 * _k + 3, :] = Pe[3 * _k:3 * _k + 3, :] @ Tinv
                anchor_info = {'scale': float(T_reg[0, 0]), 'z_flipped': 'locked',
                               'reg_rms_mm': reg_rms,
                               'frac_particles_below_glass':
                                   float(np.mean((Xe[2, :] / Xe[3, :]) < z_glass_top))}
            cameras_mm = build_cams(Ce, Re)
            lut = build_lut(cameras_mm)

        # --- pixel-space de-refraction delta ---
        delta = derefraction_delta(Pe, Xe, cameras_mm, lut, IdMat_cols)

        # --- divergence guard (max_shift_px > 0) --------------------------------
        # A correction larger than any physically possible refraction shift means the
        # seed re-solved in this pass has broken down (run evidence: a plausible first
        # pass of 7-364 px, then thousands to millions of px). Discard this pass - its
        # correction AND the geometry that produced it - and stop with the last bounded
        # state; with no bounded pass yet, the observations stay uncorrected.
        if max_shift_px and max_shift_px > 0:
            _md = float(np.nanmax(np.abs(delta))) if np.size(delta) else 0.0
            if not np.isfinite(_md) or _md > max_shift_px:
                guard_stopped = True
                if _last_good is not None:
                    (Ws_work, Pe, Xe, Ce, Re, cameras_mm, lut, anchor_info) = _last_good
                    _kept = f"keeping pass {it - 1}"
                else:
                    Ws_work = Ws_raw.copy()
                    _kept = "no bounded pass yet - observations left uncorrected"
                history.append({'iter': it, 'max_delta_px': _md, 'delta_change_px': np.nan,
                                'scale': anchor_info['scale'], 'reg_rms_mm': reg_rms,
                                'guard_stopped': True})
                if verbose:
                    print(f"[derefract {it}] max|delta|={_md:8.3f}px exceeds the divergence guard "
                          f"({max_shift_px:g} px) -> pass discarded, {_kept}")
                it = it - 1
                break

        # --- correct from RAW (fixed-point, never compounded) ---
        Ws_work = Ws_raw.copy()
        Ws_work[:, cols] -= delta

        # --- convergence on the correction field ---
        if delta_prev is None:
            change = np.inf
        else:
            change = float(np.max(np.abs(delta - delta_prev)))
        max_delta = float(np.max(np.abs(delta)))
        history.append({'iter': it, 'max_delta_px': max_delta,
                        'delta_change_px': change, 'scale': anchor_info['scale'],
                        'reg_rms_mm': reg_rms})
        if verbose:
            extra = "" if it == 1 else f"  reg_rms={reg_rms:8.2f}mm"
            print(f"[derefract {it}] max|delta|={max_delta:8.3f}px  "
                  f"change={change:8.3f}px{extra}")

        _last_good = (Ws_work.copy(), Pe, Xe, Ce, Re, cameras_mm, lut, anchor_info)
        if change < tol:
            break
        delta_prev = delta

    return {
        'Ws_corrected': Ws_work,
        'Pe': Pe, 'Xe': Xe, 'Ce': Ce, 'Re': Re,
        'cameras_mm': cameras_mm, 'lut': lut,
        'n_iter': it, 'history': history, 'anchor_info': anchor_info,
        'geometry_ok': geometry_ok, 'frame': frame,
        'guard_stopped': guard_stopped,
    }
