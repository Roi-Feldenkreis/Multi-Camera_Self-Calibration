"""
Camera_anchor.py
================

Anchors a projective reconstruction to known per-camera positions by
fitting the best similarity transform (uniform scale + rotation + 
translation) from the reconstructed camera centres onto the known
ground-truth or measured positions.

Two anchor functions are exposed:

    apply_camera_anchor_xyz(Pe, Xe, Ce, Re, known_positions, ...)
        Full XYZ anchor — fits a 7-DOF similarity transform 
        (1 scale + 3 rotation + 3 translation) to 4 known camera 
        positions (12 constraints, well-overdetermined).

    apply_camera_anchor_z(Pe, Xe, Ce, Re, known_z_heights, ...)
        Z-only anchor — uses only the four Z heights as constraints.
        4 constraints to determine scale + Z-translation + a partial
        constraint on rotation about the vertical axis.  Falls back
        to the standard metric_anchor for the in-plane rotation.

Both functions return:
    Pe_out, Xe_out, Ce_out, Re_out, info

where ``info`` records the fitted scale, rotation, translation,
residual error, and any safety bounds that were triggered.

WHY THIS EXISTS
---------------
The standard metric_anchor scales the reconstruction so that the MEAN
of all camera Z heights equals a configured scalar.  This is a single
constraint and cannot capture the per-camera Z spread of a real rig.
For the caldata_refracted_step3 dataset the true cam-Z range is 
[318, 464] mm — a 146 mm spread that the scalar anchor averages away.

By using all known camera positions (or just their Z components) we
turn the anchor into an overdetermined least-squares fit, which is the
mathematically correct generalisation of the existing single-scalar
target.

USAGE
-----
This module is meant to be called from GoCal_safe.py in place of (or
in addition to) the existing apply_metric_anchor, when a known camera
prior is available.  See the README block at the top of GoCal_safe.py
for integration notes.
"""
from __future__ import annotations
import numpy as np
from typing import Optional


# =============================================================
# Helper: best similarity transform from one point set to another
# =============================================================
def _fit_similarity(src: np.ndarray, dst: np.ndarray) -> tuple:
    """
    Find the similarity transform (scale s, rotation R, translation t)
    that best maps src points onto dst points in the least-squares sense:

        dst ≈ s * R @ src + t

    Both src and dst are (3, N) arrays of point coordinates.

    Returns (s, R, t, residual_rms) where R is 3x3 with det(R)=+1.

    This is the Umeyama (1991) algorithm: a closed-form SVD solution
    to the absolute orientation problem.  It is the same algorithm
    used internally by SciPy's procrustes routines but written out
    here so the residual diagnostics are exposed.
    """
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    assert src.shape == dst.shape and src.shape[0] == 3, \
        f"src and dst must both be (3, N); got {src.shape} and {dst.shape}"
    N = src.shape[1]

    # Center both point sets at their centroids.
    src_mean = src.mean(axis=1, keepdims=True)
    dst_mean = dst.mean(axis=1, keepdims=True)
    src_c    = src - src_mean
    dst_c    = dst - dst_mean

    # Cross-covariance and its SVD.
    H = src_c @ dst_c.T / N
    U, S, Vt = np.linalg.svd(H)

    # Force det(R) = +1 to avoid reflection (R must be a proper rotation).
    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[2, 2] = -1.0
    R = Vt.T @ D @ U.T

    # Uniform scale.  Trace of (R @ H) is the optimal alignment of 
    # rotated-source onto destination; var_src normalises.
    var_src = float(np.mean(np.sum(src_c ** 2, axis=0)))
    if var_src < 1e-15:
        # Degenerate input — all source points coincident.  Use identity.
        return 1.0, np.eye(3), dst_mean.ravel() - src_mean.ravel(), float('inf')
    s = float(np.trace(np.diag(S) @ D)) / var_src

    # Translation closes the centroid match.
    t = (dst_mean - s * R @ src_mean).ravel()

    # Residual RMS — how good the fit is.  Used as a diagnostic.
    src_mapped = s * R @ src + t[:, None]
    residual_rms = float(np.sqrt(np.mean(np.sum((src_mapped - dst) ** 2, axis=0))))

    return s, R, t, residual_rms


# =============================================================
# Apply a (scale, rotation, translation) to the whole reconstruction
# =============================================================
def _apply_similarity_to_reconstruction(
    Pe: np.ndarray, Xe: np.ndarray, Ce: np.ndarray, Re: np.ndarray,
    s: float, R: np.ndarray, t: np.ndarray,
) -> tuple:
    """
    Apply the similarity transform x_new = s * R @ x + t to:

      - particle homogeneous coordinates Xe   (4, M)
      - camera centres homogeneous coords Ce  (4, N)
      - projection matrices Pe                (3N, 4)
      - rotation matrices Re                  (3N, 3)

    Reprojection error is invariant under similarity transforms applied
    consistently to all three structures.  This is a fundamental fact 
    about projective geometry: scaling, rotating, and translating the
    world frame does not change image points if the cameras also rotate
    by R, translate by t, and the focal lengths are unchanged.

    Math:
       Let H = [[s*R, t],
                [0,   1]]  be the 4x4 similarity matrix.
       Then  X_new = H @ X
             C_new = H @ C
             P_new = P @ H^{-1}
             K @ R_new @ [I | -C_new] = (K @ R_old) @ R^T @ [I | -C_new']
             so R_new = R_old @ R^T  (and K stays the same)
    """
    n_cams = Ce.shape[1]

    # Build the 4x4 similarity H and its inverse.
    H = np.eye(4)
    H[:3, :3] = s * R
    H[:3,  3] = t
    H_inv = np.eye(4)
    H_inv[:3, :3] = R.T / s
    H_inv[:3,  3] = -R.T @ t / s

    # Apply to particles and camera centres.
    Xe_out = H @ Xe
    Ce_out = H @ Ce

    # Apply to projection matrices: P_new = P @ H_inv.
    Pe_out = Pe @ H_inv

    # Apply to rotation matrices: R_new = R_old @ R^T (per camera).
    Re_out = Re.copy()
    for i in range(n_cams):
        Re_out[3*i:3*i+3, :] = Re[3*i:3*i+3, :] @ R.T

    return Pe_out, Xe_out, Ce_out, Re_out


# =============================================================
# Z-flip helpers
# =============================================================
def _apply_z_flip(
    Pe: np.ndarray, Xe: np.ndarray, Ce: np.ndarray, Re: np.ndarray,
) -> tuple:
    """
    Reflect the entire reconstruction through the Z=0 plane.

    The Svoboda projective factorisation produces a reconstruction
    that is correct only up to a Z-axis reflection (an "up vs down"
    ambiguity).  A proper-rotation similarity fit cannot recover the
    correct orientation if the reconstruction is on the wrong side
    of this ambiguity, because that would require det(R) = -1.

    This helper mirrors the exact mechanics used in Metric_anchor's
    Z-flip step: negate the Z components of camera centres, of
    particle coordinates, and of the third column of each projection
    matrix.  Rotation matrices are NOT modified (this matches the
    behaviour of Metric_anchor).  Reprojection is exactly invariant
    under this operation, which is the key correctness check.

    Returns fresh copies of Pe, Xe, Ce (with Z negated); Re is aliased.
    """
    Pe_f = Pe.copy()
    Xe_f = Xe.copy()
    Ce_f = Ce.copy()
    Pe_f[:, 2] = -Pe_f[:, 2]
    Xe_f[2, :] = -Xe_f[2, :]
    Ce_f[2, :] = -Ce_f[2, :]
    return Pe_f, Xe_f, Ce_f, Re


def _try_both_z_orientations(
    Pe: np.ndarray, Xe: np.ndarray, Ce: np.ndarray, Re: np.ndarray,
    known_for_fit: np.ndarray,
    z_only: bool = False,
    glass_top: float = 0.0,
    verbose: bool = True,
) -> tuple:
    """
    Try both Z-orientations of the reconstruction and pick the better one.

    "Better" is defined by a combination of two criteria:

      1. Fit residual (primary): how well the similarity transform maps
         reconstructed cameras to known positions.  Lower is better.
      2. Particle-below-glass fraction (tiebreaker): after applying the
         similarity, what fraction of particles end up with z <= 0
         (below the glass top).  Higher is better — the multimedia BA
         pre-validation refuses to run if this fraction is too small,
         so even a slightly worse fit residual is preferable if it puts
         the particles on the correct side of glass.

    Decision rule:
      - If one orientation's fit residual is < 80% of the other's:
        pick the lower-residual one regardless of particle geometry.
      - Otherwise (residuals within 25% of each other):
        pick the one with more particles below glass.

    This matches the spirit of Metric_anchor's two-sided scoring while
    being adapted to the camera-position-prior anchor.

    Parameters
    ----------
    Pe, Xe, Ce, Re : the reconstruction (as in the anchor functions)
    known_for_fit  : (3, N) known camera positions to fit to
    z_only         : if True, only the Z component of known_for_fit is
                     used as constraint (for the Z-only anchor)
    glass_top      : Z coordinate of the glass top (mm), default 0.0
    verbose        : print the decision

    Returns
    -------
    Pe_used, Xe_used, Ce_used, Re_used : the orientation that was selected
                                          (either input or Z-flipped)
    s, R, t, residual_rms              : the similarity fit on that orientation
    z_flipped                          : True if the flipped orientation was chosen
    diagnostic                         : dict with both candidates' numbers
    """
    Ce_xyz = (Ce[:3, :] / Ce[3:4, :])
    Pe_f, Xe_f, Ce_f, Re_f = _apply_z_flip(Pe, Xe, Ce, Re)
    Ce_f_xyz = (Ce_f[:3, :] / Ce_f[3:4, :])

    if z_only:
        # Z-only fitting: use the 1D Z-fit approach.  Returns
        # (s, R=I, t with t[0]=t[1]=0, residual on Z only).
        s_nf, R_nf, t_nf, res_nf = _fit_z_only(Ce_xyz, known_for_fit)
        s_f,  R_f,  t_f,  res_f  = _fit_z_only(Ce_f_xyz, known_for_fit)
    else:
        s_nf, R_nf, t_nf, res_nf = _fit_similarity(Ce_xyz,   known_for_fit)
        s_f,  R_f,  t_f,  res_f  = _fit_similarity(Ce_f_xyz, known_for_fit)

    # Predict where particles land after applying each similarity.
    def _particle_below_frac(Xe_in, s, R, t):
        Xe_xyz = (Xe_in[:3, :] / Xe_in[3:4, :])
        Xe_mapped = s * R @ Xe_xyz + t[:, None]
        return float(np.mean(Xe_mapped[2, :] <= glass_top))

    frac_nf = _particle_below_frac(Xe,   s_nf, R_nf, t_nf)
    frac_f  = _particle_below_frac(Xe_f, s_f,  R_f,  t_f)

    diagnostic = {
        'noflip_residual_mm':       float(res_nf),
        'flip_residual_mm':         float(res_f),
        'noflip_particle_below_frac': frac_nf,
        'flip_particle_below_frac':   frac_f,
    }

    # Decision rule.
    # If one residual is clearly better (within 80% of the other), use it.
    # Otherwise prefer the one with more particles below glass.
    res_ratio = min(res_nf, res_f) / max(res_nf, res_f, 1e-9)
    if res_ratio < 0.80:
        # Big residual gap — let residual decide.
        choose_flip = (res_f < res_nf)
        decision_reason = 'residual'
    else:
        # Residuals comparable — let particle geometry decide.
        choose_flip = (frac_f > frac_nf)
        decision_reason = 'particle_below_frac'

    diagnostic['decision_reason'] = decision_reason
    diagnostic['z_flipped'] = bool(choose_flip)

    if verbose:
        flag_nf = '<-- KEEP' if not choose_flip else ''
        flag_f  = '<-- FLIP' if choose_flip else ''
        print(f"    Z-orient test:")
        print(f"      no-flip: residual {res_nf:6.2f} mm, "
              f"particles below glass {frac_nf*100:5.1f}%  {flag_nf}")
        print(f"      flipped: residual {res_f:6.2f} mm, "
              f"particles below glass {frac_f*100:5.1f}%  {flag_f}")
        print(f"      decision by {decision_reason}")

    if choose_flip:
        return Pe_f, Xe_f, Ce_f, Re_f, s_f, R_f, t_f, res_f, True, diagnostic
    else:
        return Pe, Xe, Ce, Re, s_nf, R_nf, t_nf, res_nf, False, diagnostic


def _fit_z_only(src_xyz: np.ndarray, known_z: np.ndarray) -> tuple:
    """
    Z-only similarity fit: scale + Z-translation only, identity rotation.

    src_xyz   : (3, N) reconstructed camera positions
    known_z   : either (N,) array of known Z heights, or (3, N) array
                (in which case only the Z row is used)

    Returns (s, R=I, t with t[0]=t[1]=0, z_residual_rms).
    """
    z_recon = src_xyz[2, :]
    if known_z.ndim == 1:
        z_known = known_z
    else:
        z_known = known_z[2, :]

    z_recon_c = z_recon - z_recon.mean()
    z_known_c = z_known - z_known.mean()

    denom = float(np.dot(z_recon_c, z_recon_c))
    if denom < 1e-9:
        return 1.0, np.eye(3), np.zeros(3), float('inf')

    s = float(np.dot(z_recon_c, z_known_c)) / denom

    R = np.eye(3)
    t = np.zeros(3)
    t[2] = float(z_known.mean() - s * z_recon.mean())

    z_mapped = s * z_recon + t[2]
    residual = float(np.sqrt(np.mean((z_mapped - z_known) ** 2)))
    return s, R, t, residual


# =============================================================
# Stage 1: full XYZ anchor
# =============================================================
def apply_camera_anchor_xyz(
    Pe: np.ndarray,
    Xe: np.ndarray,
    Ce: np.ndarray,
    Re: np.ndarray,
    known_positions: np.ndarray,
    noise_std_mm: float = 0.0,
    rng_seed: Optional[int] = None,
    verbose: bool = True,
) -> tuple:
    """
    Anchor a reconstruction to known per-camera XYZ positions.

    Parameters
    ----------
    Pe : (3N, 4) projection matrices (stacked)
    Xe : (4, M) particle homogeneous coordinates
    Ce : (4, N) camera centres in homogeneous coordinates
    Re : (3N, 3) rotation matrices (stacked)
    known_positions : (3, N) ground-truth or measured camera positions in mm.
        Must be in the SAME ORDER as the cameras in Pe/Ce.
    noise_std_mm : float, default 0.0
        Standard deviation of Gaussian noise added to each component of
        each camera position before fitting.  Use 10.0 mm to simulate
        ruler-measurement uncertainty; 0.0 for noiseless (sanity check).
    rng_seed : Optional[int]
        Seed for the noise RNG.  Set this for reproducibility.
    verbose : bool

    Returns
    -------
    Pe_out, Xe_out, Ce_out, Re_out, info
    """
    info = {
        'mode': 'xyz',
        'n_cameras': Ce.shape[1],
        'noise_std_mm': noise_std_mm,
        'applied': False,
    }

    Ce_xyz = (Ce[:3, :] / Ce[3:4, :])

    # Add measurement noise to the known positions, if requested.
    known = np.asarray(known_positions, dtype=float).copy()
    if known.shape != Ce_xyz.shape:
        info['reason'] = (f'shape mismatch: known {known.shape} '
                          f'vs reconstructed {Ce_xyz.shape}')
        if verbose:
            print(f"  Camera anchor SKIPPED: {info['reason']}")
        return Pe, Xe, Ce, Re, info

    if noise_std_mm > 0.0:
        rng = np.random.default_rng(rng_seed)
        known = known + rng.normal(0.0, noise_std_mm, size=known.shape)

    # Try both Z-orientations of the reconstruction and pick the better.
    # The selected orientation may be flipped relative to the input;
    # Pe_use, Xe_use, Ce_use, Re_use are the (possibly flipped) version
    # that the fitted similarity will be applied to.
    (Pe_use, Xe_use, Ce_use, Re_use,
     s, R, t, residual_rms,
     z_flipped, z_diag) = _try_both_z_orientations(
        Pe, Xe, Ce, Re, known_for_fit=known,
        z_only=False, glass_top=0.0, verbose=verbose,
    )

    info['scale'] = s
    info['rotation'] = R.tolist()
    info['translation'] = t.tolist()
    info['fit_residual_rms_mm'] = residual_rms
    info['z_flipped'] = z_flipped
    info['z_flip_diagnostic'] = z_diag

    # Safety bounds.  If the scale is wildly off or the residual is huge,
    # something went wrong upstream and we should refuse to apply.
    if not np.isfinite(s) or s <= 0.0:
        info['reason'] = f'invalid scale {s}'
        if verbose:
            print(f"  Camera anchor SKIPPED: {info['reason']}")
        return Pe, Xe, Ce, Re, info

    SCALE_LIMITS = (1e-3, 1e+3)
    if not (SCALE_LIMITS[0] <= s <= SCALE_LIMITS[1]):
        info['reason'] = (f'scale {s:.3e} outside safety bounds '
                          f'{SCALE_LIMITS}')
        if verbose:
            print(f"  Camera anchor SKIPPED: {info['reason']}")
        return Pe, Xe, Ce, Re, info

    # Apply the transform to the orientation-selected version of the
    # reconstruction (which may be Z-flipped relative to the input).
    Pe_out, Xe_out, Ce_out, Re_out = _apply_similarity_to_reconstruction(
        Pe_use, Xe_use, Ce_use, Re_use, s, R, t)

    info['applied'] = True
    info['reason'] = 'ok'

    if verbose:
        Ce_after = (Ce_out[:3, :] / Ce_out[3:4, :])
        per_cam_err = np.linalg.norm(Ce_after - known, axis=0)
        flip_str = " (Z-flipped)" if z_flipped else ""
        print(f"  Camera anchor (XYZ): scale={s:.4f}, "
              f"residual RMS={residual_rms:.2f} mm{flip_str}")
        if noise_std_mm > 0.0:
            print(f"    Noise σ = {noise_std_mm:.1f} mm added to prior")
        # Also report particle Z range so it's visible at a glance.
        Xe_after_xyz = (Xe_out[:3, :] / Xe_out[3:4, :])
        z_min, z_max = float(Xe_after_xyz[2,:].min()), float(Xe_after_xyz[2,:].max())
        below = float(np.mean(Xe_after_xyz[2,:] <= 0.0)) * 100
        print(f"    Particle Z range: [{z_min:+.1f}, {z_max:+.1f}] mm, "
              f"{below:.1f}% below glass")
        for i in range(Ce_xyz.shape[1]):
            print(f"    Cam {i+1}: err = {per_cam_err[i]:6.2f} mm "
                  f"-> ({Ce_after[0,i]:+7.1f}, {Ce_after[1,i]:+7.1f}, {Ce_after[2,i]:+7.1f})")

    return Pe_out, Xe_out, Ce_out, Re_out, info


# =============================================================
# Stage 2: Z-only anchor
# =============================================================
def apply_camera_anchor_z(
    Pe: np.ndarray,
    Xe: np.ndarray,
    Ce: np.ndarray,
    Re: np.ndarray,
    known_z_heights: np.ndarray,
    noise_std_mm: float = 0.0,
    rng_seed: Optional[int] = None,
    verbose: bool = True,
) -> tuple:
    """
    Anchor a reconstruction using ONLY known per-camera Z heights.

    This is intentionally weaker than the XYZ anchor: it constrains
    the scale (via the spread of Z values), the absolute Z position
    (via the mean), and a partial constraint on rotation about the
    vertical axis (via the relative Z order of the cameras).  X and Y
    of cameras are NOT used as constraints.

    The rationale: in a real experiment you can measure camera heights
    with a ruler much more easily than you can measure their X, Y
    positions.  This anchor finds the minimum prior knowledge needed
    to produce a good multimedia BA start.

    Approach: fit a similarity with rotation constrained to be about
    the world Z-axis (so the horizontal plane is preserved).  This is
    a 5-parameter fit: scale, 1 rotation angle (yaw), and a 3-vector
    translation.  We solve it by:

      1. Subtracting the centroid of (reconstructed XY) and (known XY)
         to remove translation in XY.
      2. Subtracting the mean Z of both sides to remove Z translation.
      3. Fitting (s, yaw) by minimising the Z-only error (since X, Y
         of cameras are not known, they cannot enter the fit).
      4. Recovering translation in XY from the centroid relation,
         and translation in Z from the mean-Z relation.

    Because X and Y of cameras are not pinned, the in-plane rotation
    is determined only by the Z-pattern of cameras (i.e. the relative
    heights).  This is sufficient when cameras are at meaningfully
    different heights, which is the case for caldata_refracted_step3
    (range +318 to +464 mm).

    Note: for now this routine uses the simplest possible Z-only
    approach: it scales by (Z_range_known / Z_range_reconstructed),
    translates so means align, and KEEPS the horizontal frame from
    the reconstruction unchanged.  More elaborate fits are possible
    but this is the honest minimum.

    Parameters
    ----------
    Pe, Xe, Ce, Re : as in apply_camera_anchor_xyz
    known_z_heights : (N,) array of known camera Z heights (mm).
    noise_std_mm : float, default 0.0
        Gaussian noise added to known_z_heights to simulate
        ruler-measurement uncertainty.
    rng_seed : Optional[int]
    verbose : bool

    Returns
    -------
    Pe_out, Xe_out, Ce_out, Re_out, info
    """
    info = {
        'mode': 'z_only',
        'n_cameras': Ce.shape[1],
        'noise_std_mm': noise_std_mm,
        'applied': False,
    }

    Ce_xyz = (Ce[:3, :] / Ce[3:4, :])
    z_recon = Ce_xyz[2, :]
    z_known = np.asarray(known_z_heights, dtype=float).ravel().copy()

    if z_known.shape != z_recon.shape:
        info['reason'] = (f'shape mismatch: known z {z_known.shape} '
                          f'vs reconstructed z {z_recon.shape}')
        if verbose:
            print(f"  Camera anchor (Z) SKIPPED: {info['reason']}")
        return Pe, Xe, Ce, Re, info

    if noise_std_mm > 0.0:
        rng = np.random.default_rng(rng_seed)
        z_known = z_known + rng.normal(0.0, noise_std_mm, size=z_known.shape)

    # Sanity check: reconstructed Z values must have some spread, otherwise
    # there's nothing for the scale to lock onto.
    if float(np.var(z_recon)) < 1e-12:
        info['reason'] = 'reconstructed Z values are all equal'
        if verbose:
            print(f"  Camera anchor (Z) SKIPPED: {info['reason']}")
        return Pe, Xe, Ce, Re, info

    # Try both Z-orientations of the reconstruction (flip and no-flip).
    # For the Z-only case the "rotation" component of the similarity is
    # the identity and the fit is reduced to (s, t_z), but the two-sided
    # test still matters because it picks the global Z-sign of the
    # reconstruction.
    (Pe_use, Xe_use, Ce_use, Re_use,
     s, R, t, residual_rms,
     z_flipped, z_diag) = _try_both_z_orientations(
        Pe, Xe, Ce, Re, known_for_fit=z_known,
        z_only=True, glass_top=0.0, verbose=verbose,
    )

    if not np.isfinite(s) or s <= 0.0:
        info['reason'] = f'invalid scale {s}'
        if verbose:
            print(f"  Camera anchor (Z) SKIPPED: {info['reason']}")
        return Pe, Xe, Ce, Re, info

    SCALE_LIMITS = (1e-3, 1e+3)
    if not (SCALE_LIMITS[0] <= s <= SCALE_LIMITS[1]):
        info['reason'] = (f'scale {s:.3e} outside safety bounds '
                          f'{SCALE_LIMITS}')
        if verbose:
            print(f"  Camera anchor (Z) SKIPPED: {info['reason']}")
        return Pe, Xe, Ce, Re, info

    # Apply the transform to the orientation-selected version of the
    # reconstruction (which may be Z-flipped relative to the input).
    Pe_out, Xe_out, Ce_out, Re_out = _apply_similarity_to_reconstruction(
        Pe_use, Xe_use, Ce_use, Re_use, s, R, t)

    Ce_after = (Ce_out[:3, :] / Ce_out[3:4, :])
    per_cam_z_err = Ce_after[2, :] - z_known
    z_residual_rms = float(np.sqrt(np.mean(per_cam_z_err ** 2)))

    info['scale'] = s
    info['translation'] = t.tolist()
    info['z_residual_rms_mm'] = z_residual_rms
    info['z_flipped'] = z_flipped
    info['z_flip_diagnostic'] = z_diag
    info['applied'] = True
    info['reason'] = 'ok'

    if verbose:
        flip_str = " (Z-flipped)" if z_flipped else ""
        print(f"  Camera anchor (Z-only): scale={s:.4f}, "
              f"Z residual RMS={z_residual_rms:.2f} mm{flip_str}")
        if noise_std_mm > 0.0:
            print(f"    Noise σ = {noise_std_mm:.1f} mm added to prior")
        Xe_after_xyz = (Xe_out[:3, :] / Xe_out[3:4, :])
        z_min, z_max = float(Xe_after_xyz[2,:].min()), float(Xe_after_xyz[2,:].max())
        below = float(np.mean(Xe_after_xyz[2,:] <= 0.0)) * 100
        print(f"    Particle Z range: [{z_min:+.1f}, {z_max:+.1f}] mm, "
              f"{below:.1f}% below glass")
        for i in range(z_recon.size):
            print(f"    Cam {i+1}: Z err = {per_cam_z_err[i]:+6.2f} mm "
                  f"(recon Z = {Ce_after[2,i]:+7.1f}, "
                  f"known Z = {z_known[i]:+7.1f})")

    return Pe_out, Xe_out, Ce_out, Re_out, info


# =============================================================
# Self-test
# =============================================================
if __name__ == "__main__":
    """
    Sanity test: construct a synthetic reconstruction with known cameras,
    apply a known similarity transform (scale + rotation + translation),
    and verify that the anchor recovers the inverse transform.
    """
    print("=" * 60)
    print("Camera_anchor self-test")
    print("=" * 60)
    np.random.seed(42)

    # Ground-truth cameras (mimicking caldata_refracted_step3).
    cam_truth = np.array([
        [+481.76, +270.19, +442.75],
        [+153.97, +584.16, +398.96],
        [+113.55, +372.87, +464.47],
        [+384.44, +115.55, +318.12],
    ]).T  # (3, 4)
    print(f"\nGround truth cameras (mm):")
    for i in range(4):
        print(f"  Cam {i+1}: ({cam_truth[0,i]:+8.2f}, "
              f"{cam_truth[1,i]:+8.2f}, {cam_truth[2,i]:+8.2f})")

    # Synthetic projective reconstruction: scaled, rotated, translated.
    # This mimics what the projective pipeline would produce — internally
    # consistent but in a "wrong" coordinate frame.
    rng = np.random.default_rng(0)
    s_true = 0.02                       # scale 50x off
    theta = np.radians(37.0)
    Rz = np.array([[ np.cos(theta), -np.sin(theta), 0],
                   [ np.sin(theta),  np.cos(theta), 0],
                   [ 0, 0, 1]])
    t_true = np.array([5.0, -3.0, 2.0])

    cam_recon = (cam_truth - t_true[:, None]) / s_true
    cam_recon = Rz.T @ cam_recon   # apply inverse rotation
    print(f"\nReconstructed cameras (projective frame):")
    for i in range(4):
        print(f"  Cam {i+1}: ({cam_recon[0,i]:+8.3f}, "
              f"{cam_recon[1,i]:+8.3f}, {cam_recon[2,i]:+8.3f})")

    # Wrap into the (Pe, Xe, Ce, Re) format expected by the anchor.
    n_cams = 4
    Ce = np.vstack([cam_recon, np.ones((1, n_cams))])
    n_pts = 50
    pts = rng.normal(size=(3, n_pts)) * 50.0
    Xe = np.vstack([pts, np.ones((1, n_pts))])
    Pe = np.tile(np.hstack([np.eye(3), np.zeros((3, 1))]), (n_cams, 1))
    Re = np.tile(np.eye(3), (n_cams, 1))

    # ----------------------------------------------------------
    # Test 1 — full XYZ anchor with noiseless prior
    # ----------------------------------------------------------
    print(f"\n[Test 1] XYZ anchor, noiseless prior")
    Pe1, Xe1, Ce1, Re1, info1 = apply_camera_anchor_xyz(
        Pe, Xe, Ce, Re, known_positions=cam_truth,
        noise_std_mm=0.0, verbose=True)
    assert info1['applied']
    assert info1['fit_residual_rms_mm'] < 1e-6, \
        f"noiseless residual should be ~0; got {info1['fit_residual_rms_mm']}"
    print("  PASS — recovered ground truth to machine precision")

    # ----------------------------------------------------------
    # Test 2 — full XYZ anchor with 10 mm noise
    # ----------------------------------------------------------
    print(f"\n[Test 2] XYZ anchor, 10 mm noise on prior")
    Pe2, Xe2, Ce2, Re2, info2 = apply_camera_anchor_xyz(
        Pe, Xe, Ce, Re, known_positions=cam_truth,
        noise_std_mm=10.0, rng_seed=1, verbose=True)
    assert info2['applied']
    # With 10 mm noise on 4 cameras, residual should be ~ 10 mm.
    assert info2['fit_residual_rms_mm'] < 30.0, \
        f"10 mm noise should give residual ~10 mm; got {info2['fit_residual_rms_mm']}"
    print(f"  PASS — residual {info2['fit_residual_rms_mm']:.2f} mm "
          f"is consistent with 10 mm noise")

    # ----------------------------------------------------------
    # Test 3 — Z-only anchor with 5 mm noise
    # ----------------------------------------------------------
    print(f"\n[Test 3] Z-only anchor, 5 mm noise")
    z_known = cam_truth[2, :]
    Pe3, Xe3, Ce3, Re3, info3 = apply_camera_anchor_z(
        Pe, Xe, Ce, Re, known_z_heights=z_known,
        noise_std_mm=5.0, rng_seed=2, verbose=True)
    assert info3['applied']
    # Z residual should be near the noise level.
    assert info3['z_residual_rms_mm'] < 15.0, \
        f"5 mm Z noise should give Z residual ~5 mm; got {info3['z_residual_rms_mm']}"
    print(f"  PASS — Z residual {info3['z_residual_rms_mm']:.2f} mm "
          f"is consistent with 5 mm noise")

    # ----------------------------------------------------------
    # Test 4 — reprojection invariance under the XYZ anchor
    # ----------------------------------------------------------
    print(f"\n[Test 4] Reprojection invariance under similarity transform")
    # Before anchor: project particles to image space using Pe.
    img_before = Pe @ Xe  # (3n, M)
    img_before = img_before / img_before[2::3, :].repeat(3, axis=0)

    # After anchor (Test 1): same calculation with Pe1, Xe1.
    img_after = Pe1 @ Xe1
    img_after = img_after / img_after[2::3, :].repeat(3, axis=0)

    diff = float(np.max(np.abs(img_before - img_after)))
    print(f"  Max pixel difference: {diff:.3e}")
    assert diff < 1e-6, f"reprojection should be invariant; got {diff}"
    print("  PASS — reprojection is invariant under the similarity transform")

    # ----------------------------------------------------------
    # Test 5 — reflection avoidance
    # ----------------------------------------------------------
    print(f"\n[Test 5] Reflection avoidance (det(R) = +1)")
    R5 = np.array(info1['rotation'])
    det_R = float(np.linalg.det(R5))
    print(f"  det(R) = {det_R:.6f}")
    assert abs(det_R - 1.0) < 1e-6, f"R must be a proper rotation; det = {det_R}"
    print("  PASS — R is a proper rotation (no reflection)")

    # ----------------------------------------------------------
    # Test 6 — Z-flip detection: reconstruction with inverted Z
    # ----------------------------------------------------------
    print(f"\n[Test 6] Z-flip detection")
    # Take the existing reconstruction and flip its Z axis.  This mimics
    # a Svoboda reconstruction that happens to be on the wrong side of
    # the Z-sign ambiguity.  The anchor should detect the flip and
    # correct it, ending with cameras at the right positions.
    Ce_flipped = Ce.copy()
    Ce_flipped[2, :] = -Ce_flipped[2, :]
    Xe_flipped = Xe.copy()
    Xe_flipped[2, :] = -Xe_flipped[2, :]
    Pe_flipped = Pe.copy()
    Pe_flipped[:, 2] = -Pe_flipped[:, 2]
    # Particles below glass after the flip, so that the flipped
    # orientation is the geometrically correct one.
    Xe_flipped[2, :] -= 200.0  # shift particles down

    Pe6, Xe6, Ce6, Re6, info6 = apply_camera_anchor_xyz(
        Pe_flipped, Xe_flipped, Ce_flipped, Re,
        known_positions=cam_truth, noise_std_mm=0.0, verbose=True)
    assert info6['applied']
    print(f"  z_flipped flag: {info6['z_flipped']}")
    print(f"  noflip residual: {info6['z_flip_diagnostic']['noflip_residual_mm']:.2f} mm")
    print(f"  flip residual:   {info6['z_flip_diagnostic']['flip_residual_mm']:.2f} mm")
    assert info6['z_flipped'] is True, \
        "Should have flipped Z (input was flipped relative to truth)"
    assert info6['fit_residual_rms_mm'] < 1e-6, \
        f"After correct flip, residual should be ~0; got {info6['fit_residual_rms_mm']}"
    print("  PASS — Z-flip correctly detected and applied")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
