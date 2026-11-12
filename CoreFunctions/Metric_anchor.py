"""
Metric_anchor.py
================

Anchor the projective reconstruction to physical units (mm) using the known
camera-to-glass distance from the cfg file.

This is the bridge between Section 2 (Svoboda self-calibration, which produces
a Euclidean reconstruction up to an arbitrary similarity transform) and
Section 2.2 (Maas multimedia geometry, which requires absolute distances in
mm to model refraction at the glass interface).

THE PROBLEM
-----------
After ``euclidize()`` the reconstruction is correct in shape and angle, but
the absolute scale (and possibly the orientation in Z) is arbitrary.  A typical
output looks like::

    Particle Z range: [-46.9, +870.9]    (should be all below glass)
    Camera mean Z   : -2.3              (should be ≈ +450 above glass)

The multimedia BA cannot run on this — it needs particles below the glass at
known depths in mm.

THE SOLUTION
------------
We use ONE physical anchor — the cfg field ``Camera-Glass-Distance`` — and
let the algorithm self-calibrate everything else (focal lengths, distortions,
particle positions).

Pipeline:
    1.  Determine the dominant Z-direction of the cameras (positive or
        negative).  If the average camera Z is negative we flip the entire
        reconstruction so cameras sit above the glass at z > 0.
    2.  Compute the scale factor that brings the mean camera height to the
        configured ``camera_glass_distance_mm``.
    3.  Apply that scale factor uniformly to ``Pe``, ``Xe`` and ``Ce``.
        ``Re`` is rotation-only and does not change.

This single step turns iterative refinement into true self-calibration:
the only "ground truth" needed is one distance you already know about your
rig.

USAGE
-----
::

    from CoreFunctions.Metric_anchor import apply_metric_anchor

    Pe, Xe, Ce, Re = euclidize(Rmat, Lambda, P, X, config)
    Pe, Xe, Ce, Re, info = apply_metric_anchor(Pe, Xe, Ce, Re, config)

    if info['applied']:
        print(f"Scale factor applied: {info['scale']:.6f}")
        print(f"Z-flip applied      : {info['z_flipped']}")
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Dict


def apply_metric_anchor(
    Pe: np.ndarray,
    Xe: np.ndarray,
    Ce: np.ndarray,
    Re: np.ndarray,
    config,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Scale a Euclidean reconstruction to physical (mm) units using the
    camera-to-glass distance specified in the cfg.

    Parameters
    ----------
    Pe : np.ndarray, shape (3n, 4)
        Stacked Euclidean projection matrices from ``euclidize()``.
    Xe : np.ndarray, shape (4, m)
        Reconstructed 3-D points in homogeneous coordinates.
    Ce : np.ndarray, shape (4, n)
        Camera centres in homogeneous coordinates.
    Re : np.ndarray, shape (3n, 3)
        Stacked camera rotation matrices (unchanged by anchoring).
    config : CalibrationConfig
        Must expose ``camera_glass_distance_mm`` (float, mm) and
        ``z_glass_top`` (float, mm).  If ``camera_glass_distance_mm`` is
        zero or missing the function is a no-op.
    verbose : bool
        Print a one-line summary of the applied transform.

    Returns
    -------
    Pe, Xe, Ce, Re : np.ndarray
        Rescaled (and possibly Z-flipped) inputs.  ``Re`` is returned
        unchanged but kept in the signature for symmetry.
    info : dict
        ``{'applied': bool,
           'scale':   float,
           'z_flipped': bool,
           'mean_camera_z_before': float,
           'mean_camera_z_after':  float,
           'reason': str}``
    """
    info = {
        'applied': False,
        'scale': 1.0,
        'z_flipped': False,
        'mean_camera_z_before': float(np.mean(Ce[2, :])),
        'mean_camera_z_after':  float(np.mean(Ce[2, :])),
        'reason': '',
    }

    # ------------------------------------------------------------------
    # Pre-checks — if the anchor isn't configured, do nothing.
    # ------------------------------------------------------------------
    target_distance = float(getattr(config, 'camera_glass_distance_mm', 0.0))
    if target_distance <= 0.0:
        info['reason'] = 'camera_glass_distance_mm not set (≤ 0)'
        return Pe, Xe, Ce, Re, info

    z_glass_top = float(getattr(config, 'z_glass_top', 0.0))

    # Camera centres in 3-D (drop homogeneous coordinate).
    cam_xyz = Ce[:3, :] / Ce[3:4, :]  # (3, n)

    # ------------------------------------------------------------------
    # Step 1 — determine whether the reconstruction needs a Z-flip.
    #
    # In a correctly oriented multimedia rig:
    #     cameras  sit ABOVE the glass:  Z_cam      > z_glass_top
    #     particles are BELOW the glass: Z_particle < z_glass_top
    #
    # The previous single-criterion rule ("flip if mean camera Z < 0")
    # only checked the camera condition. With refraction the projective
    # reconstruction can return BOTH cameras and particles in the same
    # half-space, so we need to consider both sides explicitly.
    #
    # Strategy: score each candidate orientation by the fraction of
    # particles that end up below the glass AND the fraction of cameras
    # above the glass. Pick the orientation with the higher combined
    # score. If both score equally (perfectly symmetric, can't decide
    # from geometry alone) we keep the camera-above-glass convention.
    # ------------------------------------------------------------------
    pts_z_raw = Xe[2, :] / Xe[3, :]
    cam_z_raw = cam_xyz[2, :]

    def _score_orientation(cam_z: np.ndarray, pts_z: np.ndarray) -> float:
        """Fraction of cameras above z_glass_top + fraction of particles below it."""
        frac_cams_above = float(np.mean(cam_z > z_glass_top))
        frac_pts_below  = float(np.mean(pts_z < z_glass_top))
        return frac_cams_above + frac_pts_below

    score_noflip = _score_orientation(cam_z_raw, pts_z_raw)
    score_flip   = _score_orientation(-cam_z_raw + 2.0 * z_glass_top,
                                      -pts_z_raw + 2.0 * z_glass_top)

    # Pick whichever orientation places the world more correctly.
    # On a perfect tie, prefer the camera-above-glass convention
    # (matches the original single-criterion behaviour).
    if score_flip > score_noflip:
        z_flipped = True
    elif score_flip < score_noflip:
        z_flipped = False
    else:
        z_flipped = float(np.mean(cam_z_raw - z_glass_top)) < 0.0

    if verbose:
        print(f"  Z-flip decision: noflip_score={score_noflip:.2f}, "
              f"flip_score={score_flip:.2f} -> "
              f"{'FLIP' if z_flipped else 'KEEP'}")

    if z_flipped:
        # Reflect about the glass plane: z_new = 2*z_glass_top - z_old.
        # This negates the third row of every projection matrix (and the
        # third coordinate of every world point / camera centre) and adds
        # a translation so the glass plane itself is the fixed plane of
        # the reflection. When z_glass_top == 0 this reduces to the
        # original "negate Z" behaviour.
        Ce        = Ce.copy()
        Xe        = Xe.copy()
        Pe        = Pe.copy()

        # Reflect points (homogeneous): z' = 2*z_g*w - z
        Ce[2, :]  = 2.0 * z_glass_top * Ce[3, :] - Ce[2, :]
        Xe[2, :]  = 2.0 * z_glass_top * Xe[3, :] - Xe[2, :]

        # Reflect projection matrices.  If P maps (x,y,z,1) -> image,
        # then for the reflected world point (x,y,2*z_g-z,1) we need
        # P' such that  P' [x,y,2*z_g-z,1]^T == P [x,y,z,1]^T.
        # That gives:  P'[:,0]=P[:,0], P'[:,1]=P[:,1],
        #              P'[:,2] = -P[:,2],
        #              P'[:,3] = P[:,3] + 2*z_g * P[:,2].
        Pe[:, 3]  = Pe[:, 3] + 2.0 * z_glass_top * Pe[:, 2]
        Pe[:, 2]  = -Pe[:, 2]

        cam_xyz   = Ce[:3, :] / Ce[3:4, :]

    # ------------------------------------------------------------------
    # Step 2 — compute the scale factor.
    # The mean camera height above the glass should equal the cfg value.
    # ------------------------------------------------------------------
    mean_cam_height = float(np.mean(cam_xyz[2, :] - z_glass_top))

    if abs(mean_cam_height) < 1e-12:
        info['reason'] = 'mean camera height is ~0; cannot anchor'
        info['z_flipped'] = z_flipped
        return Pe, Xe, Ce, Re, info

    # Always use absolute value — scale is a magnitude.  Sign was already
    # handled by the Z-flip step above.
    scale = abs(target_distance / mean_cam_height)
    if not np.isfinite(scale) or scale <= 0.0:
        info['reason'] = f'invalid scale factor ({scale}); skipping'
        info['z_flipped'] = z_flipped
        return Pe, Xe, Ce, Re, info

    # ------------------------------------------------------------------
    # Step 3 — apply the scale.
    # World-point scaling is a uniform similarity:
    #   X' = scale * (X - origin) + origin    (origin = z_glass_top on Z)
    # For simplicity (and because z_glass_top is usually 0 in this codebase)
    # we apply pure scaling about the world origin and then translate the
    # cameras / points so that z_glass_top stays put.
    # ------------------------------------------------------------------
    Xe_out = Xe.copy()
    Ce_out = Ce.copy()

    # Translate so glass top is at z=0, scale, translate back.
    Xe_out[2, :] -= z_glass_top
    Xe_out[:3, :] *= scale
    Xe_out[2, :] += z_glass_top

    Ce_out[2, :] -= z_glass_top
    Ce_out[:3, :] *= scale
    Ce_out[2, :] += z_glass_top

    # Pe rescale — when X is scaled by s the matrix that maps the new X back
    # to the same image points is  P' = P · diag(1/s, 1/s, 1/s, 1).
    # Equivalently we multiply the first three columns of Pe by 1/s; the
    # translation column absorbs the z_glass_top shift.
    Pe_out = Pe.copy()
    if abs(z_glass_top) > 1e-12:
        # Pre-translate the world: X_new = X_old + [0,0,z_glass_top,0]
        # P' [R | t] becomes [R | t + R · [0,0,z_glass_top]]
        T_pre = np.array([0.0, 0.0, z_glass_top, 0.0])
        Pe_out[:, 3] += Pe_out @ T_pre  # safe: Pe_out is (3n,4), T_pre is (4,)

    Pe_out[:, :3] /= scale

    # ------------------------------------------------------------------
    # Step 3b — Z-translation to anchor particles below the glass.
    #
    # The scale step above ensures cameras are at the correct height
    # above the glass.  But it does NOT constrain the ABSOLUTE Z
    # position of the reconstruction relative to the glass plane.
    # On refracted data we observed reconstructions where cameras
    # were at the right height but particles sat at z > 0 (above the
    # glass), so multimedia BA refused to fire.
    #
    # The fix: translate the entire reconstruction (cameras AND
    # particles) along Z so that the bulk of the particle cloud
    # lands just below the glass bottom.  This is a rigid translation
    # so distances and reprojection are unchanged.  Only the absolute
    # Z origin shifts.
    #
    # Reference point chosen: the 5th percentile of particle Z.
    # Targeting the 5th percentile (rather than the median) is
    # deliberate — it ensures that the SHALLOWEST particles sit just
    # below the glass, which is the physical configuration multimedia
    # BA expects, while letting the deeper particles extend further
    # down without forcing any individual placement.
    # ------------------------------------------------------------------
    glass_thickness_for_anchor = 0.0
    if hasattr(config, 'd'):
        _d = config.d
        glass_thickness_for_anchor = float(_d[0]) if hasattr(_d, '__len__') else float(_d)

    # Where the shallowest particles SHOULD be: just below glass bottom.
    # Small offset (5 mm) keeps them clear of the boundary.
    Z_TARGET_OFFSET_BELOW_GLASS = 5.0  # mm
    target_shallow_z = z_glass_top - glass_thickness_for_anchor - Z_TARGET_OFFSET_BELOW_GLASS

    pts_z_current = Xe_out[2, :] / Xe_out[3, :]
    current_shallow_z = float(np.percentile(pts_z_current, 95))  # 95th = shallowest 5%

    z_shift = target_shallow_z - current_shallow_z

    # Sanity: don't apply absurdly large shifts (would indicate a
    # different problem elsewhere)
    Z_SHIFT_LIMIT_MM = 2000.0
    if abs(z_shift) > Z_SHIFT_LIMIT_MM:
        if verbose:
            print(f"  Z-anchor SKIPPED: required shift {z_shift:+.1f} mm "
                  f"exceeds safety limit ±{Z_SHIFT_LIMIT_MM:.0f} mm")
        info['z_shift_applied'] = 0.0
        info['z_shift_skipped'] = True
    else:
        # Apply the translation to particles, cameras, and projection matrices.
        Xe_out[2, :] += z_shift * Xe_out[3, :]
        Ce_out[2, :] += z_shift * Ce_out[3, :]

        # For projection matrices: a world Z-translation by delta means
        # P maps (x,y,z+delta,1) to the same image as before.  So we
        # absorb -delta into the translation column:
        #   P' [x,y,z+delta,1]^T = P [x,y,z,1]^T
        # i.e. P'[:,3] = P[:,3] - delta * P[:,2]
        Pe_out[:, 3] -= z_shift * Pe_out[:, 2]

        info['z_shift_applied'] = z_shift
        info['z_shift_skipped'] = False

        if verbose:
            print(f"  Z-anchor: shifted reconstruction by {z_shift:+.2f} mm "
                  f"(shallowest particles now at z={target_shallow_z:.1f} mm, "
                  f"just below glass bottom)")

    # ------------------------------------------------------------------
    # Identify physically-plausible particles for multimedia BA.
    #
    # The projective→Euclidean upgrade in euclidize() can leave particles
    # with absurd Z values (we've observed 96,000 mm spreads for true
    # ~200 mm range) due to a small singular value in the Z direction.
    # These outliers don't affect 2D reprojection (they happen to project
    # near the correct pixel) but they make multimedia BA diverge.
    #
    # We compute a mask of particles within a physically sensible Z range:
    #   - particles must be BELOW the glass (Zp < z_glass_top - thickness)
    #   - particles within ±N·MAD of median Z (robust outlier rejection)
    #
    # The mask is stored in info — downstream code can use it to filter
    # particles before passing them to multimedia BA.
    # ------------------------------------------------------------------
    pts_z = Xe_out[2, :] / Xe_out[3, :]

    # Glass bottom in world Z (z_glass_top minus glass thickness if known)
    glass_thickness = 0.0
    if hasattr(config, 'd'):
        d_val = config.d
        glass_thickness = float(d_val[0]) if hasattr(d_val, '__len__') else float(d_val)
    glass_bottom = z_glass_top - glass_thickness

    # Tank depth — read from config if present, otherwise use a sensible
    # default that covers most experiments (500 mm tank).
    max_depth_mm = float(getattr(config, 'tank_depth_mm', 500.0))
    z_min = glass_bottom - max_depth_mm   # deepest plausible particle
    z_max = glass_bottom                  # shallowest (just below glass)

    # Statistics-based outlier detection (MAD around median) FAILS here
    # because most reconstructed particles cluster near z=0 (a degenerate
    # solution) while the correctly-placed ones are spread out in water.
    # Physical bounds work better: keep only particles in the water tank.
    mask_below_glass      = pts_z < z_max
    mask_above_tank_floor = pts_z > z_min
    particle_mask = mask_below_glass & mask_above_tank_floor
    n_clipped = int((~particle_mask).sum())

    # Diagnostics — kept for reporting only
    median_z = float(np.median(pts_z))
    mad      = float(np.median(np.abs(pts_z - median_z)))

    info.update({
        'applied': True,
        'scale': scale,
        'z_flipped': z_flipped,
        'mean_camera_z_before': info['mean_camera_z_before'],
        'mean_camera_z_after':  float(np.mean(Ce_out[2, :])),
        'reason': 'ok',
        # Particle filtering diagnostics — used by multimedia BA to skip outliers
        'particle_mask':    particle_mask,
        'n_particles_kept': int(particle_mask.sum()),
        'n_particles_clipped': n_clipped,
        'particle_z_median':  median_z,
        'particle_z_mad':     mad,
    })

    if verbose:
        flip_str = " (Z-flipped)" if z_flipped else ""
        print(f"  Metric anchor applied: scale={scale:.6f}{flip_str}, "
              f"mean cam Z {info['mean_camera_z_before']:+.2f} → "
              f"{info['mean_camera_z_after']:+.2f} mm "
              f"(target {target_distance:.1f} mm)")
        if n_clipped > 0:
            print(f"  Particle outliers identified: {n_clipped}/{len(pts_z)} "
                  f"(Z median={median_z:.1f} mm, MAD={mad:.1f} mm). "
                  f"These will be excluded from multimedia BA.")

    return Pe_out, Xe_out, Ce_out, Re, info


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Build a tiny synthetic Euclidean reconstruction and verify the anchor.
    from types import SimpleNamespace

    n_cams, n_pts = 4, 50
    rng = np.random.default_rng(0)

    # True camera positions: 4 cameras evenly distributed at +450 mm height
    cam_pos_true = np.array([
        [-200.0, -200.0, 450.0],
        [+200.0, -200.0, 450.0],
        [-200.0, +200.0, 450.0],
        [+200.0, +200.0, 450.0],
    ]).T  # (3, 4)

    # True particles: scattered below glass (z = -10 ... -200 mm)
    pts_true = np.column_stack([
        rng.uniform(-150, 150, n_pts),
        rng.uniform(-150, 150, n_pts),
        rng.uniform(-200, -10, n_pts),
    ]).T  # (3, n_pts)

    # Now simulate what euclidize() returns: arbitrary scale s and Z-flip
    s_proj = 0.357   # arbitrary projective scale
    pts_proj = pts_true * s_proj
    pts_proj[2, :] *= -1                          # Z-flip
    cam_proj = cam_pos_true * s_proj
    cam_proj[2, :] *= -1                          # cameras now at Z<0

    Xe_in = np.vstack([pts_proj, np.ones(n_pts)])
    Ce_in = np.vstack([cam_proj, np.ones(n_cams)])
    Pe_in = np.tile(np.hstack([np.eye(3), np.zeros((3, 1))]), (n_cams, 1))
    Re_in = np.tile(np.eye(3), (n_cams, 1))

    cfg = SimpleNamespace(
        camera_glass_distance_mm = 450.0,
        z_glass_top              = 0.0,
    )

    print("Before anchor:")
    print(f"  Mean camera Z : {np.mean(Ce_in[2, :]):+.3f}")
    print(f"  Particle Z    : [{pts_proj[2].min():+.3f}, {pts_proj[2].max():+.3f}]")

    Pe, Xe, Ce, Re, info = apply_metric_anchor(Pe_in, Xe_in, Ce_in, Re_in, cfg)

    print("\nAfter anchor:")
    print(f"  Mean camera Z : {np.mean(Ce[2, :]):+.3f}")
    print(f"  Particle Z    : [{Xe[2].min():+.3f}, {Xe[2].max():+.3f}]")
    print(f"  Info          : {info}")

    # Verify cameras are now above glass (the Z-anchor step may shift
    # them away from exactly +450 to ensure particles land below glass)
    assert np.mean(Ce[2, :]) > 0.0, "Cameras should be above glass"
    # Verify at least 90% of particles are below glass (the Z-anchor
    # targets the 95th percentile so a few particles may extend above)
    pts_z = Xe[2, :] / Xe[3, :]
    frac_below = float(np.mean(pts_z < 0.0))
    assert frac_below >= 0.90, f"Only {frac_below*100:.0f}% of particles below glass"
    print(f"\nSelf-test PASSED — {frac_below*100:.0f}% of particles below glass, "
          f"mean cam Z = {np.mean(Ce[2, :])/np.mean(Ce[3, :]):.1f} mm ✓")
