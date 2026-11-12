"""
multimedia_bundle.py — Multimedia bundle adjustment in metric world space.

Corrects the systematic bias introduced by the standard projective BA
(bundle_PX_proj) treating refracted observations as pinhole. Uses the full
Maas (1996) multimedia functional model (observation_equation_221) as the
residual function, and numerical_jacobian_221 for camera parameter derivatives.

Coordinate system
-----------------
Everything in this module is in metric world space or pixel space.
There is no coordinate barrier (see eval_y_and_dy_EXPLANATION.md).

    cameras_mm[k].position  : world mm  (from euclidize → _build_camera_params)
    X_world[:, n]           : world mm  (Xe dehomogenised)
    x_obs, y_obs            : pixel     (principal point already added in GoCal)
    observation_equation_221: world mm + pixel → pixel residuals
    numerical_jacobian_221  : ∂(pixel residuals) / ∂(16 Maas camera params)

Pipeline position
-----------------
Runs AFTER the standard BA has converged and reprerror() has set
cam[i]['recandvis'] / cam[i]['visandrec']:

    fill_mm_bundle → euclidize → cameras_mm (biased)
                                      ↓
                          multimedia_bundle_adjustment   ← THIS FILE
                                      ↓
                          cameras_mm (corrected)  →  LUT rebuild  →  reprerror
"""

import numpy as np
from copy import deepcopy
from scipy.optimize import least_squares
from typing import List, Optional, Dict, Tuple

from Multimedia.Iterative_radial_shift import observation_equation_221, numerical_jacobian_221
from Multimedia.Multimedia_params import CameraParams


# ── 16-parameter layout (matches numerical_jacobian_221 column order) ─────────
_PARAM_NAMES = [
    'X0', 'Y0', 'Z0',          # 0-2  camera centre (mm)
    'omega', 'phi', 'kappa',   # 3-5  Euler angles (radians)
    'c',                        # 6    principal distance (pixels)
    'xh', 'yh',                 # 7-8  principal point (pixels)
    'k1', 'k2', 'k3',          # 9-11 radial distortion
    'p1', 'p2',                 # 12-13 tangential distortion
    'a1', 'a2',                 # 14-15 affine/shear correction
]

# Default: optimise position (0-2) and orientation (3-5) only.
# These absorb the pinhole bias. Intrinsics come from a separate calibration.
_DEFAULT_CAM_MASK = np.array([0, 1, 2, 3, 4, 5], dtype=int)


# ── Private helpers ────────────────────────────────────────────────────────────

def _pack_cam(cam_obj: CameraParams) -> np.ndarray:
    """Flatten one CameraParams into a (16,) array in _PARAM_NAMES order."""
    return np.array([
        cam_obj.position[0], cam_obj.position[1], cam_obj.position[2],
        cam_obj.angles[0],   cam_obj.angles[1],   cam_obj.angles[2],
        cam_obj.c,   cam_obj.xh,  cam_obj.yh,
        cam_obj.k1,  cam_obj.k2,  cam_obj.k3,
        cam_obj.p1,  cam_obj.p2,
        cam_obj.a1,  cam_obj.a2,
    ], dtype=float)


def _unpack_cam(cam_obj: CameraParams, all16: np.ndarray) -> CameraParams:
    """Create a new CameraParams from a (16,) array, preserving cam_id and mm."""
    return CameraParams(
        cam_id   = cam_obj.cam_id,
        position = all16[0:3].copy(),
        angles   = all16[3:6].copy(),
        c        = float(all16[6]),
        xh       = float(all16[7]),
        yh       = float(all16[8]),
        k1       = float(all16[9]),
        k2       = float(all16[10]),
        k3       = float(all16[11]),
        p1       = float(all16[12]),
        p2       = float(all16[13]),
        a1       = float(all16[14]),
        a2       = float(all16[15]),
        mm       = cam_obj.mm,
    )


def _build_observations(cam: list, K: int) -> list:
    """
    Build a flat list of (cam_idx, xe_col, x_obs, y_obs) from cam structures.

    cam[k]['recandvis'] : indices into Xe columns (3D point index)
    cam[k]['visandrec'] : indices into cam[k]['xgt'] columns (2D obs index)
    cam[k]['xgt']       : observed pixel coordinates (row 0=u, row 1=v)
                          with principal point already added back by GoCal

    Both sets of indices are set by reprerror() — this function must be
    called after reprerror() has run.
    """
    obs = []
    for k in range(K):
        rec = cam[k]['recandvis']   # → Xe column indices
        vis = cam[k]['visandrec']   # → xgt column indices
        for xe_col, xgt_col in zip(rec, vis):
            x_obs = float(cam[k]['xgt'][0, xgt_col])
            y_obs = float(cam[k]['xgt'][1, xgt_col])
            obs.append((k, int(xe_col), x_obs, y_obs))
    return obs


def _point_jacobian(cam_obj: CameraParams, particle: np.ndarray,
                    x_obs: float, y_obs: float,
                    eps: float = 1e-4) -> np.ndarray:
    """
    Central-difference Jacobian of observation_equation_221 w.r.t. particle.

    Returns (2, 3): ∂[vx, vy] / ∂[X, Y, Z] in world mm.
    Finite-differences through the full multimedia functional model —
    no closed-form derivative needed for the iterative radial shift.
    """
    J = np.zeros((2, 3))
    for j in range(3):
        p_fwd = particle.copy(); p_fwd[j] += eps
        p_bwd = particle.copy(); p_bwd[j] -= eps
        try:
            vx_f, vy_f, _ = observation_equation_221(p_fwd, cam_obj, x_obs, y_obs)
            vx_b, vy_b, _ = observation_equation_221(p_bwd, cam_obj, x_obs, y_obs)
            J[0, j] = (vx_f - vx_b) / (2.0 * eps)
            J[1, j] = (vy_f - vy_b) / (2.0 * eps)
        except Exception:
            pass  # degenerate ray — leave zeros
    return J


# ── Geometry pre-validation ──────────────────────────────────────────────────

def to_physical_frame(cameras_mm, Xe3, camera_glass_distance_mm):
    """
    Transform euclidize-frame cameras and particles to the physical glass frame.

    The euclidize reconstruction is in an arbitrary Euclidean frame — unknown
    scale, origin, and orientation relative to the physical world. This function
    resolves all three unknowns using ONE physical measurement:

        camera_glass_distance_mm : perpendicular distance from the camera ring
                                   to the front of the glass surface (mm).
                                   Measurable with a ruler. Always known in a
                                   PTV experiment.

    Algorithm
    ---------
    1. Rotation (glass normal direction):
       The glass normal ≈ mean viewing direction of cameras = (particle_center
       − camera_center) normalised. Camera looks perpendicular to glass in a
       ring PTV setup.

    2. Scale (mm per euclidize unit):
       Project cameras onto the glass normal. The mean projection distance from
       camera ring to the scene origin (≈ glass surface) is Z0_euc.
       scale = camera_glass_distance_mm / Z0_euc

    3. Origin (where is Z=0 = glass surface?):
       After scale: camera position along normal = camera_glass_distance_mm.
       Glass top is at that position minus camera_glass_distance_mm = 0.
       i.e., z_glass_top = 0 in the transformed frame — consistent with the
       physical convention already in the .cfg.

    Parameters
    ----------
    cameras_mm : list of CameraParams
        Camera objects with position in euclidize frame.
    Xe3 : np.ndarray (3, N)
        Dehomogenised 3D points in euclidize frame.
    camera_glass_distance_mm : float
        Physical camera-to-glass distance in mm.

    Returns
    -------
    cam_pos_phys : np.ndarray (K, 3)   camera positions in physical mm frame
    Xe3_phys     : np.ndarray (3, N)   particle positions in physical mm frame
    R_to_phys    : np.ndarray (3, 3)   rotation from euclidize frame to physical
    scale        : float               mm per euclidize unit
    valid        : bool                False if geometry is degenerate
    """
    K = len(cameras_mm)
    cam_pos_euc = np.array([c.position for c in cameras_mm])   # (K, 3)

    cam_center = cam_pos_euc.mean(axis=0)          # mean camera position
    pt_center  = Xe3.mean(axis=1)                  # mean particle position (3,)

    # Glass normal = direction from cameras toward particles
    glass_normal_euc = pt_center - cam_center
    norm_gn = np.linalg.norm(glass_normal_euc)
    if norm_gn < 1e-10:
        return None, None, None, None, False
    glass_normal_euc = glass_normal_euc / norm_gn  # unit vector

    # Project each camera position onto the glass normal
    # Z0_euc = mean signed distance from camera ring to the origin along the normal
    cam_proj = cam_pos_euc @ glass_normal_euc      # (K,) projections along normal
    cam_proj_mean = cam_proj.mean()
    # The "glass" is between camera and particles; treat cam_proj_mean as camera Z
    # after origin shift — i.e., Z0_euc = cam_proj_mean (relative to a scene origin near 0)
    Z0_euc = abs(cam_proj_mean)
    if Z0_euc < 1e-10:
        return None, None, None, None, False

    # Scale: camera_glass_distance_mm corresponds to Z0_euc units
    scale = camera_glass_distance_mm / Z0_euc

    # Build rotation matrix: glass_normal_euc maps to physical -Z axis
    # (cameras above glass → physical Z > 0; particles below → physical Z < -t)
    # physical_Z axis = -glass_normal_euc (flip: larger euclidize "depth" = more negative physical Z)
    z_phys = -glass_normal_euc                    # physical +Z points AWAY from scene

    # Choose x_phys to be roughly aligned with the dominant XY direction
    # (use the camera with largest XY spread for numerical stability)
    cam_xy = cam_pos_euc - cam_center
    cam_xy_proj = cam_xy - (cam_xy @ glass_normal_euc[:, None]) * glass_normal_euc
    spread = np.linalg.norm(cam_xy_proj, axis=1)
    ref_cam_idx = np.argmax(spread)
    x_phys_raw = cam_xy_proj[ref_cam_idx]
    x_phys_raw -= np.dot(x_phys_raw, z_phys) * z_phys
    xn = np.linalg.norm(x_phys_raw)
    if xn < 1e-10:
        x_phys_raw = np.array([1.0, 0.0, 0.0])
        x_phys_raw -= np.dot(x_phys_raw, z_phys) * z_phys
        xn = np.linalg.norm(x_phys_raw)
    x_phys = x_phys_raw / xn
    y_phys = np.cross(z_phys, x_phys)

    # Rotation matrix: columns are physical axes expressed in euclidize coords
    R_to_phys = np.column_stack([x_phys, y_phys, z_phys])   # (3, 3)

    # Apply transform to cameras and particles
    # physical = scale × R_to_phys^T × (euclidize − cam_center) + [0, 0, Z0_mm]
    # where [0, 0, Z0_mm] places cameras at Z = Z0_mm above glass (Z_glass_top = 0)
    origin_offset = np.array([0.0, 0.0, camera_glass_distance_mm])

    cam_pos_phys = scale * (cam_pos_euc - cam_center) @ R_to_phys + origin_offset  # (K,3)
    Xe3_phys     = scale * (Xe3.T - cam_center) @ R_to_phys + origin_offset        # (N,3)
    Xe3_phys     = Xe3_phys.T                                                       # (3,N)

    return cam_pos_phys, Xe3_phys, R_to_phys, scale, True


# ---------------------------------------------------------------------------
# FrameConvention — the SINGLE authority for the camera/cloud/glass frame.
# ---------------------------------------------------------------------------

class FrameConvention:
    """
    One frame, decided once, read by everyone.

    Built ONCE (on pass 1) from to_physical_frame(). It stores the 4x4 similarity
    T (euclidize -> physical) and exposes it so that EVERY downstream consumer —
    triangulation, the LUT, the BA validator — works in the same physical frame
    instead of re-deriving its own. Convention (matches the LUT and the validator):

        glass top    at z = z_glass_top   (= 0)
        glass bottom at z = z_glass_top - t
        cameras      above  (z > z_glass_top)
        particles    below  (z < glass_bottom)

    Because to_physical_frame uses a PROPER rotation from the unambiguous
    camera->particle direction, there is no Z-reflection flip ambiguity — the
    wrong-side cloud that the old simple_metric_anchor produced cannot occur.
    """

    def __init__(self, T, scale, t, z_glass_top, valid, K):
        self.T = T                       # 4x4 euclidize -> physical
        self.Tinv = np.linalg.inv(T) if valid else None
        self.scale = scale
        self.t = float(t)
        self.z_glass_top = float(z_glass_top)
        self.glass_bottom = float(z_glass_top) - float(t)
        self.valid = valid
        self.K = K

    @classmethod
    def from_euclidize(cls, cameras_mm, Xe3, camera_glass_distance_mm,
                       t, z_glass_top=0.0, glass_percentile=95.0):
        """
        Establish the frame ONCE from a euclidize reconstruction.

        Orientation: proper rotation from the camera->particle direction (robust,
        no Z-reflection flip ambiguity).  Glass placement: at the top of the
        particle cloud (glass_percentile of particle depth), so particles land
        BELOW it.  Scale: camera_glass_distance / (mean camera height above that
        glass plane).  This fixes the under-determination that defeats both
        simple_metric_anchor (fragile flip) and to_physical_frame (assumes the
        euclidize origin is the glass).
        """
        cam_pos_euc = np.array([c.position for c in cameras_mm])     # (K,3)
        Xe3 = np.asarray(Xe3)
        cam_center = cam_pos_euc.mean(axis=0)
        pt_center = Xe3.mean(axis=1)

        normal = pt_center - cam_center                              # cameras -> particles
        nn = np.linalg.norm(normal)
        if nn < 1e-10:
            return cls(np.eye(4), 1.0, t, z_glass_top, False, len(cameras_mm))
        z_phys = -normal / nn                                        # physical +Z: particles -> cameras

        # in-plane axes (numerically stable pick, same idea as to_physical_frame)
        cam_xy = cam_pos_euc - cam_center
        cam_xy_proj = cam_xy - (cam_xy @ z_phys)[:, None] * z_phys
        ref = np.argmax(np.linalg.norm(cam_xy_proj, axis=1))
        x_raw = cam_xy_proj[ref]
        x_raw -= np.dot(x_raw, z_phys) * z_phys
        if np.linalg.norm(x_raw) < 1e-10:
            x_raw = np.array([1.0, 0.0, 0.0]); x_raw -= np.dot(x_raw, z_phys) * z_phys
        x_phys = x_raw / np.linalg.norm(x_raw)
        y_phys = np.cross(z_phys, x_phys)
        R = np.column_stack([x_phys, y_phys, z_phys])               # euc axes -> phys

        # oriented z (before scale): (X - cam_center) . z_phys
        oz_cam = (cam_pos_euc - cam_center) @ z_phys                 # ~0
        oz_pt  = (Xe3.T - cam_center) @ z_phys                       # negative (particles below)
        glass_oriented = float(np.percentile(oz_pt, glass_percentile))  # top of cloud
        cam_height = float(oz_cam.mean()) - glass_oriented          # > 0
        if abs(cam_height) < 1e-12:
            return cls(np.eye(4), 1.0, t, z_glass_top, False, len(cameras_mm))
        scale = camera_glass_distance_mm / cam_height

        # X_phys = scale * R^T (X_euc - cam_center) - [0,0, scale*glass_oriented] + [0,0,z_glass_top]
        T = np.eye(4)
        T[:3, :3] = scale * R.T
        shift = -scale * (R.T @ cam_center)
        shift[2] += z_glass_top - scale * glass_oriented
        T[:3, 3] = shift
        return cls(T, scale, t, z_glass_top, True, len(cameras_mm))

    # ---- transforms: bring any reconstruction into THE frame ----
    def points_to_frame(self, Xh):
        """Xh (4,N) homogeneous euclidize -> (4,N) homogeneous physical."""
        Y = self.T @ (Xh / Xh[3:4, :])
        return Y / Y[3:4, :]

    def pe_to_frame(self, Pe):
        """Pe (3K,4) euclidize -> (3K,4) physical (proj invariant w/ points_to_frame)."""
        Pe_out = np.zeros_like(Pe)
        for k in range(Pe.shape[0] // 3):
            Pe_out[3*k:3*k+3, :] = Pe[3*k:3*k+3, :] @ self.Tinv
        return Pe_out

    def cam_positions_to_frame(self, cam_pos_euc):
        """(K,3) euclidize camera centres -> (K,3) physical."""
        H = np.vstack([cam_pos_euc.T, np.ones(cam_pos_euc.shape[0])])
        return (self.T @ H)[:3].T

    # ---- the ONE depth definition every consumer must use ----
    def depth_below_glass(self, X3_phys):
        """Zp for the Maas/LUT model. Positive when the point is below the glass."""
        return -(np.asarray(X3_phys)[2] - self.glass_bottom)

    # ---- the ONE validity check ----
    def is_valid_geometry(self, cam_pos_phys, X3_phys):
        cam_z = np.asarray(cam_pos_phys)[:, 2]
        pz = np.asarray(X3_phys)[2, :]
        cams_above = int(np.sum(cam_z <= self.z_glass_top))
        frac_below = float(np.mean(pz < self.glass_bottom))
        ok = (cams_above == 0) and (frac_below > 0.8)
        return ok, {'cams_below_glass': cams_above,
                    'frac_particles_below': frac_below}


def _validate_geometry(cameras_mm, Xe3, z_glass_top, verbose):
    """
    Pre-flight check before multimedia BA.

    Detects two failure modes:
      1. SCALE MISMATCH  — Xe3 is in projective/normalised units (~0.1–0.3)
         while t is in real mm (5.0). Zp = -(particle_z + t) then gives Zp<0
         for every particle, causing ValueError in multimedia_radial_shift.
         The try/except returns (vx=vy=0) silently -> RMS=0, 1 eval.
      2. CAMERAS BELOW GLASS — Z0 = cam_z - z_glass_top <= 0.

    Returns (ok:bool, message:str). ok=False means the BA should be skipped.
    """
    mm           = cameras_mm[0].mm
    t            = mm.t_total          # glass thickness in mm
    K            = len(cameras_mm)
    glass_bottom = z_glass_top - t

    cam_z   = np.array([c.position[2] for c in cameras_mm])
    Z0_vals = cam_z - z_glass_top
    n_bad_cam = int(np.sum(Z0_vals <= 0))

    particle_z  = Xe3[2, :]
    Zp_vals     = -(particle_z - glass_bottom)  # positive when below glass
    pct_bad_pt  = 100.0 * float(np.sum(Zp_vals <= 0)) / max(len(particle_z), 1)

    cam_c_vals = [c.c for c in cameras_mm]
    n_bad_c    = int(np.sum(np.array(cam_c_vals) < 5.0))

    issues = []

    if n_bad_cam == K:
        issues.append(
            "ALL cameras are at or below the glass top surface "
            f"(z_glass_top={z_glass_top} mm). "
            f"Camera Z positions: {[round(float(z),4) for z in cam_z]}. "
            f"Heights above glass: {[round(float(z),4) for z in Z0_vals]}. "
            "LIKELY CAUSE: euclidize() frame not aligned with physical world — "
            "provide .rad calibration files so euclidize() recovers correct metric "
            "scale and orientation."
        )
    elif n_bad_cam > 0:
        # Strict: ANY camera below the glass means the projective reconstruction
        # is too inconsistent for multimedia BA to recover from.  The refraction
        # math (Iterative_radial_shift) assumes the camera sits above the air-glass
        # interface; a sub-glass camera causes the iterative solver to diverge
        # (radial shifts of thousands of mm).  Better to skip BA than to corrupt
        # the geometry further — the next outer iteration may have a better
        # starting reconstruction once goradf has refined the intrinsics.
        issues.append(
            f"{n_bad_cam}/{K} cameras below glass top. "
            f"Heights above glass: {[round(float(z),4) for z in Z0_vals]}"
        )

    if pct_bad_pt > 80.0:
        issues.append(
            f"{pct_bad_pt:.0f}% of particles have Zp<=0 "
            f"(glass bottom at z={glass_bottom:.2f} mm, "
            f"particle Z range [{float(particle_z.min()):.4f}, {float(particle_z.max()):.4f}]). "
            f"Glass thickness t={t:.1f} mm. "
            "LIKELY CAUSE: Xe is in projective/normalised units, not physical mm. "
            "Provide .rad calibration files with the true focal length so that "
            "euclidize() recovers the correct metric scale."
        )

    if n_bad_c > 0:
        issues.append(
            f"{n_bad_c}/{K} cameras have focal length c<5 px "
            f"(values: {[round(float(c),3) for c in cam_c_vals]}). "
            "Missing .rad calibration files — multimedia BA needs calibrated intrinsics."
        )

    if issues:
        msg = (
            "  Multimedia BA SKIPPED — pre-validation failed:\n"
            + "\n".join(f"    [{i+1}] {iss}" for i, iss in enumerate(issues))
            + "\n  Cameras returned unchanged."
        )
        return False, msg

    return True, "OK"


def validate_multimedia_geometry(cameras_mm, Xe3, z_glass_top,
                                 camera_glass_distance_mm=0.0, verbose=False):
    """
    Public gate for all multimedia operations in GoCal.

    Returns (ok, message, Xe3_out, cam_pos_out) where:
      ok           : bool — True if geometry is valid for multimedia operations
      message      : str  — explanation when ok=False
      Xe3_out      : (3,N) array — particles in the frame used for validation
      cam_pos_out  : (K,3) array — camera positions in the same frame

    When camera_glass_distance_mm > 0: applies to_physical_frame() to convert
    the arbitrary euclidize frame to physical mm before checking.  This is the
    normal path when no .rad files are available.

    When camera_glass_distance_mm == 0: validates the frame as-is (requires
    .rad calibration files so euclidize already produces metric output).
    """
    if camera_glass_distance_mm > 0.0:
        cam_pos_phys, Xe3_phys, R, scale, valid = to_physical_frame(
            cameras_mm, Xe3, camera_glass_distance_mm)
        if not valid:
            return False, "to_physical_frame degenerate — camera/particle geometry invalid", Xe3, np.array([c.position for c in cameras_mm])
        # Build temporary cameras_mm with physical positions for validation
        from copy import deepcopy
        cameras_mm_phys = []
        for k, cam in enumerate(cameras_mm):
            c2 = deepcopy(cam)
            c2.position = cam_pos_phys[k]
            cameras_mm_phys.append(c2)
        ok, msg = _validate_geometry(cameras_mm_phys, Xe3_phys, z_glass_top, verbose)
        return ok, msg, Xe3_phys, cam_pos_phys
    else:
        ok, msg = _validate_geometry(cameras_mm, Xe3, z_glass_top, verbose)
        if not ok:
            # Append actionable instruction when no Camera-Glass-Distance is set
            action = (
                "  ACTION REQUIRED: add to the [Multimedia] section of multicamselfcal.cfg:"
                "\n    Camera-Glass-Distance: <distance_in_mm>"
                "\n  where <distance_in_mm> is the perpendicular distance (mm) from"
                "\n  the camera ring to the glass surface (measure with a ruler)."
                "\n  Without it the physical world frame cannot be established."
            )
            msg = msg + "\n" + action
        return ok, msg, Xe3, np.array([c.position for c in cameras_mm])


# ── Public API ─────────────────────────────────────────────────────────────────

def multimedia_bundle_adjustment(
    cameras_mm: List[CameraParams],
    Xe: np.ndarray,
    cam: list,
    z_glass_top: float = 0.0,
    opt: Optional[Dict] = None,
) -> Tuple[List[CameraParams], np.ndarray, Dict]:
    """
    Multimedia bundle adjustment in metric world space (Maas 1996).

    Refines camera parameters using observation_equation_221 as the residual
    and numerical_jacobian_221 for the camera-parameter Jacobian. Operates
    entirely in metric world / pixel space — no coordinate barrier.

    Parameters
    ----------
    cameras_mm : list of CameraParams
        Initial camera parameters from _build_camera_params() after euclidize().
        Typically biased by the preceding standard pinhole BA.
    Xe : np.ndarray, shape (4, N)
        Homogeneous 3D points in metric world coordinates from euclidize().
    cam : list of dict
        Camera observation structures from GoCal. Must have been populated by
        a preceding reprerror() call (provides recandvis, visandrec, xgt).
    z_glass_top : float
        Z-coordinate of the air/glass interface in mm.
    opt : dict, optional
        'cam_params'  : list[int] — indices into the 16-param vector to
                        optimise. Default [0,1,2,3,4,5] (position + angles).
        'fix_points'  : bool — if False, also refine 3D point positions.
                        Default True (camera-only refinement).
        'verbose'     : bool. Default True.
        'max_nfev'    : int, max function evaluations. Default 2000.
        'ftol'        : float. Default 1e-8.
        'xtol'        : float. Default 1e-8.
        'gtol'        : float. Default 1e-8.

    Returns
    -------
    cameras_mm_new : list of CameraParams
        Refined camera parameters. Same length as cameras_mm.
    Xe_new : np.ndarray, shape (4, N)
        Refined 3D points (identical to Xe when fix_points=True).
    info : dict
        'n_obs'       : int — number of (camera, point) pairs used
        'rms_before'  : float — pixel RMS before optimisation
        'rms_after'   : float — pixel RMS after optimisation
        'n_iter'      : int — number of function evaluations
        'success'     : bool
        'message'     : str — scipy termination reason
        'cam_shifts'  : np.ndarray (K,) — position shift per camera (mm)
        'cam_params'  : list[str] — names of optimised parameters
    """
    if opt is None:
        opt = {}

    cam_mask   = np.asarray(opt.get('cam_params', _DEFAULT_CAM_MASK), dtype=int)
    fix_points = bool(opt.get('fix_points', True))
    verbose    = bool(opt.get('verbose', True))
    max_nfev   = int(opt.get('max_nfev', 2000))
    # residual per axis for a ray the model cannot compute; larger than any
    # residual a valid ray reaches in practice, so failing never pays
    fail_px    = float(opt.get('fail_px', 1000.0))
    _n_failed  = [0]                                   # failed rays in the last residual call
    ftol       = float(opt.get('ftol', 1e-8))
    xtol       = float(opt.get('xtol', 1e-8))
    gtol       = float(opt.get('gtol', 1e-8))

    K         = len(cameras_mm)
    N         = Xe.shape[1]
    n_cam_p   = len(cam_mask)           # free params per camera

    # Dehomogenise Xe -> (3, N) metric world in mm
    Xe3 = Xe[:3, :] / Xe[3:4, :]

    # Pre-flight geometry check — detects scale mismatch and camera geometry errors
    allow_invalid = bool(opt.get('allow_invalid_geometry', False))
    ok, msg = _validate_geometry(cameras_mm, Xe3, z_glass_top, verbose)
    if not ok and not allow_invalid:
        if verbose:
            print(msg)
        info_skip = {
            'n_obs':      0,
            'rms_before': float('nan'),
            'rms_after':  float('nan'),
            'n_iter':     0,
            'success':    False,
            'message':    'Skipped: geometry pre-validation failed',
            'cam_shifts': np.zeros(K),
            'cam_params': [_PARAM_NAMES[i] for i in cam_mask],
        }
        return cameras_mm, Xe, info_skip
    if not ok and allow_invalid and verbose:
        print("  [mmBA] geometry pre-validation FAILED but allow_invalid_geometry=True "
              "— attempting joint BA anyway:")
        print(msg)

    # Build (camera, point, observation) triplets from cam structures
    observations = _build_observations(cam, K)
    n_obs = len(observations)

    if n_obs == 0:
        raise ValueError(
            'multimedia_bundle_adjustment: no observations found in cam structures. '
            'Call reprerror() before calling this function.')

    if verbose:
        opt_names = [_PARAM_NAMES[i] for i in cam_mask]
        print(f'Multimedia BA: {K} cameras · {N} points · {n_obs} observations')
        print(f'  Optimising: {opt_names}  |  fix_points={fix_points}')

    # ── Known inter-camera distances (soft scale/shape constraint) ─────────────
    # opt['cam_dist'] resolves the scale ambiguity that reprojection alone cannot:
    # a wrong overall scale slides all inter-camera distances together, so a
    # penalty on those distances closes the flat "baseline valley". It is SOFT
    # (measured by hand, a few mm off), so it pulls toward the known geometry but
    # lets the images push back. Format: list of (i, j, d_ij_mm). Weight sqrt_w
    # (opt['cam_dist_weight'], px-per-mm) balances it against the pixel residuals;
    # because reprojection is nearly flat along the scale direction, even a modest
    # weight closes the valley without disturbing the image fit.
    cam_dist = opt.get('cam_dist', None)
    cam_dist_w = float(opt.get('cam_dist_weight', 1.0))
    dist_pairs = []
    if cam_dist:
        # only pairs whose position params (X0,Y0,Z0) are actually free can be
        # constrained; if positions are frozen the penalty is constant → skip.
        pos_free = all(p in list(cam_mask) for p in (0, 1, 2))
        if pos_free:
            for (i, j, d) in cam_dist:
                if 0 <= i < K and 0 <= j < K and i != j and d is not None and d > 0:
                    dist_pairs.append((int(i), int(j), float(d)))
    n_pen = len(dist_pairs)
    sqrt_w = np.sqrt(max(cam_dist_w, 0.0))
    if n_pen and verbose:
        print(f'  Inter-camera distance constraint: {n_pen} pairs, weight={cam_dist_w:g} px/mm')

    # ── Known camera-to-glass distances along the line of sight (soft) ─────────
    # opt['cam_los'] = [(i, d_mm)]: hand-measured distance from camera i, along its
    # optical axis, to the glass top surface (plane z = z_glass_top). Model:
    #     d_i = h_i / |cos(omega_i) cos(phi_i)|,  h_i = Z0_i - z_glass_top
    # (cos(omega)cos(phi) = A[2,2] = vertical component of the optical axis).
    # The inter-camera distances fix the rig's shape and scale but leave its height
    # and tilt relative to the glass free; this penalty pins them.
    cam_los = opt.get('cam_los', None)
    cam_los_w = float(opt.get('cam_los_weight', 1.0))
    los_list = []
    if cam_los:
        los_free = all(p in list(cam_mask) for p in (2, 3, 4))   # Z0, omega, phi
        if los_free:
            for (i, d) in cam_los:
                if 0 <= i < K and d is not None and d > 0:
                    los_list.append((int(i), float(d)))
    n_los = len(los_list)
    sqrt_wl = np.sqrt(max(cam_los_w, 0.0))
    if n_los and verbose:
        print(f'  Camera-glass line-of-sight constraint: {n_los} cameras, weight={cam_los_w:g} px/mm')
    # opt['intr_prior'] = [(i, name, target, weight)]: soft prior on an intrinsic
    # parameter ('c', 'xh' or 'yh') of camera i, as a residual row
    # sqrt(weight) * (value - target). Used when the intrinsics are estimated rather
    # than taken from a calibration file (e.g. principal point near the image centre).
    # Applied only to parameters that are being optimised.
    pri_list = []
    for (i, name, target, w) in (opt.get('intr_prior', None) or []):
        if name in ('c', 'xh', 'yh') and _PARAM_NAMES.index(name) in list(cam_mask) and 0 <= int(i) < K and w > 0:
            pri_list.append((int(i), name, float(target), float(np.sqrt(w))))
    n_pri = len(pri_list)
    if n_pri and verbose:
        print(f'  Intrinsic priors: {n_pri} terms')
    n_extra = n_pen + n_los + n_pri

    def _los_model(cam):
        """(predicted line-of-sight distance, h, g) for one camera."""
        om, ph = float(cam.angles[0]), float(cam.angles[1])
        g = np.cos(om) * np.cos(ph)
        h = float(cam.position[2]) - z_glass_top
        return h / max(abs(g), 1e-9), h, g

    # position columns within a camera's parameter block (for the penalty Jacobian)
    _pos_cols = [j for j, idx in enumerate(cam_mask) if _PARAM_NAMES[idx] in ('X0', 'Y0', 'Z0')]
    _pos_axis = {_PARAM_NAMES[cam_mask[j]]: j for j in _pos_cols}  # 'X0'->local col, etc.

    # ── Snapshot of all 16 params per camera (frozen values for non-optimised) ─
    cam_all16 = np.vstack([_pack_cam(c) for c in cameras_mm])   # (K, 16)

    # ── Pack initial parameter vector p0 ──────────────────────────────────────
    # Layout: [cam0_sel, cam1_sel, ..., camK_sel, pt0_xyz, pt1_xyz, ...]
    p0_cam = cam_all16[:, cam_mask].flatten()                     # (K * n_cam_p,)
    if not fix_points:
        p0_pts = Xe3.T.flatten()                                  # (N * 3,)
        p0 = np.concatenate([p0_cam, p0_pts])
    else:
        p0 = p0_cam

    # ── Closure: recover current cameras and points from p ────────────────────
    def _unpack_p(p):
        cams_cur = []
        for k in range(K):
            all16 = cam_all16[k].copy()
            all16[cam_mask] = p[k * n_cam_p:(k + 1) * n_cam_p]
            cams_cur.append(_unpack_cam(cameras_mm[k], all16))
        if fix_points:
            pts_cur = Xe3                                         # (3, N) unchanged
        else:
            pts_cur = p[K * n_cam_p:].reshape(N, 3).T            # (3, N)
        return cams_cur, pts_cur

    # ── Residual function ──────────────────────────────────────────────────────
    def residual_fn(p):
        cams_cur, pts_cur = _unpack_p(p)
        res = np.zeros(2 * n_obs + n_extra)
        _n_failed[0] = 0
        for l, (k, xe_col, x_obs, y_obs) in enumerate(observations):
            particle = pts_cur[:, xe_col]
            try:
                vx, vy, _ = observation_equation_221(
                    particle, cams_cur[k], x_obs, y_obs)
            except Exception:
                # A ray the model cannot compute (camera or particle on the wrong
                # side of the glass, total internal reflection) must COST, not score
                # perfectly: with 0 here the optimiser could lower its cost by
                # pushing particles out of the water. Fixed penalty per axis.
                vx, vy = fail_px, fail_px
                _n_failed[0] += 1
            res[2 * l]     = vx
            res[2 * l + 1] = vy
        # inter-camera distance penalty rows: sqrt_w * (‖Ci-Cj‖ - d_ij)
        for m, (i, j, d) in enumerate(dist_pairs):
            dvec = cams_cur[i].position - cams_cur[j].position
            res[2 * n_obs + m] = sqrt_w * (float(np.linalg.norm(dvec)) - d)
        # camera-glass line-of-sight rows: sqrt_wl * (h/|g| - d_i)
        for m, (i, d) in enumerate(los_list):
            res[2 * n_obs + n_pen + m] = sqrt_wl * (_los_model(cams_cur[i])[0] - d)
        # intrinsic prior rows: sqrt(w) * (value - target)
        for m, (i, name, target, sw) in enumerate(pri_list):
            res[2 * n_obs + n_pen + n_los + m] = sw * (float(getattr(cams_cur[i], name)) - target)
        return res

    # ── Jacobian function ──────────────────────────────────────────────────────
    def jacobian_fn(p):
        cams_cur, pts_cur = _unpack_p(p)
        n_params = len(p)
        J = np.zeros((2 * n_obs + n_extra, n_params))

        for l, (k, xe_col, x_obs, y_obs) in enumerate(observations):
            particle = pts_cur[:, xe_col]
            row = 2 * l

            # Camera Jacobian: only the optimised columns (cam_mask) are computed —
            # bit-identical to selecting them from the full (2, 16) Jacobian. When no
            # camera parameter is free (cloud pass), nothing is computed at all.
            if n_cam_p:
                try:
                    J[row:row+2, k*n_cam_p:(k+1)*n_cam_p] = numerical_jacobian_221(
                        particle, cams_cur[k], x_obs, y_obs, mask=cam_mask)  # (2, n_cam_p)
                except Exception:
                    pass   # leave zeros for degenerate ray

            # Point Jacobian: central finite differences through obs_eq_221
            if not fix_points:
                try:
                    J_pt = _point_jacobian(
                        cams_cur[k], particle, x_obs, y_obs)   # (2, 3)
                    col_pt = K * n_cam_p + xe_col * 3
                    J[row:row+2, col_pt:col_pt+3] = J_pt
                except Exception:
                    pass

        # penalty rows: d‖Ci-Cj‖/dCi = (Ci-Cj)/‖Ci-Cj‖ ; opposite sign for Cj
        axes = ('X0', 'Y0', 'Z0')
        for m, (i, j, d) in enumerate(dist_pairs):
            prow = 2 * n_obs + m
            dvec = cams_cur[i].position - cams_cur[j].position
            nrm = float(np.linalg.norm(dvec))
            if nrm < 1e-9:
                continue
            u = dvec / nrm                                     # unit vector
            for a, axis in enumerate(axes):
                loc = _pos_axis.get(axis)
                if loc is None:
                    continue
                J[prow, i * n_cam_p + loc] = sqrt_w * u[a]
                J[prow, j * n_cam_p + loc] = -sqrt_w * u[a]

        # line-of-sight rows: d = h/|g|, g = cos(om)cos(ph), sg = sign(g)
        #   dd/dZ0 = 1/|g|;  dd/dom = h*sg*sin(om)cos(ph)/g^2;  dd/dph = h*sg*cos(om)sin(ph)/g^2
        _loc = {_PARAM_NAMES[cam_mask[jj]]: jj for jj in range(n_cam_p)}
        for m, (i, d) in enumerate(los_list):
            prow = 2 * n_obs + n_pen + m
            _, h, g = _los_model(cams_cur[i])
            if abs(g) < 1e-9:
                continue
            sg = 1.0 if g > 0 else -1.0
            om, ph = float(cams_cur[i].angles[0]), float(cams_cur[i].angles[1])
            J[prow, i * n_cam_p + _loc['Z0']]    = sqrt_wl / abs(g)
            J[prow, i * n_cam_p + _loc['omega']] = sqrt_wl * h * sg * np.sin(om) * np.cos(ph) / g**2
            J[prow, i * n_cam_p + _loc['phi']]   = sqrt_wl * h * sg * np.cos(om) * np.sin(ph) / g**2

        # intrinsic prior rows: d/d(param) = sqrt(w)
        for m, (i, name, target, sw) in enumerate(pri_list):
            J[2 * n_obs + n_pen + n_los + m, i * n_cam_p + _loc[name]] = sw

        return J

    # ── Initial RMS (pixel part only; the penalty rows are excluded so the
    #    reported reprojection RMS stays meaningful) ─────────────────────────────
    def _pixel_rms(res):
        px = res[:2 * n_obs]
        return float(np.sqrt(np.mean(px ** 2)))
    def _pen_rms(res):
        if not n_pen:
            return 0.0
        return float(np.sqrt(np.mean((res[2 * n_obs:2 * n_obs + n_pen] / max(sqrt_w, 1e-12)) ** 2)))
    def _los_rms(res):
        if not n_los:
            return 0.0
        return float(np.sqrt(np.mean((res[2 * n_obs + n_pen:2 * n_obs + n_pen + n_los] / max(sqrt_wl, 1e-12)) ** 2)))
    res0 = residual_fn(p0)
    rms_before = _pixel_rms(res0)
    if verbose:
        print(f'  RMS before: {rms_before:.4f} px'
              + (f'  |  cam-dist misfit: {_pen_rms(res0):.2f} mm' if n_pen else '')
              + (f'  |  glass-LOS misfit: {_los_rms(res0):.2f} mm' if n_los else ''))

    # ── Run the optimiser ──────────────────────────────────────────────────────
    # If a position trust radius is given, bound the camera-centre parameters
    # (X0,Y0,Z0) to +/- max_pos_step_mm around the seed and use the bounded
    # trust-region solver, so the BA descends from the clean seed instead of
    # making thousand-millimetre excursions. Angles stay free.
    max_pos_step = opt.get('max_pos_step_mm', None)
    if max_pos_step is not None and max_pos_step > 0:
        lo = np.full_like(p0, -np.inf)
        hi = np.full_like(p0,  np.inf)
        # which entries of the per-camera parameter block are X0,Y0,Z0?
        pos_local = [j for j, idx in enumerate(cam_mask) if _PARAM_NAMES[idx] in ('X0', 'Y0', 'Z0')]
        for k in range(K):
            for j in pos_local:
                col = k * n_cam_p + j
                lo[col] = p0[col] - max_pos_step
                hi[col] = p0[col] + max_pos_step
        # When points are free, bound each point coordinate too, so the joint
        # optimisation can move the cloud (e.g. back through the interface) but
        # cannot let points diverge. Radius scaled larger than the camera step
        # because the cloud may legitimately need to move further than a camera.
        if not fix_points:
            pt_step = float(opt.get('max_pt_step_mm', 5.0 * max_pos_step))
            base = K * n_cam_p
            lo[base:] = p0[base:] - pt_step
            hi[base:] = p0[base:] + pt_step
        result = least_squares(
            residual_fn, p0, jac=jacobian_fn,
            method='trf', bounds=(lo, hi),
            max_nfev=max_nfev, ftol=ftol, xtol=xtol, gtol=gtol, verbose=0,
        )
    else:
        result = least_squares(
            residual_fn, p0, jac=jacobian_fn,
            method='lm',        # unbounded Levenberg-Marquardt (original behaviour)
            max_nfev=max_nfev, ftol=ftol, xtol=xtol, gtol=gtol, verbose=0,
        )

    # ── Unpack result ──────────────────────────────────────────────────────────
    cams_new, pts_new = _unpack_p(result.x)

    residual_fn(result.x)                  # sets _n_failed for the final solution
    rms_after = _pixel_rms(result.fun)
    pen_after = _pen_rms(result.fun)
    los_after = _los_rms(result.fun)
    if verbose:
        print(f'  RMS after:  {rms_after:.4f} px'
              + (f'  |  cam-dist misfit: {pen_after:.2f} mm' if n_pen else '')
              + (f'  |  glass-LOS misfit: {los_after:.2f} mm' if n_los else '')
              + (f'  |  FAILED rays: {_n_failed[0]}/{n_obs}' if _n_failed[0] else '')
              + f'  |  {result.nfev} evals  |  {result.message}')

    # Build Xe_new — update 3D positions if points were refined
    Xe_new = Xe.copy()
    if not fix_points:
        Xe_new[:3, :] = pts_new
        Xe_new[3, :]  = 1.0

    # Per-camera position shift (mm) — diagnostic quality indicator
    cam_shifts = np.array([
        np.linalg.norm(cams_new[k].position - cameras_mm[k].position)
        for k in range(K)
    ])
    if verbose:
        for k in range(K):
            print(f'  Camera {cameras_mm[k].cam_id}: '
                  f'position shift = {cam_shifts[k]:.4f} mm')

    info = {
        'n_obs':       n_obs,
        'rms_before':  rms_before,
        'rms_after':   rms_after,
        'n_iter':      result.nfev,
        'success':     bool(result.success),
        'message':     result.message,
        'cam_shifts':  cam_shifts,
        'cam_dist_pairs':  n_pen,
        'cam_dist_misfit': pen_after,   # RMS |recovered - known| over pairs (mm)
        'n_failed_rays':   _n_failed[0],  # rays the model could not compute (penalised)
        'cam_los_n':       n_los,
        'cam_los_misfit':  los_after,   # RMS |recovered - measured| line-of-sight (mm)
        'cam_params':  [_PARAM_NAMES[i] for i in cam_mask],
    }

    return cams_new, Xe_new, info


# ── Multimedia reprojection error ─────────────────────────────────────────────

def multimedia_reprerror(cam, Pe, Xe, FRAMES, inliers, cameras_mm, lut):
    """
    Compute multimedia-corrected reprojection error for each camera.

    This is the multimedia counterpart of reprerror() in
    EstimateReprojectionError.py, which is kept as a pure pinhole function.
    Call this AFTER the inner outlier loop has converged and cameras_mm/lut
    are available — not inside the loop.

    For each visible, reconstructed point the predicted image position is
    computed via:
        1. dehomogenise Xe[:, n]      → particle (3,) in world mm
        2. lut[cam_id].shift_point()  → P_bar (Maas eq. 8)
        3. collinearity_condition()   → (x_pred, y_pred) in pixel space

    Sets on each cam[i]:
        recandvis  — column indices into Xe (points reconstructed AND visible)
        visandrec  — column indices into xgt (matching observed positions)
        xe_mm      — (2, n) multimedia-predicted image positions
        err2d      — per-point Euclidean error (pixels)
        mean2Derr  — mean error (pixels)
        std2Derr   — std deviation of error (pixels)

    cam[i]['xe'] (standard pinhole projection) is NOT touched — it belongs
    to the pinhole framework and is managed by reprerror().

    Parameters
    ----------
    cam : list of dict
        Camera structures with 'idlin' and 'xgt' already populated.
    Pe : np.ndarray (3*CAMS x 4)
        Euclidean projection matrices from euclidize().
    Xe : np.ndarray (4 x N)
        Euclidean 3D points from euclidize() — metric world coordinates.
    FRAMES : int
        Total number of frames (columns in the original measurement matrix).
    inliers : dict
        Must contain 'idx' — array of inlier frame indices.
    cameras_mm : list of CameraParams
        One per camera, built by build_camera_params() after euclidize().
    lut : MultiCameraLUT
        Per-camera radial-shift lookup table built by MultiCameraLUT.build_all().

    Returns
    -------
    cam : list of dict
        Updated with multimedia error fields on each camera.
    """
    from Multimedia.Iterative_radial_shift import collinearity_condition

    CAMS = Pe.shape[0] // 3

    for i in range(CAMS):
        # ── Index bookkeeping (identical to reprerror) ────────────────────────
        mask_rec = np.zeros(FRAMES, dtype=int)
        mask_vis = np.zeros(FRAMES, dtype=int)

        mask_rec[inliers['idx']] = 1
        mask_vis[cam[i]['idlin']] = 1
        mask_both = mask_vis & mask_rec

        unmask_rec = np.cumsum(mask_rec)
        unmask_vis = np.cumsum(mask_vis)

        selection = np.logical_and(
            ~np.logical_xor(mask_rec, mask_both), mask_rec
        ).astype(bool)

        cam[i]['recandvis'] = (unmask_rec[selection] - 1).astype(int)
        cam[i]['visandrec'] = (unmask_vis[selection] - 1).astype(int)

        # ── Multimedia prediction per point ───────────────────────────────────
        n_pts   = len(cam[i]['recandvis'])
        xe_mm   = np.zeros((2, n_pts))
        cam_obj = cameras_mm[i]
        lut_i   = lut[cam_obj.cam_id]
        xgt_pts = cam[i]['xgt'][0:2, cam[i]['visandrec']]

        for k, rec_idx in enumerate(cam[i]['recandvis']):
            X_h      = Xe[:, rec_idx]
            particle = X_h[:3] / X_h[3]              # dehomogenise to world mm
            try:
                P_bar = lut_i.shift_point(particle, cam_obj)   # Maas eq. 8
                x_pred, y_pred = collinearity_condition(P_bar, cam_obj)  # eq. 17
            except Exception:
                # Degenerate ray (total internal reflection) — fall back to
                # standard pinhole prediction for this point only
                xe_pin = Pe[3*i:3*i+3, :] @ X_h
                x_pred = float(xe_pin[0] / xe_pin[2])
                y_pred = float(xe_pin[1] / xe_pin[2])
            xe_mm[0, k] = x_pred
            xe_mm[1, k] = y_pred

        cam[i]['xe_mm']      = xe_mm
        cam[i]['err2d']      = np.sqrt(np.sum((xe_mm - xgt_pts) ** 2, axis=0))
        cam[i]['mean2Derr']  = float(np.mean(cam[i]['err2d']))
        cam[i]['std2Derr']   = float(np.std(cam[i]['err2d']))

    return cam


# ── Reprojection error with the BA's OWN model (single authority) ────────────

def multimedia_point_residuals(cam, cameras_mm, Xe):
    """
    Per-point reprojection error computed with EXACTLY the model and the
    observation set the multimedia bundle adjustment minimises:
    _build_observations() + observation_equation_221 (exact Maas model, no LUT).

    Use this for every reported reprojection number in the multimedia pipeline,
    so the final statistics and graphs agree with the BA's 'reproj' value:
        sqrt(mean(err2d**2) / 2)  ==  BA rms_after   (per-axis RMS)
    (exact, except for degenerate rays: the BA scores them as 0, here they are
    NaN and are excluded and counted by evalreprerror).

    Requires cam[k]['recandvis'], ['visandrec'], ['xgt'] (set by reprerror()).

    Sets on each cam[k]:
        err2d     — per-point Euclidean error (px), NaN for degenerate rays
        mean2Derr — mean of finite err2d (px)
        std2Derr  — std of finite err2d (px)
    """
    K = len(cameras_mm)
    pts = Xe[:3, :] / Xe[3:4, :]
    per_cam = [[] for _ in range(K)]
    for (k, xe_col, x_obs, y_obs) in _build_observations(cam, K):
        try:
            vx, vy, _ = observation_equation_221(pts[:, xe_col], cameras_mm[k],
                                                 x_obs, y_obs)
            per_cam[k].append(float(np.hypot(vx, vy)))
        except Exception:
            per_cam[k].append(float('nan'))          # degenerate ray (TIR)
    for k in range(K):
        e = np.asarray(per_cam[k], float)
        cam[k]['err2d'] = e
        ef = e[np.isfinite(e)]
        cam[k]['mean2Derr'] = float(np.mean(ef)) if ef.size else float('nan')
        cam[k]['std2Derr'] = float(np.std(ef)) if ef.size else float('nan')
    return cam
