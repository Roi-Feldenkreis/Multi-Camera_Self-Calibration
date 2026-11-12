"""
Iterative_radial_shift .py
=================
Section 2.2.1 — Multimedia module for the collinearity condition.

Maas (1996), Section 2.2.1, equations (2)–(8) + eq. (17).

PURPOSE
-------
This is the CALIBRATION functional model.

For each object point P(X,Y,Z), it computes a radially shifted point
P̄(X̄, Ȳ, Z) that, when substituted into the standard collinearity
condition, correctly predicts the image coordinates despite refraction.

This is the approach to use for:
    ✓  System calibration (spatial resection, bundle adjustment)
    ✓  Least-squares parameter estimation (all 16 camera parameters)
    ✓  Self-calibration with multiple cameras
    ✗  Particle tracking — use Forward_ray_tracing.py instead (faster)

The full collinearity condition (Maas eq. 17) is also provided here,
ready to be used as observation equations in a least-squares system.

KEY EQUATIONS FROM MAAS (1996)
-------------------------------
Geometry constraint (eq. 2):
    Z₀·tan(β₁) + t·tan(β₂) + Zₚ·tan(β₃) = R
    R̄ = (Z₀ + t + Zₚ)·tan(β₁)

Snell's Law (eq. 3):
    n₁·sin(β₁) = n₂·sin(β₂) = n₃·sin(β₃)

Iterative scheme (eqs. 4–7):
    R̄₍₀₎ = √((Xₚ-X₀)² + (Yₚ-Y₀)²)          initial estimate
    β₁    = atan(R̄ / (Z₀ + t + Zₚ))            eq. (5)
    β₂, β₃ via Snell                             eq. (6)
    dR    = R̄ - (Z₀·tanβ₁ + t·tanβ₂ + Zₚ·tanβ₃)  eq. (7)
    R̄₍₁₎ = R̄₍₀₎ + dR    →  repeat until |dR| < ε

Cartesian shift (eq. 8):
    X̄ₚ = X₀ + (Xₚ-X₀)·(R̄/R)
    Ȳₚ = Y₀ + (Yₚ-Y₀)·(R̄/R)
    Z̄ₚ = Zₚ

Collinearity condition (eq. 17):
    (x'ᵢ, y'ᵢ) = f(X₀,Y₀,Z₀, ω,φ,κ, c,xₕ,yₕ, k₁,k₂,k₃, p₁,p₂, a₁,a₂, Xᵢ,Yᵢ,Zᵢ)

Dependencies
------------
    snell_core.py
    multimedia_params.py
"""

from __future__ import annotations
import numpy as np

from Multimedia.Snell_core import snell_all_angles, radial_distance_from_nadir
from Multimedia.Multimedia_params import CameraParams


# ---------------------------------------------------------------------------
# Core: Maas iterative radial-shift algorithm (eqs. 4–8)
# ---------------------------------------------------------------------------

def multimedia_radial_shift(
    particle: np.ndarray,
    cam: CameraParams,
    eps: float = 1e-6,
    max_iter: int = 100,
    ocf: float = 1.0,) -> tuple[np.ndarray, int, float]:
    """
    Compute the radially shifted point P̄ for one camera–particle pair.

    This is the heart of Section 2.2.1. It iteratively solves for the
    R̄ value such that the refracted ray from the camera through the
    glass wall lands exactly on the particle P.

    Maas (1996) equations (4)–(8).

    Coordinate system assumption
    ----------------------------
    The X-Y plane is parallel to the glass/water interface.
    Z₀ = distance from camera to glass top surface (positive, upward).
    Zₚ = distance from glass bottom surface to particle (positive, downward).
    Both are derived internally from cam.position and particle.

    Parameters
    ----------
    particle : np.ndarray, shape (3,)
        True 3-D particle position [X, Y, Z] in mm.
    cam : CameraParams
        Camera with position, multimedia params (n1, n2, d, n3).
    eps : float
        Convergence criterion |dR| < eps in mm (Maas uses ε = 0.001 mm).
    max_iter : int
        Maximum iterations before raising a warning.
    ocf : float
        Over-compensation factor for convergence acceleration.
        Maas suggests ocf ≈ 1.1–1.8 depending on geometry.
        Default 1.0 = no over-compensation (safest).

    Returns
    -------
    P_bar : np.ndarray, shape (3,)
        The radially shifted point [X̄, Ȳ, Z] to use in the
        collinearity condition instead of P.
    n_iter : int
        Number of iterations taken.
    final_dR : float
        Residual |dR| at convergence (mm).

    Notes
    -----
    The nadir point N is at [X₀, Y₀, 0.0] in the coordinate system
    where the glass/water interface is at z = 0.
    The sign convention for Z₀ and Zₚ follows Maas Figure 3:
        - Z₀ > 0: camera is above the glass top surface
        - Zₚ > 0: particle is below the glass bottom surface
    """
    mm = cam.mm
    X0, Y0, Z0_cam = cam.position
    Xp, Yp, Zp_world = particle

    # --- Derive Z₀ and Zₚ from world coordinates ---
    # We need to know where the glass surfaces are.
    # Convention: glass top is at z = 0, glass bottom at z = -t_total.
    # Z₀ = height of camera above glass top = Z0_cam - 0 = Z0_cam
    # Zₚ = depth of particle below glass bottom = -(Zp_world + t_total)
    # (particle below glass → Zp_world < -t_total → Zₚ > 0)
    t = mm.t_total
    Z0 = Z0_cam                     # camera height above glass top
    Zp = -(Zp_world + t)            # particle depth below glass bottom

    if Z0 <= 0:
        raise ValueError(
            f"Camera must be above the glass (Z0_cam > 0). Got Z0_cam={Z0_cam}"
        )
    if Zp < 0:
        raise ValueError(
            f"Particle must be below the glass bottom. "
            f"Got particle Z={Zp_world}, glass bottom at Z={-t}."
        )

    # --- Nadir point N = [X₀, Y₀, 0] ---
    nadir = np.array([X0, Y0, 0.0])

    # --- Initial estimate of R̄ (eq. 4) ---
    R = radial_distance_from_nadir(particle, nadir)   # true horizontal distance
    R_bar = R                                          # initial guess = true R

    if R_bar < 1e-9:
        # Particle is directly below the nadir — no radial shift
        return particle.copy(), 0, 0.0

    # --- Iterative refinement (eqs. 5–7) ---
    # Goal: find R̄ such that the refracted ray at angle β₁ = atan(R̄/(Z₀+t+Zₚ))
    # actually reaches the TRUE particle at radius R.
    # Convergence condition (Maas eq. 7): |R - R_ray| < ε
    n_iter = 0
    final_dR = np.inf

    for n_iter in range(1, max_iter + 1):
        # eq. (5): angle in air based on current virtual radius R̄
        beta1 = np.arctan(R_bar / (Z0 + t + Zp))

        # eq. (6): angles in glass layers and water via Snell's Law
        beta2_list, beta3 = snell_all_angles(beta1, mm.n1, mm.n2, mm.n3)

        # Radius the refracted ray ACTUALLY reaches at depth Zp
        R_ray = (Z0 * np.tan(beta1)
                 + sum(d_i * np.tan(b2)
                       for d_i, b2 in zip(mm.d, beta2_list))
                 + Zp * np.tan(beta3))

        # Residual (eq. 7): how far the refracted ray falls short of the
        # TRUE particle radius R (fixed). dR > 0 when ray bends inward.
        dR = R - R_ray          # ← compare to fixed R, not varying R̄
        final_dR = abs(dR)

        if final_dR < eps:
            break

        # Update R̄ with optional over-compensation (Maas: ocf ≈ 1.1–1.8)
        R_bar = R_bar + ocf * dR
    else:
        import warnings
        warnings.warn(
            f"multimedia_radial_shift did not converge in {max_iter} "
            f"iterations (final |dR|={final_dR:.6f} mm). "
            f"Check geometry or increase max_iter.",
            RuntimeWarning,
        )

    # --- Convert R̄ back to Cartesian (eq. 8) ---
    # Scale the horizontal offset from nadir by the ratio R̄/R
    if R < 1e-9:
        X_bar, Y_bar = Xp, Yp
    else:
        scale = R_bar / R
        X_bar = X0 + (Xp - X0) * scale
        Y_bar = Y0 + (Yp - Y0) * scale

    P_bar = np.array([X_bar, Y_bar, Zp_world])   # Z is unchanged (eq. 8)

    return P_bar, n_iter, final_dR


# ---------------------------------------------------------------------------
# Collinearity condition (standard, on shifted point)
# ---------------------------------------------------------------------------

def collinearity_condition(
    P_bar: np.ndarray,
    cam: CameraParams,
) -> tuple[float, float]:
    """
    Standard photogrammetric collinearity condition.

    Applied to the SHIFTED point P̄ (output of multimedia_radial_shift),
    this gives the predicted image coordinates incorporating the
    multimedia geometry correction.

    Maas (1996) eq. (17):
        (x', y') = f(X₀,Y₀,Z₀, ω,φ,κ, c, xₕ,yₕ, k₁..k₃, p₁,p₂, a₁,a₂, X̄,Ȳ,Z)

    Standard form (no distortion):
        x' = -c · (r₁₁(X̄-X₀) + r₁₂(Ȳ-Y₀) + r₁₃(Z̄-Z₀)) / denominator + xₕ
        y' = -c · (r₂₁(X̄-X₀) + r₂₂(Ȳ-Y₀) + r₂₃(Z̄-Z₀)) / denominator + yₕ
    where denominator = r₃₁(X̄-X₀) + r₃₂(Ȳ-Y₀) + r₃₃(Z̄-Z₀)

    Parameters
    ----------
    P_bar : np.ndarray, shape (3,)
        Radially shifted object point [X̄, Ȳ, Z̄].
    cam : CameraParams
        Camera parameters.

    Returns
    -------
    x_pred, y_pred : float
        Predicted image coordinates in mm (before distortion correction).
    """
    A = cam.rotation_matrix
    dX = P_bar[0] - cam.position[0]
    dY = P_bar[1] - cam.position[1]
    dZ = P_bar[2] - cam.position[2]

    # Numerators and denominator of the collinearity condition
    num_x = A[0, 0]*dX + A[0, 1]*dY + A[0, 2]*dZ
    num_y = A[1, 0]*dX + A[1, 1]*dY + A[1, 2]*dZ
    denom = A[2, 0]*dX + A[2, 1]*dY + A[2, 2]*dZ

    if abs(denom) < 1e-12:
        raise ValueError(
            "Collinearity denominator ≈ 0: point is at or behind the "
            "camera's principal plane. Check geometry."
        )

    x_pred = -cam.c * (num_x / denom) + cam.xh
    y_pred = -cam.c * (num_y / denom) + cam.yh

    return float(x_pred), float(y_pred)


def apply_lens_distortion(
    x: float,
    y: float,
    cam: CameraParams) -> tuple[float, float]:
    """
    Apply radial and decentering distortion correction.

    Maas (1996) Section 2.1: 5 distortion parameters (k₁,k₂,k₃, p₁,p₂).

    The distorted image coordinate:
        r² = x² + y²
        Δx_rad = x(k₁r² + k₂r⁴ + k₃r⁶)
        Δy_rad = y(k₁r² + k₂r⁴ + k₃r⁶)
        Δx_dec = p₁(r² + 2x²) + 2p₂xy
        Δy_dec = p₂(r² + 2y²) + 2p₁xy
        x_dist = x + Δx_rad + Δx_dec
        y_dist = y + Δy_rad + Δy_dec

    Parameters
    ----------
    x, y : float
        Ideal (undistorted) image coordinates in mm.
    cam : CameraParams
        Camera with distortion parameters k1,k2,k3,p1,p2.

    Returns
    -------
    x_dist, y_dist : float
        Distorted image coordinates in mm.
    """
    r2 = x**2 + y**2
    radial = cam.k1 * r2 + cam.k2 * r2**2 + cam.k3 * r2**3
    dx_rad = x * radial
    dy_rad = y * radial
    dx_dec = cam.p1 * (r2 + 2*x**2) + 2*cam.p2 * x * y
    dy_dec = cam.p2 * (r2 + 2*y**2) + 2*cam.p1 * x * y
    return x + dx_rad + dx_dec, y + dy_rad + dy_dec


def apply_ad_correction(
    x: float,
    y: float,
    cam: CameraParams) -> tuple[float, float]:
    """
    Apply the A/D affine transformation (Maas eq. 1).

        x̃ = a₀ + a₁·x' + a₂·y'
        ỹ = b₀ + b₁·x' + b₂·y'

    Only the two free parameters (a₁, a₂) are implemented here;
    (a₀, b₀, b₁, b₂) are absorbed into the camera orientation.

    Parameters
    ----------
    x, y : float
        Image coordinates before A/D correction.
    cam : CameraParams
        Camera with a1 (scale), a2 (shear).

    Returns
    -------
    x_corr, y_corr : float
    """
    x_corr = (1.0 + cam.a1) * x + cam.a2 * y
    y_corr = y                                   # y not affected by a1,a2
    return x_corr, y_corr


# ---------------------------------------------------------------------------
# Full observation equation (eq. 17 + multimedia)
# ---------------------------------------------------------------------------

def observation_equation_221(
    particle: np.ndarray,
    cam: CameraParams,
    x_observed: float,
    y_observed: float,
    eps: float = 1e-6,
    ocf: float = 1.0) -> tuple[float, float, np.ndarray]:
    """
    Full Section 2.2.1 observation equation for one camera–point pair.

    Combines:
        1. Multimedia radial shift  (eqs. 4–8)       → P̄
        2. Collinearity condition   (eq. 17)           → (x_pred, y_pred)
        3. Residuals                                   → (v_x, v_y)

    This is the building block for the least-squares calibration system.
    The Jacobian (partial derivatives w.r.t. all unknowns) can be computed
    by numerical differentiation around this function.

    Parameters
    ----------
    particle : np.ndarray, shape (3,)
        3-D object point (either a calibration target or particle position).
    cam : CameraParams
        Current camera parameter estimates.
    x_observed, y_observed : float
        Measured image coordinates (corrected for distortion + A/D) in mm.
    eps : float
        Convergence tolerance for the radial-shift iteration.
    ocf : float
        Over-compensation factor for the iteration.

    Returns
    -------
    v_x, v_y : float
        Image residuals: observed − predicted (mm).
        In the least-squares sense, these are the observation corrections.
    P_bar : np.ndarray, shape (3,)
        The shifted object point used (for diagnostics).
    """
    # Step 1: multimedia radial shift
    P_bar, _, _ = multimedia_radial_shift(particle, cam, eps=eps, ocf=ocf)

    # Step 2: collinearity on shifted point
    x_pred, y_pred = collinearity_condition(P_bar, cam)

    # Step 3: residuals
    v_x = x_observed - x_pred
    v_y = y_observed - y_pred

    return v_x, v_y, P_bar


# ---------------------------------------------------------------------------
# Numerical Jacobian (for least-squares calibration)
# ---------------------------------------------------------------------------

def numerical_jacobian_221(
    particle: np.ndarray,
    cam: CameraParams,
    x_observed: float,
    y_observed: float,
    delta: float = 1e-5,
    mask=None) -> np.ndarray:
    """
    Compute the numerical Jacobian of the observation equation
    with respect to the 16 camera parameters.

    Used in the linearized least-squares calibration:
        J · Δp = v
    where Δp is the correction to camera parameters and v is the
    vector of residuals.

    The 16 parameters (Maas eq. 17) in order:
        [X₀, Y₀, Z₀, ω, φ, κ, c, xₕ, yₕ, k₁, k₂, k₃, p₁, p₂, a₁, a₂]

    Parameters
    ----------
    particle : np.ndarray, shape (3,)
        3-D object point.
    cam : CameraParams
        Current camera parameter estimates.
    x_observed, y_observed : float
        Observed image coordinates in mm.
    delta : float
        Step size for finite difference.

    Returns
    -------
    mask : sequence of int, optional
        Indices (into the 16 parameters above) of the columns to compute.
        None = all 16 (original behaviour). Each column is an independent
        forward difference, so the requested columns are bit-identical to
        the same columns of the full Jacobian; the rest are simply not
        computed (the BA optimises only the masked parameters).

    J : np.ndarray, shape (2, 16) — or (2, len(mask)) when mask is given
        Jacobian matrix: rows = [x, y], columns = camera params (in mask order).
    """
    import copy

    param_names = [
        'position_0', 'position_1', 'position_2',   # X₀, Y₀, Z₀
        'angles_0',   'angles_1',   'angles_2',      # ω, φ, κ
        'c', 'xh', 'yh',                             # interior orientation
        'k1', 'k2', 'k3',                            # radial distortion
        'p1', 'p2',                                  # decentering distortion
        'a1', 'a2',                                  # A/D correction
    ]

    # Reference residuals
    vx0, vy0, _ = observation_equation_221(particle, cam, x_observed, y_observed)

    cols = list(range(16)) if mask is None else [int(m) for m in mask]
    J = np.zeros((2, len(cols)))

    for out_col, col in enumerate(cols):
        name = param_names[col]
        # shallow copy: the perturbed attribute is replaced on the copy below
        # (position/angles get fresh arrays), so the original is never touched;
        # the shared MultimediaParams is read-only here.
        cam_plus = copy.copy(cam)

        # Perturb the appropriate parameter
        if name.startswith('position_'):
            idx = int(name[-1])
            cam_plus.position = cam.position.copy()
            cam_plus.position[idx] += delta
        elif name.startswith('angles_'):
            idx = int(name[-1])
            cam_plus.angles = cam.angles.copy()
            cam_plus.angles[idx] += delta
        else:
            setattr(cam_plus, name, getattr(cam, name) + delta)

        vx1, vy1, _ = observation_equation_221(
            particle, cam_plus, x_observed, y_observed
        )

        J[0, out_col] = (vx1 - vx0) / delta
        J[1, out_col] = (vy1 - vy0) / delta

    return J


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from Multimedia.Multimedia_params import MultimediaParams, CameraParams, N_AIR, N_GLASS, N_WATER

    print("=== Iterative_radial_shift .py self-test ===\n")

    mm = MultimediaParams(n1=N_AIR, n2=[N_GLASS], d=[5.0], n3=N_WATER)

    cam = CameraParams(
        cam_id=0,
        position=np.array([0.0, 0.0, 300.0]),  # 300 mm above glass top
        angles=np.zeros(3),
        c=10.0,
        mm=mm,
    )

    # A particle 50 mm into water (glass bottom is at z = -5.0)
    particle = np.array([30.0, 15.0, -55.0])

    print(f"Camera position: {cam.position}")
    print(f"Particle position: {particle}")
    print(f"Glass: n2={mm.n2}, thickness={mm.d} mm")
    print(f"Media: n1={mm.n1} (air), n3={mm.n3} (water)\n")

    # --- Test 1: radial shift ---
    P_bar, n_iter, dR = multimedia_radial_shift(particle, cam, eps=1e-6, ocf=1.3)
    print(f"Radial shift result:")
    print(f"  P    = {particle}")
    print(f"  P̄    = {P_bar}")
    print(f"  ΔR   = {np.linalg.norm(P_bar[:2] - particle[:2]):.4f} mm")
    print(f"  Converged in {n_iter} iterations, |dR|={dR:.2e} mm\n")

    # --- Test 2: collinearity condition ---
    x_pred, y_pred = collinearity_condition(P_bar, cam)
    print(f"Predicted image coordinates: x'={x_pred:.4f} mm, y'={y_pred:.4f} mm\n")

    # --- Test 3: observation equation ---
    x_obs, y_obs = x_pred + 0.01, y_pred - 0.005   # simulate small measurement noise
    vx, vy, _ = observation_equation_221(particle, cam, x_obs, y_obs)
    print(f"Observation residuals: vx={vx:.4f} mm, vy={vy:.4f} mm")
    print(f"(Expected ~0.01 and ~-0.005 since we added known noise)\n")

    # --- Test 4: numerical Jacobian ---
    J = numerical_jacobian_221(particle, cam, x_obs, y_obs)
    print(f"Jacobian shape: {J.shape}  (2 obs × 16 params)")
    print(f"Jacobian row 0 (∂x/∂params): {np.round(J[0], 5)}")
    print(f"Jacobian row 1 (∂y/∂params): {np.round(J[1], 5)}")

    print("\nAll tests passed.")
