"""
snell_core.py
=============
Shared physics core: Snell's Law and geometry primitives.

This module is the foundation shared by BOTH solution paths:
  - Section 2.2.2  (forward ray tracing  → Forward_ray_tracing.py)
  - Section 2.2.1  (inverse radial shift  → Iterative_radial_shift .py)

The central relationship is Maas (1996) equation (3):

    n₁ · sin(β₁) = n₂ · sin(β₂) = n₃ · sin(β₃)

All functions operate on plain floats or NumPy arrays.
No dependency on multimedia_params.py so this module can be
imported in isolation for testing.

References
----------
Maas, H.-G. (1996), eqs. (3), (6), (11), (13).
"""

from __future__ import annotations
import numpy as np


# ---------------------------------------------------------------------------
# Snell's Law: angle form (used by 2.2.1)
# ---------------------------------------------------------------------------

def snell_angle(beta_in: float, n_in: float, n_out: float) -> float:
    """
    Apply Snell's Law in angle form.

        n_in · sin(β_in) = n_out · sin(β_out)

    Maas (1996) eq. (6):
        β₂ = asin( (n₁/n₂) · sin(β₁) )
        β₃ = asin( (n₁/n₃) · sin(β₁) )

    Parameters
    ----------
    beta_in : float
        Angle of incidence in radians.
    n_in, n_out : float
        Refractive indices of incoming and outgoing media.

    Returns
    -------
    float
        Angle of refraction in radians.

    Raises
    ------
    ValueError
        If total internal reflection would occur (sin > 1).
    """
    sin_out = (n_in / n_out) * np.sin(beta_in)
    if np.any(np.abs(sin_out) > 1.0):
        raise ValueError(
            f"Total internal reflection: n_in={n_in}, n_out={n_out}, "
            f"beta_in={np.rad2deg(beta_in):.2f}°"
        )
    return np.arcsin(sin_out)


def snell_all_angles(
    beta1: float,
    n1: float,
    n2_list: list[float],
    n3: float,
) -> tuple[list[float], float]:
    """
    Compute refraction angles through all glass layers and into the fluid.

    For a single-layer stack (nlay = 1) this directly gives Maas eq. (6):
        β₂ = asin( (n₁/n₂) · sin(β₁) )
        β₃ = asin( (n₁/n₃) · sin(β₁) )

    For multiple layers the invariant n₁·sin(β₁) = const propagates
    through each layer in sequence.

    Parameters
    ----------
    beta1 : float
        Angle of incidence in air (medium 1), radians.
    n1 : float
        Refractive index of air.
    n2_list : list of float
        Refractive indices of glass layers.
    n3 : float
        Refractive index of fluid.

    Returns
    -------
    beta2_list : list of float
        Angles inside each glass layer, radians.
    beta3 : float
        Angle inside the fluid, radians.
    """
    beta2_list = []
    beta_current = beta1
    n_current    = n1

    for n2 in n2_list:
        beta2 = snell_angle(beta_current, n_current, n2)
        beta2_list.append(beta2)
        beta_current = beta2
        n_current    = n2

    # Into the fluid (Maas notes n₁·sinβ₁ = n₃·sinβ₃ directly)
    beta3 = snell_angle(beta1, n1, n3)

    return beta2_list, beta3


# ---------------------------------------------------------------------------
# Snell's Law: direction-cosine form (used by 2.2.2)
# ---------------------------------------------------------------------------

def snell_direction_cosines(
    direction: np.ndarray,
    normal: np.ndarray,
    n_in: float,
    n_out: float,
) -> np.ndarray:
    """
    Apply Snell's Law in 3-D vector form.

    Given an incoming unit ray direction and a surface normal, compute
    the refracted ray direction.

    Maas (1996) eqs. (11) and (13):

        [a₂]   n₁   [a₁]   ( n₁             ┌─────────────────────────── )  [λ]
        [b₂] = ── · [b₁] - ( ── cos(i) - √(1-(n₁/n₂)² + (n₁/n₂ cos(i))²)  [μ]
        [c₂]   n₂   [c₁]   ( n₂                                              )  [ν]

    where (λ,μ,ν) = (0,0,1) for a horizontal flat interface (normal along Z),
    and cos(i) = c (the z-component of the incoming direction).

    This implementation handles an arbitrary interface normal.

    Parameters
    ----------
    direction : np.ndarray, shape (3,)
        Incoming unit ray direction vector.
    normal : np.ndarray, shape (3,)
        Unit surface normal pointing INTO the incoming medium.
    n_in, n_out : float
        Refractive indices.

    Returns
    -------
    np.ndarray, shape (3,)
        Refracted unit ray direction.

    Raises
    ------
    ValueError
        On total internal reflection.
    """
    direction = np.asarray(direction, dtype=float)
    normal    = np.asarray(normal,    dtype=float)

    # Normalise defensively
    direction = direction / np.linalg.norm(direction)
    normal    = normal    / np.linalg.norm(normal)

    ratio   = n_in / n_out
    cos_in  = -np.dot(direction, normal)      # cos of incidence angle
    sin2_in = 1.0 - cos_in**2
    sin2_out = ratio**2 * sin2_in

    if sin2_out > 1.0:
        raise ValueError(
            f"Total internal reflection: n_in={n_in}, n_out={n_out}, "
            f"cos(i)={cos_in:.4f}"
        )

    cos_out = np.sqrt(1.0 - sin2_out)

    # Vector Snell's Law (Hecht / Born & Wolf form)
    refracted = ratio * direction + (ratio * cos_in - cos_out) * normal
    return refracted / np.linalg.norm(refracted)


# ---------------------------------------------------------------------------
# Geometry: ray–plane intersection
# ---------------------------------------------------------------------------

def ray_plane_intersection(
    ray_origin: np.ndarray,
    ray_dir: np.ndarray,
    plane_z: float,
) -> np.ndarray:
    """
    Find where a ray intersects a horizontal plane z = plane_z.

    Used to locate the break-point Pв on the air/glass and glass/water
    interfaces (Maas 1996, Figure 3, eqs. 10 and 12).

    Parameters
    ----------
    ray_origin : np.ndarray, shape (3,)
        Starting point of the ray.
    ray_dir : np.ndarray, shape (3,)
        Direction of the ray (need not be normalised).
    plane_z : float
        Z-coordinate of the horizontal interface.

    Returns
    -------
    np.ndarray, shape (3,)
        Intersection point [X, Y, plane_z].

    Raises
    ------
    ValueError
        If the ray is parallel to the plane.
    """
    ray_origin = np.asarray(ray_origin, dtype=float)
    ray_dir    = np.asarray(ray_dir,    dtype=float)

    if abs(ray_dir[2]) < 1e-12:
        raise ValueError("Ray is parallel to the horizontal interface plane.")

    t = (plane_z - ray_origin[2]) / ray_dir[2]
    return ray_origin + t * ray_dir


# ---------------------------------------------------------------------------
# Geometry: radial distance from nadir
# ---------------------------------------------------------------------------

def radial_distance_from_nadir(
    point: np.ndarray,
    nadir: np.ndarray,
) -> float:
    """
    Compute the radial distance R in the X-Y plane between a 3-D point
    and the camera nadir point N.

    Maas (1996) eq. (4):
        R̄₍₀₎ = √((Xₚ - X₀)² + (Yₚ - Y₀)²)

    Parameters
    ----------
    point : np.ndarray, shape (3,)   — particle P or shifted P̄
    nadir : np.ndarray, shape (3,)   — camera nadir N = [X₀, Y₀, 0]

    Returns
    -------
    float : radial distance in mm
    """
    dx = point[0] - nadir[0]
    dy = point[1] - nadir[1]
    return float(np.sqrt(dx**2 + dy**2))


# ---------------------------------------------------------------------------
# Geometry: angle of incidence from radial geometry
# ---------------------------------------------------------------------------

def incidence_angle_from_radial(
    R: float,
    Z0: float,
    t_total: float,
    Zp: float,
) -> float:
    """
    Compute β₁ (angle of incidence in air) from the radial geometry.

    Maas (1996) eq. (5):
        β₁ = atan( R̄ / (Z₀ + t + Zₚ) )

    where Z₀ = height of camera above the glass top surface,
          t   = total glass thickness,
          Zₚ  = depth of particle below the glass bottom surface.

    All distances in mm, with Z₀, t, Zₚ all positive.

    Parameters
    ----------
    R      : float — radial distance in the X-Y plane (mm)
    Z0     : float — camera height above glass top (mm)
    t_total: float — total glass thickness (mm)
    Zp     : float — particle depth below glass bottom (mm)

    Returns
    -------
    float : β₁ in radians
    """
    return float(np.arctan(R / (Z0 + t_total + Zp)))


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== snell_core.py self-test ===\n")

    # Test 1: Snell's Law angle form
    beta1 = np.deg2rad(20.0)   # 20° in air
    b2, b3 = snell_all_angles(beta1, n1=1.0, n2_list=[1.5], n3=1.333)
    print(f"β₁ = {np.rad2deg(beta1):.2f}°  (air)")
    print(f"β₂ = {np.rad2deg(b2[0]):.4f}°  (glass n=1.5)   — expected ~13.18°")
    print(f"β₃ = {np.rad2deg(b3):.4f}°   (water n=1.333)  — expected ~14.90°")

    # Test 2: Direction-cosine form gives same angle
    normal = np.array([0.0, 0.0, 1.0])
    d_in   = np.array([np.sin(beta1), 0.0, -np.cos(beta1)])  # ray going down
    d_out  = snell_direction_cosines(d_in, normal, 1.0, 1.5)
    angle_out = np.arcsin(np.sqrt(d_out[0]**2 + d_out[1]**2))
    print(f"\nDirection-cosine β₂ = {np.rad2deg(angle_out):.4f}°  "
          f"— matches angle form: {np.isclose(angle_out, b2[0])}")

    # Test 3: Ray-plane intersection
    origin = np.array([0.0, 0.0, 300.0])
    direction = np.array([0.2, 0.0, -1.0])
    hit = ray_plane_intersection(origin, direction, plane_z=0.0)
    print(f"\nRay from {origin} hits z=0 plane at {hit}")

    print("\nAll self-tests passed.")
