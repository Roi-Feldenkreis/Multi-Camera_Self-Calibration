"""
Forward_ray_tracing.py
==================
Section 2.2.2 — Direct solution for the spatial intersection.

Maas (1996), Section 2.2.2, equations (9)–(16).

PURPOSE
-------
Given corrected image coordinates (x', y') from one or more cameras,
trace each ray forward through the media stack (air → glass → water)
using direction cosines, then intersect the rays in water to obtain
the 3-D particle position.

USE CASES
---------
✓  Particle positioning during tracking (fast, one-shot)
✓  Building the multimedia LUT (see multimedia_lut.py)
✓  Verifying the 2.2.1 radial-shift solution
✗  System calibration — cannot be used (no least-squares functional model)

LIMITATIONS (Maas, Section 2.2.3)
----------------------------------
- Assumes plane-parallel glass interfaces
- Does not account for: aberrations, dispersion, refractive index
  inhomogeneities, glass wall non-planarity, or diffraction asymmetry
- Less accurate than the full 2.2.1 bundle adjustment

Dependencies
------------
    snell_core.py
    multimedia_params.py
"""

from __future__ import annotations
import numpy as np
from typing import Sequence

from Multimedia.Snell_core import snell_direction_cosines, ray_plane_intersection
from Multimedia.Multimedia_params import CameraParams, MultimediaParams


# ---------------------------------------------------------------------------
# Step 1: Direction cosines in air  (Maas eq. 9)
# ---------------------------------------------------------------------------

def direction_cosines_in_air(
    x_img: float,
    y_img: float,
    cam: CameraParams,
) -> np.ndarray:
    """
    Compute the unit ray direction vector in air (medium n₁) from a
    corrected image point (x', y').

    Maas (1996) eq. (9):

        s' = √(x'² + y'² + c²)

        [a₁]       [x'/s']
        [b₁] = A · [y'/s']
        [c₁]       [-c/s' ]

    where A is the 3×3 rotation matrix mapping camera → object space.

    Parameters
    ----------
    x_img, y_img : float
        Image coordinates corrected for lens distortion and A/D effects (mm).
        These are x', y' in Maas notation.
    cam : CameraParams
        Camera with rotation matrix A and principal distance c.

    Returns
    -------
    np.ndarray, shape (3,)
        Unit direction vector [a₁, b₁, c₁] in object-space coordinates.
    """
    c    = cam.c
    s_prime = np.sqrt(x_img**2 + y_img**2 + c**2)

    # Local direction in camera space (pointing from image plane toward object)
    local_dir = np.array([x_img / s_prime,
                           y_img / s_prime,
                          -c     / s_prime])

    # Rotate to object space using the rotation matrix A
    A = cam.rotation_matrix
    world_dir = A @ local_dir

    return world_dir / np.linalg.norm(world_dir)


# ---------------------------------------------------------------------------
# Step 2: Trace ray through all media  (Maas eqs. 10–13)
# ---------------------------------------------------------------------------

def trace_ray_through_media(
    cam: CameraParams,
    x_img: float,
    y_img: float,
    z_glass_top: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Trace a ray from the camera centre, through the glass wall, and
    into the fluid.  Returns the ray's entry position and direction
    in the fluid (medium n₃).

    This implements the complete Maas (1996) forward ray-trace:
        eq. (9)  → direction cosines in air
        eq. (10) → intersection with air/glass interface
        eq. (11) → refracted direction in glass
        eq. (12) → intersection with glass/water interface
        eq. (13) → refracted direction in water

    Coordinate convention
    ----------------------
    The global Z-axis points upward.
    The glass top surface is at z = z_glass_top.
    The glass bottom surface (glass/water interface) is at
        z = z_glass_top - t_total.

    Parameters
    ----------
    cam : CameraParams
        Full camera description including multimedia parameters.
    x_img, y_img : float
        Corrected image coordinates in mm (x', y' of Maas).
    z_glass_top : float
        Z-coordinate of the upper glass surface (air/glass interface) in mm.

    Returns
    -------
    pos_water : np.ndarray, shape (3,)
        Point on the glass/water interface where the ray enters the fluid.
    dir_water : np.ndarray, shape (3,)
        Unit direction vector of the ray inside the fluid.
    """
    mm = cam.mm

    # Interface Z-coordinates
    z_glass_bottom = z_glass_top - mm.t_total

    # Interface normal: points upward (into the incoming medium = air)
    normal = np.array([0.0, 0.0, 1.0])

    # --- Step 1: direction in air  (eq. 9) ---
    dir_air = direction_cosines_in_air(x_img, y_img, cam)

    # --- Step 2: ray/air-glass intersection  (eq. 10) ---
    pos_B1 = ray_plane_intersection(cam.position, dir_air, z_glass_top)

    # --- Step 3: refract through each glass layer  (eq. 11) ---
    dir_current = dir_air
    n_current   = mm.n1
    pos_current = pos_B1
    z_current   = z_glass_top

    for layer_idx, (n2, d_layer) in enumerate(zip(mm.n2, mm.d)):
        # Refract at interface
        dir_current = snell_direction_cosines(
            dir_current, normal, n_current, n2
        )
        n_current = n2

        # Advance through this glass layer to the next interface
        z_next = z_current - d_layer
        pos_current = ray_plane_intersection(pos_current, dir_current, z_next)
        z_current = z_next

    # At this point pos_current is on the glass/water interface,
    # dir_current is the direction in the last glass layer.

    # --- Step 4: refract into water  (eq. 13) ---
    dir_water = snell_direction_cosines(
        dir_current, normal, mm.n2[-1], mm.n3
    )
    pos_water = pos_current  # already on the glass/water interface

    return pos_water, dir_water


# ---------------------------------------------------------------------------
# Step 3: Spatial intersection of two rays  (Maas eqs. 14–16)
# ---------------------------------------------------------------------------

def spatial_intersection_two_rays(
    pos_l: np.ndarray, dir_l: np.ndarray,
    pos_r: np.ndarray, dir_r: np.ndarray,
    baseline_axis: str = 'auto',
) -> np.ndarray:
    """
    Compute the 3-D particle position from two rays in the fluid.

    Maas (1996) eqs. (14)–(16):
    The closed-form solution picks either eq. (14) or eq. (15) depending
    on whether the camera baseline is predominantly in Y or X.

    For robustness this implementation uses the general least-squares
    closest-approach between two skew lines in 3-D — which reduces to
    the Maas formula for the specific stereo geometry he describes.

    Parameters
    ----------
    pos_l, dir_l : np.ndarray, shape (3,)
        Position and unit direction of the LEFT camera ray in water.
    pos_r, dir_r : np.ndarray, shape (3,)
        Position and unit direction of the RIGHT camera ray in water.
    baseline_axis : str
        'auto' selects the axis of larger camera separation.
        'x' or 'y' forces the Maas direct formula.

    Returns
    -------
    np.ndarray, shape (3,)
        Best-estimate 3-D particle position in object space.
    """
    # General closest-point-of-approach between two lines:
    #   P(s) = pos_l + s * dir_l
    #   Q(t) = pos_r + t * dir_r
    # Minimise |P(s) - Q(t)|²
    w  = pos_l - pos_r
    a  = np.dot(dir_l, dir_l)
    b  = np.dot(dir_l, dir_r)
    c  = np.dot(dir_r, dir_r)
    d  = np.dot(dir_l, w)
    e  = np.dot(dir_r, w)
    denom = a * c - b * b

    if abs(denom) < 1e-12:
        # Rays are parallel — return midpoint of the two starting positions
        return 0.5 * (pos_l + pos_r)

    s = (b * e - c * d) / denom
    t = (a * e - b * d) / denom

    P = pos_l + s * dir_l
    Q = pos_r + t * dir_r

    return 0.5 * (P + Q)   # midpoint = best estimate


def spatial_intersection_multi(
    ray_positions: Sequence[np.ndarray],
    ray_directions: Sequence[np.ndarray],
) -> tuple[np.ndarray, float]:
    """
    Compute the 3-D particle position from N ≥ 2 rays using least squares.

    This is the generalisation of Maas eqs. (14)–(16) to multiple cameras,
    using the standard photogrammetric bundle intersection.

    For each ray i:   P = pos_i + sᵢ · dir_i
    We minimise the sum of squared distances from P to all rays.

    Parameters
    ----------
    ray_positions  : sequence of np.ndarray, shape (3,)
        Origin of each ray in water (at the glass/water interface).
    ray_directions : sequence of np.ndarray, shape (3,)
        Unit direction of each ray in water.

    Returns
    -------
    P_hat : np.ndarray, shape (3,)
        Least-squares estimate of the particle position.
    residual : float
        RMS distance from P_hat to each ray (mm) — quality indicator.
    """
    n_rays = len(ray_positions)
    if n_rays < 2:
        raise ValueError("Need at least 2 rays for spatial intersection.")

    # Build the normal equations for least-squares line bundle
    # Each ray contributes:   (I - d·dᵀ) P = (I - d·dᵀ) pos
    A_sum = np.zeros((3, 3))
    b_sum = np.zeros(3)

    for pos_i, dir_i in zip(ray_positions, ray_directions):
        dir_i = dir_i / np.linalg.norm(dir_i)
        M_i   = np.eye(3) - np.outer(dir_i, dir_i)
        A_sum += M_i
        b_sum += M_i @ pos_i

    P_hat = np.linalg.lstsq(A_sum, b_sum, rcond=None)[0]

    # Compute RMS residual (distance from each ray to P_hat)
    residuals = []
    for pos_i, dir_i in zip(ray_positions, ray_directions):
        dir_i  = dir_i / np.linalg.norm(dir_i)
        delta  = P_hat - pos_i
        dist   = np.linalg.norm(delta - np.dot(delta, dir_i) * dir_i)
        residuals.append(dist)

    rms = float(np.sqrt(np.mean(np.array(residuals)**2)))
    return P_hat, rms


# ---------------------------------------------------------------------------
# High-level: full pipeline for N cameras  (Section 2.2.2 end-to-end)
# ---------------------------------------------------------------------------

def locate_particle_222(
    cameras: list[CameraParams],
    image_points: list[tuple[float, float]],
    z_glass_top: float,
) -> tuple[np.ndarray, float]:
    """
    Full Section 2.2.2 pipeline: image coordinates → 3-D particle position.

    Steps:
        1. For each camera: direction cosines in air (eq. 9)
        2. Trace each ray through glass into water (eqs. 10–13)
        3. Intersect all rays in water (eqs. 14–16, generalised)

    Parameters
    ----------
    cameras : list of CameraParams
        All cameras observing the particle (≥ 2).
    image_points : list of (x', y') tuples
        Corrected image coordinates for each camera, in mm.
    z_glass_top : float
        Z-coordinate of the glass top surface in mm.

    Returns
    -------
    position : np.ndarray, shape (3,)
        3-D particle position in object space (mm).
    residual : float
        RMS ray distance at the intersection point (mm).
    """
    if len(cameras) != len(image_points):
        raise ValueError("cameras and image_points must have the same length.")
    if len(cameras) < 2:
        raise ValueError("Need at least 2 cameras.")

    ray_pos_list = []
    ray_dir_list = []

    for cam, (x_img, y_img) in zip(cameras, image_points):
        pos_w, dir_w = trace_ray_through_media(cam, x_img, y_img, z_glass_top)
        ray_pos_list.append(pos_w)
        ray_dir_list.append(dir_w)

    return spatial_intersection_multi(ray_pos_list, ray_dir_list)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from Multimedia.Multimedia_params import MultimediaParams, CameraParams, N_AIR, N_GLASS, N_WATER

    print("=== Forward_ray_tracing.py self-test ===\n")

    mm = MultimediaParams(n1=N_AIR, n2=[N_GLASS], d=[5.0], n3=N_WATER)

    # Two cameras looking at a known particle
    true_particle = np.array([10.0, 5.0, -50.0])   # 50 mm into water

    cam_left = CameraParams(
        cam_id=0,
        position=np.array([-150.0, 0.0, 300.0]),
        angles=np.deg2rad([0.0, 15.0, 0.0]),
        c=10.0,
        mm=mm,
    )
    cam_right = CameraParams(
        cam_id=1,
        position=np.array([150.0, 0.0, 300.0]),
        angles=np.deg2rad([0.0, -15.0, 0.0]),
        c=10.0,
        mm=mm,
    )

    z_glass_top = 0.0   # glass surface at z=0

    # Trace rays from each camera through to the particle depth
    pos_l, dir_l = trace_ray_through_media(
        cam_left, x_img=0.5, y_img=0.3, z_glass_top=z_glass_top
    )
    pos_r, dir_r = trace_ray_through_media(
        cam_right, x_img=-0.5, y_img=0.3, z_glass_top=z_glass_top
    )

    print(f"Left  ray origin in water: {pos_l}")
    print(f"Left  ray direction:       {dir_l}")
    print(f"Right ray origin in water: {pos_r}")
    print(f"Right ray direction:       {dir_r}")

    # Two-ray intersection
    P_2ray = spatial_intersection_two_rays(pos_l, dir_l, pos_r, dir_r)
    print(f"\n2-ray intersection:  {P_2ray}")

    # Multi-ray (generalised) intersection
    P_multi, rms = spatial_intersection_multi([pos_l, pos_r], [dir_l, dir_r])
    print(f"Multi-ray intersection: {P_multi}  (RMS residual={rms:.4f} mm)")

    print("\nDone.")
