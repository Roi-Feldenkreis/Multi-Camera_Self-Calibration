"""
multimedia_params.py
====================
Data structures for the multimedia geometry model.

Based on:
    Maas, H.-G. (1996). Contributions of Digital Photogrammetry to 3-D PTV.
    In: Three-Dimensional Velocity and Vorticity Measuring and Image Analysis
    Techniques, Kluwer Academic Publishers, pp. 191-207.

These structures mirror the C structs in liboptv (OpenPTV):
    - MultimediaParams  ←→  mm_np
    - CameraParams      ←→  control_par / Calibration combined
    - ImagePoint        ←→  (x', y') corrected image coordinates
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from typing import List


# ---------------------------------------------------------------------------
# Physical constants / typical defaults
# ---------------------------------------------------------------------------
N_AIR   = 1.000293   # refractive index of air  (n₁)
N_GLASS = 1.500      # refractive index of glass (n₂), typical borosilicate
N_WATER = 1.333      # refractive index of water (n₃) at ~20 °C, 589 nm


# ---------------------------------------------------------------------------
# MultimediaParams
# ---------------------------------------------------------------------------
@dataclass
class MultimediaParams:
    """
    Holds the three-media stack: air → glass → water.

    Mirrors  mm_np  in liboptv/include/parameters.h

    Attributes
    ----------
    n1 : float
        Refractive index of medium 1 (air, camera side).  Typically 1.0.
    n2 : list of float
        Refractive indices of the glass layer(s).
        Most experiments use a single glass wall → n2 = [1.5].
    d  : list of float
        Thickness of each glass layer in **mm**, one value per n2 entry.
    n3 : float
        Refractive index of medium 3 (fluid, e.g. water).  Typically ~1.333.

    Notes
    -----
    Maas (1996) treats a single plane-parallel glass wall, so nlay = 1.
    Kotowski (1988) generalized to arbitrary, non-parallel interfaces.
    The liboptv implementation stores up to 3 layers (n2[3], d[3]).
    """
    n1: float = N_AIR
    n2: List[float] = field(default_factory=lambda: [N_GLASS])
    d:  List[float] = field(default_factory=lambda: [5.0])   # mm
    n3: float = N_WATER

    def __post_init__(self):
        if len(self.n2) != len(self.d):
            raise ValueError(
                f"n2 and d must have the same length "
                f"(got {len(self.n2)} and {len(self.d)})"
            )
        if len(self.n2) == 0:
            raise ValueError("Must have at least one glass layer.")

    @property
    def nlay(self) -> int:
        """Number of glass layers."""
        return len(self.n2)

    @property
    def t_total(self) -> float:
        """Total glass thickness in mm (= t in Maas 1996)."""
        return float(np.sum(self.d))

    def is_air_only(self) -> bool:
        """
        Return True when all refractive indices equal 1.0
        (no multimedia correction needed).
        Mirrors the liboptv check:
            if (n1 != 1 || n2[0] != 1 || n3 != 1) → init LUT
        """
        return (
            abs(self.n1 - 1.0) < 1e-9
            and all(abs(n - 1.0) < 1e-9 for n in self.n2)
            and abs(self.n3 - 1.0) < 1e-9
        )

    def __repr__(self) -> str:
        return (
            f"MultimediaParams("
            f"n1={self.n1}, n2={self.n2}, d={self.d} mm, n3={self.n3})"
        )


# ---------------------------------------------------------------------------
# CameraParams
# ---------------------------------------------------------------------------
@dataclass
class CameraParams:
    """
    Full description of one camera in the multi-camera system.

    Combines exterior orientation, interior orientation, lens distortion,
    and A/D correction — the 16 parameters of Maas (1996), eq. (17).

    Exterior orientation  (6 parameters)
    ─────────────────────────────────────
    position  : camera projective centre O = (X₀, Y₀, Z₀) in mm
    angles    : (ω, φ, κ) in radians — rotations about X, Y, Z axes

    Interior orientation  (3 parameters)
    ─────────────────────────────────────
    c         : principal distance (focal length equivalent) in mm
    xh, yh    : principal point offset (image centre) in mm

    Lens distortion  (5 parameters)
    ────────────────────────────────
    k1, k2, k3 : radial distortion coefficients
    p1, p2     : decentering distortion coefficients

    A/D conversion  (2 parameters, Maas eq. 1)
    ────────────────────────────────────────────
    a1 : scale correction in x-image-coordinate direction
    a2 : shear correction

    Attributes
    ----------
    cam_id : int
        Camera index (0-based).
    position : np.ndarray, shape (3,)
        [X₀, Y₀, Z₀] in mm.
    angles : np.ndarray, shape (3,)
        [ω, φ, κ] in radians.
    c : float
        Principal distance in mm.
    xh : float
        x principal point in mm.
    yh : float
        y principal point in mm.
    k1, k2, k3 : float
        Radial distortion.
    p1, p2 : float
        Decentering distortion.
    a1, a2 : float
        A/D affine correction (Maas eq. 1).
    mm : MultimediaParams
        The multimedia stack shared by all cameras.
    """
    cam_id:   int   = 0
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    angles:   np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Interior orientation
    c:  float = 10.0    # mm
    xh: float = 0.0     # mm
    yh: float = 0.0     # mm

    # Lens distortion
    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    p1: float = 0.0
    p2: float = 0.0

    # A/D affine correction (Maas eq. 1)
    a1: float = 0.0
    a2: float = 0.0

    # Multimedia stack
    mm: MultimediaParams = field(default_factory=MultimediaParams)

    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=float)
        self.angles   = np.asarray(self.angles,   dtype=float)
        if self.position.shape != (3,):
            raise ValueError("position must be a 3-vector.")
        if self.angles.shape != (3,):
            raise ValueError("angles must be a 3-vector.")

    @property
    def rotation_matrix(self) -> np.ndarray:
        """
        Build the 3×3 rotation matrix A from (ω, φ, κ).

        Convention: A = Rz(κ) · Ry(φ) · Rx(ω)
        Maps object-space vectors into the camera coordinate system.

        Returns
        -------
        A : np.ndarray, shape (3, 3)
        """
        omega, phi, kappa = self.angles
        co, so = np.cos(omega), np.sin(omega)
        cp, sp = np.cos(phi),   np.sin(phi)
        ck, sk = np.cos(kappa), np.sin(kappa)

        # Standard photogrammetric rotation (Rx · Ry · Rz convention)
        A = np.array([
            [ cp*ck,           cp*sk,          -sp   ],
            [ so*sp*ck - co*sk, so*sp*sk + co*ck, so*cp],
            [ co*sp*ck + so*sk, co*sp*sk - so*ck, co*cp],
        ])
        return A

    @property
    def nadir_point(self) -> np.ndarray:
        """
        N = orthogonal projection of camera centre O onto the X-Y plane.
        Maas (1996): nadir point used as the origin for the radial shift ΔR.

        Returns
        -------
        N : np.ndarray, shape (3,)  — [X₀, Y₀, 0]
        """
        return np.array([self.position[0], self.position[1], 0.0])

    def __repr__(self) -> str:
        return (
            f"CameraParams(cam_id={self.cam_id}, "
            f"pos={self.position}, "
            f"angles={np.rad2deg(self.angles)} deg, "
            f"c={self.c} mm)"
        )


# ---------------------------------------------------------------------------
# ImagePoint
# ---------------------------------------------------------------------------
@dataclass
class ImagePoint:
    """
    A 2-D image coordinate, optionally corrected for distortion.

    Attributes
    ----------
    x, y    : float  — raw pixel or metric image coordinates
    x_corr, y_corr : float — corrected for lens distortion + A/D (eq. 1)
    """
    x: float = 0.0
    y: float = 0.0
    x_corr: float | None = None
    y_corr: float | None = None

    def corrected(self) -> tuple[float, float]:
        """Return corrected coordinates, falling back to raw if not set."""
        xc = self.x_corr if self.x_corr is not None else self.x
        yc = self.y_corr if self.y_corr is not None else self.y
        return xc, yc
