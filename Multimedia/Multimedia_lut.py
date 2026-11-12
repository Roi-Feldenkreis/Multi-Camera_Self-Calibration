"""
multimedia_lut.py
=================
Multimedia Look-Up Table (LUT) builder.

This module bridges Section 2.2.2 and Section 2.2.1.

Maas (1996), Section 2.2.1 (last paragraph):

    "A consequent extension of this idea is the implementation of the
    radial shift into a lookup-table (input: Depth in water Zᵢ and
    radial distance R from nadir point, output: Radial displacement
    factor R̄/R). The computational effort needed for the initialization
    of these lookup-tables (one for each camera) will be compensated by
    the reduced processing time for the particle image sequences, even
    if they are short."

HOW THE LUT CONNECTS 2.2.2 AND 2.2.1
--------------------------------------
The LUT is populated using the FORWARD ray-trace of Section 2.2.2:
  1. For each (R, Zp) grid point, fire a ray from the camera in air
     with a known incidence angle β₁.
  2. Trace through the glass and water (Section 2.2.2 physics).
  3. Record the resulting (R_actual, Zp) and the ratio R̄/R.
  4. Store: (R, Zp) → R̄/R

During calibration (Section 2.2.1), instead of iterating to convergence,
look up R̄/R from the table and compute the Cartesian shift immediately.

USAGE
-----
    from multimedia_lut import MultimediaLUT
    from multimedia_params import CameraParams, MultimediaParams

    mm  = MultimediaParams(n1=1.0, n2=[1.5], d=[5.0], n3=1.333)
    cam = CameraParams(position=[0,0,300], ...)
    
    lut = MultimediaLUT(cam, z_glass_top=0.0)
    lut.build(R_max=200.0, Zp_max=200.0, n_R=100, n_Zp=100)

    # Fast lookup during calibration
    P_bar = lut.shift_point(particle, cam)

Dependencies
------------
    snell_core.py
    multimedia_params.py
    Forward_ray_tracing.py    (for LUT population — the 2.2.2 forward trace)
    Iterative_radial_shift .py     (for the fallback iterative method)
"""

from __future__ import annotations
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from Multimedia.Snell_core import snell_all_angles, radial_distance_from_nadir
from Multimedia.Multimedia_params import CameraParams, MultimediaParams


# ---------------------------------------------------------------------------
# MultimediaLUT
# ---------------------------------------------------------------------------

class MultimediaLUT:
    """
    Per-camera lookup table for the multimedia radial displacement factor.

    The LUT stores R̄/R as a function of (R, Zp) where:
        R   = horizontal distance from camera nadir to particle (mm)
        Zp  = depth of particle below the glass bottom surface (mm)
        R̄/R = the ratio by which R must be scaled to get the corrected R̄

    Attributes
    ----------
    cam : CameraParams
        The camera this LUT belongs to.
    z_glass_top : float
        Z-coordinate of the glass top surface in mm.
    R_grid : np.ndarray, shape (n_R,)
        R grid values in mm.
    Zp_grid : np.ndarray, shape (n_Zp,)
        Zp grid values in mm.
    ratio_grid : np.ndarray, shape (n_R, n_Zp)
        R̄/R values on the grid.
    interpolator : scipy.interpolate.RegularGridInterpolator
        Bilinear interpolator for fast lookup.
    is_built : bool
        Whether the LUT has been populated.
    """

    def __init__(self, cam: CameraParams, z_glass_top: float = 0.0):
        self.cam         = cam
        self.z_glass_top = z_glass_top
        self.R_grid      = None
        self.Zp_grid     = None
        self.ratio_grid  = None
        self.interpolator = None
        self.is_built    = False

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(
        self,
        R_max: float = 300.0,
        Zp_max: float = 300.0,
        n_R: int = 100,
        n_Zp: int = 100,
        verbose: bool = True,
    ) -> None:
        """
        Populate the LUT using the Maas iterative formula (eqs. 4–7).

        For each grid point (R, Zp), we compute the exact R̄ via the
        iterative scheme and store R̄/R.

        This is done analytically (using snell_all_angles from snell_core.py),
        which is equivalent to what the 2.2.2 forward trace computes but
        expressed in the radial-geometry language of 2.2.1.

        Parameters
        ----------
        R_max : float
            Maximum radial distance to cover (mm).
        Zp_max : float
            Maximum particle depth below glass bottom (mm).
        n_R, n_Zp : int
            Grid resolution (number of points in each dimension).
        verbose : bool
            Print progress.
        """
        mm = self.cam.mm
        Z0_cam = self.cam.position[2]
        t      = mm.t_total

        # Camera height above glass top
        Z0 = Z0_cam  # assuming glass top at z=0

        # Grid — start from small positive value to avoid singularity at R=0
        R_vals  = np.linspace(0.01, R_max,  n_R)
        Zp_vals = np.linspace(0.01, Zp_max, n_Zp)

        ratio_grid = np.ones((n_R, n_Zp))   # default: no correction = 1.0

        if verbose:
            print(f"Building multimedia LUT for camera {self.cam.cam_id}...")
            print(f"  Grid: R=[0,{R_max}] mm  ×  Zp=[0,{Zp_max}] mm  "
                  f"({n_R}×{n_Zp} = {n_R*n_Zp} points)")

        # Whole grid at once (vectorised Maas eqs. 4–7). Identical equations,
        # tolerance and per-cell stopping rule as the scalar _solve_R_bar, which
        # is kept as the reference implementation (see lut_equivalence_test.py).
        ratio_grid = self._solve_ratio_grid(R_vals, Zp_vals, Z0, t, mm)

        self.R_grid     = R_vals
        self.Zp_grid    = Zp_vals
        self.ratio_grid = ratio_grid

        # Build interpolator
        self.interpolator = RegularGridInterpolator(
            (R_vals, Zp_vals),
            ratio_grid,
            method='linear',
            bounds_error=False,
            fill_value=None,   # extrapolate at edges
        )
        self.is_built = True

        if verbose:
            print(f"  Done. Ratio range: [{ratio_grid.min():.4f}, "
                  f"{ratio_grid.max():.4f}]")

    @staticmethod
    def _solve_ratio_grid(R_vals, Zp_vals, Z0, t, mm,
                          eps: float = 1e-6, max_iter: int = 50,
                          ocf: float = 1.3) -> np.ndarray:
        """
        Vectorised version of _solve_R_bar over the full (R, Zp) grid.

        Every cell follows exactly the scalar recipe:
            R̄ <- R;  repeat: β1 = atan(R̄/(Z0+t+Zp));  Snell through the layers;
            R_ray = Z0·tanβ1 + Σ dᵢ·tanβ2ᵢ + Zp·tanβ3;  dR = R − R_ray;
            stop if |dR| < eps, else R̄ += ocf·dR
        A cell that converges stops updating (the scalar 'break'); a cell that
        hits total internal reflection returns R̄ = R (the scalar 'return R').
        Snell's law: n1·sinβ1 = n2ᵢ·sinβ2ᵢ = n3·sinβ3.

        Returns ratio_grid (n_R, n_Zp) = R̄ / R.
        """
        Rm, Zm = np.meshgrid(np.asarray(R_vals, float), np.asarray(Zp_vals, float),
                             indexing='ij')
        R_bar  = Rm.copy()
        active = np.ones(Rm.shape, dtype=bool)
        n2s    = [float(n) for n in mm.n2]
        ds     = [float(d) for d in mm.d]

        for _ in range(max_iter):
            if not active.any():
                break
            beta1 = np.arctan(R_bar / (Z0 + t + Zm))
            s     = mm.n1 * np.sin(beta1)

            # total internal reflection -> R̄ = R, cell frozen
            tir = np.zeros(Rm.shape, dtype=bool)
            for n in n2s + [float(mm.n3)]:
                tir |= np.abs(s / n) > 1.0
            tir &= active
            if tir.any():
                R_bar  = np.where(tir, Rm, R_bar)
                active &= ~tir

            with np.errstate(invalid='ignore'):
                layer_sum = 0.0
                for d_i, n_i in zip(ds, n2s):
                    layer_sum = layer_sum + d_i * np.tan(np.arcsin(s / n_i))
                beta3 = np.arcsin(s / mm.n3)
            R_ray = Z0 * np.tan(beta1) + layer_sum + Zm * np.tan(beta3)
            dR    = Rm - R_ray

            converged = active & (np.abs(dR) < eps)
            active   &= ~converged
            R_bar     = np.where(active, R_bar + ocf * dR, R_bar)

        return R_bar / Rm

    def _solve_R_bar(
        self,
        R: float,
        Z0: float,
        t: float,
        Zp: float,
        mm: MultimediaParams,
        eps: float = 1e-6,
        max_iter: int = 50,
        ocf: float = 1.3,) -> float:
        """
        Solve for R̄ using Maas eqs. (4)–(7).

        This is the same iterative core as Iterative_radial_shift .multimedia_radial_shift
        but operates directly on scalar (R, Z0, t, Zp) geometry — no
        CameraParams needed — making it fast for the grid sweep.
        """
        R_bar = R   # initial estimate (eq. 4)

        for _ in range(max_iter):
            beta1 = np.arctan(R_bar / (Z0 + t + Zp))

            try:
                beta2_list, beta3 = snell_all_angles(
                    beta1, mm.n1, mm.n2, mm.n3
                )
            except ValueError:
                # Total internal reflection — clamp to R_bar = R (no shift)
                return R

            R_ray = (Z0 * np.tan(beta1)
                     + sum(d_i * np.tan(b2)
                           for d_i, b2 in zip(mm.d, beta2_list))
                     + Zp * np.tan(beta3))

            # dR = how far refracted ray falls short of the TRUE target R
            dR = R - R_ray      # ← fixed R is the target
            if abs(dR) < eps:
                break
            R_bar += ocf * dR

        return R_bar

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def lookup_ratio(self, R: float, Zp: float) -> float:
        """
        Look up the displacement ratio R̄/R for given (R, Zp).

        Parameters
        ----------
        R : float — radial distance from nadir in mm
        Zp : float — particle depth below glass bottom in mm

        Returns
        -------
        float : R̄/R  (1.0 if no multimedia correction)
        """
        if not self.is_built:
            raise RuntimeError("LUT has not been built. Call build() first.")

        if abs(R) < 1e-9:
            return 1.0

        ratio = float(self.interpolator([[R, Zp]])[0])
        return ratio

    def shift_point(
        self,
        particle: np.ndarray,
        cam: CameraParams) -> np.ndarray:
        """
        Fast LUT-based radial shift: equivalent to multimedia_radial_shift
        from Iterative_radial_shift .py but without iteration.

        Parameters
        ----------
        particle : np.ndarray, shape (3,)
            True 3-D particle position [X, Y, Z] in mm.
        cam : CameraParams
            Camera (uses position for nadir; should match self.cam).

        Returns
        -------
        P_bar : np.ndarray, shape (3,)
            Radially shifted point for use in the collinearity condition.
        """
        mm = cam.mm
        t  = mm.t_total
        X0, Y0, _ = cam.position

        nadir = np.array([X0, Y0, 0.0])
        R = radial_distance_from_nadir(particle, nadir)

        # Derive Zp from world Z of particle
        Zp = -(particle[2] + t)
        if Zp < 0:
            Zp = 0.0

        ratio = self.lookup_ratio(R, Zp)
        R_bar = R * ratio

        if R < 1e-9:
            return particle.copy()

        scale = R_bar / R
        X_bar = X0 + (particle[0] - X0) * scale
        Y_bar = Y0 + (particle[1] - Y0) * scale

        return np.array([X_bar, Y_bar, particle[2]])

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def max_shift_mm(self) -> float:
        """Return the maximum ΔR = |R̄ - R| observed across the grid (mm)."""
        if not self.is_built:
            return 0.0
        R_mesh, _ = np.meshgrid(self.R_grid, self.Zp_grid, indexing='ij')
        delta_R = np.abs((self.ratio_grid - 1.0) * R_mesh)
        return float(delta_R.max())

    def plot(self, ax=None):
        """
        Plot the displacement ratio R̄/R as a 2-D contour over (R, Zp).

        Requires matplotlib.
        """
        import matplotlib.pyplot as plt

        if not self.is_built:
            raise RuntimeError("LUT not built.")

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        R_mesh, Zp_mesh = np.meshgrid(self.R_grid, self.Zp_grid, indexing='ij')
        delta = (self.ratio_grid - 1.0) * R_mesh   # ΔR in mm

        cf = ax.contourf(R_mesh, Zp_mesh, delta, levels=20, cmap='viridis')
        plt.colorbar(cf, ax=ax, label='ΔR = R̄ - R  (mm)')
        ax.set_xlabel('R — radial distance from nadir (mm)')
        ax.set_ylabel('Zp — depth in water (mm)')
        ax.set_title(
            f'Multimedia radial displacement ΔR\n'
            f'Camera {self.cam.cam_id}  |  '
            f'n₁={self.cam.mm.n1}, n₂={self.cam.mm.n2}, '
            f'n₃={self.cam.mm.n3}'
        )
        return ax


# ---------------------------------------------------------------------------
# Multi-camera LUT manager
# ---------------------------------------------------------------------------

class MultiCameraLUT:
    """
    Manages one LUT per camera in a multi-camera self-calibration system.

    Usage
    -----
        mc_lut = MultiCameraLUT(cameras, z_glass_top=0.0)
        mc_lut.build_all()

        # In calibration loop:
        P_bar = mc_lut[cam_id].shift_point(particle, camera)
    """

    def __init__(
        self,
        cameras: list[CameraParams],
        z_glass_top: float = 0.0,
    ):
        self.cameras     = cameras
        self.z_glass_top = z_glass_top
        self.luts: dict[int, MultimediaLUT] = {}

        for cam in cameras:
            self.luts[cam.cam_id] = MultimediaLUT(cam, z_glass_top)

    def build_all(self, **kwargs) -> None:
        """Build the LUT for every camera. Kwargs are passed to LUT.build()."""
        for cam_id, lut in self.luts.items():
            lut.build(**kwargs)

    def __getitem__(self, cam_id: int) -> MultimediaLUT:
        return self.luts[cam_id]

    def is_multimedia(self) -> bool:
        """Return True if any camera has non-trivial multimedia parameters."""
        return any(not cam.mm.is_air_only() for cam in self.cameras)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from Multimedia.Multimedia_params import MultimediaParams, CameraParams, N_AIR, N_GLASS, N_WATER
    from Multimedia.Iterative_radial_shift import multimedia_radial_shift

    print("=== multimedia_lut.py self-test ===\n")

    mm = MultimediaParams(n1=N_AIR, n2=[N_GLASS], d=[5.0], n3=N_WATER)
    cam = CameraParams(
        cam_id=0,
        position=np.array([0.0, 0.0, 300.0]),
        angles=np.zeros(3),
        c=10.0,
        mm=mm,
    )

    # Build LUT
    lut = MultimediaLUT(cam, z_glass_top=0.0)
    lut.build(R_max=200.0, Zp_max=200.0, n_R=80, n_Zp=80)

    print(f"\nMax ΔR across the grid: {lut.max_shift_mm():.3f} mm\n")

    # Test: compare LUT shift vs iterative shift for several particles
    test_particles = [
        np.array([30.0,  15.0, -55.0]),
        np.array([80.0,   0.0, -30.0]),
        np.array([ 0.0,  50.0, -100.0]),
        np.array([120.0, 80.0, -20.0]),
    ]

    print(f"{'Particle':>35}  {'Iter ΔR':>10}  {'LUT ΔR':>10}  {'Error':>10}")
    print("-" * 75)

    for p in test_particles:
        # Iterative (exact)
        P_bar_iter, _, _ = multimedia_radial_shift(p, cam, eps=1e-7)
        dR_iter = np.linalg.norm(P_bar_iter[:2] - p[:2])

        # LUT (approximate)
        P_bar_lut = lut.shift_point(p, cam)
        dR_lut = np.linalg.norm(P_bar_lut[:2] - p[:2])

        error = abs(dR_iter - dR_lut)
        print(f"P={p}  {dR_iter:>10.4f}  {dR_lut:>10.4f}  {error:>10.6f} mm")

    print("\nMultiCameraLUT demo:")
    cams = [
        CameraParams(cam_id=i, position=np.array([i*100.0-150, 0, 300]),
                     angles=np.zeros(3), c=10.0, mm=mm)
        for i in range(3)
    ]
    mc_lut = MultiCameraLUT(cams, z_glass_top=0.0)
    mc_lut.build_all(R_max=200.0, Zp_max=200.0, n_R=40, n_Zp=40, verbose=False)
    print(f"Built LUTs for {len(mc_lut.luts)} cameras.")
    print(f"Is multimedia: {mc_lut.is_multimedia()}")
    print("\nAll tests passed.")
