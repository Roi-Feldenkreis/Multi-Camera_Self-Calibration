# Multimedia Geometry for Multi-Camera Self-Calibration

> Python implementation of Maas (1996) multimedia photogrammetry for
> 3D-PTV systems where cameras observe particles through glass and water.

---

## Table of Contents

1. [Background and Motivation](#1-background-and-motivation)
2. [The Physical Problem](#2-the-physical-problem)
3. [Mathematical Model — Maas (1996)](#3-mathematical-model--maas-1996)
   - 3.1 [Section 2.2.1 — Radial Shift (Calibration Model)](#31-section-221--radial-shift-calibration-model)
   - 3.2 [Section 2.2.2 — Direct Ray Tracing (Particle Positioning)](#32-section-222--direct-ray-tracing-particle-positioning)
   - 3.3 [Connection Between the Two Sections](#33-connection-between-the-two-sections)
   - 3.4 [Non-Modelled Effects](#34-non-modelled-effects)
4. [Relationship to OpenPTV / PyPTV](#4-relationship-to-openptv--pyptv)
5. [File Architecture](#5-file-architecture)
6. [Module Reference](#6-module-reference)
   - 6.1 [multimedia_params.py](#61-multimedia_paramspy)
   - 6.2 [snell_core.py](#62-snell_corepy)
   - 6.3 [Forward_ray_tracing.py](#63-ray_tracing_222py)
   - 6.4 [Iterative_radial_shift .py](#64-multimedia_221py)
   - 6.5 [multimedia_lut.py](#65-multimedia_lutpy)
7. [Pipeline Integration](#7-pipeline-integration)
8. [Quick-Start Usage](#8-quick-start-usage)
9. [Key Equations Reference](#9-key-equations-reference)
10. [Dependencies](#10-dependencies)
11. [References](#11-references)

---

## 1. Background and Motivation

In a typical 3D Particle Tracking Velocimetry (3D-PTV) or multi-camera
self-calibration experiment for fluid mechanics, cameras sit in **air** but
the measurement volume is **water**, separated by a **glass wall**.

Light from each particle therefore crosses three optical media before
reaching the camera sensor:

```
  Particle (water, n ≈ 1.333)  →  glass wall (n ≈ 1.5)  →  air (n = 1.0)  →  camera
```

At each interface, light refracts according to Snell's Law. This means
every ray is **bent twice** — once at the water/glass boundary and once
at the glass/air boundary. The standard pinhole camera (collinearity)
model, which assumes light travels in straight lines from object to sensor,
is **systematically wrong** if this refraction is ignored.

This module provides the complete mathematical machinery to account for
this "multimedia geometry", following the authoritative treatment by
**Maas (1996)**, the same paper that forms the theoretical foundation of
the OpenPTV / liboptv ecosystem.

---

## 2. The Physical Problem

### The setup

```
          Camera O(X₀, Y₀, Z₀)
               |
               |  β₁  ray in air (n₁ ≈ 1.0)
               |
     ══════════╪════════════  glass top surface (z = 0)
               |  β₂  ray in glass (n₂ ≈ 1.5)   thickness t
     ══════════╪════════════  glass bottom / water surface (z = -t)
               |  β₃  ray in water (n₃ ≈ 1.333)
               |
               P(Xₚ, Yₚ, Zₚ)   ← true particle position
               P̄(X̄, Ȳ, Zₚ)    ← apparent (shifted) position
```

Typical refractive index values:

| Medium | Refractive index | Typical value |
|--------|-----------------|---------------|
| Air    | n₁              | 1.000293      |
| Glass  | n₂              | 1.5 (borosilicate) |
| Water  | n₃              | 1.333 (at 20 °C, 589 nm) |

### Why the standard model fails

Because rays are bent toward the denser medium (water → glass at the
first interface), the apparent position of the particle as seen from the
camera is displaced **radially outward** from the camera nadir. If you
triangulate using uncorrected rays, every 3D point will be at the wrong
location. This is not a small effect: at a 20° incidence angle the radial
displacement is on the order of 1–2 mm for a typical tank experiment.

---

## 3. Mathematical Model — Maas (1996)

The paper introduces two complementary solution strategies.

### 3.1 Section 2.2.1 — Radial Shift (Calibration Model)

**This is the approach to use for system calibration.**

Instead of modifying the ray-tracing, this method computes a **radial
shift ΔR** of the object point P, producing a corrected point P̄ that,
when substituted into the standard collinearity condition, gives the
correct predicted image coordinates.

**Key insight:** because the glass wall is a plane-parallel plate and the
interfaces are horizontal, refraction has no effect on the Z-coordinate
of the particle — it only causes a radial displacement in the X-Y plane
relative to each camera's **nadir point** N = (X₀, Y₀, 0).

#### The geometry (Maas eq. 2 and 3)

```
Z₀·tan(β₁) + t·tan(β₂) + Zₚ·tan(β₃) = R         (geometry)
n₁·sin(β₁) = n₂·sin(β₂) = n₃·sin(β₃)             (Snell's Law)
```

where:
- `Z₀` = height of camera above glass top surface
- `t`  = total glass thickness
- `Zₚ` = depth of particle below glass bottom
- `R`  = true horizontal distance from camera nadir to particle
- `R̄`  = apparent (shifted) horizontal distance

#### The iterative algorithm (Maas eqs. 4–8)

```
1.  R̄₍₀₎ = √((Xₚ - X₀)² + (Yₚ - Y₀)²)          initial estimate (eq. 4)

2.  β₁ = atan( R̄ / (Z₀ + t + Zₚ) )               angle in air (eq. 5)

3.  β₂ = asin( (n₁/n₂)·sin(β₁) )                 via Snell in glass (eq. 6)
    β₃ = asin( (n₁/n₃)·sin(β₁) )                 via Snell in water (eq. 6)

4.  R_ray = Z₀·tan(β₁) + t·tan(β₂) + Zₚ·tan(β₃) actual ray reach (eq. 7)

5.  dR = R - R_ray                                 residual

6.  R̄ ← R̄ + ocf·dR                               update (ocf ≈ 1.1–1.8)

7.  Repeat steps 2–6 until |dR| < ε  (e.g. ε = 0.001 mm)
```

⚠️ **Critical note on the iteration direction:**  
`dR = R - R_ray` — the iteration adjusts the virtual angle `β₁` until
the refracted ray reaches the **fixed true particle radius R**. The
over-compensation factor `ocf` (Maas suggests 1.1–1.8) accelerates
convergence significantly. Convergence is typically reached in 10–15
iterations.

#### Back to Cartesian (Maas eq. 8)

```
X̄ₚ = X₀ + (Xₚ - X₀)·(R̄/R)
Ȳₚ = Y₀ + (Yₚ - Y₀)·(R̄/R)
Z̄ₚ = Zₚ                         ← Z is unchanged
```

The shifted point P̄ is then used in the **standard collinearity condition**:

```
x' = -c · (r₁₁(X̄-X₀) + r₁₂(Ȳ-Y₀) + r₁₃(Z̄-Z₀)) / D  + xₕ
y' = -c · (r₂₁(X̄-X₀) + r₂₂(Ȳ-Y₀) + r₂₃(Z̄-Z₀)) / D  + yₕ
```

where D is the denominator involving the third row of the rotation matrix.

#### Why this enables least-squares calibration

Because the multimedia correction is formulated as a shift of the object
point **before** the collinearity condition, the entire expression can be:
- Linearised around current parameter estimates
- Differentiated to produce a Jacobian `J[2 × 16]`
- Used as observation equations in a bundle adjustment

The 16 camera parameters (Maas eq. 17) are:

```
[X₀, Y₀, Z₀,   ω, φ, κ,   c, xₕ, yₕ,   k₁, k₂, k₃,   p₁, p₂,   a₁, a₂]
 ──────────────  ────────  ───────────  ───────────────  ──────  ──────
 exterior (6)    interior   radial dist  decentering      A/D
                 orient.(3) (3 params)   (2 params)       (2)
```

### 3.2 Section 2.2.2 — Direct Ray Tracing (Particle Positioning)

**This is the approach used by OpenPTV / PyPTV for particle tracking.**

Given corrected image coordinates `(x', y')`, this method traces the ray
forward through each medium using direction cosines, then intersects the
rays from multiple cameras in the water volume.

#### Direction cosines in air (Maas eq. 9)

```
s' = √(x'² + y'² + c²)

[a₁]       [x'/s']
[b₁]  = A · [y'/s']       A = camera rotation matrix
[c₁]       [-c/s' ]
```

#### Refraction at air/glass interface (Maas eq. 11)

```
[a₂]    n₁   [a₁]   ( n₁             ________________________  ) [λ]
[b₂]  = ── · [b₁] - ( ──·cos(i₁) - √(1-(n₁/n₂)²+(n₁/n₂·cos i₁)²) ) [μ]
[c₂]    n₂   [c₁]   ( n₂                                          ) [ν]
```

where (λ,μ,ν) = (0,0,1) for a horizontal interface, so `cos(i₁) = c₁`.

#### Intersection with glass/water interface (Maas eq. 12)

```
[X_B2]   d    [a₂]   [X_B1]
[Y_B2] = ── · [b₂] + [Y_B1]
[Z_B2]   c₂   [c₂]   [Z_B1]
```

#### Refraction into water (Maas eq. 13) — same form as eq. 11

#### Spatial intersection (Maas eqs. 14–16)

Once rays from all cameras are in the water, their intersection gives
the 3D particle position. The implementation uses the general
**least-squares closest approach** between N skew lines, which generalises
Maas's closed-form stereo formula to any number of cameras.

#### Limitations of Section 2.2.2

As Maas explicitly states, this method:
- **Cannot** be used for calibration (no functional model for resection)
- Cannot provide partial derivatives for a least-squares system
- Is a one-shot geometric solver with no error propagation

### 3.3 Connection Between the Two Sections

The two sections are solving **the same geometry from opposite ends**:

```
                     2.2.2 direction (FORWARD)
                  ─────────────────────────────►
Image pixel (x',y') → direction cosines → refract → intersect in water → 3D point P

3D point P → compute ΔR → shifted P̄ → collinearity → image pixel (x',y')
                  ◄─────────────────────────────────
                     2.2.1 direction (INVERSE / ITERATIVE)
```

The shared physics core — Snell's Law — is identical in both:
- In 2.2.2: expressed as direction-cosine vector Snell's Law
- In 2.2.1: expressed as angle-form Snell's Law

The **Look-Up Table (LUT)** is the structural bridge:
the LUT is *populated* using the 2.2.2 forward physics (fast, analytical),
then *used* to accelerate the 2.2.1 inverse iteration (replacing the loop
with a single interpolation lookup).

### 3.4 Non-Modelled Effects

Maas (1996, Section 2.2.3) identifies six sources of systematic error
that the mathematical model does **not** capture:

| Effect | Impact |
|--------|--------|
| Optical aberrations (lenses designed for air) | Degraded image quality with convergent cameras |
| Dispersion in water (1.4% index variation across visible spectrum vs 0.008% in air) | Colour fringing; blur in B&W images |
| Refractive index inhomogeneities (temperature/pressure gradients) | Deviations from flat-interface model |
| Non-planar glass walls | Falsified incidence angles |
| Reduced ray intersection angle (rays bent toward denser medium) | Z-coordinate errors ≈ 2× worse than X/Y |
| Diffusion and absorption in water | Reduced image contrast |

These effects set the practical accuracy limit of the method regardless
of algorithmic refinement.

---

## 4. Relationship to OpenPTV / PyPTV

The repositories at [alexlib/pyptv](https://github.com/alexlib/pyptv) and
[OpenPTV/openptv](https://github.com/OpenPTV/openptv) implement the same
multimedia geometry in C (`liboptv/src/ray_tracing.c`) with Python bindings
via Cython.

| Feature | OpenPTV / PyPTV | This module |
|---------|-----------------|-------------|
| Primary use case | Particle tracking | Multi-camera self-calibration |
| Algorithm | Section 2.2.2 (direct ray tracing) | Section 2.2.1 (radial shift, calibration model) |
| Multimedia detection | `if n1 != 1 or n2[0] != 1 or n3 != 1` → init mmLUT | `MultimediaParams.is_air_only()` |
| LUT | `init_mmLUT()` per camera | `MultimediaLUT.build()` per camera |
| Language | ANSI C + Cython | Pure Python + NumPy |

The C struct `mm_np` in liboptv maps directly to `MultimediaParams` here:
```c
// liboptv (C)           →   multimedia_params.py (Python)
mm_np.n1                 →   MultimediaParams.n1
mm_np.n2[3]              →   MultimediaParams.n2  (list)
mm_np.d[3]               →   MultimediaParams.d   (list, mm)
mm_np.n3                 →   MultimediaParams.n3
mm_np.nlay               →   MultimediaParams.nlay (property)
```

---

## 5. File Architecture

```
multimedia_params.py       ← data structures (instantiated once, passed everywhere)
       │
       ▼
snell_core.py              ← shared physics (Snell's Law in angle + vector form)
       │
   ┌───┴───────────────────┐
   ▼                       ▼
Forward_ray_tracing.py    Iterative_radial_shift .py
(Section 2.2.2)       (Section 2.2.1)
Forward ray trace     Radial shift + collinearity
Particle tracking     Calibration functional model
       │                       │
       └───────────┬───────────┘
                   ▼
           multimedia_lut.py
           Built using 2.2.2 physics
           Accelerates 2.2.1 iterations
```

### Dependency summary

| File | Imports | Used by pipeline at |
|------|---------|---------------------|
| `multimedia_params.py` | numpy only | Steps 1, 2, 3.3, 3.10 |
| `snell_core.py` | numpy only | (internal — never called directly) |
| `Forward_ray_tracing.py` | snell_core, multimedia_params | Step 3.6 |
| `Iterative_radial_shift .py` | snell_core, multimedia_params | Steps 3.2, 3.3, 3.8, 3.9 |
| `multimedia_lut.py` | snell_core, multimedia_params, scipy | Steps 3.7 (build), 3.8, 3.9 (use) |

---

## 6. Module Reference

### 6.1 `multimedia_params.py`

Data structures. Instantiated once at startup and passed through the
entire pipeline.

#### `MultimediaParams`

```python
from multimedia_params import MultimediaParams

mm = MultimediaParams(
    n1 = 1.000293,    # air (camera side)
    n2 = [1.5],       # glass layer(s)
    d  = [5.0],       # glass thickness in mm
    n3 = 1.333,       # water (fluid side)
)

mm.nlay          # → 1  (number of glass layers)
mm.t_total       # → 5.0 mm
mm.is_air_only() # → False (multimedia correction needed)
```

For multi-layer glass (e.g. two panes):
```python
mm = MultimediaParams(n2=[1.5, 1.5], d=[4.0, 4.0])
```

#### `CameraParams`

```python
from multimedia_params import CameraParams
import numpy as np

cam = CameraParams(
    cam_id   = 0,
    position = np.array([0.0, 0.0, 300.0]),   # mm
    angles   = np.deg2rad([0.0, 15.0, 0.0]),  # ω, φ, κ in radians
    c        = 10.0,    # principal distance (mm)
    xh       = 0.0,     # principal point x (mm)
    yh       = 0.0,     # principal point y (mm)
    k1=0.0, k2=0.0, k3=0.0,   # radial distortion
    p1=0.0, p2=0.0,            # decentering distortion
    a1=0.0, a2=0.0,            # A/D scale + shear (Maas eq. 1)
    mm = mm,
)

cam.rotation_matrix   # → 3×3 ndarray
cam.nadir_point       # → [X₀, Y₀, 0]
```

---

### 6.2 `snell_core.py`

Physics primitives. Not called directly by the pipeline — used internally
by the other three modules. Import only if writing your own physics tests.

```python
from snell_core import snell_angle, snell_direction_cosines

# Angle form (used by 2.2.1)
beta2 = snell_angle(beta1, n_in=1.0, n_out=1.5)   # radians

# Vector form (used by 2.2.2)
d_refracted = snell_direction_cosines(
    direction = np.array([0.34, 0.0, -0.94]),
    normal    = np.array([0.0,  0.0,  1.0]),
    n_in=1.0, n_out=1.5
)
```

---

### 6.3 `Forward_ray_tracing.py`

Section 2.2.2 — forward ray tracing for particle positioning and LUT building.

```python
from Forward_ray_tracing import locate_particle_222, trace_ray_through_media

# Full pipeline: image coords → 3D particle position
position, residual_mm = locate_particle_222(
    cameras      = [cam_left, cam_right, cam_top],
    image_points = [(x1, y1), (x2, y2), (x3, y3)],  # corrected, mm
    z_glass_top  = 0.0,
)

# Single camera ray trace (for manual use or LUT building)
pos_water, dir_water = trace_ray_through_media(
    cam         = cam,
    x_img       = 0.5,   # corrected image coordinate, mm
    y_img       = 0.3,
    z_glass_top = 0.0,
)
```

---

### 6.4 `Iterative_radial_shift .py`

Section 2.2.1 — the calibration functional model.
This is the core of your bundle adjustment.

```python
from Iterative_radial_shift  import (
    multimedia_radial_shift,
    observation_equation_221,
    numerical_jacobian_221,
    apply_lens_distortion,
    apply_ad_correction,
)

# --- Distortion correction (Step 3.2) ---
x_corr, y_corr = apply_lens_distortion(x_raw, y_raw, cam)
x_corr, y_corr = apply_ad_correction(x_corr, y_corr, cam)

# --- Radial shift alone (if needed directly) ---
P_bar, n_iter, dR = multimedia_radial_shift(
    particle = np.array([30.0, 15.0, -55.0]),
    cam      = cam,
    eps      = 1e-6,    # convergence: |dR| < 0.001 mm
    ocf      = 1.3,     # over-compensation factor (Maas: 1.1–1.8)
)

# --- Full observation equation: residuals for bundle adjustment ---
vx, vy, P_bar = observation_equation_221(
    particle   = np.array([30.0, 15.0, -55.0]),
    cam        = cam,
    x_observed = 0.89,   # corrected image coordinate, mm
    y_observed = 0.44,
)
# vx, vy are the image residuals (observed − predicted)

# --- Jacobian: J[2 × 16] for normal equations ---
J = numerical_jacobian_221(
    particle   = np.array([30.0, 15.0, -55.0]),
    cam        = cam,
    x_observed = 0.89,
    y_observed = 0.44,
)
# J[0, :] = ∂(vx)/∂(X₀, Y₀, Z₀, ω, φ, κ, c, xₕ, yₕ, k₁..k₃, p₁,p₂, a₁,a₂)
# J[1, :] = ∂(vy)/∂(same 16 params)
```

#### Bundle adjustment sketch

```python
# In your Levenberg-Marquardt loop:
residuals = []
jacobians = []

for point, cam, (x_obs, y_obs) in all_observations:
    vx, vy, _ = observation_equation_221(point, cam, x_obs, y_obs)
    J         = numerical_jacobian_221(point, cam, x_obs, y_obs)
    residuals.append([vx, vy])
    jacobians.append(J)

v = np.array(residuals).flatten()
J = np.vstack(jacobians)
# → solve: (JᵀJ + λI)·Δp = Jᵀv  (Levenberg-Marquardt)
```

---

### 6.5 `multimedia_lut.py`

Per-camera lookup table that accelerates the 2.2.1 iteration.
Build once after Step 3.7 (when camera positions are known).

```python
from multimedia_lut import MultimediaLUT, MultiCameraLUT

# Single camera
lut = MultimediaLUT(cam, z_glass_top=0.0)
lut.build(
    R_max  = 300.0,   # max radial distance from nadir (mm)
    Zp_max = 200.0,   # max particle depth in water (mm)
    n_R    = 100,     # grid resolution
    n_Zp   = 100,
)

# Fast lookup instead of iterating:
P_bar = lut.shift_point(particle, cam)

# Multi-camera system (recommended)
mc_lut = MultiCameraLUT(cameras, z_glass_top=0.0)
mc_lut.build_all(R_max=300.0, Zp_max=200.0)

P_bar = mc_lut[cam_id].shift_point(particle, cam)

# Diagnostics
print(f"Max ΔR in this camera's volume: {lut.max_shift_mm():.2f} mm")
lut.plot()   # requires matplotlib
```

LUT accuracy vs. iterative:

| Grid size | Typical error | Build time |
|-----------|--------------|------------|
| 50 × 50   | ~0.0005 mm   | < 1 s      |
| 100 × 100 | ~0.0001 mm   | ~2 s       |
| 200 × 200 | ~0.00003 mm  | ~8 s       |

---

## 7. Pipeline Integration

The following maps each module to the exact step in Svoboda's
multi-camera self-calibration pipeline.

```
Step 1  Load Config
        └─ multimedia_params.py
           · MultimediaParams(n1, n2, d, n3)  — add to .cfg file
           · CameraParams(...)                 — one per camera

Step 2  Radial Params
        └─ multimedia_params.py
           · CameraParams.k1, k2, k3, p1, p2  — distortion coefficients
           · CameraParams.a1, a2              — A/D affine params

Step 3.1  Load Data
          └─ (no multimedia call)

Step 3.2  Undistortion
          └─ Iterative_radial_shift .py
             · apply_lens_distortion(x, y, cam)
             · apply_ad_correction(x, y, cam)
             These implement Maas eq.(1): the a₁ scale + a₂ shear correction

Step 3.3  Principal Point Offset
          └─ multimedia_params.py
             · CameraParams.xh, .yh — used inside collinearity_condition()

Step 3.4  RANSAC
          └─ (no multimedia call — operates on already-corrected 2D points)

Step 3.5  Pack Tracks
          └─ (no multimedia call)

Step 3.6  Projective SFM
          └─ Forward_ray_tracing.py          ← REPLACE standard triangulation
             · trace_ray_through_media(cam, x_img, y_img, z_glass_top)
             · locate_particle_222(cameras, image_points, z_glass_top)
             Without this, your initial 3D points are wrong from the start.

Step 3.7  Euclidean Upgrade
          └─ multimedia_lut.py           ← BUILD THE LUT HERE (once)
             · MultiCameraLUT(cameras, z_glass_top)
             · .build_all(R_max, Zp_max, n_R, n_Zp)
             Camera positions are now known → best moment to build.
             The LUT persists across all subsequent iterations.

Step 3.8  Reprojection Error
          └─ Iterative_radial_shift .py  +  multimedia_lut.py
             · observation_equation_221(particle, cam, x_obs, y_obs)
               → vx, vy = the multimedia-corrected reprojection residuals
             · lut.shift_point(particle, cam)  ← fast path

Step 3.9  Bundle Adjustment
          └─ Iterative_radial_shift .py  +  multimedia_lut.py  ← DEEPEST CHANGE
             · observation_equation_221(...)  → residual vector
             · numerical_jacobian_221(...)    → J[2×16] normal equations
             · lut.shift_point(...)           → speeds up each Jacobian eval
             Iterative_radial_shift  IS the functional model for the entire BA.

Step 3.10  Save Results
           └─ multimedia_params.py
              · Save MultimediaParams alongside .ori files
              · Save all 16 CameraParams per camera
              The multimedia params are part of the calibration result.

Step 3.11  Check Convergence
           └─ If not converged → loop back to Step 2
              · LUT does NOT need rebuilding (cameras haven't moved much)
              · Rebuild LUT only if camera positions changed significantly
```

---

## 8. Quick-Start Usage

### Minimum working example

```python
import numpy as np
from multimedia_params import MultimediaParams, CameraParams
from Iterative_radial_shift      import observation_equation_221, numerical_jacobian_221
from multimedia_lut     import MultiCameraLUT

# 1. Define the media stack
mm = MultimediaParams(
    n1 = 1.0,     # air
    n2 = [1.5],   # glass
    d  = [5.0],   # 5 mm thick
    n3 = 1.333,   # water
)

# 2. Define cameras (glass top at z = 0)
cameras = [
    CameraParams(cam_id=0, position=np.array([-150, 0, 300]),
                 angles=np.deg2rad([0, 12, 0]), c=10.0, mm=mm),
    CameraParams(cam_id=1, position=np.array([150, 0, 300]),
                 angles=np.deg2rad([0, -12, 0]), c=10.0, mm=mm),
    CameraParams(cam_id=2, position=np.array([0, 150, 300]),
                 angles=np.deg2rad([-12, 0, 0]), c=10.0, mm=mm),
    CameraParams(cam_id=3, position=np.array([0, -150, 300]),
                 angles=np.deg2rad([12, 0, 0]), c=10.0, mm=mm),
]

# 3. Build LUT (once, after Step 3.7)
lut = MultiCameraLUT(cameras, z_glass_top=0.0)
lut.build_all(R_max=300.0, Zp_max=200.0, n_R=100, n_Zp=100)

# 4. In bundle adjustment — one observation:
particle    = np.array([10.0, -5.0, -40.0])   # current 3D estimate
x_obs, y_obs = 0.32, -0.15                    # corrected image coords (mm)

vx, vy, P_bar = observation_equation_221(particle, cameras[0], x_obs, y_obs)
J             = numerical_jacobian_221(particle, cameras[0], x_obs, y_obs)

print(f"Residuals: vx={vx:.4f} mm, vy={vy:.4f} mm")
print(f"Jacobian shape: {J.shape}")   # (2, 16)
```

### No multimedia (air-only experiment)

```python
mm_air = MultimediaParams(n1=1.0, n2=[1.0], d=[0.0], n3=1.0)
# mm_air.is_air_only() → True
# All modules still work — refraction just has no effect (ΔR = 0)
```

### Checking the multimedia correction magnitude

```python
lut = MultimediaLUT(cameras[0], z_glass_top=0.0)
lut.build(R_max=200.0, Zp_max=150.0)
print(f"Max radial shift ΔR: {lut.max_shift_mm():.2f} mm")
lut.plot()   # contour map of ΔR over the (R, Zp) grid
```

---

## 9. Key Equations Reference

| Equation | Description | Location |
|----------|-------------|----------|
| Maas (1) | A/D affine transformation | `apply_ad_correction()` |
| Maas (2) | Radial geometry constraint | `multimedia_radial_shift()` |
| Maas (3) | Snell's Law (angle form) | `snell_all_angles()` |
| Maas (4) | Initial R̄ estimate | `multimedia_radial_shift()` |
| Maas (5) | β₁ from current R̄ | `multimedia_radial_shift()` |
| Maas (6) | β₂, β₃ via Snell | `snell_all_angles()` |
| Maas (7) | dR correction | `multimedia_radial_shift()` |
| Maas (8) | Cartesian shift → P̄ | `multimedia_radial_shift()` |
| Maas (9) | Direction cosines in air | `direction_cosines_in_air()` |
| Maas (10) | Ray/air-glass intersection | `trace_ray_through_media()` |
| Maas (11) | Refraction into glass | `snell_direction_cosines()` |
| Maas (12) | Ray/glass-water intersection | `trace_ray_through_media()` |
| Maas (13) | Refraction into water | `snell_direction_cosines()` |
| Maas (14–16) | Spatial intersection | `spatial_intersection_multi()` |
| Maas (17) | Full collinearity condition | `collinearity_condition()` |

---

## 10. Dependencies

```
numpy   >= 1.23     (array maths, core dependency)
scipy   >= 1.9      (RegularGridInterpolator in multimedia_lut.py)
```

Optional (for plotting):
```
matplotlib          (MultimediaLUT.plot())
```

Install:
```bash
pip install numpy scipy matplotlib
```

All five modules are pure Python — no C extensions, no Cython, no
dependency on liboptv or OpenPTV.

---

## 11. References

**Primary reference — the mathematical foundation of this entire module:**

> Maas, H.-G. (1996). *Contributions of Digital Photogrammetry to 3-D PTV*.
> In: Th. Dracos (ed.), Three-Dimensional Velocity and Vorticity Measuring
> and Image Analysis Techniques, Kluwer Academic Publishers, pp. 191–207.

**Related OpenPTV / PyPTV repositories:**

> alexlib/pyptv — Python GUI for the OpenPTV library, liboptv.
> https://github.com/alexlib/pyptv

> OpenPTV/openptv — Core C library (liboptv) with Python bindings.
> https://github.com/OpenPTV/openptv

**General multimedia photogrammetry (referenced in Maas 1996):**

> Kotowski, R. (1988). Phototriangulation in Multi-Media-Photogrammetry.
> International Archives of Photogrammetry and Remote Sensing, Vol. XXVII.
> (Generalisation to arbitrary number of non-parallel interfaces.)

> Okamoto, A. & Höhle, J. (1972). Allgemeines analytisches
> Orientierungsverfahren in der Zwei- und Mehrmedien-Photogrammetrie.
> Bildmessung und Luftbildwesen, 2/1972, 3/1972.

**Snell's Law and refractive indices:**

- Standard refractive index of water at 20 °C, 589 nm: **n = 1.333**
- Standard refractive index of borosilicate glass: **n = 1.5**
- Refractive index of air at standard conditions: **n = 1.000293**
- Dispersion in water across visible spectrum: ~1.4%
  (vs ~0.008% in air — Hoehle 1971, cited in Maas 1996)
