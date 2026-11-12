# bundle_PX_proj.py — Multimedia Geometry Changes Explanation

## Context: What This File Does and Where Step 3.9 Lives

`bundle_PX_proj` performs Levenberg-Marquardt bundle adjustment. Its internal
pipeline is:

```
Input: P0 (3K×4 projective cameras), X0 (4×N 3D points), q (2K×N observations)
  ↓
vgg_conditioner_from_image → conditioning matrices H[k]
  ↓
P0_cond = H[k] @ P0[k]       condition cameras
q_cond  = H[k] @ q[k]        condition observations
P0_cond = normP(P0_cond)      normalize cameras
X0_cond = normx(X0)           normalize 3D points
  ↓
Build observation vector y and visibility mask qivis
Build tangent spaces TX (for X) and TP (for P)
  ↓
levmarq(objective, p0, opt)   Levenberg-Marquardt loop
  objective(p) → eval_y_and_dy(p, P0_cond, TP, X0_cond, TX, y, qivis, RADIAL)
  ↓
Extract P and X from optimized p, undo conditioning on P
  ↓
Output: P (3K×4), X (4×N)
```

---

## The Architectural Constraint

The residuals and Jacobian are computed entirely inside `eval_y_and_dy`, which
is not provided in this repository. `eval_y_and_dy` receives the LM parameter
vector `p`, the conditioned/normalized projective P and X, and the observation
vector `y` — all in a single conditioned projective coordinate system.

`cameras_mm` (from `_build_camera_params` in GoCal) lives in **metric world
coordinates**. These two coordinate systems cannot be mixed directly. Fully
replacing the projection model inside the LM loop requires modifying
`eval_y_and_dy` to call `observation_equation_221` and `numerical_jacobian_221`
from `Multimedia.multimedia_221`.

---

## What Was Done in This Session

Given the architectural constraint, four concrete, correct changes are made:

### Change 1 — Import Alignment

```python
from Multimedia.ray_tracing_222 import trace_ray_through_media, spatial_intersection_multi
```

Direct import matching the `Multimedia/` package structure used throughout
the project. No `try/except` guard — consistent with project convention.

---

### Change 2 — Helper Function `_mm_refine_X_from_q`

A private function added before `bundle_PX_proj`.

**Purpose:** After the LM converges, use the raw 2D observations `q` and
metric camera positions `cameras_mm` to compute better 3D point positions
through multimedia spatial intersection. This replaces the BA's projective
estimate of X with metric world coordinates for successfully intersected points.

**Why `q` and not the conditioned `y`:** `q` (2K × N Euclidean) is in pixel
space with the principal point subtracted — the same coordinate system as
`cameras_mm[k].c`, `.xh`, `.yh`. The conditioned `q_cond` and observation
vector `y` are in the normalized conditioning space and cannot be used directly
with `trace_ray_through_media`.

**Algorithm per point `n`:**

```
for each camera k in range(K):
    x_obs = q[2k, n],  y_obs = q[2k+1, n]
    if NaN → skip
    pos_w, dir_w = trace_ray_through_media(cameras_mm[k], x_obs, y_obs, z_glass_top)

if len(ray_positions) >= 2:
    P_hat, rms = spatial_intersection_multi(ray_positions, ray_directions)
    X_ref[:3, n] = P_hat    ← metric world coordinates
    X_ref[3,  n] = 1.0
```

**Returns:** refined X copy, count of refined points, mean RMS ray distance (mm).

---

### Change 3 — Function Signature Extension

```python
# Before
def bundle_PX_proj(P0, X0, q, imsize, nl_params_all_cams=None, opt=None)

# After
def bundle_PX_proj(P0, X0, q, imsize, nl_params_all_cams=None, opt=None,
                   cameras_mm=None, lut=None, z_glass_top=0.0)
```

All new parameters default to their "disabled" values — every existing call
site continues to work unchanged.

| Parameter | Purpose |
|---|---|
| `cameras_mm` | Metric camera objects for post-BA multimedia refinement. `None` on first iteration. |
| `lut` | Stored in opt for future `eval_y_and_dy` integration. Not yet used in LM residuals. |
| `z_glass_top` | Z-coordinate of air/glass interface (mm). Passed to `trace_ray_through_media`. |

---

### Change 4 — Store Multimedia Params in opt

```python
opt['cameras_mm']  = cameras_mm
opt['lut']         = lut
opt['z_glass_top'] = z_glass_top
```

Stored at the top of `bundle_PX_proj`, immediately after option defaults are
set, so they flow into every downstream call — including the `objective` closure
that calls `eval_y_and_dy`. When `eval_y_and_dy.py` is provided and modified
to call `observation_equation_221`, it will read `cameras_mm` and `lut` from
the `opt` dict it already receives. **Zero changes will be needed here.**

---

### Change 5 — Post-BA Multimedia Spatial Intersection (Step 3.9)

Inserted immediately before `return P, X`:

```python
if cameras_mm is not None:
    X, n_ref, mean_rms = _mm_refine_X_from_q(
        q, X, K, N, cameras_mm, z_glass_top)
    if opt.get('verbose', 0):
        print(f'Multimedia post-BA refinement: {n_ref}/{N} points refined, '
              f'mean RMS={mean_rms:.4f} mm')
```

**What X contains after this step:**

| Point | X[:,n] contents |
|---|---|
| Successfully refined | `[Xw, Yw, Zw, 1]` — metric world coordinates |
| Not refined (< 2 rays) | Original BA estimate (projective/normalized space) |

This mixed X then flows through `fill_mm_bundle` → `GoCal` →
`euclidize(Rmat, Lambda, P, X, config)`. `euclidize` uses both P and X to
recover metric structure. For the multimedia-refined points (where X is already
in world coordinates), `euclidize` receives a better starting structure,
progressively improving the metric reconstruction over global iterations.

---

## What Was Not Changed

| Component | Reason |
|---|---|
| `eval_y_and_dy` call | Core residual computation — requires `eval_y_and_dy.py` source |
| `levmarq` loop | Unchanged — standard Levenberg-Marquardt |
| `vgg_conditioner_from_image` | Unchanged — standard image conditioning |
| P extraction and unconditiong | Unchanged — standard projective recovery |
| X extraction from tangent space | Unchanged — standard recovery |

---

## The Iteration Pattern

```
Iteration 0
  cameras_mm = None
  bundle_PX_proj(..., cameras_mm=None)
    → standard BA, no post-BA refinement
    → X in conditioned projective space
  euclidize(P, X) → metric Xe, Ce, Re
  build cameras_mm from Ce, Re

Iteration 1+
  cameras_mm available from previous euclidize()
  bundle_PX_proj(..., cameras_mm=cameras_mm)
    → standard BA (unchanged)
    → post-BA: _mm_refine_X_from_q replaces X with metric world coords
    → X now mixed (world coords + remaining projective)
  euclidize(P, X_refined) → better metric Xe
  rebuild cameras_mm → improved for next iteration
```

---

## What Remains for Full Step 3.9

The complete multimedia BA integration requires modifying `eval_y_and_dy.py`
to use the multimedia functional model. Specifically:

```python
# Inside eval_y_and_dy, the standard projection:
#   residual = q_observed - π_pinhole(P, X)
# Should become:
#   residual = q_observed - collinearity(shift_point(X_world, cam), cam)
# And the Jacobian:
#   J = numerical_jacobian_221(X_world, cam, q_obs_x, q_obs_y)
```

The infrastructure is ready:
- `cameras_mm` and `lut` are in `opt` — accessible to `eval_y_and_dy`
- The Multimedia package is imported
- The signature and opt dict are fully wired

When `eval_y_and_dy.py` is provided, the change needed there is:
1. Read `cameras_mm = opt.get('cameras_mm')` and `lut = opt.get('lut')`
2. If both are not None, replace `π_pinhole(P, X)` with the multimedia model
3. Replace the analytical Jacobian with `numerical_jacobian_221`

No changes to `bundle_PX_proj.py` will be needed at that point.

---

## Coordinate System Note

`q[2k, n]` and `q[2k+1, n]` are in pixel space with the principal point
subtracted (done by GoCal before calling `fill_mm_bundle`). This matches the
coordinate system expected by `trace_ray_through_media`: image coordinates
centered at the principal point, consistent with `cameras_mm[k].c` (focal
length in pixels) and `cameras_mm[k].xh/yh` (principal point offsets).

---

## References

- Maas, H.-G. (1996). *Contributions of Digital Photogrammetry to 3-D PTV.*
  Section 2.2.1: observation equation for calibration (eqs. 4–8, 17).
  Section 2.2.2: direct spatial intersection (eqs. 9–16).
- See also: `fill_mm_bundle_EXPLANATION.md`, `GoCal_EXPLANATION.md`,
  `EstimateReprojectionError_EXPLANATION.md`
