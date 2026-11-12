# fill_mm_bundle.py — Multimedia Geometry Changes Explanation

## Context: Why This File and Why Step 3.6 Is Here

The integration map (Step 3.6) requires replacing standard DLT triangulation
with multimedia ray intersection so that the initial 3D structure already
accounts for refraction before `euclidize()` sees it.

`fill_mm_bundle` wraps two sub-steps:
1. `fill_mm` — projective factorisation (rank-4 SVD)
2. `bundle_PX_proj` — nonlinear refinement

`fill_mm` works entirely in **projective space**: it factorises the
measurement matrix without any knowledge of real-world camera positions.
Multimedia ray tracing (`locate_particle_222`) requires **metric** camera
positions, which only exist after `euclidize()` is called in `GoCal.py`.

`fill_mm` therefore **cannot** be replaced. Instead, the multimedia correction
is applied as a **post-processing refinement of X** between `fill_mm` and
`bundle_PX_proj`, using metric camera positions from the *previous* global
iteration.

---

## The Iteration Pattern

```
Iteration 0
  cameras_mm = None
  fill_mm_bundle(M, imsize, opt,
                 cameras_mm=None)    ← standard projective X
  euclidize(P, X)                    ← get Ce, Re
  build cameras_mm from Ce, Re       ← now have metric positions

Iteration 1+
  fill_mm_bundle(M, imsize, opt,
                 cameras_mm=cameras_mm)   ← multimedia X refinement applied
  euclidize(P, X_refined)           ← Euclidean upgrade from better X
  rebuild cameras_mm from new Ce, Re
```

On the first call, `cameras_mm=None` and the function behaves identically to
the original. From the second iteration onward, X is refined before being
passed to `euclidize()`, progressively correcting the 3D structure for
refraction.

---

## Changes Made

### Change 1 — Module Import (top of file)

```python
from Multimedia.ray_tracing_222 import trace_ray_through_media, spatial_intersection_multi
```

Direct import matching the `Multimedia/` package structure used throughout the
project (same convention as `GoCal.py` imports from `Multimedia.Multimedia_lut`
and `Multimedia.Multimedia_params`). No `try/except` guard needed since the
`Multimedia/` package is a project requirement.

---

### Change 2 — Helper Function `_multimedia_refine_X`

A new private function added **before** `fill_mm_bundle` in the file.

**Purpose:** For each recovered 3D point, collect refracted rays from every
camera that observes it and intersect them in the water volume to replace
the projective estimate from `fill_mm`.

**Algorithm per point `j` in `r2` (recovered points):**

```
for each camera i in r1 (recovered cameras):
    obs = M[3i:3i+3, j]              read 2D observation (homogeneous)
    if NaN → skip this camera
    x_img = obs[0] / obs[2]          normalise to metric 2D
    y_img = obs[1] / obs[2]
    pos_w, dir_w = trace_ray_through_media(cameras_mm[i], x_img, y_img, z_glass_top)
                                      forward ray trace: air → glass → water

intersect rays → spatial_intersection_multi(ray_positions, ray_directions)
               → P_hat (3D position), rms (ray-distance quality)

X[:3, j] = P_hat                     replace projective estimate
X[3,  j] = 1.0                       homogeneous coordinate
```

**Guard conditions:**
- `cameras_mm is not None` — metric positions available from previous euclidize()
- `len(ray_positions) >= 2` — need at least 2 rays to intersect
- Per-ray `try/except` — skips rays that cause total internal reflection

**Returns:**
- `X_ref` — refined copy of X
- `n_refined` — number of points successfully replaced
- `mean_rms` — mean RMS ray distance in mm (quality indicator)

---

### Change 3 — Function Signature Extension

```python
# Before
def fill_mm_bundle(M, imsize, opt=None)

# After
def fill_mm_bundle(M, imsize, opt=None,
                   cameras_mm=None,
                   lut=None,
                   z_glass_top=0.0)
```

| Parameter | Type | Purpose |
|---|---|---|
| `cameras_mm` | `list[CameraParams]` or `None` | Metric camera objects from previous iteration. `None` on first call → no refinement. |
| `lut` | `MultiCameraLUT` or `None` | Reserved for Step 3.9 (`bundle_PX_proj` integration). Accepted but not used in this file yet. |
| `z_glass_top` | `float` | Z-coordinate of air/glass interface in mm. Passed to `trace_ray_through_media`. |

All three default to their "disabled" values so every existing call site
continues to work with zero behaviour change.

---

### Change 4 — Multimedia Refinement Step in the Function Body

Inserted between `fill_mm` and the bundle adjustment block:

```python
# Step 2: Multimedia spatial intersection refinement (Maas 2.2.2, Step 3.6)
info['mm_refined']   = False
info['mm_n_refined'] = 0
info['mm_mean_rms']  = 0.0

if cameras_mm is not None and len(u1) < m and len(u2) < n:
    r1 = np.setdiff1d(np.arange(m), u1)
    r2 = np.setdiff1d(np.arange(n), u2)
    X, n_ref, mean_rms = _multimedia_refine_X(
        M, X, r1, r2, cameras_mm, z_glass_top)
    info['mm_refined']   = True
    info['mm_n_refined'] = n_ref
    info['mm_mean_rms']  = mean_rms
```

Three new fields in `info` (always present, safe to read):

| Field | Type | Meaning |
|---|---|---|
| `info['mm_refined']` | bool | True if multimedia refinement ran |
| `info['mm_n_refined']` | int | Number of points replaced by multimedia intersection |
| `info['mm_mean_rms']` | float | Mean RMS ray distance in mm — lower is better |

---

## GoCal.py Updates (required alongside this file)

Both `fill_mm_bundle` calls in `GoCal.py` now pass the new parameters:

```python
# Inner outlier loop
P, X, u1, u2, info = fill_mm_bundle(  # (2,CAMS) to match MATLAB pp(:,1:2)'
    linear.Ws[:, inliers['idx']],
    config.principal_point[:, :2].T,
    options,
    cameras_mm=cameras_mm,
    z_glass_top=config.z_glass_top)

# Final bundle adjustment
P, X, u1, u2, info = fill_mm_bundle(  # (2,CAMS) to match MATLAB pp(:,1:2)'
    linear.Ws[:, inliers['idx']],
    config.principal_point[:, :2].T,
    options,
    cameras_mm=cameras_mm,
    z_glass_top=config.z_glass_top)
```

`cameras_mm` is `None` on the first outer iteration and populated from the
previous `euclidize()` call on every subsequent one.

---

## Indexing: cameras_mm vs M Row Triples

`M` has shape `(3m × n)` where rows `3i, 3i+1, 3i+2` belong to camera `i`.
`cameras_mm` is built in `GoCal._build_camera_params()` with the same ordering:
`cameras_mm[i]` corresponds to rows `3i:3i+3` in `linear.Ws`.

Since `fill_mm_bundle` receives `linear.Ws[:, inliers['idx']]`, the column
count changes but the row order is preserved. `cameras_mm[i]` therefore
always maps correctly to `M[3i:3i+3, :]`.

---

## What Was Not Changed

| Component | Reason |
|---|---|
| `fill_mm` | Works in projective space — cannot use metric multimedia geometry |
| `bundle_PX_proj` call | Unchanged — Step 3.9 requires `bundle_PX_proj.py` to be uploaded |
| Error reporting (`info['err']`) | Unchanged — BA error still computed from projective residual |
| `u1`, `u2` | Unchanged — unrecovered camera/point lists come from `fill_mm` |

---

## Backward Compatibility

When `cameras_mm=None` (first iteration or air-only experiments):
- `_multimedia_refine_X` is never called
- `info['mm_refined']` = False
- X is identical to the projective output of `fill_mm`
- The function returns identically to the original

---

## What Remains for Step 3.9

`lut` is accepted in the signature and stored, but `bundle_PX_proj` is not
yet modified. Once `bundle_PX_proj.py` is uploaded, the multimedia functional
model (`observation_equation_221`, `numerical_jacobian_221`) will replace its
internal projection — at that point `lut` will be forwarded into it.

---

## References

- Maas, H.-G. (1996). *Contributions of Digital Photogrammetry to 3-D PTV.*
  Section 2.2.2: Direct solution for the spatial intersection (eqs. 9–16).
- See also: `GoCal_EXPLANATION.md`, `EstimateReprojectionError_EXPLANATION.md`
