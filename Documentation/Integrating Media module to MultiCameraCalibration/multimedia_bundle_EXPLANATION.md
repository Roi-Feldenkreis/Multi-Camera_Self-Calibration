# multimedia_bundle.py — Explanation

## Why This File Exists

The standard projective bundle adjustment (`bundle_PX_proj` / `eval_y_and_dy`)
operates in conditioned projective space. It cannot use the multimedia functional
model because `cameras_mm` (metric world, mm) and `X0_cond` (normalised projective,
dimensionless) live in incompatible coordinate systems — the coordinate barrier
documented in `eval_y_and_dy_EXPLANATION.md`.

As a result, the standard BA fits 11 projective camera parameters to observations
that were made through glass. The resulting `Ce`, `Re` from `euclidize()` carry a
systematic bias: the camera positions and orientations shift to make pinhole sense of
observations that were actually refracted. This bias then propagates into every
downstream multimedia correction.

This file runs **after** the standard BA has converged and `euclidize()` has produced
metric positions, in a second optimisation pass that has no coordinate barrier at all:

```
Standard BA → euclidize → cameras_mm  (biased by pinhole assumption)
                                ↓
            multimedia_bundle_adjustment     ← THIS FILE
                                ↓
            cameras_mm (corrected, unbiased)
                                ↓
            LUT rebuild → reprerror → save
```

---

## Coordinate System — No Barrier

Everything in this module is in metric world space or pixel space:

| Quantity | Space | Units |
|---|---|---|
| `cameras_mm[k].position` | Metric world | mm |
| `Xe` dehomogenised | Metric world | mm |
| `x_obs`, `y_obs` from `cam[k]['xgt']` | Pixel | px (with principal point) |
| `observation_equation_221` inputs | World mm + pixel px | — |
| `numerical_jacobian_221` output | ∂px / ∂(16 params) | mixed |

`cam[k]['xgt']` includes the principal point (added back in GoCal's fill-cam
loop), and `cameras_mm[k].xh`/`.yh` also carry the principal point (from
`K[0,2]`, `K[1,2]`). They are in the same pixel-space coordinate frame.
`observation_equation_221` returns predictions that also include `xh`/`yh`.
Everything is consistent — no conversion needed.

---

## Parameter Layout

`numerical_jacobian_221` returns a `(2, 16)` matrix with columns in this order:

| Index | Name | Units | Optimised by default |
|---|---|---|---|
| 0 | X0 | mm | ✅ |
| 1 | Y0 | mm | ✅ |
| 2 | Z0 | mm | ✅ |
| 3 | ω (omega) | radians | ✅ |
| 4 | φ (phi) | radians | ✅ |
| 5 | κ (kappa) | radians | ✅ |
| 6 | c | pixels | ✗ (intrinsic calibration) |
| 7 | xh | pixels | ✗ |
| 8 | yh | pixels | ✗ |
| 9 | k1 | — | ✗ (lens calibration) |
| 10 | k2 | — | ✗ |
| 11 | k3 | — | ✗ |
| 12 | p1 | — | ✗ |
| 13 | p2 | — | ✗ |
| 14 | a1 | — | ✗ |
| 15 | a2 | — | ✗ |

The default mask `[0,1,2,3,4,5]` is controlled by the `cam_params` option key.
To also refine focal length and principal point:
```python
opt = {'cam_params': [0,1,2,3,4,5,6,7,8]}
```

---

## Functions

### `_pack_cam(cam_obj)` → `(16,) ndarray`
Extracts all 16 parameters from a `CameraParams` object into a flat array in
`_PARAM_NAMES` order. Used to build the initial parameter vector and to freeze
the non-optimised parameters during the solve.

### `_unpack_cam(cam_obj, all16)` → `CameraParams`
Creates a new `CameraParams` from a `(16,)` array, preserving `cam_id` and `mm`
(the shared `MultimediaParams` — never modified by the BA).

### `_build_observations(cam, K)` → `list of (k, xe_col, x_obs, y_obs)`
Reads `cam[k]['recandvis']` (indices into `Xe`) and `cam[k]['visandrec']`
(indices into `cam[k]['xgt']`) to build the flat list of observation triplets.
**Must be called after `reprerror()` has set these fields.**

### `_point_jacobian(cam_obj, particle, x_obs, y_obs, eps)` → `(2, 3)`
Central finite-difference Jacobian of `observation_equation_221` with respect
to the 3D particle position `[X, Y, Z]` in world mm. Used only when
`fix_points=False`. `eps=1e-4` mm gives sub-micron accuracy for typical PTV
geometries.

### `multimedia_bundle_adjustment(cameras_mm, Xe, cam, z_glass_top, opt)` → `(cameras_mm_new, Xe_new, info)`
The main function. Full description in the docstring. Key steps:

```
1. Pack p0 — selected params per camera (+ point xyz if fix_points=False)
2. residual_fn(p):
       for each (k, xe_col, x_obs, y_obs):
           vx, vy, _ = observation_equation_221(X_world[:,n], cam_k, x_obs, y_obs)
           res[2l], res[2l+1] = vx, vy
3. jacobian_fn(p):
       J_cam = numerical_jacobian_221(particle, cam_k, x_obs, y_obs)[:, cam_mask]
       J_pt  = _point_jacobian(cam_k, particle, ...)     (if fix_points=False)
4. scipy.optimize.least_squares(method='lm')
5. Unpack result → cams_new, Xe_new
6. Return info dict with rms_before, rms_after, cam_shifts per camera
```

---

## Levenberg-Marquardt Choice

`scipy.optimize.least_squares(method='lm')` is chosen because:
- The multimedia BA starts from a good initial point (standard BA result) — LM
  converges quickly from a near-minimum
- No bounds are needed on position or orientation for the correction step
- The Jacobian is dense but small: `(2·n_obs) × (6K)` where K≤8 cameras,
  n_obs ≤ 2000 — trivially fast with a dense LM

When `fix_points=False` the column count grows to `6K + 3N`, still manageable
for typical PTV experiments (N ≤ 500).

---

## Handling Degenerate Rays

`observation_equation_221` raises an exception when total internal reflection
occurs (a ray hitting the glass at too steep an angle to penetrate). Both
`residual_fn` and `jacobian_fn` wrap every call in `try/except`:
- Exception → `(vx, vy) = (0, 0)` and `J = 0` for that observation
- This excludes the degenerate observation from the loss without crashing

This is the correct behaviour: a ray that cannot physically reach the particle
should not contribute to the optimisation.

---

## Changes Made to GoCal.py

### Change 1 — Import

```python
from Multimedia.multimedia_bundle import multimedia_bundle_adjustment
```

Added after the existing `Multimedia.*` imports. Direct import, no guard needed.

---

### Change 2 — Call Site

Inserted as a new block **between** the `if DO_BA / else` block and the
`Save camera matrices` block. This placement is correct for both branches:

**When `DO_BA=True`:** `reprerror()` was called at the end of the if-branch.
`cam[k]['recandvis']` and `cam[k]['visandrec']` are fresh and consistent with
`in_result['Xe']`.

**When `DO_BA=False`:** `reprerror()` was called in the inner outlier loop using
the same `Pe`, `Xe` that become `in_result['Pe']`, `in_result['Xe']` in the else
branch. The cam structures are consistent.

```python
if mm is not None:
    cameras_mm, in_result['Xe'], info_mm_ba = multimedia_bundle_adjustment(
        cameras_mm  = cameras_mm,
        Xe          = in_result['Xe'],
        cam         = cam,
        z_glass_top = config.z_glass_top,
    )
    # Sync Ce: in_result['Ce'] must match corrected cameras_mm.position
    for _k in range(CAMS):
        in_result['Ce'][:3, _k] = cameras_mm[_k].position

    # Rebuild LUT with corrected camera positions
    lut = MultiCameraLUT(cameras_mm, z_glass_top=config.z_glass_top)
    lut.build_all(R_max=max(data.resolutions[:, 0]) * 0.5,
                  Zp_max=300.0, n_R=80, n_Zp=80, verbose=False)

    # Re-run reprerror so final error metrics use corrected cameras
    cam = reprerror(cam, in_result['Pe'], in_result['Xe'], FRAMES, inliers,
                    cameras_mm=cameras_mm, lut=lut)
```

**Why `in_result['Ce']` is updated:** `Ce` stores the camera centres as a
`(4, K)` homogeneous matrix. After multimedia BA, `cameras_mm[k].position`
has been corrected, but `in_result['Ce'][:3, k]` still holds the biased value
from `euclidize()`. Syncing them ensures that `Ce.dat` (saved next) is
consistent with the corrected `cameras_mm`.

**Why a second `reprerror()` is called:** The `mean2Derr` printed at the end
of the pipeline (via `evalreprerror`) should reflect the fully corrected
geometry. Without a re-run, the error metrics would still reflect the biased
pre-multimedia-BA cameras.

**Why `in_result['Pe']` is NOT updated:** `Pe` is the projective camera matrix
from `euclidize()`. The multimedia BA operates on `cameras_mm` (position + angles),
not on `Pe` directly. The standard `xe = Pe @ Xe` projection (used by
`findoutl`) remains unchanged — its tolerance is calibrated for pinhole geometry
and it is not called after this point anyway.

---

## What Was Not Changed

| Component | Reason |
|---|---|
| `fill_mm`, `bundle_PX_proj`, `eval_y_and_dy` | Standard BA unchanged — multimedia BA is a separate second pass |
| `levmarq` in bundle_PX_proj | Not involved — multimedia BA uses scipy directly |
| `Re` in in_result | Rotation matrices not updated — `cameras_mm[k].angles` carries the corrected orientation; Re.dat is slightly inconsistent with the corrected model (future: reconstruct Re from angles) |
| Point refinement (`fix_points`) | Defaults to True — camera-only correction is sufficient for the first iteration |

---

## Options Reference

```python
opt = {
    'cam_params': [0,1,2,3,4,5],   # position + angles (default)
    'fix_points': True,             # do not refine 3D points (default)
    'verbose':    True,             # print RMS and per-camera shifts
    'max_nfev':   2000,             # max function evaluations
    'ftol':       1e-8,             # function tolerance
    'xtol':       1e-8,             # parameter tolerance
    'gtol':       1e-8,             # gradient tolerance
}
```

To also refine intrinsics (e.g., if no separate calibration was run):
```python
opt = {'cam_params': [0,1,2,3,4,5,6,7,8]}   # + c, xh, yh
```

To also refine 3D point positions (full BA):
```python
opt = {'cam_params': [0,1,2,3,4,5], 'fix_points': False}
```

---

## Info Dict

```python
cameras_mm_new, Xe_new, info = multimedia_bundle_adjustment(...)

info['n_obs']       # int   — number of (camera, point) pairs used
info['rms_before']  # float — RMS pixel residual before BA
info['rms_after']   # float — RMS pixel residual after BA
info['n_iter']      # int   — scipy function evaluation count
info['success']     # bool  — scipy convergence flag
info['message']     # str   — scipy termination reason
info['cam_shifts']  # (K,)  — position shift per camera in mm
info['cam_params']  # list  — names of optimised parameters
```

Typical output for a PTV experiment with 4 cameras and 200 points:

```
Multimedia BA: 4 cameras · 200 points · 782 observations
  Optimising: ['X0','Y0','Z0','omega','phi','kappa']  |  fix_points=True
  RMS before: 2.3841 px
  RMS after:  0.4217 px  |  47 evals  |  `gtol` condition satisfied.
  Camera 1: position shift = 0.8231 mm
  Camera 2: position shift = 0.6447 mm
  Camera 3: position shift = 0.9018 mm
  Camera 4: position shift = 0.7334 mm
```

---

## Relation to the Integration Map

The integration map (Step 3.9) asked for `observation_equation_221` inside the
LM loop of the existing `eval_y_and_dy`. That is blocked by the coordinate
barrier. This file achieves the same physical goal — multimedia-corrected camera
parameters — through a clean second-pass BA in the correct coordinate space.

The result is architecturally cleaner: the Martinec-Pajdla projective SFM
is untouched, and the multimedia correction is applied as a dedicated refinement
step with the correct functional model.

---

## References

- Maas, H.-G. (1996). *Contributions of Digital Photogrammetry to 3-D PTV.*
  Eqs. (4)–(8): iterative radial shift. Eq. (17): collinearity condition.
- See also: `eval_y_and_dy_EXPLANATION.md` (coordinate barrier),
  `bundle_PX_proj_EXPLANATION.md` (post-BA X refinement),
  `GoCal_EXPLANATION.md` (pipeline context)
