# eval_y_and_dy.py & bundle_PX_proj.py — Step 3.9 Changes Explanation

## What These Files Do

`eval_y_and_dy` is the innermost function of the Levenberg-Marquardt bundle
adjustment. For each LM iteration it computes:
- `y_residual` — predicted minus observed 2D positions for all visible points
- `J` — the sparse analytical Jacobian of the residual w.r.t. all parameters

`bundle_PX_proj` calls `eval_y_and_dy` through a closure inside `levmarq`.

---

## The Coordinate System Barrier

This is the central fact governing every decision in this file. The two
coordinate systems at play are:

```
eval_y_and_dy (projective conditioned space)
────────────────────────────────────────────
P0      = normP( H_k @ P_projective )   conditioned camera
X0      = normx( X_projective )         conditioned 3D point
y       = p2e( H_k @ q_observed )       conditioned observations
X_LM    = X0 + TX[n] @ iX_n            current X during iteration
                                         ← all in conditioned normalized space

cameras_mm (metric world space)
────────────────────────────────
position = Ce[:3, i]                    metric world coords (mm)
c, xh, yh = K[0,0], K[0,2], K[1,2]    pixel units
lut       = built for R in [0,300] mm   metric world radius
```

`observation_equation_221(X_world, cam, x_obs, y_obs)` requires:
- `X_world` in **metric mm** — NOT available during LM (X is conditioned/normalized)
- `x_obs, y_obs` in **pixel space** — available in `q_orig`
- `cam` in **metric world** — available in `cameras_mm[k]`

To go from conditioned LM space to metric world requires:
1. Inverse of `normx()` — not stored anywhere accessible
2. The full projective→metric transform from `euclidize()` — not in scope

**Conclusion:** replacing `x = P @ X` with `observation_equation_221` inside
the LM loop is architecturally impossible without restructuring the BA to work
in metric world space — a larger change beyond the scope of these files.

---

## Where the Multimedia Correction Actually Lives

The full multimedia correction for Step 3.9 is correctly split across three
points that ARE in the right coordinate space:

| Stage | File | Function | What it does |
|---|---|---|---|
| Before BA | `fill_mm_bundle.py` | `_multimedia_refine_X` | Refines X via refracted rays before LM sees it |
| After BA | `bundle_PX_proj.py` | `_mm_refine_X_from_q` | Post-LM X refinement in metric world coords |
| Error metrics | `EstimateReprojectionError.py` | `reprerror` | Multimedia-corrected reprojection error |

The LM itself optimizes in conditioned projective space (correct for the
Martinec-Pajdla framework). The multimedia geometry is applied around it.

---

## Changes Made to eval_y_and_dy.py

### Change 1 — Module Imports

```python
from Multimedia.multimedia_221 import collinearity_condition
from Multimedia.multimedia_221 import multimedia_radial_shift
```

Direct imports matching the `Multimedia/` package structure. These are
imported now so they are available when the metric-space residual path is
activated in a future restructuring. No try/except — consistent with project
convention.

---

### Change 2 — Signature Extension

```python
# Before
def eval_y_and_dy(p, P0, TP, X0, TX, y, qivis, RADIAL=False)

# After
def eval_y_and_dy(p, P0, TP, X0, TX, y, qivis, RADIAL=False,
                  cameras_mm=None, lut=None, z_glass_top=0.0, q_orig=None)
```

Four new parameters, all with safe defaults. Every existing call site
continues to work unchanged.

| Parameter | Type | Purpose |
|---|---|---|
| `cameras_mm` | `list[CameraParams]` or None | Metric camera objects from previous euclidize(). Available for metric-space residual. |
| `lut` | `MultiCameraLUT` or None | Per-camera LUT for radial shift. Available for metric-space residual. |
| `z_glass_top` | float | Z of air/glass interface (mm). Available for ray tracing. |
| `q_orig` | `np.ndarray (2K×N)` or None | Original pixel observations BEFORE H_k conditioning. Consistent with cameras_mm. |

**Why `q_orig` matters:** `y` (the target observations in `eval_y_and_dy`) is
in conditioned space. `q_orig` is in pixel space, consistent with
`cameras_mm[k].c`, `.xh`, `.yh`. When the metric residual path is activated,
`q_orig[2k, n]` and `q_orig[2k+1, n]` are the correct observed values to pass
to `observation_equation_221`.

---

### Change 3 — Local Storage and `_mm_active` Flag

```python
_cameras_mm  = cameras_mm
_lut         = lut
_z_glass_top = z_glass_top
_q_orig      = q_orig
_mm_active   = (cameras_mm is not None and lut is not None
                and q_orig is not None)
```

Stored immediately after `K, N = qivis.shape`. The `_mm_active` flag is the
single gate for the future metric residual path — when that path is added, it
will be `if _mm_active:` before the projection computation.

---

### Change 4 — Docstring Coordinate System Note

The docstring now contains a `Coordinate system note` section explaining:
- What space everything operates in
- Why cameras_mm cannot be used directly in the LM loop
- Where the multimedia correction lives instead
- What is needed to activate the metric residual path

This is part of the code, not just documentation.

---

## Changes Made to bundle_PX_proj.py

### Objective Closure Updated to Wire q_orig

```python
# Before
def objective(p_vec):
    return eval_y_and_dy(p_vec, P0_cond, TP, X0_cond, TX, y, qivis, RADIAL)

# After
def objective(p_vec):
    return eval_y_and_dy(p_vec, P0_cond, TP, X0_cond, TX, y, qivis, RADIAL,
                         cameras_mm=cameras_mm, lut=lut,
                         z_glass_top=z_glass_top, q_orig=q)
```

`q` (pixel-space observations, before conditioning) is available in the
closure scope of `bundle_PX_proj` and is now forwarded to `eval_y_and_dy`
as `q_orig`. This completes the full wiring:

```
GoCal.py
  cameras_mm, z_glass_top
    ↓
bundle_PX_proj(cameras_mm=cameras_mm, z_glass_top=z_glass_top)
  q (pixel observations available in closure scope)
    ↓
eval_y_and_dy(..., cameras_mm=cameras_mm, lut=lut,
              z_glass_top=z_glass_top, q_orig=q)
  _mm_active = True when all four params provided
```

All parameters are present and correctly routed. **No further changes to
bundle_PX_proj or eval_y_and_dy are needed when the metric path is activated.**

---

## What Was Not Changed

| Component | Reason |
|---|---|
| `P = P0 + TP @ iP` reconstruction | Correct — projective tangent space |
| `X = X0 + TX @ iX` reconstruction | Correct — projective tangent space |
| `x = P @ X` projection | Unchanged — standard pinhole in conditioned space |
| `y_residual = y_pred - y` | Unchanged — standard residual |
| Analytical Jacobian (dqdP, dqdX) | Unchanged — self-consistent in conditioned space |
| `levmarq` loop | Unchanged — standard LM |
| `vgg_conditioner_from_image` | Unchanged |

---

## What Remains for Full LM Multimedia Integration

To replace the `x = P @ X` projection with `observation_equation_221` inside
the LM loop, the following restructuring is required (future work):

```
Required:
  1. A metric-space tangent space parameterization:
       X_world = X0_world + TX_world @ iX
       cam_params = cam0 + T_cam @ ip
     instead of conditioned projective P and X.

  2. Passing X0_world (metric world coordinates) into eval_y_and_dy.

  3. Inside eval_y_and_dy, replace:
       x = P @ X
       q = p2e(x)
       residual = q - q_orig_k
     with:
       vx, vy, _ = observation_equation_221(X_world[:,n], cameras_mm[k],
                                            q_orig[2k,n], q_orig[2k+1,n])
       residual = [vx, vy]

  4. Replace analytical Jacobian with:
       J_cam = numerical_jacobian_221(X_world[:,n], cameras_mm[k],
                                     q_orig[2k,n], q_orig[2k+1,n])
       # J_cam is (2, 16) — maps to 16 Maas parameters per camera

  5. Restructure levmarq parameter vector to hold the 16 Maas params
     per camera + 3 world coords per point, instead of 11 projective
     params per camera + 3 tangent-space increments per point.
```

All data needed for steps 3 and 4 (`cameras_mm`, `lut`, `q_orig`,
`z_glass_top`) is already wired in and accessible via `_cameras_mm`,
`_lut`, `_q_orig`, `_z_glass_top`.

---

## Complete Step 3.9 Status After These Changes

| Sub-step | Status | Location |
|---|---|---|
| Multimedia params wired into LM objective | ✅ Done | bundle_PX_proj → eval_y_and_dy |
| q_orig (pixel observations) available in LM | ✅ Done | bundle_PX_proj closure |
| Post-BA X refinement via refracted rays | ✅ Done | bundle_PX_proj._mm_refine_X_from_q |
| LM residual replaced with multimedia model | 🔲 Future | Requires metric parameterization |
| Multimedia Jacobian in LM | 🔲 Future | Requires metric parameterization |

---

## References

- Maas, H.-G. (1996). *Contributions of Digital Photogrammetry to 3-D PTV.*
  Eqs. (4)–(8): iterative radial shift. Eq. (17): collinearity condition.
- See also: `bundle_PX_proj_EXPLANATION.md`, `fill_mm_bundle_EXPLANATION.md`,
  `GoCal_EXPLANATION.md`
