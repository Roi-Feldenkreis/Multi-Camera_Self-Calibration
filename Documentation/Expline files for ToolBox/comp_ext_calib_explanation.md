# `comp_ext_calib` — Function Explanation

## What does it do?

`comp_ext_calib` is the **orchestrator of the extrinsic estimation stage** in
the camera calibration pipeline. It loops over every calibration image, runs
the full two-step pose estimation (linear init → non-linear refinement), and
manages image activation — deactivating any view whose pose is numerically
unreliable and warning the user.

It sits at the junction of the pipeline where individual per-image pose
estimates are produced that will later be packed into the global `solution`
vector by functions like `extract_parameters`.

---

## Where it fits in the full calibration pipeline

```
Intrinsic estimation (fc, cc, kc, alpha_c already known)
    ↓
comp_ext_calib  ← THIS FUNCTION
    ├─ for each active image:
    │     ├─ compute_extrinsic_init   (Step 1: linear init, fast, approximate)
    │     └─ compute_extrinsic_refine (Step 2: Gauss-Newton, precise)
    │
    └─ returns omc_list, Tc_list (one pose per image)
         ↓
extract_parameters  (packs poses into the global solution vector)
comp_error_calib    (uses poses to compute reprojection error)
```

---

## Previously Converted Functions Used

| Function | Role here |
|---|---|
| `check_active_images.py` | Called twice — once at the start to initialise `active_images` and `ind_active`, once at the end to resync after any deactivations during the loop |
| `compute_extrinsic_refine.py` | Called per active image for Gauss-Newton non-linear refinement. Returns the refined pose and the final Jacobian `JJ_kk` used for conditioning checks |
| `normalize.py` | Used inside the fallback linear initialiser (`_default_extrinsic_init`) to remove intrinsics from observed points |
| `rodrigues.py` | Used inside the fallback initialiser to convert the recovered rotation matrix back to a rotation vector |

---

## Inputs & Outputs

| Name | Type | Description |
|------|------|-------------|
| `x_list` | list of (2, N_k) | Observed 2D corners per image |
| `X_list` | list of (3, N_k) | Corresponding 3D world points per image |
| `fc` | (2,) | Focal lengths [fx, fy] |
| `cc` | (2,) | Principal point [cx, cy] |
| `kc` | (5,) | Distortion coefficients |
| `alpha_c` | float | Skew coefficient |
| `active_images` | list of int | 1 = use, 0 = skip (None = all active) |
| `thresh_cond` | float | Condition number threshold for deactivation |
| `check_cond` | bool | Whether to deactivate ill-conditioned views |
| `max_iter` | int | Max Gauss-Newton iterations per image |
| `omc_list` *(out)* | list of (3,) | Rotation vectors (NaN if inactive) |
| `Tc_list` *(out)* | list of (3,) | Translation vectors (NaN if inactive) |
| `active_images` *(out)* | list of int | Updated flags after deactivations |
| `N_points_views` *(out)* | list of int | Number of points used per image |
| `deactivated_images` *(out)* | list of int | Indices newly deactivated this run |

---

## Step-by-step Breakdown

### Step 1 — Initialise active image flags
```python
n_ima, active_images, ind_active = check_active_images(n_ima, active_images)
```
Validates the `active_images` array and builds `ind_active`. Any image that
is missing data or has NaN corners is immediately flagged and skipped with a
warning.

### Step 2 — Linear pose initialisation (per active image)
```python
omc_init, Tc_init = compute_extrinsic_init_fn(x_kk, X_kk, fc, cc, kc, alpha_c)
```
Computes a fast closed-form starting estimate. In the full CalTech toolbox
this is `compute_extrinsic_init.m` (a homography decomposition for planar
targets, or a DLT for non-planar ones). The built-in fallback
`_default_extrinsic_init` implements a planar DLT that:
- Normalises image points via `normalize.py` (removes intrinsics/distortion)
- Solves a homography by SVD
- Decomposes H into R and T, projects R onto SO(3) via SVD
- Converts R to `omc_init` via `rodrigues.py`

This step intentionally ignores distortion — it only needs to be good enough
to give the Gauss-Newton loop a starting point in the right basin.

### Step 3 — Non-linear refinement (per active image)
```python
omckk, Tckk, Rckk, JJ_kk = compute_extrinsic_refine(
    omc_init, Tc_init, x_kk, X_kk, fc, cc, kc, alpha_c, max_iter, thresh_cond
)
```
Runs the full Gauss-Newton loop from our converted
`compute_extrinsic_refine.py`. This minimises pixel reprojection error under
the complete distortion model. Typically converges in 5–10 iterations.

### Step 4 — Conditioning and NaN checks
After refinement, two safety checks deactivate an image if:
- `cond(JJ_kk) > thresh_cond` — the Jacobian is near-singular, meaning the
  pose is geometrically degenerate (e.g. the calibration pattern is seen
  nearly edge-on, or there are too few points). This indicates the view
  gives unreliable information and should be excluded.
- `isnan(omckk[0])` — the optimiser diverged. Happens rarely but handled
  gracefully by silently deactivating the image.

### Step 5 — Re-sync active image flags
```python
_, active_images, ind_active = check_active_images(n_ima, active_images)
```
After the loop, `check_active_images` is called again to rebuild `ind_active`
consistently — any images deactivated during the loop are now properly
reflected in all returned arrays.

---

## Key MATLAB → Python Differences

| Topic | MATLAB | Python |
|-------|--------|--------|
| Dynamic variables | `eval(['x_' num2str(kk)])` | `x_list[kk]` — indexed list |
| `exist(['x_' num2str(kk)])` | Checks if workspace var exists | `x_kk is None` check |
| `desactivated_images` (sic) | Grows with `[... kk]` | `deactivated_images.append(kk)` |
| Script re-run | `check_active_images;` called twice | `check_active_images()` called twice explicitly |
| `fprintf` warnings | Printed to stdout | `print()` |

The most significant structural improvement is replacing the `eval`-based
dynamic variable lookup (`x_1`, `x_2`, ...) with a plain Python list. This
makes the code debuggable and removes the need for workspace-level variable
management entirely.
