# `go_calib_optim_iter` — Function Explanation

## What does it do?

`go_calib_optim_iter` is the **top-level calibration function**. It takes the
raw detected corner locations from all calibration images and produces the
final camera model — intrinsics (focal length, principal point, distortion,
skew) and per-image extrinsics (rotation + translation) — by jointly
minimising pixel reprojection error over all parameters simultaneously.

It is the function you call to run a full calibration. Everything else in this
toolbox is a subroutine that it coordinates.

---

## Where it fits in the full pipeline

```
Input: x_list (corners), X_list (3D grid), image size nx×ny
    |
    v
go_calib_optim_iter   <- THIS FUNCTION
    |
    +-- Initialise intrinsics (fc, cc, kc, alpha_c)
    |
    +-- comp_ext_calib          (Step 4: linear + GN init for all poses)
    |       +-- compute_extrinsic_init      (linear DLT per image)
    |       +-- compute_extrinsic_refine    (GN refine per image)
    |       +-- check_active_images
    |
    +-- Main GN loop           (Step 6: joint update of ALL parameters)
    |       +-- project_points2  (per image, per iteration)
    |               +-- rigid_motion
    |                       +-- rodrigues
    |       +-- compute_extrinsic_refine  (optional re-refine per step)
    |
    +-- comp_error_calib        (Step 7: reprojection error at solution)
    |       +-- project_points2
    |
    +-- extract_parameters      (Step 9: unpack flat vector to named params)
    |
    v
Output: fc, cc, kc, alpha_c, KK, per-image poses, uncertainties, err_std
```

---

## Previously Converted Functions and Their Roles Here

| Function | Role in this file |
|---|---|
| `check_active_images` | Called 5+ times total: once to initialise flags, once after extrinsic init, once per GN iteration, once for uncertainty estimation |
| `comp_ext_calib` | Called once to initialise all per-image poses before the main loop. Internally runs the two-step linear→GN pose init for every active image |
| `compute_extrinsic_refine` | Called inside the optional extrinsic re-refinement step after each GN update of the intrinsics. Ensures poses stay consistent with the evolving intrinsics |
| `project_points2` | Called per active image per GN iteration. Returns both the reprojected pixel positions and all 6 analytic Jacobians that build the global JJ3 matrix. Also called again to rebuild JJ3 for uncertainty estimation |
| `comp_error_calib` | Called once after convergence to compute the final reprojection error and `sigma_x` for uncertainty scaling |
| `extract_parameters` | Called once at the end to unpack the flat solution vector into human-readable named parameters with their uncertainties |
| `rodrigues` | Called to convert rotation vectors to matrices when building the final pose lists for `comp_error_calib` |

---

## The Global Parameter Vector

All parameters are packed into a single flat vector of length `15 + 6·n_ima`:

```
param[0:2]              fc       (focal lengths fx, fy)
param[2:4]              cc       (principal point cx, cy)
param[4]                alpha_c  (skew coefficient)
param[5:10]             kc       (5 distortion coefficients)
param[10:15]            zeros    (reserved padding — keeps image slots at 15+6k)
param[15+6k : 15+6k+3]  omc_k   (rotation vector for image k)
param[15+6k+3 : 15+6k+6] Tc_k   (translation vector for image k)
```

This layout is shared with `extract_parameters` and `comp_error_calib`,
making the toolbox internally consistent.

---

## The Main Gauss-Newton Loop — Step by Step

### Building JJ3 — the global block Jacobian

For each active image `k`, `project_points2` returns:
- `dxdf`, `dxdc`, `dxdalpha`, `dxdk` — sensitivities to intrinsics → stacked into `A` (10×2N)
- `dxdom`, `dxdT` — sensitivities to pose → stacked into `B` (6×2N)

These are accumulated into the global (15+6n)×(15+6n) block matrix:

```
JJ3[0:10, 0:10]         += A @ A.T    (intrinsic-intrinsic, all images sum)
JJ3[base:base+6, ...]    = B @ B.T    (extrinsic-extrinsic, this image only)
JJ3[0:10, base:base+6]   = A @ B.T   (intrinsic-extrinsic coupling)
```

This block structure is the Schur complement form of the full Jacobian — it
makes the normal equations efficiently solvable without forming the huge
(2·N_total) × (15+6n) full Jacobian.

### The selected_variables mask

Not all parameters are estimated at every run. The mask `sel` (called
`selected_variables` in MATLAB) selects the subset of active parameters:

```python
sel[0:2]  = est_fc          # which focal lengths to update
sel[2:4]  = center_optim    # update principal point?
sel[4]    = est_alpha        # update skew?
sel[5:10] = est_dist         # which distortion coefficients?
sel[15+6k:15+6k+6] = active_images[k]  # this image's pose?
```

Only the `ind_Jac`-indexed rows and columns of JJ3 are solved, keeping
inactive parameters frozen.

### The smoothed update step

```python
alpha_smooth2 = 1 - (1 - alpha_smooth)^(iter+1)
param_innov   = alpha_smooth2 * solve(JJ3_sub, ex3_sub)
```

`alpha_smooth` damps the step size on cold starts (when fc, cc, kc are
unknown). It ramps from `alpha_smooth` toward 1 over iterations, switching
from a slow damped descent to a full Newton step as the Jacobian improves.

### Optional extrinsic re-refinement (`recompute_extrinsic=True`)

After updating the intrinsics, the stored extrinsics are no longer optimal.
`compute_extrinsic_refine` is run per image with the new intrinsics to
correct this. This is an extra GN solve per image per outer iteration, but it
greatly widens the basin of convergence — the outer loop needs far fewer
iterations with it enabled.

---

## Parameter Uncertainty Estimation

After convergence:
1. `sigma_x = std(all reprojection residuals)` — estimate of pixel noise level
2. JJ3 is rebuilt at the final solution
3. `solution_error[ind_Jac] = 3 * sqrt(diag(inv(JJ3_sub))) * sigma_x`

These are approximate 3-sigma intervals on each estimated parameter, under
the assumption that observation noise is i.i.d. Gaussian with std `sigma_x`.

---

## Key MATLAB → Python Differences

| Topic | MATLAB | Python |
|-------|--------|--------|
| Script vs function | Runs as a script in shared workspace | Python function with explicit arguments and return dict |
| `~exist('fc')` | Checks workspace variable | `fc=None` default argument |
| `eval(['X_' num2str(kk)])` | Dynamic variable lookup | `X_list[kk]` indexed list |
| `sparse(...)` | Built-in sparse matrix | Dense `np.zeros` (sparse via `scipy.sparse` in production) |
| `inv(JJ3)` | Explicit matrix inverse | `np.linalg.solve` (more stable) + `pinv` for uncertainty |
| `fprintf(1,...)` | Printed to stdout | `print()` with `verbose` flag to suppress |
| `return` at end | Script `return` exits early | Normal function return |
| `comp_error_calib` call | Uses workspace `ex` directly | Returns `ex_all`; `project_points2` passed explicitly to avoid pinhole fallback |

The most important structural change is converting the script into a proper
function with an explicit return value. In MATLAB the calibration results live
in the shared workspace (a form of global state); here they are cleanly
returned as a dictionary.
