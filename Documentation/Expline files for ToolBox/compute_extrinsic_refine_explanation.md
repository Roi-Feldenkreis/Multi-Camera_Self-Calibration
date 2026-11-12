# `compute_extrinsic_refine` — Function Explanation

## What does it do?

`compute_extrinsic_refine` performs **non-linear refinement of a camera's
extrinsic parameters** (its pose — rotation and translation relative to a
calibration target). Starting from a rough initial estimate, it iteratively
minimises the **reprojection error** — the pixel-space distance between
observed 2D feature points and the points predicted by the current camera
model — until it converges to a precise solution.

### Where it fits in the calibration pipeline

```
Linear initialisation (e.g. homography decomposition)
    ↓  produces omc_init, Tc_init  (rough, ignores distortion)
compute_extrinsic_refine  ← THIS FUNCTION
    ↓  produces omckk, Tckk  (precise, full distortion model)
Used by: extract_parameters, comp_error_calib
```

The linear step gives a fast but approximate pose. This function refines it
under the *full* camera model (including radial and tangential distortion)
to minimise pixel error directly.

---

## Previously Converted Functions Used

| Function | Role here |
|---|---|
| `project_points2.py` | Called at every iteration — reprojects 3D points with the current pose estimate and returns `dxpdom` (2N×3) and `dxpdT` (2N×3), the Jacobians that form J |
| `rodrigues.py` | Called once at the end to convert the converged rotation vector `omckk` into the rotation matrix `Rckk` |

Both functions use the same **F-order (column-major) Jacobian convention**,
so their outputs chain directly into the Gauss-Newton update without any
reordering.

---

## Inputs & Outputs

| Name | Shape | Description |
|------|-------|-------------|
| `omc_init` | (3,) | Initial rotation vector |
| `Tc_init` | (3,) | Initial translation vector |
| `x_kk` | (2, N) | Observed 2D image feature locations (pixels) |
| `X_kk` | (3, N) | Corresponding 3D world points |
| `fc` | (2,) | Focal lengths [fx, fy] |
| `cc` | (2,) | Principal point [cx, cy] |
| `kc` | (5,) | Distortion coefficients |
| `alpha_c` | float | Skew coefficient |
| `max_iter` | int | Maximum iterations (default 20) |
| `thresh_cond` | float | Stop if cond(J) exceeds this (default ∞) |
| `omckk` *(out)* | (3,) | Refined rotation vector |
| `Tckk` *(out)* | (3,) | Refined translation vector |
| `Rckk` *(out)* | (3,3) | Rotation matrix from `rodrigues(omckk)` |
| `JJ` *(out)* | (2N, 6) | Final Jacobian at convergence |

---

## The Gauss-Newton Algorithm — Step by Step

### Step 1 — Project the current estimate
```python
xp, dxpdom, dxpdT, *_ = project_points2(X_kk, om, T, fc, cc, kc, alpha_c)
```
Uses our converted `project_points2.py` to get both the predicted pixel
positions and their Jacobians w.r.t. `om` and `T`.

### Step 2 — Compute the residual
```python
ex = (x_kk - xp).ravel(order='F')    # (2N,)
```
The residual is the difference between observed and projected pixel positions,
flattened in F-order (interleaved x/y) to match the Jacobian row ordering.

### Step 3 — Stack the Jacobian
```python
JJ = np.hstack([dxpdom, dxpdT])       # (2N, 6)
```
The 6-column Jacobian combines sensitivity to rotation (3 columns) and
translation (3 columns). This is the heart of why `project_points2` was
designed to return those derivatives.

### Step 4 — Check conditioning
If `cond(J) > thresh_cond`, the normal equations are too ill-conditioned to
solve reliably, so the loop exits early. This protects against degenerate
configurations (e.g. all points on a single line).

### Step 5 — Gauss-Newton normal equations
```
(JᵀJ) · Δp = Jᵀ · e
```
Solving for `Δp` gives the parameter update that locally minimises
`‖e - J·Δp‖²`. We use `np.linalg.solve` rather than `inv(JᵀJ) @` for
numerical stability (avoids explicitly inverting the 6×6 matrix).

### Step 6 — Update and check convergence
```python
change = ‖Δp‖ / ‖p_new‖
```
When the relative parameter change drops below `1e-10`, the solution has
converged. After the loop, `rodrigues(omckk)` converts the final rotation
vector to a matrix.

---

## Why Gauss-Newton (not Gradient Descent)?

The MATLAB comment says "Gradient Descent" but the update rule
`Δp = (JᵀJ)⁻¹ Jᵀ e` is actually **Gauss-Newton**. This distinction matters —
understanding *why* Gauss-Newton is the right choice here explains the entire
Jacobian-centric design of the toolbox.

### The problem: non-linear least squares

The goal is to find the pose `p = [om; T]` (6 parameters) that minimises the
sum of squared reprojection errors across all N calibration points:

```
minimise  F(p) = ‖e(p)‖²  =  Σ ‖x_observed_k - x_projected(X_k, p)‖²
```

Because `x_projected` is a nonlinear function of `p` (it involves rotations,
perspective division, and polynomial distortion), there is no closed-form
solution — we need an iterative optimiser.

### Three candidate methods

**Gradient descent** steps in the direction of the negative gradient of F:
```
Δp = α · Jᵀe
```
It only uses first-order slope information. Near the solution it oscillates
in narrow valleys of the cost surface, taking tiny steps and converging very
slowly — sometimes thousands of iterations.

**Newton's method** uses the full second-order Hessian `H = ∂²F/∂p²` to
account for curvature and make perfectly scaled steps:
```
Δp = H⁻¹ · Jᵀe
```
This converges quadratically (the number of correct digits doubles each
iteration). The problem: computing the true Hessian requires second
derivatives of `x_projected` with respect to `p` — very expensive to derive
analytically and to compute at every step.

**Gauss-Newton** approximates the Hessian as `H ≈ JᵀJ`, discarding the
second-derivative terms:
```
Δp = (JᵀJ)⁻¹ · Jᵀe
```

### Why the Gauss-Newton approximation is valid here

The true Hessian of `F = ‖e(p)‖²` is:
```
H = JᵀJ  +  Σ eₖ · ∂²eₖ/∂p²
```
The second term involves the residuals `eₖ` weighted by second derivatives.
This term is small (and safely ignorable) when:

1. **Residuals are small** — which is exactly true here since we start from
   a decent linear initialisation. The refinement starts a few pixels away
   from the true solution, not far from it.
2. **The model fits the data well** — a well-calibrated camera with low noise
   has small residuals at the optimum by definition.

When these conditions hold, `JᵀJ` is an excellent Hessian approximation and
Gauss-Newton achieves the same near-quadratic convergence as full Newton —
without needing any second derivatives at all.

### Why this matters for the toolbox design

This is the reason every single function in this toolbox is built around
computing and returning **Jacobians**:

| Function | Jacobians returned | Used by |
|---|---|---|
| `rodrigues` | `dRdom` (9×3) | `rigid_motion` |
| `rigid_motion` | `dYdom`, `dYdT` (3N×3 each) | `project_points2` |
| `project_points2` | `dxpdom`, `dxpdT`, `dxpdf`, ... (2N×3/2 each) | `compute_extrinsic_refine`, global calibration |

They all feed into `JᵀJ` and `Jᵀe`. Without Gauss-Newton as the core
algorithm, all that Jacobian machinery would be unnecessary — gradient
descent only needs `Jᵀe`, and a finite-difference approach wouldn't need
analytic Jacobians at all. The analytic Jacobians make the update step both
fast (no finite-difference perturbations) and accurate (no numerical noise).

---

## Key MATLAB → Python Differences

| Topic | MATLAB | Python |
|-------|--------|--------|
| `nargin` defaults | Nested `if nargin < N` | `None`-defaulted keyword args |
| `inv(JJ2) * JJ' * ex(:)` | Explicit matrix inverse | `np.linalg.solve(JJ2, JJ.T @ ex)` — numerically stabler |
| `ex(:)` flatten | MATLAB column-major (F-order) | `.ravel(order='F')` |
| `[dxdom dxdT]` hstack | Side-by-side matrix concat | `np.hstack([dxpdom, dxpdT])` |
| `rodrigues(omckk)` | Returns R only | Returns `(R, dR)` — we unpack `Rckk, _` |
