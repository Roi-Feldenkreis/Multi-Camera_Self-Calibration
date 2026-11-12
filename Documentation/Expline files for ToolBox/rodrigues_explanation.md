# `rodrigues` — Function Explanation

## What does it do?

`rodrigues` converts between the two most common representations of a 3D rotation:

- A **rotation vector** `om` (3 elements) — the axis of rotation scaled by the angle in radians. Compact, easy to optimise over.
- A **rotation matrix** `R` (3×3) — the full linear map that rotates 3D points. Required for most geometry computations.

The function detects which representation was passed in and performs the appropriate conversion, also returning a **Jacobian matrix** that tells optimisers (e.g. Levenberg-Marquardt during camera calibration) how `R` or `om` changes with small perturbations.

---

## Inputs & Outputs

| Input shape | Output | Jacobian `dout` |
|-------------|--------|-----------------|
| (3,) or (3,1) or (1,3) | R : (3,3) rotation matrix | dRdin : (9,3) — d(R.ravel(F))/d(om) |
| (3,3) | om : (3,) rotation vector | domdR : (3,9) — d(om)/d(R.ravel(F)) |

---

## Case 1 — Rotation Vector → Matrix

Uses the **Rodrigues rotation formula**:
```
R = I·cos(θ) + [ω]×·sin(θ) + ωωᵀ·(1−cos(θ))
```
where `θ = ‖om‖` is the rotation angle and `ω = om/θ` is the unit axis.

### Steps

**Step 1 — Near-zero check.** If `θ < machine_epsilon`, return the identity matrix. The Jacobian in this limit is the linearised version of the formula evaluated at zero rotation.

**Step 2 — Build intermediate quantities.** Compute `α=cos(θ)`, `β=sin(θ)`, `γ=1−cos(θ)`, the skew-symmetric cross-product matrix `[ω]×` (`omegav`), and the outer product `A = ωωᵀ`.

**Step 3 — Compute R.** Apply the Rodrigues formula directly.

**Step 4 — Compute the Jacobian via chain rule.** The derivation uses four intermediate Jacobian matrices chained together:
```
dRdin = dRdm1 · dm1dm2 · dm2dm3 · dm3din
```
Each matrix captures how one set of intermediate variables depends on the previous. This avoids having to differentiate `R` with respect to `om` directly, which would be messy.

---

## Case 2 — Rotation Matrix → Vector

Extracts the rotation angle and axis from `R` using the matrix logarithm approach.

### Steps

**Step 1 — Project onto SO(3).** Uses SVD (`R = U·Σ·Vᵀ`) to snap any numerically imperfect matrix onto the nearest valid rotation matrix: `R_clean = U·Vᵀ`.

**Step 2 — Extract angle.** `θ = arccos((trace(R)−1)/2)`

**Step 3 — Three sub-cases based on `sin(θ)`:**

| Condition | Situation | Action |
|-----------|-----------|--------|
| `sin(θ) ≥ 1e-5` | General rotation | Full Jacobian via chain rule |
| `sin(θ) < 1e-5`, `trace > 0` | Near-zero rotation | Return `om = [0,0,0]`, linearised Jacobian |
| `sin(θ) < 1e-5`, `trace ≤ 0` | Rotation angle ≈ π | Special closed-form; Jacobian is undefined (returns NaN with a warning) |

---

## Key MATLAB → Python Differences

| Topic | MATLAB | Python |
|-------|--------|--------|
| Matrix flattening | `M(:)` is column-major | `M.ravel(order='F')` |
| Reshape | `reshape(v,3,3)` is column-major | `v.reshape(3,3,order='F')` |
| `eps` | Built-in scalar | `np.finfo(float).eps` |
| `[1 5 9]` index (1-based) | Diagonal elements of 9-vector | Python 0-based: `[0, 4, 8]` |

### Critical gotcha — F-order flattening

The single most important conversion detail: all matrices stored as row-vectors in the Jacobian (`R(:)`, `omegav(:)`, `A(:)`) use **column-major (Fortran) order** in MATLAB. In Python you must use `order='F'` explicitly in both `ravel` and `reshape` calls. Using the default C-order produces a silently wrong Jacobian with large errors.

To apply the Jacobians correctly in Python:
```python
# Vector → matrix Jacobian
R_new ≈ R + (dRdin @ dom).reshape(3, 3, order='F')

# Matrix → vector Jacobian
om_new ≈ om + domdR @ dR.ravel(order='F')
```
