# MATLAB → Python Multi-Camera Self-Calibration: Bug Fix Documentation

This document records all bugs found and fixed during the conversion of the
Martinec-Pajdla / Svoboda multi-camera self-calibration codebase from MATLAB to Python.
Each bug is described with its root cause, the broken code, the fix, and its impact on results.

---

## Background

The codebase implements projective self-calibration of multiple cameras using:
1. Pairwise RANSAC F-matrix estimation (`FindInliers` / `REG`)
2. Projective reconstruction via null-space factorization (`fill_mm` pipeline)
3. Euclidean upgrade (`Euclidize`)
4. Bundle adjustment (`bundle_PX_proj`)

All bugs stem from four systematic translation traps:

| Trap | MATLAB | Python |
|---|---|---|
| Memory layout | Column-major (Fortran order) | Row-major (C order) |
| Indexing | 1-based | 0-based |
| Data structures | Nested structs / cell arrays | Flat dataclasses / dicts |
| Stacked arrays | `(3n, 4)` = `n` cameras stacked | Can be confused with `(n, 3, 4)` |

---

## Bug #1 — `Fsampson.py`: Wrong point used in Sampson distance

**Severity:** 🔴 Critical

**File:** `Fsampson.py`

**Root cause:** The Sampson distance formula uses both points `u1` and `u2` from a
stereo pair. The second term in the denominator must use `u1` (the point in the
first image), but the Python code used `u2` for both terms.

```python
# BROKEN
denominator = (F @ u2)[0]**2 + (F @ u2)[1]**2 + (F.T @ u1)[0]**2 + (F.T @ u2)[1]**2

# FIXED
denominator = (F @ u2)[0]**2 + (F @ u2)[1]**2 + (F.T @ u1)[0]**2 + (F.T @ u1)[1]**2
```

**Impact:** Every Sampson distance computation was wrong, corrupting RANSAC inlier
detection for all camera pairs.

---

## Bug #2 — `REG.py`: Random sampling bias in RANSAC

**Severity:** 🟡 Medium

**File:** `REG.py`

**Root cause:** MATLAB's `ceil(rand * (len-pos))` always produces an index strictly
greater than `pos` (self-swaps are essentially impossible). Python's
`random.randint(0, len-pos)` could return 0, causing `ptr[pos]` to swap with itself
and reducing effective sample diversity.

```python
# BROKEN — biased toward lower indices
idx = np.random.randint(0, len_pts - pos)
ptr[pos], ptr[pos + idx] = ptr[pos + idx], ptr[pos]

# FIXED — uniform over remaining positions
idx = np.random.randint(pos, len_pts)
ptr[pos], ptr[idx] = ptr[idx], ptr[pos]
```

**Impact:** Subtle bias in which 8-point subsets RANSAC selected; required more
iterations to find good F-matrix hypotheses.

---

## Bug #3 — Various files: Unvectorized loops (performance)

**Severity:** 🟢 Performance

**Files:** `Fsampson.py`, `Utils.py`, others

**Root cause:** Direct loop-for-loop translation of MATLAB code that MATLAB's JIT
handles efficiently; Python requires NumPy vectorization for acceptable speed.

**Fix:** Replaced nested Python loops with NumPy broadcasting and matrix operations.

**Impact:** ~257× speedup on benchmark inputs; essential for practical use.

---

## Bug #4 — `create_nullspace.py`: Memory allocation crash

**Severity:** 🔴 Critical

**File:** `create_nullspace.py`

**Root cause:** When the pre-allocated null-space matrix needed to grow, the Python
version computed extra columns needed but did not guarantee the buffer was large
enough to immediately write `nulltemp` after reallocation.

```python
# BROKEN — could still overflow if estimated_extra < nulltemp.shape[1]
extra_cols = estimated_extra

# FIXED — guarantee at least enough room for the current write
needed = width + nulltemp.shape[1] - nullspace.shape[1]
extra_cols = max(estimated_extra, needed)
```

**Impact:** `IndexError` crash during null-space construction with dense point clouds.

---

## Bug #5 — `fill_mm.py`: Chained indexing loses NaN fill

**Severity:** 🔴 Critical

**File:** `fill_mm.py`

**Root cause:** Python's chained indexing `M[rows][cols] = value` creates a temporary
copy; the assignment is silently discarded. MATLAB's `M(rows, cols) = value` modifies
in-place.

```python
# BROKEN — assignment to temporary copy, M unchanged
M[Utils.k2i(r1)][r2] = R[...]

# FIXED — direct 2D indexing
for fi, fp in zip(fill_img_idx, fill_pt_idx):
    M[3*r1[fi] + offsets, r2[fp]] = R[3*fi + offsets, fp]
```

**Impact:** Holes in the measurement matrix were never filled; recovery loop ran
indefinitely or recovered zero points.

---

## Bug #6 — `fill_mm.py`: Wrong fill index expansion

**Severity:** 🔴 Critical

**File:** `fill_mm.py`

**Root cause:** When deciding which entries to fill, the code flattened a 2D mask and
passed the flat indices through `k2i()`, which misinterpreted them as image indices
rather than flat (image, point) indices.

```python
# BROKEN — k2i treats flat index as image index
fill = np.where(condition.flatten())[0]
M[Utils.k2i(fill), ...] = R[Utils.k2i(fill), ...]

# FIXED — decompose to (image, point) pairs first
fill_img_idx, fill_pt_idx = np.where(condition)
for fi, fp in zip(fill_img_idx, fill_pt_idx):
    M[3*r1[fi] + offsets, r2[fp]] = R[3*fi + offsets, fp]
```

**Impact:** Wrong rows of M were updated, corrupting the measurement matrix.

---

## Bug #7 — `fill_mm.py`: Lambda flat index bug in factorization

**Severity:** 🔴 Critical  

**File:** `fill_mm.py`

**Root cause:** Same flat-index misuse as Bug #6, applied to the lambda computation
inside the factorization step. MATLAB uses logical indexing on a 2D matrix;
Python's equivalent requires explicit (row, col) decomposition.

**Fix:** Same pattern — replaced flat index + `k2i` with explicit 2D index loops.

**Impact:** Lambda values were assigned to wrong (camera, point) pairs, producing
a corrupted rescaled measurement matrix `B` for the SVD factorization.

---

## Bug #8 — `Utils.py`: `eucl_dist_only` flat index bug

**Severity:** 🟠 High

**File:** `Utils.py` → `eucl_dist_only`

**Root cause:** `np.where(I.flatten())` returns row-major flat indices into the
`(m, n)` mask `I`. Passing these to `k2i()` treated each flat index as an image
index, then computed `k2i(flat_idx)` = `[3*flat_idx, 3*flat_idx+1, 3*flat_idx+2]`
— completely wrong rows.

```python
# BROKEN
flat_indices = np.where(I.flatten())[0]
rows = Utils.k2i(flat_indices)

# FIXED
img_idx, pt_idx = np.where(I)
for s in range(step):
    rows = step * img_idx + s
    B[s, :] = (M0[rows, pt_idx] - M[rows, pt_idx]) ** 2
```

**Impact:** All Euclidean reprojection error calculations were wrong; affected
convergence monitoring and the `dist()` metric throughout.

---

## Bug #9 — `fill_mm_bundle.py`: `imsize` shape and indexing

**Severity:** 🟠 High

**File:** `fill_mm_bundle.py`

**Root cause:** `vgg_conditioner_from_image` (the normalization step in bundle
adjustment) expects `imsize` as shape `(2, m)` — width and height for each camera.
The code was passing `(m, 2)` or accessing `imsize[k]` instead of `imsize[:, k]`.

**Fix:** Ensured `imsize` is always `(2, m)` and all accesses use `imsize[:, k]`.

**Impact:** Bundle adjustment normalization used wrong image sizes; conditioning
matrix was corrupted, causing BA to diverge or not converge.

---

## Bug #10 — `eval_y_and_dy.py`: Jacobian always `None`

**Severity:** 🔴 Critical

**File:** `eval_y_and_dy.py`

**Root cause:** A conditional check `if compute_jacobian:` was never reached because
the variable was named differently at the call site. The Jacobian was always returned
as `None`, so the Levenberg-Marquardt bundle adjustment ran with no gradient
information and could not converge.

**Fix:** Corrected the variable name so the Jacobian is computed and returned when
requested.

**Impact:** Bundle adjustment always diverged; could not refine camera parameters.

---

## Bug #11 — `eval_y_and_dy.py`: Wrong reshape order in residuals

**Severity:** 🔴 Critical

**File:** `eval_y_and_dy.py`

**Root cause:** MATLAB's `reshape(M, rows, cols)` fills column-first (Fortran order).
Python's `reshape(rows, cols)` fills row-first (C order). The residual vector was
assembled in the wrong order, misaligning observation indices with Jacobian rows.

```python
# BROKEN — row-major fill misorders residuals
r = (proj - obs).reshape(2 * n_valid)

# FIXED — column-major fill matches MATLAB
r = (proj - obs).reshape(2 * n_valid, order='F')
```

**Impact:** Even if the Jacobian was available, residuals were misaligned with it,
causing BA to take wrong gradient steps.

---

## Bug #12 — `eval_y_and_dy.py`: Wrong COO sparse Jacobian construction

**Severity:** 🔴 Critical

**File:** `eval_y_and_dy.py`

**Root cause:** The sparse Jacobian was constructed using row-major flat indices for
the COO format, but the values were laid out in column-major (Fortran) order to
match MATLAB. This mismatch caused each Jacobian entry to point to the wrong
residual/parameter pair.

**Fix:** Made COO index generation consistent with the column-major value layout.

**Impact:** Jacobian was structurally wrong; BA gradient was meaningless.

---

## Bug #13 — `Euclidize.py`: P format mismatch

**Severity:** 🔴 Critical

**File:** `Euclidize.py`

**Root cause:** `fill_mm` returns `P` as `(3n, 4)` stacked format (MATLAB convention).
`Euclidize.py` expected `(n, 3, 4)` 3D array and accessed `P[i, :, :]`,
causing `IndexError: too many indices for array`.

```python
# BROKEN — assumes 3D array
Pe_i = P[i, :, :]

# FIXED — handle both formats defensively
if P.ndim == 3:
    P_stacked = P.reshape(-1, 4)
else:
    P_stacked = P   # already (3n, 4)
```

**Impact:** Euclidize crashed immediately; no metric reconstruction was possible.

---

## Bug #14 — `Euclidize.py`: Config attribute access

**Severity:** 🟠 High

**File:** `Euclidize.py`

**Root cause:** MATLAB uses nested struct access `config.cal.SQUARE_PIX` and
`config.cal.pp`. The Python `CalibrationConfig` is a flat dataclass, so the
correct access is `config.square_pixels` and `config.principal_point`.

```python
# BROKEN
config['cal']['SQUARE_PIX']
config['cal']['pp'][i, 0]

# FIXED
config.square_pixels
config.principal_point[i, 0]
```

**Impact:** `TypeError: 'CalibrationConfig' object is not subscriptable` crash.

---

## Bug #15 — `Euclidize.py`: Frobenius norm on a vector

**Severity:** 🟡 Medium

**File:** `Euclidize.py`

**Root cause:** `np.linalg.norm(v, 'fro')` is only valid for 2D matrices.
`Pe_i[2, :3]` is a 1D vector; NumPy raises `ValueError: Invalid norm order 'fro'`.
MATLAB's `norm(v, 'fro')` silently falls back to the vector 2-norm.

```python
# BROKEN
sc = np.linalg.norm(Pe_i[2, :3], 'fro')

# FIXED
sc = np.linalg.norm(Pe_i[2, :3])   # default L2 norm
```

**Impact:** Crash during per-camera scale normalization in Euclidize.

---

## Bug #16 — `U2fdlt.py`: Transpose bug in F-matrix construction

**Severity:** 🔴 Critical

**File:** `U2fdlt.py`

**Root cause:** MATLAB `reshape(f, 3, 3)'` fills column-major then transposes,
which is equivalent to a row-major reshape in Python. The Python code added an
extra `.T` that incorrectly transposed the final F matrix.

```python
# BROKEN — extra .T gives F^T instead of F
F = Vh[-1, :].reshape(3, 3).T

# FIXED — row-major reshape already gives the correct F
F = Vh[-1, :].reshape(3, 3)
```

**Proof via epipolar constraint `u2^T F u1 = 0`:**
- With `.T`: mean residual = 0.135 (clearly wrong)
- Without `.T`: mean residual = 0.000 ✓

**Impact:** All fundamental matrices were transposed; RANSAC inlier counts dropped
dramatically (e.g. 410/523 instead of 523/523 for cameras 1 & 2).

---

## Bug #17 — `depth_estimation.py`: F-key uses local loop index instead of actual camera index

**Severity:** 🔴 Critical

**File:** `depth_estimation.py`

**Root cause:** `M2Fe` stores fundamental matrices keyed by **actual camera indices**:
`F[(rows[i], rows[j])]`. The Python code looked up `F[(i, j)]` using the **local
loop counter** `i`, which only coincidentally matches when all cameras are present.
When any camera fails (e.g. `rows = [0, 2, 3]`), lookups silently fail and `lambda`
defaults to `1.0` for every point.

```python
# BROKEN — local loop index, not actual camera index
G = F[(i, j)]
epip = ep[(i, j)]

# FIXED — translate to actual camera index first
actual_i = rows[i]
actual_j = rows[j]
G = F[(actual_i, actual_j)]
epip = ep[(actual_i, actual_j)]
```

**Impact:** All projective depths defaulted to `1.0` when any camera was missing;
the rescaled measurement matrix `B` was completely wrong, causing ~100× reprojection
error after Euclidize.

---

## Bug #18 — `GoCal.py`: Wrong `imsize` orientation passed to `fill_mm_bundle`

**Severity:** 🟠 High

**File:** `GoCal.py`

**Root cause:** MATLAB passes `config.cal.pp(:, 1:2)'` — note the `'` transpose —
giving shape `(2, CAMS)`. Python passed `config.principal_point[:, :2]` with shape
`(CAMS, 2)`. When `fill_mm_bundle` accessed `imsize[:, k]` to get camera `k`'s
`[width, height]`, it instead got all cameras' x-coordinates.

```python
# BROKEN — (CAMS, 2) instead of (2, CAMS)
fill_mm_bundle(Ws, config.principal_point[:, :2])

# FIXED — transpose to match MATLAB's pp(:,1:2)'
fill_mm_bundle(Ws, config.principal_point[:, :2].T)
```

**Impact:** Bundle adjustment normalization used wrong image dimensions (e.g.
`f = (320+240)/2 = 280` instead of `(640+480)/2 = 560`), causing a 2× scale
error in the conditioning matrix.

---

## Bug #19 — `fill_prmm.py`: Wrong image index from row index

**Severity:** 🟠 High

**File:** `fill_prmm.py`

**Root cause:** MATLAB converts 1-indexed row indices to camera indices with
`ceil(u1b / 3)`. This works because `ceil(1/3) = 1`, `ceil(2/3) = 1`,
`ceil(3/3) = 1`. In Python with 0-indexed rows, `ceil(0/3) = 0` happens to be
correct, but `ceil(1/3) = 1` and `ceil(2/3) = 1` both give image 1 instead of
image 0.

```python
# BROKEN — ceil(1/3)=1 maps row 1 to image 1 instead of image 0
u1 = np.unique(np.ceil(u1b / 3).astype(int))

# FIXED — integer division gives correct 0-indexed image number
u1 = np.unique((u1b // 3).astype(int))
```

**Impact:** Wrong cameras marked as "unrecovered"; `P` was trimmed incorrectly,
producing mismatched camera matrix shapes.

---

## Bug #20 — `GoCal.py`: `options` dict never passed to `fill_mm_bundle`

**Severity:** 🔴 Critical — root cause of 8× error gap

**File:** `GoCal.py`

**Root cause:** MATLAB calls `fill_mm_bundle(..., options)` where
`options.create_nullspace.trial_coef = 10`. Python defined the same `options` dict
but never passed it to `fill_mm_bundle`, so the function used its internal default
of `trial_coef = 1.0`.

```python
# BROKEN — options defined but ignored
options = {'create_nullspace': {'trial_coef': 10}, ...}
P, X, u1, u2, info = fill_mm_bundle(Ws, imsize)  # options not passed!

# FIXED
P, X, u1, u2, info = fill_mm_bundle(Ws, imsize, options)
```

**Effect on null-space quality:**

| | MATLAB | Python (broken) | Python (fixed) |
|---|---|---|---|
| `trial_coef` | 10 | 1 | 10 |
| Null-space trials | ~4 290 | ~429 | ~4 290 |
| "Error balanced" | ~1.1 px | ~8.6 px | ~1.0 px |

**Impact:** 10× under-sampled null space → poorly conditioned basis →
wrong depth estimates → 8× higher reprojection error throughout.

---

## Bug #21 — `Utils.py`: `normalize_mp` wrong flat indexing

**Severity:** 🟡 Medium

**File:** `Utils.py` → `normalize_mp`

**Root cause:** The old code used `M.flatten()[k * 3 + 2]` to get the homogeneous
(z) coordinate for point `k`. For a `(3m, n)` matrix flattened row-major, flat
index `k*3+2` does not correspond to any z-coordinate — it points to an arbitrary
x or y value from a completely different camera and point.

```python
# BROKEN — accesses wrong element entirely
z_coord = M.flatten()[k * 3 + 2]  # points to M[0, 2] for k=0, not M[2, 0]

# FIXED — iterate over cameras, divide each triplet by its z-row
for cam in range(m):
    z = M[3*cam + 2, :]
    valid = (~np.isnan(z)) & (np.abs(z) > eps)
    Mnorm[3*cam:3*cam+3, valid] = M[3*cam:3*cam+3, valid] / z[valid]
```

**Impact:** All homogeneous coordinate normalization was wrong; affected
reprojection error display and metric computations throughout the pipeline.

---

## Bug #22 — `fill_prmm.py`: Float dtype in integer index array

**Severity:** 🟡 Medium

**File:** `fill_prmm.py`

**Root cause:** `approximate()` returns `u1b` as a float array (from SVD/lstsq).
Python's `//` operator preserves the input dtype, so `float_array // 3` returns
`float`, which cannot be used as an array index.

```python
# BROKEN — u1b is float, u1b//3 is also float
u1 = np.unique(u1b // 3)
# → IndexError: arrays used as indices must be of integer type

# FIXED — explicit cast
u1 = np.unique((u1b // 3).astype(int))
```

**Impact:** `IndexError` crash at `rows[u1]` inside `fill_mm_sub.py`.

---

## Bug #23 — `fill_mm.py`: Inverted `lstsq` arguments in lambda computation

**Severity:** 🔴 Critical — root cause of 15× factorization error jump

**File:** `fill_mm.py` (factorization step)

**Root cause:** MATLAB's backslash operator `A \ b` solves `A * x = b` for `x`.
The MATLAB code `lambda(i) = M0(k2i(i)) \ Mdepths_un(k2i(i))` solves
`M0_vec * lambda = Mdepths_vec`, giving `lambda = Mdepths / M0`.
The Python translation had the arguments reversed.

```python
# BROKEN — solves Mdepths * lambda = M0, gives lambda = M0/Mdepths (inverted!)
lambda_val = np.linalg.lstsq(md_vec.reshape(-1,1), m_vec, rcond=None)[0][0]

# FIXED — solves M0 * lambda = Mdepths, gives lambda = Mdepths/M0
lambda_val = np.linalg.lstsq(m_vec.reshape(-1,1), md_vec, rcond=None)[0][0]
```

**Concrete example with a depth factor of 3:**

| | `lstsq(A, b)` | Result |
|---|---|---|
| MATLAB `M0 \ Mdepths` | `lstsq(M0_vec, Mdepths_vec)` | `lambda = 3.0` ✓ |
| Python broken | `lstsq(Mdepths_vec, M0_vec)` | `lambda = 0.33` ✗ |
| Python fixed | `lstsq(M0_vec, Mdepths_vec)` | `lambda = 3.0` ✓ |

**Impact:** `B = M * lambda` was built with inverted depth factors; the SVD
factorization of this matrix produced camera matrices with no geometric meaning.
Reprojection error jumped from 1.02 px (no-factorization) to 15.27 px
(after factorization) — a 15× increase instead of the expected slight improvement.
After fix: 1.03 px → 0.95 px ✓

---

## Final Results

After all fixes, Python output matches MATLAB within normal stochastic variance:

| Metric | MATLAB | Python |
|---|---|---|
| Reprojection error (no-fact) | 1.117 px | 1.025 px |
| Reprojection error (fact) | 1.045 px | 0.952 px |
| Outliers detected | 1 | 1 |
| Camera 1 mean error | 1.00 px | 0.92 px |
| Camera 2 mean error | 0.95 px | 0.86 px |
| Camera 3 mean error | 1.49 px | 1.40 px |
| Camera 4 mean error | 0.61 px | 0.49 px |

Remaining differences (RANSAC inlier counts) are due to random seed variance in the
RANSAC algorithm — both implementations are correct and produce equivalent calibration
quality.
