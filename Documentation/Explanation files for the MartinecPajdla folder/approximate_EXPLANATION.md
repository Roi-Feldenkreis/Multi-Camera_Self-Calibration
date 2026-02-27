# approximate Function - Explanation

## What It Does
Computes **r-rank approximation** of measurement matrix M using null-space methods:
```
M ≈ P × X
```
Where:
- **P** = basis matrix (m × r)
- **X** = coefficient matrix (r × n)
- **r** = target rank

---

## Input/Output

### Input
- `M`: (m × n) measurement matrix (may have NaN for missing data)
- `r`: Target rank for approximation
- `P`: (m × r) initial basis matrix
- `opt`: Options dict with:
  - `'tol'`: Tolerance for rank computations
  - `'verbose'`: Print progress messages

### Output
- `P`: Updated basis matrix (may have fewer rows)
- `X`: (r × n) coefficient matrix
- `u1`: Unrecovered row indices
- `u2`: Unrecovered column indices

---

## Algorithm Overview

### Main Steps

1. **Find valid basis rows**
   - Rows of P with sufficient magnitude

2. **Approximate columns** (`approx_matrix`)
   - For each column, solve: `P[rows] @ x = M[rows, col]`
   - Stores missing columns

3. **Extend to missing rows** (`extend_matrix`)
   - For each missing row, solve: `p @ X = M[row, :]`
   - Recovers as many rows as possible

4. **Extend to missing columns** (transpose trick)
   - Apply `extend_matrix` to M^T
   - Recovers as many columns as possible

---

## Step-by-Step Process

### Step 1: Identify Valid Rows
```python
row_means = np.mean(np.abs(P.T), axis=0)
rows = np.where(row_means > tol)[0]
nonrows = setdiff(range(m), rows)
```
- Rows with non-zero basis coefficients
- These rows already fit the r-rank model

### Step 2: Approximate Each Column
```python
for each column j:
    valid_rows = where(M[:, j] not NaN)
    if rank(P[valid_rows, :]) == r:
        solve: P[valid_rows] @ x = M[valid_rows, j]
        X[:, j] = x
        M_approx[:, j] = P @ x
    else:
        mark column as missing
```

**Key equation:**
```
P[valid_rows] × X[:, j] = M[valid_rows, j]
```

Solve using least squares (left division in MATLAB).

### Step 3: Extend to Missing Rows
```python
for each missing row i:
    valid_cols = where(M[i, :] not NaN)
    if rank(X[:, valid_cols]) == r:
        solve: p @ X[:, valid_cols] = M[i, valid_cols]
        E[i, :] = p @ X
    else:
        mark row as unrecovered
```

**Key equation:**
```
p × X[:, valid_cols] = M[i, valid_cols]
```

Solve using least squares (right division in MATLAB).

### Step 4: Extend to Missing Columns
```python
# Apply extend_matrix to transpose
M_T, X_new = extend_matrix(M.T, M_approx.T, P.T, ...)
X_extended = X_new.T
```

Use same logic as Step 3 but on transposed data.

### Step 5: Error Checking
```python
if any column of X is all zeros:
    raise error("nullspace problem")
```

Ensures valid approximation.

---

## Helper Functions

### approx_matrix(M, P, r, opt)
Immerses columns of M into basis P.

**For each column j:**
1. Find valid (non-NaN) rows
2. Check if P[rows] has rank r
3. If yes: solve `P[rows] @ x = M[rows, j]`
4. If no: mark column as missing

**Returns:**
- `Mapp`: Approximated matrix
- `misscols`: Columns that couldn't be approximated
- `X`: Coefficient matrix

### extend_matrix(M, subM, X, rows, nonrows, r, tol)
Fills missing rows of M.

**For each missing row i:**
1. Find valid (non-NaN) columns
2. Check if X[cols] has rank r
3. Solve `p @ X[cols] = M[i, cols]`
4. Check error: if `max(|p @ X|) > 1/tol`, mark as unrecovered
5. Otherwise: `E[i, :] = p @ X`

**Returns:**
- `E`: Extended matrix
- `unrecovered`: Rows that couldn't be recovered
- `Pnonrows`: Basis coefficients for recovered rows

### setdiff_custom(a, b)
Custom set difference that maintains order.

Unlike NumPy's `setdiff1d` which sorts, this keeps original order.

### member(e, s)
Checks if element e is in set s.

---

## Key MATLAB → Python Conversions

| MATLAB | Python | Description |
|--------|--------|-------------|
| `P(rows,:)\M(rows,j)` | `np.linalg.lstsq(P[rows,:], M[rows,j])` | Left division (solve Ax=b) |
| `M(i,cols)/X(:,cols)` | `np.linalg.lstsq(X[:,cols].T, M[i,cols]).T` | Right division (solve xA=b) |
| `rank(A, tol)` | `np.linalg.matrix_rank(A, tol=tol)` | Matrix rank |
| `find(~isnan(M(:,j)))` | `np.where(~np.isnan(M[:,j]))[0]` | Find non-NaN |
| `isempty(arr)` | `len(arr) == 0` | Check empty |
| `A(m,n) = NaN` | `A = np.full((m,n), np.nan)` | NaN allocation |
| `setdiff(a,b)` | `np.setdiff1d(a,b)` | Set difference |
| `union(a,b)` | `np.union1d(a,b)` | Set union |
| `sum(~sum(X))` | `np.any(np.sum(X, axis=0)==0)` | Check zero columns |

---

## Mathematical Background

### Low-Rank Approximation
For matrix M with missing data, find best rank-r approximation:
```
min ||M - P×X||²_F  (over non-missing entries)
```

Where:
- P ∈ ℝ^(m×r)
- X ∈ ℝ^(r×n)
- r < min(m, n)

### Null-Space Method
The function uses null-space of cross-product spaces to:
1. Find r-dimensional subspace that best fits M
2. Project each column onto this subspace
3. Extend to missing rows/columns when possible

### Why It Works
If M truly has rank r:
- Each column lies in span(P)
- Each row can be written as linear combination of X rows
- Missing data can be recovered if enough constraints exist

---

## Usage Example

```python
from approximate import approximate
import numpy as np

# Low-rank matrix with missing data
m, n, r = 100, 150, 5
P_true = np.random.randn(m, r)
X_true = np.random.randn(r, n)
M = P_true @ X_true

# Add missing entries
M[np.random.rand(m, n) < 0.2] = np.nan

# Initial basis (from SVD)
M_filled = M.copy()
M_filled[np.isnan(M_filled)] = 0
U, s, _ = np.linalg.svd(M_filled, full_matrices=False)
P_init = U[:, :r]

# Options
opt = {
    'tol': 1e-6,
    'verbose': True
}

# Compute approximation
P_approx, X_approx, u1, u2 = approximate(M, r, P_init, opt)

# Reconstruct
M_approx = P_approx @ X_approx

print(f"Unrecovered rows: {len(u1)}")
print(f"Unrecovered cols: {len(u2)}")

# Error on valid entries
valid = ~np.isnan(M)
error = np.mean((M[valid] - M_approx[valid])**2)
print(f"MSE: {error}")
```

---

## Common Use Cases

### 1. Structure from Motion
```python
# M contains 2D point tracks across images
# Approximate with rank-4 (3D structure + projective)
P, X, u1, u2 = approximate(M, r=4, P_init, opt)
# P contains camera matrices
# X contains 3D point coordinates
```

### 2. Missing Data Completion
```python
# Fill in missing measurements
M_complete = P @ X
# Use for entries where M had NaN
```

### 3. Denoising
```python
# Low-rank approximation removes noise
M_denoised = P @ X
```

---

## Error Handling

### "nullspace problem 1" or "nullspace problem 2"
**Cause:** Some column of X became all zeros
**Solution:**
- Increase rank r
- Improve initial basis P
- Check for degenerate data

### High unrecovered counts (u1, u2)
**Cause:** Insufficient valid data to recover all rows/cols
**Solution:**
- More measurements needed
- Data too sparse
- Reduce rank r

### Reconstruction error too high
**Cause:** M doesn't truly have rank r, or too much noise
**Solution:**
- Increase r
- Use robust methods (RANSAC, etc.)
- Preprocess/denoise data

---

## Implementation Notes

### Left Division (MATLAB's `\`)
```python
# MATLAB: x = A\b
# Python: x = np.linalg.lstsq(A, b, rcond=None)[0]
```
Solves Ax = b for x (overdetermined system).

### Right Division (MATLAB's `/`)
```python
# MATLAB: x = b/A
# Python: x = np.linalg.lstsq(A.T, b.T, rcond=None)[0].T
```
Solves xA = b for x (underdetermined system).

### Rank Computation
```python
rank = np.linalg.matrix_rank(A, tol=tolerance)
```
Uses SVD: rank = number of singular values > tol.

---

## Comparison with SVD

### SVD Approach
```python
U, s, Vt = np.linalg.svd(M)
M_approx = U[:,:r] @ np.diag(s[:r]) @ Vt[:r,:]
```
**Problem:** Cannot handle missing data (NaN)

### This Function
- Handles missing data explicitly
- Uses existing basis P
- Can extend to missing rows/columns incrementally
- Better for sparse/incomplete measurements

---

## Performance Tips

1. **Good initial P:** Use SVD on filled data
2. **Set verbose=False:** For production code
3. **Appropriate tolerance:** 1e-6 typical, adjust for scale
4. **Check u1, u2:** If many unrecovered, data may be too sparse

---

## Notes

- Function modifies P in-place for recovered rows
- NaN handling is critical - all logic checks for valid data
- Order-preserving set operations (setdiff_custom) important for indexing
- Transpose trick allows code reuse (extend rows → extend columns)
- Tolerances critical for rank checks and error thresholds
