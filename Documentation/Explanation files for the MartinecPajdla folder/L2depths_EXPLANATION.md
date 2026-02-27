# L2depths Function - Explanation

## What It Does
Computes **projective depths** from a basis matrix L by solving for depth scale factors that make measurements consistent with the basis.

**Key idea:** If M lies in the span of L, we can compute missing depths by solving a linear system.

---

## Input/Output

### Input
- `L`: (3m × r) basis matrix for the projective space
- `M`: (3m × n) measurement matrix
- `Idepths`: (m × n) binary depth indicator
  - 1 = depth already known/scaled
  - 0 = depth unknown (needs computation)
- `opt`: Options dict with `'verbose'` (default True)

### Output
- `Mdepths`: (3m × n) measurements scaled by computed depths
- `lambda`: (m × n) depth scale factors

---

## Algorithm Overview

For each point (column j):
1. Find images with valid measurements
2. Identify which need depth computation (Idepths=0)
3. Build constraint matrix using `spread_depths_col`
4. Solve linear system: `L @ coeffs = submatrix @ [1, depths]`
5. Apply computed depths to measurements

---

## Step-by-Step Process

### Step 1: Initialize
```python
Mdepths = M.copy()
lambda_vals = zeros(m, n)
```

Start with original measurements and zero depths.

### Step 2: Process Each Point
```python
for j in range(n):
    full = where(~isnan(M[0::3, j]))
    mis_rows = intersect(where(Idepths[:, j] == 0), full)
```

- `full`: Images with valid measurements for point j
- `mis_rows`: Images needing depth computation

### Step 3: Build Submatrix
```python
rowsbig = k2i(full, step=3)
col_data = M[rowsbig, j]
depth_col = Idepths[full, j]

submatrix = spread_depths_col(col_data, depth_col)
```

**What spread_depths_col does:**
- Column 0: Measurements with known depths (Idepths=1)
- Columns 1+: Each unknown depth gets its own column

**Example:**
```
Idepths = [1, 0, 0, 1]
submatrix = [known_depths | depth_1 | depth_2 | more_known]
```

### Step 4: Set Up Linear System
```python
right = submatrix[:, 0]
A = [L[rows, :] | -submatrix[:, 1:]]
```

**System:** 
```
L @ res[:r] - submatrix[:,1:] @ res[r:] = submatrix[:,0]
```

**Rearranged:**
```
[L | -submatrix[:,1:]] @ res = submatrix[:,0]
```

Where:
- `res[:r]` = basis coefficients
- `res[r:]` = unknown depth scales

### Step 5: Check Solvability
```python
if matrix_rank(A) < A.shape[1]:
    # Cannot compute - kill data
    Mdepths[invalid_rows, j] = NaN
    lambda[invalid_rows, j] = NaN
```

If underdetermined, depths can't be uniquely computed.

### Step 6: Solve for Depths
```python
res = lstsq(A, right)
```

Solves the linear system for depth coefficients.

### Step 7: Apply Depths
```python
# Known depths (column 0) get lambda = 1
i = indices_in_column_0
lambda[i, j] = 1.0
Mdepths[k2i(i), j] = M[k2i(i), j]

# Unknown depths get computed lambda
for ii in range(other_columns):
    i = indices_in_column_ii
    lambda[i, j] = res[r + ii]
    Mdepths[k2i(i), j] = M[k2i(i), j] * lambda[i, j]
```

---

## Mathematical Background

### Why This Works

If measurements M lie in a rank-r subspace spanned by L:
```
M ≈ L @ X
```

For each point with some known depths, we can write:
```
[λ₁·u₁]       [x₁]
[λ₂·u₂]   = L·[x₂]
[λ₃·u₃]       [x₃]
[λ₄·u₄]       [x₄]
```

Where some λᵢ are known (=1), others unknown.

### The Linear System

Separate known and unknown:
```
[u_known]     [x₁]   [λ_unk · u_unk]
[       ] = L·[x₂] - [              ]
            [x₃]
            [x₄]
```

Rearrange:
```
[L | -u_unk] @ [x, λ_unk] = u_known
```

Solve for x and λ_unk simultaneously.

---

## Key MATLAB → Python Conversions

| MATLAB | Python | Description |
|--------|--------|-------------|
| `M(1:3:end,j)` | `M[0::3, j]` | Every 3rd row |
| `k2i(full)` | `Utils.k2i(full, step=3)` | Index conversion |
| `spread_depths_col(...)` | `Utils.spread_depths_col(...)` | Depth spreading |
| `rank(A)` | `np.linalg.matrix_rank(A)` | Matrix rank |
| `A\b` | `np.linalg.lstsq(A, b)[0]` | Left division |
| `[A B]` | `np.hstack([A, B])` | Horizontal concat |
| `isfield(opt,'f')` | `'f' in opt` | Check field |
| `lambda(m,n)=0` | `lambda_vals = np.zeros((m,n))` | Allocation |
| `intersect(a,b)` | `np.intersect1d(a,b)` | Set intersection |

---

## Usage Example

```python
from L2depths import L2depths
import numpy as np

# Basis matrix (rank-4 subspace)
L = ...  # (3m x 4) matrix

# Measurements
M = ...  # (3m x n) matrix

# Depth indicators
Idepths = ...  # (m x n) binary matrix
# 1 where depth already known, 0 where needs computation

# Compute depths
Mdepths, lambda_vals = L2depths(L, M, Idepths, opt={'verbose': True})

# Use scaled measurements
# Mdepths contains M scaled by computed depths
# lambda_vals contains the scale factors
```

---

## Common Use Cases

### 1. Projective Reconstruction
```python
# After factorization: M ≈ L @ X
# Compute consistent depths
Mdepths, lambda_vals = L2depths(L, M, Idepths)
```

### 2. Missing Depth Recovery
```python
# Some depths known, others unknown
Idepths[known_depths] = 1
Idepths[unknown_depths] = 0

# Compute missing depths
Mdepths, lambda_vals = L2depths(L, M, Idepths)
```

### 3. Depth Refinement
```python
# Initial depths may be noisy
# Re-compute using basis constraint
# (function ignores initial depths)
```

---

## Output Interpretation

### Mdepths Matrix
- Original measurements scaled by computed depths
- `Mdepths[3*i:3*i+3, j] = M[3*i:3*i+3, j] * lambda[i, j]`
- NaN where depth couldn't be computed

### Lambda Matrix
- Scale factors for each measurement
- `lambda[i, j] = 1.0` for known depths
- `lambda[i, j] = computed` for unknown depths  
- `lambda[i, j] = NaN` where computation failed

**Example:**
```
lambda = [[1.0,  0.5,  NaN],    # Image 0
          [1.0,  1.0,  0.8],    # Image 1
          [0.7,  NaN,  1.0]]    # Image 2
```

---

## Why Depths Can't Be Computed

### Rank Deficiency
```python
if matrix_rank(A) < A.shape[1]:
    # Underdetermined system
```

**Causes:**
- Too few valid measurements
- Linear dependencies in constraints
- Degenerate configuration

**Result:** Set depths to NaN

---

## Integration with Other Functions

### Uses Utils Functions
```python
rowsbig = Utils.k2i(full, step=3)
submatrix = Utils.spread_depths_col(col_data, depth_col)
```

### Typical Workflow
```python
# 1. Get basis from factorization
P, X = approximate(M, r, ...)

# 2. Use one factor as basis
L = P

# 3. Compute depths
Mdepths, lambda_vals = L2depths(L, M, Idepths)

# 4. Use scaled measurements for refinement
```

---

## Differences from depth_estimation

### depth_estimation
- Uses **epipolar geometry** (F, epipoles)
- Computes depths from **geometric constraints**
- Sequential/central image modes
- One reference image

### L2depths
- Uses **algebraic basis** (L matrix)
- Computes depths from **subspace constraints**
- All points independently
- No reference image needed

**When to use which:**
- **depth_estimation**: Early in pipeline, from epipolar geometry
- **L2depths**: After factorization, to refine depths

---

## Error Handling

### SVD Convergence Failure
```python
try:
    rank_A = np.linalg.matrix_rank(A)
except LinAlgError:
    rank_A = 0  # Treat as rank deficient
```

Gracefully handles numerical issues.

### Missing Data
```python
if len(mis_rows) == 0:
    # No unknown depths - skip this point
    continue
```

Only processes points needing depth computation.

---

## Performance Notes

### Complexity
- **O(n · m · r²)** where n=points, m=images, r=rank
- Each point solved independently

### Optimization
- Vectorize if many points have same depth pattern
- Cache k2i conversions for reused indices

### Memory
- Pre-allocates lambda matrix
- Copies M for output (doesn't modify input)

---

## Verification

### Check Reconstruction Error
```python
# After computation, verify:
for j in range(n):
    reconstruction = L @ coeffs
    error = norm(Mdepths[:, j] - reconstruction)
    # Should be small
```

### Check Depth Consistency
```python
# Known depths should remain 1.0
known = Idepths == 1
assert np.allclose(lambda_vals[known], 1.0)
```

---

## Notes

- **No exploitation of known depths:** Function recomputes all depths from scratch based on L
- **Known depths serve as reference:** They define scale (set to 1.0)
- **Unknown depths computed relative to known:** Maintains consistency
- **NaN handling critical:** Invalid computations marked as NaN
- **Basis must span measurements:** If M ∉ span(L), results undefined

---

## Typical Values

### lambda
- Usually in range [0.5, 2.0]
- Extreme values (>5 or <0.2) indicate problems
- Many NaN suggests rank issues or sparse data

### Success Rate
- Good: >80% non-NaN depths
- Fair: 50-80%  
- Poor: <50% (check data quality or rank)
