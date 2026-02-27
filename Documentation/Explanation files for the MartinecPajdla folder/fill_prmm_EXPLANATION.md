# fill_prmm Function - Explanation

## What It Does
Computes the **null-space** and **fills PRMM** (Projective Reconstruction Measurement Matrix):
1. Creates null-space from random 4-tuples
2. Computes basis L from null-space via SVD
3. Computes depths from basis
4. Approximates with rank-r factorization
5. Returns camera matrices P and 3D points X

**Main pipeline function** that orchestrates the complete reconstruction.

---

## Input/Output

### Input
- `M`: Measurement matrix (3m × n)
- `Idepths`: Depth indicator matrix (m × n)
  - 1 = depth known, 0 = unknown
- `central`: Central image index (0 for sequence mode)
- `opt`: Options dict with:
  - `'create_nullspace'`: Options for null-space creation
  - `'verbose'`: Print progress
  - `'tol'`: Tolerance for computations
- `info`: Information dict (will be updated)

### Output
- `P`: Camera matrices
- `X`: 3D points
- `u1`: Unrecovered image indices
- `u2`: Unrecovered point indices
- `lambda`: Depth scale factors
- `info`: Updated info dict with statistics

---

## Algorithm Overview

### Pipeline
```
1. create_nullspace(M, Idepths) → NULLSPACE
2. nullspace2L(NULLSPACE) → basis L
3. svd_suff_data(S) → check if valid
4. L2depths(L, M) → Mdepths, lambda
5. approximate(Mdepths) → P, X
```

### Why This Order?
- **Null-space first:** Captures constraints from all point correspondences
- **Basis extraction:** Reduces to essential rank-4 subspace
- **Depth computation:** Makes measurements consistent
- **Factorization:** Extracts camera and point parameters

---

## Step-by-Step Process

### Step 1: Create Null-Space
```python
NULLSPACE, result = create_nullspace(M, Idepths, central, opt)
```

Randomly samples 4-tuples of points and builds null-space.

**Statistics tracked:**
- `tried`: Number of 4-tuples attempted
- `used`: Number that contributed to null-space
- `failed`: Number that failed
- `tried_perc`: Percentage of all possible 4-tuples
- `used_perc`: Success rate

### Step 2: Compute Statistics
```python
tried_perc = tried / C(n, 4) * 100
used_perc = used / tried * 100
```

Where C(n, 4) = n choose 4 combinations.

### Step 3: Compute Basis from Null-Space
```python
L, S = nullspace2L(NULLSPACE, r=4, opt)
```

Extracts last 4 columns of U from SVD (or eigendecomposition).

**Two methods:**
- **Narrow:** `cols < 10*rows` → SVD directly
- **Wide:** `cols ≥ 10*rows` → Eigendecomposition of N@N^T

### Step 4: Check Data Sufficiency
```python
if svd_suff_data(S, r=4, threshold):
    # Proceed with reconstruction
else:
    # Return empty (insufficient data)
```

Checks if (n-r)-th singular value > threshold.

### Step 5: Compute Depths
```python
Mdepths, lambda = L2depths(L, M, Idepths, opt)
```

Makes measurements consistent with basis L.

### Step 6: Factorization
```python
P, X, u1b, u2 = approximate(Mdepths, r=4, L, opt)
```

Rank-4 approximation: Mdepths ≈ P @ X

### Step 7: Process Unrecovered Indices
```python
u1 = unique(ceil(u1b / 3))  # Row indices → image indices
killb = setdiff(k2i(u1), u1b)  # Rows to remove
```

Converts row-level indices to image-level indices and adjusts.

### Step 8: Adjust Lambda
```python
lambda = lambda[valid_images, valid_points]
```

Subsets lambda to match recovered P and X.

---

## Helper Functions

### nullspace2L(NULLSPACE, r, opt)

Computes basis L from null-space.

**For narrow matrices** (cols < 10×rows):
```python
U, s, Vt = svd(NULLSPACE)
S = diag(s)
L = U[:, -r:]  # Last r columns
```

**For wide matrices** (cols ≥ 10×rows):
```python
eigenvalues, U = eig(NULLSPACE @ NULLSPACE.T)
# Sort descending
sv = sqrt(eigenvalues)
S = diag(sv)
L = U[:, -r:]
```

**Why two methods?**
- SVD is O(mn²) for m×n matrix
- Eigendecomposition is O(m³) but on m×m matrix
- Wide matrices: m << n, so eig faster

**Returns:**
- `L`: (m × r) basis matrix
- `S`: Singular value matrix

### svd_suff_data(S, r, threshold)

Checks if data is sufficient for rank-r reconstruction.

**Logic:**
```python
if S[n-r-1, n-r-1] > threshold:
    return True  # Sufficient
else:
    return False  # Insufficient
```

**Why (n-r)-th singular value?**
- Want to extract last r columns of U
- If sv[n-r] is tiny, those columns are unreliable
- Indicates null-space rank is too low

**Degenerate cases return False:**
- Empty matrix
- Not enough columns
- Too few rows

---

## Key MATLAB → Python Conversions

| MATLAB | Python | Description |
|--------|--------|-------------|
| `info.sequence{end}` | `info['sequence'][-1]` | Last list element |
| `1:m` | `np.arange(m)` | Range |
| `ceil(u1b/3)` | `np.ceil(u1b/3).astype(int)` | Ceiling to int |
| `union(a, [])` | `np.unique(a)` | Unique elements |
| `setdiff(a, b)` | `np.setdiff1d(a, b)` | Set difference |
| `isempty(x)` | `len(x) == 0` or `x.size == 0` | Empty check |
| `k2i(u1)` | `Utils.k2i(u1, step=3)` | Index conversion |
| `comb(n, k)` | `Utils.comb(n, k)` | Combination |
| `clear NULLSPACE` | `del NULLSPACE` | Free memory |
| `sprintf(' %f', vals)` | `' '.join(f'{v:.6f}' for v in vals)` | Format numbers |

---

## Information Dictionary Structure

### Input/Output
```python
info = {
    'create_nullspace': { ... },  # Options used
    'sequence': [
        {
            'tried': int,
            'tried_perc': float,
            'used': int,
            'used_perc': float,
            'failed': int,
            'size_nullspace': tuple
        }
    ],
    'Mdepths': array  # If successful
}
```

### Statistics Tracked
- **tried**: 4-tuples attempted
- **tried_perc**: % of all possible C(n,4)
- **used**: 4-tuples that worked
- **used_perc**: Success rate (used/tried %)
- **failed**: 4-tuples that failed
- **size_nullspace**: Shape of null-space matrix

---

## Usage Example

```python
from fill_prmm import fill_prmm
import numpy as np

# Measurement matrix with missing data
M = ...  # (3m x n)
Idepths = ...  # (m x n) binary

# Options
opt = {
    'create_nullspace': {
        'trial_coef': 1.0,
        'threshold': 0.01,
        'verbose': True
    },
    'verbose': True,
    'tol': 1e-6,
    'info_separately': True
}

# Info structure
info = {'sequence': []}

# Fill PRMM
P, X, u1, u2, lambda_vals, info = fill_prmm(
    M, Idepths, central=0, opt=opt, info=info
)

if P.size > 0:
    print(f"Reconstruction successful!")
    print(f"P: {P.shape}, X: {X.shape}")
else:
    print(f"Insufficient data")
```

---

## Common Use Cases

### 1. Structure from Motion Pipeline
```python
# After getting point correspondences
M = build_measurement_matrix(correspondences)
Idepths = initialize_depths(M)

P, X, u1, u2, lambda_vals, info = fill_prmm(...)

# Use P, X for further refinement
```

### 2. Missing Data Reconstruction
```python
# Fill in missing measurements
P_filled, X_filled, ... = fill_prmm(M_sparse, ...)
M_complete = P_filled @ X_filled
```

### 3. Iterative Refinement
```python
for iteration in range(max_iter):
    P, X, u1, u2, lambda_vals, info = fill_prmm(...)
    if P.size > 0:
        # Refine depths, re-run
        Idepths_new = update_depths(lambda_vals)
```

---

## Mathematical Background

### Why Null-Space?

For rank-4 projective reconstruction:
```
M = P @ X  where rank(M) = 4
```

4-tuple constraints:
```
For 4 points, their measurements span a 4D space
Null-space of [m_1, m_2, m_3, m_4] captures this
```

### SVD of Null-Space
```
NULLSPACE = U @ S @ V^T
```

Where:
- U's last 4 columns span the measurement space
- S's last 4 values indicate reliability
- Small singular values → unreliable

### Sufficient Data Criterion
```
S[n-r-1, n-r-1] > threshold
```

Ensures the r-dimensional subspace is well-determined.

---

## Pipeline Integration

### Full Workflow
```python
# 1. Initial measurements
M_raw = get_correspondences()

# 2. Balance
M_bal = balance_triplets(M_raw)

# 3. Fill PRMM (this function)
P_init, X_init, ... = fill_prmm(M_bal, ...)

# 4. Bundle adjustment
P_final, X_final, _ = qPXbundle_cmp(P_init, X_init, q)
```

### Uses These Functions
1. **create_nullspace** - Generates constraints
2. **L2depths** - Computes consistent depths
3. **approximate** - Extracts P, X
4. **Utils.comb** - Combination counting
5. **Utils.k2i** - Index conversion

---

## Output Interpretation

### Successful Reconstruction
```python
if P.size > 0:
    # P: (3k x 4) cameras for k recovered images
    # X: (4 x n') points for n' recovered points
    # u1: Unrecovered image indices
    # u2: Unrecovered point indices
    # lambda: (k x n') depth scales
```

### Failed Reconstruction
```python
else:
    # P, X, lambda are empty arrays
    # u1 = all images
    # u2 = all points
```

**Causes of failure:**
- Empty null-space (no valid 4-tuples)
- Insufficient singular values
- Too much missing data

---

## Convergence and Quality

### Good Indicators
- High `used_perc` (>50%)
- Low `failed` count
- Singular values well above threshold
- Few unrecovered images/points

### Poor Indicators
- Low `used_perc` (<10%)
- High `failed` count
- Singular values near threshold
- Many unrecovered images/points

### Typical Values
- **tried_perc**: 0.1% - 5% (depends on trial_coef)
- **used_perc**: 20% - 80%
- **Nullspace size**: (3m, 100-1000) for reasonable data

---

## Performance

### Complexity
- **create_nullspace**: O(trials × 4 × m)
- **nullspace2L**: O(min(m³, m²n))
- **L2depths**: O(m × n)
- **approximate**: O(iterations × m × n)
- **Total**: Dominated by create_nullspace

### Memory
- **NULLSPACE**: Can be large (3m × trials)
- Deleted after computing L to save memory
- Peak usage during SVD/eigendecomposition

---

## Debugging Tips

### Empty Null-Space
**Cause:** All 4-tuples failed
**Check:**
- Sufficient data? (need ≥4 common points)
- trial_coef too small?
- threshold too strict?

### Insufficient Data Flag
**Cause:** sv[n-r] < threshold
**Check:**
- Increase trial_coef (more 4-tuples)
- Lower threshold (riskier)
- More/better measurements

### Many Unrecovered
**Cause:** Sparse data, rank issues
**Check:**
- Data quality
- Missing data pattern
- Try central mode instead of sequence

---

## Advanced Features

### Two-Method SVD
The function automatically chooses:
- **SVD** for narrow matrices
- **Eigendecomposition** for wide matrices

Optimizes for computational efficiency.

### Adaptive Index Killing
```python
for ib in killb[:-1]:
    # Adjust indices as rows removed
```

Ensures proper indexing after removing rows.

---

## Notes

- **Rank fixed at 4:** Projective reconstruction assumption
- **Info structure updated:** Tracks statistics across pipeline
- **Memory conscious:** Deletes NULLSPACE after use
- **Robust checks:** Multiple validation steps
- **Modular design:** Uses 3 already-converted functions
- **Utils integration:** Uses comb() and k2i() from Utils.py

---

## Comparison with Direct Methods

### This Approach (Null-Space)
**Pros:**
- Handles missing data naturally
- Robust to noise
- Uses all available constraints

**Cons:**
- Slower (random sampling)
- Non-deterministic (depends on samples)

### Direct SVD
**Pros:**
- Fast
- Deterministic

**Cons:**
- Cannot handle missing data
- Less robust to outliers

---

## Error Handling

### Empty Results
Function returns empty arrays but doesn't raise errors.

**Check success:**
```python
P, X, u1, u2, lambda_vals, info = fill_prmm(...)
if P.size == 0:
    print("Reconstruction failed")
    # Check info['sequence'][-1] for diagnostics
```

### Partial Recovery
```python
if len(u1) > 0 or len(u2) > 0:
    print(f"Partial: {len(u1)} images, {len(u2)} points lost")
```

---

## Testing

The module includes comprehensive tests:
```bash
python fill_prmm.py
```

Tests:
1. Full pipeline (sequence mode)
2. nullspace2L (both methods)
3. svd_suff_data (good/bad cases)
