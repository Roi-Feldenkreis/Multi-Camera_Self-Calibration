# create_nullspace Function - Explanation

## What It Does
Creates **null-space vectors** for 3D reconstruction from perspective cameras by:
1. Randomly selecting 4-tuples of points
2. Building constraint submatrices
3. Computing null-spaces of submatrices
4. Aggregating into global null-space

Used in structure-from-motion to find rank-deficient subspaces.

---

## Input/Output

### Input
- `M`: (3m × n) measurement matrix (m images, n points)
- `depths`: (m × n) binary matrix
  - 1 = point scaled by known depth
  - 0 = point not scaled
- `central`: Central image index (0 for sequence mode)
- `opt`: Options dict
  - `'trial_coef'`: Multiplier for trials (default 1)
  - `'threshold'`: SVD threshold (default 0.01)
  - `'verbose'`: Print progress (default False)

### Output
- `nullspace`: (3m × k) null-space matrix
- `result`: Dict with statistics
  - `'tried'`: Number of trials attempted
  - `'used'`: Number of successful 4-tuples
  - `'failed'`: Number of failed 4-tuples

---

## Algorithm Overview

### Main Loop
```python
for trial in range(num_trials):
    1. Randomly select 4 points (columns)
    2. Find images (rows) with all 4 points
    3. Build submatrix from depths and measurements
    4. Compute null-space of submatrix
    5. Add to global null-space if valid
```

### Why 4-tuples?
With perspective cameras, 4 point correspondences provide enough constraints to extract useful null-space information while being:
- **Overdetermined** enough for robust estimation
- **Small** enough to find frequently in sparse data

---

## Step-by-Step Process

### Step 1: Initialize
```python
I = ~isnan(M[0::3, :])  # Valid measurements
num_trials = round(trial_coef * n)
nullspace = zeros(3m, num_trials)
```

Track which measurements are valid (non-NaN).

### Step 2: Determine Scaled Columns
```python
if central > 0:
    cols_scaled[I[central, :]] = 1
```

In central mode, mark columns visible in central image as scaled.

### Step 3: Select 4-Tuple
```python
for t in range(4):
    c, cols = random_element(cols)
    cols_chosen.append(c)
    rows = intersect(rows, where(I[:, c]))
    
    if t < 3:
        rows, cols = cut_useless(...)
```

**Process:**
1. Pick random column
2. Keep only rows with this column
3. Filter useless rows/cols before next pick
4. Repeat 4 times

### Step 4: Build Submatrix
```python
rowsbig = k2i(rows, step=3)
for j in range(4):
    col_data = M[rowsbig, cols_chosen[j]]
    depth_col = depths[rows, cols_chosen[j]]
    submatrix = [... spread_depths_col(col_data, depth_col)]
```

**What spread_depths_col does:**
- Separates scaled and unscaled measurements
- Creates columns for known depths in one column
- Creates separate columns for unknown depths

### Step 5: Compute Null-Space
```python
subnull = nulleps(submatrix, threshold)
```

Finds vectors with singular values ≤ threshold.

**nulleps algorithm:**
```python
u, s, vt = svd(M)
sigsvs = sum(s > tol)
return u[:, sigsvs:]  # Small singular value vectors
```

### Step 6: Add to Global Null-Space
```python
if valid:
    nulltemp = zeros(3m, subnull.shape[1])
    nulltemp[rowsbig, :] = subnull
    nullspace[:, width:width+k] = nulltemp
    width += k
```

Embed local null-space into global coordinates.

### Step 7: Dynamic Memory
```python
if width + new_cols > nullspace.shape[1]:
    # Allocate more columns
    mean_added = width / trial
    extra = round(mean_added * remaining_trials)
    nullspace = expand(nullspace, extra)
```

Grows array as needed based on average success rate.

---

## Helper Functions

### cut_useless(I, cols_scaled, cols_chosen, rows, cols, demanded, scaled_ensured)
Filters rows and columns to maintain feasibility.

**Logic:**
1. **Check scaling requirements:** Ensure enough scaled columns
2. **Filter columns:** Keep those with ≥ demanded_rows valid rows
3. **Filter rows:** Keep those with ≥ demanded valid columns

**Purpose:** Ensures remaining choices can form valid 4-tuple.

### random_element(arr)
Picks random element from array.

```python
idx = random.randint(0, len(arr))
element = arr[idx]
rest = delete(arr, idx)
return element, rest
```

### nulleps(M, tol)
Computes null-space with tolerance.

**Returns vectors where:**
- Singular value ≤ tol
- These are "near null-space" vectors

**Why not exact null-space?**
- Noise in measurements
- Want vectors that nearly satisfy constraints
- More robust than strict null-space

---

## Mathematical Background

### Null-Space in Structure from Motion
For 4 points across m views:
```
[M_1,1  M_1,2  M_1,3  M_1,4]   [x_1]   [0]
[M_2,1  M_2,2  M_2,3  M_2,4] × [x_2] = [0]
[M_3,1  M_3,2  M_3,3  M_3,4]   [x_3]   [0]
    ...                         [x_4]
```

Where M_i,j represents point j in image i.

**Null-space vectors** encode constraints between points.

### Why This Matters
The null-space captures:
- Epipolar constraints between views
- Projective depth relationships
- Scene structure dependencies

Used for:
- Rank constraints in factorization
- Missing data recovery
- Projective reconstruction

---

## Key MATLAB → Python Conversions

| MATLAB | Python | Description |
|--------|--------|-------------|
| `M(1:3:end,:)` | `M[0::3,:]` | Every 3rd row from start |
| `find(I(:,c) > 0)` | `np.where(I[:,c])[0]` | Find indices |
| `intersect(a,b)` | `np.intersect1d(a,b)` | Set intersection |
| `[c, rest] = random_element(arr)` | `c, rest = random_element(arr)` | Random selection |
| `k2i(rows)` | `Utils.k2i(rows, step=3)` | Index conversion |
| `spread_depths_col(...)` | `Utils.spread_depths_col(...)` | Depth spreading |
| `svd(M)` | `np.linalg.svd(M)` | SVD |
| `size(M,1)` | `M.shape[0]` | Matrix dimension |
| `mod(x,y)` | `x % y` | Modulo |
| `round(x)` | `round(x)` | Rounding |

---

## Usage Example

```python
from create_nullspace import create_nullspace
import numpy as np

# Measurement matrix (3m x n)
M = ...  # Your multi-view measurements

# Depths indicator (m x n)
depths = ...  # 1 where depth known, 0 otherwise

# Options
opt = {
    'trial_coef': 1.0,    # Try n trials
    'threshold': 0.01,    # SVD threshold
    'verbose': True       # Show progress
}

# Compute null-space
nullspace, result = create_nullspace(M, depths, central=0, opt=opt)

print(f"Null-space shape: {nullspace.shape}")
print(f"Success rate: {result['used']}/{result['tried']}")

# Use null-space for reconstruction
# ... (in factorization algorithms)
```

---

## Common Use Cases

### 1. Projective Factorization
```python
# Create null-space
N, _ = create_nullspace(M, depths, 0, opt)

# Use in factorization
# M ≈ P × X where constraints from N
```

### 2. Missing Data Completion
```python
# Null-space encodes constraints
# Use to fill missing measurements
```

### 3. Rank Estimation
```python
# Number of null-space vectors indicates
# rank deficiency of measurement matrix
```

---

## Result Statistics

### 'tried'
Total number of 4-tuple trials attempted.

### 'used'
Number of successful trials that contributed null-space vectors.

**Low success rate causes:**
- Sparse data (many NaN)
- Insufficient point overlap
- Poor depth scaling

### 'failed'
Number of trials where no valid rows remained.

**High failure causes:**
- Points not visible across enough views
- Scaling requirements too strict

---

## Progress Output (verbose=True)

```
Used 4-tuples (.=10): (Allocating memory...)
-.10%-.20%-.30%-.40%-.50%-.60%-.70%-.80%-.90%
(Cutting unused memory...) (0.123 sec)
```

- `.` = 10 successful 4-tuples
- `-` = Underdetermined submatrix (rows ≤ cols)
- Numbers = Progress percentage
- Time = Total computation time

---

## Error Handling

### SVD Convergence Failure
```python
try:
    u, s, vt = np.linalg.svd(M)
except LinAlgError:
    return empty_nullspace
```

Returns empty null-space if SVD fails (numerical issues).

### Empty Rows
```python
if len(rows) == 0:
    failed = True
    continue
```

Skips 4-tuple if no valid configuration exists.

### Memory Management
Dynamically expands null-space array based on success rate:
```python
mean_added = width / trial
extra = round(mean_added * remaining)
```

---

## Differences from Standard Null-Space

### Standard null(A)
- Exact null-space: A × x = 0
- Only vectors with zero singular values

### This Function (nulleps)
- **Approximate null-space**: A × x ≈ 0
- Vectors with singular values ≤ threshold
- More robust to noise
- Captures "near constraints"

### Why This Matters
Real measurements have:
- Noise
- Imperfect correspondences
- Numerical errors

Strict null-space would be empty. Approximate null-space captures useful constraints.

---

## Performance Tips

1. **Adjust trial_coef:**
   - `< 1`: Faster but fewer null-space vectors
   - `> 1`: More vectors but slower

2. **Tune threshold:**
   - Smaller: Stricter null-space
   - Larger: More vectors but less accurate

3. **Ensure data quality:**
   - More complete measurements → higher success rate
   - Proper scaling → better null-space

4. **Monitor success rate:**
   - Low rate → adjust parameters or improve data

---

## Integration with Other Functions

### Uses Utils Functions
```python
rowsbig = Utils.k2i(rows, step=3)
spread = Utils.spread_depths_col(col_data, depth_col)
```

### Used By Factorization Algorithms
The null-space is used in:
- Rank-deficient factorization
- Projective reconstruction
- Bundle adjustment initialization

---

## Notes

- Random selection makes this non-deterministic
- More trials → more complete null-space
- Central mode handles scaling differently than sequence
- Empty null-space doesn't mean failure - data may already be full rank
- Threshold of 0.01 works well for normalized data
