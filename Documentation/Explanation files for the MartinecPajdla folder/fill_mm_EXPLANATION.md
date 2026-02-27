# fill_mm Function - Explanation

## What It Does
**Main projective reconstruction function** from measurement matrix:
1. Selects best reconstruction strategy (sequence or central image)
2. Iteratively recovers structure and motion
3. Fills missing measurements
4. Performs final SVD factorization
5. Returns complete P and X

**Top-level pipeline function** - orchestrates the entire reconstruction process.

---

## Input/Output

### Input
- `M`: Measurement matrix (3m × n) with NaNs for unknown
- `opt`: Options dict with:
  - `'strategy'` (-1): Strategy selection
    - `-1`: Try all, choose best
    - `0`: Sequence mode only
    - `-2`: All central image strategies
    - `k > 0`: Use image k as central
  - `'create_nullspace'`: Null-space options
  - `'tol'` (1e-6): Tolerance
  - `'no_factorization'` (False): Skip final SVD
  - `'metric'` (1): Error metric (1=Euclidean, 2=std)
  - `'verbose'` (True): Print progress

### Output
- `P`: Camera matrices (3k × 4)
- `X`: 3D points (4 × n')
- `u1`: Unrecovered image indices
- `u2`: Unrecovered point indices
- `info`: Detailed information dict

---

## Algorithm Overview

### High-Level Flow
```
1. Preprocess: Remove points visible in < 2 images
2. Setup strategies (sequence/central)
3. While not converged:
   a. Compute predictions for all strategies
   b. Choose best strategy
   c. Reconstruct sub-scene (fill_mm_sub)
   d. Fill missing data
   e. Update visibility
4. Final SVD factorization
5. Return P, X, u1, u2, info
```

### Strategy Selection
- **Predictor F:** Number of missing points (lower = better)
- **Predictor S:** Number of scaled points (higher = better)
- Best strategy: `max(F)`, then `max(S)` as tiebreaker

---

## Step-by-Step Process

### Step 1: Preprocessing
```python
# Remove points visible in < 2 images
visible_count = sum(~isnan(M[0::3, :]))
valid_points = where(visible_count >= 2)
M = M[:, valid_points]
```

Need ≥2 views for triangulation.

### Step 2: Strategy Setup
```python
if strategy == -1:
    # All strategies
    Omega = [{'name': 'seq'}] + 
            [{'name': 'cent', 'ind': i} for i in range(m)]
elif strategy == 0:
    Omega = [{'name': 'seq'}]
elif strategy == -2:
    Omega = [{'name': 'cent', 'ind': i} for i in range(m)]
else:
    Omega = [{'name': 'cent', 'ind': strategy}]
```

### Step 3: Iterative Recovery
```python
while recoverable > 0 and added:
    # Compute predictions
    S, F, strengths = compute_predictions(Omega, I)
    
    # Choose best strategy
    sg = argmax(F), then argmax(S)
    
    # Set rows/cols for strategy
    rows, cols, central = set_rows_cols(...)
    
    # Reconstruct
    P, X, ... = fill_mm_sub(M_subset, ...)
    
    # Fill holes
    M[missing] = P @ X
    
    # Update
    I = ~isnan(M[0::3, :])
```

### Step 4: Final Factorization
```python
# Compute depths
for i in valid_depths:
    lambda[i] = M0 \ Mdepths

# Build rescaled B
B = M .* lambda

# Balance
B = balance_triplets(B)

# SVD
U, S, Vt = svd(B)
P = U[:, :4] @ sqrt(S[:4, :4])
X = sqrt(S[:4, :4]) @ Vt[:4, :]
```

---

## Helper Functions

### compute_predictions(Omega, I)
Computes predictor values for all strategies.

**For sequence:**
```python
b, lengths = subseq_longest(I)
F = sum(I == 0)  # Missing points
S = sum(lengths)  # Total length
```

**For central:**
```python
result = strength(central, I)
F = result['strength'][0]  # Missing in submatrix
S = result['num_scaled']    # Scaled points
```

### strength(central, I, general)
Computes central image strength.

**Process:**
1. Find images with ≥8 correspondences to central
2. Mark unusable images as all-zero
3. Find columns with ≥2 visible points
4. Extract submatrix
5. Compute strength and num_scaled

**Returns:**
- `strength`: [missing, total] in submatrix
- `good_rows`: Usable images
- `good_cols`: Usable points
- `Isub`: Submatrix
- `num_scaled`: Points scalable from central

### set_rows_cols(Omega, sg, F, S, strengths, I, info, opt)
Sets rows and columns for selected strategy.

**Sequence mode:**
```python
rows = all images
cols = all points
central = 0
```

**Central mode:**
```python
rows = strengths[central]['good_rows']
cols = strengths[central]['good_cols']
central = selected index
```

### normM(M) and normMback(P, T)
Hartley normalization and its inverse.

**normM:**
```python
for k in range(m):
    Tk = normu(M[k, valid_cols])
    M[k] = Tk @ M[k]
    T[k] = Tk
```

**normMback:**
```python
for k in range(m):
    P[k] = inv(Tk) @ P[k]
```

---

## Key MATLAB → Python Conversions

| MATLAB | Python | Description |
|--------|--------|-------------|
| `isfield(opt, 'f')` | `'f' in opt` | Dict key check |
| `exist('OCTAVE_VERSION', 'builtin')` | — | Skip (Python-specific) |
| `fflush(stdout)` | `flush=True` in print | Flush output |
| `keyboard` | `raise RuntimeError` | Debug breakpoint |
| `Omega{end+1}` | `Omega.append({})` | Append to list |
| `1:m` | `np.arange(m)` | Range |
| `find(condition)` | `np.where(condition)[0]` | Find indices |
| `setdiff(a, b)` | `np.setdiff1d(a, b)` | Set difference |
| `union(a, b)` | `np.union1d(a, b)` | Set union |
| `intersect(a, b)` | `np.intersect1d(a, b)` | Intersection |
| `M0(k2i(i)) \ Md(k2i(i))` | `lstsq(Md, M0)[0]` | Least squares |
| `normu(...)` | `Utils.normu(...)` | From Utils |
| `dist(...)` | `Utils.dist(...)` | From Utils |
| `subseq_longest(...)` | `Utils.subseq_longest(...)` | From Utils |

---

## Strategy Selection Logic

### Predictor F (Feasibility)
- **Missing points** in reconstruction
- Lower = better (fewer holes to fill)

### Predictor S (Size)
- **Total points** available
- Higher = better (more data)

### Selection
```python
candidates = argmax(F)
if tie:
    best = argmax(S[candidates])
return best
```

---

## Information Dictionary

### Structure
```python
info = {
    'omega': {
        'name': 'seq' or 'cent',
        'ind': central_index (if cent),
        'F': feasibility_score,
        'S': size_score
    },
    'sequence': [
        {
            'central': index,
            'scaled': percentage,
            'missing': percentage,
            'used_pts': percentage
        }
    ],
    'err': {
        'no_fact': error_before_factorization,
        'fact': error_after_factorization
    },
    'Mdepths': measurement_with_depths,
    'opt': options_used
}
```

---

## Usage Example

```python
from fill_mm import fill_mm
import numpy as np

# Measurement matrix with missing data
M = ...  # (3m × n) with NaNs

# Options
opt = {
    'strategy': -1,  # Try all strategies
    'create_nullspace': {
        'trial_coef': 1.0,
        'threshold': 0.01
    },
    'verbose': True,
    'tol': 1e-6,
    'no_factorization': False,
    'metric': 1
}

# Reconstruct
P, X, u1, u2, info = fill_mm(M, opt)

if P.size > 0:
    print(f"Success! {P.shape[0]//3} cameras, {X.shape[1]} points")
    print(f"Error: {info['err']['fact']:.6f}")
else:
    print(f"Failed: {len(u1)} images, {len(u2)} points lost")
```

---

## Common Use Cases

### 1. Automatic Strategy Selection
```python
opt = {'strategy': -1, ...}  # Best of all
P, X, u1, u2, info = fill_mm(M, opt)
```

### 2. Sequence Reconstruction
```python
opt = {'strategy': 0, ...}  # Video sequence
P, X, u1, u2, info = fill_mm(M, opt)
```

### 3. Central Image Mode
```python
opt = {'strategy': 3, ...}  # Image 3 as central
P, X, u1, u2, info = fill_mm(M, opt)
```

### 4. Fast Preview (No Factorization)
```python
opt = {'no_factorization': True, ...}
P, X, u1, u2, info = fill_mm(M, opt)
# Faster but lower quality
```

---

## Mathematical Background

### Projective Reconstruction
```
M = P @ X  (rank 4)
```

Unknown: P (3m × 4), X (4 × n)

### Iterative Filling
```
Iteration 1: Fill some missing entries
Iteration 2: Fill more (now more data)
...
Converge: All fillable entries filled
```

### Final Factorization
```
B = rescaled measurements
B = U @ S @ V^T  (SVD)
P = U[:, :4] @ sqrt(S[:4, :4])
X = sqrt(S[:4, :4]) @ V[:, :4]^T
```

Ensures rank-4 structure.

---

## Convergence

### Termination Conditions
1. **Success:** `recoverable == 0` (all fillable filled)
2. **Stagnation:** `added == 0` (no progress)
3. **Max iterations:** `iteration > 10` (safety)

### Typical Behavior
```
Iteration 1: Fill 50% of holes
Iteration 2: Fill 30% more
Iteration 3: Fill 15% more
Iteration 4: Fill 5% more → converged
```

---

## Error Metrics

### Metric 1: Euclidean
```python
error = sqrt(sum((M - P@X)²)) / num_points
```

Average reprojection error.

### Metric 2: Standard Deviation
```python
D = normalize(M) - normalize(P@X)
error = std([D_x, D_y])
```

Spread of errors.

---

## Pipeline Integration

### Complete Workflow
```python
# 1. Get measurements
M = build_measurement_matrix(correspondences)

# 2. Main reconstruction (this function)
P, X, u1, u2, info = fill_mm(M, opt)

# 3. Bundle adjustment
P_refined, X_refined, _ = qPXbundle_cmp(P, X, q)

# 4. Metric upgrade (if available)
```

### Uses These Functions
1. **fill_mm_sub** - Sub-scene reconstruction
2. **balance_triplets** - Matrix balancing
3. **Utils.normu** - Hartley normalization
4. **Utils.dist** - Error computation
5. **Utils.subseq_longest** - Sequence finding
6. **Utils.k2i** - Index conversion

---

## Performance

### Complexity
- **Per iteration:** O(strategies × fill_mm_sub)
- **fill_mm_sub:** Dominated by fill_prmm
- **Factorization:** O(min(m³, mn²))
- **Total:** 1-5 iterations typical

### Memory
- **M, M0:** 3m × n each
- **P, X:** 3m × 4, 4 × n
- **Temporary:** During SVD
- Peak during factorization

---

## Debugging Tips

### No Progress (Nothing Recovered)
**Cause:** Insufficient correspondences
**Check:**
- Each image pair has ≥8 common points?
- Try different strategy
- Increase point count

### High Error
**Cause:** Poor geometry or noise
**Check:**
- Camera motion sufficient?
- Noise level
- Try different metric

### Iteration Limit
**Cause:** Stuck in loop
**Check:**
- Degenerate configuration?
- Increase threshold
- Simplify problem

---

## Strategy Comparison

### Sequence Mode
**Best for:**
- Video sequences
- Ordered images
- Similar consecutive views

**Pros:**
- Uses all pairwise geometry
- Robust to missing data

**Cons:**
- Can accumulate drift
- May not use best pairs

### Central Mode
**Best for:**
- Convergent cameras
- One high-quality reference
- Turntable/circular motion

**Pros:**
- Single reference frame
- Consistent scale
- Clear geometry

**Cons:**
- Requires good central
- May exclude images

### Auto Mode (strategy=-1)
**Best for:**
- Unknown configuration
- Mixed scenarios
- Robustness priority

**Pros:**
- Tries all options
- Finds best automatically

**Cons:**
- Slower (tries multiple)
- May be overkill

---

## Advanced Features

### Iterative Recovery
Progressively fills missing data as more becomes known.

### Strategy Selection
Automatically chooses best approach based on data.

### Two-Phase Reconstruction
1. Projective (this function)
2. Bundle adjustment (separate)

### Normalization
Hartley normalization for numerical stability.

---

## Notes

- **Preprocessing crucial:** Removes unusable points
- **Strategy selection automatic:** Finds best approach
- **Iterative filling:** Progressively completes M
- **Final factorization:** Ensures rank-4 structure
- **Error tracking:** Both before/after factorization
- **Modular design:** Uses fill_mm_sub + balance_triplets
- **Highest-level function:** Complete reconstruction pipeline

---

## Testing

The module includes comprehensive test:
```bash
python fill_mm.py
```

**Note:** Test may fail if random data doesn't have sufficient correspondences between image pairs. Real data with proper tracking typically works well.

---

## Comparison with Other Approaches

### This (Iterative + SVD)
**Pros:**
- Handles missing data naturally
- Robust strategy selection
- Iterative filling

**Cons:**
- May take multiple iterations
- Needs sufficient initial data

### Direct Factorization
**Pros:**
- Single step
- Fast

**Cons:**
- Cannot handle missing data
- Less robust

### Bundle Adjustment
**Pros:**
- Highest quality
- Handles all distortion

**Cons:**
- Needs good initialization
- Slower

**Best practice:** Use fill_mm for initialization, then qPXbundle_cmp for refinement.
