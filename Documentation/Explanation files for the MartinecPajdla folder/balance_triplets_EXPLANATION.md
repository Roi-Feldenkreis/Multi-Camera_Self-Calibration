# balance_triplets Function - Explanation

## What It Does
Balances measurement matrix by iterative **column and row-triplet normalization**:
1. Normalize each column to weighted unit length
2. Normalize each row triplet to weighted unit area
3. Repeat until convergence

**Goal:** Uniform weighting across points and cameras, accounting for missing data.

---

## Input/Output

### Input
- `M`: (3m × n) measurement matrix
- `opt`: Options dict
  - `'verbose'`: Display progress (default True)
  - `'info_separately'`: Print on separate line (default True)

### Output
- `B`: Balanced measurement matrix (same shape as M)

---

## Algorithm Overview

### Iterative Balancing
```
while not converged and iterations < 20:
    1. Normalize columns → weight = #valid_rows
    2. Normalize row triplets → weight = #valid_cols
    3. Check change from previous iteration
```

### Convergence Criteria
Stop when ALL are satisfied:
- `change` ≤ 0.01
- `diff_rows` ≤ 1
- `diff_cols` ≤ 1

OR max 20 iterations reached.

---

## Step-by-Step Process

### Initialization
```python
B = M.copy()
change = inf
diff_rows = inf
diff_cols = inf
iteration = 0
```

Start with original matrix.

### Step 1: Normalize Each Column
```python
for each column l:
    rows = valid rows (non-NaN)
    rowsb = k2i(rows, step=3)
    
    s = sum(B[rowsb, l]²)
    supposed_weight = len(rows)
    
    B[rowsb, l] /= sqrt(s / supposed_weight)
```

**What it does:**
- Computes sum of squares for valid entries
- Normalizes so sum = #valid_rows
- Columns with more data get more weight

**Example:**
```
Column with 3 valid rows:
  Before: [10, 20, 30] → sum² = 1400
  After:  scale by sqrt(1400/3) ≈ 21.6
  Result: sum² ≈ 3
```

### Step 2: Normalize Each Row Triplet
```python
for each camera k:
    cols = valid cols (non-NaN)
    
    s = sum(B[3*k:3*k+3, cols]²)
    supposed_weight = len(cols)
    
    B[3*k:3*k+3, cols] /= sqrt(s / supposed_weight)
```

**What it does:**
- Computes sum of squares for triplet
- Normalizes so sum = #valid_cols
- Cameras with more points get more weight

### Step 3: Compute Change
```python
change = sum over all triplets:
    sum((B[triplet] - Bold[triplet])²)
```

Measures how much matrix changed this iteration.

### Step 4: Update Metrics
```python
diff_cols = max over columns:
    |actual_weight - supposed_weight|

diff_rows = max over triplets:
    |actual_weight - supposed_weight|
```

Tracks how far from target weights.

---

## Mathematical Background

### Why Balance?

Unbalanced matrices have:
- Different scales for different cameras
- Different scales for different points
- Biased optimization (favors large values)

### Goal Weights

**Columns:** Sum of squares = #valid_rows
- Column with k valid rows → weight k
- Empty columns → weight 0

**Row triplets:** Sum of squares = #valid_cols
- Triplet with k valid cols → weight k
- Empty triplets → weight 0

### Overall Weight

After balancing:
```
Total weight ≈ m × n
```

Where m = #cameras, n = #points.

Each point-camera pair contributes ~1 on average.

---

## Convergence Behavior

### Typical Pattern
```
Iteration 1: Large changes, high diff
Iteration 2-5: Rapid convergence
Iteration 6-10: Fine-tuning
Iteration 10+: Minimal changes
```

### Why Two Steps?

Column normalization affects row weights.
Row normalization affects column weights.

→ Must alternate until both stabilize.

---

## Key MATLAB → Python Conversions

| MATLAB | Python | Description |
|--------|--------|-------------|
| `M(1:3:end,l)` | `M[0::3, l]` | Every 3rd row |
| `M(3*k,:)` | `M[3*k, :]` | Row 3k |
| `M(3*k-2:3*k,cols)` | `M[3*k:3*k+3, cols]` | Triplet rows |
| `k2i(rows)` | `Utils.k2i(rows, step=3)` | Index conversion |
| `find(~isnan(...))` | `np.where(~np.isnan(...))[0]` | Find non-NaN |
| `tic/toc` | `time.time()` | Timing |
| `isfield(opt,'f')` | `'f' in opt` | Field check |
| `fprintf(1,'.')` | `print('.', end='', flush=True)` | Progress dot |

---

## Usage Example

```python
from balance_triplets import balance_triplets
import numpy as np

# Unbalanced measurement matrix
M = ...  # (3m x n) with varying scales

# Balance it
opt = {'verbose': True, 'info_separately': True}
B = balance_triplets(M, opt)

# Now use balanced matrix
# - Better for factorization
# - Better for optimization
# - Uniform error weighting
```

---

## Common Use Cases

### 1. Pre-processing for Factorization
```python
# Balance before factorization
B = balance_triplets(M)
P, X = approximate(B, r, ...)
```

Prevents bias toward large values.

### 2. Iterative Refinement
```python
# Balance during optimization
for iteration in range(max_iter):
    M_refined = optimize_step(M)
    M = balance_triplets(M_refined)
```

Maintains numerical stability.

### 3. Multi-Scale Data
```python
# Data from different cameras with different scales
M_balanced = balance_triplets(M_raw)
```

Equalizes contribution from all cameras.

---

## Output Characteristics

### Balanced Matrix Properties
- Columns have weighted unit length
- Row triplets have weighted unit area
- Missing data accounted for in weighting
- NaN pattern preserved
- Relative structure preserved

### Typical Changes
- Large scale factors reduced
- Small scale factors increased
- Overall variance decreased
- Condition number improved

---

## Convergence Monitoring

### Progress Indicators
```
Balancing PRMM.....  
```
Each `.` = one iteration.

### Convergence Metrics
- **change**: Total squared difference from previous
- **diff_cols**: Max column weight deviation
- **diff_rows**: Max row weight deviation

**Good convergence:**
```
change < 0.01
diff_cols < 1
diff_rows < 1
```

---

## Edge Cases

### All NaN Column
```python
if no valid rows:
    skip column (no normalization)
```

### All NaN Triplet
```python
if no valid cols:
    skip triplet (no normalization)
```

### Near-Zero Weight
```python
if s < 1e-10:
    skip (avoid division by zero)
```

Prevents inf/NaN from numerical issues.

---

## Differences from Standard Normalization

### Standard Column Normalization
```python
col /= norm(col)
```
Problem: Ignores missing data.

### This Function
```python
col /= sqrt(sum² / #valid_entries)
```
Benefit: Accounts for sparsity.

### Why It Matters
Column with 10 valid entries should have 10× weight of column with 1 entry.

---

## Performance

### Complexity
- **Per iteration:** O(m × n)
- **Typical iterations:** 5-10
- **Total:** O(m × n)

Very efficient!

### Memory
- Creates one copy of M
- In-place operations thereafter
- Minimal overhead

---

## Integration with Other Functions

### Typical Workflow
```python
# 1. Balance
B = balance_triplets(M)

# 2. Factorize
P, X = approximate(B, r, ...)

# 3. Refine
M_refined = ...  # some optimization

# 4. Re-balance
B2 = balance_triplets(M_refined)
```

Balancing is often applied multiple times in a pipeline.

---

## Verification

### Check Column Weights
```python
for col in range(n):
    rows = valid_rows(col)
    weight = sum(B[rows, col]²)
    expected = len(rows)
    assert abs(weight - expected) < 1.0
```

### Check Row Weights
```python
for k in range(m):
    cols = valid_cols(k)
    weight = sum(B[3*k:3*k+3, cols]²)
    expected = len(cols)
    assert abs(weight - expected) < 1.0
```

---

## Advantages

1. **Numerically Stable:** Prevents overflow/underflow
2. **Unbiased:** Equal weight to all measurements
3. **Sparse-Aware:** Handles missing data properly
4. **Fast:** Converges in few iterations
5. **Preserves Structure:** Only scales, doesn't change relationships

---

## When to Use

### Use balancing BEFORE:
- Matrix factorization
- Structure from motion
- Bundle adjustment
- Any optimization on M

### DON'T need balancing for:
- Already normalized data
- Single-camera problems
- Non-optimization tasks

---

## Notes

- **Convergence guaranteed:** Weights always improve or stay same
- **Multiple solutions:** Different orders give slightly different results
- **Preserves rank:** Doesn't change rank of M
- **Preserves null-space:** Doesn't change fundamental subspaces
- **Only rescales:** No rotation or translation
- **NaN-safe:** Protects against division by zero
