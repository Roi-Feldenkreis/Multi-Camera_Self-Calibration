# fill_mm_sub Function - Explanation

## What It Does
**Projective reconstruction of a normalized sub-scene:**
1. Estimates fundamental matrices and epipoles
2. Computes depth scale factors
3. Builds rescaled measurement matrix
4. Balances matrix
5. Fills missing data using PRMM algorithm
6. Returns cameras P, points X, and depth scales

**High-level reconstruction function** for a single sub-scene or sequence.

---

## Input/Output

### Input
- `Mfull`: Complete known measurements (3m × n)
  - Used for best estimate of fundamental matrices
  - May have fewer NaNs than M
- `M`: Measurement matrix subset (3m × n)
  - Actual data to reconstruct
  - May have more missing data
- `central`: Central image index
  - `0` or `None`: Sequence mode
  - `k > 0`: Use image k as central
- `opt`: Options dict with:
  - `'verbose'`: Print progress
  - `'create_nullspace'`: Options for null-space creation
  - `'tol'`: Tolerance for computations
- `info`: Information dict (will be updated)

### Output
- `P`: Camera matrices (3k × 4) for k recovered images
- `X`: 3D points (4 × n') for n' recovered points
- `lambda`: Depth scale factors (k × n')
- `u1`: Unrecovered image indices
- `u2`: Unrecovered point indices
- `info`: Updated info dict

---

## Algorithm Overview

### Pipeline
```
1. M2Fe(Mfull) → F, ep, rows, nonrows
2. depth_estimation(M) → lambda, Ilamb
3. Build rescaled B = M .* lambda
4. balance_triplets(B) → balanced B
5. fill_prmm(B) → P, X, lambda1
6. Merge lambda and lambda1
```

### Why This Order?
- **Fundamental matrices first:** Need epipolar geometry
- **Depth estimation:** Makes measurements projectively consistent
- **Rescaling:** Incorporates depth information
- **Balancing:** Improves numerical conditioning
- **Fill PRMM:** Completes reconstruction

---

## Step-by-Step Process

### Step 1: Compute Visibility
```python
I = ~isnan(M[0::3, :])  # (m × n)
```

Determines which points are visible in which cameras.

### Step 2: Estimate Fundamental Matrices
```python
F, ep, rows, nonrows = M2Fe(Mfull, central)
```

**Returns:**
- `F`: Dict of fundamental matrices
- `ep`: Dict of epipoles
- `rows`: Images with valid geometry
- `nonrows`: Images without valid geometry

**Early exit:** If `len(rows) < 2`, cannot reconstruct.

### Step 3: Determine Central Image
```python
if not central:
    rows_central = 0  # Sequence mode
else:
    rows_central = find(rows == central)
```

Finds position of central image in `rows` array.

### Step 4: Compute Depth Scale Factors
```python
M_subset = M[k2i(rows), :]
lambda, Ilamb = depth_estimation(M_subset, F, ep, rows, rows_central)
```

**Returns:**
- `lambda`: (len(rows) × n) depth scales
- `Ilamb`: (len(rows) × n) depth validity mask

### Step 5: Prepare Visualization Info
```python
info['show_prmm']['I'] = I
info['show_prmm']['Idepths'] = zeros(m, n)
info['show_prmm']['Idepths'][rows, :] = Ilamb
```

Stores information for later visualization.

### Step 6: Build Rescaled Matrix B
```python
for i in range(len(rows)):
    B[k2i(i), :] = M[k2i(rows[i]), :] * (ones(3,1) @ lambda[i,:])
```

**Rescaling formula:**
```
B_i = M_i .* [1; 1; 1] * lambda_i
```

Each row triplet scaled by corresponding depth.

### Step 7: Balance Matrix
```python
B = balance_triplets(B, opt)
```

Normalizes columns and row triplets for better conditioning.

### Step 8: Fill PRMM
```python
P, X, u1, u2, lambda1, info = fill_prmm(B, Ilamb, central, opt, info)
```

Completes the reconstruction using null-space method.

### Step 9: Adjust Indices
```python
r1 = setdiff(1:len(rows), u1)  # Valid image indices
r2 = setdiff(1:n, u2)           # Valid point indices
```

Converts local indices to original indices.

### Step 10: Merge Lambda Values
```python
lambda = lambda[r1, r2]

# Update with newly computed depths
new = ~Ilamb[r1,r2] & I[r1,r2]
lambda[new] = lambda1[new]
```

Combines original depths with newly computed ones.

### Step 11: Compute Error
```python
error = eucl_dist_only(B[k2i(r1), r2], P@X, valid_mask, step=3)
```

Measures reconstruction quality.

### Step 12: Update Unrecovered Images
```python
u1 = union(nonrows, rows[u1])
```

Includes both:
- Images without valid geometry (`nonrows`)
- Images that failed reconstruction (`rows[u1]`)

---

## Key MATLAB → Python Conversions

| MATLAB | Python | Description |
|--------|--------|-------------|
| `M(1:3:end,:)` | `M[0::3, :]` | Every 3rd row |
| `isempty(central)` | `central is None or central == 0` | Empty check |
| `1:m` | `np.arange(m)` | Range |
| `find(rows == central)` | `np.where(rows == central)[0]` | Find indices |
| `k2i(rows)` | `Utils.k2i(rows, step=3)` | Index conversion |
| `[1;1;1]*lambda(i,:)` | `ones(3,1) @ lambda[i,:].reshape(1,-1)` | Broadcasting |
| `setdiff(a, b)` | `np.setdiff1d(a, b)` | Set difference |
| `union(a, b)` | `np.union1d(a, b)` | Set union |
| `~isnan(B(3*r1,r2))` | `~np.isnan(B[3*r1, :][:, r2])` | Validity mask |
| `eucl_dist_only(...)` | `Utils.eucl_dist_only(...)` | Euclidean distance |

---

## Information Dictionary

### Input/Output Structure
```python
info = {
    'show_prmm': {
        'I': array,        # Visibility matrix (m × n)
        'Idepths': array   # Depth validity (m × n)
    },
    'sequence': [...],     # From fill_prmm
    'create_nullspace': {...},  # Options used
    'Mdepths': array      # From fill_prmm
}
```

### Purpose
- **I**: Which points visible in which cameras
- **Idepths**: Which depths are known vs computed
- Used by visualization functions

---

## Usage Example

```python
from fill_mm_sub import fill_mm_sub
import numpy as np

# Measurement matrices
Mfull = ...  # (3m × n) complete measurements
M = ...      # (3m × n) subset with more missing data

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

# Reconstruct (sequence mode)
P, X, lambda_vals, u1, u2, info = fill_mm_sub(
    Mfull, M, central=0, opt=opt, info=info
)

if P.size > 0:
    print(f"Success! P: {P.shape}, X: {X.shape}")
    # Use P, X for further processing
else:
    print(f"Failed: {len(u1)} images, {len(u2)} points lost")
```

---

## Common Use Cases

### 1. Sequence Reconstruction
```python
# Multiple images in sequence
P, X, lambda_vals, u1, u2, info = fill_mm_sub(
    Mfull, M, central=0, opt=opt, info=info
)
```

### 2. Central Image Mode
```python
# Using image 3 as central reference
P, X, lambda_vals, u1, u2, info = fill_mm_sub(
    Mfull, M, central=3, opt=opt, info=info
)
```

### 3. Incremental Reconstruction
```python
# Start with good subset
P_init, X_init, ... = fill_mm_sub(M_good, M_good, ...)

# Add more data
P_refined, X_refined, ... = fill_mm_sub(M_good, M_more, ...)
```

---

## Mathematical Background

### Projective Depth Scaling

Original measurements:
```
m_i^j = P_i @ X_j
```

With unknown depths λ:
```
M_i^j = λ_i^j · m_i^j
```

Rescaled measurements:
```
B_i^j = M_i^j  (already scaled)
```

Then:
```
B ≈ P @ X  (rank 4)
```

### Why Mfull vs M?

- **Mfull**: More complete, better for F matrices
- **M**: Actual target for reconstruction
- Allows using different data sources

### Two-Stage Depth Computation

**Stage 1:** `depth_estimation` → Initial depths from epipolar geometry

**Stage 2:** `fill_prmm` → Refines depths during factorization

**Merge:** Combines both sources of depth information

---

## Pipeline Integration

### Full Reconstruction Workflow
```python
# 1. Get measurements
Mfull, M = get_measurements()

# 2. Reconstruct sub-scene (this function)
P, X, lambda_vals, u1, u2, info = fill_mm_sub(...)

# 3. Bundle adjustment
P_refined, X_refined, _ = qPXbundle_cmp(P, X, q)
```

### Uses These Functions
1. **M2Fe** - Epipolar geometry
2. **depth_estimation** - Initial depths
3. **balance_triplets** - Matrix conditioning
4. **fill_prmm** - Complete reconstruction
5. **Utils.k2i** - Index conversion
6. **Utils.eucl_dist_only** - Error computation

---

## Output Interpretation

### Successful Reconstruction
```python
if P.size > 0:
    # P: (3k × 4) for k = len(rows) - len(u1) images
    # X: (4 × n') for n' = n - len(u2) points
    # lambda: (k × n') depth scales
    # u1: Failed image indices
    # u2: Failed point indices
```

### Failed Reconstruction
```python
else:
    # P, X, lambda are empty
    # u1 = all image indices
    # u2 = all point indices
```

**Common failure causes:**
- < 2 images with valid geometry
- Insufficient point correspondences
- Too much missing data
- Degenerate camera configuration

---

## Error Computation

### Balanced Error
```python
error = eucl_dist_only(B_subset, P@X, valid_mask, step=3)
```

**Measures:**
- Euclidean distance between measurements and reconstruction
- Only on valid (non-NaN) entries
- Step=3 treats row triplets as single points

**Interpretation:**
- Small error (< 1): Good reconstruction
- Medium error (1-10): Acceptable
- Large error (> 10): Poor quality

---

## Lambda Merging Strategy

### Initial Lambda (from depth_estimation)
```python
lambda_init[i,j] where Ilamb[i,j] = 1
```

Known from epipolar geometry.

### Refined Lambda (from fill_prmm)
```python
lambda1[i,j] where Ilamb[i,j] = 0
```

Computed during factorization.

### Final Lambda
```python
if Ilamb[i,j]:
    lambda[i,j] = lambda_init[i,j]  # Keep original
else:
    lambda[i,j] = lambda1[i,j]      # Use refined
```

Combines both sources intelligently.

---

## rows vs nonrows

### rows
Images with valid epipolar geometry:
- At least 8 correspondences
- Fundamental matrix computable
- Used in reconstruction

### nonrows
Images without valid geometry:
- Too few correspondences
- Degenerate configuration
- Excluded from reconstruction
- Added to u1 at the end

---

## Index Conversions

### Local to Global
```python
# fill_prmm returns u1, u2 in local coordinates
# Need to map back to original image indices

u1_local = [0, 2]       # Failed images in rows
rows = [1, 3, 5, 7, 9]  # Used images

u1_global = rows[u1_local] = [1, 5]  # In original numbering
```

### Final u1
```python
u1_final = union(nonrows, u1_global)
```

Includes both excluded and failed images.

---

## Visualization Preparation

### info.show_prmm Structure
```python
{
    'I': (m × n),        # Visibility
    'Idepths': (m × n)   # Depth validity
}
```

**I[i,j]:**
- `True`: Point j visible in image i
- `False`: Point j not visible

**Idepths[i,j]:**
- `1`: Depth known from epipolar geometry
- `0`: Depth unknown (may be computed)

**Usage:** Visualization tools can show which depths were original vs computed.

---

## Performance

### Complexity
- **M2Fe**: O(m² × n)
- **depth_estimation**: O(m × n)
- **balance_triplets**: O(iterations × m × n)
- **fill_prmm**: Dominant (null-space creation)
- **Total**: Dominated by fill_prmm

### Memory
- **Mfull, M, B**: Each 3m × n
- **lambda, Ilamb**: Each m × n
- **P, X**: Much smaller (3k × 4, 4 × n')
- Peak during null-space creation

---

## Debugging Tips

### Empty Output
**Cause:** `len(rows) < 2`
**Check:**
- Are there enough correspondences in Mfull?
- Try different central image
- Check data quality

### High Error
**Cause:** Poor reconstruction
**Check:**
- Balance not converging?
- Insufficient 4-tuples in fill_prmm?
- Increase trial_coef

### Many Unrecovered
**Cause:** Sparse data, weak geometry
**Check:**
- Visibility pattern
- Camera motion (avoid planar)
- Try sequence vs central mode

---

## Sequence vs Central Mode

### Sequence Mode (central=0)
**Pro:**
- Uses all pairwise geometry
- Better for video sequences
- More robust to missing data

**Con:**
- Can accumulate drift
- No single reference frame

### Central Mode (central=k)
**Pro:**
- Single reference frame
- Consistent scale
- Good for convergent cameras

**Con:**
- Requires good central image
- May exclude some images
- Needs sufficient correspondences to central

---

## Advanced Features

### Dual Matrix Input
```python
fill_mm_sub(Mfull, M, ...)
```

Allows:
- Using more complete data for F matrices
- Reconstructing sparser subset
- Handling different quality levels

### Lambda Merging
Intelligently combines:
- Epipolar-based depths (reliable)
- Factorization-based depths (fill gaps)

### Error Reporting
Provides reconstruction quality metric.

---

## Notes

- **Requires ≥2 images** with valid geometry
- **Rescaling is crucial** for projective consistency
- **Balancing improves** numerical stability
- **Lambda merging** gives best of both worlds
- **info structure** enables visualization
- **Modular design** uses 5 converted functions

---

## Error Handling

### Function doesn't raise errors
Returns empty results on failure.

**Check success:**
```python
P, X, lambda_vals, u1, u2, info = fill_mm_sub(...)

if P.size == 0:
    print("Reconstruction failed")
    print(f"Lost: {len(u1)} images, {len(u2)} points")
else:
    print("Success!")
```

### Partial Success
```python
if len(u1) > 0:
    print(f"Warning: {len(u1)} images not recovered")
```

---

## Testing

The module includes comprehensive tests:
```bash
python fill_mm_sub.py
```

Tests both sequence and central modes with synthetic data.
