# depth_estimation Function - Explanation

## What It Does
Computes **scale factors (projective depths)** for 3D reconstruction from multiple views. These scale factors λ allow recovering 3D structure from 2D measurements.

---

## Input/Output

### Input
- `M`: (3m × n) measurements array (m images, n points)
- `F`: Dictionary of fundamental matrices (from M2Fe)
- `ep`: Dictionary of epipoles (from M2Fe)
- `rows`: Valid image indices (from M2Fe)
- `central`: 0 = sequence mode, >0 = central image number

### Output
- `lambda`: (m × n) array of scale factors
- `Ilamb`: (m × n) boolean array of valid depths

---

## What Are Projective Depths?

In projective reconstruction, a 3D point **X** projects to image point **u**:
```
u = λ * P * X
```

Where:
- **P** = camera projection matrix
- **λ** = scale factor (projective depth)

The scale factor is needed because we can only recover structure up to a projective transformation without calibration.

---

## Step-by-Step Process

### Step 1: Initialize Reference Image
```python
if central > 0:
    j = central  # Central image as reference
    Ilamb[j, :] = ~isnan(M[3*j, :])
else:
    j = 0  # First image as reference
    b = subseq_longest(valid_mask)
```

**Sequence mode:** Reference changes for each image  
**Central mode:** One fixed reference image

### Step 2: Set Reference Depths to 1
```python
lambda_vals = np.ones((m, n))
```
Reference image always has λ = 1 (defines scale)

### Step 3: Compute Depths for Other Images
```python
for i in images_except_reference:
    if not central:
        j = i - 1  # Previous image in sequence
    
    G = F[(i, j)]      # Fundamental matrix
    epip = ep[(i, j)]   # Epipole
```

For each image i relative to reference j:

### Step 4: Compute Scale Factor per Point
```python
for each point p:
    u = cross(epipole, point_i)  # Epipolar line
    v = G @ point_j               # Transformed point
    
    λ_i = (u^T @ v / ||u||²) * λ_j
```

**Key formula:**
```
λ_i(p) = (u^T · v) / ||u||² × λ_j(p)
```

Where:
- **u** = epipole × point_i (epipolar line direction)
- **v** = F × point_j (corresponding epipolar constraint)

### Step 5: Handle Invalid Points
```python
if not valid:
    lambda_vals[i, p] = 1.0  # Default for recovery
```
Points without correspondences get λ = 1

---

## Key Equations

### Epipolar Constraint
```
u_j^T @ F @ u_i = 0
```

### Depth Relation
The depth relationship comes from:
```
λ_i × u_i = P_i @ X
λ_j × u_j = P_j @ X
```

Combined with epipolar geometry:
```
λ_i = (e × u_i)^T @ (F @ u_j) / ||e × u_i||² × λ_j
```

Where **e** is the epipole.

---

## Usage Example

```python
from depth_estimation import depth_estimation
from M2Fe import M2Fe
import numpy as np

# Multi-view measurements
M = ...  # (3*m, n) array

# Get epipolar geometry
F, ep, rows, nonrows = M2Fe(M, central=0)

# Compute depths
lambda_vals, Ilamb = depth_estimation(M, F, ep, rows, central=0)

# Use depths for 3D reconstruction
for i in range(m):
    for p in range(n):
        if Ilamb[i, p]:
            # Scale the 2D point
            X_proj = lambda_vals[i, p] * M[3*i:3*i+3, p]
            print(f"Point {p} in image {i}: {X_proj}")
```

---

## Two Modes Explained

### Sequence Mode (central=0)
```
Image 0 (ref) → Image 1 → Image 2 → Image 3
  λ=1           λ_1        λ_2        λ_3
```
- Each image uses previous as reference
- Depths accumulate through sequence
- Good for video sequences

### Central Mode (central=k)
```
Image 0 ←→ Image k (ref) ←→ Image 2
  λ_0        λ=1              λ_2
           ↕
        Image 3
          λ_3
```
- All images use central image as reference
- Independent depth estimates
- Good for hub-spoke configurations

---

## Output Arrays

### lambda (m × n)
- Scale factors for each point in each image
- Reference image always has λ = 1
- Other images have λ relative to reference

**Example:**
```
lambda = [[1.0,  1.0,  1.0],    # Image 0 (ref)
          [0.85, 0.82, 0.88],   # Image 1
          [0.71, 0.69, 0.75]]   # Image 2
```

### Ilamb (m × n)
- Boolean indicating valid depth estimates
- False if point missing or computation failed

**Example:**
```
Ilamb = [[True,  True,  True],   # All valid
         [True,  False, True],   # Point 1 missing
         [True,  True,  True]]
```

---

## Key MATLAB → Python Conversions

| MATLAB | Python | Note |
|--------|--------|------|
| `M(3*i-2:3*i,p)` | `M[3*i:3*i+3,p]` | 0-based indexing |
| `F(i,j,1:3,1:3)` | `F[(i,j)]` | Dictionary access |
| `ep(i,j,1:3)` | `ep[(i,j)]` | Dictionary access |
| `cross(a,b)` | `np.cross(a,b)` | Cross product |
| `norm(u)^2` | `np.linalg.norm(u)**2` | Squared norm |
| `u'*v` | `u.T @ v` | Dot product |
| `ones(m,n)` | `np.ones((m,n))` | Initialization |
| `isnan(x)` | `np.isnan(x)` | NaN check |

---

## Integration with Other Functions

### Requires M2Fe Output
```python
F, ep, rows, _ = M2Fe(M, central)
lambda_vals, Ilamb = depth_estimation(M, F, ep, rows, central)
```

The F and ep dictionaries from M2Fe are used directly.

### Uses Utils.subseq_longest
```python
b, _ = Utils.subseq_longest(valid_mask)
```
Finds longest valid subsequence for sequence mode.

---

## Common Use Cases

### 1. 3D Reconstruction
```python
# Compute depths
lambda_vals, Ilamb = depth_estimation(M, F, ep, rows, 0)

# Reconstruct 3D points
for p in range(n_points):
    for i in range(n_images):
        if Ilamb[i, p]:
            X_i = lambda_vals[i, p] * M[3*i:3*i+3, p]
            # X_i is now scaled correctly
```

### 2. Missing Data Handling
```python
# Ilamb tells which depths are valid
valid_points = np.where(Ilamb[image_idx, :])[0]
lambda_valid = lambda_vals[image_idx, valid_points]
```

### 3. Multi-View Consistency
```python
# Check depth consistency across views
for p in range(n_points):
    depths = [lambda_vals[i, p] for i in range(m) if Ilamb[i, p]]
    std = np.std(depths)
    if std > threshold:
        print(f"Point {p} has inconsistent depths")
```

---

## Mathematical Background

### Projective Reconstruction
Without camera calibration, we can only recover 3D structure up to a projective transformation:
```
X_proj = H @ X_true
```

Where H is a 4×4 projective transformation.

### Role of Scale Factors
The scale factors λ encode the depth ambiguity:
```
λ_i × u_i ≈ P_i @ X_proj
```

### Epipolar Geometry Connection
The fundamental matrix F relates λ values between views:
```
λ_i = f(λ_j, F, epipole, points)
```

This function computes this relationship.

---

## Error Cases

### No Valid Correspondences
```python
if not Ilamb[i, p]:
    lambda_vals[i, p] = 1.0
```
Default to 1 to allow later scene recovery.

### Degenerate Configuration
```python
if u_norm_sq < 1e-10:
    lambda_vals[i, p] = 1.0
```
If epipolar line is degenerate, use default.

### Missing F Matrix
```python
if (i, j) not in F:
    continue
```
Skip if fundamental matrix wasn't estimated.

---

## Notes

- Reference image always has λ = 1 (defines scale)
- Sequence mode: depths propagate through sequence
- Central mode: depths independent for each image
- Invalid points get λ = 1 for later recovery
- Cross product computes epipolar line direction
- Fundamental matrix transforms between views
