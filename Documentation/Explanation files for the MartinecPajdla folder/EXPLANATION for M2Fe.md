# M2Fe Function - Explanation

## What It Does
Estimates **fundamental matrices** and **epipoles** for multiple views, either:
- **Sequence mode**: Consecutive image pairs (1-2, 2-3, 3-4, ...)
- **Central mode**: All images vs one central image (1-c, 2-c, 3-c, ...)

---

## Input/Output

### Input
- `M`: (3m × N) array where m = number of images, N = number of points
  - Each image occupies 3 rows (homogeneous coordinates)
  - M[0:3, :] = image 0, M[3:6, :] = image 1, etc.
- `central`: 
  - 0 = sequence mode
  - >0 = central image mode (image number)

### Output
- `F`: Dictionary {(i,j): F_matrix} - fundamental matrices
- `ep`: Dictionary {(i,j): epipole} - epipoles
- `rows`: Valid image indices
- `nonrows`: Failed image indices

---

## Step-by-Step Process

### Step 1: Determine Image Pairs
```python
if central > 0:
    rows = [0..central-1, central+1..m]  # All except central
else:
    rows = [1..m]  # From second image onwards
```

### Step 2: Estimate F for Each Pair
```python
for k in rows:
    j = central if central > 0 else k-1
    u_pair = stack(M[3*k:3*k+3], M[3*j:3*j+3])
    G = u2FI(u_pair)
```
- Extracts point correspondences for images k and j
- Calls `u2FI()` to estimate fundamental matrix
- If fails, marks image k as invalid

### Step 3: Compute Epipole
```python
u, s, vt = svd(G)
epipole = u[:, 2]  # Last column (smallest singular value)
```
- Epipole is null space of F^T (right null space of F)
- Use SVD to find it robustly

### Step 4: Store Results
```python
F[(k, j)] = G
ep[(k, j)] = epipole
```
- Dictionary keys are image pair indices

### Step 5: Post-Processing
```python
if central > 0:
    rows = union(rows, [central])  # Add central back
else:
    rows = [0] + rows  # Add first image
```

### Step 6: Handle Failures (Sequence Mode Only)
```python
if nonrows and central == 0:
    # Find longest continuous valid subsequence
    b, length = subseq_longest(indicator_array)
    rows = [b, b+1, ..., b+length-1]
```
- If some images failed, keep only longest continuous run
- Ensures temporal coherence in sequence

---

## Key MATLAB → Python Conversions

| MATLAB | Python | Note |
|--------|--------|------|
| `M(3*k-2:3*k,:)` | `M[3*k:3*k+3,:]` | 0-based indexing |
| `[1:n]` | `np.arange(n)` | Range |
| `[arr1, arr2]` | `np.concatenate([arr1, arr2])` | Concatenate |
| `setdiff(a,b)` | `np.setdiff1d(a,b)` | Set difference |
| `union(a,b)` | `np.union1d(a,b)` | Set union |
| `F(k,j,1:3,1:3)` | `F[(k,j)]` = matrix | Dictionary |

---

## Usage Examples

### Example 1: Sequence Mode
```python
from M2Fe import M2Fe
import numpy as np

# M has 5 images (15 rows), 50 points
M = ...  # Your multi-view data

# Estimate F for consecutive pairs
F, ep, rows, nonrows = M2Fe(M, central=0)

# Access fundamental matrix between images 2 and 1
F_21 = F[(2, 1)]
epipole_21 = ep[(2, 1)]

print(f"Valid images: {rows}")
print(f"Failed images: {nonrows}")
```

### Example 2: Central Image Mode
```python
# Compare all images to image 2
F, ep, rows, nonrows = M2Fe(M, central=2)

# Access fundamental matrix between images 0 and 2
F_02 = F[(0, 2)]
epipole_02 = ep[(0, 2)]
```

---

## Return Value Structure

### F Dictionary
Keys: `(k, j)` tuples  
Values: 3×3 fundamental matrices

**Sequence mode:**
- F[(1, 0)], F[(2, 1)], F[(3, 2)], ...

**Central mode (central=2):**
- F[(0, 2)], F[(1, 2)], F[(3, 2)], F[(4, 2)], ...

### ep Dictionary
Same key structure as F  
Values: 3×1 epipole vectors

---

## What Are Epipoles?

The **epipole** is the projection of one camera center onto the other camera's image plane.

For images i and j with fundamental matrix F:
- Epipole in image j: `F @ e_i = 0` (left null space)
- Epipole in image i: `F^T @ e_j = 0` (right null space)

This function computes the **right null space** epipole using SVD.

---

## Common Use Cases

### 1. Structure from Motion
Build 3D reconstruction from image sequence:
```python
F, ep, rows, _ = M2Fe(M, central=0)
# Use F matrices to triangulate 3D points
```

### 2. Camera Calibration
Estimate camera matrices from fundamental matrices:
```python
F, ep, rows, _ = M2Fe(M, central=reference_image)
# Use F and epipoles to recover camera matrices
```

### 3. Quality Check
Find which images have good correspondences:
```python
F, ep, rows, nonrows = M2Fe(M, central=0)
print(f"Good images: {len(rows)}")
print(f"Bad images: {len(nonrows)}")
```

---

## Integration with Other Functions

### Uses u2FI
```python
G = u2FI(u_pair, normalization='norm')
```
- Estimates fundamental matrix for each pair
- Returns 0 if insufficient points

### Uses Utils.subseq_longest
```python
b, length = Utils.subseq_longest(I)
```
- Finds longest continuous subsequence of valid images
- Only in sequence mode with failures

---

## Error Handling

### All Images Fail
- `F = {}` (empty dictionary)
- `ep = {}` (empty dictionary)
- `rows = []` (empty array)
- `nonrows = [0,1,2,...,m]` (all images)

### Some Images Fail (Sequence Mode)
- Function finds longest continuous valid subsequence
- Returns only images in that subsequence
- Other images go to `nonrows`

### Some Images Fail (Central Mode)
- Invalid images go to `nonrows`
- Valid images stay in `rows`
- Central image always in `rows`

---

## Mathematical Background

For two views with fundamental matrix **F**:
```
u_j^T @ F @ u_i = 0
```

Where:
- u_i, u_j are corresponding points in images i and j
- F encodes the epipolar geometry

The epipole e_j satisfies:
```
F^T @ e_j = 0
```

Computed via SVD: F = U S V^T, then e_j = U[:, 2] (last column)

---

## Notes

- Requires ≥8 point correspondences per image pair
- Uses Hartley normalization (via u2FI) for numerical stability
- Sequence mode ensures temporal coherence by finding longest valid run
- Dictionary keys use 0-based Python indexing
