# u2FI Function - Step-by-Step Explanation

## What It Does
Estimates the **Fundamental Matrix (F)** between two camera views using the 8-point algorithm.

The fundamental matrix relates corresponding points: `u2^T * F * u1 = 0`

---

## Step-by-Step Process

### Step 1: Validate Input
```python
valid_mask = np.sum(~np.isnan(u[0::3, :]), axis=0) == 2
sampcols = np.where(valid_mask)[0]
if len(sampcols) < 8:
    return 0
```
- Finds columns with valid (non-NaN) point pairs
- Requires minimum 8 point correspondences
- Returns 0 if insufficient points

### Step 2: Normalize Points (Optional)
```python
if donorm:
    A1 = Utils.normu(u[0:3, sampcols])
    A2 = Utils.normu(u[3:6, sampcols])
    u1 = A1 @ u[0:3, sampcols]
    u2 = A2 @ u[3:6, sampcols]
```
- Transforms points to have centroid at origin
- Scales to average distance √2 from origin
- **Critical for numerical stability!**

### Step 3: Build Constraint Matrix
```python
Z = np.zeros((ptNum, 9))
for i in range(ptNum):
    Z[i, :] = np.outer(u1[:, i], u2[:, i]).flatten()
```
- Each row contains Kronecker product of point pair
- For points [x1, y1, 1] and [x2, y2, 1], creates:
  `[x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1]`

### Step 4: Solve Eigenproblem
```python
M = Z.T @ Z
V, d = seig(M)
F = V[:, 0].reshape(3, 3)
```
- Computes M = Z^T * Z
- Finds eigenvector with smallest eigenvalue
- This minimizes ||Z*f||² subject to ||f||=1
- Reshapes into 3×3 matrix

### Step 5: Enforce Rank-2 Constraint
```python
uu, us, vt = np.linalg.svd(F)
us[2] = 0
F = uu @ np.diag(us) @ vt
```
- Fundamental matrix must have rank 2 (det(F) = 0)
- Sets smallest singular value to zero
- Reconstructs F from modified SVD

### Step 6: Denormalize
```python
if donorm or normalization == 'usenorm':
    F = A1.T @ F @ A2
```
- Transforms F back to original coordinate system
- If normalized: F_original = A1^T * F_normalized * A2

### Step 7: Normalize by Norm
```python
F = F / np.linalg.norm(F, 'fro')
```
- Scales F so Frobenius norm = 1
- Standard convention (F is defined up to scale)

### Step 8: Final Rank Check
```python
if np.linalg.matrix_rank(F, tol=1e-6) > 2:
    uu, us, vt = np.linalg.svd(F)
    us[2] = 0
    F = uu @ np.diag(us) @ vt
```
- Ensures rank is still 2 after normalization
- Re-applies rank-2 constraint if needed

---

## Key MATLAB → Python Conversions

| MATLAB | Python | Description |
|--------|--------|-------------|
| `u(1:3,:)` | `u[0:3,:]` | 0-based indexing |
| `A*B` | `A @ B` | Matrix multiplication |
| `A'` | `A.T` | Transpose |
| `[U,S,V] = svd(F)` | `U, s, Vt = np.linalg.svd(F)` | SVD (note: V is transposed) |
| `isempty(A)` | `A.size == 0` | Empty array check |
| `reshape(v,3,3)` | `v.reshape(3, 3)` | Reshape array |

---

## Usage Example

```python
from Utils import Utils
from u2FI import u2FI
import numpy as np

# Point correspondences (6×N array)
# Rows 0-2: image 1 points [x1, y1, 1]
# Rows 3-5: image 2 points [x2, y2, 1]
u = np.array([
    [100, 150, 200, 250, 300, 350, 400, 450],  # x1
    [120, 170, 180, 220, 240, 280, 300, 320],  # y1
    [1,   1,   1,   1,   1,   1,   1,   1],    # w1
    [110, 160, 210, 260, 310, 360, 410, 460],  # x2
    [125, 175, 185, 225, 245, 285, 305, 325],  # y2
    [1,   1,   1,   1,   1,   1,   1,   1]     # w2
])

# Estimate fundamental matrix
F = u2FI(u)

if not isinstance(F, int):
    print("Success!")
    print(f"F = \n{F}")
    
    # Verify epipolar constraint (should be ≈0)
    for i in range(u.shape[1]):
        residual = u[3:6, i].T @ F @ u[0:3, i]
        print(f"Point {i}: {residual:.2e}")
```

---

## Function Parameters

### u2FI(u, normalization='norm', A1=None, A2=None)

**Args:**
- `u`: (6, N) array of point correspondences
- `normalization`: 
  - `'norm'` - apply normalization (default, recommended)
  - `'nonorm'` - skip normalization
  - `'usenorm'` - use provided A1, A2 matrices
- `A1`, `A2`: Optional normalization matrices (3×3)

**Returns:**
- `F`: (3×3) fundamental matrix if successful
- `0`: if failed (insufficient points or degenerate)

---

## Why Normalization Matters

Without normalization, points like (0-1000 pixels) create ill-conditioned systems:
- Matrix elements vary by orders of magnitude
- Numerical errors dominate
- Results are unreliable

With normalization:
- Points centered at origin
- Average distance = √2
- Matrix elements have similar scales
- **Accuracy improves by 10-100×**

---

## Mathematical Background

The fundamental matrix satisfies:
```
u2^T * F * u1 = 0
```

Expanding:
```
x2*(F11*x1 + F12*y1 + F13) + 
y2*(F21*x1 + F22*y1 + F23) + 
   (F31*x1 + F32*y1 + F33) = 0
```

For n points, we get system `Z*f = 0` where f = vec(F).

Solution: eigenvector of Z^T*Z with smallest eigenvalue.

Properties of F:
- 3×3 matrix
- Rank 2 (det(F) = 0)
- 7 degrees of freedom
- Defined up to scale

---

## Common Issues

**"Need at least 8 points"**
- Provide ≥8 point correspondences
- Each point pair gives one constraint

**"Normalization failed"**
- All points at same location
- Check input data validity

**Large residuals**
- Points have outliers → use RANSAC
- Points poorly distributed → use better features
- Degenerate configuration → check camera setup

---

## References

1. Hartley & Zisserman, "Multiple View Geometry in Computer Vision"
2. Hartley, "In defense of the eight-point algorithm", IEEE TPAMI, 1997
