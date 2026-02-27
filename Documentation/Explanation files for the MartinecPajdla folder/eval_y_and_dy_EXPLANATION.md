# eval_y_and_dy Function - Explanation

## What It Does
Evaluates **bundle adjustment objective function** and its **Jacobian**:
- Computes residuals between predicted and observed image points
- Optionally computes sparse Jacobian for optimization
- Supports radial distortion

Used in non-linear least squares optimization for camera calibration and 3D reconstruction.

---

## Input/Output

### Input
- `p`: Parameter vector ((11+NR)×K + 3×N)
  - Camera parameters: `[iP(1); u0(1); kappa(1); ... iP(K); ...]`
  - Point parameters: `[iX(1); iX(2); ... iX(N)]`
  - NR = 3 if RADIAL, 0 otherwise
- `P0`: Base camera matrices (3K × 4)
- `TP`: List of K camera transformations (12 × 11 each)
- `X0`: Base 3D points (4 × N)
- `TX`: List of N point transformations (4 × 3 each)
- `y`: Observed image points (2×n_visible)
- `qivis`: Visibility matrix (K × N), 1=visible, 0=hidden
- `RADIAL`: Include radial distortion (bool)

### Output
- `y_residual`: Residual vector (predicted - observed)
- `J`: Sparse Jacobian matrix (optional)

---

## Imaging Model

### Forward Model
```
X = X0 + TX @ iX          (3D point)
P = P0 + TP @ iP          (camera matrix)
x = P @ X                 (projection to 3D)
u = p2e(x) = [x/z, y/z]  (to 2D)
q = raddist(u, u0, kappa) (radial distortion)
```

### Residuals
```
residual = q_predicted - q_observed
```

---

## Step-by-Step Process

### Step 1: Extract Parameters
```python
for k in range(K):
    iP_k = p[k*(11+NR):(k+1)*(11+NR)-NR]
    P[k] = P0[k] + TP[k] @ iP_k
    
    if RADIAL:
        u0[k] = p[k*(11+NR)+11:k*(11+NR)+13]
        kappa[k] = p[k*(11+NR)+13]
```

Extracts camera and distortion parameters.

### Step 2: Compute 3D Points
```python
for n in range(N):
    iX_n = p[(11+NR)*K + n*3:(11+NR)*K + (n+1)*3]
    X[:, n] = X0[:, n] + TX[n] @ iX_n
```

Computes 3D point positions.

### Step 3: Project to Images
```python
for k in range(K):
    x[k] = P[k] @ X          # 3D projection
    u[k] = p2e(x[k])         # to 2D
    
    if RADIAL:
        q[k] = raddist(u[k], u0[k], kappa[k])
```

Projects 3D points to 2D images with optional distortion.

### Step 4: Compute Residuals
```python
q_visible = q[qivis]
residual = q_visible - y_observed
```

Extracts visible points and computes residuals.

### Step 5: Compute Jacobian (Optional)
For each visible point:
```python
dxdP = kron(X.T, I₃) @ TP    # dx/diP
dxdX = P @ TX                 # dx/diX
dudx = [I₂ | -u] / z          # du/dx

if RADIAL:
    dqdu, dqdu0, dqdkappa = raddist_deriv(...)
else:
    dqdu = I₂

dqdP = dqdu @ dudx @ dxdP    # Chain rule
dqdX = dqdu @ dudx @ dxdX
```

Uses chain rule to compute derivatives.

---

## Radial Distortion

### Model
```
q = u0 + (1 + κ·r²)·(u - u0)
```

Where:
- `u` = undistorted point
- `u0` = distortion center
- `κ` = radial distortion coefficient
- `r² = ||u - u0||²`

### Forward Function (raddist_apply)
```python
du = u - u0
r2 = du.T @ du
q = u0 + (1 + kappa * r2) * du
```

### Derivatives (raddist_deriv)

**dq/du:**
```python
f = 1 + kappa * r²
dqdu = f·I + 2·kappa·(du ⊗ du)
```

**dq/du0:**
```python
dqdu0 = I - f·I - 2·kappa·(du ⊗ du)
```

**dq/dκ:**
```python
dqdkappa = r² · du
```

---

## Parameter Vector Structure

### Without Radial (NR=0)
```
p = [iP(1)      | iP(2)      | ... | iP(K)      | iX(1)  | ... | iX(N)  ]
    [11 params  | 11 params  | ... | 11 params  | 3 par  | ... | 3 par  ]
    
Total: 11K + 3N
```

### With Radial (NR=3)
```
p = [iP(1), u0(1), κ(1) | iP(2), u0(2), κ(2) | ... | iX(1)  | ... | iX(N)  ]
    [11, 2, 1 params    | 11, 2, 1 params    | ... | 3 par  | ... | 3 par  ]
    
Total: 14K + 3N
```

---

## Jacobian Structure

### Sparse Matrix
- **Rows:** 2 × n_visible (one row per coordinate)
- **Columns:** len(p)
- **Non-zeros:** ~(11+NR+3) per visible point

### Block Structure
```
For visible point (k, n):
  Rows [2l:2l+2] correspond to this point
  Cols [(11+NR)*k:(11+NR)*(k+1)] → dq/diP_k
  Cols [(11+NR)*K + 3*n:(11+NR)*K + 3*(n+1)] → dq/diX_n
  
  If RADIAL:
    Cols [(11+NR)*k+11:(11+NR)*k+13] → dq/du0_k
    Cols [(11+NR)*k+13] → dq/dκ_k
```

---

## Key MATLAB → Python Conversions

| MATLAB | Python | Description |
|--------|--------|-------------|
| `k2i(k)` | `3*k:3*k+3` or `Utils.k2i(k, step=3)` | Row triplet indices |
| `nhom(x)` | `Utils.p2e(x)` | Homogeneous to Euclidean |
| `kron(A,B)` | `np.kron(A,B)` | Kronecker product |
| `reshape(...,[3 4])` | `.reshape(3, 4)` | Reshape |
| `sparse(i,j,v,m,n)` | `csr_matrix((v,(i,j)), shape=(m,n))` | Sparse matrix |
| `nargout` | Return tuple | Multiple returns |
| `find(qivis)` | `np.where(qivis)` | Find indices |

---

## Usage Example

```python
from eval_y_and_dy import eval_y_and_dy
import numpy as np

# Setup bundle adjustment problem
K, N = 5, 100  # 5 cameras, 100 points
p = np.random.randn(14*K + 3*N) * 0.01  # Initial parameters
y_obs = ...  # Observed image points
qivis = ...  # Visibility matrix

# Evaluate residuals and Jacobian
residual, J = eval_y_and_dy(
    p, P0, TP, X0, TX, y_obs, qivis, RADIAL=True
)

# Use in optimization
# scipy.optimize.least_squares(eval_y_and_dy, p, jac=True, ...)
```

---

## Common Use Cases

### 1. Bundle Adjustment
```python
# Refine camera and point parameters
def objective(p):
    residual, J = eval_y_and_dy(p, ...)
    return residual, J

result = least_squares(objective, p0, jac=True)
```

### 2. Camera Calibration
```python
# Fix 3D points, optimize cameras only
# (subset of parameters)
```

### 3. Structure Refinement
```python
# Fix cameras, optimize 3D points only
# (different subset of parameters)
```

---

## Mathematical Background

### Bundle Adjustment
Minimizes reprojection error:
```
min Σ ||q_observed - project(P, X)||²
```

Over camera parameters P and point parameters X.

### Non-linear Least Squares
```
min ||f(p)||²

Update: p ← p - (J^T J)^(-1) J^T f(p)
```

Where J is the Jacobian.

### Chain Rule for Derivatives
```
dq/dp = (dq/du)(du/dx)(dx/dP)(dP/dp)
```

Each derivative computed separately then multiplied.

---

## Numerical Stability

### Homogeneous Coordinates
```python
u = [x/z, y/z] instead of [x, y]
```
Handles points at infinity gracefully.

### Sparse Jacobian
```python
scipy.sparse.csr_matrix
```
Only stores non-zero elements → memory efficient.

### Numerical Gradients Check
```python
# Finite difference
dq_numerical = (q(p+eps) - q(p)) / eps

# Should match analytical
error = ||dq_numerical - dq_analytical||
```

Our tests show error < 1e-6 ✓

---

## Performance

### Complexity
- **Function evaluation:** O(K×N) projections
- **Jacobian computation:** O(n_visible × (11+NR+3))

### Optimization
- Use sparse Jacobian (most entries are zero)
- Only compute Jacobian when needed
- Vectorize over points when possible

---

## Integration with Other Functions

### Typical Workflow
```python
# 1. Initialize from structure-from-motion
P0, X0 = initialize_from_sfm(...)

# 2. Setup transformations
TP = [make_camera_transform(k) for k in range(K)]
TX = [make_point_transform(n) for n in range(N)]

# 3. Bundle adjustment
p_opt = optimize(eval_y_and_dy, p0, ...)

# 4. Extract final cameras and points
P_final = extract_cameras(p_opt, P0, TP)
X_final = extract_points(p_opt, X0, TX)
```

---

## Parameter Transformations

### Why TP and TX?

Instead of optimizing P and X directly:
```
P = P0 + TP @ iP
X = X0 + TX @ iX
```

**Benefits:**
1. **Constraints:** TP/TX can enforce structure
2. **Parameterization:** Reduce from 12 to 11 DoF for P
3. **Initialization:** Start near good solution P0, X0

### Camera Parameterization
- 12 elements in 3×4 matrix
- But only 11 degrees of freedom (scale ambiguity)
- TP: 12×11 removes one DoF

### Point Parameterization
- 4 elements in homogeneous coordinates
- But only 3 degrees of freedom
- TX: 4×3 handles homogeneous properly

---

## Radial Distortion Notes

### Brown's Model
This implements radial-only distortion (1 parameter κ).

Full Brown's model includes:
- Radial: κ₁r² + κ₂r⁴ + κ₃r⁶
- Tangential: p₁(r²+2u²) + 2p₂uv

This function uses simplified: κr²

### When to Use
- **Include RADIAL:** Real cameras, wide-angle lenses
- **Skip RADIAL:** Synthetic data, telecentric lenses, pre-undistorted images

---

## Error Handling

### Missing Observations
```python
qivis[k, n] = 0  # Point n not visible in camera k
# Automatically excluded from residuals and Jacobian
```

### Degenerate Cases
- Empty visibility → residual size 0
- Zero parameters → identity transform
- Numerical issues → check condition number of J^T J

---

## Debugging Tips

1. **Check residual size:** Should be 2 × sum(qivis)
2. **Check Jacobian size:** (residual_size, param_size)
3. **Numerical gradient:** Compare analytical vs finite difference
4. **Visualize:** Plot residuals, check for outliers
5. **Conditioning:** eigenvalues(J^T J) should not span >10 orders

---

## Notes

- Function can be called with/without Jacobian computation
- Sparse Jacobian essential for large problems (K>10, N>100)
- Radial distortion adds 3 parameters per camera
- Parameters should be scaled appropriately for optimization
- Initial guess p should be close to optimum for convergence
