# bundle_PX_proj Function - Explanation

## What It Does
**Projective bundle adjustment with image-based conditioning:**
1. Applies image conditioning (centering + scaling)
2. Normalizes cameras and points
3. Sets up tangent space parameterization  
4. Optimizes using Levenberg-Marquardt
5. Undoes conditioning and returns refined P, X

**Key difference from qPXbundle_cmp:** Uses image-specific preconditioning for better numerical stability.

---

## Input/Output

### Input
- `P0`: Initial camera matrices (3K × 4)
- `X0`: Initial 3D points (4 × N)
- `q`: Observations (2K × N) in Euclidean coordinates
- `imsize`: Image sizes (2 × K), `imsize[:, k]` = [width, height]
- `nl_params_all_cams`: Radial distortion (optional, not implemented)
- `opt`: Options dict with:
  - `'verbose'` (1): Print progress
  - `'verbose_short'` (0): Print condensed progress
  - `'res_scale'` (1): Scale for printed residuals
  - `'max_niter'` (10000): Max iterations
  - `'max_stepy'` (100*eps): Convergence threshold
  - `'lam_init'` (1e-4): Initial λ (damping)

### Output
- `P`: Optimized camera matrices (3K × 4)
- `X`: Optimized 3D points (4 × N)

**Note:** Preconditioning is automatic - no manual preparation needed!

---

## Algorithm Overview

### Main Steps
```
1. Image conditioning: H_k for each camera k
2. Apply: P0 → H @ P0, q → H @ q
3. Normalize: P, X to unit norm
4. Parameterize: Tangent space (TP, TX)
5. Optimize: Levenberg-Marquardt
6. Undo conditioning: P → inv(H) @ P
7. Return: Refined P, X
```

### Levenberg-Marquardt
```
while not converged:
    J = Jacobian
    D = -(J^T J + λI)^(-1) J^T r
    if improves:
        p ← p + D
        λ ← λ / 10
    else:
        λ ← λ * 10
```

---

## Step-by-Step Process

### Step 1: Image Conditioning
```python
for k in range(K):
    H_k = vgg_conditioner_from_image(imsize[:, k])
    P0[k] = H_k @ P0[k]
    q[k] = p2e(H_k @ hom(q[k]))
```

**Why condition?**
- Centers image at origin
- Scales to unit size
- Improves numerical stability

### Step 2: Normalize
```python
P0 = normP(P0)  # ||P_k||_F = 1
X0 = normx(X0)  # ||X_n|| = 1
```

Further numerical conditioning.

### Step 3: Form Observations
```python
qq = q.reshape(2, K*N)
qivis = all(~isnan(qq))  # Visibility
y = qq[qivis].flatten()  # Observed points only
```

### Step 4: Tangent Space Parameterization
```python
# For points
for n in range(N):
    Q, R = qr(X0[:, n])
    TX[n] = Q[1:, :].T  # (4 × 3)

# For cameras
for k in range(K):
    Q, R = qr(P0[k].flatten())
    TP[k] = Q[1:, :].T  # (12 × 11)
```

**Parameterization:**
```
X = X0 + TX @ iX  (iX is 3×1)
P = P0 + TP @ iP  (iP is 11×1)
```

Removes scale ambiguity.

### Step 5: Initialize Parameters
```python
p0 = [zeros(11) for k in K] + [zeros(3*N)]
```

Start at current estimate (P0, X0).

### Step 6: Levenberg-Marquardt Optimization
```python
while not converged:
    r, J = objective(p)
    D = -(J^T J + λI)^(-1) J^T r
    
    if ||r(p+D)||² < ||r(p)||²:
        p = p + D
        λ = λ / 10
    else:
        λ = λ * 10
```

### Step 7: Extract Results
```python
for k in range(K):
    P_k = P0[k] + TP[k] @ iP_k
    P[k] = inv(H_k) @ P_k  # Undo conditioning

for n in range(N):
    X[:, n] = X0[:, n] + TX[n] @ iX_n
```

---

## Helper Functions

### vgg_conditioner_from_image(imsize)
Creates conditioning matrix for image.

**Formula:**
```python
f = (width + height) / 2

C = [[1/f,   0,  -width/(2*f) ],
     [  0, 1/f, -height/(2*f) ],
     [  0,   0,        1       ]]
```

**Effect:**
- Translates center to origin: `(-w/2, -h/2)` → `(0, 0)`
- Scales to unit size: `w/f, h/f` → `1, 1`

**Inverse:** (computed analytically, not `np.linalg.inv`)
```python
invC = [[f, 0, width/2 ],
        [0, f, height/2],
        [0, 0,    1    ]]
```

### levmarq(F, p0, opt)
Levenberg-Marquardt optimizer.

**Algorithm:**
1. Evaluate F(p) → (residual, Jacobian)
2. Solve: `(J^T J + λI) D = -J^T r`
3. Test step: accept if improves, else increase λ
4. Repeat until convergence

**Adaptive λ:**
- Success: `λ ← max(λ/10, 1e-15)` (→ Gauss-Newton)
- Failure: `λ ← min(λ*10, 1e5)` (→ gradient descent)

### hom(x)
Euclidean to homogeneous coordinates.

```python
x = [[1, 2, 3],
     [4, 5, 6]]

x_hom = [[1, 2, 3],
         [4, 5, 6],
         [1, 1, 1]]  ← Added
```

Inverse of `p2e`.

### normP(P) and normx(X)
Normalize to unit norm.

```python
# Cameras
P_k = P_k / ||P_k||_F

# Points
X_n = X_n / ||X_n||
```

---

## Key MATLAB → Python Conversions

| MATLAB | Python | Description |
|--------|--------|-------------|
| `nargin < 6` | `opt is None` | Arg count check |
| `isfield(opt, 'f')` | `'f' in opt` | Dict key check |
| `k2i(k)` | `Utils.k2i(np.array([k]), step=3)` | Index conversion |
| `k2i(k, 2)` | `Utils.k2i(np.array([k]), step=2)` | For 2D points |
| `nhom(...)` | `Utils.p2e(...)` | Homogeneous to Euclidean |
| `hom(...)` | `hom(...)` | Euclidean to homogeneous |
| `[qX, dummy] = qr(X)` | `Q, R = np.linalg.qr(X, mode='complete')` | Full QR |
| `qX(2:end, :)'` | `Q[1:, :].T` | Skip first row, transpose |
| `reshape(v, [3 4])` | `v.reshape(3, 4, order='F')` | Fortran order |
| `feval(F, p, ...)` | `F(p, ...)` | Call function |
| `issparse(J)` | `hasattr(J, 'toarray')` | Check if sparse |
| `speye(n)` | `speye(n, format='csr')` | Sparse identity |
| `inv(H)` | `np.linalg.inv(H)` | Matrix inverse |

---

## Image Conditioning Details

### Why Condition?

**Problem:** Image coordinates in [0, width] × [0, height]
- Large values → poor conditioning
- Off-center → asymmetric gradients

**Solution:** Transform to [-1, 1] × [-1, 1] centered at origin

### Conditioning Matrix

**Forward (C):**
```
Point (x, y) in image → (x', y') conditioned

x' = (x - width/2) / f
y' = (y - height/2) / f

where f = (width + height) / 2
```

**Inverse (invC):**
```
x = f*x' + width/2
y = f*y' + height/2
```

### Application

**Cameras:**
```python
P_cond = H @ P_original
```

**Points:**
```python
q_hom = hom(q)           # (2×N) → (3×N)
q_cond_hom = H @ q_hom   # Apply conditioning
q_cond = p2e(q_cond_hom) # Back to Euclidean
```

**Undo (after optimization):**
```python
P_final = inv(H) @ P_optimized
```

---

## Levenberg-Marquardt Details

### Damping Parameter λ

**Small λ (→0):**
- Gauss-Newton method
- Fast convergence near minimum
- Assumes local quadratic model

**Large λ (→∞):**
- Gradient descent
- Robust far from minimum
- Slow but stable

### Adaptive Strategy

```python
if step_improves:
    λ = max(λ / 10, 1e-15)  # Trust local model more
else:
    λ = min(λ * 10, 1e5)    # Trust local model less
```

### Convergence Criteria

1. **Step size:** `max|r(p) - r(p-1)| < threshold`
2. **Max iterations:** `niter >= max_niter`
3. **Max failures:** `nfail >= 20`

---

## Objective Function

### Forward Model
```python
for each visible point (k, n):
    P_k = P0_k + TP_k @ iP_k
    X_n = X0_n + TX_n @ iX_n
    x = P_k @ X_n
    q = p2e(x)
    residual = q - q_observed
```

### Jacobian
```python
for each visible point:
    dxdP = kron(X_n^T, I_3) @ TP_k
    dxdX = P_k @ TX_n
    dudx = [I_2 | -u] / z
    
    dq/diP = dudx @ dxdP
    dq/diX = dudx @ dxdX
```

Sparse matrix with ~14 entries per visible point.

---

## Usage Example

```python
from bundle_PX_proj import bundle_PX_proj
import numpy as np

# Initial estimates
P0 = ...  # (3K × 4)
X0 = ...  # (4 × N)
q = ...   # (2K × N) observations
imsize = np.array([[width, height]] * K).T  # (2 × K)

# Options
opt = {
    'verbose': True,
    'max_niter': 100,
    'lam_init': 1e-3
}

# Bundle adjustment
P, X = bundle_PX_proj(P0, X0, q, imsize, None, opt)

print(f"Optimized: {P.shape[0]//3} cameras, {X.shape[1]} points")
```

---

## Common Use Cases

### 1. After Structure-from-Motion
```python
# Get initial estimate
P_init, X_init = fill_mm(M, opt)

# Refine with bundle adjustment
P_final, X_final = bundle_PX_proj(P_init, X_init, q, imsize)
```

### 2. Multi-Resolution Refinement
```python
# Start coarse
P, X = bundle_PX_proj(P0, X0, q_coarse, imsize_coarse)

# Refine fine
P, X = bundle_PX_proj(P, X, q_fine, imsize_fine)
```

### 3. Iterative Reconstruction
```python
for iteration in range(max_iter):
    P, X = bundle_PX_proj(P, X, q, imsize, opt)
    # Update outlier rejection, etc.
```

---

## Mathematical Background

### Projective Bundle Adjustment
```
min Σ_k Σ_n ||q_kn - π(P_k, X_n)||²
```

Subject to projective constraints.

### Image Conditioning
Based on Hartley & Zisserman "Multiple View Geometry":
- Improves condition number of normal equations
- Reduces numerical errors
- Essential for reliable optimization

### Tangent Space Parameterization
Removes scale ambiguity:
- P has 12 parameters but only 11 DOF
- Parameterize on 11D manifold
- Avoids singular Hessian

---

## Convergence Behavior

### Typical Pattern
```
Iter 0: rms = 10.5, max = 45.2
Iter 1: rms = 2.3,  max = 12.1, λ = 1e-4
Iter 2: rms = 0.8,  max = 4.2,  λ = 1e-5
Iter 3: rms = 0.5,  max = 2.1,  λ = 1e-6
...
Converged!
```

### Progress Indicators
- **rms:** Root mean square residual
- **max:** Maximum residual
- **stepmax:** Maximum change in residual
- **λ:** Damping parameter

---

## Comparison with Other Methods

### bundle_PX_proj vs qPXbundle_cmp

| Feature | bundle_PX_proj | qPXbundle_cmp |
|---------|---------------|---------------|
| Conditioning | Image-based | Hartley norm |
| Optimizer | Custom LM | Custom LM |
| Radial | Not implemented | Supported |
| API | Separate H per camera | Single normalization |

### When to Use Each

**bundle_PX_proj:**
- Different image sizes
- Prefer image-based conditioning
- Standard projective bundle

**qPXbundle_cmp:**
- Need radial distortion
- Uniform treatment
- Legacy compatibility

---

## Performance

### Complexity
- **Per iteration:** O(n_visible × (K+N))
- **Typical iterations:** 5-15
- **Jacobian:** Sparse (O(n_visible) storage)

### Memory
- **Dense:** P, X, parameters
- **Sparse:** Jacobian
- **Peak:** During solve `(J^T J + λI) D = b`

---

## Debugging Tips

### Divergence
**Cause:** Poor initialization, λ too small
**Fix:**
- Increase `lam_init` (try 1e-3 or 1e-2)
- Better P0, X0 from fill_mm

### Slow Convergence
**Cause:** Large λ, poor conditioning
**Fix:**
- Check conditioning matrices
- Verify imsize is correct
- Better initialization

### NaN in Results
**Cause:** Singular system, bad conditioning
**Fix:**
- Check visibility matrix
- Ensure sufficient correspondences
- Verify imsize

---

## Advanced Features

### Image-Specific Conditioning
Each camera gets its own H_k based on its imsize.

**Benefit:** Handles varying image resolutions naturally.

### Sparse Jacobian
Only non-zero entries stored/computed.

**Benefit:** Efficient for large problems.

### Adaptive Damping
λ adjusts based on step success.

**Benefit:** Fast convergence + robustness.

---

## Notes

- **Automatic conditioning:** No manual preparation needed
- **Handles missing data:** Via qivis mask
- **Sparse Jacobian:** Essential for large K, N
- **No radial distortion:** Not implemented (use qPXbundle_cmp if needed)
- **Image sizes matter:** Must match actual images
- **Undo conditioning:** Automatically applied to results

---

## Testing

The module includes comprehensive tests:
```bash
python bundle_PX_proj.py
```

Tests:
1. Full bundle adjustment
2. Conditioning matrices (forward + inverse)
3. hom/p2e round-trip
4. Helper functions

---

## Integration with Pipeline

### Complete Workflow
```python
# 1. Initial reconstruction
P_init, X_init, u1, u2, info = fill_mm(M, opt)

# 2. Bundle adjustment (this function)
P_ba, X_ba = bundle_PX_proj(P_init, X_init, q, imsize, None, opt)

# 3. Optional: Further refinement with radial
# P_final, X_final, radial = qPXbundle_cmp(P_ba, X_ba, q, radial_init)
```

### Uses These Functions
- `Utils.k2i()` - Index conversion
- `Utils.p2e()` - Homogeneous to Euclidean
- Internal: `hom()`, `normP()`, `normx()`

---

## Error Metrics

### Printed During Optimization

**rms (root mean square):**
```
rms = sqrt(mean(residual²))
```

**max:**
```
max = max(|residual|)
```

**stepmax:**
```
stepmax = max(|r_new - r_old|)
```

All scaled by `res_scale` for printing.

---

## Comparison with SciPy

### Why Custom Levenberg-Marquardt?

**Advantages:**
- Full control over λ adaptation
- Sparse Jacobian support built-in
- Verbose output tailored for bundle
- Same behavior as MATLAB original

**Could use scipy.optimize.least_squares:**
```python
from scipy.optimize import least_squares
result = least_squares(objective, p0, jac=True, method='lm')
```

But custom implementation matches original better.
