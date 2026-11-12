# `apply_distortion` — Function Explanation

## What does it do?

When a camera captures an image, the lens introduces geometric distortions — straight lines in the real world appear curved. `apply_distortion` **simulates this effect**: it takes idealized, undistorted normalized image coordinates `x` and returns the distorted coordinates `xd` that the lens would actually produce.

It models two types of distortion:

- **Radial distortion** — causes barrel or pincushion warping (points shift outward or inward from the centre)
- **Tangential distortion** — caused by the lens not being perfectly parallel to the image sensor

---

## Inputs & Outputs

| Name | Shape | Description |
|------|-------|-------------|
| `x` | (2, N) | Undistorted normalized image coordinates (row 0 = x, row 1 = y) |
| `k` | (1–5,) | Distortion coefficients: k1, k2, p1, p2, k3 |
| `xd` *(output)* | (2, N) | Distorted image coordinates |
| `dxddk` *(output)* | (2N, len(k)) | Jacobian of `xd` w.r.t. `k` |

The coefficient vector `k` maps as follows:

| Index | Symbol | Effect |
|-------|--------|--------|
| k[0] | k1 | Radial, r² term |
| k[1] | k2 | Radial, r⁴ term |
| k[2] | p1 | Tangential |
| k[3] | p2 | Tangential |
| k[4] | k3 | Radial, r⁶ term |

---

## Step-by-step Breakdown

### Step 1 — Pad `k` to length 5
Some callers only pass 2 or 4 coefficients (simpler models). The function always needs 5 slots internally, so it zero-pads the rest. The original length is saved so the Jacobian can be trimmed back at the end.

### Step 2 — Compute radius powers (r², r⁴, r⁶)
```
r² = x² + y²
```
These measure how far each point is from the image centre. Distortion grows with distance, so higher powers are needed to model the effect accurately at the edges of the image.

### Step 3 — Radial scale factor `cdist`
```
cdist = 1 + k1·r² + k2·r⁴ + k3·r⁶
```
This polynomial scales each point outward or inward. A point at the centre (r = 0) gives `cdist = 1`, meaning no change. Points near the edges get a larger correction.

### Step 4 — Apply radial distortion
```
xd1 = x · cdist
```
Each coordinate is multiplied by `cdist`. This is the dominant effect and models barrel/pincushion warping.

### Step 5 — Compute tangential distortion `delta_x`
```
delta_x[0] = p1·(2xy)       + p2·(r² + 2x²)
delta_x[1] = p1·(r² + 2y²)  + p2·(2xy)
```
Tangential distortion is an asymmetric shift driven by p1 and p2. It models the lens being slightly tilted relative to the sensor plane.

### Step 6 — Combine both effects
```
xd = xd1 + delta_x
```
The final distorted point is the sum of the radially-distorted position and the tangential shift.

### Step 7 — Jacobian `dxddk`
The derivative of `xd` with respect to each coefficient in `k`. This is used by camera calibration optimisers (e.g. Levenberg-Marquardt) to efficiently minimise reprojection error without needing to approximate gradients numerically.

---

## Key MATLAB → Python Gotcha

MATLAB's `reshape(M, 2, n)` is **column-major** (Fortran order). NumPy defaults to row-major (C order). When using the Jacobian output, always write:

```python
# ✅ Correct
arr.reshape(2, n, order='F')

# ❌ Wrong — silently gives incorrect results
arr.reshape(2, n)
```
