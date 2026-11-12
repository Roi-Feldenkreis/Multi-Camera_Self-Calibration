# `normalize` — Function Explanation

## What does it do?

`normalize` converts raw **pixel coordinates** into **normalized image coordinates** — the undistorted, focal-length-independent coordinate system that lies between the camera sensor and the 3D world. This is a critical preprocessing step in camera calibration and 3D reconstruction: most geometry algorithms (triangulation, essential/fundamental matrix estimation, etc.) expect normalized coordinates as input, not raw pixels.

The function reverses three things the camera physically did to the 3D point:
1. Scaling by focal length and shifting by the principal point
2. Skewing due to a non-rectangular pixel grid
3. Warping by the lens distortion

---

## Inputs & Outputs

| Name | Shape | Description |
|------|-------|-------------|
| `x_kk` | (2, N) | Pixel feature locations |
| `fc` | (2,) | Focal lengths [fx, fy] (default [1, 1]) |
| `cc` | (2,) | Principal point [cx, cy] (default [0, 0]) |
| `kc` | (5,) | Distortion coefficients [k1, k2, p1, p2, k3] (default zeros) |
| `alpha_c` | float | Skew coefficient (default 0) |
| `xn` *(output)* | (2, N) | Normalized undistorted image coordinates |

---

## Step-by-step Breakdown

### Step 1 — Subtract principal point and divide by focal length
```python
x_distort[0] = (x_kk[0] - cx) / fx
x_distort[1] = (x_kk[1] - cy) / fy
```
This maps pixels back to the unit focal plane. The principal point `cc` is where the optical axis hits the sensor; the focal length `fc` is the scale factor the camera applied during projection. Reversing both moves the point into a coordinate system centred on the optical axis with unit scale.

### Step 2 — Undo skew
```python
x_distort[0] -= alpha_c * x_distort[1]
```
Some cameras have a slight shear between pixel rows and columns (non-90° pixel axes). The skew coefficient `alpha_c` models this. Only the x-coordinate is corrected because skew shifts x as a function of y.

### Step 3 — Compensate for lens distortion
```python
xn = comp_distortion_oulu(x_distort, kc)
```
Calls the iterative Oulu undistortion function (already converted) to remove radial and tangential lens distortion. This step is skipped entirely when `kc` is all zeros (ideal, distortion-free lens) to avoid unnecessary computation.

---

## Dependency

`normalize` imports and calls `comp_distortion_oulu` from the already-converted `comp_distortion_oulu.py`. Make sure both files are in the same directory (or on `PYTHONPATH`).

---

## Key MATLAB → Python Differences

| Topic | MATLAB | Python |
|-------|--------|--------|
| Default arguments | `nargin` cascade with nested `if` blocks | Keyword arguments with `None` defaults |
| Zero-check | `norm(kc) ~= 0` | `np.linalg.norm(kc) != 0` |
| Array indexing | 1-based `x_kk(1,:)` | 0-based `x_kk[0, :]` |

The MATLAB `nargin` cascade (checking how many arguments were passed and setting defaults accordingly) is replaced cleanly with Python's `None`-defaulted keyword arguments, which is more readable and idiomatic.
