# `compute_extrinsic_init` & `compute_homography` — Explanation

---

## What does `compute_extrinsic_init` do?

Given a set of detected 2D image points and their corresponding known 3D world
positions, this function computes an **initial estimate** of the camera's
extrinsic parameters — the rotation `R` (or equivalently `om`) and translation
`T` that explain how the 3D scene maps to the image.

"Initial" is the key word: the result does not need to be perfect because the
calibration toolbox will refine it with a nonlinear optimiser afterwards.
Two different algorithms are used depending on whether the 3D structure is
planar or not.

---

## What does `compute_homography` do?

`compute_homography` is the **missing dependency** referenced in the original
MATLAB code but never provided. It computes the 3×3 planar homography `H`
that maps 2D calibration pattern points to normalised image coordinates,
using the **normalised Direct Linear Transform (DLT)** algorithm.

It is only needed in the planar case of `compute_extrinsic_init`.

---

## Previously Converted Functions Used

| Function | Where used |
|---|---|
| `normalize.py` | Step 1 — converts pixel coords to normalised image coords |
| `rodrigues.py` | Steps 2c, 2d, non-planar case — vector↔matrix conversions |
| `project_points2.py` | Self-test only — generates synthetic test data |

---

## Inputs & Outputs (`compute_extrinsic_init`)

| Name | Shape | Description |
|------|-------|-------------|
| `x_kk` | (2, N) | Detected pixel feature locations |
| `X_kk` | (3, N) | Corresponding 3D world points |
| `fc` | (2,) | Focal lengths |
| `cc` | (2,) | Principal point |
| `kc` | (5,) | Distortion coefficients |
| `alpha_c` | float | Skew coefficient |
| `omckk` *(out)* | (3,) | Rotation vector |
| `Tckk` *(out)* | (3,1) | Translation vector |
| `Rckk` *(out)* | (3,3) | Rotation matrix |

---

## Step-by-step Breakdown

### Step 1 — Normalise image points (calls `normalize.py`)
```python
xn = normalize(x_kk, fc, cc, kc, alpha_c)
```
Converts raw pixel coordinates to the normalised image plane by removing the
principal point offset, focal length scaling, skew, and lens distortion. All
subsequent geometry is done in this clean coordinate system.

### Step 2 — Test for planarity
The 3D points are centred and their scatter matrix `YYᵀ` is decomposed by SVD.
The flatness ratio `r = S[2]/S[1]` measures how planar the structure is:
- `r ≈ 0` → planar (one singular value is near zero)
- `r ≈ 1` → fully 3D

Fewer than 5 points also forces the planar path since DLT needs at least 6.

---

### PLANAR PATH (r < 1e-3 or N < 5)

**Step 2a — Align the plane to Z=0**
The right singular vectors `V` of the scatter matrix give the plane's
orientation. The eigenvector corresponding to the smallest singular value
is the plane's normal. `R_transform = Vᵀ` rotates the plane so it lies
flat in the Z=0 plane, making the homography computable.

**Step 2b — Compute planar homography (calls `compute_homography.py`)**
```python
H = compute_homography(xn, X_new[:2, :])
```
`H` is a 3×3 matrix mapping 2D pattern points (now in the Z=0 plane) to
normalised image coordinates. It encodes `[r1 | r2 | t]` — the first two
columns of `R` and the translation — up to a scale factor.

**Step 2c — Orthogonalise to get a valid rotation**
The columns of `H` are close to orthonormal but not exact (due to noise and
the DLT approximation). Gram-Schmidt orthogonalisation is applied:
```python
u1 = H[:,0] / ||H[:,0]||
u2 = (H[:,1] - u1·H[:,1]·u1) / ||...||
u3 = u1 × u2     # cross product ensures right-handed frame
R = [u1 | u2 | u3]
```
`rodrigues` is then used to convert to and back from a rotation vector,
which projects `R` exactly onto SO(3).

**Step 2d — Compose with the pre-transform**
Since we rotated the world into Z=0 before computing H, the final pose must
account for that pre-rotation:
```
Rckk = R · R_transform
Tckk = Tckk + R · T_transform
```

---

### NON-PLANAR PATH — Direct Linear Transform (DLT)

Builds a (2N × 12) system `J · p = 0` where `p` encodes all 12 entries of
`[R | T]`. Each point correspondence contributes 2 rows. The solution is
the right-singular vector of `JᵀJ` corresponding to the smallest singular
value. The raw solution is then projected onto SO(3) via SVD to produce a
valid rotation matrix. The translation is recovered by normalising with the
Frobenius norm of the rotation.

This gives a rough but unbiased initial estimate across arbitrary 3D point
clouds, suitable as a starting point for nonlinear refinement.

---

## `compute_homography` — Algorithm Detail

Uses the **normalised DLT**:

1. **Normalise** both point sets (translate centroid to origin, scale mean
   distance to √2). This dramatically improves numerical conditioning.
2. **Build** the 2N×9 linear system `A·h = 0` from the cross-product
   constraint `xn × (H·Xp) = 0`.
3. **Solve** via SVD — the last right singular vector of `A` is `h`,
   reshaped to 3×3 to give `H_normalised`.
4. **Denormalise**: `H = inv(T_xn) · H_n · T_Xp`

---

## Key MATLAB → Python Differences

| Topic | MATLAB | Python |
|-------|--------|--------|
| `svd(YY)` returns `[U,S,V]` | S is diagonal matrix, V columns are right singular vectors | `np.linalg.svd` returns `(U, s, Vt)` — `s` is 1D, `Vt` rows are right singular vectors |
| `reshape(V(1:9,12), 3,3)` | Column-major reshape of the 12th column | `v12[:9].reshape(3,3)` — note: DLT here gives correct result in C-order because J is laid out accordingly |
| `svd(JJ)` last column | `V(:,12)` — last column | `Vsvd[-1]` — last row of `Vt` |
| `norm(X_kk(3,:)) < eps` | Original planarity test (commented out) | Replaced by SVD ratio `r = S[2]/S[1]` (the actual test used) |
| `compute_homography` | Called but not provided | Implemented from scratch using normalised DLT |
