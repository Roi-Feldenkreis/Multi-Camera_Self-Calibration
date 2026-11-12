# `rigid_motion` — Function Explanation

## What does it do?

`rigid_motion` applies a **rigid body transformation** to a set of 3D points:

```
Y = R * X + T
```

where `R` is the rotation matrix built from the rotation vector `om` (via
`rodrigues`), and `T` is a translation vector. This is the core operation
that moves points from the **world coordinate frame** into the **camera
coordinate frame** — an essential step in any 3D projection pipeline.

It also returns the **Jacobians** of the output `Y` with respect to both
`om` and `T`, which are needed by the calibration optimiser to refine the
camera's pose efficiently.

---

## Inputs & Outputs

| Name | Shape | Description |
|------|-------|-------------|
| `X` | (3, N) | 3D points in the world frame |
| `om` | (3,) | Rotation vector (default: zeros → identity) |
| `T` | (3,) | Translation vector (default: zeros) |
| `Y` *(out)* | (3, N) | 3D points in the camera frame |
| `dYdom` *(out)* | (3N, 3) | Jacobian of Y w.r.t. om |
| `dYdT` *(out)* | (3N, 3) | Jacobian of Y w.r.t. T |

The 3N rows of both Jacobians are interleaved point-by-point:
rows `[0,1,2]` → point 0, rows `[3,4,5]` → point 1, etc.

---

## Step-by-step Breakdown

### Step 1 — Get R and dRdom from `rodrigues`
```python
R, dRdom = rodrigues(om)   # R: (3,3),  dRdom: (9,3)
```
`rodrigues` is called with the rotation vector `om` and returns both the
rotation matrix and its Jacobian. The Jacobian `dRdom` has 9 rows — one per
element of R flattened in column-major (Fortran) order — and 3 columns for
the 3 components of `om`.

### Step 2 — Apply the transformation
```python
Y = R @ X + T[:, np.newaxis]   # (3, N)
```
Each column of `X` is rotated and then shifted by `T`.

### Step 3 — Build `dYdR` (Jacobian of Y w.r.t. R)
Since `Y[:,k] = R @ X[:,k] + T`, differentiating with respect to each element
of R gives a sparse block structure. Using column-major R indexing (columns
0,3,6 correspond to `R[0,:]`; columns 1,4,7 to `R[1,:]`; columns 2,5,8 to
`R[2,:]`):

```python
dYdR[0::3, 0::3] = X.T   # x-outputs depend on row 0 of R
dYdR[1::3, 1::3] = X.T   # y-outputs depend on row 1 of R
dYdR[2::3, 2::3] = X.T   # z-outputs depend on row 2 of R
```

This column indexing must match the column-major R ordering used in `dRdom`
from `rodrigues` — the two Jacobians must speak the same "language" for the
chain rule to work correctly.

### Step 4 — Build `dYdT` (Jacobian of Y w.r.t. T)
Since each point's x-coordinate depends only on `T[0]`, y only on `T[1]`,
z only on `T[2]`, `dYdT` is a simple repeating identity block:
```python
dYdT[0::3, 0] = 1.0   # all x-outputs
dYdT[1::3, 1] = 1.0   # all y-outputs
dYdT[2::3, 2] = 1.0   # all z-outputs
```

### Step 5 — Chain rule: `dYdom`
```python
dYdom = dYdR @ dRdom    # (3N, 9) @ (9, 3) → (3N, 3)
```
The Jacobian of Y with respect to `om` is obtained by chaining `dYdR`
through `dRdom`. This works correctly only because both matrices use the
same column-major R flattening convention.

---

## Alignment with `rodrigues`

The critical connection between `rigid_motion` and `rodrigues` is the
**column-major (F-order) R flattening** convention:

| Role | Matrix | Convention |
|------|--------|------------|
| `rodrigues` output | `dRdom` — (9,3) | Rows = R flattened F-order |
| `rigid_motion` intermediate | `dYdR` — (3N,9) | Columns = R flattened F-order |
| Chain rule product | `dYdom = dYdR @ dRdom` | Consistent F-order cancels out |

If either side used C-order flattening, the matrix multiply would be
silently wrong. The numerical Jacobian tests (gains ~1M) confirm both
are consistent.

To apply `dYdom` correctly in Python (as the MATLAB test block does):
```python
Y_pred = Y + (dYdom @ dom).reshape(3, N, order='F')
```

---

## Key MATLAB → Python Differences

| Topic | MATLAB | Python |
|-------|--------|--------|
| `nargin` defaults | `if nargin < 3` cascade | `None`-defaulted keyword args |
| `repmat(T,[1 n])` | Tiles T across N columns | `T[:, np.newaxis]` broadcast |
| `nargout > 1` guard | Only computes Jacobians if caller needs them | Always computed (Python has no `nargout`) |
| `size(X)` | `[m, n]` | `X.shape[1]` for N points |
