# `project_points2` — Function Explanation

## What does it do?

`project_points2` is the **complete forward projection function** of the
calibration toolbox. Given a set of 3D world points and a full camera model,
it computes where each point appears in the image as a pixel coordinate. It
also returns seven Jacobian matrices — one for each set of parameters — which
the calibration optimiser needs to efficiently minimise reprojection error.

The projection pipeline has five sequential stages:

```
World 3D points X
    ↓  rigid_motion (rotation + translation)
Camera-frame points Xc = R·X + T
    ↓  perspective divide
Normalised image coords x = Xc[:2] / Xc[2]
    ↓  radial + tangential distortion
Distorted coords xd
    ↓  skew correction
xd3
    ↓  focal length scaling + principal point shift
Pixel coords xp  ✓
```

---

## Previously Converted Functions Used

| Function | Role here |
|---|---|
| `rigid_motion.py` | Step 1 — transforms X from world to camera frame, returns `Y`, `dYdom`, `dYdT` |
| `rodrigues.py` | Called internally by `rigid_motion` to build R from `om` |

Both share the same **column-major (F-order) Jacobian convention**, so their
outputs chain correctly into `project_points2`'s Jacobian calculations.

---

## Inputs & Outputs

| Name | Shape | Description |
|------|-------|-------------|
| `X` | (3, N) | 3D world points |
| `om` | (3,) | Rotation vector |
| `T` | (3,) | Translation vector |
| `f` | (2,) or float | Focal lengths [fx, fy] |
| `c` | (2,) | Principal point [cx, cy] |
| `k` | (5,) | Distortion coefficients [k1,k2,p1,p2,k3] |
| `alpha` | float | Skew coefficient |
| `xp` *(out)* | (2, N) | Pixel coordinates |
| `dxpdom` *(out)* | (2N, 3) | Jacobian w.r.t. om |
| `dxpdT` *(out)* | (2N, 3) | Jacobian w.r.t. T |
| `dxpdf` *(out)* | (2N, 2) or (2N, 1) | Jacobian w.r.t. f |
| `dxpdc` *(out)* | (2N, 2) | Jacobian w.r.t. c |
| `dxpdk` *(out)* | (2N, 5) | Jacobian w.r.t. k |
| `dxpdalpha` *(out)* | (2N, 1) | Jacobian w.r.t. alpha |

---

## Step-by-step Breakdown

### Step 1 — Rigid motion (calls `rigid_motion.py`)
```python
Y, dYdom, dYdT = rigid_motion(X, om, T)
```
Rotates and translates the 3D points into the camera coordinate frame. The
returned Jacobians `dYdom` (3N×3) and `dYdT` (3N×3) are passed forward into
all subsequent Jacobian calculations via the chain rule.

### Step 2 — Perspective divide
```python
x = Y[:2] / Y[2]     # normalised coords, (2, N)
```
Projects 3D camera-frame points onto the unit focal plane. The Jacobians
`dxdom` and `dxdT` (both 2N×3) are derived using the quotient rule:
```
d(x/z)/d(param) = (1/z) · d(x)/d(param)  −  (x/z²) · d(z)/d(param)
```
The `bb` and `cc` terms implement this `-x/z²` scaling for the z-row of
the rigid motion Jacobian.

### Step 3 — Distortion (radial + tangential)
Mirrors `apply_distortion.py` but computed **forward** (not iterative
inversion) and with full Jacobians tracked throughout.

**Radial:** `cdist = 1 + k1·r² + k2·r⁴ + k3·r⁶` scales each normalised
point outward or inward.

**Tangential:** `delta_x` shifts the point asymmetrically using p1, p2.

Jacobians `dxd2dom`, `dxd2dT`, `dxd2dk` are built by differentiating each
sub-expression with respect to the upstream quantities and chaining them.

### Step 4 — Skew correction
```python
xd3[0] = xd2[0] + alpha * xd2[1]
xd3[1] = xd2[1]
```
Only the x-coordinate is affected. The Jacobian `dxd3dalpha = xd2[1]`
captures how a change in skew shifts the x-pixel.

### Step 5 — Pixel coordinates
```python
xp = f * xd3 + c
```
Two sub-cases: if `f` is a 2-vector (separate fx, fy), the x and y pixels
are scaled independently. If `f` is a scalar, both are scaled equally and
`dxpdf` becomes a single column.

`dxpdc` is trivially the identity repeated — shifting the principal point
shifts every pixel by the same amount.

---

## Key MATLAB → Python Differences

| Topic | MATLAB | Python |
|-------|--------|--------|
| `nargin` defaults | Nested `if nargin < N` | `None`-defaulted keyword args |
| `nargout` guard | Jacobians only when needed | Always computed |
| `repmat(T,[1 n])` | Tile T across columns | `T[:, np.newaxis]` broadcast |
| Variable name clashes | `bb`, `cc` reused for different things | Renamed (`bb`/`cc` for pinhole; `aa`/`bb`/`cc` for tangential) |
| Apply Jacobian | `reshape(J*dp, 2, n)` column-major | `(J @ dp).reshape(2, n, order='F')` |

### Note on variable name reuse
The original MATLAB reuses `bb` and `cc` twice — first for the perspective
divide terms, then for the tangential distortion terms. In Python these are
kept as separate local variables to avoid confusion, with `aa`/`bb`/`cc`
used only in the tangential section.
