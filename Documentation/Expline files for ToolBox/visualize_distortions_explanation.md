# `visualize_distortions` — Function Explanation

## What does it do?

`visualize_distortions` produces three diagnostic plots that visually inspect
how a calibrated lens distorts the image. After running calibration, you want
to understand *how much* and *where* the distortion occurs. These plots show
the distortion as a vector field (arrows showing where each pixel moves) and
contour lines of displacement magnitude in pixels.

**Figure 1 — Complete distortion:** the full model (radial + tangential combined)

**Figure 2 — Tangential only:** isolated effect of p1 and p2

**Figure 3 — Radial only:** isolated effect of k1, k2, k3

---

## Previously Converted Functions Used

| Function | Role here |
|---|---|
| `apply_distortion.py` | Called three times — once per figure — with different `kc` subsets to isolate each distortion component |
| `quiver.py` | Informs the design but see note below about native `ax.quiver` |

### Note on `quiver.py` vs `ax.quiver`

Our converted `calib_quiver` from `quiver.py` is designed for **grid-shaped**
`x, y` inputs (2D arrays). In this script, `px` and `py` are **flat 1D
scatter arrays** (441 points from a 21×21 grid, but passed as column vectors).
The MATLAB original passes them the same way and MATLAB's `quiver` handles
both cases. Since Python's matplotlib `ax.quiver` natively handles 1D scatter
inputs identically, we use `ax.quiver` directly here to keep the call clean.
`calib_quiver` is still imported and available, and would be the right choice
if the caller had 2D grid arrays.

---

## Inputs & Outputs

| Name | Type | Description |
|------|------|-------------|
| `nx`, `ny` | int | Image width and height in pixels |
| `fc` | (2,) | Focal lengths [fx, fy] |
| `cc` | (2,) | Principal point [cx, cy] |
| `kc` | (5,) | Distortion coefficients [k1, k2, p1, p2, k3] |
| `alpha_c` | float | Skew coefficient |
| `fc_error` | (2,) | Uncertainty estimates on fc |
| `cc_error` | (2,) | Uncertainty estimates on cc |
| `kc_error` | (5,) | Uncertainty estimates on kc |
| `alpha_c_error` | float | Uncertainty on skew |
| `err_std` | (2,) | Reprojection error std [x-std, y-std] |
| `save_path` | str | Optional path prefix to save PNGs |
| *(output)* | fig1, fig2, fig3 | Matplotlib Figure objects |

---

## Step-by-step Breakdown

### Step 1 — Build the normalised ray grid
A 21×21 grid of pixel positions is sampled uniformly across the image:
```python
gx = np.linspace(0, nx-1, 21)
gy = np.linspace(0, ny-1, 21)
mx, my = np.meshgrid(gx, gy)
```
These pixel positions are then back-projected through the inverse of the
intrinsic matrix K to get **normalised undistorted image coordinates**:
```python
rays = inv(KK) @ [px; py; 1]
x = rays[:2] / rays[2]
```
This is where `alpha_c` matters — K includes the skew term, so
`inv(K)` automatically accounts for it when computing normalised coords.

### Step 2 — Apply distortion (calls `apply_distortion.py`)
For each of the three figures, `apply_distortion` is called with a different
subset of `kc`:

| Figure | kc passed |
|--------|-----------|
| Complete | `[k1, k2, p1, p2, k3]` — all coefficients |
| Tangential | `[0, 0, p1, p2, 0]` — only p1, p2 |
| Radial | `[k1, k2, 0, 0, k3]` — only k1, k2, k3 |

The distorted normalised coordinates are then projected back to pixel space
using the focal length and principal point to give `(px2, py2)`.

### Step 3 — Compute and plot displacement
```python
dx = px2 - px    # pixel displacement due to distortion
dy = py2 - py
dr = sqrt(dx² + dy²)   # magnitude on the 21×21 grid
```
The quiver plot shows arrows from the undistorted to the distorted pixel
position. The contour plot overlaid on the same axes shows iso-magnitude
lines, making it easy to see how strongly the lens distorts at each distance
from the centre.

### Step 4 — Annotate with camera parameters
Each figure gets a text block along the bottom edge listing all calibration
parameters and their uncertainties — useful for quick visual inspection after
calibration.

---

## Key MATLAB → Python Differences

| Topic | MATLAB | Python |
|-------|--------|--------|
| `meshgrid` order | `meshgrid(x, y)` → same as NumPy | Same — `np.meshgrid(gx, gy)` |
| `reshape(mx', nnx*nny, 1)` | Transpose then flatten column-major | `mx.T.ravel()` |
| `reshape(dr, nny, nnx)'` | Column-major reshape then transpose | `.reshape(nnx, nny).T` |
| `inv(KK)` | Built-in `inv` | `np.linalg.inv(KK)` |
| `axis ij` | Y-axis inverted (image convention) | `ax.invert_yaxis()` |
| `figure(N); clf` | Create/clear named figure | `plt.subplots()` — new figure each call |
| `axes(position)` text overlay | Invisible axes for text placement | `fig.add_axes([0,0,1,1], visible=False)` |
| `contour` + `clabel` | Built-in | `ax.contour()` + `ax.clabel()` |
