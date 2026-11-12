# `quiver` — Function Explanation

## What does it do?

`quiver` draws a **vector field plot** — each point on a 2D grid gets an arrow showing the direction and magnitude of a vector at that location. Typical uses include visualising fluid flow, gradients, electric fields, or any quantity that has both magnitude and direction at each point in space.

---

## Inputs & Outputs

| Argument | Type | Description |
|----------|------|-------------|
| `x`, `y` | array-like | Grid positions (optional — defaults to index grid) |
| `u`, `v` | array-like | Vector components at each grid point |
| `s` | float | Scale factor; `0` disables autoscaling (default `1`) |
| `color` | str | Matplotlib colour, e.g. `'r'`, `'blue'` |
| `linestyle` | str | Matplotlib linestyle, e.g. `'-'`, `'--'` |
| `marker` | str | Marker drawn at vector base; suppresses arrowheads |
| `filled` | bool | Fill the base marker (default `False`) |
| `ax` | Axes | Matplotlib axes to draw on (default: current axes) |

**Returns:** a list `[shaft_line, arrowhead_line, marker_line]` of `Line2D` objects (elements may be `None` if not drawn).

---

## Step-by-step Breakdown

### Step 1 — Parse arguments
The function accepts flexible call signatures, just like MATLAB's `varargin`. String arguments are stripped from the end to detect `'filled'` and matplotlib format strings (e.g. `'r--'`, `'ko'`). If a marker character is found, arrowheads are suppressed and the marker is drawn at the base instead.

### Step 2 — Build the grid
If only `u, v` are given (no `x, y`), a default integer index grid is generated with `np.meshgrid`. This matches MATLAB's `xyzchk` behaviour which produces a uniform grid when positions are not specified.

### Step 3 — Autoscale arrow lengths
To prevent arrows from overlapping or being invisibly small, the function normalises vector lengths against the average grid cell size:
```
scale = 0.9 × s / max_normalised_length
```
The average cell size is estimated from the range of `x` and `y` divided by the number of grid points. The scale factor `s` (default 1) lets the caller stretch or shrink arrows further. Setting `s = 0` skips this step entirely.

### Step 4 — Draw arrow shafts
All shafts are drawn as a **single `Line2D`** by building NaN-separated coordinate arrays. Each vector contributes three values: `[tail, tip, NaN]`. The NaN breaks the line so each shaft appears disconnected, but only one plot call is needed — which is efficient for large grids.

### Step 5 — Draw arrowheads
Each arrowhead is a V-shape made of two line segments meeting at the tip. The wing points are calculated using `alpha` (controls how far back the wings start) and `beta` (controls how wide they are):
```
left_wing  = tip - alpha × (u + beta × v)
right_wing = tip - alpha × (u − beta × v)
```
A tiny `eps` is added to `v` to avoid division by zero for zero-length vectors. Again, all arrowheads are packed into one `Line2D` using NaN separators.

### Step 6 — Draw base markers (optional)
If a marker character was found in the format string, markers are plotted at each vector's tail position. If `filled=True`, the marker face colour is set to match the shaft colour.

---

## Key MATLAB → Python Differences

| Topic | MATLAB | Python |
|-------|--------|--------|
| Grid gradient order | `gradient(z, dx, dy)` returns `(px, py)` | `np.gradient(z, dy, dx)` returns `(gy, gx)` — **axes are swapped** |
| NaN-separated lines | Natural with MATLAB `plot` | Replicated with NaN-padded arrays into a single `ax.plot` call |
| `colstyle` parsing | Built-in MATLAB function | Replicated with regex on the format string |
| `eps` | Machine epsilon built-in | `np.finfo(float).eps` |
| Hold state | `ishold` / `hold on/off` | Managed implicitly by matplotlib axes |
