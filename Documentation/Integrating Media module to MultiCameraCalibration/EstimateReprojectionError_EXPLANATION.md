# EstimateReprojectionError.py — Multimedia Geometry Changes Explanation

## What Was Changed

`reprerror` computes the 2D reprojection error for each camera after each
`euclidize()` call. Two additions integrate the Maas (1996) multimedia
functional model without altering the standard pinhole path.

---

## Overview of Changes

```
reprerror(cam, Pe, Xe, FRAMES, inliers)           ← original signature
reprerror(cam, Pe, Xe, FRAMES, inliers,            ← new signature
          cameras_mm=None, lut=None)

For each camera i:

  [UNCHANGED] xe = Pe[i] @ Xe          standard pinhole projection
  [UNCHANGED] Mask + index computation

  Error computation (two paths):
    cameras_mm is None → [UNCHANGED] xe_pts from standard xe
    cameras_mm provided → [NEW] xe_pts from multimedia model
      for each matched point:
        particle  = dehomogenise Xe[:,j]
        P_bar     = lut[i].shift_point(particle, cam_obj)   (Maas eq.8)
        x_pred, y_pred = collinearity_condition(P_bar, cam_obj)  (eq.17)

  [UNCHANGED] err2d = ||xe_pts - xgt_pts||
```

---

## Function Signature Change

### Before
```python
def reprerror(cam, Pe, Xe, FRAMES, inliers):
```

### After
```python
def reprerror(cam, Pe, Xe, FRAMES, inliers, cameras_mm=None, lut=None):
```

Both new arguments default to `None`. Every existing call site that omits
them runs the original code path unchanged.

---

## New Parameters

### `cameras_mm`
**Type:** `list[CameraParams]` or `None`

One `CameraParams` object per camera, built by `_build_camera_params()` in
`GoCal.py` immediately after `euclidize()`. Carries the 16 Maas orientation
and intrinsic parameters plus the shared `MultimediaParams` stack (n1/n2/d/n3).

### `lut`
**Type:** `MultiCameraLUT` or `None`

Per-camera lookup table mapping `(R, Zp)` → `R̄/R`. Built alongside
`cameras_mm`. Replaces the Maas iterative scheme (eqs. 4–7) with a single
bilinear interpolation per point.

---

## Why cam[i]['xe'] Is Left Unchanged

`cam[i]['xe']` is the standard pinhole projection `Pe[i] @ Xe`.
`findoutl()` reads this field to classify outliers using `INL_TOL`.
That threshold is calibrated for the standard model. Replacing `xe` with
multimedia-corrected coordinates would silently change which points are
rejected — potentially degrading the reconstruction on early iterations
when the LUT is still approximate.

The multimedia correction therefore applies **only** to the error metrics
(`err2d`, `mean2Derr`, `std2Derr`) — the values the outer loop monitors
for convergence.

---

## Multimedia Error Computation — Step by Step

For each point `k` in `cam[i]['recandvis']`:

### Step 1 — Dehomogenise the 3D point
```python
X_h      = Xe[:, rec_idx]          # shape (4,)
particle = X_h[:3] / X_h[3]        # shape (3,)  world coords in mm
```

### Step 2 — Radial shift via LUT (Maas eqs. 4–8)
```python
P_bar = lut[i].shift_point(particle, cam_obj)
```
`shift_point` looks up `R̄/R` for this camera from the prebuilt grid, scales
the horizontal offset from the nadir point, and returns the shifted position
`P̄(X̄, Ȳ, Z)`. Each camera computes a different `P̄` for the same particle.

### Step 3 — Collinearity condition (Maas eq. 17)
```python
x_pred, y_pred = collinearity_condition(P_bar, cam_obj)
```
Standard photogrammetric collinearity applied to `P̄` instead of `P`,
giving the predicted image coordinates in pixel space.

### Step 4 — Error
```python
cam[i]['err2d'] = ||[x_pred, y_pred] - xgt_pts||
```
Same formula as the standard path — only the source of `xe_pts` differs.

---

## New Output Field

### `cam[i]['xe_mm']`
**Shape:** `(2, n_matched)`

The multimedia-predicted image positions for the matched point subset.
Stored for diagnostics — compare against `cam[i]['xe'][0:2, recandvis]`
to visualise how much the multimedia correction shifts each predicted point.

---

## Coordinate System

`collinearity_condition` returns coordinates that include `cam_obj.xh` and
`cam_obj.yh` (principal point offset in mm). `cam[i]['xgt']` was shifted
back to pixel space in `GoCal.py` by adding `config.principal_point[i]`.
The two frames are consistent because `_build_camera_params()` sets:
```python
xh = K[0, 2]   # from the intrinsic matrix
yh = K[1, 2]
```
which matches the principal point used in `GoCal.py`.

---

## Backward Compatibility

```python
# Original call — unchanged behaviour
cam = reprerror(cam, Pe, Xe, FRAMES, inliers)

# Multimedia call — new behaviour for error metrics only
cam = reprerror(cam, Pe, Xe, FRAMES, inliers,
                cameras_mm=cameras_mm, lut=lut)
```

When `cameras_mm=None` and `lut=None` (the defaults), the function executes
identically to the original — the standard path is taken and no multimedia
code runs.

---

## Performance

The multimedia path loops over matched points per camera rather than using
the batched matrix multiply of the standard path. For typical experiments
(50–200 matched points per camera) this is fast enough for the calibration
loop. If performance becomes an issue, `lut.shift_point` and
`collinearity_condition` can be vectorised.

---

## References

- Maas, H.-G. (1996). *Contributions of Digital Photogrammetry to 3-D PTV*.
  Eqs. (4)–(8): iterative radial shift. Eq. (17): collinearity condition.
- See also: `GoCal_EXPLANATION.md`, `multimedia_lut.py`, `Iterative_radial_shift .py`
