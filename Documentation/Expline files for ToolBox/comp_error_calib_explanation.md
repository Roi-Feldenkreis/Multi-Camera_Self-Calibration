# `comp_error_calib` & `check_active_images` — Explanation

---

## Why does `check_active_images.m` only have 3 lines?

This is the most important thing to understand about these two files.

In the full CalTech Camera Calibration Toolbox, `check_active_images` is a
**MATLAB script** (not a function). It runs inside the shared MATLAB workspace,
reads variables like `n_ima`, `active_images`, and `ind_active` that already
exist there, and may rebuild or validate them. The full version is much longer.

**The 3-line version you have is a minimal stub** — a safe fallback used when
the toolbox is run with a single image and no images have been explicitly
deactivated. Its three lines just hard-code the simplest possible state:

```matlab
ind_active = 1;      % index of the one active image
n_ima = 1;           % total number of images = 1
active_images = 1;   % that one image is active
```

In other words: *"there is exactly one image and it is active."* Nothing more
is needed in that minimal case, so nothing more is written. In the full
multi-image toolbox workflow, this script would contain logic to loop over all
images, check whether each one has valid calibration data loaded, and build
the full `active_images` boolean array.

In Python we convert it into a proper utility function with sensible defaults
that generalises naturally to any number of images.

---

## What does `comp_error_calib` do?

After a camera has been calibrated (i.e. after intrinsic parameters `fc`, `cc`,
`kc` and per-image poses `omc`, `Tc` have been estimated), this script
**recomputes how accurately the model explains the observed data**.

For each active calibration image it:
1. Takes the estimated 3D world points `X`
2. Re-projects them onto the image plane using the current camera model
3. Computes the pixel-level difference between where corners were *detected*
   (`x`) and where the model *predicts* them (`y`)

The final metric — **reprojection error** — is the standard deviation of these
pixel differences across all active images. A well-calibrated camera typically
achieves under 1 pixel RMS reprojection error.

---

## Inputs & Outputs

| Name | Type | Description |
|------|------|-------------|
| `X_list` | list of (3, N_k) | 3D world points per image |
| `x_list` | list of (2, N_k) | Detected 2D corners per image |
| `omc_list` | list of (3,) | Rotation vectors per image |
| `Tc_list` | list of (3,1) | Translation vectors per image |
| `fc` | (2,) | Focal lengths [fx, fy] |
| `cc` | (2,) | Principal point [cx, cy] |
| `kc` | (5,) | Distortion coefficients |
| `alpha_c` | float | Skew coefficient |
| `active_images` | list of int | 1 = use image, 0 = skip (None = all active) |
| `ex_list` *(out)* | list of (2, N_k) | Per-image pixel errors (NaN if inactive) |
| `ex` *(out)* | (2, M) | All errors concatenated (active images only) |
| `x_all` *(out)* | (2, M) | All detected corners (active images only) |
| `y_all` *(out)* | (2, M) | All reprojected points (active images only) |
| `err_std` *(out)* | (2,) | Std of reprojection error [x-std, y-std] |

---

## Step-by-step Breakdown

### Step 1 — Check active images
Calls `check_active_images` to validate the `active_images` array and default
to all-active if none is provided. Sets `n_ima`.

### Step 2 — Loop over images
For each image `kk`:
- **If active and pose is valid** (not NaN): call `project_points2` to
  re-project the 3D points, then subtract from detected corners to get `ex_kk`.
  Accumulate into global arrays `ex`, `x_all`, `y_all`.
- **If inactive or pose is NaN**: store `NaN` for that image's error so
  downstream code can detect it was skipped.

### Step 3 — Compute error statistics
```python
err_std = np.std(ex, axis=1)   # one std per axis (x and y)
```
This summarises the overall quality of the calibration across all active images.

---

## Key MATLAB → Python Differences

| Topic | MATLAB | Python |
|-------|--------|--------|
| Dynamic variables | `eval(['x_' num2str(kk)])` builds variable names at runtime | Replaced with indexed lists `x_list[kk]` |
| Script vs function | `check_active_images` is a script sharing the workspace | Converted to a proper function with explicit arguments |
| `std` axis | `std(ex')'` transposes twice to get per-axis std | `np.std(ex, axis=1)` directly |
| `exist('alpha_c')` | Checks if workspace variable exists | Replaced with a default argument `alpha_c=0.0` |

The biggest structural change is replacing MATLAB's `eval`-based dynamic
variable names (`x_1`, `x_2`, ...) with Python lists. This is cleaner, more
readable, and avoids the security and debuggability problems of `eval`.
