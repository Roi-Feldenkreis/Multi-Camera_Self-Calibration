# Multi-Camera Self-Calibration — Configuration File Guide

This document explains every setting in the `multicamselfcal.cfg` configuration
file used by the Python (and MATLAB) multi-camera self-calibration pipeline.

---

## File Format

The configuration file uses standard INI format with `[Section]` headers and
`Key: Value` pairs. Comments start with `#` or `%`.

```ini
[Section]
Key: Value
```

> **Important:** Python uses `Key: Value` (colon separator). MATLAB's parser
> maps hyphenated keys to underscored struct fields automatically, so
> `Use-Nth-Frame` becomes `config.cal.USE_NTH_FRAME` in MATLAB.

---

## Complete Example — Air Only (standard pinhole)

```ini
[Files]
Basename: cam
Image-Extension: jpg

[Images]
Subpix: 0.5

[Calibration]
Num-Cameras: 4
Num-Projectors: 0
Nonlinear-Parameters: 50    0    1    0    0    0
Nonlinear-Update: 1   0   1   0   0   0
Initial-Tolerance: 10
Do-Global-Iterations: 1
Global-Iteration-Threshold: 0.5
Global-Iteration-Max: 5
Num-Cameras-Fill: 2
Do-Bundle-Adjustment: 1
Undo-Radial: 0
Min-Points-Value: 30
N-Tuples: 3
Square-Pixels: 1
Use-Nth-Frame: 2
Align-Existing: 0
```

## Complete Example — Multimedia (through-glass / underwater)

The only addition is the optional `[Multimedia]` section at the end.
Everything else stays identical.

```ini
[Files]
Basename: cam
Image-Extension: jpg

[Images]
Subpix: 0.5

[Calibration]
Num-Cameras: 4
Num-Projectors: 0
Nonlinear-Parameters: 50    0    1    0    0    0
Nonlinear-Update: 1   0   1   0   0   0
Initial-Tolerance: 10
Do-Global-Iterations: 1
Global-Iteration-Threshold: 0.5
Global-Iteration-Max: 5
Num-Cameras-Fill: 2
Do-Bundle-Adjustment: 1
Undo-Radial: 0
Min-Points-Value: 30
N-Tuples: 3
Square-Pixels: 1
Use-Nth-Frame: 2
Align-Existing: 0

[Multimedia]
N1: 1.0
N2: 1.5
D:  5.0
N3: 1.333
Z-Glass-Top: 0.0
```

---

## `[Paths]` Section *(optional)*

### `Data`
**Type:** string (directory path)

Path to the folder containing the input data files (`points.dat`, `IdMat.dat`,
`Res.dat`). Also used as the output directory for calibration results.

```ini
[Paths]
Data: C:/my_data/experiment_01/
```

> **Note:** If this section is omitted, both Python and MATLAB default to the
> directory containing the cfg file itself. This is the recommended approach —
> place the cfg file in the data folder and omit `[Paths]` entirely to avoid
> platform-specific path issues.

---

## `[Files]` Section

### `Basename`
**Type:** string
**Example:** `cam`, `basename`, `strawlab_test`

The base name prefix used for per-camera output files. Camera-specific files
are named `{Basename}{camera_id}.rad`, `{Basename}{camera_id}.cal`, etc.

```ini
Basename: cam
# Produces: cam1.rad, cam2.rad, cam3.rad, cam4.rad
```

### `Image-Extension`
**Type:** string
**Default:** `jpg`

File extension for camera images. Used when locating image files for
visualisation (not required for the self-calibration pipeline itself).

```ini
Image-Extension: jpg
```

---

## `[Images]` Section

### `Subpix`
**Type:** float
**Default:** `0.333`
**Recommended:** `0.5`

Scale factor for sub-pixel accuracy in LED detection. Controls the precision
of the corner/LED detection step that generates `points.dat`. A value of `0.5`
is a good compromise between speed and accuracy. For high-resolution images or
larger LEDs, try `1.0`.

```ini
Subpix: 0.5
```

### `Resolution` *(optional)*
**Type:** two integers
**Default:** `640 480`

Default image resolution `[width height]` in pixels. Used as a fallback when
`Res.dat` is not present in the data folder. If your dataset includes a `Res.dat`
file, this setting is ignored.

```ini
Resolution: 752 480
```

---

## `[Calibration]` Section

This is the most important section. All parameters that control the calibration
algorithm are defined here.

---

### `Num-Cameras`
**Type:** integer
**Required:** yes

The total number of cameras in the rig. **Must match the actual number of cameras
in `points.dat`** (`points.dat` has `3 × Num-Cameras` rows). If there is a
mismatch, Python will raise a clear error message telling you the correct value.

```ini
Num-Cameras: 4
```

> **Validation:** Python checks `Num-Cameras` against `points.dat` row count on
> startup and raises:
> ```
> CONFIG / DATA MISMATCH
> Config says: 4 cameras
> points.dat has: 3 cameras (9 rows ÷ 3 = 3)
> Please fix: Num-Cameras: 3
> ```

---

### `Num-Projectors`
**Type:** integer
**Default:** `0`

Number of projectors in the rig (used in mixed camera+projector setups).
For standard camera-only rigs, always set to `0`.

```ini
Num-Projectors: 0
```

---

### `Nonlinear-Parameters`
**Type:** 6 floats
**Default:** `50 0 1 0 0 0`

Initial values for the non-linear intrinsic parameter estimation (used by the
CalTech toolbox in the `goradf` refinement step):

| Index | Parameter | Meaning |
|---|---|---|
| 1 | `initFOV` | Initial field of view in degrees (e.g. `50`) |
| 2 | `center_optim` | `1` = estimate principal point, `0` = fix at image centre |
| 3 | `est_dist[0]` | `1` = estimate first radial distortion coefficient `kc1` |
| 4 | `est_dist[1]` | `1` = estimate second radial distortion coefficient `kc2` |
| 5 | `est_dist[2]` | `1` = estimate tangential distortion `kc3` |
| 6 | `est_dist[3]` | `1` = estimate tangential distortion `kc4` |

```ini
Nonlinear-Parameters: 50    0    1    0    0    0
# → FOV=50°, fix principal point, estimate kc1 only
```

---

### `Nonlinear-Update`
**Type:** 6 integers (0 or 1)
**Default:** `1 0 1 0 0 0`

Controls which non-linear parameters are updated during global iterations
(when `Do-Global-Iterations: 1`). Same indexing as `Nonlinear-Parameters`.

```ini
Nonlinear-Update: 1   0   1   0   0   0
# → Allow updating FOV and kc1 during global iterations
```

---

### `Initial-Tolerance`
**Type:** float (pixels)
**Default:** `10.0`
**Typical values:** `2.0` – `10.0`

The initial RANSAC inlier tolerance in pixels. A point is accepted as an inlier
if its epipolar distance is less than this threshold.

- **Higher value** (e.g. `10.0`): More points survive RANSAC, works better with
  uncorrected radial distortion or noisier data.
- **Lower value** (e.g. `2.0`): Stricter, keeps only very clean points, requires
  better data quality.

When `Do-Global-Iterations: 1`, this tolerance is halved on each global iteration
(10 → 6.67 → 4.44 → 2.96 → ...) until it reaches `Global-Iteration-Threshold`.

```ini
Initial-Tolerance: 10
```

---

### `Do-Global-Iterations`
**Type:** boolean (`0` or `1`)
**Default:** `0`

Controls whether the pipeline runs multiple global iterations, tightening the
RANSAC tolerance each time to progressively remove more outliers and improve
accuracy.

- `0`: Run once with `Initial-Tolerance` and stop.
- `1`: Keep iterating, halving the tolerance each time, until the error is below
  `Global-Iteration-Threshold` or `Global-Iteration-Max` iterations are reached.

```ini
Do-Global-Iterations: 1
```

> **Recommended:** Set to `1` for best results. Each iteration refines the
> distortion model and removes remaining outliers, typically improving the final
> reprojection error by 20–50%.

---

### `Global-Iteration-Threshold`
**Type:** float (pixels)
**Default:** `0.5`

The target reprojection error in pixels. Global iterations stop when the mean
2D reprojection error across all cameras falls below this value.

```ini
Global-Iteration-Threshold: 0.5
```

---

### `Global-Iteration-Max`
**Type:** integer
**Default:** `5`

Maximum number of global iterations regardless of whether the threshold has
been reached. Prevents infinite loops on difficult datasets.

```ini
Global-Iteration-Max: 5
```

---

### `Num-Cameras-Fill`
**Type:** integer
**Default:** `0`

The maximum number of cameras that may miss a 3D point and still have that
point included in the reconstruction. Controls how aggressively the
Martinec-Pajdla missing data filling algorithm works.

- `0`: Only use points visible in **all** cameras (most conservative).
- `1`: Allow 1 camera to miss a point (includes more points).
- `CAMS - 3`: Maximum useful value — always requires at least 3 cameras to
  see each point (minimum for reconstruction).

```ini
Num-Cameras-Fill: 2
# → Points visible in at least (Num-Cameras - 2) cameras are used
```

> **Trade-off:** Higher values give more points and better coverage but
> increase the risk of incorrectly filled (hallucinated) measurements.
> For 4 cameras, `Num-Cameras-Fill: 1` or `2` is typical.

---

### `Do-Bundle-Adjustment`
**Type:** boolean (`0` or `1`)
**Default:** `1`

Controls whether Levenberg-Marquardt bundle adjustment is run as a final
refinement step after the linear projective reconstruction.

- `0`: Use the linear solution only (faster but less accurate).
- `1`: Refine with BA (recommended — typically improves error by 10–30%).

```ini
Do-Bundle-Adjustment: 1
```

---

### `Undo-Radial`
**Type:** boolean (`0` or `1`)
**Default:** `0`

Whether to correct for radial lens distortion before running the self-calibration.
Requires per-camera `.rad` files (named `{Basename}{id}.rad`) to be present in
the data folder.

- `0`: No distortion correction (use for cameras with low distortion, or on the
  first run when `.rad` files are not yet available).
- `1`: Undistort using the parameters from `.rad` files before RANSAC and
  reconstruction.

```ini
Undo-Radial: 0
```

> **Workflow:** Run with `Undo-Radial: 0` first to generate `.rad` files via
> `goradf`. Then set `Undo-Radial: 1` and `Do-Global-Iterations: 1` for
> subsequent refinement passes.

---

### `Min-Points-Value`
**Type:** integer
**Default:** `30`

Minimum number of 3D-2D correspondences required per camera to consider the
calibration reliable. If any camera has fewer visible points than this after
RANSAC filtering, a warning is raised.

```ini
Min-Points-Value: 30
```

---

### `N-Tuples`
**Type:** integer
**Default:** `3`
**Valid range:** `2` – `5`

Size of the camera tuples used in the null-space construction step of the
Martinec-Pajdla algorithm. Larger values use more camera combinations for
a more robust null space but require more memory and computation.

```ini
N-Tuples: 3
```

---

### `Square-Pixels`
**Type:** boolean (`0` or `1`)
**Default:** `1`

Assumption about pixel aspect ratio:

- `1`: Square pixels (aspect ratio = 1, `fx = fy`). Correct for virtually all
  modern digital cameras.
- `0`: Allow non-square pixels (estimates `fx` and `fy` independently). Only
  needed for very old or specialised sensors.

```ini
Square-Pixels: 1
```

---

### `Use-Nth-Frame`
**Type:** integer
**Default:** `1`

Frame downsampling factor. Only every N-th frame from `points.dat` is used:

- `1`: Use all frames (maximum data, best quality).
- `2`: Use every 2nd frame (half the data).
- `5`: Use every 5th frame (20% of data, fastest).

```ini
Use-Nth-Frame: 2
```

> **Guidance:**
> - For datasets with many frames (> 500), `Use-Nth-Frame: 2` or `3` gives
>   a good speed/quality trade-off.
> - For datasets with few frames (< 200), always use `1`.
> - For quick test runs, use `5` or higher.
> - **Important:** Both MATLAB and Python must use the same value to produce
>   comparable results. Mismatched values are the most common cause of
>   unexplained result differences between the two tools.

---

### `Align-Existing`
**Type:** boolean (`0` or `1`)
**Default:** `0`

Whether to align the reconstructed camera coordinate frame with a
pre-existing set of camera centres. Requires an `original_cam_centers.dat`
file in the data folder.

- `0`: No alignment (standard self-calibration).
- `1`: Align with existing camera centres after reconstruction.

```ini
Align-Existing: 0
```

---

## `[Multimedia]` Section *(optional)*

This section enables the **multimedia geometry correction** described by
Maas (1996) for experiments where cameras in air observe a volume through
a glass wall into water or another fluid. When this section is absent (the
default), the pipeline uses the standard pinhole model — no behaviour change
for air-only experiments.

**How it works:** Light from each particle crosses three optical media —
water → glass → air — and refracts at each interface according to Snell's
Law. Without correction this refraction causes a systematic radial outward
shift of every apparent particle position. The `[Multimedia]` section tells
the pipeline the refractive-index stack so that shift can be computed and
removed before calibration and tracking.

**To disable:** Delete the section entirely, or set `N2: 1.0` and `N3: 1.0`.
The `config.use_multimedia` property returns `False` in that case and every
multimedia call is skipped — there is no performance penalty.

---

### `N1`
**Type:** float
**Default:** `1.0`
**Physical meaning:** Refractive index of the **camera-side medium** (air).

Almost always `1.0`. Change only if the cameras themselves are submerged or
surrounded by a medium other than air.

```ini
N1: 1.0
```

---

### `N2`
**Type:** one or more space-separated floats
**Default:** `1.0` (no glass — multimedia disabled)
**OpenPTV default:** `1.5`
**Physical meaning:** Refractive index of each **glass layer** in the wall.

For a single glass pane (the most common case) give one value. For a
double-pane wall (e.g. two acrylic sheets with an air gap) give two
space-separated values. The number of values must equal the number of
values in `D`.

| Material | Typical n |
|---|---|
| Borosilicate glass | 1.50 |
| Soda-lime glass | 1.52 |
| Acrylic / Perspex | 1.49 |
| Fused silica (quartz) | 1.46 |

```ini
# Single glass pane
N2: 1.5

# Double pane (two acrylic sheets)
N2: 1.49   1.49
```

> **Important:** `N2` and `D` must always have the same number of values.
> Python will raise a `ValueError` if they differ.

---

### `D`
**Type:** one or more space-separated floats (millimetres)
**Default:** `0.0`
**OpenPTV default:** `5.0`
**Physical meaning:** Thickness of each glass layer in **millimetres**.

One value per layer, matching the entries in `N2`. Measure the actual wall
thickness with a calliper — errors of even 0.5 mm in `D` produce measurable
systematic error in the Z-coordinate (depth direction).

```ini
# 5 mm single pane
D: 5.0

# Two 4 mm panes
D: 4.0   4.0
```

---

### `N3`
**Type:** float
**Default:** `1.0` (air — multimedia disabled)
**OpenPTV default:** `1.333`
**Physical meaning:** Refractive index of the **object-side medium** (the fluid
the particles are suspended in).

| Fluid | Typical n at 20 °C, 589 nm |
|---|---|
| Water | 1.333 |
| Glycerol 50% | 1.395 |
| Glycerol 100% | 1.474 |
| Mineral oil | 1.46–1.48 |
| Air | 1.000 |

```ini
# Water
N3: 1.333

# 50 % glycerol solution
N3: 1.395
```

> **Note:** The refractive index of water varies with temperature (~0.001 per
> °C near 20 °C) and with dissolved salts. For high-precision work, measure n
> with a refractometer at the actual experimental temperature.

---

### `Z-Glass-Top`
**Type:** float (millimetres)
**Default:** `0.0`
**Physical meaning:** Z-coordinate of the **air/glass interface** (top surface
of the glass wall) in the world coordinate system.

This value defines where the glass wall sits along the optical axis in your
experiment's coordinate frame. The glass bottom surface is automatically
computed as `Z-Glass-Top − sum(D)`.

- If you define your world coordinate system so that the glass top surface
  is at Z = 0, set `Z-Glass-Top: 0.0` (the most common choice).
- If your world origin is elsewhere (e.g. at the centre of the measurement
  volume), set this to the actual Z-coordinate of the glass surface in mm.

```ini
# Glass top at world origin (most common)
Z-Glass-Top: 0.0

# Glass top 300 mm above world origin
Z-Glass-Top: 300.0
```

---

### Multi-layer example

For a double acrylic wall (two 4 mm sheets, 50% glycerol solution):

```ini
[Multimedia]
N1: 1.0
N2: 1.49   1.49
D:  4.0    4.0
N3: 1.395
Z-Glass-Top: 0.0
```

---

## Input Data Files

The pipeline expects the following files in the data folder:

| File | Shape | Description |
|---|---|---|
| `points.dat` | `(3×CAMS, FRAMES)` | Joint image matrix. Each group of 3 rows is `[x, y, 1]` homogeneous coordinates for one camera. Missing detections are `NaN`. |
| `IdMat.dat` | `(CAMS, FRAMES)` | Visibility matrix. `1` = point detected in this camera/frame, `0` = not detected. |
| `Res.dat` | `(CAMS, 2)` | Image resolutions. Each row is `[width, height]` for one camera. Optional — falls back to `Resolution` from cfg if absent. |
| `{Basename}{id}.rad` | text | Per-camera radial distortion parameters (output of `goradf`, input for `Undo-Radial: 1`). Optional. |

---

## Output Files

After a successful run the pipeline writes these files to the data folder:

| File | Description |
|---|---|
| `Pmats.dat` | Stacked Euclidean projection matrices `(3×CAMS, 4)` |
| `Xe.dat` | Reconstructed 3D points `(4, N)` in homogeneous coordinates |
| `Re.dat` | Rotation matrices stacked `(3×CAMS, 3)` |
| `Ce.dat` | Camera centres `(4, CAMS)` in homogeneous coordinates |
| `{Basename}{id}.rad` | Estimated intrinsic parameters + distortion per camera |
| `points4cal{id}.dat` | 3D-2D correspondences per camera for further processing |
| `reprerrors.png` | Line plot of mean and std reprojection error per camera |
| `reprerrors_bars.png` | Bar chart of mean and std reprojection error per camera |

---

## Recommended Settings by Use Case

### Quick test run
```ini
Use-Nth-Frame: 5
Do-Global-Iterations: 0
Do-Bundle-Adjustment: 1
Initial-Tolerance: 10
```

### Standard production run
```ini
Use-Nth-Frame: 1
Do-Global-Iterations: 1
Global-Iteration-Threshold: 0.5
Global-Iteration-Max: 5
Do-Bundle-Adjustment: 1
Initial-Tolerance: 10
Num-Cameras-Fill: 1
```

### High-distortion cameras (first pass, no rad files yet)
```ini
Initial-Tolerance: 15
Undo-Radial: 0
Do-Global-Iterations: 0
Num-Cameras-Fill: 2
```

### High-distortion cameras (subsequent pass, rad files available)
```ini
Initial-Tolerance: 5
Undo-Radial: 1
Do-Global-Iterations: 1
Global-Iteration-Threshold: 0.5
Nonlinear-Parameters: 50    0    1    1    0    0
```

### Through-glass / underwater PTV (standard water tank)
Add the `[Multimedia]` section alongside any of the above. These values match
the OpenPTV defaults and are the correct starting point for a typical
borosilicate or acrylic tank experiment with water.

```ini
[Multimedia]
N1: 1.0
N2: 1.5
D:  5.0
N3: 1.333
Z-Glass-Top: 0.0
```

> **Workflow for multimedia experiments:**
> 1. Run without `[Multimedia]` first (or with `N2: 1.0`, `N3: 1.0`) to get
>    an initial linear calibration.
> 2. Add the `[Multimedia]` section with measured glass thickness and fluid
>    refractive index.
> 3. Re-run with `Do-Global-Iterations: 1` and `Undo-Radial: 1` — the bundle
>    adjustment will now use the multimedia functional model (Maas eq. 17).
> 4. Check `summary.txt` in the Results folder to confirm multimedia was
>    detected and the correct values were used.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `CONFIG / DATA MISMATCH` error | `Num-Cameras` in cfg doesn't match `points.dat` row count | Set `Num-Cameras` to the value shown in the error message |
| Very high reprojection error (> 3px) | `Use-Nth-Frame` too high — too few points | Reduce `Use-Nth-Frame` to `1` or `2` |
| Only 1 outlier-removal pass | Old code with `break` in loop | Update `GoCal.py` to the latest version |
| MATLAB gives different results to Python | Stale MATLAB workspace | Run `clear all` in MATLAB before each run |
| `Not enough points` error | `Num-Cameras-Fill` too low | Increase `Num-Cameras-Fill` by 1 |
| BA diverges | `Initial-Tolerance` too tight for data quality | Increase `Initial-Tolerance` (try `15` or `20`) |
| Camera `.rad` file not found warning | `Basename` in cfg doesn't match actual file names | Check that `{Basename}{id}.rad` files exist in the data folder |
| Multimedia correction not applied | `[Multimedia]` section absent or `N2`/`N3` left at `1.0` | Add the section with correct values; check `summary.txt` for "ENABLED" |
| `ValueError: n2 and d must have the same length` | `N2` and `D` have different numbers of values | Count the space-separated values — they must match exactly |
| Z-coordinate errors ~2× worse than X/Y | Expected behaviour in multimedia geometry (Maas 1996) | This is a physical limitation, not a bug — rays bend toward denser medium, reducing depth resolution |
| Reprojection error higher after adding `[Multimedia]` | `D` (glass thickness) is wrong | Measure glass thickness with a calliper; even 1 mm error matters |
| `RuntimeError: multimedia_params.py is not on the Python path` | Module file missing | Place `multimedia_params.py` in the same folder as `CalibrationSystem.py` |
