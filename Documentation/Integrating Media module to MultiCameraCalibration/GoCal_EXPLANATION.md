# GoCal.py — Multimedia Geometry Changes Explanation

## What Was Changed

`GoCal.py` is the main pipeline orchestrator. Five additions integrate the
Maas (1996) multimedia geometry correction without modifying any existing logic.

---

## Overview of Changes

```
load_config / load_data
  ↓
[NEW] Import multimedia modules (guarded try/except)
[NEW] Build mm = MultimediaParams from config
  ↓
Main while loop
  ↓
  Radial undistortion block
  [NEW] Apply A/D correction (Maas eq.1) to linear.Ws
  ↓
  RANSAC / pack / cam structures     ← unchanged
  ↓
  Inner outlier loop
    fill_mm_bundle → euclidize
    [NEW] Build CameraParams + MultiCameraLUT from Ce, Re
    reprerror (now receives cameras_mm, lut)
  ↓
  Final Bundle Adjustment
    fill_mm_bundle → euclidize
    [NEW] Rebuild CameraParams + LUT from refined Ce, Re
    reprerror (now receives cameras_mm, lut)
  ↓
  Save camera matrices
  [NEW] Save multimedia_params.json
```

---

## Change 1 — Module Import

### Location
Top of file, after existing imports.

### What
```python
    from Iterative_radial_shift  import apply_ad_correction
    from multimedia_lut import MultiCameraLUT
    from multimedia_params import CameraParams
```


---

## Change 2 — One-Time Setup

### Location
Inside `main()`, after `load_data()`, before the `while` loop.

### What
```python
if config.use_multimedia:
    mm = config.to_multimedia_params()
else:
    mm = None

lut        = None   # populated after first euclidize()
cameras_mm = None   # populated after first euclidize()
```

### Why
`mm` (MultimediaParams) is built once from the `[Multimedia]` section of
the `.cfg` file. `lut` and `cameras_mm` cannot be built yet because camera
positions are not metric until after `euclidize()`.

---

## Change 3 — A/D Correction on linear.Ws

### Location
Inside the existing `if UNDO_RADIAL and INT_RADIAL_PARAM:` block, inside
the per-camera loop, immediately after `undoradial()`.

### What
```python
if mm is not None:
    raw = data.camera_parameters[i].get('raw_params', {})
    a1  = float(raw.get('a1', 0.0))
    a2  = float(raw.get('a2', 0.0))
    if abs(a1) > 1e-10 or abs(a2) > 1e-10:
        x_row = linear.Ws[3*i,   :]
        y_row = linear.Ws[3*i+1, :]
        linear.Ws[3*i,   :] = (1.0 + a1) * x_row + a2 * y_row
```

### Why
`linear.Ws` is the single coordinate matrix used by every downstream step:
RANSAC, `fill_mm_bundle`, `reprerror`. Applying the Maas eq.(1) A/D
correction here propagates it automatically to everything that follows.

On iteration 0 `a1=a2=0` (no `.rad` files yet), so this block is a no-op.

### What is NOT corrected here
The multimedia radial shift ΔR (Maas eqs.4–8) cannot be applied to
`linear.Ws` because it depends on the 3D position of each point, which does
not exist at this stage. It is applied per-point inside `reprerror`.

---

## Change 4 — CameraParams + LUT Build

### Location
Inside the `while outliers:` loop, immediately after every `euclidize()` call.
Repeated identically after the final BA `euclidize()` call.

### What
```python
if mm is not None:
    cameras_mm = _build_camera_params(Ce, Re, config, mm, CAMS)
    lut = MultiCameraLUT(cameras_mm, z_glass_top=config.z_glass_top)
    lut.build_all(
        R_max  = max(data.resolutions[:, 0]) * 0.5,
        Zp_max = 300.0,
        n_R    = 80,
        n_Zp   = 80,
        verbose= False,
    )
```

### Why
`euclidize()` is the first step that produces **metric** camera positions
(`Ce`) and orientations (`Re`). Before this point, cameras are described
only in projective space. The LUT requires real-world positions to compute
nadir points and radial distances.

`_build_camera_params()` converts `Ce` and `Re` into one `CameraParams`
object per camera, each carrying all 16 Maas parameters plus the shared
`MultimediaParams`. `MultiCameraLUT.build_all()` then sweeps an (R, Zp)
grid per camera, storing R̄/R, enabling fast radial-shift lookup during
`reprerror`.

`cameras_mm` and `lut` are updated on every `euclidize()` call so they
always reflect the current best camera geometry.

---

## Helper Function — _build_camera_params

### What
```python
def _build_camera_params(Ce, Re, config, mm, CAMS) -> list[CameraParams]
```

Converts the Euclidean reconstruction output into a list of `CameraParams`.
Rotation matrices are converted to Euler angles (ω, φ, κ) using SciPy if
available, with an atan2 fallback. Intrinsics (`c`, `xh`, `yh`) and
distortion coefficients come from the `.rad` files in
`config._camera_parameters`.

---

## Change 5 — Save multimedia_params.json

### Location
After `savecalpar()`, before the alignment block.

### What
```python
if mm is not None:
    mm_dict = {'n1': mm.n1, 'n2': mm.n2, 'd': mm.d,
               'n3': mm.n3, 'z_glass_top': config.z_glass_top}
    with open(os.path.join(config.data_path, 'multimedia_params.json'), 'w') as f:
        json.dump(mm_dict, f, indent=2)
```

### Why
The multimedia parameters are part of the calibration result. Any downstream
particle tracking system needs them to reproduce the same refraction
correction. Saving them alongside `Pmats.dat` makes the output folder
self-contained.

---

## What Was Not Changed

| Component | Reason |
|---|---|
| `fill_mm_bundle` calls | Operates on 2D image coordinates — no multimedia geometry |
| `findinl` / RANSAC | Operates on already-corrected `linear.Ws` |
| `findoutl` | Reads `cam[i]['xe']` from standard pinhole projection — unchanged |
| `cam_dict` loop | Data carrier only — no camera geometry |
| `evalreprerror` | Post-processing — unchanged |
| `savecalpar` | Saves projection matrices — unchanged |

---

## Backward Compatibility

When the `[Multimedia]` section is absent from the `.cfg` file:
- `config.use_multimedia` returns `False`
- `mm = None`
- Every `if mm is not None:` block is skipped
- The pipeline runs identically to the original

---

## Parameter Tuning

| Parameter | Default | Notes |
|---|---|---|
| `R_max` | `max(resolution) * 0.5` | Adjust to the actual observation volume radius in mm |
| `Zp_max` | `300.0` | Maximum particle depth below the glass bottom surface in mm |
| `n_R`, `n_Zp` | `80` | LUT grid resolution. 80×80 gives < 0.0002 mm interpolation error |

---

## References

- Maas, H.-G. (1996). *Contributions of Digital Photogrammetry to 3-D PTV*.
  Eqs. (1)–(8), (17).
- See also: `GoCal_EXPLANATION.md`, `EstimateReprojectionError_EXPLANATION.md`,
  `CalibrationSystem_EXPLANATION.md`
