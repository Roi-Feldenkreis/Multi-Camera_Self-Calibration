# `goradf` — Function Explanation

## What does it do?

`goradf` is the **final function in the complete calibration pipeline**. It is
the single entry point that a user calls after running MultiCamSelfCal and
MultiCamValidation. For each camera in a multi-camera rig, it:

1. Loads the 3D↔2D point correspondences from a BlueC `.rad` file
2. Runs the full joint intrinsic+extrinsic calibration
3. Validates the result and falls back to a safe default if calibration failed
4. Saves distortion visualisation plots
5. Writes the final camera matrix `KK` and distortion vector `kc` to a
   BlueC-format config file

Everything below this function in the pipeline is a subroutine.

---

## Where it fits — the complete pipeline

```
MultiCamSelfCal  (external — produces .rad files)
MultiCamValidation
    |
    v
goradf(config, par2estimate, INL_TOL)   <-- THIS FUNCTION
    |
    +-- for each camera i:
    |
    +-- preparedata(cam_i.rad)
    |       reads BlueC ASCII file -> X (3,N), x (2,N)
    |
    +-- go_calib_optim_iter(x_list, X_list, ...)
    |       the full CalTech pipeline:
    |         check_active_images
    |         comp_ext_calib
    |             compute_extrinsic_init + compute_extrinsic_refine
    |                 project_points2
    |                     rigid_motion -> rodrigues
    |         Gauss-Newton loop (project_points2, compute_extrinsic_refine)
    |         comp_error_calib
    |         extract_parameters
    |
    +-- validate: NaN check + err_std > 2*INL_TOL -> fallback KK
    |
    +-- visualize_distortions(...)       save PNG plots per camera
    |       apply_distortion x3 (complete / tangential / radial)
    |
    +-- _write_bluec_config(...)         write KK and kc to ASCII file
    |
    v
selfcalib dict: goradproblem flag, KK_list, kc_list, results per camera
```

---

## Previously Converted Functions and Their Roles Here

| Function | Role in goradf |
|---|---|
| `preparedata` | Called once per camera at the start of the loop. Reads the BlueC `.rad` ASCII file and returns `X (3,N)` and `x (2,N)` — the raw inputs for calibration |
| `go_calib_optim_iter` | The calibration engine — called once per camera with a single-image `x_list`/`X_list`. Returns `KK`, `kc`, `fc`, `cc`, `err_std`, `solution`, and all uncertainties. Internally chains every other function in the toolbox |
| `visualize_distortions` | Called after a successful calibration. Generates three distortion field plots (complete / tangential / radial) and saves them as PNG files using the `save_path` prefix |

---

## Inputs & Outputs

### `par2estimate` — the 6-element parameter control vector

| Index | Value | What it controls |
|-------|-------|-----------------|
| 0 | `initFOV` | Initial camera field of view in degrees (used to seed focal length) |
| 1 | `center_optim` | 1 = estimate principal point, 0 = fix at image centre |
| 2–5 | `est_dist[0:4]` | Which of the first 4 distortion coefficients to estimate |

The 5th kc coefficient (6th-order radial) is always fixed to 0 — BlueC only
stores 4 distortion values, so `est_dist` is padded with a 0 before being
passed to `go_calib_optim_iter`.

### `config` — the camera rig configuration

```python
config = {
    'cal': {
        'cams2use': [1, 2, 3],         # camera indices to calibrate
        'Res': np.array([[1280, 960],   # image resolution per camera
                         [1280, 960],
                         [1280, 960]]),
    },
    'files': {
        'points4cal': '/data/cam%d.rad',  # input: 3D/2D correspondences
        'rad':        '/out/cam%d.cfg',   # output: KK + kc config file
    },
    'paths': {
        'data': '/out/',                  # directory for distortion plots
    },
}
```

### Return value — `selfcalib` dict

| Key | Type | Description |
|-----|------|-------------|
| `goradproblem` | int | 0 = all cameras calibrated OK; 1 = at least one failed |
| `results` | dict[i → dict] | Full `go_calib_optim_iter` output per camera |
| `KK_list` | dict[i → (3,3)] | Final camera matrix per camera |
| `kc_list` | dict[i → (4,)] | First 4 distortion coefficients per camera |

---

## The Failure Detection and Fallback

```python
if param_has_nan or np.any(err_std > 2 * INL_TOL):
    KK = [[700, 0, 320], [0, 700, 240], [0, 0, 1]]   # safe default
    kc = zeros(5)
    selfcalib['goradproblem'] = 1
```

Two failure conditions trigger the fallback:
- **NaN in solution vector** — the Gauss-Newton optimiser diverged
- **err_std > 2·INL_TOL** — reprojection error is too large to trust

The fallback uses `fc=700px`, `cc=(320,240)` (a reasonable guess for a
640×480 camera) and zero distortion. This is deliberately conservative —
a bad calibration is worse than no calibration, and the downstream system
can at least function with this default.

---

## The BlueC Config File Format

The output file stores the 9 elements of `K` and 4 distortion coefficients
as key-value pairs with 16-decimal-place precision:

```
K11 = 823.4500000000000000
K12 = 0.0000000000000000
K13 = 319.5000000000000000
...
K33 = 1.0000000000000000

kc1 = -0.2500000000000000
kc2 =  0.1200000000000000
kc3 =  0.0000000000000000
kc4 =  0.0000000000000000
```

Only `kc1..kc4` are written because the BlueC format predates the 5-parameter
distortion model. The 5th coefficient is used internally during calibration
but not exported.

---

## Key MATLAB → Python Differences

| Topic | MATLAB | Python |
|-------|--------|--------|
| `go_calib_optim_iter` | A script — writes to shared workspace variables | A function — results returned as a dict |
| `config.cal.cams2use` | MATLAB struct field access | Python dict: `config['cal']['cams2use']` |
| `sprintf(config.files.points4cal, i)` | `sprintf` format string | Python `%` string formatting: `config['files']['points4cal'] % i` |
| `visualize_distortions` | Script — reads workspace, saves EPS | Function — takes explicit args, saves PNG |
| `print -depsc file.eps` | MATLAB print to EPS | `save_path` kwarg in `visualize_distortions` |
| `clear fc kc alpha_c cc nx ny` | Clears workspace variables to avoid contaminating next camera | Naturally scoped — Python local variables are cleared at loop end |
| `fopen/fclose/fprintf(fid,...)` | C-style file I/O | Python `with open(...) as fid` context manager |
| Octave branch | `if ~Octave, visualize..., end` | Single code path — Octave compatibility not needed in Python |

The most important difference is the elimination of the **shared workspace**.
In MATLAB, `go_calib_optim_iter` is a script that reads and writes variables
like `fc`, `cc`, `kc`, `KK`, `err_std` directly into the caller's workspace.
This means `goradf.m` must explicitly `clear` those variables at the end of
each camera loop to prevent one camera's result contaminating the next.
In Python, function scope makes this impossible — each call to
`go_calib_optim_iter` returns a fresh dict, and there is no shared state.
