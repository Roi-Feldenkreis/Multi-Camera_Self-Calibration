# Multi-Camera_Self-Calibration

Target-free self-calibration of a multi-camera rig for 3-D Particle Tracking Velocimetry (PTV),
including cameras that look into water through a flat glass wall.

The project combines two methods:

- **Svoboda — Quick Guide to Multi-Camera Self-Calibration** (Computer Vision Lab, ETH Zürich,
  BiWi-TR-263, 2003): projective self-calibration from a moving point (or particle cloud),
  with Martinec–Pajdla filling of missing observations, RANSAC outlier rejection, Euclidean
  upgrade and bundle adjustment.
- **Maas — Contributions of Digital Photogrammetry to 3-D PTV** (ETH Zürich, 1996): the
  multimedia model for refraction through an air / glass / water interface.

The original MATLAB MultiCamSelfCal and the Caltech camera-calibration toolbox are ported to
Python, and the multimedia model is built into the reconstruction and the bundle adjustment.

## How it works

1. **Input**: per-camera 2-D point observations (`Points.dat`, `IdMat.dat`, `Res.dat`) and a
   `multicamselfcal.cfg` describing the rig. For real recordings these files are produced from
   the image sequences by `FindingPoints/im2points_real.py` (see below).
2. **Outlier rejection**: pairwise RANSAC on epipolar geometry (`RANSAC_Algorithm/`).
3. **Projective reconstruction**: Martinec–Pajdla factorisation with missing data
   (`MartinecPajdla/`).
4. **De-refraction bootstrap** (multimedia configs): image points are corrected for refraction
   with the Maas radial-shift model, using a look-up table (`Multimedia/Multimedia_lut.py`),
   and the seed geometry is checked for mirror solutions (`seed_gatekeeper.py`,
   signed-volume chirality test).
5. **Euclidean upgrade and metric anchor**: self-calibration to a Euclidean frame; scale is fixed
   from the measured camera-to-glass distances (`CoreFunctions/Derefraction.py`).
6. **Multimedia bundle adjustment**: Levenberg–Marquardt over cameras and particles with the
   refraction model and an inter-camera distance penalty (`Multimedia/multimedia_bundle.py`).
   A single frame authority (`Multimedia/Frame_convention.py`) keeps all modules in one frame.
7. **Output**: camera matrices and centres (`camN.cal`, `camN.Pmat.cal`, `Pst.dat`, `Cst.dat`),
   reprojection-error plots and, on synthetic data, a ground-truth alignment report.

Without a `[Multimedia]` section the refraction steps are skipped and the classic Svoboda pipeline runs.

## Quick start

```
pip install -r requirements.txt
```

1. Point `config_paths.txt` at a dataset:
   ```
   config_file=<root>\TestSystem\<group>\<dataset>\multicamselfcal.cfg
   data_folder=<root>\TestSystem\<group>\<dataset>
   ```
2. From the project root run `python GoCal_derefract.py`.

Results are written into the dataset folder. Refraction is enabled by the `[Multimedia]` section
of the `.cfg` (refractive indices and glass thickness, `Camera-Glass-Distance`, LUT resolution
`LUT-nR` / `LUT-nZp` / `Zp-Max`, intrinsics mode). See
`Documentation/Configuration File Guide.md` for all keys.

## Finding points in real images

`FindingPoints/im2points_real.py` turns synchronized image sequences (one folder per camera)
into the pipeline input files. It detects a single bright target (submerged LED or bead) in
every frame, following Svoboda's im2points recipe:

- background model from separate background frames (`--bg`), or from the sequence itself;
- blob mask `diff > thr_frac * max(diff)`, rejected if the peak is below `k_sigma` x the
  background noise, or if there is more than one blob, or it is too small, too large or too
  elongated (motion blur);
- sub-pixel position from a 2-D Gaussian fit (fallback: weighted centroid).

It writes `IdMat.dat`, `Points.dat` and `Res.dat` to `--out`. Points are stored raw (distorted
and refracted), because `GoCal_derefract.py` corrects both. After export it runs sanity checks
into `<out>/checks/`: a report, sensor-coverage plots and spot-check overlays of detections.

```
python FindingPoints\im2points_real.py ^
    --cam "D:\exp1\cam1\*.tif" --cam "D:\exp1\cam2\*.tif" ^
    --cam "D:\exp1\cam3\*.tif" --cam "D:\exp1\cam4\*.tif" ^
    --bg  "D:\exp1\bg1\*.tif"  --bg  "D:\exp1\bg2\*.tif" ^
    --bg  "D:\exp1\bg3\*.tif"  --bg  "D:\exp1\bg4\*.tif" ^
    --out "D:\exp1\calib_input" --one-based
```

Give one `--cam` per camera, in pipeline camera order. Detection parameters can be set on the
command line (`--thr-frac`, `--k-sigma`, `--min-peak`, `--min-area`, `--max-area`, `--max-ecc`,
`--win-half`, `--subpix gauss|centroid`, `--channel`, `--bg-max-frames`); `--workers N` runs
cameras in parallel and `--no-checks` skips the sanity checks.

To test the detector, `FindingPoints/make_test.py` renders a synthetic 4-camera LED sequence
with known positions and reflections into `t/`; `FindingPoints/compare.py <out_dir> [<out_dir2>]`
scores detection outputs against that ground truth (missed and false detections, RMS error)
and compares two runs.

## Layout

| Path | Contents |
|---|---|
| `GoCal_derefract.py` | Main pipeline entry point |
| `seed_gatekeeper.py` | Seed chirality/geometry gatekeeper (imported by the pipeline) |
| `config_paths.txt` | Which dataset the pipeline runs on |
| `CommonCfgAndIO/` | Config loading and I/O (`CalibrationSystem`) |
| `CoreFunctions/` | Self-calibration core: Euclidize, inliers/outliers, de-refraction, anchors |
| `MartinecPajdla/` | Projective reconstruction with missing data (fill_mm and helpers) |
| `Multimedia/` | Refraction model: Snell core, LUT, radial shift, multimedia bundle adjustment, frame convention |
| `CalTechToolBoxPython/` | Caltech toolbox port (non-linear distortion model, `goradf`) |
| `RadialDistortions/` | Distortion undo (Oulu / Heikkilä models, `.rad` reader) |
| `RANSAC_Algorithm/` | Pairwise RANSAC for epipolar outlier rejection |
| `OutputFunctions/` | Saving calibration, reprojection-error plots, alignment and scene plots |
| `FindingPoints/` | Point detection: `im2points_real.py` (real-data front end), Svoboda ports `im2points.py` / `getpoint.py` / `im2imstat.py` / `mycorr2.py`, `ShowPoints.py` overlays, `make_test.py` + `compare.py` synthetic detector test |
| `LocalAlignments/`, `Align and test/` | Aligning the result to a world frame |
| `Documentation/` | Module explanations, config guide, bug log, run-result summaries |
| `scripts/data_generation/` | Synthetic dataset builders and config editing (`set_cfg_key.py`) |
| `scripts/tests/` | Numerical equivalence / sensitivity tests for the multimedia model |
| `scripts/experiments/` | `.bat` runners for the thesis experiment series (logs go to `logs/<group>/`) |
| `TestSystem/` | Datasets, grouped by experiment (below) |

### TestSystem groups

| Group | Datasets |
|---|---|
| `reference/` | `Media test`, `DATA20100906_134124`, `caldata20130726_122220`, `OpenPTV_cavity` |
| `base/` | `caldata_flipped_step2*`, `caldata_refracted_step3*`, `caldata_refracted_wide`, `caldata_refracted_wide400` |
| `spread/` | Point-cloud spread seeds `*_wide400_s2..s5` |
| `rig/` | Steeper rig `*_r07*` |
| `realism/` | Realistic distance/LOS inputs `*_real` |
| `intrinsics/` | Intrinsics estimation `*_s1_est`, `*_s1_estfs`, `*_s1_radlos` |
| `cause/` | Noise-cause isolation `*_c1n4`, `*_c4n0`, `*_c4n123` |
| `monte_carlo_results/` | MATLAB-vs-Python Monte Carlo comparison runs |

## Scripts

Run everything from the project root, e.g.
`python scripts\data_generation\generate_refracted_caldata.py TestSystem\base\caldata_flipped_step2 TestSystem\base\caldata_refracted_step3`.
The experiment `.bat` files `cd` to the project root themselves.

## Scripts

Run from the project root, for example
`python scripts\data_generation\generate_refracted_caldata.py TestSystem\base\caldata_flipped_step2 TestSystem\base\caldata_refracted_step3`.
The `.bat` files in `scripts/experiments/` reproduce the experiment series; each one `cd`s to
the project root, edits the dataset configs with `set_cfg_key.py`, runs the pipeline and writes
its log to `logs/<group>/` (not tracked by git).

## Requirements

Python 3.11 with NumPy, SciPy, Matplotlib, OpenCV, Pillow and scikit-image
(`requirements.txt`). Developed on Windows 11 / Anaconda.

## References

- T. Svoboda, D. Martinec, T. Pajdla, "A convenient multi-camera self-calibration for virtual
  environments", PRESENCE 14(4), 2005; and the MultiCamSelfCal quick guide (BiWi-TR-263, 2003).
- D. Martinec, T. Pajdla, "Structure from many perspective images with occlusions", ECCV 2002.
- H.-G. Maas, "Contributions of digital photogrammetry to 3-D PTV", 1996.
- J. Heikkilä, O. Silvén, "A four-step camera calibration procedure with implicit image
  correction", CVPR 1997.
- P. Sturm, B. Triggs, "A factorization based algorithm for multi-image projective structure
  and motion", ECCV 1996.
