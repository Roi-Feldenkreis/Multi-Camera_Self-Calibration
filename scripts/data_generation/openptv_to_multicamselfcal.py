"""
openptv_to_multicamselfcal.py
==============================

Convert an OpenPTV ``test_cavity``-style calibration dataset into the file
format expected by our Multi-Camera Self-Calibration pipeline.

WHAT YOU NEED (from an OpenPTV experiment folder)
--------------------------------------------------
``cal/`` directory containing:
    calblock.txt          — 3D coordinates of the calibration target
                             dots (cols: id, X, Y, Z in mm)
    cam{N}.tif_targets    — detected 2D centroids in pixels for camera N
                             (cols: id, x, y, n, nx, ny, sumg, tnr) — we only
                             need columns x and y, indexed by the same point
                             ID as in calblock.txt
    cam{N}.ori            — (optional, for ground-truth comparison)
                             camera position+orientation and refractive
                             glass plane
``parameters/`` directory containing:
    ptv.par               — main parameter file (image resolution, n cameras)
    cal_ori.par           — calibration parameters
    multimedia.par        — multimedia geometry (n1, n2, n3, d)

NOTE on point-correspondence assumption
----------------------------------------
We assume calblock.txt enumerates the calibration target points 1..N and
each cam{N}.tif_targets file uses the SAME point IDs in its first column
(this is the standard OpenPTV convention after sortgrid).  If your dataset
has been through OpenPTV's sortgrid step, this will be true.

If the target files don't have matched IDs, you need to run OpenPTV's
sortgrid on them first — this script doesn't do correspondence matching.

WHAT THIS SCRIPT PRODUCES
--------------------------
A folder structured for our pipeline:
    <output>/points.dat            (3K x N) homogenised 2D pixel measurements
    <output>/IdMat.dat             (K x N) visibility matrix (0/1)
    <output>/Res.dat               (K x 2) image resolutions
    <output>/ground_truth.json     ground truth for alignment_plot.py
    <output>/multicamselfcal.cfg   ready-to-run config file
    <output>/cam{N}.rad            starting intrinsics from ``cam{N}.ori``

USAGE
-----
::

    python openptv_to_multicamselfcal.py /path/to/openptv/expt /path/to/output

The first argument is the OpenPTV experiment folder (containing ``cal/``
and ``parameters/`` subdirectories).  The second is where to write the
output files for our pipeline.
"""

from __future__ import annotations
import os
import sys
import json
import argparse
import re
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# File parsers
# ---------------------------------------------------------------------------

def parse_calblock(path: Path) -> np.ndarray:
    """
    Parse OpenPTV's calblock.txt.

    Columns: id, X, Y, Z  (4 columns, whitespace-separated, mm)
    Returns
    -------
    arr : (N, 4) array with [id, X, Y, Z] per row.
    """
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] < 4:
        raise ValueError(f"calblock.txt: expected ≥4 columns, got {data.shape[1]}")
    return data[:, :4]


def parse_targets(path: Path) -> dict:
    """
    Parse OpenPTV's per-image target file (cam{N}.tif_targets).

    OpenPTV target-file columns (8 cols):
        detection_index, x_pixel, y_pixel, n, nx, ny, sumg, tnr
    where ``tnr`` is the 3D-target-point-ID assigned by sortgrid (or -1
    if the detection was not matched to a 3D point).

    The first line of the file is a header containing the total number
    of detections; we skip it if it has only one column.

    Returns
    -------
    dict mapping 3D-point-id (int) -> (x_pixel, y_pixel) in image coordinates.
    """
    if not path.exists():
        return {}

    pts = {}
    with open(path) as fh:
        for ln_idx, line in enumerate(fh):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            cols = line.split()
            # Skip the single-integer header on line 0
            if len(cols) < 8:
                continue
            try:
                x   = float(cols[1])
                y   = float(cols[2])
                tnr = int(float(cols[7]))   # 3D point id (after sortgrid)
            except ValueError:
                continue
            if tnr < 0:
                continue   # unmatched detection
            pts[tnr] = (x, y)
    return pts


def parse_ori(path: Path) -> dict | None:
    """
    Parse an OpenPTV cam{N}.ori file for ground-truth camera pose.

    File layout (from OpenPTV tutorial)::

        line 1:  x y z                     ← camera position (mm)
        line 2:  omega phi kappa           ← rotation angles (radians)
        lines 3–5: 3×3 rotation matrix     ← precomputed from angles
        line 6:  xp yp                     ← principal point offset (mm)
        line 7:  f                         ← focal length (mm)
        line 8:  glass_x glass_y glass_z   ← glass-plane position (mm)
    """
    if not path.exists():
        return None

    with open(path) as fh:
        lines = [ln.strip() for ln in fh if ln.strip() and not ln.startswith('#')]

    try:
        position   = list(map(float, lines[0].split()))[:3]
        angles_rad = list(map(float, lines[1].split()))[:3]
        R          = np.array([list(map(float, lines[2+i].split()))[:3]
                                for i in range(3)])
        xp_yp      = list(map(float, lines[5].split()))[:2]
        focal_mm   = float(lines[6].split()[0])
        glass_pos  = list(map(float, lines[7].split()))[:3]
    except (IndexError, ValueError) as e:
        print(f"  Warning: could not fully parse {path.name}: {e}")
        return None

    return {
        'position':    position,
        'angles':      angles_rad,
        'R':           R.tolist(),
        'xp_yp':       xp_yp,
        'focal_mm':    focal_mm,
        'glass_pos':   glass_pos,
    }


def parse_par_file(path: Path) -> list:
    """
    Parse an OpenPTV .par file (one value per line, no key/value pairs).

    These files are positional — the meaning of each line depends on the
    file type.  We return the raw list of values; the caller knows the
    meaning of each entry.
    """
    if not path.exists():
        return []
    out = []
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            out.append(s)
    return out


def parse_ptv_par(path: Path) -> dict:
    """
    Parse ptv.par for image dimensions, pixel size, and multimedia params.

    OpenPTV ptv.par layout (positional)::
        line 1:  n_cams
        lines 2..(1+2n):  cam image path, cal image path  (alternating)
        line 2+2n:  hp_image (filter halfwidth)
        line 3+2n:  vp_image
        line 4+2n:  format (0=raw 1=tif)
        line 5+2n:  image_width
        line 6+2n:  image_height
        line 7+2n:  pixel_size_x (mm)
        line 8+2n:  pixel_size_y (mm)
        line 9+2n:  chfield
        line 10+2n: mmlut_flag (1 = multimedia ON)
        line 11+2n: n_water (refractive index n3)
        line 12+2n: n_glass (refractive index n2)
        line 13+2n: d_glass (glass thickness mm)
    """
    rows = parse_par_file(path)
    out = {}
    if not rows:
        return out
    try:
        n_cams = int(rows[0])
    except ValueError:
        return out
    out['n_cams'] = n_cams

    # The numeric block starts after the image+calibration paths
    paths_block_end = 1 + 2 * n_cams
    nums = []
    for r in rows[paths_block_end:]:
        try:
            nums.append(float(r))
        except ValueError:
            pass

    # Expected positional layout in the numeric block
    if len(nums) >= 6:
        out['image_width']  = int(nums[3])
        out['image_height'] = int(nums[4])
        out['pixel_size_mm'] = float(nums[5])    # x (square pixels assumed)
    if len(nums) >= 11:
        out['mmlut_flag']    = int(nums[8])
        out['n_water']       = float(nums[9])    # n3
        out['n_glass']       = float(nums[10])   # n2
    if len(nums) >= 12:
        out['d_glass']       = float(nums[11])
    return out


def parse_multimedia_par(path: Path) -> dict | None:
    """
    Parse multimedia.par if present.

    File layout::
        line 1:  n_layers   (usually 1 for a single glass wall)
        line 2:  n1         (air refractive index)
        line 3:  n2         (glass refractive index)
        line 4:  n3         (water refractive index)
        line 5:  d          (glass thickness in mm)
    """
    rows = parse_par_file(path)
    if len(rows) < 5:
        return None
    try:
        return {
            'n_layers': int(rows[0]),
            'n1':       float(rows[1]),
            'n2':       float(rows[2]),
            'n3':       float(rows[3]),
            'd':        float(rows[4]),
        }
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Conversion driver
# ---------------------------------------------------------------------------

def convert(expt_dir: Path, out_dir: Path, basename: str = 'cam') -> dict:
    """
    Convert an OpenPTV experiment folder to MultiCamSelfCal format.
    """
    cal_dir   = expt_dir / 'cal'
    param_dir = expt_dir / 'parameters'

    if not cal_dir.is_dir():
        raise FileNotFoundError(f"No 'cal' subdirectory in {expt_dir}")

    # ---- Read ptv.par for image dimensions and camera count ---------
    ptv = parse_ptv_par(param_dir / 'ptv.par')
    n_cams = ptv.get('n_cams') or 4
    img_w  = ptv.get('image_width')  or 1280
    img_h  = ptv.get('image_height') or 1024
    print(f"  Cameras: {n_cams}, image size: {img_w}x{img_h}")

    # ---- Read calblock.txt — 3D ground-truth points ----------------
    # The 3D calibration target file goes by several names depending on
    # dataset and version. Try common ones.
    calblock_candidates = [
        cal_dir / 'calblock.txt',
        cal_dir / 'target_on_a_side.txt',
        cal_dir / 'calblock_20.txt',
        cal_dir / 'calblock_10.txt',
        cal_dir / 'cal_block.txt',
    ]
    calblock_path = next((p for p in calblock_candidates if p.exists()), None)
    if calblock_path is None:
        raise FileNotFoundError(
            f"No calibration target file found in {cal_dir}. "
            f"Tried: {[p.name for p in calblock_candidates]}")
    print(f"  Calibration target file: {calblock_path.name}")
    calblock = parse_calblock(calblock_path)
    point_ids_3d = calblock[:, 0].astype(int)
    xyz_3d       = calblock[:, 1:4]
    n_points     = len(point_ids_3d)
    print(f"  Calibration target points: {n_points}")

    # ---- Read per-camera target files (2D detections) ---------------
    cam_targets = []
    for i in range(1, n_cams + 1):
        # Try common naming conventions
        candidates = [
            cal_dir / f'cam{i}.tif_targets',
            cal_dir / f'cam{i}_targets',
            cal_dir / f'cam{i}.targets',
            cal_dir / f'cam{i}_cal.tif_targets',
        ]
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            raise FileNotFoundError(
                f"No targets file for camera {i}. Tried: "
                f"{[str(p.name) for p in candidates]}")
        pts = parse_targets(path)
        cam_targets.append(pts)
        print(f"  Cam {i}: {len(pts)} detected dots ({path.name})")

    # ---- Build the measurement matrix points.dat (3K, N) -----------
    # For each calibration point j (column in calblock), for each cam i,
    # fill rows 3i:3i+3 of points.dat with [u, v, 1] if the point is
    # visible in cam i, else NaN.
    points = np.full((3 * n_cams, n_points), np.nan)
    idmat  = np.zeros((n_cams, n_points), dtype=int)

    for j, pid in enumerate(point_ids_3d):
        for i, pts in enumerate(cam_targets):
            if pid in pts:
                u, v = pts[pid]
                points[3*i,   j] = u
                points[3*i+1, j] = v
                points[3*i+2, j] = 1.0
                idmat[i, j] = 1

    # ---- Resolution matrix -----------------------------------------
    res = np.tile([img_w, img_h], (n_cams, 1))

    # ---- Read multimedia parameters --------------------------------
    # First try the dedicated multimedia.par, then fall back to ptv.par
    mm = parse_multimedia_par(param_dir / 'multimedia.par')
    if mm is None and 'n_water' in ptv:
        mm = {
            'n_layers': 1,
            'n1':       1.0,
            'n2':       ptv['n_glass'],
            'n3':       ptv['n_water'],
            'd':        ptv.get('d_glass', 0.0),
        }
    if mm:
        print(f"  Multimedia: n1={mm['n1']}, n2={mm['n2']}, "
              f"n3={mm['n3']}, d={mm['d']} mm")

    # ---- Read per-camera .ori files for ground truth + starting intrinsics
    oris = []
    for i in range(1, n_cams + 1):
        # OpenPTV uses cam{N}.tif.ori for newer datasets; cam{N}.ori for older.
        ori_candidates = [cal_dir / f'cam{i}.tif.ori', cal_dir / f'cam{i}.ori']
        ori = None
        for p in ori_candidates:
            if p.exists():
                ori = parse_ori(p)
                break
        if ori is None:
            print(f"  Warning: cam{i}.ori not parsed — no ground truth for camera {i}")
        oris.append(ori)

    # ---- Write everything ------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savetxt(out_dir / 'points.dat', points, fmt='%.6f')
    np.savetxt(out_dir / 'IdMat.dat',  idmat,  fmt='%d')
    np.savetxt(out_dir / 'Res.dat',    res,    fmt='%d')

    # ---- Write starting .rad files ---------------------------------
    # Convert focal length from mm to pixels.  OpenPTV stores focal length
    # in mm and pixel size separately; we assume square pixels and that
    # focal_mm / pixel_size_mm = focal_px.  Typical pixel size is 10 µm.
    # The user can refine these starting values later.
    pixel_size_mm = ptv.get('pixel_size_mm', 0.010)   # use ptv.par value if available
    print(f"  Pixel size: {pixel_size_mm} mm")
    for i, ori in enumerate(oris, start=1):
        if ori is None:
            # Fallback to a sensible default if no .ori
            fx = fy = 800.0
            cx = img_w / 2.0
            cy = img_h / 2.0
        else:
            fx = fy = ori['focal_mm'] / pixel_size_mm
            cx = img_w / 2.0 + ori['xp_yp'][0] / pixel_size_mm
            cy = img_h / 2.0 + ori['xp_yp'][1] / pixel_size_mm
        rad_path = out_dir / f'{basename}{i}.rad'
        with open(rad_path, 'w') as fh:
            fh.write(f"K11 = {fx:.16f}\n")
            fh.write(f"K12 = 0.0000000000000000\n")
            fh.write(f"K13 = {cx:.16f}\n")
            fh.write(f"K21 = 0.0000000000000000\n")
            fh.write(f"K22 = {fy:.16f}\n")
            fh.write(f"K23 = {cy:.16f}\n")
            fh.write(f"K31 = 0.0000000000000000\n")
            fh.write(f"K32 = 0.0000000000000000\n")
            fh.write(f"K33 = 1.0000000000000000\n\n")
            fh.write(f"kc1 = 0.0000000000000000\n")
            fh.write(f"kc2 = 0.0000000000000000\n")
            fh.write(f"kc3 = 0.0000000000000000\n")
            fh.write(f"kc4 = 0.0000000000000000\n")

    # ---- Write ground_truth.json for alignment ---------------------
    gt = {
        'cameras': [
            {
                'id':              i + 1,
                'position_mm':     ori['position'] if ori else None,
                'rotation_matrix': ori['R'] if ori else None,
                'intrinsics': {
                    'fx': ori['focal_mm'] / pixel_size_mm if ori else None,
                    'fy': ori['focal_mm'] / pixel_size_mm if ori else None,
                    'cx': img_w / 2.0,
                    'cy': img_h / 2.0,
                } if ori else None,
            }
            for i, ori in enumerate(oris)
        ],
        'particles_mm': xyz_3d.tolist(),
        'unit':         'mm',
        'source':       str(expt_dir),
    }
    with open(out_dir / 'ground_truth.json', 'w') as fh:
        json.dump(gt, fh, indent=2)

    # ---- Write the cfg file ----------------------------------------
    cfg_path = out_dir / 'multicamselfcal.cfg'
    with open(cfg_path, 'w') as fh:
        fh.write("[Files]\n")
        fh.write(f"Basename: {basename}\n")
        fh.write("Image-Extension: tif\n\n")
        fh.write("[Images]\n")
        fh.write("Subpix: 0.5\n\n")
        fh.write("[Calibration]\n")
        fh.write(f"Num-Cameras: {n_cams}\n")
        fh.write("Num-Projectors: 0\n")
        fh.write("Nonlinear-Parameters: 50    0    1    0    0    0\n")
        fh.write("Nonlinear-Update: 1   0   1   0   0   0\n")
        fh.write("Initial-Tolerance: 10\n")
        fh.write("Do-Global-Iterations: 1\n")
        fh.write("Global-Iteration-Threshold: 0.5\n")
        fh.write("Global-Iteration-Max: 5\n")
        fh.write("Num-Cameras-Fill: 2\n")
        fh.write("Do-Bundle-Adjustment: 1\n")
        fh.write("Undo-Radial: 0\n")
        fh.write("Min-Points-Value: 30\n")
        fh.write("N-Tuples: 3\n")
        fh.write("Square-Pixels: 1\n")
        fh.write("Use-Nth-Frame: 1\n")
        fh.write("Align-Existing: ground_truth.json\n")
        if mm:
            fh.write("\n[Multimedia]\n")
            fh.write(f"N1: {mm['n1']}\n")
            fh.write(f"N2: {mm['n2']}\n")
            fh.write(f"D:  {mm['d']}\n")
            fh.write(f"N3: {mm['n3']}\n")
            fh.write("Z-Glass-Top: 0.0\n")
            # Estimate camera-to-glass distance from cam1.ori if possible
            cg_dist = 100.0  # default fallback
            if oris[0] is not None:
                # Distance from camera position to glass plane along Z
                cam_z = oris[0]['position'][2]
                glass_z = oris[0]['glass_pos'][2]
                cg_dist = abs(cam_z - glass_z)
                if cg_dist < 1.0:
                    cg_dist = 100.0  # fall back if values look degenerate
            fh.write(f"Camera-Glass-Distance: {cg_dist:.1f}\n")

    # ---- Summary --------------------------------------------------
    visible_in_all = int((idmat.sum(axis=0) == n_cams).sum())
    visible_in_geq2 = int((idmat.sum(axis=0) >= 2).sum())
    summary = {
        'n_cams':         n_cams,
        'n_points':       n_points,
        'visible_in_all': visible_in_all,
        'visible_in_geq2': visible_in_geq2,
        'image_size':     (img_w, img_h),
        'multimedia':     mm,
    }
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('experiment',
                    help='path to OpenPTV experiment folder (must contain cal/ and parameters/)')
    ap.add_argument('output',
                    help='path to output folder (will be created if needed)')
    ap.add_argument('--basename', default='cam',
                    help="experiment basename (matches Basename: in the cfg; default 'cam')")
    args = ap.parse_args()

    expt = Path(args.experiment).resolve()
    out  = Path(args.output).resolve()

    print(f"Converting OpenPTV experiment:")
    print(f"  Source: {expt}")
    print(f"  Output: {out}")
    print()

    summary = convert(expt, out, basename=args.basename)

    print()
    print("Summary:")
    print(f"  Cameras            : {summary['n_cams']}")
    print(f"  Calibration points : {summary['n_points']}")
    print(f"  Visible in all cams: {summary['visible_in_all']}")
    print(f"  Visible in ≥2 cams : {summary['visible_in_geq2']}")
    print(f"  Image size         : {summary['image_size'][0]} x {summary['image_size'][1]}")
    if summary['multimedia']:
        mm = summary['multimedia']
        print(f"  Multimedia         : n1={mm['n1']}, n2={mm['n2']}, "
              f"n3={mm['n3']}, d={mm['d']} mm")
    print()
    print(f"✓ Output written to: {out}")
    print(f"  Copy these files into your data folder, then update")
    print(f"  the cfg path in CalibrationSystem and run GoCal.py.")
