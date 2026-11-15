"""
Find the LED in all frames of all cameras and write the Svoboda input files
(Python port of MultiCamSelfCal/FindingPoints/im2points.m).

Outputs (in config['out_dir']):
    IdMat.dat   M x N   1 = point found, 0 = not found
    Points.dat  3M x N  rows [u; v; 1] per camera, NaN where not found
    Res.dat     M x 2   (width, height) per camera
    frame_ids.txt       frame number of each column

FIX summary vs. previous version:
  * `from getpoint import getpoint` (module was called as a function)
  * image-processing parameters passed from config['imgs'] (was config['files'])
  * Points written as 3M x N (was reshape(-1, 3) -> (M*N) x 3, unreadable by gocal)
  * NaN written as 'NaN' (MATLAB- and NumPy-readable)
  * frame alignment: frame number = last integer in the file name (slicing was wrong)
  * statistics delegated to im2imstat (no duplicated code), float, cached as .npy
  * optional 1-based coordinates to match MATLAB output (config['one_based'])
  * script body wrapped in main() -> importable, Windows-safe
"""
import os
import re
import glob
import time
import json
import numpy as np

from getpoint import getpoint
from im2imstat import im2imstat

_LAST_INT = re.compile(r"(\d+)(?!.*\d)")


def frame_number(path):
    m = _LAST_INT.search(os.path.basename(path))
    if m is None:
        raise ValueError(f"No frame number in file name: {path}")
    return int(m.group(1))


def default_config():
    return {
        "cams": {1: "images/cam1/*.png", 2: "images/cam2/*.png", 3: "images/cam3/*.png"},
        "out_dir": "calib_input",
        "imgs": {"LEDsize": 7, "LEDcolor": "gray", "subpix": 1 / 3, "LEDthr": 70},
        "step4stat": 1,
        "one_based": True,
        "showfig": False,
    }


def _write_nan_matrix(path, mat, fmt="%.6f"):
    with open(path, "w") as f:
        for row in mat:
            f.write(" ".join("NaN" if np.isnan(x) else fmt % x for x in row) + "\n")


def im2points(config):
    cam_ids = list(config["cams"].keys())
    M = len(cam_ids)
    out_dir = config["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    # --- image lists and frame alignment (FrameMat) -----------------------
    frames = {}
    for cid in cam_ids:
        files = sorted(glob.glob(config["cams"][cid]))
        if len(files) < 4:
            raise ValueError(f"Camera {cid}: not enough images for {config['cams'][cid]}")
        frames[cid] = {frame_number(f): f for f in files}
    frame_ids = sorted(set().union(*[set(d) for d in frames.values()]))
    N = len(frame_ids)

    # --- background statistics --------------------------------------------
    t0 = time.time()
    stats = im2imstat({cid: sorted(frames[cid].values()) for cid in cam_ids},
                      config["imgs"].get("LEDcolor", "gray"),
                      os.path.join(out_dir, "stats"), step=config["step4stat"])
    print(f"Elapsed time for statistics: {time.time() - t0:.1f} s")

    # --- find points --------------------------------------------------------
    print(f"Finding points: {M} cameras, {N} frames")
    off = 1.0 if config.get("one_based", True) else 0.0
    IdMat = np.zeros((M, N), dtype=int)
    Points = np.full((3 * M, N), np.nan)
    Res = np.zeros((M, 2), dtype=int)
    for i, cid in enumerate(cam_ids):
        t1 = time.time()
        avIM, stdIM = stats[cid]
        Res[i] = (avIM.shape[1], avIM.shape[0])
        reasons = {}
        for j, fid in enumerate(frame_ids):
            path = frames[cid].get(fid)
            if path is None:
                reasons["missing_file"] = reasons.get("missing_file", 0) + 1
                continue
            pos, err, why = getpoint(path, config.get("showfig", False), config["imgs"],
                                     avIM, stdIM, return_reason=True)
            reasons[why] = reasons.get(why, 0) + 1
            if not err:
                IdMat[i, j] = 1
                Points[3 * i:3 * i + 3, j] = (pos[0] + off, pos[1] + off, 1.0)
        summary = ", ".join(f"{k}={v}" for k, v in sorted(reasons.items()))
        print(f"  camera {cid}: {IdMat[i].sum()} points ({summary}), {time.time() - t1:.1f} s")

    np.savetxt(os.path.join(out_dir, "IdMat.dat"), IdMat, fmt="%d")
    _write_nan_matrix(os.path.join(out_dir, "Points.dat"), Points)
    np.savetxt(os.path.join(out_dir, "Res.dat"), Res, fmt="%d")
    with open(os.path.join(out_dir, "frame_ids.txt"), "w") as f:
        f.write("\n".join(map(str, frame_ids)) + "\n")

    cpf = IdMat.sum(axis=0)
    print("Overall statistics from im2points:")
    print(f"  frames (possible 3D points): {N}, cameras: {M}")
    print(f"  detected in >=1 cam: {(cpf > 0).sum()}, >=3 cams: {(cpf > 2).sum()}, all: {(cpf == M).sum()}")
    return IdMat, Points, Res, frame_ids


def main():
    import sys
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            cfg = json.load(f)
        cfg["cams"] = {int(k): v for k, v in cfg["cams"].items()}
    else:
        cfg = default_config()
    im2points(cfg)


if __name__ == "__main__":
    main()
