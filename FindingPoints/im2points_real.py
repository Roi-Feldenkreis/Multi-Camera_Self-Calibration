"""
im2points_real.py
=================
Real-data front end for the MultiCamSelfCal / GoCal_derefract pipeline.

Detects a single bright target (submerged LED / bead) in every frame of every
camera, and writes the three Svoboda input files:

    IdMat.dat   M x N   1 = detected, 0 = not detected
    Points.dat  3M x N  rows [u; v; 1] per camera, NaN where not detected
    Res.dat     M x 2   image resolution per camera (width, height)

Detection follows Svoboda's im2points recipe:
    diff = frame - background_mean
    blob mask = diff > 0.5 * max(diff)   (Svoboda: 4/5; see thr_frac)
    reject if: peak < K_SIGMA * background_std at the peak pixel
               more than one blob / blob too large / too small / too eccentric
    sub-pixel position: 2D Gaussian fit (fallback: weighted centroid)

Points are stored RAW (distorted, refracted). Do not undistort or de-refract:
GoCal_derefract does both internally.

After export, sanity checks run automatically and are written to
<out>/checks/ (report + coverage plots + spot-check overlays).

Usage (Windows example, one --cam per camera, in pipeline camera order):

    python im2points_real.py ^
        --cam "D:\\exp1\\cam1\\*.tif" --cam "D:\\exp1\\cam2\\*.tif" ^
        --cam "D:\\exp1\\cam3\\*.tif" --cam "D:\\exp1\\cam4\\*.tif" ^
        --bg  "D:\\exp1\\bg1\\*.tif"  --bg  "D:\\exp1\\bg2\\*.tif" ^
        --bg  "D:\\exp1\\bg3\\*.tif"  --bg  "D:\\exp1\\bg4\\*.tif" ^
        --out "D:\\exp1\\calib_input" --one-based

Requires: numpy, scipy, opencv-python, matplotlib.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, asdict

import cv2
import numpy as np
from scipy import ndimage
from scipy.optimize import least_squares


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DetectionConfig:
    thr_frac: float = 0.5          # blob mask = diff > thr_frac * max(diff); Svoboda uses 4/5,
                                   # 0.5 gives a mask big enough for meaningful area/shape tests
    k_sigma: float = 5.0           # peak must exceed k_sigma * background std
    min_peak: float = 10.0         # absolute minimum peak above background (grey levels)
    min_area: int = 1              # px, smallest accepted blob
    max_area: int = 400            # px, largest accepted blob (set ~ 4x expected LED area)
    max_ecc: float = 0.85          # 0 = circle, ->1 = line (motion blur)
    win_half: int = 7              # half-size of the fitting window around the blob
    subpix: str = "gauss"          # "gauss" or "centroid"
    channel: int = -1              # colour channel for RGB images (-1 = convert to grey)
    bg_max_frames: int = 200       # max frames used for the background model
    one_based: bool = True         # True = MATLAB (1-based) pixel coordinates


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

_LAST_INT = re.compile(r"(\d+)(?!.*\d)")


def frame_index(path: str, pattern: re.Pattern = _LAST_INT) -> int:
    """Frame number = last integer in the file name (e.g. cam1_000123.tif -> 123)."""
    m = pattern.search(os.path.basename(path))
    if m is None:
        raise ValueError(f"No frame number found in file name: {path}")
    return int(m.group(1))


def list_frames(glob_pattern: str) -> dict[int, str]:
    files = sorted(glob.glob(glob_pattern))
    if not files:
        raise FileNotFoundError(f"No files match: {glob_pattern}")
    out: dict[int, str] = {}
    for f in files:
        idx = frame_index(f)
        if idx in out:
            raise ValueError(f"Duplicate frame index {idx}: {out[idx]} and {f}")
        out[idx] = f
    return out


def read_image(path: str, channel: int) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"Cannot read image: {path}")
    if img.ndim == 3:
        if channel < 0:
            img = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2GRAY)
        else:
            img = img[..., channel]          # OpenCV order: 0=B, 1=G, 2=R
    return img.astype(np.float64)


# ---------------------------------------------------------------------------
# Background model (streaming mean / std, Welford)
# ---------------------------------------------------------------------------

def background_model(paths: list[str], cfg: DetectionConfig):
    if len(paths) > cfg.bg_max_frames:
        sel = np.linspace(0, len(paths) - 1, cfg.bg_max_frames).round().astype(int)
        paths = [paths[i] for i in sel]
    n, mean, m2 = 0, None, None
    for p in paths:
        x = read_image(p, cfg.channel)
        if mean is None:
            mean = np.zeros_like(x)
            m2 = np.zeros_like(x)
        n += 1
        d = x - mean
        mean += d / n
        m2 += d * (x - mean)
    std = np.sqrt(m2 / max(n - 1, 1))
    # floor std so that a perfectly static pixel does not give std = 0
    std = np.maximum(std, 1.0)
    return mean, std


# ---------------------------------------------------------------------------
# Single-frame detection
# ---------------------------------------------------------------------------

def _gauss2d(p, xx, yy):
    a, x0, y0, sx, sy, b = p
    return a * np.exp(-0.5 * (((xx - x0) / sx) ** 2 + ((yy - y0) / sy) ** 2)) + b


def _subpixel(diff: np.ndarray, cy: float, cx: float, cfg: DetectionConfig):
    h, w = diff.shape
    r = cfg.win_half
    y0, y1 = max(int(round(cy)) - r, 0), min(int(round(cy)) + r + 1, h)
    x0, x1 = max(int(round(cx)) - r, 0), min(int(round(cx)) + r + 1, w)
    patch = diff[y0:y1, x0:x1]
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float64)

    # weighted centroid (also the Gaussian start point)
    wts = np.clip(patch - np.median(patch), 0, None)
    s = wts.sum()
    if s <= 0:
        return None
    u_c = (wts * xx).sum() / s
    v_c = (wts * yy).sum() / s
    if cfg.subpix == "centroid":
        return u_c, v_c

    p0 = [patch.max() - np.median(patch), u_c, v_c, 1.5, 1.5, np.median(patch)]
    lb = [0, x0, y0, 0.3, 0.3, -np.inf]
    ub = [np.inf, x1 - 1, y1 - 1, r, r, np.inf]
    try:
        res = least_squares(
            lambda p: (_gauss2d(p, xx, yy) - patch).ravel(),
            p0, bounds=(lb, ub), method="trf")
        if not res.success:
            return u_c, v_c
        u, v = res.x[1], res.x[2]
        # guard against the fit running away from the blob
        if abs(u - u_c) > 2.0 or abs(v - v_c) > 2.0:
            return u_c, v_c
        return u, v
    except Exception:
        return u_c, v_c


def detect(img: np.ndarray, bg_mean: np.ndarray, bg_std: np.ndarray,
           cfg: DetectionConfig):
    """Return (u, v, reason). u, v are None when rejected; reason is a short code."""
    diff = img - bg_mean
    py, px = np.unravel_index(np.argmax(diff), diff.shape)
    peak = diff[py, px]
    if peak < cfg.min_peak:
        return None, None, "no_peak"
    if peak < cfg.k_sigma * bg_std[py, px]:
        return None, None, "below_ksigma"

    mask = diff > cfg.thr_frac * peak
    lab, nlab = ndimage.label(mask, structure=np.ones((3, 3), bool))
    if nlab != 1:
        return None, None, "multi_blob"
    area = int(mask.sum())
    if area < cfg.min_area:
        return None, None, "too_small"
    if area > cfg.max_area:
        return None, None, "too_large"

    ys, xs = np.nonzero(mask)
    if area >= 5:                  # shape is meaningless on 1-4 px
        cov = np.cov(np.vstack([xs, ys]).astype(np.float64))
        ev = np.sort(np.linalg.eigvalsh(cov))
        if ev[1] > 0:
            ecc = np.sqrt(max(0.0, 1.0 - ev[0] / ev[1]))
            if ecc > cfg.max_ecc:
                return None, None, "eccentric"

    res = _subpixel(diff, ys.mean(), xs.mean(), cfg)
    if res is None:
        return None, None, "fit_failed"
    u, v = res
    if cfg.one_based:
        u, v = u + 1.0, v + 1.0
    return u, v, "ok"


# ---------------------------------------------------------------------------
# Per-camera worker
# ---------------------------------------------------------------------------

def process_camera(cam_id: int, frames: dict[int, str], frame_ids: list[int],
                   bg_paths: list[str] | None, cfg: DetectionConfig):
    if bg_paths:
        bg_mean, bg_std = background_model(bg_paths, cfg)
    else:
        # No background sequence: Svoboda fallback = model from the calibration
        # frames themselves (the moving target averages out).
        bg_mean, bg_std = background_model([frames[i] for i in frame_ids if i in frames], cfg)

    h, w = bg_mean.shape
    uv = np.full((2, len(frame_ids)), np.nan)
    reasons: dict[str, int] = {}
    for j, fid in enumerate(frame_ids):
        if fid not in frames:
            reasons["missing_file"] = reasons.get("missing_file", 0) + 1
            continue
        img = read_image(frames[fid], cfg.channel)
        if img.shape != (h, w):
            raise ValueError(f"cam {cam_id + 1}: frame {fid} has shape {img.shape}, expected {(h, w)}")
        u, v, why = detect(img, bg_mean, bg_std, cfg)
        reasons[why] = reasons.get(why, 0) + 1
        if u is not None:
            uv[:, j] = (u, v)
    return cam_id, uv, (w, h), reasons


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def _write_matrix(path: str, mat: np.ndarray, fmt: str):
    with open(path, "w") as f:
        for row in mat:
            f.write(" ".join("NaN" if np.isnan(x) else fmt % x for x in row) + "\n")


def write_outputs(out_dir: str, uv_all: np.ndarray, res: np.ndarray):
    """uv_all: M x 2 x N (NaN where missing). Writes IdMat.dat, Points.dat, Res.dat."""
    m, _, n = uv_all.shape
    idmat = (~np.isnan(uv_all[:, 0, :])).astype(int)
    points = np.full((3 * m, n), np.nan)
    for i in range(m):
        points[3 * i:3 * i + 2, :] = uv_all[i]
        points[3 * i + 2, :] = np.where(idmat[i] == 1, 1.0, np.nan)
    os.makedirs(out_dir, exist_ok=True)
    np.savetxt(os.path.join(out_dir, "IdMat.dat"), idmat, fmt="%d")
    _write_matrix(os.path.join(out_dir, "Points.dat"), points, "%.6f")
    np.savetxt(os.path.join(out_dir, "Res.dat"), res, fmt="%d")
    return idmat, points


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

def sanity_checks(out_dir: str, cam_frames: list[dict[int, str]],
                  frame_ids: list[int], cfg: DetectionConfig,
                  n_spot: int = 20, seed: int = 0) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chk = os.path.join(out_dir, "checks")
    os.makedirs(chk, exist_ok=True)
    lines: list[str] = []

    # 1. reload exactly as a consumer would
    idmat = np.loadtxt(os.path.join(out_dir, "IdMat.dat"), ndmin=2)
    points = np.loadtxt(os.path.join(out_dir, "Points.dat"), ndmin=2)
    res = np.loadtxt(os.path.join(out_dir, "Res.dat"), ndmin=2)
    m, n = idmat.shape
    ok_shape = points.shape == (3 * m, n) and res.shape == (m, 2)
    lines.append(f"[{'PASS' if ok_shape else 'FAIL'}] shapes: IdMat {idmat.shape}, "
                 f"Points {points.shape}, Res {res.shape}")

    # 2. IdMat vs Points consistency + homogeneous row
    consistent = True
    for i in range(m):
        nan_u = np.isnan(points[3 * i])
        if not np.array_equal(nan_u, idmat[i] == 0):
            consistent = False
        third = points[3 * i + 2, idmat[i] == 1]
        if third.size and not np.allclose(third, 1.0):
            consistent = False
    lines.append(f"[{'PASS' if consistent else 'FAIL'}] NaN exactly where IdMat = 0; third row = 1")

    # 3. detection rate per camera + cameras per frame
    lines.append("")
    lines.append("Detection rate per camera:")
    for i in range(m):
        lines.append(f"  cam {i + 1}: {int(idmat[i].sum())}/{n} = {100 * idmat[i].mean():.1f}%")
    cpf = idmat.sum(axis=0).astype(int)
    hist = np.bincount(cpf, minlength=m + 1)
    lines.append("Cameras per frame (count of frames):")
    for k, c in enumerate(hist):
        lines.append(f"  {k} cams: {c}")
    n3 = int((cpf >= 3).sum())
    lines.append(f"[{'PASS' if n3 >= 1000 else 'WARN'}] frames seen by >= 3 cameras: {n3} "
                 f"(target 1000-3000)")

    # 4. pairwise RANSAC fundamental matrix
    lines.append("")
    lines.append("Pairwise epipolar RANSAC (inlier ratio, target > 90%):")
    for a in range(m):
        for b in range(a + 1, m):
            common = (idmat[a] == 1) & (idmat[b] == 1)
            k = int(common.sum())
            if k < 15:
                lines.append(f"  [WARN] cam {a + 1}-{b + 1}: only {k} common points")
                continue
            pa = points[3 * a:3 * a + 2, common].T.astype(np.float64)
            pb = points[3 * b:3 * b + 2, common].T.astype(np.float64)
            F, inl = cv2.findFundamentalMat(pa, pb, cv2.FM_RANSAC, 1.0, 0.999)
            if F is None or inl is None:
                lines.append(f"  [FAIL] cam {a + 1}-{b + 1}: RANSAC failed ({k} pts)")
                continue
            r = inl.ravel().mean()
            tag = "PASS" if r > 0.9 else "WARN"
            lines.append(f"  [{tag}] cam {a + 1}-{b + 1}: {100 * r:.1f}% of {k} pts")

    # 5. coverage plots
    off = 1.0 if cfg.one_based else 0.0
    fig, axes = plt.subplots(1, m, figsize=(4 * m, 3.5), squeeze=False)
    for i in range(m):
        ax = axes[0, i]
        w, h = res[i]
        ax.plot(points[3 * i], points[3 * i + 1], ".", ms=1.5)
        ax.add_patch(plt.Rectangle((off, off), w, h, fill=False, ls="--", color="k"))
        ax.set_xlim(off - 0.02 * w, off + 1.02 * w)
        ax.set_ylim(off + 1.02 * h, off - 0.02 * h)
        ax.set_aspect("equal")
        ax.set_title(f"cam {i + 1}: {int(idmat[i].sum())} pts")
    fig.tight_layout()
    fig.savefig(os.path.join(chk, "coverage.png"), dpi=130)
    plt.close(fig)
    lines.append("")
    lines.append("Coverage plot: checks/coverage.png (points should fill each sensor incl. corners)")

    # 6. spot-check overlays
    rng = np.random.default_rng(seed)
    cand = np.nonzero(cpf >= 1)[0]
    pick = rng.choice(cand, size=min(n_spot, cand.size), replace=False) if cand.size else []
    spot_dir = os.path.join(chk, "spot")
    os.makedirs(spot_dir, exist_ok=True)
    for j in pick:
        fid = frame_ids[j]
        tiles = []
        for i in range(m):
            path = cam_frames[i].get(fid)
            if path is None:
                continue
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img.ndim == 3:
                img = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2GRAY)
            img = cv2.normalize(img.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            if idmat[i, j]:
                u, v = points[3 * i, j] - off, points[3 * i + 1, j] - off
                cv2.circle(img, (int(round(u)), int(round(v))), 12, (0, 0, 255), 1)
            cv2.putText(img, f"cam {i + 1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            tiles.append(img)
        if tiles:
            hmax = max(t.shape[0] for t in tiles)
            tiles = [cv2.copyMakeBorder(t, 0, hmax - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
            cv2.imwrite(os.path.join(spot_dir, f"frame_{fid:06d}.png"), np.hstack(tiles))
    lines.append(f"Spot-check overlays: checks/spot/ ({len(pick)} frames, red circle = detection)")

    report = "\n".join(lines)
    with open(os.path.join(chk, "report.txt"), "w") as f:
        f.write(report + "\n")
    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(cam_globs: list[str], bg_globs: list[str] | None, out_dir: str,
        cfg: DetectionConfig, workers: int = 1, checks: bool = True):
    m = len(cam_globs)
    if m < 3:
        raise ValueError("At least 3 cameras are required.")
    if bg_globs and len(bg_globs) != m:
        raise ValueError("Give one --bg per --cam, in the same order.")

    cam_frames = [list_frames(g) for g in cam_globs]
    frame_ids = sorted(set().union(*[set(cf) for cf in cam_frames]))
    n_common = len(set.intersection(*[set(cf) for cf in cam_frames]))
    print(f"{m} cameras, {len(frame_ids)} frame indices ({n_common} present in all cameras)")
    if n_common < len(frame_ids):
        print("  WARNING: some cameras are missing frames -> check synchronisation / dropped frames")

    bg_lists = [sorted(glob.glob(g)) for g in bg_globs] if bg_globs else [None] * m

    uv_all = np.full((m, 2, len(frame_ids)), np.nan)
    res = np.zeros((m, 2), int)
    jobs = [(i, cam_frames[i], frame_ids, bg_lists[i], cfg) for i in range(m)]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(process_camera, *zip(*jobs)))
    else:
        results = [process_camera(*j) for j in jobs]

    for cam_id, uv, (w, h), reasons in results:
        uv_all[cam_id] = uv
        res[cam_id] = (w, h)
        summary = ", ".join(f"{k}={v}" for k, v in sorted(reasons.items()))
        print(f"  cam {cam_id + 1} ({w}x{h}): {summary}")

    write_outputs(out_dir, uv_all, res)
    with open(os.path.join(out_dir, "frame_ids.txt"), "w") as f:
        f.write("\n".join(map(str, frame_ids)) + "\n")
    with open(os.path.join(out_dir, "detection_config.txt"), "w") as f:
        for k, v in asdict(cfg).items():
            f.write(f"{k} = {v}\n")
    print(f"Wrote IdMat.dat, Points.dat, Res.dat, frame_ids.txt to {out_dir}")

    if checks:
        print()
        print(sanity_checks(out_dir, cam_frames, frame_ids, cfg))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cam", action="append", required=True,
                   help="glob for one camera's calibration frames (repeat, in camera order)")
    p.add_argument("--bg", action="append", default=None,
                   help="glob for one camera's background frames (repeat, same order as --cam)")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--workers", type=int, default=1, help="parallel processes (one camera each)")
    p.add_argument("--no-checks", action="store_true")
    base = DetectionConfig()
    p.add_argument("--one-based", dest="one_based", action="store_true", default=base.one_based,
                   help="MATLAB 1-based pixel coordinates (default)")
    p.add_argument("--zero-based", dest="one_based", action="store_false",
                   help="0-based pixel coordinates (OpenCV/NumPy)")
    for f in ("thr_frac", "k_sigma", "min_peak", "max_ecc"):
        p.add_argument("--" + f.replace("_", "-"), type=float, default=getattr(base, f))
    for f in ("min_area", "max_area", "win_half", "channel", "bg_max_frames"):
        p.add_argument("--" + f.replace("_", "-"), type=int, default=getattr(base, f))
    p.add_argument("--subpix", choices=["gauss", "centroid"], default=base.subpix)
    a = p.parse_args(argv)

    cfg = DetectionConfig(**{k: getattr(a, k) for k in asdict(base)})
    run(a.cam, a.bg, a.out, cfg, workers=a.workers, checks=not a.no_checks)


if __name__ == "__main__":
    main()
