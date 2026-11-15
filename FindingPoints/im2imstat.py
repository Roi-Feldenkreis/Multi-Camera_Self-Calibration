"""
Background statistics (mean and std image) per camera
(Python port of MultiCamSelfCal/FindingPoints/im2imstat.m).

FIX summary vs. previous version:
  * computes ONLY the statistics (finding points lived here too -> duplicated im2points)
  * float64 throughout, saved as .npy (uint8 rounding destroyed std values of 1-3 grey levels)
  * channel selection identical to getpoint (single authority: getpoint.select_channel)
  * streaming (Welford) mean/std -> no list of all images in memory
  * glob instead of os.listdir on a pattern
"""
import os
import glob
import numpy as np

from getpoint import read_channel


def image_stats(paths, led_color, step=1):
    sel = paths[::max(int(step), 1)]
    n, mean, m2 = 0, None, None
    for p in sel:
        x = read_channel(p, led_color)
        if mean is None:
            mean, m2 = np.zeros_like(x), np.zeros_like(x)
        n += 1
        d = x - mean
        mean += d / n
        m2 += d * (x - mean)
    if n < 2:
        raise ValueError("Need at least 2 images for statistics")
    return mean, np.sqrt(m2 / (n - 1))


def im2imstat(cam_paths, led_color, stat_dir, step=1, force=False):
    """
    cam_paths : dict cam_id -> sorted list of image paths
    Returns dict cam_id -> (avIM, stdIM). Cached as avIM_cam<id>.npy / stdIM_cam<id>.npy.
    """
    os.makedirs(stat_dir, exist_ok=True)
    out = {}
    for cam_id, paths in cam_paths.items():
        fa = os.path.join(stat_dir, f"avIM_cam{cam_id}.npy")
        fs = os.path.join(stat_dir, f"stdIM_cam{cam_id}.npy")
        if not force and os.path.exists(fa) and os.path.exists(fs):
            print(f"Stats for camera {cam_id} already exist")
            out[cam_id] = (np.load(fa), np.load(fs))
            continue
        print(f"Computing average and std images for camera {cam_id}")
        av, sd = image_stats(paths, led_color, step)
        np.save(fa, av)
        np.save(fs, sd)
        out[cam_id] = (av, sd)
    return out


if __name__ == "__main__":
    import sys
    cams = {i + 1: sorted(glob.glob(g)) for i, g in enumerate(sys.argv[2:])}
    im2imstat(cams, "gray", sys.argv[1])
