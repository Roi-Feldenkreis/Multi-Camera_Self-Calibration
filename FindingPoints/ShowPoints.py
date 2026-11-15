"""
Overlay detected points on the original images
(Python port of MultiCamSelfCal/FindingPoints/showpoints.m).

FIX summary vs. previous version:
  * camera rows 3*i, 3*i+1 (was MATLAB 3*cam_id-2, 3*cam_id-1)
  * camera = row index i, not its ID; frames aligned via frame_ids.txt
  * 1-based coordinates converted back to image pixels
  * grey / 16-bit images normalised to 8-bit RGB before drawing in colour
  * montage saved as PNG (EPS needs Ghostscript on Windows); frame subset selectable
"""
import os
import re
import glob
import json
import numpy as np
from PIL import Image, ImageDraw

_LAST_INT = re.compile(r"(\d+)(?!.*\d)")


def _frame_map(pattern):
    return {int(_LAST_INT.search(os.path.basename(f)).group(1)): f for f in sorted(glob.glob(pattern))}


def _to_rgb(path):
    a = np.asarray(Image.open(path)).astype(np.float64)
    if a.ndim == 3:
        a = a[..., :3]
    lo, hi = a.min(), a.max()
    a = np.zeros_like(a) if hi <= lo else (a - lo) / (hi - lo) * 255
    img = Image.fromarray(a.astype(np.uint8))
    return img.convert("RGB")


def show_points(config, frames_to_show=None, compose=True):
    d = config["out_dir"]
    IdMat = np.loadtxt(os.path.join(d, "IdMat.dat"), ndmin=2)
    Ws = np.loadtxt(os.path.join(d, "Points.dat"), ndmin=2)
    frame_ids = np.loadtxt(os.path.join(d, "frame_ids.txt"), dtype=int, ndmin=1)
    off = 1.0 if config.get("one_based", True) else 0.0
    cam_ids = list(config["cams"].keys())
    fmaps = [_frame_map(config["cams"][c]) for c in cam_ids]

    out = os.path.join(d, "showpoints")
    os.makedirs(out, exist_ok=True)
    cols = range(len(frame_ids)) if frames_to_show is None else frames_to_show
    for j in cols:
        fid = int(frame_ids[j])
        tiles = []
        for i, cid in enumerate(cam_ids):
            path = fmaps[i].get(fid)
            if path is None:
                continue
            img = _to_rgb(path)
            dr = ImageDraw.Draw(img)
            dr.text((15, 20), f"Camera: {cid:02d} Frame: {fid:04d}", fill="lime")
            if IdMat[i, j] > 0:
                x, y = Ws[3 * i, j] - off, Ws[3 * i + 1, j] - off
                dr.ellipse((x - 12, y - 12, x + 12, y + 12), outline="lime", width=2)
                dr.ellipse((x - 7, y - 7, x + 7, y + 7), outline="red", width=1)
            else:
                dr.text((15, 40), "No point found", fill="lime")
            if compose:
                tiles.append(img)
            else:
                img.save(os.path.join(out, f"point.cam{cid}.{fid:04d}.png"))
        if compose and tiles:
            W = sum(t.width for t in tiles)
            H = max(t.height for t in tiles)
            mont = Image.new("RGB", (W, H))
            x0 = 0
            for t in tiles:
                mont.paste(t, (x0, 0))
                x0 += t.width
            mont.save(os.path.join(out, f"points.composed.{fid:04d}.png"))
    return out


if __name__ == "__main__":
    import sys
    with open(sys.argv[1]) as f:
        cfg = json.load(f)
    cfg["cams"] = {int(k): v for k, v in cfg["cams"].items()}
    show_points(cfg, frames_to_show=range(10))
