"""
write_camera_geometry.py - replace the [CameraGeometry] section of a config with the
block for the cameras in a ground_truth.json (inter-camera distances, line-of-sight
distances, cyclic order), using make_camera_geometry.py's functions.

Default: NOISE-FREE block (true distances).
--noisy: hand-measurement noise exactly as make_camera_geometry.py draws it
         (distances sigma 4 mm, line of sight sigma 5 mm, same random streams), so the
         written values equal what that tool prints for the same --seed.

Usage (project root):
    python write_camera_geometry.py --gt <ground_truth.json> --cfg <multicamselfcal.cfg> [--noisy --seed 3]
"""
import argparse
import re
import shutil
from pathlib import Path

import numpy as np
from make_camera_geometry import load_true_positions, pairwise, cyclic_order_ccw, load_true_los, emit

ap = argparse.ArgumentParser()
ap.add_argument("--gt", required=True); ap.add_argument("--cfg", required=True)
ap.add_argument("--noisy", action="store_true", help="add hand-measurement noise (as make_camera_geometry.py)")
ap.add_argument("--sigma", type=float, default=4.0); ap.add_argument("--sigma-los", type=float, default=5.0)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
pos = load_true_positions(a.gt)
pairs = pairwise(pos); los = load_true_los(a.gt)
if a.noisy:
    rng = np.random.default_rng(a.seed)
    pairs = [(i, j, d + float(rng.normal(0.0, a.sigma))) for (i, j, d) in pairs]
    rng_los = np.random.default_rng(a.seed + 1000)
    los = {i: d + float(rng_los.normal(0.0, a.sigma_los)) for i, d in sorted(los.items())}
    header = f"NOISY hand-measured distances (sigma={a.sigma} mm, line of sight sigma={a.sigma_los} mm, seed={a.seed})"
else:
    header = "NOISE-FREE distances for these cameras (write_camera_geometry.py)"
block = emit(pairs, cyclic_order_ccw(pos), header, los).strip("\n")
cfg = Path(a.cfg); text = cfg.read_text()
shutil.copy2(cfg, cfg.with_suffix(".cfg.bak"))
m = re.search(r"^[ \t]*\[CameraGeometry\][ \t]*$", text, re.M)
blk = "\n".join(l for l in block.splitlines() if not l.startswith("# NOISE-FREE") and not l.startswith("# synthetic"))
if m:
    nxt = re.search(r"^[ \t]*\[[^\]]+\][ \t]*$", text[m.end():], re.M)
    end = m.end() + nxt.start() if nxt else len(text)
    start = text.rfind("\n", 0, m.start()) + 1
    text = text[:start] + blk + "\n\n" + text[end:].lstrip("\n")
else:
    text = text.rstrip("\n") + "\n\n" + blk + "\n"
cfg.write_text(text)
print(block)
