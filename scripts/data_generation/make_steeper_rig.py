"""
make_steeper_rig.py - same rig, steeper camera views.

Copies a Step 2 folder and changes ONLY the camera poses in its ground_truth.json:
every camera keeps its height and keeps looking at the same aim point (the centre of
the original particle cloud, where the generator aimed the cameras), but its
horizontal offset from that point is scaled by --factor. A factor below 1 makes the
cameras look more steeply down through the glass. Each camera is turned by the rotation
Q that takes its old viewing direction to the new one: R_new = R_old @ Q^T.
The particles are left as they are; run make_wide_cloud.py on the new folder to draw a
cloud visible to the NEW cameras. Stale points.dat / IdMat.dat are removed.

Usage (project root):
    python make_steeper_rig.py <step2_folder> <new_step2_folder> [--factor 0.7]
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def rot_from_to(a, b):
    a = np.asarray(a, float) / np.linalg.norm(a); b = np.asarray(b, float) / np.linalg.norm(b)
    v = np.cross(a, b); c = float(a @ b)
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * (1.0 / (1.0 + c))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step2_folder"); ap.add_argument("new_folder")
    ap.add_argument("--factor", type=float, default=0.7)
    args = ap.parse_args()
    src, dst = Path(args.step2_folder), Path(args.new_folder)
    if dst.exists():
        raise SystemExit(f"{dst} already exists - choose a new folder name")
    gt = json.load(open(src / "ground_truth.json"))
    target = np.asarray(gt["particles_mm"], float).mean(axis=0)      # the generator's aim point
    print(f"aim point (centre of the original cloud): {np.round(target, 1)} mm, factor {args.factor}")
    for c in sorted(gt["cameras"], key=lambda c: c["id"]):
        C = np.asarray(c["position_mm"], float); R = np.asarray(c["rotation_matrix"], float)
        Cn = C.copy(); Cn[:2] = target[:2] + args.factor * (C[:2] - target[:2])
        Q = rot_from_to(target - C, target - Cn)
        c["position_mm"] = Cn.tolist(); c["rotation_matrix"] = (R @ Q.T).tolist()
        a0 = np.degrees(np.arccos(abs((target - C)[2]) / np.linalg.norm(target - C)))
        a1 = np.degrees(np.arccos(abs((target - Cn)[2]) / np.linalg.norm(target - Cn)))
        print(f"  cam{c['id']}: position {np.round(C, 1)} -> {np.round(Cn, 1)} mm, viewing angle {a0:.1f} -> {a1:.1f} deg")
    gt["source"] = gt.get("source", "") + f" | cameras made steeper by make_steeper_rig.py (factor={args.factor})"
    gt["rig_change"] = {"generator": "make_steeper_rig.py", "factor": args.factor, "aim_point_mm": target.tolist()}
    shutil.copytree(src, dst)
    for stale in ("points.dat", "IdMat.dat"):
        if (dst / stale).exists():
            (dst / stale).unlink()
    with open(dst / "ground_truth.json", "w") as f:
        json.dump(gt, f, indent=2)
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
