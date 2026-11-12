"""
make_wide_cloud.py — same cameras, wider particle cloud.

Copies a Step 2 folder and replaces ONLY the particle positions in its
ground_truth.json with a cloud that covers much more of each camera's image
(and a larger depth range). Everything else — cameras, intrinsics, Res.dat —
stays identical, so the existing [CameraGeometry] block still applies.
Then run generate_refracted_caldata.py on the new folder as usual.

Every new particle is
  * in the water (below the glass bottom, with a margin), and
  * visible to ALL cameras after refraction (same Maas shift + pinhole
    projection as generate_refracted_caldata.py), with a margin from the edges.

Usage (from the project root):
    python make_wide_cloud.py <step2_folder> <new_step2_folder> [--scale 3] [--seed 1]

    --scale      lateral extent and depth range relative to the current cloud (default 3)
    --n          number of particles (default: same as the current cloud)
    --max-depth  deepest allowed particle below the glass top, mm (default 600)
    --seed       random seed (default 1)
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np

import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))  # repo root on path
from Multimedia.Multimedia_params import MultimediaParams, CameraParams
from Multimedia.Iterative_radial_shift import multimedia_radial_shift


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step2_folder")
    ap.add_argument("new_folder")
    ap.add_argument("--scale", type=float, default=3.0)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--max-depth", type=float, default=600.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--edge-margin-px", type=float, default=10.0)
    args = ap.parse_args()

    src, dst = Path(args.step2_folder), Path(args.new_folder)
    if dst.exists():
        raise SystemExit(f"{dst} already exists - choose a new folder name")
    gt = json.load(open(src / "ground_truth.json"))

    # media (same defaults as generate_refracted_caldata.py, overridden by the file if present)
    m = gt.get("multimedia", {})
    glass = gt.get("glass", {})
    n1, n2, n3 = m.get("n1", 1.0), m.get("n2", 1.46), m.get("n3", 1.33)
    t = float(glass.get("thickness_mm", m.get("glass_thickness_mm", 5.0)))
    z_top = float(glass.get("z_top", m.get("z_glass_top", 0.0)))
    mm = MultimediaParams(n1=n1, n2=[n2], d=[t], n3=n3)

    res = np.atleast_2d(np.loadtxt(src / "Res.dat"))
    W, H = float(res[0, 0]), float(res[0, 1])

    cams = sorted(gt["cameras"], key=lambda c: c["id"])
    geo = []
    for c in cams:
        I = c["intrinsics"]
        geo.append((np.array(c["rotation_matrix"], float), np.array(c["position_mm"], float),
                    I["fx"], I["fy"], I["cx"], I["cy"],
                    CameraParams(position=np.array(c["position_mm"], float), mm=mm)))

    def project_all(P):
        """(n_cams, 2) pixel coords of one particle, refracted (Maas) + pinhole, as Step 3 does."""
        out = []
        for R, C, fx, fy, cx, cy, cp in geo:
            pb, _, _ = multimedia_radial_shift(P, cp, eps=1e-9, max_iter=300, ocf=1.3)
            pc = R @ (pb - C)
            if pc[2] <= 0:
                return None
            out.append((fx * pc[0] / pc[2] + cx, fy * pc[1] / pc[2] + cy))
        return np.array(out)

    P_old = np.array(gt["particles_mm"], float)
    n = args.n or len(P_old)
    cen = P_old.mean(0)
    half = 0.5 * np.ptp(P_old, axis=0)
    z_hi = z_top - t - 5.0                                   # in the water, 5 mm below the glass
    z_lo = max(z_top - args.max_depth, cen[2] - half[2] * args.scale)
    m_px = args.edge_margin_px

    rng = np.random.default_rng(args.seed)
    new, tried = [], 0
    while len(new) < n:
        tried += 1
        if tried > 200000:
            raise SystemExit("could not place enough visible particles - try a smaller --scale")
        p = np.array([rng.uniform(cen[0] - half[0] * args.scale, cen[0] + half[0] * args.scale),
                      rng.uniform(cen[1] - half[1] * args.scale, cen[1] + half[1] * args.scale),
                      rng.uniform(z_lo, z_hi)])
        try:
            uv = project_all(p)
        except Exception:
            continue
        if uv is None:
            continue
        if np.all((uv[:, 0] > m_px) & (uv[:, 0] < W - m_px) & (uv[:, 1] > m_px) & (uv[:, 1] < H - m_px)):
            new.append(p)
    P_new = np.array(new)

    # report: image coverage per camera, old vs new
    def coverage(P):
        uv = np.array([project_all(p) for p in P])           # (n, cams, 2)
        return [100 * np.ptp(uv[:, k, 0]) * np.ptp(uv[:, k, 1]) / (W * H) for k in range(len(geo))]
    cov_old, cov_new = coverage(P_old), coverage(P_new)

    shutil.copytree(src, dst)
    # Files that describe the OLD particles are stale now. generate_refracted_caldata.py
    # compares its new pixels with Step 2's points.dat before writing its outputs; with
    # a different particle count that comparison runs past the end of the old file and
    # crashes. Without the file it skips the comparison (it is only a printed check).
    for stale in ("points.dat", "IdMat.dat"):
        if (dst / stale).exists():
            (dst / stale).unlink()
            print(f"removed stale {stale} (it described the old particles)")
    gt["particles_mm"] = P_new.tolist()
    gt["source"] = (gt.get("source", "") + f" | particles replaced by make_wide_cloud.py "
                    f"(scale={args.scale}, seed={args.seed}, n={n})")
    gt["particle_cloud"] = {"generator": "make_wide_cloud.py", "scale": args.scale,
                            "seed": args.seed, "n": n}
    with open(dst / "ground_truth.json", "w") as f:
        json.dump(gt, f, indent=2)

    deepest = float(z_top - P_new[:, 2].min())                # depth below the glass top
    r_far = max(float(np.hypot(P_new[:, 0] - g[1][0], P_new[:, 1] - g[1][1]).max()) for g in geo)
    zp_max = int(np.ceil((deepest + 40.0) / 50.0) * 50)       # LUT depth axis, with margin
    n_zp = int(round(160 * zp_max / 300.0))                   # keep the 160x160 grid's depth spacing
    r_max = int(np.ceil((r_far + 150.0) / 50.0) * 50)         # LUT radial axis, generous margin (seed frames vary)
    n_r = int(round(160 * r_max / 376.0))                     # keep the 160x160 grid's radial spacing
    print(f"wrote {dst}  ({n} particles, scale {args.scale}, seed {args.seed}, {tried} draws)")
    print("image coverage per camera, old -> new: "
          + ", ".join(f"cam{c['id']} {a:.0f}% -> {b:.0f}%" for c, a, b in zip(cams, cov_old, cov_new)))
    print(f"depth below glass top: old {z_top - P_old[:, 2].max():.0f}-{z_top - P_old[:, 2].min():.0f} mm, "
          f"new {z_top - P_new[:, 2].max():.0f}-{deepest:.0f} mm; "
          f"largest radial distance from a camera nadir: {r_far:.0f} mm")
    print("SET IN THE NEW DATASET'S multicamselfcal.cfg, section [Multimedia]:")
    print(f"    LUT-Zp-Max = {zp_max}")
    print(f"    LUT-nZp    = {n_zp}")
    print(f"    LUT-R-Max  = {r_max}")
    print(f"    LUT-nR     = {n_r}")

if __name__ == "__main__":
    main()
