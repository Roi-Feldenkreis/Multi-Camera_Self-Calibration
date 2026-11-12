#!/usr/bin/env python3
"""
make_camera_geometry.py
-----------------------
Generate a synthetic [CameraGeometry] section for multicamselfcal.cfg from the
true camera positions in ground_truth.json.

Because no rig with measured inter-camera distances is available, this simulates
the six pairwise camera-to-camera distances a person would measure by hand:
it takes the TRUE distances from the ground truth and adds a few millimetres of
noise (default sigma = 4 mm) to mimic measuring between camera bodies rather than
optical centres. The result is a soft constraint the multimedia bundle adjustment
uses to fix the scale ambiguity.

Usage:
    python make_camera_geometry.py \
        --gt  /path/to/ground_truth.json \
        --sigma 4.0 \
        --weight 1.0 \
        [--seed 0] [--emit-clean]

Prints a ready-to-paste [CameraGeometry] block. With --emit-clean it also prints
a second block using the exact (noise-free) distances, useful as an upper-bound
sanity check (how well the pipeline does if the measurements were perfect).
"""
import argparse
import itertools
import json
import numpy as np


def load_true_positions(gt_path):
    with open(gt_path) as f:
        gt = json.load(f)
    cams = gt["cameras"]
    # sort by id so indices are deterministic and 0-based
    cams = sorted(cams, key=lambda c: c["id"])
    pos = {}
    for idx, c in enumerate(cams):
        pos[idx] = np.asarray(c["position_mm"], dtype=float)
    return pos


def load_true_los(gt_path):
    """True distance from each camera, along its optical axis, to the glass top
    surface (mm) — what a person measures by hand along the line of sight.
    Ground-truth rotations are world->camera with the optical axis as the third
    row; the axis is signed to point down toward the glass."""
    with open(gt_path) as f:
        gt = json.load(f)
    z_glass = float(gt.get("glass", {}).get("z_top", 0.0))
    out = {}
    for idx, c in enumerate(sorted(gt["cameras"], key=lambda c: c["id"])):
        a = np.asarray(c["rotation_matrix"], dtype=float)[2]
        if a[2] > 0:
            a = -a
        h = float(c["position_mm"][2]) - z_glass
        out[idx] = h / -a[2]
    return out


def pairwise(pos):
    out = []
    for i, j in itertools.combinations(sorted(pos), 2):
        d = float(np.linalg.norm(pos[i] - pos[j]))
        out.append((i, j, d))
    return out


def cyclic_order_ccw(pos):
    """Cameras in a cyclic order around the rig, oriented so the true constellation
    has POSITIVE signed tetrahedron volume. This encodes the rig handedness (which
    the six scalar distances cannot) and lets the gatekeeper reject the mirror-image
    reconstruction via the sign of the signed volume — robust to every reflection
    plane, unlike an in-plane winding test."""
    idx = sorted(pos)
    P = np.array([pos[i] for i in idx])
    c = P[:, :2] - P[:, :2].mean(0)
    ang = np.arctan2(c[:, 1], c[:, 0])
    order = [idx[k] for k in np.argsort(ang)]          # start from an angular sort
    # orient so the true signed volume is positive (reverse if not)
    o = order[:4]
    a, b, cc, d = (np.array(pos[o[0]]), np.array(pos[o[1]]),
                   np.array(pos[o[2]]), np.array(pos[o[3]]))
    vol = float(np.dot(np.cross(b - a, cc - a), d - a))
    if vol < 0:
        order = order[::-1]
    return order


def emit(pairs, order, header, los=None):
    lines = [f"# {header}", "[CameraGeometry]",
             "# Weight is fixed in code (not read from here).",
             f"CyclicOrder = {', '.join(str(k) for k in order)}"]
    for (i, j, d) in pairs:
        lines.append(f"Dist-{i}-{j} = {d:.2f}")
    if los:
        lines.append("# camera-to-glass distance along each camera's line of sight (mm)")
        for i in sorted(los):
            lines.append(f"LOS-{i} = {los[i]:.2f}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, help="path to ground_truth.json")
    ap.add_argument("--sigma", type=float, default=4.0,
                    help="hand-measurement noise std (mm), default 4.0")
    ap.add_argument("--sigma-los", type=float, default=5.0,
                    help="line-of-sight hand-measurement noise std (mm), default 5.0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--emit-clean", action="store_true",
                    help="also print a noise-free block")
    args = ap.parse_args()

    pos = load_true_positions(args.gt)
    clean = pairwise(pos)
    order = cyclic_order_ccw(pos)

    rng = np.random.default_rng(args.seed)
    noisy = [(i, j, d + float(rng.normal(0.0, args.sigma))) for (i, j, d) in clean]
    # separate random stream: the inter-camera distances above stay identical
    los_clean = load_true_los(args.gt)
    rng_los = np.random.default_rng(args.seed + 1000)
    los_noisy = {i: d + float(rng_los.normal(0.0, args.sigma_los)) for i, d in sorted(los_clean.items())}

    print(emit(noisy, order,
               f"synthetic hand-measured distances (sigma={args.sigma} mm, "
               f"line of sight sigma={args.sigma_los} mm, seed={args.seed})", los_noisy))
    print()
    print("# true distances (for reference, do NOT need to paste):")
    for (i, j, d) in clean:
        print(f"#   {i}-{j}: {d:.2f} mm")
    for i, d in sorted(los_clean.items()):
        print(f"#   LOS-{i}: {d:.2f} mm")
    if args.emit_clean:
        print()
        print(emit(clean, order, "NOISE-FREE distances (perfect-measurement upper bound)", los_clean))


if __name__ == "__main__":
    main()
