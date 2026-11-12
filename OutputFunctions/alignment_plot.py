"""
alignment_plot.py — Ground truth alignment and calibration result visualisation.

Loads a ground truth file (camera centres only, or full JSON with particles),
aligns the reconstructed cameras to the ground truth using a similarity
transform (Procrustes: scale + rotation + translation), then produces a
3D comparison plot and a text error report.

Supported ground truth file formats
-------------------------------------
1. .dat  — plain text, one camera centre per line, 3 space-separated floats.
           May be in any consistent unit or scale (normalised, mm, …).
           Only camera positions are compared.
           Example (original_cam_centers.dat):
               0.38877566795033641 -0.25224323017973899 0.54269896014519803
               ...

2. .json — structured JSON produced by the synthetic dataset generator.
           Must contain 'cameras' list with 'position_mm' per camera.
           May optionally contain 'particles_world_mm' for a particle cloud.
           Example (ground_truth.json):
               {"cameras": [{"cam_id": 1, "position_mm": [400,0,450]}, ...],
                "particles_world_mm": [[x,y,z], ...]}

Pipeline position
-----------------
Called from GoCal.py → Alignment routines block, after save and before
evalreprerror, using config.align_existing as the ground truth file path.
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict


# ── Ground truth loader ────────────────────────────────────────────────────────

def load_ground_truth(filepath: str) -> Dict:
    """
    Load a ground truth file into a standard dict.

    Parameters
    ----------
    filepath : str
        Path to a .dat or .json ground truth file.

    Returns
    -------
    gt : dict with keys:
        'cam_centres'   : np.ndarray (K, 3) — camera positions (any unit)
        'particles'     : np.ndarray (N, 3) or None — particle world coords
        'source'        : str — file format ('dat' or 'json')
        'unit'          : str — 'mm' when known from JSON, 'unknown' for .dat
        'n_cams'        : int
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Ground truth file not found: {filepath}")

    suffix = filepath.suffix.lower()

    if suffix == '.dat':
        data = np.loadtxt(filepath)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.shape[1] < 3:
            raise ValueError(f"Expected ≥3 columns in {filepath}, got {data.shape[1]}")
        return {
            'cam_centres': data[:, :3],
            'particles':   None,
            'source':      'dat',
            'unit':        'unknown',
            'n_cams':      data.shape[0],
        }

    elif suffix == '.json':
        with open(filepath) as f:
            raw = json.load(f)

        if 'cameras' not in raw:
            raise ValueError(f"JSON ground truth must contain a 'cameras' key")

        centres = np.array([c['position_mm'] for c in raw['cameras']])
        particles = None
        if 'particles_world_mm' in raw:
            particles = np.array(raw['particles_world_mm'])

        return {
            'cam_centres': centres,
            'particles':   particles,
            'source':      'json',
            'unit':        'mm',
            'n_cams':      centres.shape[0],
        }

    else:
        raise ValueError(f"Unsupported ground truth format: {suffix}. Use .dat or .json")


# ── Procrustes similarity alignment ───────────────────────────────────────────

def procrustes_align(source: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """
    Find the similarity transform (scale, rotation, translation) that maps
    source → target with minimum sum-of-squared distances.

    Both inputs are (K, 3) arrays — one row per camera.

    Algorithm
    ---------
    1. Centre both point sets (subtract centroid)
    2. Normalise both to unit RMS radius (removes scale)
    3. SVD to find optimal rotation: R = V @ Uᵀ
    4. Recover scale as ratio of RMS radii
    5. Compose: aligned = scale * (source_c @ Rᵀ) + target_centroid

    Parameters
    ----------
    source : (K, 3) — reconstructed camera centres (Ce[:3, :].T)
    target : (K, 3) — ground truth camera centres

    Returns
    -------
    aligned : (K, 3) — source after transform
    transform : dict with keys:
        'scale'        : float — scale factor applied to source
        'R'            : (3, 3) — rotation matrix
        't'            : (3,)  — translation vector
        'rms_before'   : float — RMS error before alignment (in target units)
        'rms_after'    : float — RMS error after alignment (in target units)
        'errors'       : (K,)  — per-camera Euclidean error after alignment
    """
    assert source.shape == target.shape, \
        f"Camera count mismatch: source={source.shape[0]}, target={target.shape[0]}"
    K = source.shape[0]

    # RMS error before alignment (raw, before any transform)
    rms_before = float(np.sqrt(np.mean(np.linalg.norm(source - target, axis=1) ** 2)))

    # Step 1: centre
    src_c = source - source.mean(axis=0)
    tgt_c = target - target.mean(axis=0)

    # Step 2: scale factors (RMS radius)
    src_scale = float(np.sqrt(np.mean(np.linalg.norm(src_c, axis=1) ** 2)))
    tgt_scale = float(np.sqrt(np.mean(np.linalg.norm(tgt_c, axis=1) ** 2)))

    if src_scale < 1e-12:
        raise ValueError("Reconstructed camera centres are all at the same point")

    src_n = src_c / src_scale
    tgt_n = tgt_c / tgt_scale

    # Step 3: optimal rotation via SVD of cross-covariance
    H   = src_n.T @ tgt_n          # (3, 3)
    U, S, Vt = np.linalg.svd(H)
    V   = Vt.T
    # Correct for reflection
    det = np.linalg.det(V @ U.T)
    D   = np.diag([1.0, 1.0, det])
    R   = V @ D @ U.T              # (3, 3) rotation matrix

    # Step 4: scale = tgt_scale / src_scale
    scale = tgt_scale / src_scale

    # Step 5: compose full transform
    t = target.mean(axis=0) - scale * (source.mean(axis=0) @ R.T)
    aligned = scale * (source @ R.T) + t

    errors    = np.linalg.norm(aligned - target, axis=1)   # per-camera (K,)
    rms_after = float(np.sqrt(np.mean(errors ** 2)))

    transform = {
        'scale':      scale,
        'R':          R,
        't':          t,
        'rms_before': rms_before,
        'rms_after':  rms_after,
        'errors':     errors,
    }
    return aligned, transform


def apply_transform(points: np.ndarray, transform: Dict) -> np.ndarray:
    """Apply a Procrustes transform dict to an arbitrary (N, 3) point array."""
    return transform['scale'] * (points @ transform['R'].T) + transform['t']


# ── Text error report ──────────────────────────────────────────────────────────

def _build_error_report(transform: Dict, gt: Dict, cam_ids: list,
                        aligned_cams: np.ndarray) -> str:
    """Build a formatted text error table."""
    lines = [
        "=" * 60,
        "CALIBRATION vs GROUND TRUTH — ALIGNMENT REPORT",
        "=" * 60,
        f"Ground truth source : {gt['source'].upper()} file",
        f"Ground truth unit   : {gt['unit']}",
        f"Similarity transform applied (Procrustes)",
        f"  Scale factor : {transform['scale']:.6f}",
        f"  RMS error before alignment : {transform['rms_before']:.4f}",
        f"  RMS error after  alignment : {transform['rms_after']:.4f}  "
        f"[{gt['unit']}]",
        "",
        "Per-camera position error (after alignment):",
        f"  {'Camera':<10} {'Error':>12}",
        "  " + "-" * 24,
    ]
    for cam_id, err in zip(cam_ids, transform['errors']):
        lines.append(f"  Cam {cam_id:<6} {err:>12.4f}  [{gt['unit']}]")
    lines += [
        "  " + "-" * 24,
        f"  {'Mean':<10} {transform['errors'].mean():>12.4f}  [{gt['unit']}]",
        f"  {'Max':<10} {transform['errors'].max():>12.4f}  [{gt['unit']}]",
        "=" * 60,
    ]
    return "\n".join(lines)


# ── 3D comparison plot ─────────────────────────────────────────────────────────

def _make_comparison_plot(gt: Dict, aligned_cams: np.ndarray,
                          transform: Dict, in_result: Dict,
                          cam_ids: list, data_path: str) -> str:
    """
    Produce a 3D matplotlib figure comparing GT and reconstructed cameras
    (and optionally particles). Saves to data_path/alignment_comparison.png.
    Returns the path to the saved figure.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except ImportError:
        print("  matplotlib not available — skipping plot")
        return ''

    K       = aligned_cams.shape[0]
    gt_cams = gt['cam_centres']   # (K, 3) — target (GT)

    fig = plt.figure(figsize=(13, 10), constrained_layout=True)
    fig.patch.set_facecolor('#0d1117')

    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#0d1117')
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor('#30363d')
    ax.tick_params(colors='#8b949e', labelsize=8)
    ax.xaxis.label.set_color('#8b949e')
    ax.yaxis.label.set_color('#8b949e')
    ax.zaxis.label.set_color('#8b949e')

    # ── GT cameras ────────────────────────────────────────────────────────────
    ax.scatter(gt_cams[:, 0], gt_cams[:, 1], gt_cams[:, 2],
               c='#3fb950', s=120, marker='*', zorder=5, label='Ground truth cameras',
               edgecolors='#ffffff', linewidths=0.5)
    for k in range(K):
        ax.text(gt_cams[k, 0], gt_cams[k, 1], gt_cams[k, 2],
                f'  GT{cam_ids[k]}', color='#3fb950', fontsize=7.5, zorder=6)

    # ── Reconstructed cameras (aligned) ───────────────────────────────────────
    ax.scatter(aligned_cams[:, 0], aligned_cams[:, 1], aligned_cams[:, 2],
               c='#f78166', s=80, marker='o', zorder=5, label='Reconstructed cameras',
               edgecolors='#ffffff', linewidths=0.5)
    for k in range(K):
        ax.text(aligned_cams[k, 0], aligned_cams[k, 1], aligned_cams[k, 2],
                f'  R{cam_ids[k]}', color='#f78166', fontsize=7.5, zorder=6)

    # ── Error lines connecting GT ↔ reconstructed ─────────────────────────────
    for k in range(K):
        xs = [gt_cams[k, 0], aligned_cams[k, 0]]
        ys = [gt_cams[k, 1], aligned_cams[k, 1]]
        zs = [gt_cams[k, 2], aligned_cams[k, 2]]
        ax.plot(xs, ys, zs, color='#d29922', linewidth=1.2,
                linestyle='--', alpha=0.8, zorder=4)
        mid = np.array([np.mean(xs), np.mean(ys), np.mean(zs)])
        err = transform['errors'][k]
        ax.text(mid[0], mid[1], mid[2],
                f'{err:.3f}', color='#d29922', fontsize=7, alpha=0.9)

    # ── Reconstructed 3D points (particle cloud) ──────────────────────────────
    Xe = in_result.get('Xe')
    if Xe is not None:
        pts = Xe[:3, :] / Xe[3:4, :]      # dehomogenise
        pts_aligned = apply_transform(pts.T, transform)
        # subsample if too many
        step = max(1, pts_aligned.shape[0] // 800)
        ps = pts_aligned[::step]
        ax.scatter(ps[:, 0], ps[:, 1], ps[:, 2],
                   c='#58a6ff', s=3, alpha=0.25, zorder=2,
                   label=f'Reconstructed points (n={pts.shape[1]})')

    # ── GT particles (if JSON format) ─────────────────────────────────────────
    if gt['particles'] is not None:
        gp = gt['particles']
        step = max(1, gp.shape[0] // 800)
        gps = gp[::step]
        ax.scatter(gps[:, 0], gps[:, 1], gps[:, 2],
                   c='#bc8cff', s=3, alpha=0.20, zorder=1,
                   label=f'GT particles (n={gp.shape[0]})')

    # ── Axes and title ────────────────────────────────────────────────────────
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    unit = gt['unit']
    rms  = transform['rms_after']
    ax.set_title(
        f'Calibration Alignment  —  RMS error: {rms:.4f} [{unit}]  '
        f'|  Scale: {transform["scale"]:.4f}',
        color='#e6edf3', fontsize=11, pad=14,
    )

    legend = ax.legend(loc='upper left', fontsize=8,
                       facecolor='#161b22', edgecolor='#30363d')
    for text in legend.get_texts():
        text.set_color('#e6edf3')

    # ── Error bar chart (inset table) ─────────────────────────────────────────
    ax2 = fig.add_axes([0.72, 0.12, 0.24, 0.22])
    ax2.set_facecolor('#161b22')
    bar_colors = ['#3fb950' if e < rms else '#f78166' for e in transform['errors']]
    bars = ax2.bar(
        [f'C{cid}' for cid in cam_ids],
        transform['errors'],
        color=bar_colors, width=0.55, zorder=3,
    )
    ax2.axhline(rms, color='#d29922', linewidth=1, linestyle='--',
                label=f'RMS={rms:.3f}')
    ax2.set_ylabel(f'Error [{unit}]', color='#8b949e', fontsize=8)
    ax2.tick_params(colors='#8b949e', labelsize=8)
    ax2.set_title('Per-camera error', color='#8b949e', fontsize=8, pad=4)
    for spine in ax2.spines.values():
        spine.set_edgecolor('#30363d')
    ax2.set_facecolor('#0d1117')
    ax2.legend(fontsize=7, facecolor='#161b22', edgecolor='#30363d',
               labelcolor='#d29922')

    out_path = os.path.join(data_path, 'alignment_comparison.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


# ── Public orchestrator ────────────────────────────────────────────────────────

def run_alignment(in_result: Dict, config, data_path: Optional[str] = None) -> Dict:
    """
    Align calibration results to a ground truth file and produce a
    comparison plot + text report.

    Parameters
    ----------
    in_result : dict
        Pipeline result dict with keys 'Ce' (4 × K homogeneous camera centres)
        and optionally 'Xe' (4 × N homogeneous 3D points).
    config : CalibrationConfig
        config.align_existing must be a non-empty string (path to GT file).
    data_path : str, optional
        Directory where outputs are saved. Defaults to config.data_path.

    Returns
    -------
    align : dict with keys:
        'X'         : np.ndarray (4, N) — points in GT coordinate frame
                      (for use by the points4cal save loop)
        'Ce_aligned': np.ndarray (4, K) — camera centres in GT frame
        'transform' : dict — scale, R, t, errors (from procrustes_align)
        'gt'        : dict — loaded ground truth
        'report'    : str — formatted text error report
    """
    gt_path = config.align_existing
    if not gt_path or gt_path.lower() in ('0', 'false', 'no', 'none', ''):
        return {}

    if data_path is None:
        data_path = config.data_path

    # Resolve path: try absolute, then relative to data_path
    if not os.path.isabs(gt_path):
        candidate = os.path.join(data_path, gt_path)
        if os.path.exists(candidate):
            gt_path = candidate

    print(f'\nAlignment to ground truth: {gt_path}')

    # Load ground truth
    gt = load_ground_truth(gt_path)
    n_gt = gt['n_cams']

    # Extract reconstructed camera centres: Ce is (4, K) homogeneous
    Ce = in_result['Ce']
    Ce_xyz = (Ce[:3, :] / Ce[3:4, :]).T   # → (K, 3)
    K = Ce_xyz.shape[0]

    if n_gt != K:
        print(f"  WARNING: GT has {n_gt} cameras, reconstruction has {K}. "
              f"Using first min({n_gt},{K}) cameras for alignment.")
        n_use = min(n_gt, K)
        Ce_xyz  = Ce_xyz[:n_use]
        gt['cam_centres'] = gt['cam_centres'][:n_use]
        gt['n_cams'] = n_use
        K = n_use

    cam_ids = list(config.cameras_2_use[:K]) if config.cameras_2_use else list(range(1, K+1))

    # Procrustes alignment: source = reconstructed, target = GT
    aligned_cams, transform = procrustes_align(Ce_xyz, gt['cam_centres'])

    # Apply transform to 3D points
    Xe_aligned = None
    Ce_hom_aligned = np.ones((4, K))
    Ce_hom_aligned[:3, :] = aligned_cams.T

    Xe = in_result.get('Xe')
    if Xe is not None:
        pts = Xe[:3, :] / Xe[3:4, :]          # (3, N)
        pts_aligned = apply_transform(pts.T, transform).T   # (3, N)
        Xe_aligned = np.vstack([pts_aligned, np.ones((1, pts.shape[1]))])

    # Build text report
    report = _build_error_report(transform, gt, cam_ids, aligned_cams)
    print(report)

    # Save text report
    report_path = os.path.join(data_path, 'alignment_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f'  Alignment report saved: {report_path}')

    # Produce comparison plot
    plot_path = _make_comparison_plot(
        gt, aligned_cams, transform, in_result, cam_ids, data_path)
    if plot_path:
        print(f'  Comparison plot saved: {plot_path}')

    return {
        'X':          Xe_aligned if Xe_aligned is not None else Xe,
        'Ce_aligned': Ce_hom_aligned,
        'transform':  transform,
        'gt':         gt,
        'report':     report,
    }
