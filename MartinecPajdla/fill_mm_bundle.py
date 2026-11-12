"""
fill_mm_bundle - Projective reconstruction from measurement matrix with bundle adjustment.

This is the highest-level wrapper function that combines:
1. fill_mm - Main projective reconstruction
2. bundle_PX_proj - Bundle adjustment refinement
"""

import numpy as np
import time
from typing import Tuple, Dict, Optional
from MartinecPajdla.Utils import Utils
from MartinecPajdla.fill_mm import fill_mm
from MartinecPajdla.bundle_PX_proj import bundle_PX_proj

# Note: Multimedia ray-tracing imports were removed because the function
# below (_multimedia_refine_X) is intentionally disabled — see the comment
# in fill_mm_bundle() Step 2 for the rationale. If you ever re-enable that
# code path, move the imports inside _multimedia_refine_X as a lazy import
# so this module continues to load without the Multimedia package.

def _multimedia_refine_X(M, X, r1, r2, cameras_mm, z_glass_top):
    # Lazy import — only happens if/when this function is called.
    from Multimedia.Forward_ray_tracing import (
        trace_ray_through_media,
        spatial_intersection_multi,
    )
    """
    Refine 3D point positions using multimedia spatial intersection (Maas 2.2.2).

    Called after fill_mm returns projective X, when metric camera positions
    from a previous iteration are available in cameras_mm. For each recovered
    point, the function collects refracted rays from every camera that observes
    it and intersects them using spatial_intersection_multi().

    Parameters
    ----------
    M : np.ndarray, shape (3m x n)
        Measurement matrix with NaN for missing observations.
    X : np.ndarray, shape (4 x n)
        Projective 3D points from fill_mm (modified in-place copy).
    r1 : np.ndarray
        Indices of recovered cameras (rows in M are 3*r1 : 3*r1+3).
    r2 : np.ndarray
        Indices of recovered points (columns in M).
    cameras_mm : list of CameraParams
        One per camera row-triple in M, from the previous euclidize() call.
    z_glass_top : float
        Z-coordinate of the air/glass interface in mm.

    Returns
    -------
    X_ref : np.ndarray, shape (4 x n)
        Copy of X with refined positions for all successfully intersected points.
    n_refined : int
        Number of points where multimedia intersection replaced the projective estimate.
    mean_rms : float
        Mean RMS ray distance across all refined points (mm). Quality indicator.
    """
    X_ref = X.copy()
    rms_values = []
    n_refined = 0

    for j in r2:
        ray_positions  = []
        ray_directions = []

        for i in r1:
            obs = M[3*i:3*i+3, j]
            if np.any(np.isnan(obs)) or abs(obs[2]) < 1e-12:
                continue  # this camera does not see this point

            # Normalise homogeneous image coords to metric 2D
            x_img = obs[0] / obs[2]
            y_img = obs[1] / obs[2]

            try:
                pos_w, dir_w = trace_ray_through_media(
                    cameras_mm[i], x_img, y_img, z_glass_top)
                ray_positions.append(pos_w)
                ray_directions.append(dir_w)
            except (ValueError, IndexError):
                continue  # total internal reflection or geometry error — skip ray

        if len(ray_positions) < 2:
            continue  # need at least 2 rays to intersect

        P_hat, rms = spatial_intersection_multi(ray_positions, ray_directions)
        X_ref[:3, j] = P_hat
        X_ref[3,  j] = 1.0
        rms_values.append(rms)
        n_refined += 1

    mean_rms = float(np.mean(rms_values)) if rms_values else 0.0
    return X_ref, n_refined, mean_rms


def fill_mm_bundle(M: np.ndarray,
                   imsize: np.ndarray,
                   opt: Optional[Dict] = None,
                   cameras_mm=None,
                   lut=None,
                   z_glass_top: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Projective reconstruction from measurement matrix with optional bundle adjustment.
    Calls fill_mm for initial reconstruction, then optionally refines with bundle adjustment.
    
    Args:
        M: Measurement matrix (3m x n) with NaNs for unknown elements
        imsize: Image sizes (2 x m), imsize[:, i] is size of image i
        opt: Options dict with:
            - 'no_BA' (False): Whether to skip bundle adjustment
            - 'verbose' (True): Print progress
            - 'verbose_short': Short progress (see bundle_PX_proj)
            - ... other options from fill_mm
        cameras_mm: list of CameraParams, one per camera row-triple in M.
            None on the first global iteration (no metric positions yet).
            Provided from the second iteration onward — enables multimedia
            spatial intersection refinement of X (Step 3.6, Maas 2.2.2).
        lut: MultiCameraLUT — reserved for Step 3.9 (bundle_PX_proj integration).
            Not used inside this function yet; passed through for future use.
        z_glass_top: Z-coordinate of the air/glass interface in mm.
    
    Returns:
        P: Camera matrices (3k x 4) for k recovered images
        X: 3D points (4 x n') for n' recovered points
        u1: Unrecovered image indices
        u2: Unrecovered point indices
        info: Information dict with:
            - 'R_lin': Linear estimation (P @ X before bundle adjustment)
            - 'err': Error dict with 'BA' key if bundle adjustment was run
            - 'mm_refined': True if multimedia X refinement was applied
            - 'mm_n_refined': number of points refined
            - 'mm_mean_rms': mean RMS ray distance after intersection (mm)
            - ... other fields from fill_mm
    
    This is the complete pipeline:
        1. fill_mm: Structure-from-motion reconstruction
        2. Multimedia spatial intersection (optional, Step 3.6)
        3. bundle_PX_proj: Nonlinear refinement (optional)
    """
    
    # Set default options
    if opt is None:
        opt = {}
    if 'no_BA' not in opt:
        opt['no_BA'] = False
    if 'verbose' not in opt:
        opt['verbose'] = True
    
    # Step 1: Initial reconstruction with fill_mm
    P, X, u1, u2, info = fill_mm(M, opt)
    
    # Store linear reconstruction result
    info['R_lin'] = P @ X

    # Step 2: Multimedia spatial intersection refinement (Maas 2.2.2, Step 3.6)
    # Replaces projective X positions with refracted-ray intersections in water.
    # Only active from iteration 2 onward, when cameras_mm holds metric positions
    # from the previous euclidize() call. On iteration 0 cameras_mm is None.
    m = M.shape[0] // 3  # Number of images
    n = M.shape[1]        # Number of points

    info['mm_refined']   = False
    info['mm_n_refined'] = 0
    info['mm_mean_rms']  = 0.0

    # NOTE: _multimedia_refine_X is intentionally disabled.
    #
    # Reason: it replaces projective X values (~0–1, dimensionless) with
    # metric world coordinates in mm (~100–400). bundle_PX_proj (below) calls
    # normx() on X immediately afterward — normx() computes a scale factor from
    # the RMS of all values. With mixed scales (some ~0.1, some ~300), the
    # conditioning blows up and the BA diverges (observed error: ~5000 instead
    # of ~0.3 in projective units).
    #
    # The multimedia X refinement is correctly handled AFTER euclidize() by
    # multimedia_bundle_adjustment() in GoCal.py, where X is already in
    # metric world coordinates. No action needed here.
    #
    # if cameras_mm is not None and len(u1) < m and len(u2) < n:
    #     X, n_ref, mean_rms = _multimedia_refine_X(M, X, r1, r2, ...)

    # Step 3: Bundle adjustment (if requested and feasible)
    # Only run bundle adjustment if:
    # - not disabled by option
    # - some cameras were recovered (len(u1) < m)
    # - some points were recovered (len(u2) < n)
    if not opt['no_BA'] and len(u1) < m and len(u2) < n:
        if opt['verbose']:
            print('Bundle adjustment...', end='', flush=True)
            start_time = time.time()
        
        # Get valid indices
        r1 = np.setdiff1d(np.arange(m), u1)  # Recovered images
        r2 = np.setdiff1d(np.arange(n), u2)  # Recovered points
        
        # Prepare observations: normalize_cut converts to Euclidean 2D
        M_subset = M[Utils.k2i(r1, step=3), :][:, r2]
        q = Utils.normalize_cut(M_subset)
        
        # Run bundle adjustment.
        # IMPORTANT: imsize must be (2, m) — 2 rows (width, height), m columns (cameras).
        # If passed as (m, 2) (transposed), fix it here.
        imsize_use = imsize if imsize.shape[0] == 2 else imsize.T
        # Pass only the recovered cameras imsize columns so indices align with P/q.
        # MATLAB passes full imsize and uses columns 1..K, assuming first K are recovered.
        # The correct approach is to index by r1 explicitly.
        P, X = bundle_PX_proj(P, X, q, imsize_use[:, r1])
        
        # Note: Old bundler was qPXbundle_cmp, now using bundle_PX_proj
        
        if opt['verbose']:
            elapsed = time.time() - start_time
            print(f' ({elapsed:.3f} sec)')
        
        # Compute error after bundle adjustment
        metric = info['opt'].get('metric', 1) if 'opt' in info else 1
        info['err']['BA'] = Utils.dist(M_subset, P @ X, metric)
        
        if opt['verbose']:
            print(f"Error (after BA): {info['err']['BA']:.6f}")
        else:
            print(f" {info['err']['BA']:.6f}")
    else:
        # Bundle adjustment skipped
        if not opt['verbose']:
            print()  # Newline
    
    return P, X, u1, u2, info


if __name__ == "__main__":
    # Test fill_mm_bundle function
    print("=" * 60)
    print("fill_mm_bundle - Complete Reconstruction Pipeline Test")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Create realistic test data
    m_images = 4
    n_points = 30
    
    # Image sizes
    imsize = np.array([[640, 480]] * m_images).T  # (2 x m)
    
    # Generate cameras
    P_cameras = []
    for k in range(m_images):
        angle = k * 0.2
        R = np.array([[np.cos(angle), -np.sin(angle), 0],
                      [np.sin(angle), np.cos(angle), 0],
                      [0, 0, 1]])
        t = np.array([[k * 1.0], [0], [0]])
        P_cameras.append(np.hstack([R, t]))
    
    P_true = np.vstack(P_cameras)
    
    # Generate 3D points
    X_true = np.vstack([
        np.random.randn(3, n_points) * 4,
        np.ones((1, n_points))
    ])
    
    # Project to get measurements
    M = P_true @ X_true
    
    # Add noise
    M += np.random.randn(3 * m_images, n_points) * 0.05
    
    # Add missing data carefully
    # First 15 points visible in all cameras (for fundamental matrix)
    missing = np.zeros((3 * m_images, n_points), dtype=bool)
    missing[:, 15:] = np.random.rand(3 * m_images, n_points - 15) < 0.15
    
    # Ensure each point visible in ≥2 cameras
    for j in range(n_points):
        visible = ~missing[0::3, j]
        if np.sum(visible) < 2:
            for k in range(min(2, m_images)):
                missing[3*k:3*k+3, j] = False
    
    M[missing] = np.nan
    
    print(f"\nTest setup:")
    print(f"  Images: {m_images}")
    print(f"  Points: {n_points}")
    print(f"  Image size: {imsize[:, 0]}")
    print(f"  Missing: {np.sum(missing)}/{M.size}")
    print(f"  Fully visible points: {np.sum(np.all(~np.isnan(M[0::3, :]), axis=0))}")
    
    # Test 1: With bundle adjustment (default)
    print("\n" + "-" * 60)
    print("TEST 1: Complete Pipeline (with Bundle Adjustment)")
    print("-" * 60)
    
    opt = {
        'strategy': 0,  # Sequence mode
        'create_nullspace': {
            'trial_coef': 1.0,
            'threshold': 0.01,
            'verbose': False
        },
        'verbose': True,
        'no_factorization': False,
        'metric': 1,
        'no_BA': False,  # Enable bundle adjustment
        'max_niter': 20,
        'lam_init': 1e-3
    }
    
    try:
        P, X, u1, u2, info = fill_mm_bundle(M, imsize, opt)
        
        print(f"\n{'='*60}")
        print("RESULTS:")
        print(f"{'='*60}")
        
        if P.size > 0:
            print(f"✓ Reconstruction successful!")
            print(f"  P shape: {P.shape}")
            print(f"  X shape: {X.shape}")
            print(f"  Recovered images: {m_images - len(u1)}/{m_images}")
            print(f"  Recovered points: {n_points - len(u2)}/{n_points}")
            
            if 'err' in info:
                if 'no_fact' in info['err']:
                    print(f"  Error (no fact): {info['err']['no_fact']:.6f}")
                if 'fact' in info['err']:
                    print(f"  Error (factorized): {info['err']['fact']:.6f}")
                if 'BA' in info['err']:
                    print(f"  Error (bundle adj): {info['err']['BA']:.6f}")
            
            if 'R_lin' in info:
                print(f"  Linear result stored: {info['R_lin'].shape}")
        else:
            print("✗ Reconstruction failed")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Without bundle adjustment
    print("\n" + "-" * 60)
    print("TEST 2: Reconstruction Only (no Bundle Adjustment)")
    print("-" * 60)
    
    opt2 = opt.copy()
    opt2['no_BA'] = True  # Disable bundle adjustment
    
    try:
        P2, X2, u1_2, u2_2, info2 = fill_mm_bundle(M, imsize, opt2)
        
        print(f"\nResults:")
        if P2.size > 0:
            print(f"  P shape: {P2.shape}")
            print(f"  X shape: {X2.shape}")
            print(f"  Recovered images: {m_images - len(u1_2)}/{m_images}")
            print(f"  Recovered points: {n_points - len(u2_2)}/{n_points}")
            
            if 'err' in info2:
                if 'fact' in info2['err']:
                    print(f"  Error (factorized): {info2['err']['fact']:.6f}")
                if 'BA' in info2['err']:
                    print(f"  Error (bundle adj): {info2['err']['BA']:.6f}")
                else:
                    print(f"  Bundle adjustment: SKIPPED")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("✓ All tests completed")
    print("=" * 60)
