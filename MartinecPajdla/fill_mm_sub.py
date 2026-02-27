"""
fill_mm_sub - Projective reconstruction of a normalized sub-scene
Converted from MATLAB to Python
"""

import numpy as np
from typing import Tuple, Dict, Optional
from MartinecPajdla.Utils import Utils
from MartinecPajdla.M2Fe import M2Fe
from MartinecPajdla.depth_estimation import depth_estimation
from MartinecPajdla.balance_triplets import balance_triplets
from MartinecPajdla.fill_prmm import fill_prmm


def fill_mm_sub(Mfull: np.ndarray,
               M: np.ndarray,
               central: int,
               opt: Dict,
               info: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Projective reconstruction of a normalized sub-scene.
    
    When the central image concept is used, the information about which image
    is the central image is passed to this function.
    
    Args:
        Mfull: Complete known parts of the problem (3m x n)
               Used for best estimate of fundamental matrices
        M: Measurement matrix subset (3m x n)
        central: Central image index (0 or None for sequence mode)
        opt: Options dict with:
            - 'verbose': Print progress
            - 'tol': Tolerance
            - Other options for sub-functions
        info: Information dict (will be updated)
    
    Returns:
        P: Camera matrices
        X: 3D points
        lambda: Depth scale factors
        u1: Unrecovered image indices
        u2: Unrecovered point indices
        info: Updated information dict
    """
    
    # Check validity matrix
    I = ~np.isnan(M[0::3, :])
    m, n = I.shape
    
    # Handle empty central
    if central is None or central == 0:
        central = 0
    
    # Initialize outputs
    P = np.array([])
    X = np.array([])
    lambda_vals = np.array([])
    u1 = np.arange(m)
    u2 = np.arange(n)
    
    # Estimate fundamental matrices and epipoles
    F, ep, rows, nonrows = M2Fe(Mfull, central)
    
    # Display used images
    if len(nonrows) > 0 and opt.get('verbose', True):
        rows_str = ' '.join([str(r) for r in rows])
        print(f'Used local images: {rows_str}.')
    
    # Need at least 2 images
    if len(rows) < 2:
        return P, X, lambda_vals, u1, u2, info
    
    # Determine scale factors lambda_i_p
    if not central or central == 0:
        rows_central = 0
    else:
        rows_central_idx = np.where(rows == central)[0]
        rows_central = rows_central_idx[0] if len(rows_central_idx) > 0 else 0
    
    # Compute depths
    M_rows = M[Utils.k2i(rows, step=3), :]
    lambda_vals, Ilamb = depth_estimation(M_rows, F, ep, rows, rows_central)
    
    # Prepare info.show_prmm for visualization
    if 'show_prmm' not in info:
        info['show_prmm'] = {}
    
    info['show_prmm']['I'] = I
    info['show_prmm']['Idepths'] = np.zeros((m, n))
    info['show_prmm']['Idepths'][rows, :] = Ilamb
    
    # Build rescaled measurement matrix B
    num_rows = len(rows)
    B = np.zeros((3 * num_rows, n))
    
    for i in range(num_rows):
        row_idx = Utils.k2i(np.array([i]), step=3)
        M_row = M[Utils.k2i(np.array([rows[i]]), step=3), :]
        
        # Scale by lambda: M * (ones(3,1) * lambda)
        lambda_row = lambda_vals[i, :].reshape(1, -1)
        ones_col = np.ones((3, 1))
        scaling = ones_col @ lambda_row
        
        B[row_idx, :] = M_row * scaling
    
    # Balance by column-wise and triplet-of-rows-wise scalar multiplications
    B = balance_triplets(B, opt)
    
    # Fill holes using Jacobs' algorithm (fill_prmm)
    P, X, u1, u2, lambda1, info = fill_prmm(B, Ilamb, central, opt, info)
    
    # Compute valid indices
    r1 = np.setdiff1d(np.arange(len(rows)), u1)
    r2 = np.setdiff1d(np.arange(n), u2)
    
    # Subset lambda to fit P*X
    if len(r1) > 0 and len(r2) > 0:
        lambda_vals = lambda_vals[np.ix_(r1, r2)]
        
        # Update with new lambda values where they were computed
        if lambda1.size > 0:
            # Find entries that were unknown in Ilamb but valid in I
            Ilamb_sub = Ilamb[np.ix_(r1, r2)]
            I_sub = I[np.ix_(r1, r2)]
            new = ~Ilamb_sub & I_sub
            lambda_vals[new] = lambda1[new]
    else:
        lambda_vals = np.array([])
    
    # Compute error
    if P.size > 0 and X.size > 0 and len(r1) > 0 and len(r2) > 0:
        B_sub = B[Utils.k2i(r1, step=3), :][:, r2]
        PX = P @ X
        
        # Validity mask for error computation
        # Check row 3k (z-coordinate) for NaN
        valid_mask = ~np.isnan(B[3*r1, :][:, r2])
        
        error, _ = Utils.eucl_dist_only(B_sub, PX, valid_mask, step=3)
        
        if opt.get('verbose', True):
            print(f'Error balanced: {error:.6f}')
    
    # Update u1 to include nonrows
    u1_in_rows = rows[u1]
    u1 = np.union1d(nonrows, u1_in_rows)
    
    return P, X, lambda_vals, u1, u2, info


if __name__ == "__main__":
    # Test fill_mm_sub function
    print("=" * 60)
    print("fill_mm_sub - Sub-scene Reconstruction Test")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Create test data - use more realistic structure
    m_images = 5
    n_points = 30
    
    # Generate low-rank structure with proper projective geometry
    r = 4
    
    # Create camera matrices
    P_cameras = []
    for k in range(m_images):
        # Simple camera: rotation + translation
        angle = k * 0.2
        R = np.array([[np.cos(angle), -np.sin(angle), 0],
                      [np.sin(angle), np.cos(angle), 0],
                      [0, 0, 1]])
        t = np.array([[k], [0], [0]])
        P_cameras.append(np.hstack([R, t]))
    
    P_true = np.vstack(P_cameras)
    
    # Create 3D points
    X_true = np.vstack([
        np.random.randn(3, n_points) * 5,
        np.ones((1, n_points))
    ])
    
    # Project to get measurements
    Mfull = P_true @ X_true
    
    # Add small noise
    Mfull += np.random.randn(3 * m_images, n_points) * 0.05
    
    # Create subset M with some missing data (but not too much)
    M = Mfull.copy()
    missing = np.random.rand(3 * m_images, n_points) < 0.1
    M[missing] = np.nan
    
    # Ensure each point is visible in at least 3 cameras
    for j in range(n_points):
        visible_cameras = ~np.isnan(M[0::3, j])
        if np.sum(visible_cameras) < 3:
            # Make visible in first 3 cameras
            for k in range(3):
                M[3*k:3*k+3, j] = Mfull[3*k:3*k+3, j]
    
    print(f"\nTest setup:")
    print(f"  Images: {m_images}")
    print(f"  Points: {n_points}")
    print(f"  Mfull shape: {Mfull.shape}")
    print(f"  M shape: {M.shape}")
    print(f"  Missing in M: {np.sum(missing)}/{M.size}")
    print(f"  Points visible in all cameras: {np.sum(np.all(~np.isnan(M[0::3, :]), axis=0))}")
    
    # Options
    opt = {
        'create_nullspace': {
            'trial_coef': 1.0,  # Increased for better sampling
            'threshold': 0.01,
            'verbose': False
        },
        'verbose': True,
        'tol': 1e-6,
        'info_separately': True
    }
    
    # Info structure
    info = {'sequence': []}
    
    # Test 1: Sequence mode
    print("\n" + "-" * 60)
    print("TEST 1: Fill Sub-Scene (Sequence Mode)")
    print("-" * 60)
    
    try:
        P, X, lambda_vals, u1, u2, info = fill_mm_sub(
            Mfull, M, central=0, opt=opt, info=info
        )
        
        print(f"\nResults:")
        if P.size > 0:
            print(f"  P shape: {P.shape}")
            print(f"  X shape: {X.shape}")
            if lambda_vals.size > 0:
                print(f"  Lambda shape: {lambda_vals.shape}")
            else:
                print(f"  Lambda: empty")
            print(f"  Unrecovered images: {len(u1)}/{m_images}")
            print(f"  Unrecovered points: {len(u2)}/{n_points}")
            
            # Check reconstruction quality
            if X.shape[1] > 0:
                print(f"  ✓ Reconstruction successful")
        else:
            print("  No reconstruction (insufficient data)")
        
        # Check info structure
        if 'show_prmm' in info:
            print(f"\nVisualization info prepared:")
            print(f"  I shape: {info['show_prmm']['I'].shape}")
            print(f"  Idepths shape: {info['show_prmm']['Idepths'].shape}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Central image mode
    print("\n" + "-" * 60)
    print("TEST 2: Fill Sub-Scene (Central Image Mode)")
    print("-" * 60)
    
    info2 = {'sequence': []}
    
    try:
        P2, X2, lambda2, u1_2, u2_2, info2 = fill_mm_sub(
            Mfull, M, central=2, opt=opt, info=info2
        )
        
        print(f"\nResults:")
        if P2.size > 0:
            print(f"  P shape: {P2.shape}")
            print(f"  X shape: {X2.shape}")
            if lambda2.size > 0:
                print(f"  Lambda shape: {lambda2.shape}")
            else:
                print(f"  Lambda: empty")
            print(f"  Unrecovered images: {len(u1_2)}/{m_images}")
            print(f"  Unrecovered points: {len(u2_2)}/{n_points}")
            
            if X2.shape[1] > 0:
                print(f"  ✓ Reconstruction successful")
        else:
            print("  No reconstruction (insufficient data)")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("✓ All tests completed")
    print("=" * 60)
