"""
fill_prmm - Compute null-space and fill PRMM (Projective Reconstruction Measurement Matrix)
Converted from MATLAB to Python
"""

import numpy as np
import time
from typing import Tuple, Dict, Optional
from MartinecPajdla.Utils import Utils
from MartinecPajdla.create_nullspace import create_nullspace
from MartinecPajdla.L2depths import L2depths
from MartinecPajdla.approximate import approximate


def fill_prmm(M: np.ndarray,
             Idepths: np.ndarray,
             central: int,
             opt: Dict,
             info: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Compute null-space and fill PRMM (Projective Reconstruction Measurement Matrix).
    
    Args:
        M: Measurement matrix (3m x n)
        Idepths: Depth indicator matrix (m x n)
        central: Central image index (0 for sequence mode)
        opt: Options dict with:
            - 'create_nullspace': Options for create_nullspace
            - 'verbose': Print progress
        info: Information dict (will be updated)
    
    Returns:
        P: Camera matrices
        X: 3D points
        u1: Unrecovered row indices
        u2: Unrecovered column indices
        lambda: Depth scale factors
        info: Updated information dict
    """
    
    # Create null-space
    NULLSPACE, result = create_nullspace(M, Idepths, central, opt['create_nullspace'])
    
    # Update info structure
    info['create_nullspace'] = opt['create_nullspace']
    
    # Get last sequence entry (or create new one if empty)
    if 'sequence' not in info or len(info['sequence']) == 0:
        info['sequence'] = [{}]
    
    last_seq = info['sequence'][-1]
    n_points = M.shape[1]
    
    # Compute statistics
    last_seq['tried'] = result['tried']
    last_seq['tried_perc'] = result['tried'] / Utils.comb(n_points, 4) * 100
    last_seq['used'] = result['used']
    last_seq['used_perc'] = result['used'] / result['tried'] * 100 if result['tried'] > 0 else 0
    last_seq['failed'] = result['failed']
    last_seq['size_nullspace'] = NULLSPACE.shape
    
    if opt['verbose']:
        print(f"Tried/used: {result['tried']}/{result['used']} "
              f"({last_seq['tried_perc']:.1e} %/ {last_seq['used_perc']:.1f} %)")
        print(f"{NULLSPACE.shape[1]} x {NULLSPACE.shape[0]}' is size of the nullspace")
    
    # Get dimensions
    m = M.shape[0] // 3  # Number of images
    n = M.shape[1]        # Number of points
    
    # Check if null-space is empty
    if NULLSPACE.shape[1] == 0:
        P = np.array([])
        X = np.array([])
        u1 = np.arange(m)
        u2 = np.arange(n)
        lambda_vals = np.array([])
    else:
        r = 4  # Rank for projective reconstruction
        
        # Compute basis from null-space
        L, S = nullspace2L(NULLSPACE, r, opt)
        del NULLSPACE  # Free memory
        
        # Get threshold
        if 'create_nullspace' not in opt or opt['create_nullspace'] is None:
            threshold = 0.01
        else:
            threshold = opt['create_nullspace'].get('threshold', 0.01)
        
        # Check if SVD has sufficient data
        if svd_suff_data(S, r, threshold):
            if opt['verbose']:
                dS = np.diag(S)
                sv_str = ' '.join([f'{v:.6f}' for v in dS[-2*r:]])
                print(f'Smallest 2r singular values: {sv_str}.')
            
            # Compute depths from basis
            Mdepths, lambda_vals = L2depths(L, M, Idepths, opt)
            info['Mdepths'] = Mdepths
            
            # Approximate with rank r
            P, X, u1b, u2 = approximate(Mdepths, r, L, opt)
            
            # Process unrecovered indices
            # Convert row indices to image indices
            u1 = np.unique((u1b // 3).astype(int))  # 0-indexed: integer division gives image index
            
            # Find rows to kill
            killb = np.setdiff1d(Utils.k2i(u1, step=3), u1b)
            
            if len(killb) > 0:
                r1b = np.setdiff1d(np.arange(3 * m), u1b)
                kill = killb.copy()
                
                # Adjust kill indices
                for idx, ib in enumerate(killb[:-1]):
                    lower = np.where(killb > ib)[0]
                    if len(lower) > 0 and lower[0] > 0:
                        if kill[lower[0] - 1] < kill[lower[0]] - 1:
                            kill[lower] = kill[lower] - 1
                
                # Remove killed rows from P
                keep_indices = np.setdiff1d(np.arange(len(r1b)), kill)
                P = P[keep_indices, :]
            
            # Adjust lambda to fit P*X
            valid_images = np.setdiff1d(np.arange(m), u1)
            valid_points = np.setdiff1d(np.arange(n), u2)
            lambda_vals = lambda_vals[np.ix_(valid_images, valid_points)]
        else:
            # Insufficient data
            P = np.array([])
            X = np.array([])
            u1 = np.arange(m)
            u2 = np.arange(n)
            lambda_vals = np.array([])
    
    return P, X, u1, u2, lambda_vals, info


def nullspace2L(NULLSPACE: np.ndarray, r: int, opt: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the basis of MM (L) from the null-space.
    
    Args:
        NULLSPACE: Null-space matrix
        r: Target rank
        opt: Options dict with 'verbose'
    
    Returns:
        L: Basis matrix (last r columns of U)
        S: Singular value matrix
    """
    
    if opt['verbose']:
        print('Computing the basis...', end='', flush=True)
        start_time = time.time()
    
    # Choose method based on matrix dimensions
    if NULLSPACE.shape[1] < 10 * NULLSPACE.shape[0]:
        # Use SVD directly
        U, s, Vt = np.linalg.svd(NULLSPACE, full_matrices=True)
        S = np.diag(s)
    else:
        # Use eigendecomposition (more efficient for wide matrices)
        # Compute NULLSPACE @ NULLSPACE^T
        SS_matrix = NULLSPACE @ NULLSPACE.T
        eigenvalues, U = np.linalg.eig(SS_matrix)
        
        # Sort by eigenvalues (descending)
        l = len(eigenvalues)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        U = U[:, sorted_indices]
        sorted_eigenvalues = eigenvalues[sorted_indices]
        
        # Convert eigenvalues to singular values
        sorted_eigenvalues = np.maximum(sorted_eigenvalues, 0)  # Ensure non-negative
        sorted_sv = np.sqrt(sorted_eigenvalues)
        
        # Create diagonal matrix
        S = np.diag(sorted_sv)
    
    # Take last r columns of U as basis
    lenU = U.shape[1]
    L = U[:, lenU - r:lenU]
    
    if opt['verbose']:
        elapsed = time.time() - start_time
        print(f' ({elapsed:.3f} sec)')
    
    return L, S


def svd_suff_data(S: np.ndarray, r: int, threshold: float) -> bool:
    """
    Check if SVD has sufficient data.
    
    Checks if the (n-r)-th singular value is above threshold,
    ensuring the null-space has sufficient rank.
    
    Args:
        S: Singular value matrix
        r: Target rank
        threshold: Minimum acceptable singular value
    
    Returns:
        True if sufficient data, False otherwise
    """
    
    Snumrows, Snumcols = S.shape
    
    # Check for degenerate cases
    if Snumrows == 0 or Snumcols + r < Snumrows or Snumrows <= r:
        return False
    
    # Check if (n-r)-th singular value is above threshold
    # MATLAB: S(Snumrows-r, Snumrows-r)
    # Python: S[Snumrows-r-1, Snumrows-r-1] (0-indexed)
    return S[Snumrows - r - 1, Snumrows - r - 1] > threshold


if __name__ == "__main__":
    # Test fill_prmm function
    print("=" * 60)
    print("fill_prmm - PRMM Filling Test")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Create test data
    m_images = 5
    n_points = 20
    
    # Generate low-rank measurement matrix
    r_true = 4
    P_true = np.random.randn(3 * m_images, r_true)
    X_true = np.random.randn(r_true, n_points)
    M = P_true @ X_true
    
    # Add noise
    M += np.random.randn(3 * m_images, n_points) * 0.1
    
    # Add missing data
    missing = np.random.rand(3 * m_images, n_points) < 0.15
    M[missing] = np.nan
    
    # Depth indicators
    Idepths = np.random.randint(0, 2, (m_images, n_points))
    
    # Ensure at least one depth per point
    for j in range(n_points):
        if np.sum(Idepths[:, j]) == 0:
            Idepths[0, j] = 1
    
    print(f"\nTest setup:")
    print(f"  Images: {m_images}")
    print(f"  Points: {n_points}")
    print(f"  True rank: {r_true}")
    print(f"  M shape: {M.shape}")
    print(f"  Missing entries: {np.sum(missing)}/{M.size}")
    print(f"  Known depths: {np.sum(Idepths)}/{Idepths.size}")
    
    # Options
    opt = {
        'create_nullspace': {
            'trial_coef': 0.5,
            'threshold': 0.01,
            'verbose': False
        },
        'verbose': True,
        'tol': 1e-6,
        'info_separately': True
    }
    
    # Info structure
    info = {
        'sequence': []
    }
    
    # Test 1: Sequence mode
    print("\n" + "-" * 60)
    print("TEST 1: Fill PRMM (Sequence Mode)")
    print("-" * 60)
    
    try:
        P, X, u1, u2, lambda_vals, info = fill_prmm(M, Idepths, central=0, opt=opt, info=info)
        
        print(f"\nResults:")
        if P.size > 0:
            print(f"  P shape: {P.shape}")
            print(f"  X shape: {X.shape}")
            print(f"  Unrecovered images: {len(u1)}/{m_images}")
            print(f"  Unrecovered points: {len(u2)}/{n_points}")
            if lambda_vals.size > 0:
                print(f"  Lambda shape: {lambda_vals.shape}")
        else:
            print("  No reconstruction (insufficient data)")
        
        # Check info
        if 'sequence' in info and len(info['sequence']) > 0:
            seq = info['sequence'][-1]
            print(f"\nNull-space statistics:")
            print(f"  Tried: {seq['tried']}")
            print(f"  Used: {seq['used']}")
            print(f"  Failed: {seq['failed']}")
            print(f"  Nullspace size: {seq['size_nullspace']}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Helper functions
    print("\n" + "-" * 60)
    print("TEST 2: Helper Functions")
    print("-" * 60)
    
    # Test nullspace2L
    test_nullspace = np.random.randn(50, 20)
    L_test, S_test = nullspace2L(test_nullspace, r=4, opt={'verbose': False})
    print(f"nullspace2L:")
    print(f"  Input shape: {test_nullspace.shape}")
    print(f"  L shape: {L_test.shape}")
    print(f"  S shape: {S_test.shape}")
    
    # Test svd_suff_data
    S_good = np.diag(np.array([10, 5, 3, 2, 1.5, 1.2, 0.9, 0.5, 0.2, 0.05]))
    S_bad = np.diag(np.array([10, 5, 3, 2, 0.005, 0.003, 0.001, 0.0005, 0.0002, 0.00005]))
    
    result_good = svd_suff_data(S_good, r=4, threshold=0.01)
    result_bad = svd_suff_data(S_bad, r=4, threshold=0.01)
    
    print(f"\nsvd_suff_data:")
    print(f"  Good data (sv={S_good[5,5]:.3f}): {result_good}")
    print(f"  Bad data (sv={S_bad[5,5]:.6f}): {result_bad}")
    
    print("\n" + "=" * 60)
    print("✓ All tests completed")
    print("=" * 60)
