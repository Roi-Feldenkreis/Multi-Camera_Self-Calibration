"""
L2depths - Compute depths of PRMM from basis L
Converted from MATLAB to Python
"""

import numpy as np
import time
from typing import Tuple
from MartinecPajdla.Utils import Utils


def L2depths(L: np.ndarray, 
            M: np.ndarray, 
            Idepths: np.ndarray, 
            opt: dict = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute depths from basis L.
    No known depths are exploited (can differ due to noise).
    
    Args:
        L: Basis matrix for the projective space
        M: Measurement matrix (3m x n)
        Idepths: Binary depth indicator matrix (m x n)
                 1 = depth known, 0 = depth unknown
        opt: Options dict with 'verbose' key (default True)
    
    Returns:
        Mdepths: Measurement matrix with scaled depths
        lambda: Depth scale factors (m x n)
    """
    
    # Default options
    if opt is None:
        opt = {}
    if 'verbose' not in opt:
        opt['verbose'] = True
    
    if opt['verbose']:
        print('Computing depths...', end='', flush=True)
        start_time = time.time()
    
    # Initialize output
    Mdepths = M.copy()
    
    # Get dimensions
    m = M.shape[0] // 3  # number of images
    n = M.shape[1]        # number of points
    
    # Allocate memory for lambda
    lambda_vals = np.zeros((m, n))
    
    # Process each point (column)
    for j in range(n):
        # Find rows with valid measurements (non-NaN)
        full = np.where(~np.isnan(M[0::3, j]))[0]
        
        # Find rows with missing depths among valid rows
        mis_rows = np.intersect1d(np.where(Idepths[:, j] == 0)[0], full)
        
        if len(mis_rows) > 0:
            # Build submatrix using depth spreading
            rowsbig = Utils.k2i(full, step=3)
            col_data = M[rowsbig, j].reshape(-1, 1)
            depth_col = Idepths[full, j].reshape(-1, 1)
            
            submatrix = Utils.spread_depths_col(col_data, depth_col)
            
            # Set up linear system
            # We want: L[rows,:] @ res[:r] = submatrix @ [1, res[r:]]
            # Rearranged: L[rows,:] @ res[:r] - submatrix[:,1:] @ res[r:] = submatrix[:,0]
            
            right = submatrix[:, 0]
            A_left = L[rowsbig, :]
            
            if submatrix.shape[1] > 1:
                A_right = -submatrix[:, 1:]
                A = np.hstack([A_left, A_right])
            else:
                A = A_left
            
            # Check if system is solvable
            try:
                rank_A = np.linalg.matrix_rank(A)
            except np.linalg.LinAlgError:
                # SVD failed - assume rank deficient
                rank_A = 0
            
            if rank_A < A.shape[1]:
                # Cannot compute depths - invalidate data
                no_depth = full[Idepths[full, j] == 0]
                kill_rows = Utils.k2i(no_depth, step=3)
                Mdepths[kill_rows, j] = np.nan
                lambda_vals[no_depth, j] = np.nan
            else:
                # Solve for depth coefficients
                res = np.linalg.lstsq(A, right, rcond=None)[0]
                
                # Apply depths to measurements
                # First: rows corresponding to first column of submatrix (depth=1)
                right_indices = np.where(right[0::3] != 0)[0]
                if len(right_indices) > 0:
                    i = full[right_indices]
                    lambda_vals[i, j] = 1.0
                    i_big = Utils.k2i(i, step=3)
                    Mdepths[i_big, j] = M[i_big, j]
                
                # Rest: rows corresponding to other columns (scaled depths)
                for ii in range(submatrix.shape[1] - 1):
                    sub_col = submatrix[0::3, ii + 1]
                    sub_indices = np.where(sub_col != 0)[0]
                    
                    if len(sub_indices) > 0:
                        i = full[sub_indices]
                        lambda_vals[i, j] = res[A_left.shape[1] + ii]
                        i_big = Utils.k2i(i, step=3)
                        Mdepths[i_big, j] = M[i_big, j] * lambda_vals[i, j]
    
    if opt['verbose']:
        elapsed = time.time() - start_time
        print(f' ({elapsed:.3f} sec)')
    
    return Mdepths, lambda_vals


if __name__ == "__main__":
    # Test L2depths function
    print("=" * 60)
    print("L2depths - Depth Computation Test")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Create test data
    m_images = 5
    n_points = 15
    r_rank = 4
    
    # Basis matrix L
    L = np.random.randn(3 * m_images, r_rank)
    
    # True coefficients
    X_true = np.random.randn(r_rank, n_points)
    
    # Generate measurements M = L @ X
    M = L @ X_true
    
    # Add some missing data
    missing = np.random.rand(3 * m_images, n_points) < 0.1
    M[missing] = np.nan
    
    # Depth indicators (m x n)
    # 1 = known depth, 0 = unknown depth
    Idepths = np.random.randint(0, 2, (m_images, n_points))
    
    # Ensure at least one known depth per point
    for j in range(n_points):
        if np.sum(Idepths[:, j]) == 0:
            Idepths[0, j] = 1
    
    print(f"\nTest data:")
    print(f"  Images: {m_images}")
    print(f"  Points: {n_points}")
    print(f"  Rank: {r_rank}")
    print(f"  M shape: {M.shape}")
    print(f"  L shape: {L.shape}")
    print(f"  Idepths shape: {Idepths.shape}")
    print(f"  Missing entries: {np.sum(missing)}")
    print(f"  Known depths: {np.sum(Idepths)}/{Idepths.size}")
    
    # Test 1: Compute depths with verbose
    print("\n" + "-" * 60)
    print("TEST 1: Compute Depths (Verbose)")
    print("-" * 60)
    
    opt = {'verbose': True}
    Mdepths, lambda_vals = L2depths(L, M, Idepths, opt)
    
    print(f"\nResults:")
    print(f"  Mdepths shape: {Mdepths.shape}")
    print(f"  Lambda shape: {lambda_vals.shape}")
    print(f"  Non-NaN in Mdepths: {np.sum(~np.isnan(Mdepths))}")
    print(f"  Non-zero lambdas: {np.sum(lambda_vals != 0)}")
    
    # Check some depths
    print(f"\nSample lambda values (first 5 points, first 3 images):")
    for i in range(min(3, m_images)):
        print(f"  Image {i}: {lambda_vals[i, :5]}")
    
    # Test 2: Quiet mode
    print("\n" + "-" * 60)
    print("TEST 2: Compute Depths (Quiet)")
    print("-" * 60)
    
    opt_quiet = {'verbose': False}
    Mdepths2, lambda_vals2 = L2depths(L, M, Idepths, opt_quiet)
    
    print(f"Computed without verbose output")
    print(f"  Results match: {np.allclose(lambda_vals, lambda_vals2, equal_nan=True)}")
    
    # Test 3: Verify depth scaling
    print("\n" + "-" * 60)
    print("TEST 3: Verify Depth Scaling")
    print("-" * 60)
    
    # Check that scaled measurements match
    for j in range(min(3, n_points)):
        for i in range(m_images):
            if not np.isnan(M[3*i, j]) and lambda_vals[i, j] != 0:
                expected = M[3*i:3*i+3, j] * lambda_vals[i, j]
                actual = Mdepths[3*i:3*i+3, j]
                
                if not np.isnan(actual[0]):
                    match = np.allclose(expected, actual, rtol=1e-5)
                    if not match:
                        print(f"  Point {j}, Image {i}: Mismatch!")
                    else:
                        print(f"  Point {j}, Image {i}: ✓")
                    break
    
    # Test 4: All known depths
    print("\n" + "-" * 60)
    print("TEST 4: All Known Depths")
    print("-" * 60)
    
    Idepths_all = np.ones((m_images, n_points), dtype=int)
    Mdepths_all, lambda_all = L2depths(L, M, Idepths_all, opt_quiet)
    
    print(f"With all depths known:")
    print(f"  Non-zero lambdas: {np.sum(lambda_all != 0)}")
    print(f"  All lambdas = 1: {np.allclose(lambda_all[lambda_all != 0], 1.0)}")
    
    # Test 5: All unknown depths
    print("\n" + "-" * 60)
    print("TEST 5: Some Unknown Depths")
    print("-" * 60)
    
    Idepths_none = np.zeros((m_images, n_points), dtype=int)
    # Set first row known to avoid all unknown
    Idepths_none[0, :] = 1
    
    Mdepths_none, lambda_none = L2depths(L, M, Idepths_none, opt_quiet)
    
    print(f"With mostly unknown depths:")
    print(f"  Non-zero lambdas: {np.sum(lambda_none != 0)}")
    print(f"  NaN lambdas: {np.sum(np.isnan(lambda_none))}")
    
    print("\n" + "=" * 60)
    print("✓ All tests completed")
    print("=" * 60)
