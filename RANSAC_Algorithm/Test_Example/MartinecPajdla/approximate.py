"""
approximate - Compute r-rank approximation of measurement matrix using null-space
Converted from MATLAB to Python
"""

import numpy as np
import time
from typing import Tuple, List


def approximate(M: np.ndarray, 
               r: int, 
               P: np.ndarray, 
               opt: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute r-rank approximation of measurement matrix M.
    The approximated matrix is P*X.
    
    Args:
        M: Measurement matrix (m x n)
        r: Target rank for approximation
        P: Basis matrix (m x r)
        opt: Options dict with 'tol' and 'verbose' keys
    
    Returns:
        P: Updated basis matrix
        X: Coefficient matrix (r x n) such that M ≈ P*X
        u1: Unrecovered row indices
        u2: Unrecovered column indices
    """
    
    m, n = M.shape
    
    # Find rows with non-zero basis
    row_means = np.mean(np.abs(P.T), axis=0)
    rows = np.where(row_means > opt['tol'])[0]
    nonrows = np.setdiff1d(np.arange(m), rows)
    
    # Immerse columns into basis
    if opt['verbose']:
        print('Immersing columns of MM into the basis...', end='', flush=True)
        start = time.time()
    
    Mapp, noncols, X = approx_matrix(M[rows, :], P[rows, :], r, opt)
    
    if opt['verbose']:
        print(f' ({time.time() - start:.3f} sec)')
    
    # Get valid columns
    cols = setdiff_custom(np.arange(n), noncols)
    
    if len(cols) < r:
        # Cannot extend matrix correctly
        u1 = np.arange(m)
        u2 = np.arange(n)
    else:
        # Extend to missing rows
        if len(nonrows) == 0:
            u1 = np.array([])
            r1 = np.arange(m)
        else:
            if opt['verbose']:
                print('Filling rows of MM which have not been computed yet...')
            
            Mapp1, u1, Pnonrows = extend_matrix(
                M[:, cols], Mapp[:, cols], X[:, cols], 
                rows, nonrows, r, opt['tol']
            )
            
            Mapp = np.full((m, n), np.nan)
            Mapp[:, cols] = Mapp1
            
            if len(u1) < len(nonrows):
                recovered = np.setdiff1d(nonrows, u1)
                P[recovered, :] = Pnonrows
            
            r1 = np.union1d(rows, np.setdiff1d(nonrows, u1))
            P = P[r1, :]
        
        # Extend to missing columns
        if len(noncols) == 0:
            u2 = np.array([])
        else:
            if opt['verbose']:
                print('Filling columns of MM which have not been computed yet...')
            
            Mapp_tr, u2, Xnoncols_tr = extend_matrix(
                M[r1, :].T, Mapp[r1, :][:, cols].T, P.T,
                cols, noncols, r, opt['tol']
            )
            
            Mapp2 = Mapp_tr.T
            Xnoncols = Xnoncols_tr.T
            
            if len(u2) < len(noncols):
                recovered_cols = np.setdiff1d(noncols, u2)
                X[:, recovered_cols] = Xnoncols
            
            X = X[:, np.union1d(cols, np.setdiff1d(noncols, u2))]
            
            # Check for zero columns
            if np.any(np.sum(np.abs(X), axis=0) == 0):
                raise ValueError('nullspace problem 1')
    
    # Final check for zero columns
    if np.any(np.sum(np.abs(X), axis=0) == 0):
        raise ValueError('nullspace problem 2')
    
    u1 = np.array(u1) if isinstance(u1, list) or isinstance(u1, np.ndarray) else u1
    u2 = np.array(u2) if isinstance(u2, list) or isinstance(u2, np.ndarray) else u2
    
    return P, X, u1, u2


def approx_matrix(M: np.ndarray, 
                 P: np.ndarray, 
                 r: int, 
                 opt: dict) -> Tuple[np.ndarray, List, np.ndarray]:
    """
    Immerse columns of M into basis P.
    
    Args:
        M: Measurement matrix subset (m x n)
        P: Basis matrix (m x r)
        r: Target rank
        opt: Options dict with 'tol' key
    
    Returns:
        Mapp: Approximated matrix (m x n)
        misscols: List of columns that couldn't be approximated
        X: Coefficient matrix (r x n)
    """
    
    m, n = M.shape
    misscols = []
    
    # Allocate memory
    Mapp = np.full((m, n), np.nan)
    X = np.full((r, n), np.nan)
    
    if P.size > 0:
        # P spans r-D space for approximating M
        for j in range(n):
            # Find valid rows for this column
            valid_rows = np.where(~np.isnan(M[:, j]))[0]
            
            if len(valid_rows) > 0:
                P_sub = P[valid_rows, :]
                
                # Check if P subset has full rank
                if np.linalg.matrix_rank(P_sub, tol=opt['tol']) == r:
                    # Solve P[rows,:] * x = M[rows,j] for x
                    X[:, j] = np.linalg.lstsq(P_sub, M[valid_rows, j], rcond=None)[0]
                    Mapp[:, j] = P @ X[:, j]
                else:
                    misscols.append(j)
            else:
                misscols.append(j)
    
    return Mapp, misscols, X


def extend_matrix(M: np.ndarray, 
                 subM: np.ndarray, 
                 X: np.ndarray, 
                 rows: np.ndarray, 
                 nonrows: np.ndarray, 
                 r: int, 
                 tol: float) -> Tuple[np.ndarray, List, np.ndarray]:
    """
    Fill rows of M which have not been computed yet.
    
    Args:
        M: Full measurement matrix
        subM: Approximation for 'rows'
        X: Coefficient matrix
        rows: Rows already computed
        nonrows: Rows still to compute
        r: Target rank
        tol: Tolerance for rank and error checks
    
    Returns:
        E: Extended matrix
        unrecovered: List of rows that couldn't be recovered
        Pnonrows: Basis for recovered nonrows
    """
    
    # Allocate memory
    E = np.full(M.shape, np.nan)
    E[rows, :] = subM
    
    unrecovered = []
    row_count = 0
    Pnonrows = np.zeros((0, r))
    
    for i in nonrows:
        # Find valid columns for this row
        valid_cols = np.where(~np.isnan(M[i, :]))[0]
        
        if len(valid_cols) == 0:
            unrecovered.append(i)
            continue
        
        X_sub = X[:, valid_cols]
        
        # Check if X subset has full rank
        if np.linalg.matrix_rank(X_sub, tol=tol) == r:
            # Solve x * X[:,cols] = M[i,cols] for x (right division)
            # This is: X[:,cols].T @ x.T = M[i,cols].T
            p_i = np.linalg.lstsq(X_sub.T, M[i, valid_cols], rcond=None)[0]
            
            # Check reconstruction error
            reconstructed = p_i @ X
            max_error = np.max(np.abs(reconstructed))
            
            if max_error > 1.0 / tol:
                unrecovered.append(i)
            else:
                # Successfully recovered
                if Pnonrows.shape[0] == 0:
                    Pnonrows = p_i.reshape(1, -1)
                else:
                    Pnonrows = np.vstack([Pnonrows, p_i])
                
                E[i, :] = p_i @ X
                row_count += 1
        else:
            unrecovered.append(i)
    
    return E, unrecovered, Pnonrows


def setdiff_custom(a: np.ndarray, b: List) -> np.ndarray:
    """
    Custom set difference (maintains order unlike np.setdiff1d).
    
    Args:
        a: Array of elements
        b: List of elements to exclude
    
    Returns:
        Elements in a but not in b
    """
    result = []
    for elem in a:
        if not member(elem, b):
            result.append(elem)
    return np.array(result)


def member(e, s) -> bool:
    """
    Check if element e is in set s.
    
    Args:
        e: Element to check
        s: Set/list to check in
    
    Returns:
        True if e in s, False otherwise
    """
    if isinstance(s, (list, np.ndarray)):
        return e in s
    return False


if __name__ == "__main__":
    # Test approximate function
    print("=" * 60)
    print("approximate - Rank Approximation Test")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Create low-rank matrix with noise
    m, n, r = 20, 30, 4
    
    # True low-rank structure: M ≈ P * X
    P_true = np.random.randn(m, r)
    X_true = np.random.randn(r, n)
    M_clean = P_true @ X_true
    
    # Add some noise
    noise = np.random.randn(m, n) * 0.1
    M = M_clean + noise
    
    # Add some missing data
    missing_mask = np.random.rand(m, n) < 0.1
    M[missing_mask] = np.nan
    
    print(f"\nGenerated test data:")
    print(f"  Matrix size: {m} x {n}")
    print(f"  Target rank: {r}")
    print(f"  Missing entries: {np.sum(missing_mask)}/{m*n}")
    
    # Initial basis (use SVD on valid data)
    M_filled = M.copy()
    M_filled[np.isnan(M_filled)] = 0
    U, s, Vt = np.linalg.svd(M_filled, full_matrices=False)
    P_init = U[:, :r]
    
    # Options
    opt = {
        'tol': 1e-6,
        'verbose': True
    }
    
    # Test 1: Basic approximation
    print("\n" + "-" * 60)
    print("TEST 1: Basic Approximation")
    print("-" * 60)
    
    try:
        P_approx, X_approx, u1, u2 = approximate(M, r, P_init, opt)
        
        print(f"\nResults:")
        print(f"  P shape: {P_approx.shape}")
        print(f"  X shape: {X_approx.shape}")
        print(f"  Unrecovered rows: {len(u1)}")
        print(f"  Unrecovered cols: {len(u2)}")
        
        # Compute approximation
        M_approx = P_approx @ X_approx
        
        # Compute error on valid entries
        valid = ~np.isnan(M)
        error = np.mean((M[valid] - M_approx[valid])**2)
        print(f"  MSE on valid entries: {error:.6f}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test 2: Perfect low-rank data
    print("\n" + "-" * 60)
    print("TEST 2: Perfect Low-Rank Data (no noise)")
    print("-" * 60)
    
    M_perfect = P_true @ X_true
    opt_quiet = {'tol': 1e-6, 'verbose': False}
    
    try:
        P_approx2, X_approx2, u1_2, u2_2 = approximate(M_perfect, r, P_init, opt_quiet)
        
        M_approx2 = P_approx2 @ X_approx2
        error2 = np.mean((M_perfect - M_approx2)**2)
        
        print(f"  MSE: {error2:.2e}")
        print(f"  Rank of approximation: {np.linalg.matrix_rank(M_approx2)}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test 3: Helper functions
    print("\n" + "-" * 60)
    print("TEST 3: Helper Functions")
    print("-" * 60)
    
    # Test setdiff_custom
    a = np.array([0, 1, 2, 3, 4, 5])
    b = [1, 3, 5]
    result = setdiff_custom(a, b)
    print(f"setdiff_custom([0,1,2,3,4,5], [1,3,5]) = {result}")
    print(f"  Expected: [0, 2, 4]")
    
    # Test member
    print(f"member(3, [1,2,3,4]) = {member(3, [1,2,3,4])}")
    print(f"member(5, [1,2,3,4]) = {member(5, [1,2,3,4])}")
    
    print("\n" + "=" * 60)
    print("✓ All tests completed")
    print("=" * 60)
