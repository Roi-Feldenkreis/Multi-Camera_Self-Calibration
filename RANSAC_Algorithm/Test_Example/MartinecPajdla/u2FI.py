"""
u2FI - Estimate fundamental matrix using orthogonal LS regression
Converted from MATLAB to Python
"""

import numpy as np
from typing import Union
from MartinecPajdla.Utils import Utils


def u2FI(u: np.ndarray, 
         normalization: str = 'norm', 
         A1: np.ndarray = None, 
         A2: np.ndarray = None) -> Union[np.ndarray, int]:
    """
    Estimate fundamental matrix using orthogonal LS regression.
    
    Args:
        u: (6, N) array of point correspondences [x1, y1, 1, x2, y2, 1]
        normalization: 'norm' (apply), 'nonorm' (skip), 'usenorm' (use provided A1, A2)
        A1: Normalization matrix for first image (optional)
        A2: Normalization matrix for second image (optional)
    
    Returns:
        F: (3, 3) fundamental matrix, or 0 if failed
    """
    
    # Find valid point correspondences (non-NaN columns with 2 points)
    valid_mask = np.sum(~np.isnan(u[0::3, :]), axis=0) == 2
    sampcols = np.where(valid_mask)[0]
    
    # Check minimum points requirement
    if len(sampcols) < 8:
        print("Error: Need at least 8 point correspondences")
        return 0
    
    # Determine if normalization should be applied
    donorm = False
    if normalization not in ['nonorm', 'usenorm']:
        donorm = True
    elif A1 is not None and A2 is not None:
        donorm = False
    
    ptNum = len(sampcols)
    
    # Normalize or extract points
    if donorm:
        A1 = Utils.normu(u[0:3, sampcols])
        A2 = Utils.normu(u[3:6, sampcols])
        
        # Check for degenerate normalization
        if A1.size == 0 or A2.size == 0:
            print("Error: Normalization failed")
            return 0
        
        u1 = A1 @ u[0:3, sampcols]
        u2 = A2 @ u[3:6, sampcols]
    else:
        u1 = u[0:3, sampcols]
        u2 = u[3:6, sampcols]
    
    # Build constraint matrix Z
    Z = np.zeros((ptNum, 9))
    for i in range(ptNum):
        Z[i, :] = np.outer(u1[:, i], u2[:, i]).flatten()
    
    # Solve using eigenvalue decomposition
    M = Z.T @ Z
    V, d = seig(M)
    F = V[:, 0].reshape(3, 3)
    
    # Enforce rank-2 constraint
    uu, us, vt = np.linalg.svd(F)
    us[2] = 0
    F = uu @ np.diag(us) @ vt
    
    # Denormalize if needed
    if donorm or normalization == 'usenorm':
        F = A1.T @ F @ A2
    
    # Normalize by Frobenius norm
    F = F / np.linalg.norm(F, 'fro')
    
    # Final rank check
    if np.linalg.matrix_rank(F, tol=1e-6) > 2:
        uu, us, vt = np.linalg.svd(F)
        us[2] = 0
        F = uu @ np.diag(us) @ vt
    
    return F


def seig(M: np.ndarray) -> tuple:
    """
    Compute sorted eigenvalues and eigenvectors.
    
    Args:
        M: Square matrix
    
    Returns:
        V: Eigenvectors sorted by eigenvalue (ascending)
        d: Sorted eigenvalues (ascending)
    """
    eigenvalues, eigenvectors = np.linalg.eig(M)
    sorted_indices = np.argsort(eigenvalues)
    d = eigenvalues[sorted_indices]
    V = eigenvectors[:, sorted_indices]
    return V, d


if __name__ == "__main__":
    # Test with synthetic data
    np.random.seed(42)
    N = 12
    
    # Generate synthetic point correspondences
    X = np.random.randn(3, N) * 100
    X[2, :] += 500
    
    K = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])
    R = np.array([[0.995, -0.0998, 0], [0.0998, 0.995, 0], [0, 0, 1]])
    t = np.array([[50], [10], [0]])
    
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t])
    
    X_homo = np.vstack([X, np.ones((1, N))])
    u1 = P1 @ X_homo
    u1 = u1 / u1[2, :]
    u2 = P2 @ X_homo
    u2 = u2 / u2[2, :]
    
    u = np.vstack([u1, u2])
    
    # Estimate F
    F = u2FI(u)
    
    if not isinstance(F, int):
        print("✓ Fundamental matrix estimated successfully!")
        print("\nF =")
        print(F)
        print(f"\nRank: {np.linalg.matrix_rank(F, tol=1e-6)}")
        print(f"Norm: {np.linalg.norm(F, 'fro'):.6f}")
        
        # Check epipolar constraint
        residuals = []
        for i in range(N):
            res = u2[:, i].T @ F @ u1[:, i]
            residuals.append(abs(res))
        print(f"\nMean epipolar residual: {np.mean(residuals):.2e}")
    else:
        print("✗ Failed to estimate F")
