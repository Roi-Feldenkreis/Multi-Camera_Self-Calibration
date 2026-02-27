"""
qPXbundle_cmp - Bundle adjustment with Levenberg-Marquardt optimization
Converted from MATLAB to Python
"""

import numpy as np
from scipy.sparse import eye as speye, csr_matrix
from scipy.sparse.linalg import spsolve
from typing import Tuple, List, Dict, Optional
from MartinecPajdla. Utils import Utils


def qPXbundle_cmp(P0: np.ndarray,
                 X0: np.ndarray,
                 q: np.ndarray,
                 radial: Optional[List[Dict]] = None) -> Tuple[np.ndarray, np.ndarray, Optional[List[Dict]]]:
    """
    Bundle adjustment using Levenberg-Marquardt optimization.
    
    Args:
        P0: Initial camera matrices (3K x 4), K = number of images
        X0: Initial 3D points (4 x N), N = number of points
        q: Measured image points (2K x N) in Euclidean coordinates
        radial: Optional radial distortion parameters (list of dicts)
    
    Returns:
        P: Optimized camera matrices (3K x 4)
        X: Optimized 3D points (4 x N)
        radial: Optimized radial distortion parameters (if provided)
    """
    
    RADIAL = (radial is not None)
    K = q.shape[0] // 2  # Number of cameras
    N = q.shape[1]        # Number of points
    
    # Normalize initial estimates
    P0 = Utils.normP(P0)
    X0 = Utils.normx(X0)
    
    # Form observation vector y
    qq = q.reshape(2, K * N, order='F')
    qivis = np.all(~np.isnan(qq), axis=0).reshape(K, N, order='F')
    qq_vis = qq[:, qivis.T.flatten()]
    y = qq_vis.flatten('F')
    
    # Setup auxiliary structure
    aux = {
        'P0': P0,
        'X0': X0,
        'RADIAL': RADIAL,
        'qivis': qivis,
        'TP': [],
        'TX': []
    }
    
    # Choose transformation matrices TP, TX
    # These describe local coordinate systems tangent to P0, X0
    for n in range(N):
        Q, R = np.linalg.qr(X0[:, n].reshape(-1, 1), mode='complete')  # Get full (4, 4) Q matrix
        aux['TX'].append(Q[1:, :].T)  # (3, 4).T = (4, 3) matrix
    
    for k in range(K):
        P_k = P0[3*k:3*k+3, :].flatten('F').reshape(-1, 1)  # (12, 1) column vector
        Q, R = np.linalg.qr(P_k, mode='complete')  # Get full (12, 12) Q matrix
        # Q is (12, 12), take rows 1-11 (skip first row), transpose to get (12, 11)
        aux['TP'].append(Q[1:, :].T)  # (11, 12).T = (12, 11)
    
    # Form initial parameter vector p0
    p0 = []
    NR = 3 if RADIAL else 0
    
    for k in range(K):
        p0.append(np.zeros(11))
        if RADIAL:
            p0.append(radial[k]['u0'])
            p0.append(np.array([radial[k]['kappa']]))
    
    p0.append(np.zeros(3 * N))
    p0 = np.concatenate([x.flatten() for x in p0])
    
    # Levenberg-Marquardt optimization
    p = p0.copy()
    lastFp = np.inf
    stepy = np.inf
    lam = 0.001
    fail_cnt = 0
    
    # Initial evaluation
    Fp, J = F(p, aux)
    residual = y - Fp
    rms = np.sqrt(np.mean(residual ** 2))
    max_res = np.max(np.abs(residual))
    print(f'  res (rms/max):  {rms:.10g} / {max_res:.10g}')
    
    # Optimization loop
    while stepy > 100 * np.finfo(float).eps and fail_cnt < 20:
        # Solve normal equations with damping
        JtJ = J.T @ J
        Jtr = J.T @ residual
        
        # Add damping
        damped = JtJ + lam * speye(JtJ.shape[0], format='csr')
        
        # Solve for step
        D = spsolve(damped, Jtr)
        
        # Evaluate at new point
        FpD, _ = F(p + D, aux)
        residual_new = y - FpD
        
        # Check if step improves objective
        if np.sum(residual ** 2) > np.sum(residual_new ** 2):
            # Accept step
            p = p + D
            lam = max(lam / 10, 1e-9)
            stepy = np.max(np.abs(Fp - lastFp))
            lastFp = Fp
            Fp, J = F(p, aux)
            residual = y - Fp
            fail_cnt = 0
            
            rms = np.sqrt(np.mean(residual ** 2))
            max_res = np.max(np.abs(residual))
            print(f'  res (rms/max), max_res_step, lam:  {rms:.10g} / {max_res:.10g}   {stepy:.10g}   {lam:g}')
        else:
            # Reject step, increase damping
            lam = min(lam * 10, 1e5)
            fail_cnt += 1
        
        # Additional stopping criterion
        stepy_threshold = 0.00005
        if stepy < stepy_threshold:
            print(f'!!! ended by condition stepy < {stepy_threshold}')
            break
    
    # Extract optimized parameters
    P = np.zeros_like(P0)
    X = np.zeros_like(X0)
    radial_out = [] if RADIAL else None
    
    for k in range(K):
        iP_k = p[k * (11 + NR):(k * (11 + NR)) + 11]
        P_k_flat = aux['TP'][k] @ iP_k
        P[3*k:3*k+3, :] = P0[3*k:3*k+3, :] + P_k_flat.reshape(3, 4, order='F')
        
        if RADIAL:
            radial_k = {
                'u0': p[(k * (11 + NR)) + 11:(k * (11 + NR)) + 13],
                'kappa': p[(k * (11 + NR)) + 13]
            }
            radial_out.append(radial_k)
    
    for n in range(N):
        iX_n = p[(11 + NR) * K + n * 3:(11 + NR) * K + (n + 1) * 3]
        X[:, n] = X0[:, n] + aux['TX'][n] @ iX_n
    
    return P, X, radial_out


def F(p: np.ndarray, aux: Dict) -> Tuple[np.ndarray, csr_matrix]:
    """
    Objective function: computes residuals and Jacobian.
    
    Similar to eval_y_and_dy but uses internal aux structure.
    """
    
    K, N = aux['qivis'].shape
    NR = 3 if aux['RADIAL'] else 0
    
    # Extract parameters
    P = np.zeros((3 * K, 4))
    X = np.zeros((4, N))
    radial = []
    
    for k in range(K):
        iP_k = p[k * (11 + NR):(k * (11 + NR)) + 11]
        P_k_flat = aux['TP'][k] @ iP_k
        P[3*k:3*k+3, :] = aux['P0'][3*k:3*k+3, :] + P_k_flat.reshape(3, 4, order='F')
        
        if aux['RADIAL']:
            radial_k = {
                'u0': p[(k * (11 + NR)) + 11:(k * (11 + NR)) + 13],
                'kappa': p[(k * (11 + NR)) + 13]
            }
            radial.append(radial_k)
    
    for n in range(N):
        iX_n = p[(11 + NR) * K + n * 3:(11 + NR) * K + (n + 1) * 3]
        X[:, n] = aux['TX'][n] @ iX_n
    X = aux['X0'] + X
    
    # Compute projected points
    x = np.zeros((3 * K, N))
    q = np.zeros((2 * K, N))
    
    for k in range(K):
        x[3*k:3*k+3, :] = P[3*k:3*k+3, :] @ X
        q[2*k:2*k+2, :] = Utils.p2e(x[3*k:3*k+3, :])
        
        if aux['RADIAL']:
            q[2*k:2*k+2, :] = Utils.raddist_apply(
                q[2*k:2*k+2, :],
                radial[k]['u0'],
                radial[k]['kappa']
            )
    
    # Extract visible points
    qq = q.reshape(2, K * N, order='F')
    qq_vis = qq[:, aux['qivis'].T.flatten()]
    y = qq_vis.flatten('F')
    
    # Compute Jacobian
    kvis, nvis = np.where(aux['qivis'])
    n_visible = len(kvis)
    
    Ji = []
    Jj = []
    Jv = []
    
    for l in range(n_visible):
        k = kvis[l]
        n = nvis[l]
        
        xl = x[3*k:3*k+3, n]
        ul = Utils.p2e(xl.reshape(3, 1)).flatten()
        
        # Compute derivatives
        dxdP = np.kron(X[:, n].reshape(1, -1), np.eye(3)) @ aux['TP'][k]
        dxdX = P[3*k:3*k+3, :] @ aux['TX'][n]
        dudx = np.hstack([np.eye(2), -ul.reshape(2, 1)]) / xl[2]
        
        if aux['RADIAL']:
            dqdu, dqdu0, dqdkappa = Utils.raddist_deriv(
                ul, radial[k]['u0'], radial[k]['kappa']
            )
        else:
            dqdu = np.eye(2)
            dqdu0 = np.zeros((2, 2))
            dqdkappa = np.zeros((2, 1))
        
        dqdP = dqdu @ dudx @ dxdP
        dqdX = dqdu @ dudx @ dxdX
        
        # Build sparse indices
        row_idx = np.arange(2) + l * 2
        col_idx_P = np.arange(11) + k * (11 + NR)
        col_idx_X = (11 + NR) * K + np.arange(3) + n * 3
        
        if aux['RADIAL']:
            col_idx_u0 = np.array([11, 12]) + k * (11 + NR)
            col_idx_kappa = np.array([13]) + k * (11 + NR)
            col_idx = np.concatenate([col_idx_P, col_idx_u0, col_idx_kappa, col_idx_X])
            values = np.hstack([dqdP, dqdu0, dqdkappa, dqdX])
        else:
            col_idx = np.concatenate([col_idx_P, col_idx_X])
            values = np.hstack([dqdP, dqdX])
        
        for i in row_idx:
            for j_idx, j_global in enumerate(col_idx):
                Ji.append(i)
                Jj.append(j_global)
                Jv.append(values[i - row_idx[0], j_idx])
    
    J = csr_matrix((Jv, (Ji, Jj)), shape=(len(y), len(p)))
    
    return y, J


if __name__ == "__main__":
    # Test qPXbundle_cmp
    print("=" * 60)
    print("qPXbundle_cmp - Bundle Adjustment Test")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Setup
    K = 3  # cameras
    N = 10  # points
    
    # Generate ground truth
    P_true = np.zeros((3 * K, 4))
    for k in range(K):
        # Simple rotation + translation
        theta = k * 0.1
        R = np.array([[np.cos(theta), -np.sin(theta), 0],
                      [np.sin(theta), np.cos(theta), 0],
                      [0, 0, 1]])
        t = np.array([[k * 2], [0], [0]])
        P_true[3*k:3*k+3, :] = np.hstack([R, t])
    
    X_true = np.vstack([
        np.random.randn(3, N) * 5,
        np.ones((1, N))
    ])
    
    # Project to get observations
    q = np.zeros((2 * K, N))
    for k in range(K):
        x = P_true[3*k:3*k+3, :] @ X_true
        q[2*k:2*k+2, :] = Utils.p2e(x)
    
    # Add noise
    q += np.random.randn(2 * K, N) * 0.5
    
    # Add some missing data
    q[:, [1, 5, 8]] = np.nan
    
    # Perturb initial guess
    P0 = P_true + np.random.randn(3 * K, 4) * 0.1
    X0 = X_true + np.random.randn(4, N) * 0.1
    
    print(f"\nProblem size:")
    print(f"  Cameras: {K}")
    print(f"  Points: {N}")
    print(f"  Observations: {2*K*N}")
    print(f"  Missing: {np.sum(np.isnan(q))}")
    
    # Test without radial
    print("\n" + "-" * 60)
    print("TEST 1: Bundle Adjustment (No Radial)")
    print("-" * 60)
    
    P_opt, X_opt, _ = qPXbundle_cmp(P0, X0, q)
    
    print(f"\nOptimization complete!")
    print(f"P shape: {P_opt.shape}")
    print(f"X shape: {X_opt.shape}")
    
    # Compute final error
    q_final = np.zeros((2 * K, N))
    for k in range(K):
        x = P_opt[3*k:3*k+3, :] @ X_opt
        q_final[2*k:2*k+2, :] = Utils.p2e(x)
    
    valid = ~np.isnan(q)
    error = np.sqrt(np.mean((q[valid] - q_final[valid]) ** 2))
    print(f"Final RMSE: {error:.6f}")
    
    print("\n" + "=" * 60)
    print("✓ Test completed")
    print("=" * 60)
