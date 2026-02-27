"""
eval_y_and_dy - Evaluate function value and Jacobian for bundle adjustment
Converted from MATLAB to Python
"""

import numpy as np
from scipy.sparse import csr_matrix
from typing import Tuple, List, Optional
from MartinecPajdla.Utils import Utils


def eval_y_and_dy(p: np.ndarray,
                 P0: np.ndarray,
                 TP: List[np.ndarray],
                 X0: np.ndarray,
                 TX: List[np.ndarray],
                 y: np.ndarray,
                 qivis: np.ndarray,
                 RADIAL: bool = False) -> Tuple[np.ndarray, csr_matrix]:
    """
    Evaluate function value and Jacobian for bundle adjustment.

    Imaging model:
        u = p2e(x)  (homogeneous to Euclidean)
        x = P @ X   (projection)
        P = P0 + TP @ iP  (camera matrix)
        X = X0 + TX @ iX  (3D point)

    Args:
        p: Parameter vector
        P0: Base camera matrices (3K x 4)
        TP: List of K transformation matrices (12 x 11 each)
        X0: Base 3D points (4 x N)
        TX: List of N transformation matrices (4 x 3 each)
        y: Observations to match
        qivis: Visibility matrix (K x N), 1=visible 0=not visible
        RADIAL: Include radial distortion (default False)

    Returns:
        y_residual: Residual vector (predicted - observed)
        J: Jacobian matrix (sparse)
    """

    K, N = qivis.shape
    NR = 3 if RADIAL else 0

    # ---------------------------------------------------------------
    # Rearrange p -> P, X
    # ---------------------------------------------------------------
    P = np.zeros((3 * K, 4))
    radial = []

    for k in range(K):
        iP_k = p[k * (11 + NR): k * (11 + NR) + 11]
        # CRITICAL: MATLAB reshape is column-major (Fortran order)
        P_k_flat = TP[k] @ iP_k
        P[3*k:3*k+3, :] = P0[3*k:3*k+3, :] + P_k_flat.reshape(3, 4, order='F')

        if RADIAL:
            radial.append({
                'u0':    p[k * (11 + NR) + 11: k * (11 + NR) + 13],
                'kappa': p[k * (11 + NR) + 13]
            })

    X = np.zeros((4, N))
    for n in range(N):
        iX_n = p[(11 + NR) * K + n * 3: (11 + NR) * K + (n + 1) * 3]
        X[:, n] = TX[n] @ iX_n
    X = X0 + X

    # ---------------------------------------------------------------
    # Compute projected points x and retina points q
    # ---------------------------------------------------------------
    x = np.zeros((3 * K, N))
    q = np.zeros((2 * K, N))

    for k in range(K):
        x[3*k:3*k+3, :] = P[3*k:3*k+3, :] @ X
        q[2*k:2*k+2, :] = Utils.p2e(x[3*k:3*k+3, :])
        if RADIAL:
            q[2*k:2*k+2, :] = Utils.raddist_apply(
                q[2*k:2*k+2, :], radial[k]['u0'], radial[k]['kappa'])

    # Extract visible observations and compute residual
    # MATLAB: qq = reshape(q,[2 K*N]); qq = qq(:,qivis);
    # Both the reshape and the logical indexing are column-major
    qq = q.reshape(2, K * N, order='F')
    vis_mask = qivis.flatten(order='F').astype(bool)   # column-major flatten
    qq_vis = qq[:, vis_mask]
    y_pred = qq_vis.flatten('F')
    y_residual = y_pred - y

    # ---------------------------------------------------------------
    # Compute Jacobian  J = dF(p)/dp
    # MATLAB only computes J when nargout >= 2. In Python we always
    # need it (levmarq always uses it), so always compute it.
    # ---------------------------------------------------------------
    kvis, nvis = np.where(qivis)   # indices of visible points

    n_visible = len(kvis)
    n_params  = len(p)
    n_obs     = len(y_residual)

    # Pre-allocate COO arrays (each visible point contributes 2 rows x (11+NR+3) cols)
    entries_per_pt = 2 * (11 + NR + 3)
    Ji = np.zeros(n_visible * entries_per_pt, dtype=int)
    Jj = np.zeros(n_visible * entries_per_pt, dtype=int)
    Jv = np.zeros(n_visible * entries_per_pt)

    cnt = 0
    for l in range(n_visible):
        k = kvis[l]
        n = nvis[l]

        xl = x[3*k:3*k+3, n]          # (3,)
        ul = Utils.p2e(xl.reshape(3, 1)).flatten()  # (2,)

        # dx/d(iP)  shape (3, 11)
        # MATLAB: kron(X(:,n)', eye(3)) * TP{k}
        dxdP = np.kron(X[:, n].reshape(1, -1), np.eye(3)) @ TP[k]

        # dx/d(iX)  shape (3, 3)
        dxdX = P[3*k:3*k+3, :] @ TX[n]

        # du/dx  shape (2, 3)
        dudx = np.hstack([np.eye(2), -ul.reshape(2, 1)]) / xl[2]

        if RADIAL:
            dqdu, dqdu0, dqdkappa = Utils.raddist_deriv(
                ul, radial[k]['u0'], radial[k]['kappa'])
        else:
            dqdu    = np.eye(2)
            dqdu0   = np.zeros((2, 0))
            dqdkappa = np.zeros((2, 0))

        # dq/d(iP)  (2, 11),  dq/d(iX)  (2, 3)
        dqdP = dqdu @ dudx @ dxdP
        dqdX = dqdu @ dudx @ dxdX

        # Row indices for this visible point (0-based)
        row_base = l * 2
        rows_2 = np.array([row_base, row_base + 1])

        # Column index ranges
        col_P = np.arange(11)          + k * (11 + NR)
        col_X = np.arange(3)           + (11 + NR) * K + n * 3

        if RADIAL:
            col_u0    = np.array([11, 12]) + k * (11 + NR)
            col_kappa = np.array([13])     + k * (11 + NR)
            cols = np.concatenate([col_P, col_u0, col_kappa, col_X])
            vals = np.hstack([dqdP, dqdu0, dqdkappa, dqdX])  # (2, 11+NR+3)
        else:
            cols = np.concatenate([col_P, col_X])
            vals = np.hstack([dqdP, dqdX])   # (2, 14)

        # Expand rows x cols into COO triplets
        n_cols_blk = len(cols)
        # rows: [row_base, row_base] * n_cols_blk, then [row_base+1, ...] * n_cols_blk
        for ri, r in enumerate(rows_2):
            s = cnt + ri * n_cols_blk
            e = s + n_cols_blk
            Ji[s:e] = r
            Jj[s:e] = cols
            Jv[s:e] = vals[ri, :]

        cnt += entries_per_pt

    J = csr_matrix((Jv, (Ji, Jj)), shape=(n_obs, n_params))

    return y_residual, J
