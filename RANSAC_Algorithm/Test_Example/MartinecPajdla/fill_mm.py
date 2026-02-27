"""
fill_mm - Projective reconstruction from measurement matrix

Main high-level reconstruction function
"""

import numpy as np
import time
from typing import Tuple, Dict, List, Optional, Any
from MartinecPajdla.Utils import Utils
from MartinecPajdla.fill_mm_sub import fill_mm_sub
from MartinecPajdla.balance_triplets import balance_triplets


def fill_mm(M: np.ndarray, opt: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Projective reconstruction from measurement matrix.
    
    Uses strategy selection (sequence or central image) and iteratively
    recovers structure and motion.
    
    Args:
        M: Measurement matrix (3m x n) with NaNs for unknown elements
        opt: Options dict with:
            - 'strategy' (-1): -1 = choose best, 0 = sequence, 
                              -2 = all central, k > 0 = central image k
            - 'create_nullspace': Options for null-space creation
              - 'trial_coef' (1.0): Coefficient for trials
              - 'threshold' (0.01): Threshold for validation
            - 'tol' (1e-6): Tolerance for approximation
            - 'no_factorization' (False): Skip final factorization
            - 'metric' (1): 1 = sqrt(sum of squares), 2 = std of coordinates
            - 'verbose' (True): Print progress
    
    Returns:
        P: Camera matrices (3k x 4) for k recovered images
        X: 3D points (4 x n') for n' recovered points
        u1: Unrecovered image indices
        u2: Unrecovered point indices
        info: Information dict with statistics
    """
    
    # Set default options
    if 'strategy' not in opt:
        opt['strategy'] = -1
    if 'create_nullspace' not in opt:
        opt['create_nullspace'] = {}
    if 'trial_coef' not in opt['create_nullspace']:
        opt['create_nullspace']['trial_coef'] = 1.0
    if 'threshold' not in opt['create_nullspace']:
        opt['create_nullspace']['threshold'] = 0.01
    if 'tol' not in opt:
        opt['tol'] = 1e-6
    if 'no_factorization' not in opt:
        opt['no_factorization'] = False
    if 'metric' not in opt:
        opt['metric'] = 1
    if 'verbose' not in opt:
        opt['verbose'] = True
    if 'verbose' not in opt['create_nullspace']:
        opt['create_nullspace']['verbose'] = opt['verbose']
    
    # Progress message
    if not opt['verbose']:
        msg = 'Repr. error in proj. space (no fact.'
        if not opt['no_factorization']:
            msg += '/fact.'
        if 'no_BA' in opt and not opt.get('no_BA', True):
            msg += '/BA'
        msg += ') is ... '
        print(msg, end='', flush=True)
    
    # Remove points visible in < 2 images
    nbeg = M.shape[1]
    visible_count = np.sum(~np.isnan(M[0::3, :]), axis=0)
    ptbeg = np.where(visible_count >= 2)[0]
    u2beg = np.setdiff1d(np.arange(nbeg), ptbeg)
    M = M[:, ptbeg]
    
    if len(u2beg) > 0:
        u2_display = u2beg[:min(20, len(u2beg))]
        u2_str = ' '.join(map(str, u2_display))
        print(f'Removed correspondences in < 2 images ({len(u2beg)}): {u2_str}')
    
    # Get dimensions
    m = M.shape[0] // 3  # Number of images
    n = M.shape[1]        # Number of points
    M0 = M.copy()
    
    # Initialize
    cols = np.array([], dtype=int)
    recoverable = np.inf
    Omega = []
    
    # Setup strategies
    if opt['strategy'] == -1:  # Use all strategies
        Omega.append({'name': 'seq'})
        for i in range(m):
            Omega.append({'name': 'cent', 'ind': i})
    elif opt['strategy'] == 0:  # Sequence only
        Omega.append({'name': 'seq'})
    elif opt['strategy'] == -2:  # All central images
        for i in range(m):
            Omega.append({'name': 'cent', 'ind': i})
    else:  # Specific central image
        Omega.append({'name': 'cent', 'ind': opt['strategy']})
    
    # Iterative recovery
    added = 1
    I = ~np.isnan(M[0::3, :])
    info = {'sequence': [], 'err': {}}
    iteration = 0
    
    while recoverable > 0 and added:
        if opt['verbose']:
            print(f'{len(cols)} (from {n}) recovered columns...')
        
        added = 0
        iteration += 1
        if iteration > 10:
            raise RuntimeError('Too many iterations in fill_mm (>10)')
        
        # Compute predictions for all strategies
        S, F, strengths = compute_predictions(Omega, I)
        
        # Try the best strategy(s)
        while (not added and np.max(F) > 0) or np.sum(F == 0) == len(F):
            
            # Choose best strategy
            Omega_F = np.where(F == np.max(F))[0]
            if len(Omega_F) == 1:
                sg = Omega_F[0]
            else:
                if opt['verbose']:
                    omega_str = ' '.join(map(str, Omega_F))
                    print(f'Omega_F: {omega_str}.')
                
                ns = S[Omega_F]
                Omega_S = Omega_F[np.where(ns == np.max(ns))[0]]
                
                if len(Omega_S) > 1 and opt['verbose']:
                    omega_str = ' '.join(map(str, Omega_S))
                    print(f'Omega_S: {omega_str}.')
                    print('!!! Maybe some other criteria is needed to choose'
                          ' the better candidate (now the 1st is taken)! !!!')
                
                sg = Omega_S[0]
            
            # Set rows and columns for this strategy
            rows, cols, central, info = set_rows_cols(
                Omega, sg, F, S, strengths, I, info, opt
            )
            
            F[sg] = -1  # Don't try this strategy anymore
            
            if opt['verbose']:
                rows_str = ' '.join(map(str, rows))
                print(f'Used images: {rows_str}.')
            
            # Normalize measurement matrix
            Mn, T = normM(M)
            
            # Find central image in rows
            if central == 0:
                central_idx = 0
            else:
                central_idx_arr = np.where(central == rows)[0]
                central_idx = central_idx_arr[0] if len(central_idx_arr) > 0 else 0
            
            # Reconstruct sub-scene
            Pn, X, lambda_vals, u1, u2, info = fill_mm_sub(
                Mn[Utils.k2i(rows, step=3), :],
                Mn[Utils.k2i(rows, step=3), :][:, cols],
                central_idx,
                opt,
                info
            )
            
            # Check if anything was recovered
            if len(u1) == len(rows) and len(u2) == len(cols):
                if opt['verbose']:
                    print('Nothing recovered.')
            else:
                if opt['verbose']:
                    if len(u1) > 0:
                        u1_str = ' '.join(map(str, u1))
                        print(f'u1 = {u1_str}')
                    if len(u2) > 0 and Pn.size > 0:
                        u2_str = ' '.join(map(str, u2))
                        print(f'u2 = {u2_str}')
                
                # Convert indices to whole M space
                r1 = rows[np.setdiff1d(np.arange(len(rows)), u1)]
                u1 = np.setdiff1d(np.arange(m), r1)
                r2 = cols[np.setdiff1d(np.arange(len(cols)), u2)]
                u2 = np.setdiff1d(np.arange(n), r2)
                
                # Denormalize
                P = normMback(Pn, T[r1, :, :])
                R = P @ X
                
                # Fill holes: use original data when known, R elsewhere
                if len(r1) > 0 and len(r2) > 0:
                    # Build (len(r1) x len(r2)) NaN-check matrix using the
                    # x-coordinate row of each triplet (sufficient for NaN test)
                    M_check = np.isnan(M[np.ix_(3*r1, r2)])  # shape (len(r1), len(r2))

                    if lambda_vals.size > 0 and lambda_vals.shape == M_check.shape:
                        lambda_nan = np.isnan(lambda_vals)
                    else:
                        lambda_nan = np.ones(M_check.shape, dtype=bool)

                    # Condition: M is NaN  OR  M is present but lambda is NaN
                    condition = M_check | (~M_check & lambda_nan)

                    # Get 2D indices — DO NOT flatten first (avoids row-vs-column-major bug)
                    fill_img_idx, fill_pt_idx = np.where(condition)
                    added = len(fill_img_idx)

                    if added > 0:
                        # Expand each (local_image, local_point) → 3 global rows + 1 global col
                        # and copy from R (local) into M (global)
                        offsets = np.array([0, 1, 2])
                        for fi, fp in zip(fill_img_idx, fill_pt_idx):
                            global_rows = 3 * r1[fi] + offsets   # global triplet rows in M
                            global_col  = r2[fp]                  # global column in M
                            local_rows  = 3 * fi + offsets        # local triplet rows in R
                            M[global_rows, global_col] = R[local_rows, fp]
                else:
                    added = 0
                
                # Update validity
                I = ~np.isnan(M[0::3, :])
                I_cols = I[:, np.where(np.sum(I, axis=0) >= 2)[0]]
                recoverable = np.sum(I_cols == 0)
                
                if opt['verbose']:
                    print(f'{added} points added, {recoverable} recoverable points remain.')
                
                # Compute error
                if len(r1) > 0 and len(r2) > 0:
                    info['err']['no_fact'] = Utils.dist(
                        M0[Utils.k2i(r1, step=3), :][:, r2],
                        R,
                        opt['metric']
                    )
                    
                    if opt['verbose']:
                        print(f"Error (no factorization): {info['err']['no_fact']:.6f}")
                    else:
                        print(f" {info['err']['no_fact']:.6f}", end='', flush=True)
    
    # Check if recovery failed
    if not added and np.sum(~I) > 0:
        print('WARNING: impossible to recover anything in fill_mm')
        P = np.array([])
        X = np.array([])
        u1 = np.arange(m)
        u2 = np.arange(nbeg)
        info['opt'] = opt
        return P, X, u1, u2, info
    
    # Check if rank can be 4
    if len(r1) < 2:
        P = np.array([])
        X = np.array([])
        u1 = np.arange(m)
        u2 = np.arange(nbeg)
        info['opt'] = opt
        return P, X, u1, u2, info
    
    # Final factorization
    if not opt['no_factorization']:
        if opt['verbose']:
            print('Factorization into structure and motion...', end='', flush=True)
            start_time = time.time()
        
        # Compute depths from Mdepths
        Mdepths_un = normMback(info['Mdepths'], T[r1, :, :])
        lambda_final = np.full((len(r1), len(r2)), np.nan)

        # Estimate lambda from known depths.
        # MATLAB: for i = find(~isnan(M0(3*r1, r2)))', lambda(i) = M0(k2i(i)) \ Mdepths_un(k2i(i))
        # The correct Python equivalent: iterate over 2D (local_img, local_pt) pairs.
        # DO NOT use flat indices + k2i - that misinterprets the flat index as an image index.
        valid_fi, valid_fp = np.where(~np.isnan(M0[np.ix_(3*r1, r2)]))
        offsets = np.array([0, 1, 2])
        for fi, fp in zip(valid_fi, valid_fp):
            # fi, fp are LOCAL indices (0..len(r1)-1, 0..len(r2)-1)
            local_rows  = 3 * fi + offsets            # rows in Mdepths_un (local)
            global_rows = 3 * r1[fi] + offsets        # rows in M0 (global)
            global_col  = r2[fp]                      # column in M0 (global)
            m_vec  = M0[global_rows, global_col]
            md_vec = Mdepths_un[local_rows, fp]
            if (not np.any(np.isnan(md_vec)) and
                    not np.any(np.isnan(m_vec)) and
                    np.linalg.norm(md_vec) > 0):
                # MATLAB: lambda(i) = M0(k2i(i)) \ Mdepths_un(k2i(i))
                # A\b = lstsq(A,b), so: lstsq(M0_vec, Mdepths_vec) → lambda = Mdepths/M0
                lambda_final[fi, fp] = np.linalg.lstsq(
                    m_vec.reshape(-1, 1), md_vec, rcond=None
                )[0][0]
        
        # Build rescaled matrix B
        B = np.zeros((3 * len(r1), M.shape[1]))
        for i in range(len(r1)):
            row_idx = Utils.k2i(np.array([i]), step=3)
            M_row = M[Utils.k2i(np.array([r1[i]]), step=3), :]
            lambda_row = lambda_final[i, :].reshape(1, -1)
            ones_col = np.ones((3, 1))
            scaling = ones_col @ lambda_row
            B[row_idx, :] = M_row * scaling
        
        # Fill missing entries with R
        # Same fix: use 2D indices, not flattened; use direct assignment not chained indexing
        fill_img_idx, fill_pt_idx = np.where(np.isnan(M0[np.ix_(3*r1, r2)]))
        if len(fill_img_idx) > 0:
            offsets = np.array([0, 1, 2])
            for fi, fp in zip(fill_img_idx, fill_pt_idx):
                B_rows = 3 * fi + offsets   # local rows in B
                R_rows = 3 * fi + offsets   # local rows in R
                B[B_rows, r2[fp]] = R[R_rows, fp]
        
        # Normalize and balance
        Bn, T_new = normM(B)
        opt['info_separately'] = False
        Bn = balance_triplets(Bn, opt)
        
        # SVD factorization
        if opt['verbose']:
            print('(running svd...', end='', flush=True)
            svd_start = time.time()
        
        u, s, vt = np.linalg.svd(Bn, full_matrices=False)
        s1 = np.sqrt(np.diag(s[:4]))
        P = u[:, :4] @ s1
        X = s1 @ vt[:4, :]
        
        if opt['verbose']:
            svd_time = time.time() - svd_start
            total_time = time.time() - start_time
            print(f'{svd_time:.3f} sec)')
        
        # Denormalize
        P = normMback(P, T_new)
        
        # Compute final error
        info['err']['fact'] = Utils.dist(
            M0[Utils.k2i(r1, step=3), :][:, r2],
            P @ X,
            opt['metric']
        )
        
        if opt['verbose']:
            print(f"Error (after factorization): {info['err']['fact']:.6f}")
        else:
            print(f" {info['err']['fact']:.6f}", end='', flush=True)
    
    # Update u2 to original indexing
    u2 = np.union1d(u2beg, ptbeg[u2])
    info['opt'] = opt
    
    return P, X, u1, u2, info


def compute_predictions(Omega: List[Dict], I: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Compute predictor functions for all strategies.
    
    Args:
        Omega: List of strategy dicts
        I: Visibility matrix (m x n)
    
    Returns:
        S: Strategy scores (number of points)
        F: Strategy feasibilities (missing points)
        strengths: Dict of strength computations for central strategies
    """
    
    S = np.zeros(len(Omega))
    F = np.zeros(len(Omega))
    strengths = {}
    
    for sg in range(len(Omega)):
        if Omega[sg]['name'] == 'seq':
            # Sequence mode
            b, lengths = Utils.subseq_longest(I)
            Isub = I.copy()
            F[sg] = np.sum(Isub == 0)
            S[sg] = np.sum(lengths)
        
        elif Omega[sg]['name'] == 'cent':
            # Central image mode
            i = Omega[sg]['ind']
            strengths[i] = strength(i, I, general=True)
            F[sg] = strengths[i]['strength'][0]
            S[sg] = strengths[i]['num_scaled']
    
    return S, F, strengths


def strength(central: int, I: np.ndarray, general: bool = False) -> Dict:
    """
    Compute the central image strength.
    
    Args:
        central: Central image index
        I: Visibility matrix (m x n)
        general: Use generalized Jacobs' algorithm
    
    Returns:
        Dict with:
            - 'strength': [missing_points, total_points]
            - 'good_rows': Usable rows
            - 'good_cols': Usable columns
            - 'Isub': Submatrix
            - 'num_scaled': Number of scaled points
    """
    
    m, n = I.shape
    good_rows = []
    I_work = I.copy()
    
    # Find rows with sufficient correspondences to central
    for i in range(m):
        if i == central:
            continue
        
        common = np.where(I_work[i, :] & I_work[central, :])[0]
        
        if len(common) >= 8:  # Need ≥8 for fundamental matrix
            good_rows.append(i)
            
            # Only common points if not general
            if not general:
                non_common = np.setdiff1d(np.arange(n), common)
                I_work[i, non_common] = 0
        else:
            I_work[i, :] = 0
    
    good_rows.append(central)
    good_rows = sorted(set(good_rows))
    
    # Need ≥2 points in each column
    present = np.sum(I_work, axis=0)
    good_cols = np.where(present >= 2)[0]
    
    # Extract submatrix
    Isub = I_work[np.ix_(good_rows, good_cols)]
    
    # Compute strength
    strength_val = [
        np.sum(Isub == 0),
        Isub.shape[0] * Isub.shape[1]
    ]
    
    # Find scaled points (those in columns known in central)
    central_idx_in_good = good_rows.index(central)
    scaled_cols = np.where(Isub[central_idx_in_good, :] == 1)[0]
    
    num_scaled = 0
    for i in range(len(good_rows)):
        scaled = np.intersect1d(
            np.where(Isub[i, :] == 1)[0],
            scaled_cols
        )
        num_scaled += len(scaled)
    
    return {
        'strength': strength_val,
        'good_rows': np.array(good_rows),
        'good_cols': good_cols,
        'Isub': Isub,
        'num_scaled': num_scaled
    }


def set_rows_cols(Omega: List[Dict],
                 sg: int,
                 F: np.ndarray,
                 S: np.ndarray,
                 strengths: Dict,
                 I: np.ndarray,
                 info: Dict,
                 opt: Dict) -> Tuple[np.ndarray, np.ndarray, int, Dict]:
    """
    Set rows and columns for selected strategy.
    
    Args:
        Omega: List of strategies
        sg: Selected strategy index
        F: Feasibility scores
        S: Strategy scores
        strengths: Strength computations
        I: Visibility matrix
        info: Information dict
        opt: Options dict
    
    Returns:
        rows: Image indices to use
        cols: Point indices to use
        central: Central image index (0 for sequence)
        info: Updated info dict
    """
    
    m, n = I.shape
    
    if Omega[sg]['name'] == 'seq':
        # Sequence mode
        rows = np.arange(m)
        cols = np.arange(n)
        Isub = I[np.ix_(rows, cols)]
        central = 0
    
    elif Omega[sg]['name'] == 'cent':
        # Central image mode
        central = Omega[sg]['ind']
        rows = strengths[central]['good_rows']
        cols = strengths[central]['good_cols']
        Isub = strengths[central]['Isub']
    
    # Update info
    info['omega'] = Omega[sg].copy()
    info['omega']['F'] = F[sg]
    info['omega']['S'] = S[sg]
    
    # Set sequence info
    sequence = set_sequence(central, S[sg], Isub, I, opt)
    info['sequence'].append(sequence)
    
    return rows, cols, central, info


def set_sequence(central: int,
                num_scaled: float,
                Isub: np.ndarray,
                I: np.ndarray,
                opt: Dict) -> Dict:
    """
    Set sequence information.
    
    Args:
        central: Central image index
        num_scaled: Number of scaled points
        Isub: Submatrix
        I: Full visibility matrix
        opt: Options dict
    
    Returns:
        Dict with sequence statistics
    """
    
    sequence = {'central': central}
    
    if Isub.size == 0:
        sequence['scaled'] = 0
        sequence['missing'] = 0
        sequence['used_pts'] = 0
    else:
        total_in_sub = np.sum(Isub)
        sequence['scaled'] = (num_scaled / total_in_sub * 100) if total_in_sub > 0 else 0
        
        sz = Isub.shape
        sequence['missing'] = np.sum(Isub == 0) / (sz[0] * sz[1]) * 100
        sequence['used_pts'] = total_in_sub / np.sum(I) * 100
    
    if opt.get('verbose', True):
        print(f"Image {central} is the central image, image points: "
              f"{sequence['scaled']:.2f} % scaled, "
              f"{sequence['missing']:.2f} % missing, "
              f"{sequence['used_pts']:.2f} % used.")
    
    return sequence


def normM(M: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize measurement matrix using Hartley normalization.
    
    Args:
        M: Measurement matrix (3m x n)
    
    Returns:
        Mr: Normalized matrix
        T: Transformation matrices (m x 3 x 3)
    """
    
    m = M.shape[0] // 3
    Mr = np.zeros_like(M)
    T = np.zeros((m, 3, 3))
    
    for k in range(m):
        # Get non-NaN entries
        valid_cols = np.where(~np.isnan(M[3*k, :]))[0]
        
        if len(valid_cols) > 0:
            # Normalize
            Tk = Utils.normu(M[Utils.k2i(np.array([k]), step=3), :][:, valid_cols])
            Mr[Utils.k2i(np.array([k]), step=3), :] = Tk @ M[Utils.k2i(np.array([k]), step=3), :]
            T[k, :, :] = Tk
        else:
            T[k, :, :] = np.eye(3)
    
    return Mr, T


def normMback(P: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    Undo normalization on camera matrices.
    
    Args:
        P: Normalized camera matrices (3k x 4)
        T: Transformation matrices (k x 3 x 3)
    
    Returns:
        P: Denormalized camera matrices
    """
    
    k = P.shape[0] // 3
    P_out = P.copy()
    
    for i in range(k):
        Tk = T[i, :, :].reshape(3, 3)
        P_out[3*i:3*i+3, :] = np.linalg.inv(Tk) @ P[3*i:3*i+3, :]
    
    return P_out


if __name__ == "__main__":
    # Test fill_mm function
    print("=" * 60)
    print("fill_mm - Main Reconstruction Test")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Create realistic test data with sufficient correspondences
    m_images = 4
    n_points = 30  # More points
    
    # Generate cameras with proper geometry
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
    
    # Project
    M = P_true @ X_true
    
    # Add noise
    M += np.random.randn(3 * m_images, n_points) * 0.05
    
    # Add missing data carefully to maintain connectivity
    # First 15 points visible in all cameras (for fundamental matrix)
    # Remaining points have random visibility
    missing = np.zeros((3 * m_images, n_points), dtype=bool)
    missing[:, 15:] = np.random.rand(3 * m_images, n_points - 15) < 0.2
    
    # Ensure each point visible in ≥2 cameras
    for j in range(n_points):
        visible = ~missing[0::3, j]
        if np.sum(visible) < 2:
            # Make visible in first 2 cameras
            for k in range(min(2, m_images)):
                missing[3*k:3*k+3, j] = False
    
    M[missing] = np.nan
    
    print(f"\nTest setup:")
    print(f"  Images: {m_images}")
    print(f"  Points: {n_points}")
    print(f"  M shape: {M.shape}")
    print(f"  Missing: {np.sum(missing)}/{M.size}")
    print(f"  Fully visible points: {np.sum(np.all(~np.isnan(M[0::3, :]), axis=0))}")
    
    # Check pairwise correspondences
    for k1 in range(m_images):
        for k2 in range(k1+1, m_images):
            common = np.sum(~np.isnan(M[3*k1, :]) & ~np.isnan(M[3*k2, :]))
            if common < 8:
                print(f"  WARNING: Images {k1}-{k2} have only {common} common points")
    
    # Options
    opt = {
        'strategy': 0,  # Sequence mode for simplicity
        'create_nullspace': {
            'trial_coef': 1.0,
            'threshold': 0.01,
            'verbose': False
        },
        'verbose': True,
        'tol': 1e-6,
        'no_factorization': False,
        'metric': 1
    }
    
    # Test: Full reconstruction
    print("\n" + "-" * 60)
    print("TEST: Full Reconstruction (Sequence Mode)")
    print("-" * 60)
    
    try:
        P, X, u1, u2, info = fill_mm(M, opt)
        
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
            
            # Check which strategy was used
            if 'omega' in info:
                print(f"\n  Strategy used: {info['omega']['name']}", end='')
                if 'ind' in info['omega']:
                    print(f" (central image {info['omega']['ind']})")
                else:
                    print(" (sequence)")
        else:
            print("✗ Reconstruction failed")
            print(f"  Lost images: {len(u1)}/{m_images}")
            print(f"  Lost points: {len(u2)}/{n_points}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("✓ Test completed")
    print("=" * 60)
