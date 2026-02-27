"""
bundle_PX_proj - Projective bundle adjustment with image conditioning
Converted from MATLAB to Python
"""

import numpy as np
from scipy.sparse import eye as speye
from typing import Tuple, Dict, Optional, List, Callable
from MartinecPajdla.Utils import Utils
from MartinecPajdla.eval_y_and_dy import eval_y_and_dy


def bundle_PX_proj(P0: np.ndarray,
                  X0: np.ndarray,
                  q: np.ndarray,
                  imsize: np.ndarray,
                  nl_params_all_cams: Optional[List[Dict]] = None,
                  opt: Optional[Dict] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Projective bundle adjustment with image conditioning.
    
    Args:
        P0: Initial camera matrices (3K x 4)
        X0: Initial 3D points (4 x N)
        q: Image observations (2K x N) in Euclidean coordinates
        imsize: Image sizes (2 x K), imsize[:, k] is size of image k
        nl_params_all_cams: Radial distortion parameters (optional, not implemented)
        opt: Options dict with:
            - 'verbose' (1): Print progress
            - 'verbose_short' (0): Print short progress
            - 'res_scale' (1): Residual scale for printing
            - 'max_niter' (10000): Max iterations
            - 'max_stepy' (100*eps): Max step for termination
            - 'lam_init' (1e-4): Initial lambda
    
    Returns:
        P: Optimized camera matrices (3K x 4)
        X: Optimized 3D points (4 x N)
    
    Note: P0, X0, q need not be preconditioned - done internally.
    """
    
    # Set default options
    if opt is None:
        opt = {}
    if 'verbose' not in opt:
        opt['verbose'] = 1
    if 'verbose_short' not in opt:
        opt['verbose_short'] = 0
    
    # Check radial distortion
    RADIAL = nl_params_all_cams is not None and len(nl_params_all_cams) > 0
    if RADIAL:
        raise NotImplementedError('Radial distortion in bundle adjustment not implemented')
    
    K = q.shape[0] // 2  # Number of cameras
    N = q.shape[1]        # Number of points
    
    # Precondition: apply image-based conditioning
    H = []
    P0_cond = P0.copy()
    q_cond = q.copy()
    
    for k in range(K):
        # Get conditioning matrix for image k
        H_k, _ = vgg_conditioner_from_image(imsize[:, k])
        H.append(H_k)
        
        # Condition camera
        P0_cond[Utils.k2i(np.array([k]), step=3), :] = \
            H_k @ P0[Utils.k2i(np.array([k]), step=3), :]
        
        # Condition image points
        q_2k = q[Utils.k2i(np.array([k]), step=2), :]
        q_hom = Utils.hom(q_2k)
        q_cond_hom = H_k @ q_hom
        q_cond[Utils.k2i(np.array([k]), step=2), :] = Utils.p2e(q_cond_hom)
    
    # Normalize
    P0_cond = Utils.normP(P0_cond)
    X0_cond = Utils.normx(X0)
    
    # Form observation vector y
    qq = q_cond.reshape(2, K * N, order='F')
    qivis = np.all(~np.isnan(qq), axis=0).reshape(K, N, order='F')
    qq_vis = qq[:, qivis.T.flatten()]
    y = qq_vis.flatten('F')
    
    # Compute transformation matrices TP, TX
    # Tangent space parameterization
    TX = []
    for n in range(N):
        Q, R = np.linalg.qr(X0_cond[:, n].reshape(-1, 1), mode='complete')
        TX.append(Q[1:, :].T)  # (4 x 3)
    
    TP = []
    for k in range(K):
        P_k = P0_cond[Utils.k2i(np.array([k]), step=3), :].flatten('F').reshape(-1, 1)
        Q, R = np.linalg.qr(P_k, mode='complete')
        TP.append(Q[1:, :].T)  # (12 x 11)
    
    # Form initial parameter vector
    NR = 3 if RADIAL else 0
    p0 = []
    
    for k in range(K):
        p0.append(np.zeros(11))
        if RADIAL:
            p0.append(nl_params_all_cams[k]['u0'])
            p0.append(np.array([nl_params_all_cams[k]['kappa']]))
    
    p0.append(np.zeros(3 * N))
    p0 = np.concatenate([x.flatten() for x in p0])
    
    # Levenberg-Marquardt optimization
    # Use eval_y_and_dy function directly (as in original MATLAB)
    def objective(p_vec):
        return eval_y_and_dy(p_vec, P0_cond, TP, X0_cond, TX, y, qivis, RADIAL)
    
    p = levmarq(objective, p0, opt)
    
    # Extract optimized parameters and undo conditioning
    P = np.zeros_like(P0)
    X = np.zeros_like(X0)
    
    for k in range(K):
        iP_k = p[k * (11 + NR):(k * (11 + NR)) + 11]
        P_k_flat = TP[k] @ iP_k
        P_k_cond = P0_cond[Utils.k2i(np.array([k]), step=3), :] + \
                   P_k_flat.reshape(3, 4, order='F')
        
        # Undo conditioning
        P[Utils.k2i(np.array([k]), step=3), :] = np.linalg.inv(H[k]) @ P_k_cond
        
        if RADIAL:
            # Extract radial parameters (not used in this implementation)
            pass
    
    for n in range(N):
        iX_n = p[(11 + NR) * K + n * 3:(11 + NR) * K + (n + 1) * 3]
        X[:, n] = X0_cond[:, n] + TX[n] @ iX_n
    
    if opt.get('verbose_short', False):
        print(')', end='', flush=True)
    
    return P, X


def levmarq(F: Callable,
           p: np.ndarray,
           opt: Dict) -> np.ndarray:
    """
    Levenberg-Marquardt optimization.
    
    Minimizes ||F(p)|| over p using Levenberg-Marquardt algorithm.
    
    Args:
        F: Function that returns (residual, Jacobian)
        p: Initial parameter vector
        opt: Options dict with:
            - 'verbose': Print progress
            - 'verbose_short': Print short progress
            - 'res_scale': Residual scale for printing
            - 'max_niter': Max iterations
            - 'max_stepy': Max step for termination
            - 'lam_init': Initial lambda
    
    Returns:
        p: Optimized parameter vector
    """
    
    # Handle options
    verbose = opt.get('verbose', 0)
    verbose_short = opt.get('verbose_short', 0)
    res_scale = opt.get('res_scale', 1.0)
    max_niter = opt.get('max_niter', 10000)
    
    # Set max_stepy with flag for whether it was user-defined
    if 'max_stepy' in opt:
        max_stepy = opt['max_stepy']
        max_stepy_undef = False
    else:
        max_stepy = 100 * np.finfo(float).eps * res_scale
        max_stepy_undef = True
    
    lam_init = opt.get('lam_init', 1e-4)
    
    # Initialize
    lastFp = np.inf
    stepy = np.inf
    lam = lam_init
    nfail = 0
    niter = 0
    
    # Initial evaluation
    Fp, J = F(p)
    
    # Create identity matrix (sparse or dense based on J)
    if hasattr(J, 'toarray'):  # Sparse
        eyeJ = speye(J.shape[1], format='csr')
    else:
        eyeJ = np.eye(J.shape[1])
    
    # Print initial residuals
    if verbose or verbose_short:
        rms = res_scale * np.sqrt(np.mean(Fp ** 2))
        max_res = res_scale * np.max(np.abs(Fp))
        
        if not verbose_short:
            print(f'                {rms:14.10g} [rms] {max_res:14.10g} [max]')
        else:
            print(f'(rms/max/stepmax: {rms:g}/{max_res:g}/', end='', flush=True)
    
    # Optimization loop
    while nfail < 20 and stepy * res_scale > max_stepy and niter < max_niter:
        
        # Compute step
        JtJ = J.T @ J
        Jtr = J.T @ Fp
        
        # Solve damped system
        if hasattr(JtJ, 'toarray'):  # Sparse
            from scipy.sparse.linalg import spsolve
            D = -spsolve(JtJ + lam * eyeJ, Jtr)
        else:
            D = -np.linalg.solve(JtJ + lam * eyeJ, Jtr)
        
        # Check for NaN or Inf
        if np.any(np.isnan(D) | np.isinf(D)):
            p[:] = np.nan
            return p
        
        # Evaluate at new point
        FpD, _ = F(p + D)
        
        # Check if step improves
        if np.sum(Fp ** 2) > np.sum(FpD ** 2):  # Success
            p = p + D
            lam = max(lam / 10, 1e-15)
            stepy = np.max(np.abs(Fp - lastFp))
            lastFp = Fp
            Fp, J = F(p)
            nfail = 0
            niter += 1
        else:  # Failure
            lam = min(lam * 10, 1e5)
            nfail += 1
        
        # Print progress if success
        if (verbose or verbose_short) and nfail == 0:
            rms = res_scale * np.sqrt(np.mean(Fp ** 2))
            max_res = res_scale * np.max(np.abs(Fp))
            step_scaled = res_scale * stepy
            
            if not verbose_short:
                print(f' {lam:7.2g} [lam]: {rms:14.10g} [rms] {max_res:14.10g} [max] {step_scaled:10.5g} [stepmax]')
            else:
                print(f' {rms:g}/{max_res:g}/{step_scaled:g}', end='', flush=True)
    
    # Check termination reason
    if not max_stepy_undef and stepy * res_scale <= max_stepy:
        print(f'\n!!! finished because of high opt.max_stepy(={max_stepy:f})')
    
    return p


def vgg_conditioner_from_image(imsize: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create similarity transform for conditioning image points.
    
    Centers image at origin and normalizes scale.
    
    Args:
        imsize: Image size [width, height] or (width, height)
    
    Returns:
        C: Conditioning matrix (3 x 3)
        invC: Inverse of C (3 x 3)
    """
    
    if imsize.ndim == 1:
        c = imsize[0]  # width
        r = imsize[1]  # height
    else:
        c = imsize[0] if imsize.size == 2 else imsize
        r = imsize[1] if imsize.size == 2 else imsize
    
    # Average of width and height
    f = (c + r) / 2
    
    # Conditioning matrix: translate to center and scale
    C = np.array([
        [1/f, 0, -c/(2*f)],
        [0, 1/f, -r/(2*f)],
        [0, 0, 1]
    ])
    
    # Inverse (more efficient than np.linalg.inv)
    invC = np.array([
        [f, 0, c/2],
        [0, f, r/2],
        [0, 0, 1]
    ])
    
    return C, invC


if __name__ == "__main__":
    # Test bundle_PX_proj function
    print("=" * 60)
    print("bundle_PX_proj - Bundle Adjustment Test")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Create test data
    K = 3  # cameras
    N = 20  # points
    
    # Image sizes
    imsize = np.array([[640, 480]] * K).T  # (2 x K)
    
    # Generate ground truth cameras
    P_cameras = []
    for k in range(K):
        angle = k * 0.3
        R = np.array([[np.cos(angle), -np.sin(angle), 0],
                      [np.sin(angle), np.cos(angle), 0],
                      [0, 0, 1]])
        t = np.array([[k * 2], [0], [5]])  # Move cameras in z
        P_cameras.append(np.hstack([R, t]))
    
    P_true = np.vstack(P_cameras)
    
    # Generate 3D points
    X_true = np.vstack([
        np.random.randn(3, N) * 3,
        np.ones((1, N))
    ])
    
    # Project to get observations
    q = np.zeros((2 * K, N))
    for k in range(K):
        x = P_true[Utils.k2i(np.array([k]), step=3), :] @ X_true
        q[Utils.k2i(np.array([k]), step=2), :] = Utils.p2e(x)
    
    # Add noise
    q += np.random.randn(2 * K, N) * 0.5
    
    # Add some missing data
    missing = np.random.rand(2 * K, N) < 0.1
    q[missing] = np.nan
    
    # Perturb initial guess
    P0 = P_true + np.random.randn(3 * K, 4) * 0.2
    X0 = X_true + np.random.randn(4, N) * 0.2
    
    print(f"\nTest setup:")
    print(f"  Cameras: {K}")
    print(f"  Points: {N}")
    print(f"  Image size: {imsize[:, 0]}")
    print(f"  Missing observations: {np.sum(missing)}/{q.size}")
    
    # Options
    opt = {
        'verbose': True,
        'verbose_short': False,
        'max_niter': 50,
        'lam_init': 1e-3
    }
    
    # Test bundle adjustment
    print("\n" + "-" * 60)
    print("TEST: Bundle Adjustment")
    print("-" * 60)
    
    try:
        P_opt, X_opt = bundle_PX_proj(P0, X0, q, imsize, None, opt)
        
        print(f"\nOptimization complete!")
        print(f"P shape: {P_opt.shape}")
        print(f"X shape: {X_opt.shape}")
        
        # Compute final reprojection error
        q_final = np.zeros((2 * K, N))
        for k in range(K):
            x = P_opt[Utils.k2i(np.array([k]), step=3), :] @ X_opt
            q_final[Utils.k2i(np.array([k]), step=2), :] = Utils.p2e(x)
        
        valid = ~np.isnan(q)
        error = np.sqrt(np.mean((q[valid] - q_final[valid]) ** 2))
        print(f"Final RMSE: {error:.6f} pixels")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test helper functions
    print("\n" + "-" * 60)
    print("TEST: Helper Functions")
    print("-" * 60)
    
    # Test vgg_conditioner_from_image
    C, invC = vgg_conditioner_from_image(np.array([640, 480]))
    print(f"Conditioning matrix:")
    print(C)
    print(f"Inverse (check):")
    print(C @ invC)  # Should be identity
    
    # Test hom/p2e
    x_euc = np.array([[1, 2, 3], [4, 5, 6]])
    x_hom = Utils.hom(x_euc)
    x_back = Utils.p2e(x_hom)
    print(f"\nEuclidean → Homogeneous → Euclidean:")
    print(f"Original: {x_euc[:, 0]}")
    print(f"Homogeneous: {x_hom[:, 0]}")
    print(f"Back: {x_back[:, 0]}")
    print(f"Error: {np.max(np.abs(x_euc - x_back)):.2e}")
    
    print("\n" + "=" * 60)
    print("✓ All tests completed")
    print("=" * 60)
