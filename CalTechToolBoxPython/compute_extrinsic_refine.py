import numpy as np
from CalTechToolBoxPython.project_points2 import project_points2
from CalTechToolBoxPython.rodrigues import rodrigues


def compute_extrinsic_refine(omc_init, Tc_init, x_kk, X_kk,
                              fc, cc, kc=None, alpha_c=0.0,
                              max_iter=20, thresh_cond=np.inf):
    """
    Refine extrinsic camera parameters by minimising reprojection error.

    Starting from an initial pose estimate (omc_init, Tc_init), this function
    runs a Gauss-Newton optimisation loop that iteratively updates the rotation
    vector and translation until convergence or max_iter is reached.

    This is the NON-LINEAR refinement step. It follows a linear initialisation
    (e.g. from a homography decomposition) and drives the reprojection error
    to a true local minimum under the full distortion model.

    Parameters
    ----------
    omc_init    : (3,)   — initial rotation vector
    Tc_init     : (3,)   — initial translation vector
    x_kk        : (2, N) — observed 2D image points (pixels)
    X_kk        : (3, N) — corresponding 3D world points
    fc          : (2,)   — focal lengths [fx, fy]
    cc          : (2,)   — principal point [cx, cy]
    kc          : (5,)   — distortion coefficients (default zeros)
    alpha_c     : float  — skew coefficient (default 0)
    max_iter    : int    — maximum Gauss-Newton iterations (default 20)
    thresh_cond : float  — stop if cond(J) exceeds this (default inf = never)

    Returns
    -------
    omckk : (3,)   — refined rotation vector
    Tckk  : (3,)   — refined translation vector
    Rckk  : (3,3)  — rotation matrix corresponding to omckk
    JJ    : (2N,6) — final Jacobian [dxpdom | dxpdT] at convergence

    Notes
    -----
    Gauss-Newton update rule:
        param_innov = inv(JᵀJ) · Jᵀ · e
    where e = x_kk − project_points2(X_kk, om, T, ...) is the residual vector
    and J = [dxpdom | dxpdT] is the (2N×6) stacked Jacobian from project_points2.

    The 2N residual rows are interleaved [ex0, ey0, ex1, ey1, ...] matching
    the F-order convention used throughout this toolbox.
    """

    if kc is None:
        kc = np.zeros(5)

    omckk = np.asarray(omc_init, dtype=float).ravel()
    Tckk  = np.asarray(Tc_init,  dtype=float).ravel()
    x_kk  = np.asarray(x_kk,    dtype=float)
    X_kk  = np.asarray(X_kk,    dtype=float)
    fc    = np.asarray(fc,       dtype=float)
    cc    = np.asarray(cc,       dtype=float)
    kc    = np.asarray(kc,       dtype=float)

    n = X_kk.shape[1]

    # Pack rotation and translation into a single 6-vector for the update step
    param = np.concatenate([omckk, Tckk])

    change = 1.0
    itr    = 0
    JJ     = np.zeros((2 * n, 6))   # will hold the final Jacobian at exit

    # ------------------------------------------------------------------ #
    # Gauss-Newton loop
    # Convergence criterion: relative parameter change < 1e-10
    # ------------------------------------------------------------------ #
    while change > 1e-10 and itr < max_iter:

        # Step 1 — Project current estimate onto the image plane.

        xp, dxpdom, dxpdT, *_ = project_points2(
            X_kk, omckk, Tckk, fc, cc, kc, alpha_c)

        # Step 2 — Residual: observed minus projected (in pixels)

        ex = (x_kk - xp).ravel(order='F')

        # Step 3 — Stack Jacobians: J = [dxpdom | dxpdT], shape (2N, 6)
        JJ = np.hstack([dxpdom, dxpdT])

        # Step 4 — Check conditioning; stop if matrix is near-singular
        if np.linalg.cond(JJ) > thresh_cond:
            break

        # Step 5 — Gauss-Newton normal equations: (JᵀJ) · Δp = Jᵀ · e
        JJ2         = JJ.T @ JJ                          # (6, 6)
        param_innov = np.linalg.solve(JJ2, JJ.T @ ex)

        # Step 6 — Update parameter vector and measure relative change
        param_up = param + param_innov
        change   = np.linalg.norm(param_innov) / (np.linalg.norm(param_up) + 1e-300)
        param    = param_up
        itr     += 1

        omckk = param[:3]
        Tckk  = param[3:6]

    # Convert final rotation vector to matrix using our rodrigues.py
    Rckk, _ = rodrigues(omckk)

    return omckk, Tckk, Rckk, JJ


# ------------------------------------------------------------------ #
# Self-test: synthesise a camera + calibration points, perturb the pose,
# then verify that refinement recovers the ground-truth pose.
# ------------------------------------------------------------------ #
if __name__ == '__main__':
    from project_points2 import project_points2

    np.random.seed(42)

    # Ground-truth camera intrinsics
    fc = np.array([800.0, 800.0])
    cc = np.array([320.0, 240.0])
    kc = np.array([-0.3, 0.15, 0.001, -0.0005, 0.0])
    alpha_c = 0.0

    # Ground-truth pose
    om_true = np.array([0.1, -0.05, 0.03])
    T_true  = np.array([0.0, 0.0, 0.5])

    # Synthetic calibration grid (20 points on a flat plane)
    X_kk = np.zeros((3, 20))
    X_kk[0], X_kk[1] = np.meshgrid(
        np.linspace(-0.1, 0.1, 5),
        np.linspace(-0.1, 0.1, 4)
    )[0].ravel(), np.meshgrid(
        np.linspace(-0.1, 0.1, 5),
        np.linspace(-0.1, 0.1, 4)
    )[1].ravel()

    # Perfect projections + tiny noise (0.1px)
    x_kk, *_ = project_points2(X_kk, om_true, T_true, fc, cc, kc, alpha_c)
    x_kk += np.random.randn(*x_kk.shape) * 0.1

    # Perturbed initial estimate (1° rotation error, 5mm translation error)
    om_init = om_true + np.array([0.02, -0.01, 0.015])
    T_init  = T_true  + np.array([0.005, -0.003, 0.008])

    # Error before refinement
    xp_init, *_ = project_points2(X_kk, om_init, T_init, fc, cc, kc, alpha_c)
    err_before = np.sqrt(np.mean((x_kk - xp_init)**2))

    # Run refinement
    omckk, Tckk, Rckk, JJ = compute_extrinsic_refine(
        om_init, T_init, x_kk, X_kk, fc, cc, kc, alpha_c
    )

    # Error after refinement
    xp_refined, *_ = project_points2(X_kk, omckk, Tckk, fc, cc, kc, alpha_c)
    err_after = np.sqrt(np.mean((x_kk - xp_refined)**2))

    print(f"RMS reprojection error  before refinement: {err_before:.4f} px")
    print(f"RMS reprojection error  after  refinement: {err_after:.4f} px")
    print(f"om recovered:  {np.round(omckk,   6)}  (true: {om_true})")
    print(f"T  recovered:  {np.round(Tckk,    6)}  (true: {T_true})")
    print(f"R orthogonality check (RᵀR - I max): {np.max(np.abs(Rckk.T @ Rckk - np.eye(3))):.2e}")
    assert err_after < err_before, "Refinement should reduce reprojection error"
    assert err_after < 0.2,        "Refined error should be close to noise level"
    print("All checks passed ✓")
