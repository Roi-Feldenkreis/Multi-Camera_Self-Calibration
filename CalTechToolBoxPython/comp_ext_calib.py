import numpy as np
from CalTechToolBoxPython.check_active_images import check_active_images
from CalTechToolBoxPython.compute_extrinsic_refine import compute_extrinsic_refine
from CalTechToolBoxPython.normalize import normalize
from CalTechToolBoxPython.rodrigues import rodrigues


def comp_ext_calib(x_list, X_list, fc, cc, kc=None, alpha_c=0.0,
                   active_images=None, thresh_cond=np.inf,
                   check_cond=True, max_iter=20,
                   compute_extrinsic_init_fn=None):
    """
    Compute extrinsic parameters (pose) for all active calibration images.

    This function is the orchestrator of the extrinsic estimation stage in
    the calibration pipeline. For each active image it:
      1. Gets a linear initial pose estimate  (compute_extrinsic_init)
      2. Refines it with Gauss-Newton         (compute_extrinsic_refine)
      3. Deactivates any image whose Jacobian is ill-conditioned or whose
         pose is invalid (NaN), and reports a warning.

    It is called once per calibration run, after the intrinsic parameters
    (fc, cc, kc, alpha_c) have been estimated.

    Parameters
    ----------
    x_list       : list of (2, N_k) arrays — observed 2D corners per image
    X_list       : list of (3, N_k) arrays — 3D world points per image
    fc           : (2,)  — focal lengths [fx, fy]
    cc           : (2,)  — principal point [cx, cy]
    kc           : (5,)  — distortion coefficients (default zeros)
    alpha_c      : float — skew coefficient (default 0)
    active_images: list of int/bool, length n_ima (None = all active)
    thresh_cond  : float — Jacobian condition number threshold (default inf)
    check_cond   : bool  — whether to deactivate ill-conditioned views
    max_iter     : int   — max Gauss-Newton iterations per image (default 20)
    compute_extrinsic_init_fn : callable — linear initialiser
        signature: (x_kk, X_kk, fc, cc, kc, alpha_c) → (omc_init, Tc_init)
        If None, a simple DLT-based fallback is used.

    Returns
    -------
    omc_list           : list of (3,)   — rotation vectors (NaN if inactive)
    Tc_list            : list of (3,)   — translation vectors (NaN if inactive)
    active_images      : list of int    — updated flags (may deactivate views)
    N_points_views     : list of int    — number of points used per image
    deactivated_images : list of int    — 0-based indices newly deactivated
    """

    if kc is None:
        kc = np.zeros(5)

    n_ima = len(x_list)
    n_ima, active_images, ind_active = check_active_images(n_ima, active_images)

    omc_list           = [np.full(3,    np.nan)] * n_ima
    Tc_list            = [np.full(3,    np.nan)] * n_ima
    N_points_views     = [0]                     * n_ima
    deactivated_images = []

    # Fallback linear initialiser if none provided
    # (mirrors compute_extrinsic_init from the CalTech toolbox)
    if compute_extrinsic_init_fn is None:
        compute_extrinsic_init_fn = _default_extrinsic_init

    for kk in range(n_ima):
        x_kk = x_list[kk]
        X_kk = X_list[kk]

        # ------------------------------------------------------------------ #
        # Guard: image data must exist and first point must not be NaN
        # (mirrors MATLAB's exist(['x_' num2str(kk)]) and isnan check)
        # ------------------------------------------------------------------ #
        data_missing = x_kk is None or X_kk is None
        data_nan     = (not data_missing) and np.isnan(x_kk.flat[0])

        if data_missing or data_nan:
            if active_images[kk]:
                print(f"Warning: Cannot calibrate with image {kk}. "
                      f"Need to extract grid corners first.\n"
                      f"         Set active_images[{kk}]=1 and re-run.")
            if data_nan:
                active_images[kk] = 0
            # Leave omc, Tc as NaN for this image
            continue

        if not active_images[kk]:
            # Image explicitly deactivated — leave NaN, skip processing
            continue

        # ------------------------------------------------------------------ #
        # Step 1 — Linear initialisation: fast closed-form pose estimate
        # Uses the normalised image coordinates and a homography / DLT.
        # This ignores distortion but gives a good starting point.
        # ------------------------------------------------------------------ #
        N_points_views[kk] = x_kk.shape[1]
        omc_init, Tc_init = compute_extrinsic_init_fn(
            x_kk, X_kk, fc, cc, kc, alpha_c
        )

        # ------------------------------------------------------------------ #
        # Step 2 — Non-linear refinement via Gauss-Newton
        # Minimises pixel reprojection error under the full distortion model.
        # Returns the refined pose and the final Jacobian JJ_kk.
        # ------------------------------------------------------------------ #
        omckk, Tckk, Rckk, JJ_kk = compute_extrinsic_refine(
            omc_init, Tc_init, x_kk, X_kk,
            fc, cc, kc, alpha_c, max_iter, thresh_cond
        )

        # ------------------------------------------------------------------ #
        # Step 3 — Conditioning check
        # If the Jacobian is ill-conditioned the pose estimate is unreliable.
        # Deactivate the image and record it so the caller can report it.
        # ------------------------------------------------------------------ #
        if check_cond and np.linalg.cond(JJ_kk) > thresh_cond:
            active_images[kk] = 0
            omckk = np.full(3, np.nan)
            Tckk  = np.full(3, np.nan)
            print(f"\nWarning: View #{kk} ill-conditioned (cond={np.linalg.cond(JJ_kk):.1e} "
                  f"> {thresh_cond:.1e}). Image set inactive.")
            deactivated_images.append(kk)

        # ------------------------------------------------------------------ #
        # Step 4 — NaN check on result
        # compute_extrinsic_refine may return NaN if the optimiser diverged.
        # Deactivate such images silently.
        # ------------------------------------------------------------------ #
        if np.isnan(omckk[0]):
            active_images[kk] = 0

        omc_list[kk] = omckk
        Tc_list[kk]  = Tckk

    # Re-run check_active_images to keep ind_active consistent after
    # any deactivations that happened during the loop
    _, active_images, ind_active = check_active_images(n_ima, active_images)

    return omc_list, Tc_list, active_images, N_points_views, deactivated_images


# ------------------------------------------------------------------ #
# Fallback linear initialiser (simplified DLT / normalised homography)
# In the full CalTech toolbox this is compute_extrinsic_init.m
# This version handles planar targets (Z=0) via a homography.
# ------------------------------------------------------------------ #
def _default_extrinsic_init(x_kk, X_kk, fc, cc, kc, alpha_c):
    """
    Linear pose initialisation for a planar calibration target (Z=0).
    Computes a homography between the world plane and the image, then
    decomposes it into R and T using the intrinsic matrix.
    """


    n = x_kk.shape[1]

    # Normalise image points (remove intrinsics + distortion approximately)
    xn = normalize(x_kk, fc, cc, kc, alpha_c)    # (2, N)

    # Build the DLT system for a planar target (X_kk[2,:] == 0 assumed)
    # Homography H maps [X;Y;1] → [x;y;1] in normalised coords
    X2d = X_kk[:2, :]    # (2, N) — use only X and Y (Z=0 plane)

    A = np.zeros((2 * n, 9))
    for i in range(n):
        Xi = np.array([X2d[0,i], X2d[1,i], 1.0])
        A[2*i,   0:3] = Xi;  A[2*i,   6:9] = -xn[0,i] * Xi
        A[2*i+1, 3:6] = Xi;  A[2*i+1, 6:9] = -xn[1,i] * Xi

    _, _, Vt = np.linalg.svd(A)
    h = Vt[-1]
    H = h.reshape(3, 3)

    # Recover r1, r2, T from H = K^{-1} already applied (normalised coords)
    r1 = H[:, 0];  lam = np.linalg.norm(r1)
    r1 = r1 / lam
    r2 = H[:, 1] / lam
    r3 = np.cross(r1, r2)
    T  = H[:, 2] / lam

    # Ensure proper rotation (project onto SO(3))
    R_approx = np.column_stack([r1, r2, r3])
    U, _, Vt = np.linalg.svd(R_approx)
    R = U @ Vt

    # Rotation vector from matrix
    om, _ = rodrigues(R)

    return om, T


# ------------------------------------------------------------------ #
# Self-test
# ------------------------------------------------------------------ #
if __name__ == '__main__':
    from project_points2 import project_points2
    from rodrigues import rodrigues

    np.random.seed(7)

    # Synthetic camera
    fc = np.array([800.0, 800.0])
    cc = np.array([320.0, 240.0])
    kc = np.array([-0.3, 0.15, 0.001, -0.0005, 0.0])
    alpha_c = 0.0

    # Three calibration views with different poses
    n_ima = 4
    x_list, X_list = [], []
    om_true_list, T_true_list = [], []

    grid_x, grid_y = np.meshgrid(np.linspace(-0.08, 0.08, 6),
                                  np.linspace(-0.06, 0.06, 5))
    X_base = np.zeros((3, 30))
    X_base[0] = grid_x.ravel()
    X_base[1] = grid_y.ravel()

    for i in range(n_ima):
        om_t = np.array([0.05*i, -0.03*i, 0.02]) + np.random.randn(3)*0.02
        T_t  = np.array([0.01*i, -0.005*i, 0.5])
        xp, *_ = project_points2(X_base, om_t, T_t, fc, cc, kc, alpha_c)
        xp += np.random.randn(*xp.shape) * 0.3
        x_list.append(xp)
        X_list.append(X_base.copy())
        om_true_list.append(om_t)
        T_true_list.append(T_t)

    # Mark image 2 as inactive
    active = [1, 1, 0, 1]

    omc_list, Tc_list, active_out, N_pts, deact = comp_ext_calib(
        x_list, X_list, fc, cc, kc, alpha_c,
        active_images=active
    )

    print(f"Images processed: active={sum(active_out)}/{n_ima}, "
          f"deactivated this run={deact}")
    for kk in range(n_ima):
        if active_out[kk]:
            xp_ref, *_ = project_points2(X_list[kk], omc_list[kk],
                                          Tc_list[kk], fc, cc, kc, alpha_c)
            rms = np.sqrt(np.mean((x_list[kk] - xp_ref)**2))
            print(f"  Image {kk}: {N_pts[kk]} pts, RMS reprojection = {rms:.3f} px")
        else:
            nan_flag = np.all(np.isnan(omc_list[kk]))
            print(f"  Image {kk}: INACTIVE (omc={'NaN' if nan_flag else 'set'})")
    print("Self-test passed ✓")
