import numpy as np
from CalTechToolBoxPython.check_active_images import check_active_images


def comp_error_calib(X_list, x_list, omc_list, Tc_list,
                     fc, cc, kc, alpha_c=0.0,
                     active_images=None,
                     project_points_fn=None):
    """
    Recompute the reprojection error for all active calibration images.

    For each active image, the 3D calibration points are re-projected using
    the current camera parameters, and the pixel-level difference between
    detected and reprojected corners is computed.

    Parameters
    ----------
    X_list   : list of (3, N_k) arrays — 3D world points per image
    x_list   : list of (2, N_k) arrays — detected 2D corners per image
    omc_list : list of (3,) arrays     — rotation vectors per image
    Tc_list  : list of (3,1) arrays    — translation vectors per image
    fc       : (2,)  — focal lengths [fx, fy]
    cc       : (2,)  — principal point [cx, cy]
    kc       : (5,)  — distortion coefficients
    alpha_c  : float — skew coefficient (default 0)
    active_images : list of int/bool, length n_ima (None = all active)
    project_points_fn : callable — project_points2(X, om, T, fc, cc, kc, alpha_c)
                        If None, a simple pinhole projection is used as fallback.

    Returns
    -------
    ex_list  : list of (2, N_k) arrays — per-image pixel errors (NaN if inactive)
    ex       : (2, M) array            — all errors concatenated (active images only)
    x_all    : (2, M) array            — all detected corners (active images only)
    y_all    : (2, M) array            — all reprojected points (active images only)
    err_std  : (2,) array              — std of reprojection error [x-std, y-std]
    """

    n_ima = len(X_list)
    n_ima, active_flags,_ = check_active_images(n_ima, active_images)

    # Fallback: simple pinhole projection (no distortion) if no function provided
    if project_points_fn is None:
        def project_points_fn(X, om, T, fc, cc, kc, alpha_c):
            from rodrigues import rodrigues
            R, _ = rodrigues(om)
            Xc = R @ X + T                          # (3, N) in camera frame
            xn = Xc[:2, :] / Xc[2:3, :]            # normalised coords
            return np.array([
                fc[0] * xn[0, :] + cc[0],
                fc[1] * xn[1, :] + cc[1]
            ])

    ex_list = []
    ex   = np.zeros((2, 0))
    x_all = np.zeros((2, 0))
    y_all = np.zeros((2, 0))

    for kk in range(n_ima):
        omckk = omc_list[kk]
        Tckk  = Tc_list[kk]

        # Skip inactive images or images with an invalid (NaN) pose
        if active_flags[kk] and not np.isnan(omckk.flat[0]):

            # Reproject 3D points onto the image plane
            y_kk = project_points_fn(
                X_list[kk], omckk, Tckk, fc, cc, kc, alpha_c
            )

            # Per-image error: detected − reprojected
            ex_kk = x_list[kk] - y_kk
            ex_list.append(ex_kk)

            # Accumulate global arrays
            ex    = np.hstack([ex,    ex_kk])
            x_all = np.hstack([x_all, x_list[kk]])
            y_all = np.hstack([y_all, y_kk])

        else:
            # Inactive image: error is undefined
            ex_list.append(np.full((2, 1), np.nan))

    # Standard deviation of reprojection error across all active points
    err_std = np.std(ex, axis=1)   # (2,) — one value per axis

    return ex_list, ex, x_all, y_all, err_std


# ------------------------------------------------------------------ #
# Self-test with synthetic data
# ------------------------------------------------------------------ #
if __name__ == '__main__':
    np.random.seed(42)
    from rodrigues import rodrigues

    n_ima = 3
    fc = np.array([800., 800.])
    cc = np.array([320., 240.])
    kc = np.zeros(5)

    # Generate synthetic 3D points and perfect projections
    X_list, x_list, omc_list, Tc_list = [], [], [], []
    for _ in range(n_ima):
        X = np.random.uniform(-0.1, 0.1, (3, 10))
        X[2, :] = 0                              # planar pattern at Z=0
        om = np.random.randn(3) * 0.1
        T  = np.array([[0.], [0.], [1.0]])
        R, _ = rodrigues(om)
        Xc = R @ X + T
        xn = Xc[:2] / Xc[2:3]
        x  = np.array([fc[0]*xn[0]+cc[0], fc[1]*xn[1]+cc[1]])
        x  += np.random.randn(*x.shape) * 0.5   # add ~0.5px noise
        X_list.append(X); x_list.append(x)
        omc_list.append(om); Tc_list.append(T)

    # Mark image 1 as inactive
    active = [1, 0, 1]
    ex_list, ex, x_all, y_all, err_std = comp_error_calib(
        X_list, x_list, omc_list, Tc_list, fc, cc, kc, active_images=active
    )

    print(f"Active images processed: {sum(active)}/{n_ima}")
    print(f"Total active points:     {ex.shape[1]}")
    print(f"Reprojection error std:  x={err_std[0]:.3f}px  y={err_std[1]:.3f}px")
    print(f"Image 1 (inactive) ex:   {ex_list[1].T}  (should be NaN)")
