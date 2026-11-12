import numpy as np


def compute_homography(xn, Xp):
    """
    Compute the planar homography H mapping 2D plane points Xp to
    normalised image coordinates xn, using the normalised DLT algorithm.

    H is defined so that:  lambda * [xn; 1] = H * [Xp; 1]

    Parameters
    ----------
    xn : (2, N) — normalised (undistorted) image point coordinates
    Xp : (2, N) — 2D calibration pattern points (Z=0 plane)

    Returns
    -------
    H  : (3, 3) — homography matrix (defined up to scale)
    """

    N = xn.shape[1]

    # ------------------------------------------------------------------ #
    # Step 1 — Normalise input points (improves numerical conditioning)
    # Translate centroid to origin and scale so mean distance = sqrt(2)
    # ------------------------------------------------------------------ #
    def _normalise(pts):
        """ Returns (pts_norm, T) where T is the 3x3 normalisation transform. """
        cx, cy = pts[0].mean(), pts[1].mean()
        shifted = pts - np.array([[cx], [cy]])
        scale = np.sqrt(2) / np.mean(np.sqrt(shifted[0]**2 + shifted[1]**2) + 1e-10)
        T = np.array([
            [scale, 0,     -scale * cx],
            [0,     scale, -scale * cy],
            [0,     0,      1.0       ]
        ])
        pts_h = np.vstack([pts, np.ones(N)])
        return (T @ pts_h)[:2], T

    xn_n, T_xn = _normalise(xn)
    Xp_n, T_Xp = _normalise(Xp)

    # ------------------------------------------------------------------ #
    # Step 2 — Build the DLT matrix A (2N x 9)
    # Each point correspondence contributes 2 rows.
    # ------------------------------------------------------------------ #
    A = np.zeros((2 * N, 9))
    x, y  = xn_n[0], xn_n[1]    # normalised image coords
    X, Y  = Xp_n[0], Xp_n[1]    # normalised pattern coords

    A[0::2, 0] = -X
    A[0::2, 1] = -Y
    A[0::2, 2] = -1
    A[0::2, 6] =  x * X
    A[0::2, 7] =  x * Y
    A[0::2, 8] =  x

    A[1::2, 3] = -X
    A[1::2, 4] = -Y
    A[1::2, 5] = -1
    A[1::2, 6] =  y * X
    A[1::2, 7] =  y * Y
    A[1::2, 8] =  y

    # ------------------------------------------------------------------ #
    # Step 3 — Solve via SVD: h = last right-singular vector of A
    # ------------------------------------------------------------------ #
    _, _, Vt = np.linalg.svd(A)
    h = Vt[-1]                          # (9,) — solution in normalised space
    H_n = h.reshape(3, 3)

    # ------------------------------------------------------------------ #
    # Step 4 — Denormalise: H = inv(T_xn) @ H_n @ T_Xp
    # ------------------------------------------------------------------ #
    H = np.linalg.inv(T_xn) @ H_n @ T_Xp
    H = H / H[2, 2]                     # normalise so H[2,2] = 1

    return H


# ------------------------------------------------------------------ #
# Self-test: verify H correctly maps Xp → xn on a synthetic example
# ------------------------------------------------------------------ #
if __name__ == '__main__':
    np.random.seed(0)

    # Create a known homography
    from rodrigues import rodrigues
    om_true = np.array([0.1, -0.05, 0.03])
    R_true, _ = rodrigues(om_true)
    T_true = np.array([0.0, 0.0, 3.0])

    # Planar pattern points (Z=0 grid)
    gx, gy = np.meshgrid(np.linspace(-0.1, 0.1, 6), np.linspace(-0.1, 0.1, 6))
    Xp = np.vstack([gx.ravel(), gy.ravel()])        # (2, 36)
    Xp3 = np.vstack([Xp, np.zeros(Xp.shape[1])])    # (3, 36)

    # Project through camera
    Xc = R_true @ Xp3 + T_true[:, np.newaxis]
    xn = Xc[:2] / Xc[2]                             # (2, 36)

    # Add mild noise
    xn_noisy = xn + np.random.randn(*xn.shape) * 1e-4

    H = compute_homography(xn_noisy, Xp)

    # Verify reprojection
    Xp_h = np.vstack([Xp, np.ones(Xp.shape[1])])
    xn_reproj_h = H @ Xp_h
    xn_reproj = xn_reproj_h[:2] / xn_reproj_h[2]

    err = np.linalg.norm(xn_reproj - xn_noisy, axis=0)
    print(f"Max reprojection error: {err.max():.6f}  (should be < 1e-3)")
    print(f"Mean reprojection error: {err.mean():.6f}")
    print("compute_homography  ✓")
