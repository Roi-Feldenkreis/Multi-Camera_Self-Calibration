import numpy as np
from CalTechToolBoxPython.normalize import normalize
from CalTechToolBoxPython.rodrigues import rodrigues
from CalTechToolBoxPython.compute_homography import compute_homography


def compute_extrinsic_init(x_kk, X_kk, fc=None, cc=None, kc=None, alpha_c=0.0):
    """
    Compute an initial estimate of extrinsic parameters (om, T, R) for a
    calibration image, given 2D image points and their 3D correspondences.

    Works for both planar and non-planar 3D structures.

    Parameters
    ----------
    x_kk    : (2, N) — detected 2D feature locations in pixel coordinates
    X_kk    : (3, N) — corresponding 3D world points
    fc      : (2,)   — focal lengths [fx, fy]        (default [1,1])
    cc      : (2,)   — principal point [cx, cy]      (default [0,0])
    kc      : (5,)   — distortion coefficients       (default zeros)
    alpha_c : float  — skew coefficient              (default 0)

    Returns
    -------
    omckk : (3,)   — rotation vector
    Tckk  : (3,1)  — translation vector
    Rckk  : (3,3)  — rotation matrix  (= rodrigues(omckk))
    """

    # Default intrinsics
    if fc      is None: fc  = np.ones(2)
    if cc      is None: cc  = np.zeros(2)
    if kc      is None: kc  = np.zeros(5)
    fc  = np.asarray(fc,  dtype=float)
    cc  = np.asarray(cc,  dtype=float)
    kc  = np.asarray(kc,  dtype=float)
    X_kk = np.asarray(X_kk, dtype=float)

    Np = X_kk.shape[1]

    # ------------------------------------------------------------------ #
    # Step 1 — Normalise image points
    xn = normalize(x_kk, fc, cc, kc, alpha_c)    # (2, N)

    # ------------------------------------------------------------------ #
    # Step 2 — Test for planarity of the 3D structure
    # ------------------------------------------------------------------ #
    X_mean = X_kk.mean(axis=1, keepdims=True)     # (3, 1)
    Y = X_kk - X_mean                             # centred points (3, N)
    YY = Y @ Y.T                                  # (3, 3) scatter matrix
    U, S, Vt = np.linalg.svd(YY)
    r = S[2] / S[1] if S[1] > 0 else 0.0         # flatness ratio

    # ------------------------------------------------------------------ #
    # PLANAR CASE  (r < 1e-3  OR  fewer than 5 points)
    # ------------------------------------------------------------------ #
    if r < 1e-3 or Np < 5:

        # --- 2a: Rotate the plane so it lies in Z=0 ---
        R_transform = Vt                           # rows of Vt are eigenvectors
        if np.linalg.norm(R_transform[:2, 2]) < 1e-6:
            R_transform = np.eye(3)
        if np.linalg.det(R_transform) < 0:
            R_transform = -R_transform

        T_transform = -(R_transform @ X_mean)     # (3, 1)
        X_new = R_transform @ X_kk + T_transform  # (3, N)  — now Z≈0

        # --- 2b: Compute the planar homography ---
        # compute_homography maps 2D pattern coords → normalised image coords
        H = compute_homography(xn, X_new[:2, :])  # (3, 3)

        # Normalise H so its first two columns have unit-average norm
        sc = (np.linalg.norm(H[:, 0]) + np.linalg.norm(H[:, 1])) / 2.0
        H = H / sc

        # --- 2c: Orthogonalise the rotation columns from H ---

        # Gram-Schmidt on the first two columns, then take their cross product.
        u1 = H[:, 0]
        u1 = u1 / np.linalg.norm(u1)
        u2 = H[:, 1] - np.dot(u1, H[:, 1]) * u1
        u2 = u2 / np.linalg.norm(u2)
        u3 = np.cross(u1, u2)
        RRR = np.column_stack([u1, u2, u3])       # (3, 3) — near-orthogonal

        omckk, _ = rodrigues(RRR)
        Rckk, _  = rodrigues(omckk)
        Tckk     = H[:, 2].reshape(3, 1)

        # --- 2d: Compose with the pre-transform ---

        Tckk = Tckk + Rckk @ T_transform          # (3, 1)
        Rckk = Rckk @ R_transform                 # (3, 3)

        # Re-project through rodrigues to enforce exact SO(3) membership
        omckk, _ = rodrigues(Rckk)
        Rckk, _  = rodrigues(omckk)

    # ------------------------------------------------------------------ #
    # NON-PLANAR CASE  — Direct Linear Transform (DLT)
    # ------------------------------------------------------------------ #
    else:

        # Build the (2N × 12) DLT matrix J
        # Each point contributes 2 equations relating [R | T] to xn
        J = np.zeros((2 * Np, 12))

        xX = xn[0:1, :] * X_kk    # (3, N)  — x * X, x * Y, x * Z
        yX = xn[1:2, :] * X_kk    # (3, N)  — y * X, y * Y, y * Z

        # Layout: J * [R(:); T] = 0
        J[0::2, [0, 3, 6]] = -X_kk.T   # −X columns
        J[1::2, [1, 4, 7]] =  X_kk.T   # +X columns
        J[0::2, [2, 5, 8]] =  xX.T     # x*X columns
        J[1::2, [2, 5, 8]] = -yX.T     # −y*X columns
        J[0::2, 11]        =  xn[0]    # x column
        J[1::2, 11]        = -xn[1]    # −y column
        J[0::2, 9]         = -1.0
        J[1::2, 10]        =  1.0

        # Solve JᵀJ via SVD — last right-singular vector is the solution
        _, _, Vsvd = np.linalg.svd(J.T @ J)
        v12 = Vsvd[-1]                 # (12,) — last row of Vt

        RR = v12[:9].reshape(3, 3)
        if np.linalg.det(RR) < 0:
            v12 = -v12
            RR  = -RR

        # Project RR onto SO(3) via SVD
        Ur, _, Vrt = np.linalg.svd(RR)
        Rckk = Ur @ Vrt

        sc   = np.linalg.norm(v12[:9]) / np.linalg.norm(Rckk)
        Tckk = (v12[9:12] / sc).reshape(3, 1)

        omckk, _ = rodrigues(Rckk)
        Rckk,  _ = rodrigues(omckk)

    return omckk, Tckk, Rckk


# ------------------------------------------------------------------ #
# Self-test: round-trip on a synthetic planar scene and a 3D scene
# ------------------------------------------------------------------ #
if __name__ == '__main__':
    from project_points2 import project_points2

    np.random.seed(42)

    # True camera parameters
    fc_true      = np.array([800.0, 800.0])
    cc_true      = np.array([320.0, 240.0])
    kc_true      = np.zeros(5)
    alpha_c_true = 0.0
    om_true      = np.array([0.05, -0.1, 0.02])
    T_true       = np.array([0.01, -0.02, 1.5])

    print("=== Test 1: Planar structure ===")
    gx, gy = np.meshgrid(np.linspace(-0.15, 0.15, 8),
                          np.linspace(-0.1,  0.1,  6))
    X_kk = np.vstack([gx.ravel(), gy.ravel(), np.zeros(gx.size)])
    x_kk, *_ = project_points2(X_kk, om_true, T_true,
                                fc_true, cc_true, kc_true, alpha_c_true)
    x_kk += np.random.randn(*x_kk.shape) * 0.3  # add 0.3px noise

    omckk, Tckk, Rckk = compute_extrinsic_init(
        x_kk, X_kk, fc_true, cc_true, kc_true, alpha_c_true)

    # Reproject with estimated pose and check error
    x_reproj, *_ = project_points2(X_kk, omckk, Tckk.ravel(),
                                    fc_true, cc_true, kc_true, alpha_c_true)
    err = np.linalg.norm(x_reproj - x_kk, axis=0)
    print(f"  Mean reprojection error: {err.mean():.3f} px")
    print(f"  Max  reprojection error: {err.max():.3f} px")
    assert err.mean() < 5.0, "Planar test failed"
    print("  ✓  Passed")

    print("\n=== Test 2: Non-planar structure ===")
    X_kk_3d = np.random.uniform(-0.15, 0.15, (3, 40))
    X_kk_3d[2] += np.random.uniform(0.0, 0.05, 40)  # varying depth → non-planar
    x_kk_3d, *_ = project_points2(X_kk_3d, om_true, T_true,
                                   fc_true, cc_true, kc_true, alpha_c_true)
    x_kk_3d += np.random.randn(*x_kk_3d.shape) * 0.3

    omckk3, Tckk3, Rckk3 = compute_extrinsic_init(
        x_kk_3d, X_kk_3d, fc_true, cc_true, kc_true, alpha_c_true)

    x_reproj3, *_ = project_points2(X_kk_3d, omckk3, Tckk3.ravel(),
                                     fc_true, cc_true, kc_true, alpha_c_true)
    err3 = np.linalg.norm(x_reproj3 - x_kk_3d, axis=0)
    print(f"  Mean reprojection error: {err3.mean():.3f} px")
    print(f"  Max  reprojection error: {err3.max():.3f} px")
    assert err3.mean() < 10.0, "Non-planar test failed"
    print("  ✓  Passed")
