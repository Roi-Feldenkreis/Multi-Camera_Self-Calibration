import numpy as np
from RadialDistortions.comp_distortion_oulu import comp_distortion_oulu


def normalize(x_kk, fc=None, cc=None, kc=None, alpha_c=0.0):
    """
    Compute normalized image coordinates from pixel coordinates.

    Parameters
    ----------
    x_kk    : (2, N) — pixel feature locations
    fc      : (2,)   — focal length [fx, fy]  (default [1, 1])
    cc      : (2,)   — principal point [cx, cy] (default [0, 0])
    kc      : (5,)   — distortion coefficients [k1,k2,p1,p2,k3] (default zeros)
    alpha_c : float  — skew coefficient (default 0)

    Returns
    -------
    xn : (2, N) — normalized (undistorted) image coordinates
    """

    # Default intrinsic parameters (mirrors MATLAB's nargin cascade)
    if fc is None:
        fc = np.array([1.0, 1.0])
    if cc is None:
        cc = np.array([0.0, 0.0])
    if kc is None:
        kc = np.zeros(5)

    fc = np.asarray(fc, dtype=float)
    cc = np.asarray(cc, dtype=float)
    kc = np.asarray(kc, dtype=float)

    # Step 1 — subtract principal point and divide by focal length
    x_distort = np.array([
        (x_kk[0, :] - cc[0]) / fc[0],
        (x_kk[1, :] - cc[1]) / fc[1]
    ])

    # Step 2 — undo skew along the x-axis
    x_distort[0, :] -= alpha_c * x_distort[1, :]

    # Step 3 — compensate for lens distortion (skipped if kc is all zeros)
    if np.linalg.norm(kc) != 0:
        xn = comp_distortion_oulu(x_distort, kc)
    else:
        xn = x_distort

    return xn


if __name__ == '__main__':
    np.random.seed(0)

    # Synthetic camera parameters
    fc      = np.array([800.0, 800.0])
    cc      = np.array([320.0, 240.0])
    kc      = np.array([0.1, -0.05, 0.001, 0.002, 0.0])
    alpha_c = 0.01

    # Random pixel points
    x_kk = np.random.uniform(0, 640, (2, 10))
    x_kk[1, :] = np.random.uniform(0, 480, 10)

    xn = normalize(x_kk, fc, cc, kc, alpha_c)
    print("Pixel coords (first 3):\n", x_kk[:, :3])
    print("Normalized coords (first 3):\n", xn[:, :3])

    # Sanity check: with no distortion and identity intrinsics,
    # normalize should be a pure shift + scale — verify on a known point
    x_test = np.array([[fc[0] + cc[0]], [fc[1] + cc[1]]])   # pixel (fx+cx, fy+cy)
    xn_test = normalize(x_test, fc, cc, kc=None, alpha_c=0)
    assert np.allclose(xn_test, [[1.0], [1.0]]), "Identity check failed"
    print("\nIdentity check passed ✓")
