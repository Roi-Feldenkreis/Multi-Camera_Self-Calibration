import numpy as np


def apply_distortion(x: np.ndarray, k: np.ndarray):
    """
    Apply radial and tangential lens distortion to normalized image coordinates.

    Parameters
    ----------
    x      : (2, N) undistorted normalized point coordinates
    k      : distortion coefficients — k1, k2, p1, p2, k3  (1–5 elements)

    Returns
    -------
    xd     : (2, N) distorted point coordinates
    dxddk  : (2N, len(k)) Jacobian of xd w.r.t. k
    """

    # Pad k to length 5 if a shorter model is used
    length_k = len(k)
    k = np.concatenate([k, np.zeros(max(0, 5 - length_k))])

    n = x.shape[1]

    # Radius powers — measure distance of each point from the image centre
    r2 = x[0, :] ** 2 + x[1, :] ** 2
    r4 = r2 ** 2
    r6 = r2 ** 3

    # Radial scale factor: cdist = 1 + k1*r^2 + k2*r^4 + k3*r^6
    cdist = 1 + k[0] * r2 + k[1] * r4 + k[4] * r6
    dcdistdk = np.column_stack([r2, r4, np.zeros(n), np.zeros(n), r6])  # (N, 5)

    # Apply radial distortion: xd1 = x * cdist
    xd1 = x * cdist[np.newaxis, :]
    dxd1dk = np.zeros((2 * n, 5))
    dxd1dk[0::2, :] = x[0, :][:, np.newaxis] * dcdistdk
    dxd1dk[1::2, :] = x[1, :][:, np.newaxis] * dcdistdk

    # Tangential distortion terms (p1=k[2], p2=k[3])
    a1 = 2.0 * x[0, :] * x[1, :]       # 2*x*y
    a2 = r2 + 2.0 * x[0, :] ** 2       # r^2 + 2*x^2
    a3 = r2 + 2.0 * x[1, :] ** 2       # r^2 + 2*y^2

    delta_x = np.array([
        k[2] * a1 + k[3] * a2,
        k[2] * a3 + k[3] * a1
    ])
    ddelta_xdk = np.zeros((2 * n, 5))
    ddelta_xdk[0::2, 2] = a1
    ddelta_xdk[0::2, 3] = a2
    ddelta_xdk[1::2, 2] = a3
    ddelta_xdk[1::2, 3] = a1

    # Combine radial + tangential
    xd = xd1 + delta_x

    # Jacobian — trim to original k length
    dxddk = dxd1dk + ddelta_xdk
    if length_k < 5:
        dxddk = dxddk[:, :length_k]

    return xd, dxddk


if __name__ == "__main__":
    np.random.seed(42)

    n = 10
    x = np.random.randn(2, n)
    k = np.array([0.1, 0.05, 0.01, 0.02, 0.001])

    xd, dxddk = apply_distortion(x, k)

    # Perturb k and check that dxddk predicts the change accurately.
    # NOTE: reshape order='F' matches MATLAB's column-major reshape.
    eps = 1e-5
    dk = eps * np.random.randn(len(k))
    xd2, _ = apply_distortion(x, k + dk)
    xd_pred = xd + (dxddk @ dk).reshape(2, n, order='F')

    ratio = np.linalg.norm(xd2 - xd) / np.linalg.norm(xd2 - xd_pred)
    print(f"Jacobian check ratio (should be >> 1): {ratio:.1f}  ✓" if ratio > 100
          else f"Jacobian check ratio: {ratio:.2f}")
