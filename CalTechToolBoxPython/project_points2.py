import numpy as np
from CalTechToolBoxPython.rigid_motion import rigid_motion


def project_points2(X, om=None, T=None, f=None, c=None, k=None, alpha=0.0):
    """
    Project 3D world points onto the image plane with full distortion model.

    Pipeline:
      1. Rigid motion:   Xc = R*X + T         (world → camera frame)
      2. Pinhole:        x  = Xc[0:2] / Xc[2] (perspective divide)
      3. Distortion:     xd = radial + tangential warp
      4. Skew:           xd3[0] += alpha * xd[1]
      5. Pixel coords:   xp = f * xd3 + c

    Parameters
    ----------
    X     : (3, N)      — 3D world points
    om    : (3,)        — rotation vector          (default zeros)
    T     : (3,)        — translation vector       (default zeros)
    f     : (2,) or float — focal length [fx, fy] (default [1,1])
    c     : (2,)        — principal point [cx, cy] (default zeros)
    k     : (5,)        — distortion coefficients  (default zeros)
    alpha : float       — skew coefficient         (default 0)

    Returns
    -------
    xp        : (2, N)   — projected pixel coordinates
    dxpdom    : (2N, 3)  — Jacobian w.r.t. om
    dxpdT     : (2N, 3)  — Jacobian w.r.t. T
    dxpdf     : (2N, 2) or (2N, 1) — Jacobian w.r.t. f
    dxpdc     : (2N, 2)  — Jacobian w.r.t. c
    dxpdk     : (2N, 5)  — Jacobian w.r.t. k
    dxpdalpha : (2N, 1)  — Jacobian w.r.t. alpha

    Notes
    -----
    All Jacobian row ordering uses MATLAB column-major (F-order) point
    interleaving: rows [0,1] → point 0, rows [2,3] → point 1, etc.
    Apply predictions as: x_pred = xp + (J @ dp).reshape(2, N, order='F')
    """

    # --- Defaults (mirrors MATLAB nargin cascade) ---
    if om    is None: om    = np.zeros(3)
    if T     is None: T     = np.zeros(3)
    if f     is None: f     = np.ones(2)
    if c     is None: c     = np.zeros(2)
    if k     is None: k     = np.zeros(5)

    X  = np.asarray(X,  dtype=float)
    om = np.asarray(om, dtype=float).ravel()
    T  = np.asarray(T,  dtype=float).ravel()
    f  = np.asarray(f,  dtype=float)
    c  = np.asarray(c,  dtype=float).ravel()
    k  = np.asarray(k,  dtype=float).ravel()
    if k.size < 5:
        k = np.concatenate([k, np.zeros(5 - k.size)])

    n = X.shape[1]

    # ------------------------------------------------------------------ #
    # Step 1 — Rigid motion: bring points into the camera frame
    # ------------------------------------------------------------------ #
    Y, dYdom, dYdT = rigid_motion(X, om, T)   # Y:(3,N), dYdom:(3N,3), dYdT:(3N,3)

    # ------------------------------------------------------------------ #
    # Step 2 — Perspective (pinhole) divide: x = [Xc/Zc ; Yc/Zc]
    # ------------------------------------------------------------------ #
    inv_Z = 1.0 / Y[2, :]                     # (N,)
    x = Y[:2, :] * inv_Z[np.newaxis, :]       # (2, N)

    # Jacobian of x w.r.t. Y: quotient rule applied per-point
    # bb[k,:] = -x[0,k] / Z[k]  (scales the Z-row contribution for x[0])
    # cc[k,:] = -x[1,k] / Z[k]  (scales the Z-row contribution for x[1])
    bb = (-x[0, :] * inv_Z)[:, np.newaxis] * np.ones((1, 3))  # (N, 3)
    cc = (-x[1, :] * inv_Z)[:, np.newaxis] * np.ones((1, 3))  # (N, 3)

    # dxdom, dxdT: (2N, 3), interleaved rows [x0,y0, x1,y1, ...]
    dxdom = np.zeros((2 * n, 3))
    dxdom[0::2, :] = inv_Z[:, np.newaxis] * dYdom[0::3, :] + bb * dYdom[2::3, :]
    dxdom[1::2, :] = inv_Z[:, np.newaxis] * dYdom[1::3, :] + cc * dYdom[2::3, :]

    dxdT = np.zeros((2 * n, 3))
    dxdT[0::2, :] = inv_Z[:, np.newaxis] * dYdT[0::3, :] + bb * dYdT[2::3, :]
    dxdT[1::2, :] = inv_Z[:, np.newaxis] * dYdT[1::3, :] + cc * dYdT[2::3, :]

    # ------------------------------------------------------------------ #
    # Step 3 — Distortion
    # ------------------------------------------------------------------ #

    # Radius powers
    r2 = x[0, :] ** 2 + x[1, :] ** 2                         # (N,)
    r4 = r2 ** 2
    r6 = r2 ** 3

    dr2dom = (2 * x[0, :, np.newaxis] * dxdom[0::2, :] +
              2 * x[1, :, np.newaxis] * dxdom[1::2, :])       # (N, 3)
    dr2dT  = (2 * x[0, :, np.newaxis] * dxdT[0::2, :]  +
              2 * x[1, :, np.newaxis] * dxdT[1::2, :])        # (N, 3)

    dr4dom = 2 * r2[:, np.newaxis] * dr2dom
    dr4dT  = 2 * r2[:, np.newaxis] * dr2dT

    dr6dom = 3 * (r2 ** 2)[:, np.newaxis] * dr2dom
    dr6dT  = 3 * (r2 ** 2)[:, np.newaxis] * dr2dT

    # --- Radial distortion ---
    cdist = 1 + k[0] * r2 + k[1] * r4 + k[4] * r6            # (N,)

    dcdistdom = k[0] * dr2dom + k[1] * dr4dom + k[4] * dr6dom # (N, 3)
    dcdistdT  = k[0] * dr2dT  + k[1] * dr4dT  + k[4] * dr6dT # (N, 3)
    dcdistdk  = np.column_stack([r2, r4, np.zeros(n), np.zeros(n), r6])  # (N, 5)

    xd1 = x * cdist[np.newaxis, :]                            # (2, N)

    # Jacobian of xd1
    coeff = np.repeat(cdist, 2)[:, np.newaxis]                # (2N, 1)  cdist broadcast

    dxd1dom = np.zeros((2 * n, 3))
    dxd1dom[0::2, :] = x[0, :, np.newaxis] * dcdistdom
    dxd1dom[1::2, :] = x[1, :, np.newaxis] * dcdistdom
    dxd1dom += coeff * dxdom

    dxd1dT = np.zeros((2 * n, 3))
    dxd1dT[0::2, :] = x[0, :, np.newaxis] * dcdistdT
    dxd1dT[1::2, :] = x[1, :, np.newaxis] * dcdistdT
    dxd1dT += coeff * dxdT

    dxd1dk = np.zeros((2 * n, 5))
    dxd1dk[0::2, :] = x[0, :, np.newaxis] * dcdistdk
    dxd1dk[1::2, :] = x[1, :, np.newaxis] * dcdistdk

    # --- Tangential distortion ---
    a1 = 2.0 * x[0, :] * x[1, :]          # (N,)
    a2 = r2 + 2.0 * x[0, :] ** 2          # (N,)
    a3 = r2 + 2.0 * x[1, :] ** 2          # (N,)

    delta_x = np.array([
        k[2] * a1 + k[3] * a2,
        k[2] * a3 + k[3] * a1
    ])                                     # (2, N)

    aa = (2*k[2]*x[1,:] + 6*k[3]*x[0,:])[:, np.newaxis]  # (N,1)
    bb = (2*k[2]*x[0,:] + 2*k[3]*x[1,:])[:, np.newaxis]  # (N,1)
    cc = (6*k[2]*x[1,:] + 2*k[3]*x[0,:])[:, np.newaxis]  # (N,1)

    ddelta_xdom = np.zeros((2 * n, 3))
    ddelta_xdom[0::2, :] = aa * dxdom[0::2, :] + bb * dxdom[1::2, :]
    ddelta_xdom[1::2, :] = bb * dxdom[0::2, :] + cc * dxdom[1::2, :]

    ddelta_xdT = np.zeros((2 * n, 3))
    ddelta_xdT[0::2, :] = aa * dxdT[0::2, :] + bb * dxdT[1::2, :]
    ddelta_xdT[1::2, :] = bb * dxdT[0::2, :] + cc * dxdT[1::2, :]

    ddelta_xdk = np.zeros((2 * n, 5))
    ddelta_xdk[0::2, 2] = a1
    ddelta_xdk[0::2, 3] = a2
    ddelta_xdk[1::2, 2] = a3
    ddelta_xdk[1::2, 3] = a1

    # Combine radial + tangential
    xd2     = xd1 + delta_x                # (2, N)
    dxd2dom = dxd1dom + ddelta_xdom
    dxd2dT  = dxd1dT  + ddelta_xdT
    dxd2dk  = dxd1dk  + ddelta_xdk

    # ------------------------------------------------------------------ #
    # Step 4 — Skew: xd3[0] = xd2[0] + alpha * xd2[1]
    # ------------------------------------------------------------------ #
    xd3 = np.array([
        xd2[0, :] + alpha * xd2[1, :],
        xd2[1, :]
    ])                                     # (2, N)

    dxd3dom = np.zeros((2 * n, 3))
    dxd3dom[0::2, :] = dxd2dom[0::2, :] + alpha * dxd2dom[1::2, :]
    dxd3dom[1::2, :] = dxd2dom[1::2, :]

    dxd3dT = np.zeros((2 * n, 3))
    dxd3dT[0::2, :] = dxd2dT[0::2, :] + alpha * dxd2dT[1::2, :]
    dxd3dT[1::2, :] = dxd2dT[1::2, :]

    dxd3dk = np.zeros((2 * n, 5))
    dxd3dk[0::2, :] = dxd2dk[0::2, :] + alpha * dxd2dk[1::2, :]
    dxd3dk[1::2, :] = dxd2dk[1::2, :]

    dxd3dalpha = np.zeros((2 * n, 1))
    dxd3dalpha[0::2, 0] = xd2[1, :]       # d(xd3[0])/d(alpha) = xd2[1]

    # ------------------------------------------------------------------ #
    # Step 5 — Pixel coordinates: xp = f * xd3 + c
    # Two sub-cases: f is a 2-vector (separate fx, fy) or a scalar
    # ------------------------------------------------------------------ #
    if f.size > 1:
        xp = xd3 * f[:, np.newaxis] + c[:, np.newaxis]        # (2, N)
        fcoeff = np.repeat(f, n)[:, np.newaxis]                # (2N, 1)

        dxpdom    = fcoeff * dxd3dom
        dxpdT     = fcoeff * dxd3dT
        dxpdk     = fcoeff * dxd3dk
        dxpdalpha = fcoeff * dxd3dalpha

        dxpdf = np.zeros((2 * n, 2))
        dxpdf[0::2, 0] = xd3[0, :]        # d(xp[0])/d(fx) = xd3[0]
        dxpdf[1::2, 1] = xd3[1, :]        # d(xp[1])/d(fy) = xd3[1]
    else:
        fval = float(f)
        xp = fval * xd3 + c[:, np.newaxis]
        dxpdom    = fval * dxd3dom
        dxpdT     = fval * dxd3dT
        dxpdk     = fval * dxd3dk
        dxpdalpha = fval * dxd3dalpha
        dxpdf     = xd3.ravel(order='F')[:, np.newaxis]       # (2N, 1)

    dxpdc = np.zeros((2 * n, 2))
    dxpdc[0::2, 0] = 1.0
    dxpdc[1::2, 1] = 1.0

    return xp, dxpdom, dxpdT, dxpdf, dxpdc, dxpdk, dxpdalpha


# ------------------------------------------------------------------ #
# Self-tests — mirror the test block at the end of project_points2.m
# ------------------------------------------------------------------ #
if __name__ == '__main__':
    np.random.seed(0)

    n  = 10
    X  = 10  * np.random.randn(3, n);  X[2, :] = np.abs(X[2, :]) + 5
    om = np.random.randn(3)
    T  = np.array([10*np.random.randn(), 10*np.random.randn(), 40.0])
    f  = 1000 * np.random.rand(2)
    c  = 100  * np.random.randn(2)
    k  = 0.5  * np.random.randn(5); k[0] *= 0.1; k[1] *= 0.01
    alpha = 0.01 * np.random.randn()

    xp, dxpdom, dxpdT, dxpdf, dxpdc, dxpdk, dxpdalpha = \
        project_points2(X, om, T, f, c, k, alpha)

    def gain(J, dp, perturb_fn):
        xp2, *_ = perturb_fn()
        xp_pred = xp + (J @ dp).reshape(2, n, order='F')
        return np.linalg.norm(xp2 - xp) / np.linalg.norm(xp2 - xp_pred)

    eps = 1e-6

    dom = eps * np.random.randn(3)
    g1 = gain(dxpdom, dom, lambda: project_points2(X, om+dom, T, f, c, k, alpha))
    print(f"dxpdom  gain: {g1:.0f}  ✓")

    dT = eps * np.random.randn(3)
    g2 = gain(dxpdT, dT, lambda: project_points2(X, om, T+dT, f, c, k, alpha))
    print(f"dxpdT   gain: {g2:.0f}  ✓")

    df = eps * np.random.randn(2)
    g3 = gain(dxpdf, df, lambda: project_points2(X, om, T, f+df, c, k, alpha))
    print(f"dxpdf   gain: {g3:.0f}  ✓")

    dc = eps * np.random.randn(2)
    g4 = gain(dxpdc, dc, lambda: project_points2(X, om, T, f, c+dc, k, alpha))
    print(f"dxpdc   gain: {g4:.0f}  ✓")

    dk = eps * np.random.randn(5)
    g5 = gain(dxpdk, dk, lambda: project_points2(X, om, T, f, c, k+dk, alpha))
    print(f"dxpdk   gain: {g5:.0f}  ✓")

    dalpha = np.array([eps * np.random.randn()])
    g6 = gain(dxpdalpha, dalpha, lambda: project_points2(X, om, T, f, c, k, alpha+dalpha[0]))
    print(f"dxpdalpha gain: {g6:.0f}  ✓")
