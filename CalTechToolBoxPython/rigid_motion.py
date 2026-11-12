import numpy as np
from CalTechToolBoxPython.rodrigues import rodrigues


def rigid_motion(X, om=None, T=None):
    """
    Apply a rigid body transformation Y = R*X + T, where R = rodrigues(om).

    Parameters
    ----------
    X  : (3, N) — 3D points in the world coordinate frame
    om : (3,)   — rotation vector (default zeros → identity)
    T  : (3,)   — translation vector (default zeros)

    Returns
    -------
    Y      : (3, N)   — 3D points in the camera coordinate frame
    dYdom  : (3N, 3)  — Jacobian of Y w.r.t. om
    dYdT   : (3N, 3)  — Jacobian of Y w.r.t. T

    Notes
    -----
    The 3N rows of the Jacobians:
    ordering: [Y[:,0], Y[:,1], ...] interleaved as
    [Y[0,0], Y[1,0], Y[2,0], Y[0,1], Y[1,1], Y[2,1], ...]
    i.e. rows 0,3,6,... → x-coords; rows 1,4,7,... → y-coords; 2,5,8,... → z-coords.

    dYdom is computed via the chain rule:
        dYdom = dYdR (3N×9) @ dRdom (9×3)
    where dRdom uses column-major R flattening (from rodrigues.py).
    """

    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(3, 1)

    # Default arguments
    if om is None:
        om = np.zeros(3)
    if T is None:
        T = np.zeros(3)

    om = np.asarray(om, dtype=float).ravel()
    T  = np.asarray(T,  dtype=float).ravel()

    n = X.shape[1]

    # Step 1 — rotation matrix and its Jacobian w.r.t. om
    # dRdom is (9, 3), rows indexed by R flattened column-major (F-order)
    R, dRdom = rodrigues(om)

    # Step 2 — apply rigid transformation: Y = R @ X + T (broadcast T)
    Y = R @ X + T[:, np.newaxis]           # (3, N)

    # Step 3 — Jacobian of Y w.r.t. R (column-major R flattening)
    # dYdR[3k,   [0,3,6]] = X[:,k]   (d Y[0,k] / d R, column-major)
    # dYdR[3k+1, [1,4,7]] = X[:,k]   (d Y[1,k] / d R, column-major)
    # dYdR[3k+2, [2,5,8]] = X[:,k]   (d Y[2,k] / d R, column-major)
    dYdR = np.zeros((3 * n, 9))
    dYdR[0::3, 0::3] = X.T   # rows: all x-outputs; cols: R[0,0], R[0,1], R[0,2] (F-order 0,3,6)
    dYdR[1::3, 1::3] = X.T   # rows: all y-outputs; cols: R[1,0], R[1,1], R[1,2] (F-order 1,4,7)
    dYdR[2::3, 2::3] = X.T   # rows: all z-outputs; cols: R[2,0], R[2,1], R[2,2] (F-order 2,5,8)

    # Step 4 — Jacobian of Y w.r.t. T (identity block, broadcast across points)
    dYdT = np.zeros((3 * n, 3))
    dYdT[0::3, 0] = 1.0
    dYdT[1::3, 1] = 1.0
    dYdT[2::3, 2] = 1.0

    # Step 5 — chain rule: dY/dom = dY/dR @ dR/dom
    dYdom = dYdR @ dRdom                   # (3N, 9) @ (9, 3) → (3N, 3)

    return Y, dYdom, dYdT


# ------------------------------------------------------------------ #
# Self-tests (mirror the test pattern used in rodrigues.py)
# ------------------------------------------------------------------ #
if __name__ == '__main__':
    np.random.seed(0)

    print("--- Test 1: dYdom Jacobian ---")
    X   = np.random.randn(3, 10)
    om  = np.random.randn(3)
    T   = np.random.randn(3)
    dom = np.random.randn(3) / 1e6

    Y1, dYdom, dYdT = rigid_motion(X, om, T)
    Y2, _, _        = rigid_motion(X, om + dom, T)

    # Y_pred = Y1 + reshape(dYdom*dom, 3, N)
    # reshape is column-major (F-order) to match the Jacobian row ordering
    Y_pred = Y1 + (dYdom @ dom).reshape(3, 10, order='F')
    gain = np.linalg.norm(Y2 - Y1) / np.linalg.norm(Y2 - Y_pred)
    print(f"  gain (should be >> 1): {gain:.0f}  ✓")

    print("\n--- Test 2: dYdT Jacobian ---")
    dT = np.random.randn(3) / 1e6
    Y3, _, _ = rigid_motion(X, om, T + dT)
    Y_pred_T = Y1 + (dYdT @ dT).reshape(3, 10, order='F')
    gain_T = np.linalg.norm(Y3 - Y1) / np.linalg.norm(Y3 - Y_pred_T)
    print(f"  gain (should be >> 1): {gain_T:.0f}  ✓")

    print("\n--- Test 3: known rotation (90° around Z) ---")
    om_z90 = np.array([0, 0, np.pi / 2])
    X_test = np.array([[1, 0, 0],
                       [0, 1, 0],
                       [0, 0, 1]], dtype=float)  # identity columns
    Y_test, _, _ = rigid_motion(X_test, om_z90, np.zeros(3))
    print(f"  X col 0 [1,0,0] → Y: {np.round(Y_test[:,0], 4)}  (expect [0,1,0])")
    print(f"  X col 1 [0,1,0] → Y: {np.round(Y_test[:,1], 4)}  (expect [-1,0,0])")
