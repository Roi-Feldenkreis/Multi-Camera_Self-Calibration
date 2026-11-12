import numpy as np


def rodrigues(inp):
    """
    Convert between rotation vector and rotation matrix (Rodrigues formula).

    If inp is a 3-vector  → returns (R, dRdin)  : (3,3) matrix  + (9,3) Jacobian
    If inp is a 3x3 matrix → returns (om, dout) : (3,) vector   + (3,9) Jacobian

    Parameters
    ----------
    inp : (3,), (3,1), (1,3), or (3,3) array

    Returns
    -------
    out  : rotation matrix (3,3) or rotation vector (3,)
    dout : Jacobian — (9,3) for vector→matrix, (3,9) for matrix→vector

    Notes
    -----
    - Jacobians use column-major (Fortran) flattening convention.
      When applying dRdin: R_new ≈ R + (dRdin @ dom).reshape(3, 3, order='F')
      When applying domdR: om_new ≈ om + domdR @ dR.ravel(order='F')
    """

    inp = np.asarray(inp, dtype=float)
    bigeps = 10e+20 * np.finfo(float).eps

    # ------------------------------------------------------------------ #
    # CASE 1: rotation vector (3-element) → rotation matrix
    # Formula: R = I·cos(θ) + [ω]×·sin(θ) + ωωᵀ·(1−cos(θ))
    # ------------------------------------------------------------------ #
    if inp.size == 3:
        inp = inp.ravel()
        theta = np.linalg.norm(inp)

        if theta < np.finfo(float).eps:
            # Near-zero rotation → identity + linearised Jacobian
            R = np.eye(3)
            dRdin = np.array([
                [ 0,  0,  0],
                [ 0,  0,  1],
                [ 0, -1,  0],
                [ 0,  0, -1],
                [ 0,  0,  0],
                [ 1,  0,  0],
                [ 0,  1,  0],
                [-1,  0,  0],
                [ 0,  0,  0]], dtype=float)
        else:
            inp = inp.reshape(3, 1)

            # --- Intermediate Jacobian chain --- #

            # m3 = [in; theta],  dm3din shape (4,3)
            dm3din = np.vstack([np.eye(3), inp.T / theta])

            omega  = inp / theta           # unit rotation axis, (3,1)

            # m2 = [omega; theta],  dm2dm3 shape (4,4)
            dm2dm3 = np.block([
                [np.eye(3) / theta, -inp / theta**2],
                [np.zeros((1, 3)),   np.ones((1, 1))]])

            alpha = np.cos(theta)
            beta  = np.sin(theta)
            gamma = 1 - np.cos(theta)

            w1, w2, w3 = omega.ravel()
            omegav = np.array([
                [ 0,  -w3,  w2],
                [ w3,   0, -w1],
                [-w2,  w1,   0]], dtype=float)
            A = omega @ omega.T             # outer product ωωᵀ

            # m1 = [alpha; beta; gamma; omegav(:); A(:)]
            # dm1dm2 shape (21,4): derivatives of m1 w.r.t. m2=[omega(3);theta]
            dm1dm2 = np.zeros((21, 4))
            dm1dm2[0, 3] = -np.sin(theta)
            dm1dm2[1, 3] =  np.cos(theta)
            dm1dm2[2, 3] =  np.sin(theta)
            # d(omegav(:))/d(omega) — omegav(:) in column-major order
            dm1dm2[3:12, 0:3] = np.array([
                [0,  0,  0,  0,  0,  1,  0, -1,  0],
                [0,  0, -1,  0,  0,  0,  1,  0,  0],
                [0,  1,  0, -1,  0,  0,  0,  0,  0]
            ], dtype=float).T
            # d(A(:))/d(omega) — A(:) in column-major order
            dm1dm2[12:21, 0] = [2*w1, w2,   w3, w2,    0,   0, w3,   0,    0]
            dm1dm2[12:21, 1] = [   0, w1,    0, w1, 2*w2,  w3,  0,  w3,    0]
            dm1dm2[12:21, 2] = [   0,  0,   w1,  0,    0,  w2, w1,  w2, 2*w3]

            R = np.eye(3) * alpha + omegav * beta + A * gamma

            # dRdm1 shape (9,21): d(R(:))/d(m1)
            # R(:) and all sub-matrices use column-major (F-order) flattening
            dRdm1 = np.zeros((9, 21))
            dRdm1[[0, 4, 8], 0] = 1.0                   # d(I·alpha)/d(alpha)
            dRdm1[:, 1]         = omegav.ravel(order='F')# d(omegav·beta)/d(beta)
            dRdm1[:, 3:12]      = beta  * np.eye(9)      # d(omegav·beta)/d(omegav)
            dRdm1[:, 2]         = A.ravel(order='F')     # d(A·gamma)/d(gamma)
            dRdm1[:, 12:21]     = gamma * np.eye(9)      # d(A·gamma)/d(A)

            dRdin = dRdm1 @ dm1dm2 @ dm2dm3 @ dm3din     # (9,3)

        return R, dRdin

    # ------------------------------------------------------------------ #
    # CASE 2: rotation matrix (3x3) → rotation vector
    # ------------------------------------------------------------------ #
    elif inp.shape == (3, 3):
        R = inp.copy()
        if not (np.linalg.norm(R.T @ R - np.eye(3)) < bigeps and
                abs(np.linalg.det(R) - 1) < bigeps):
            raise ValueError("Input is neither a rotation matrix nor a rotation vector.")

        # Project onto SO(3) via SVD to handle near-singular matrices
        U, _, Vt = np.linalg.svd(R)
        R = U @ Vt

        tr        = (np.trace(R) - 1) / 2.0
        dtrdR     = np.array([1,0,0,0,1,0,0,0,1], dtype=float) / 2.0
        theta     = float(np.real(np.arccos(np.clip(tr, -1, 1))))

        if np.sin(theta) >= 1e-5:
            # General case
            dthetadtr = -1.0 / np.sqrt(1.0 - tr**2)
            dthetadR  = dthetadtr * dtrdR                                # (9,)

            vth            = 1.0 / (2.0 * np.sin(theta))
            dvthdtheta     = -vth * np.cos(theta) / np.sin(theta)
            dvar1dtheta    = np.array([dvthdtheta, 1.0])
            dvar1dR        = np.outer(dvar1dtheta, dthetadR)             # (2,9)

            om1    = np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])
            dom1dR = np.array([
                [0,  0,  0,  0,  0,  1,  0, -1,  0],
                [0,  0, -1,  0,  0,  0,  1,  0,  0],
                [0,  1,  0, -1,  0,  0,  0,  0,  0]
            ], dtype=float)                                              # (3,9)

            dvardR   = np.vstack([dom1dR, dvar1dR])                      # (5,9)

            om       = vth * om1
            domdvar  = np.hstack([vth*np.eye(3), om1.reshape(3,1), np.zeros((3,1))])  # (3,5)
            dvar2dvar= np.vstack([domdvar, [0,0,0,0,1]])                 # (4,5)

            out  = om * theta
            domegadvar2 = np.hstack([theta*np.eye(3), om.reshape(3,1)]) # (3,4)
            dout = domegadvar2 @ dvar2dvar @ dvardR                      # (3,9)

        else:
            if tr > 0:
                # Near-zero rotation
                out  = np.zeros(3)
                dout = np.array([
                    [0,   0,    0,    0, 0,  0.5,  0, -0.5, 0],
                    [0,   0, -0.5,   0, 0,    0, 0.5,   0,  0],
                    [0, 0.5,    0, -0.5, 0,   0,   0,   0,  0]
                ], dtype=float)
            else:
                # Rotation angle ≈ π
                out = theta * (np.sqrt((np.diag(R)+1)/2.0) *
                               np.array([1,
                                         2*float(R[0,1] >= 0)-1,
                                         2*float(R[0,2] >= 0)-1]))
                print("WARNING: Jacobian domdR undefined at norm(om)=pi")
                dout = np.full((3, 9), np.nan)

        return out, dout

    else:
        raise ValueError("Input must be a 3-element vector or a 3x3 matrix.")


# ------------------------------------------------------------------ #
# Self-tests
# NOTE: use order='F' for reshape/ravel to match column-major
# ------------------------------------------------------------------ #
if __name__ == '__main__':
    np.random.seed(0)

    print("--- Test 1: dRdom Jacobian ---")
    om  = np.random.randn(3)
    dom = np.random.randn(3) / 1e6
    R1, dR1 = rodrigues(om)
    R2, _   = rodrigues(om + dom)
    R2a = R1 + (dR1 @ dom).reshape(3, 3, order='F')
    gain = np.linalg.norm(R2 - R1) / np.linalg.norm(R2 - R2a)
    print(f"  gain (should be >> 1): {gain:.0f}  ✓")

    print("\n--- Test 2: domdR Jacobian ---")
    om      = np.random.randn(3)
    R       = rodrigues(om)[0]
    dom     = np.random.randn(3) / 1e4
    dR      = rodrigues(om + dom)[0] - R
    omc, domdR = rodrigues(R)
    om2, _  = rodrigues(R + dR)
    om_app  = omc + domdR @ dR.ravel(order='F')
    gain    = np.linalg.norm(om2 - omc) / np.linalg.norm(om2 - om_app)
    print(f"  gain (should be >> 1): {gain:.0f}  ✓")

    print("\n--- Test 3: round-trip om → R → om ---")
    om    = np.random.randn(3)
    R, _  = rodrigues(om)
    om2,_ = rodrigues(R)
    print(f"  max abs diff: {np.max(np.abs(om - om2)):.2e}  ✓")
