import numpy as np
from CalTechToolBoxPython.check_active_images import check_active_images
from CalTechToolBoxPython.rodrigues import rodrigues


def extract_parameters(solution, n_ima, active_images=None, solution_error=None):
    """
    Unpack the flat optimisation solution vector into intrinsic and
    extrinsic camera parameters.

    The solution vector layout (1-indexed MATLAB → 0-indexed Python):
    ┌─────────────────────────────────────────────────────────┐
    │ Indices 0–1   : fc       (2 focal lengths)              │
    │ Indices 2–3   : cc       (principal point)              │
    │ Index  4      : alpha_c  (skew)                         │
    │ Indices 5–9   : kc       (5 distortion coefficients)    │
    │ Indices 10–14 : reserved (padding to 15 intrinsics)     │
    │ Indices 15 + 6*(k)   + 0:2  : omc for image k          │
    │ Indices 15 + 6*(k)   + 3:5  : Tc  for image k          │
    └─────────────────────────────────────────────────────────┘
    Total length: 15 + 6 * n_ima

    Parameters
    ----------
    solution       : (15 + 6*n_ima,) array — flat parameter vector
    n_ima          : int — total number of calibration images
    active_images  : list of int/bool, length n_ima (None = all active)
    solution_error : (15 + 6*n_ima,) array — std errors on each parameter
                     (None → all zeros)

    Returns
    -------
    intrinsics : dict with keys fc, cc, alpha_c, kc, KK, inv_KK,
                 fc_error, cc_error, alpha_c_error, kc_error
    extrinsics : list of n_ima dicts, each with keys
                 omc, Tc, Rc, H, omc_error, Tc_error
                 (NaN arrays for inactive images)
    """

    solution = np.asarray(solution, dtype=float)
    n_ima, active_flags, ind_active = check_active_images(n_ima, active_images)

    # Default error vector to zeros if not provided
    if solution_error is None:
        solution_error = np.zeros_like(solution)
    else:
        solution_error = np.asarray(solution_error, dtype=float)

    # ------------------------------------------------------------------ #
    # Step 1 — Extract intrinsic parameters from the first 10 elements
    # ------------------------------------------------------------------ #
    fc      = solution[0:2]
    cc      = solution[2:4]
    alpha_c = float(solution[4])
    kc      = solution[5:10]

    fc_error      = solution_error[0:2]
    cc_error      = solution_error[2:4]
    alpha_c_error = float(solution_error[4])
    kc_error      = solution_error[5:10]

    # ------------------------------------------------------------------ #
    # Step 2 — Build the calibration (intrinsic) matrix K and its inverse
    # KK = [[fx,  fx*alpha_c, cx],
    #       [ 0,  fy,         cy],
    #       [ 0,  0,          1 ]]
    # ------------------------------------------------------------------ #
    KK = np.array([
        [fc[0],  fc[0] * alpha_c,  cc[0]],
        [0,      fc[1],            cc[1]],
        [0,      0,                1.0  ]
    ])
    inv_KK = np.linalg.inv(KK)

    intrinsics = dict(
        fc=fc, cc=cc, alpha_c=alpha_c, kc=kc,
        KK=KK, inv_KK=inv_KK,
        fc_error=fc_error, cc_error=cc_error,
        alpha_c_error=alpha_c_error, kc_error=kc_error
    )

    # ------------------------------------------------------------------ #
    # Step 3 — Extract extrinsic parameters for each image

    # ------------------------------------------------------------------ #
    extrinsics = []

    for kk in range(n_ima):
        base = 15 + 6 * kk   # start index in solution for this image

        if active_flags[kk]:
            omckk = solution[base:base + 3]
            Tckk  = solution[base + 3:base + 6].reshape(3, 1)

            omckk_error = solution_error[base:base + 3]
            Tckk_error  = solution_error[base + 3:base + 6].reshape(3, 1)

            # Rotation matrix from rotation vector
            Rckk, _ = rodrigues(omckk)

            # Homography: H = K · [r1 | r2 | t], normalised so H[2,2] = 1
            Hkk = KK @ np.column_stack([Rckk[:, 0], Rckk[:, 1], Tckk.ravel()])
            Hkk = Hkk / Hkk[2, 2]

        else:
            # Inactive image — fill with NaN
            omckk      = np.full((3,),    np.nan)
            Tckk       = np.full((3, 1),  np.nan)
            Rckk       = np.full((3, 3),  np.nan)
            Hkk        = np.full((3, 3),  np.nan)
            omckk_error = np.full((3,),   np.nan)
            Tckk_error  = np.full((3, 1), np.nan)

        extrinsics.append(dict(
            omc=omckk, Tc=Tckk, Rc=Rckk, H=Hkk,
            omc_error=omckk_error, Tc_error=Tckk_error
        ))

    return intrinsics, extrinsics


# ------------------------------------------------------------------ #
# Self-test with synthetic data
# ------------------------------------------------------------------ #
if __name__ == '__main__':
    from rodrigues import rodrigues

    np.random.seed(7)
    n_ima = 4
    active = [1, 1, 0, 1]   # image 2 is inactive

    # Build a synthetic solution vector
    fc_true      = np.array([800.0, 800.0])
    cc_true      = np.array([320.0, 240.0])
    alpha_c_true = 0.001
    kc_true      = np.array([0.1, -0.05, 0.001, 0.002, 0.0])

    solution = np.zeros(15 + 6 * n_ima)
    solution[0:2]  = fc_true
    solution[2:4]  = cc_true
    solution[4]    = alpha_c_true
    solution[5:10] = kc_true

    for kk in range(n_ima):
        base = 15 + 6 * kk
        solution[base:base+3] = np.random.randn(3) * 0.1   # omc
        solution[base+3:base+6] = [0, 0, 1.0]              # Tc

    intrinsics, extrinsics = extract_parameters(solution, n_ima,
                                                active_images=active)

    print("=== Intrinsics ===")
    print(f"  fc      : {intrinsics['fc']}")
    print(f"  cc      : {intrinsics['cc']}")
    print(f"  alpha_c : {intrinsics['alpha_c']}")
    print(f"  kc      : {intrinsics['kc']}")
    print(f"  KK      :\n{intrinsics['KK']}")

    print("\n=== Extrinsics per image ===")
    for kk, ext in enumerate(extrinsics):
        status = "active" if active[kk] else "INACTIVE"
        nan_flag = np.any(np.isnan(ext['omc']))
        print(f"  Image {kk} ({status}): omc={'NaN' if nan_flag else np.round(ext['omc'],4)}")

    # Verify inactive image is NaN
    assert np.all(np.isnan(extrinsics[2]['Rc'])), "Inactive image should be NaN"
    # Verify round-trip: rodrigues(omc) should give a valid rotation matrix
    for kk in [0, 1, 3]:
        R = extrinsics[kk]['Rc']
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-10), f"R[{kk}] not orthogonal"
    print("\nAll checks passed ✓")
