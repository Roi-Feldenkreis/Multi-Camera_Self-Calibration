import numpy as np
from CalTechToolBoxPython.check_active_images import check_active_images
from CalTechToolBoxPython.comp_ext_calib import comp_ext_calib
from CalTechToolBoxPython.compute_extrinsic_refine import compute_extrinsic_refine
from CalTechToolBoxPython.project_points2 import project_points2
from CalTechToolBoxPython.comp_error_calib import comp_error_calib
from CalTechToolBoxPython.extract_parameters import extract_parameters
from CalTechToolBoxPython.rodrigues import rodrigues


def go_calib_optim_iter(
    x_list, X_list,
    nx=640, ny=480,
    fc=None, cc=None, kc=None, alpha_c=0.0,
    active_images=None,
    est_fc=None, est_aspect_ratio=True,
    center_optim=True, est_alpha=False, est_dist=None,
    max_iter=50, recompute_extrinsic=True,
    check_cond=True, thresh_cond=1e6,
    alpha_smooth=1.0, init_fov=70.0,
    compute_extrinsic_init_fn=None,
    verbose=True,
):
    """
    Main camera calibration function — jointly optimises ALL intrinsic and
    extrinsic parameters by minimising pixel reprojection error.

    This is the TOP LEVEL of the calibration pipeline.  It orchestrates:
      1. Sanitise and initialise flags and intrinsics (fc, cc, kc, alpha_c)
      2. Initialise per-image poses              -> comp_ext_calib
      3. Pack everything into one flat parameter vector
      4. Gauss-Newton loop — jointly update ALL parameters
           a. Call project_points2 per active image -> build block JJ3, ex3
           b. Solve normal equations -> update param
           c. (optional) Re-refine each pose with updated intrinsics
                                                -> compute_extrinsic_refine
      5. Recompute reprojection error            -> comp_error_calib
      6. Estimate parameter uncertainties from the final JJ3
      7. Unpack the solution into named outputs  -> extract_parameters
      8. Print results and model-selection recommendations

    Parameters
    ----------
    x_list          : list of (2, N_k) — observed 2D image corners per image
    X_list          : list of (3, N_k) — corresponding 3D world points
    nx, ny          : int  — image width / height in pixels
    fc              : (2,) — initial focal length  (None -> FOV-based init)
    cc              : (2,) — initial principal pt  (None -> image centre)
    kc              : (5,) — initial distortion    (None -> zeros)
    alpha_c         : float — initial skew          (default 0)
    active_images   : list of int — 1=use / 0=skip (None -> all active)
    est_fc          : (2,) int — which focal components to estimate
    est_aspect_ratio: bool — allow fx != fy (default True)
    center_optim    : bool — estimate principal point (default True)
    est_alpha       : bool — estimate skew (default False)
    est_dist        : (5,) int — which kc coefficients to estimate
    max_iter        : int  — max Gauss-Newton outer iterations (default 50)
    recompute_extrinsic : bool — re-refine poses after each GN step
    check_cond      : bool — deactivate ill-conditioned views
    thresh_cond     : float — Jacobian condition number threshold (1e6)
    alpha_smooth    : float — GN step damping in [0,1] (1 = full step)
    init_fov        : float — initial FOV in degrees for fc init (70)
    compute_extrinsic_init_fn : callable — linear pose initialiser or None
    verbose         : bool  — print progress (default True)

    Returns
    -------
    dict with keys:
      fc, cc, alpha_c, kc, KK, inv_KK            intrinsic parameters
      fc_error, cc_error, alpha_c_error, kc_error  3-sigma uncertainties
      omc_list, Tc_list, Rc_list                   per-image poses
      err_std                                       reprojection error std
      solution, solution_error                      flat parameter vectors
      active_images, param_list                     optimisation history
    """
    n_ima = len(x_list)
    log = print if verbose else lambda *a, **kw: None

    # ------------------------------------------------------------------ #
    # Step 1 — Sanitise optimisation flags
    # ------------------------------------------------------------------ #
    if est_fc   is None: est_fc   = np.array([1, 1])
    if est_dist is None: est_dist = np.array([1, 1, 0, 0, 0])
    est_fc   = np.asarray(est_fc,   dtype=int).ravel()
    est_dist = np.asarray(est_dist, dtype=int).ravel()
    if est_dist.size == 4:
        est_dist = np.append(est_dist, 0)

    # Resolve est_fc / est_aspect_ratio conflicts
    if not np.array_equal(est_fc, [1, 1]):
        est_aspect_ratio = True
    if not est_aspect_ratio:
        est_fc = np.array([1, 1])

    # Skew can only be estimated alongside the principal point
    if not center_optim:
        est_alpha = False

    # ------------------------------------------------------------------ #
    # Step 2 — Initialise intrinsics when not supplied
    # alpha_smooth is reduced to 0.4 for cold starts (safer convergence)
    # ------------------------------------------------------------------ #
    if cc is None:
        cc = np.array([(nx - 1) / 2.0, (ny - 1) / 2.0])
        alpha_smooth = min(alpha_smooth, 0.4)
        log("Initialising principal point at image centre.")

    if kc is None:
        kc = np.zeros(5)
        alpha_smooth = min(alpha_smooth, 0.4)
        log("Initialising distortion to zero.")
    else:
        kc = np.asarray(kc, dtype=float)
        if kc.size == 4:
            kc = np.append(kc, 0.0)

    if fc is None:
        fc = np.array([(nx / 2.0) / np.tan(np.pi * init_fov / 360.0)] * 2)
        alpha_smooth = min(alpha_smooth, 0.4)
        log(f"Initialising focal length from FOV={init_fov}deg -> fc={fc[0]:.1f} px")

    fc      = np.asarray(fc,   dtype=float)
    cc      = np.asarray(cc,   dtype=float)
    alpha_c = 0.0 if not est_alpha else float(alpha_c)

    if not est_aspect_ratio:
        fc[:] = fc.mean()           # force square pixels

    if not np.all(est_dist.astype(bool)):
        kc = kc * est_dist.astype(float)

    # ------------------------------------------------------------------ #
    # Step 3 — Active image bookkeeping
    # ------------------------------------------------------------------ #
    n_ima, active_images, ind_active = check_active_images(n_ima, active_images)

    if center_optim and len(ind_active) < 2:
        log("WARNING: Principal point rejected — need >=2 images for planar rig.")
        center_optim = False
        est_alpha    = False
    if center_optim and len(ind_active) < 5:
        log("WARNING: Principal point estimation may be unreliable (<5 images).")

    # ------------------------------------------------------------------ #
    # Step 4 — Initialise per-image poses (linear init -> GN refine)
    # ------------------------------------------------------------------ #
    log("Initialising extrinsic parameters...")
    omc_list, Tc_list, active_images, N_points_views, _ = comp_ext_calib(
        x_list, X_list, fc, cc, kc, alpha_c,
        active_images=active_images,
        thresh_cond=thresh_cond, check_cond=check_cond,
        compute_extrinsic_init_fn=compute_extrinsic_init_fn,)

    n_ima, active_images, ind_active = check_active_images(n_ima, active_images)

    # ------------------------------------------------------------------ #
    # Step 5 — Pack flat global parameter vector
    #
    # Layout:
    #   [0:2]              fc
    #   [2:4]              cc
    #   [4]                alpha_c
    #   [5:10]             kc  (5 coefficients)
    #   [10:15]            zeros  (reserved padding)
    #   [15+6*kk : 15+6*kk+3]   omc for image kk
    #   [15+6*kk+3 : 15+6*kk+6] Tc  for image kk
    #   Total: 15 + 6*n_ima
    # ------------------------------------------------------------------ #
    param = np.zeros(15 + 6 * n_ima)
    param[0:2]  = fc
    param[2:4]  = cc
    param[4]    = alpha_c
    param[5:10] = kc
    for kk in range(n_ima):
        base = 15 + 6 * kk
        om = omc_list[kk];  Tc = Tc_list[kk]
        if not np.any(np.isnan(om)):
            param[base:base+3]   = om
            param[base+3:base+6] = np.asarray(Tc).ravel()

    param_list       = [param.copy()]
    deactivated_imgs = []
    ind_Jac          = None   # selection mask — built on first iteration

    # ------------------------------------------------------------------ #
    # Step 6 — Main Gauss-Newton loop
    #
    # The global (15+6n) x (15+6n) block-structured Jacobian JJ3 is
    # assembled from per-image contributions, then one global Newton step
    # updates ALL parameters simultaneously.
    # ------------------------------------------------------------------ #
    log(f"\nMain calibration optimisation — {len(ind_active)} active images")
    log("GN iterations: ", end="")

    change = 1.0
    itr    = 0

    while change > 1e-9 and itr < max_iter:
        log(f"{itr + 1}...", end="", flush=True)

        f     = param[0:2]
        c     = param[2:4]
        alpha = param[4]
        k     = param[5:10]

        sz  = 15 + 6 * n_ima
        JJ3 = np.zeros((sz, sz))
        ex3 = np.zeros(sz)

        _, active_images, ind_active = check_active_images(n_ima, active_images)

        for kk in ind_active:
            base  = 15 + 6 * kk
            omckk = param[base:base+3]
            Tckk  = param[base+3:base+6]

            if np.isnan(omckk[0]):
                log(f"\nERROR: NaN pose for image {kk}. Stopping.")
                break

            X_kk = X_list[kk]
            x_kk = x_list[kk]

            if not est_aspect_ratio:
                # scalar f: dxdf is (2N,1); tile to (2N,2) for uniform A
                xp, dxdom, dxdT, dxdf, dxdc, dxdk, dxdalpha = \
                    project_points2(X_kk, omckk, Tckk, f[0], c, k, alpha)
                dxdf = np.tile(dxdf, (1, 2))
            else:
                xp, dxdom, dxdT, dxdf, dxdc, dxdk, dxdalpha = \
                    project_points2(X_kk, omckk, Tckk, f, c, k, alpha)

            # Residual: observed - projected, F-order interleaved (2N,)
            exkk = (x_kk - xp).ravel(order='F')

            # A: intrinsic Jacobian block  (10 x 2N)  [df dc dalpha dk]
            # B: extrinsic Jacobian block  (6  x 2N)  [dom dT]
            A = np.hstack([dxdf, dxdc, dxdalpha, dxdk]).T   # (10, 2N)
            B = np.hstack([dxdom, dxdT]).T                   # (6,  2N)

            # Block accumulation:
            #  top-left  10x10 : intrinsic cross-products (all images sum)
            #  diagonal  6x6   : extrinsic cross-product (this image only)
            #  off-diag  10x6  : intrinsic-extrinsic coupling
            JJ3[0:10, 0:10]               += A @ A.T
            JJ3[base:base+6, base:base+6]  = B @ B.T
            AB = A @ B.T                                     # (10, 6)
            JJ3[0:10, base:base+6]         = AB
            JJ3[base:base+6, 0:10]         = AB.T

            ex3[0:10]         += A @ exkk
            ex3[base:base+6]   = B @ exkk

            # Deactivate degenerate views
            if check_cond and np.linalg.cond(B.T) > thresh_cond:
                active_images[kk] = 0
                log(f"\nWarning: View #{kk} ill-conditioned (A). Set inactive.")
                deactivated_imgs.append(kk)
                param[base:base+6] = np.nan

        # Build selection mask for parameters to update this iteration.
        sel      = np.zeros(sz, dtype=bool)
        sel[0:2] = est_fc.astype(bool)
        sel[2:4] = center_optim
        sel[4]   = est_alpha
        sel[5:10]= est_dist.astype(bool)
        if not est_aspect_ratio and (np.array_equal(est_fc, [1, 1]) or
                                      np.array_equal(est_fc, [1, 0])):
            sel[1] = False   # do not update fc[1] independently
        for kk in range(n_ima):
            if active_images[kk]:
                sel[15 + 6*kk: 15 + 6*kk + 6] = True
        ind_Jac = np.where(sel)[0]

        JJ3_sub = JJ3[np.ix_(ind_Jac, ind_Jac)]
        ex3_sub = ex3[ind_Jac]

        # Smoothed GN step: alpha_smooth2 ramps from alpha_smooth to 1.
        # Damping is useful on cold starts (alpha_smooth=0.4) to avoid
        # overshooting before the Jacobian is well-conditioned.
        alpha_smooth2 = 1.0 - (1.0 - alpha_smooth) ** (itr + 1)
        try:
            param_innov = alpha_smooth2 * np.linalg.solve(JJ3_sub, ex3_sub)
        except np.linalg.LinAlgError:
            log("\nWarning: Singular Jacobian — stopping early.")
            break

        param_up           = param.copy()
        param_up[ind_Jac] += param_innov

        if not est_aspect_ratio and np.array_equal(est_fc, [1, 1]):
            param_up[1] = param_up[0]   # enforce fc[0]==fc[1]

        change = (np.linalg.norm(param_innov) /
                  (np.linalg.norm(param_up[ind_Jac]) + 1e-300))

        # Optional extrinsic re-refinement with updated intrinsics.
        # After the GN step shifts fc/cc/kc, the stored extrinsics are sub-optimal.
        if recompute_extrinsic:
            f_new = param_up[0:2];  c_new = param_up[2:4]
            a_new = param_up[4];    k_new = param_up[5:10]
            _, _, ind_now = check_active_images(n_ima, active_images)
            for kk in ind_now:
                base    = 15 + 6 * kk
                omc_cur = param_up[base:base+3]
                Tc_cur  = param_up[base+3:base+6]
                if np.isnan(omc_cur[0]):
                    continue
                omckk, Tckk, _, JJ_kk = compute_extrinsic_refine(
                    omc_cur, Tc_cur, x_list[kk], X_list[kk],
                    f_new, c_new, k_new, a_new, 20, thresh_cond
                )
                if check_cond and np.linalg.cond(JJ_kk) > thresh_cond:
                    active_images[kk] = 0
                    log(f"\nWarning: View #{kk} ill-conditioned (B). Set inactive.")
                    deactivated_imgs.append(kk)
                    omckk = np.full(3, np.nan);  Tckk = np.zeros(3)
                param_up[base:base+3]   = omckk
                param_up[base+3:base+6] = Tckk

        param = param_up
        param_list.append(param.copy())
        _, active_images, ind_active = check_active_images(n_ima, active_images)
        itr += 1

    log("\ndone")

    # ------------------------------------------------------------------ #
    # Step 7 — Unpack converged solution and compute reprojection error
    # ------------------------------------------------------------------ #
    solution  = param.copy()
    fc_sol    = solution[0:2];  cc_sol  = solution[2:4]
    alpha_sol = solution[4];    kc_sol  = solution[5:10]

    # Build pose lists from flat vector for comp_error_calib
    omc_final, Tc_final, Rc_final = [], [], []
    for kk in range(n_ima):
        base = 15 + 6 * kk
        if active_images[kk]:
            om = solution[base:base+3]
            Tc = solution[base+3:base+6]
            Rc, _ = rodrigues(om)
        else:
            om = np.full(3, np.nan);  Tc = np.full(3, np.nan)
            Rc = np.full((3, 3), np.nan)
        omc_final.append(om); Tc_final.append(Tc); Rc_final.append(Rc)

    # Pass project_points2 explicitly so the full distortion model is used
    # (avoids the simple pinhole fallback inside comp_error_calib)
    log("Estimating uncertainties...", end="")
    ex_list, ex_all, _, _, err_std = comp_error_calib(
        X_list, x_list, omc_final, Tc_final,
        fc_sol, cc_sol, kc_sol, alpha_sol,
        active_images=active_images,
        project_points_fn=lambda X, om, T, fc, cc, kc, al:
            project_points2(X, om, T, fc, cc, kc, al)[0],
    )
    sigma_x = np.std(ex_all)

    # ------------------------------------------------------------------ #
    # Step 8 — Parameter uncertainty estimation
    # Rebuild JJ3 at converged solution, then:
    #   solution_error = 3 * sqrt(diag(inv(JJ3))) * sigma_x   (3-sigma)
    # ------------------------------------------------------------------ #
    _, _, ind_active_final = check_active_images(n_ima, active_images)
    sz  = 15 + 6 * n_ima
    JJ3 = np.zeros((sz, sz))

    for kk in ind_active_final:
        base  = 15 + 6 * kk
        omckk = solution[base:base+3]
        Tckk  = solution[base+3:base+6]
        X_kk  = X_list[kk]

        if not est_aspect_ratio:
            xp, dxdom, dxdT, dxdf, dxdc, dxdk, dxdalpha = \
                project_points2(X_kk, omckk, Tckk, fc_sol[0], cc_sol, kc_sol, alpha_sol)
            dxdf = np.tile(dxdf, (1, 2))
        else:
            xp, dxdom, dxdT, dxdf, dxdc, dxdk, dxdalpha = \
                project_points2(X_kk, omckk, Tckk, fc_sol, cc_sol, kc_sol, alpha_sol)

        A = np.hstack([dxdf, dxdc, dxdalpha, dxdk]).T
        B = np.hstack([dxdom, dxdT]).T

        JJ3[0:10, 0:10]               += A @ A.T
        JJ3[base:base+6, base:base+6]  = B @ B.T
        AB = A @ B.T
        JJ3[0:10, base:base+6]         = AB
        JJ3[base:base+6, 0:10]         = AB.T

    JJ3_sub     = JJ3[np.ix_(ind_Jac, ind_Jac)]
    JJ2_inv     = np.linalg.pinv(JJ3_sub)
    param_error = np.zeros(sz)
    param_error[ind_Jac] = 3.0 * np.sqrt(np.maximum(np.diag(JJ2_inv), 0)) * sigma_x

    if not est_aspect_ratio and np.array_equal(est_fc, [1, 1]):
        param_error[1] = param_error[0]
    solution_error = param_error

    # ------------------------------------------------------------------ #
    # Step 9 — Unpack into named parameters via extract_parameters
    # ------------------------------------------------------------------ #
    intrinsics, extrinsics = extract_parameters(
        solution, n_ima, active_images=active_images,
        solution_error=solution_error
    )
    log("done")

    # ------------------------------------------------------------------ #
    # Step 10 — Print final results and model-selection recommendations
    # ------------------------------------------------------------------ #
    fc_f  = intrinsics['fc'];     fc_e  = intrinsics['fc_error']
    cc_f  = intrinsics['cc'];     cc_e  = intrinsics['cc_error']
    ac_f  = intrinsics['alpha_c'];ac_e  = intrinsics['alpha_c_error']
    kc_f  = intrinsics['kc'];     kc_e  = intrinsics['kc_error']
    angle = 90.0 - np.degrees(np.arctan(ac_f))

    log(f"\nCalibration results after optimisation (uncertainties ~3-sigma):\n")
    log(f"  Focal Length:   fc = [{fc_f[0]:.5f}   {fc_f[1]:.5f}]"
        f"  +/-  [{fc_e[0]:.5f}   {fc_e[1]:.5f}]")
    log(f"  Principal pt:   cc = [{cc_f[0]:.5f}   {cc_f[1]:.5f}]"
        f"  +/-  [{cc_e[0]:.5f}   {cc_e[1]:.5f}]")
    log(f"  Skew:      alpha_c = [{ac_f:.5f}]  +/-  [{ac_e:.5f}]"
        f"   => pixel axis angle = {angle:.5f} deg")
    log(f"  Distortion:     kc = [{' '.join(f'{v:.5f}' for v in kc_f)}]")
    log(f"                     +/- [{' '.join(f'{v:.5f}' for v in kc_e)}]")
    log(f"  Pixel error:   err = [{err_std[0]:.5f}   {err_std[1]:.5f}]")
    log("Note: uncertainties are approx 3x the standard deviations.")

    # Skew recommendation
    ac_lo, ac_hi = ac_f - ac_e / 2, ac_f + ac_e / 2
    if ac_lo < 0 < ac_hi:
        log("\nRecommendation: alpha_c is consistent with zero. "
            "Consider setting est_alpha=False and re-running.")

    # Distortion coefficient recommendations
    kc_lo = kc_f - kc_e / 2;  kc_hi = kc_f + kc_e / 2
    prob  = (kc_lo < 0) & (kc_hi > 0)
    if not (prob[2] and prob[3]):
        prob[2:4] = False
    if prob.any():
        new_est = (est_dist.astype(bool) & ~prob).astype(int)
        log(f"\nRecommendation: Some kc are consistent with zero. "
            f"Consider est_dist={list(new_est)} and re-running.")

    return dict(
        fc=fc_f, cc=cc_f, alpha_c=ac_f, kc=kc_f,
        KK=intrinsics['KK'], inv_KK=intrinsics['inv_KK'],
        fc_error=fc_e, cc_error=cc_e,
        alpha_c_error=ac_e, kc_error=kc_e,
        omc_list=[e['omc'] for e in extrinsics],
        Tc_list =[e['Tc']  for e in extrinsics],
        Rc_list =[e['Rc']  for e in extrinsics],
        err_std=err_std,
        solution=solution, solution_error=solution_error,
        active_images=active_images, param_list=param_list,)


# ------------------------------------------------------------------ #
# Self-test — synthetic camera, 5 calibration views
# ------------------------------------------------------------------ #
if __name__ == '__main__':
    np.random.seed(0)

    nx, ny  = 640, 480
    fc_true = np.array([820.0, 820.0])
    cc_true = np.array([318.0, 241.0])
    kc_true = np.array([-0.25, 0.12, 0.001, -0.0004, 0.0])

    grid_x, grid_y = np.meshgrid(np.linspace(-0.08, 0.08, 7),
                                  np.linspace(-0.06, 0.06, 5))
    X_base = np.zeros((3, 35))
    X_base[0] = grid_x.ravel()
    X_base[1] = grid_y.ravel()

    x_list, X_list = [], []
    for i in range(5):
        om = np.array([0.05*i, -0.03*i, 0.02]) + np.random.randn(3)*0.01
        T  = np.array([0.01*i, 0.0, 0.55])
        xp, *_ = project_points2(X_base, om, T, fc_true, cc_true, kc_true, 0.0)
        xp += np.random.randn(*xp.shape) * 0.5   # 0.5 px noise
        x_list.append(xp)
        X_list.append(X_base.copy())

    results = go_calib_optim_iter(
        x_list, X_list, nx=nx, ny=ny,
        est_dist=np.array([1, 1, 0, 0, 0]),
        center_optim=True, est_alpha=False,
        max_iter=30, recompute_extrinsic=True,
    )

    print(f"\nTrue  fc = {fc_true},  recovered fc = {np.round(results['fc'], 2)}")
    print(f"True  cc = {cc_true},  recovered cc = {np.round(results['cc'], 2)}")
    print(f"Pixel error std: {np.round(results['err_std'], 4)}")
    assert results['err_std'].max() < 1.5, "RMS error too high"
    print("Self-test passed ✓")
