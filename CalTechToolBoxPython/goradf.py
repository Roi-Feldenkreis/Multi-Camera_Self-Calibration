import os
import numpy as np

from CalTechToolBoxPython.preparedata import preparedata
from CalTechToolBoxPython.go_calib_optim_iter import go_calib_optim_iter
from CalTechToolBoxPython.visualize_distortions import visualize_distortions


def goradf(config, par2estimate, INL_TOL, verbose=True):
    """
    Top-level driver for per-camera non-linear intrinsic calibration.

    For every camera listed in config.cal.cams2use this function:
      1. Loads 3D/2D correspondences from a BlueC .rad file (preparedata)
      2. Runs the full CalTech calibration pipeline            (go_calib_optim_iter)
      3. Validates the result; falls back to a safe default KK if it failed
      4. Optionally saves distortion visualisation plots       (visualize_distortions)
      5. Writes the camera matrix KK and distortion vector kc to a
         BlueC-format config file

    This is the FINAL function in the full pipeline — the entry point that
    a user calls after running MultiCamSelfCal and MultiCamValidation.

    Parameters
    ----------
    config : CalibrationConfig
        config.cameras_2_use    — iterable of camera indices to calibrate
        config.principal_point  — (CAMS, 3) array; [:, :2] = [res_x/2, res_y/2]
                                  used to recover per-camera nx/ny
        config.data_path        — directory path for input/output files
        config.basename         — experiment name, used to build .rad output paths
                                  (output: {data_path}{basename}{cam_id}.rad)
        Input points read from: {data_path}points4cal{cam_id}.dat

    par2estimate : array-like, length 6
        [0]   initFOV      — initial camera field of view in degrees
        [1]   center_optim — 1=estimate principal point, 0=fix at centre
        [2:6] est_dist     — 4-element binary vector: which kc to estimate
                             (mapped to est_dist[0:4]; 5th coeff fixed=0)

    INL_TOL : float
        Inlier tolerance in pixels. If any axis of err_std exceeds
        2*INL_TOL the calibration result is considered unreliable and
        a safe default camera matrix is used instead.

    verbose : bool
        Print progress messages (default True).

    Returns
    -------
    selfcalib : dict with keys:
        'goradproblem' : int   — 0 = all cameras OK, 1 = at least one failed
        'results'      : dict  — maps camera index -> go_calib_optim_iter output
        'KK_list'      : dict  — maps camera index -> final 3x3 camera matrix
        'kc_list'      : dict  — maps camera index -> final kc (4,) vector
    """

    log = print if verbose else lambda *a, **kw: None

    # ------------------------------------------------------------------ #
    # Decode par2estimate — mirrors MATLAB's positional assignment
    # ------------------------------------------------------------------ #
    par2estimate = np.asarray(par2estimate).ravel()
    initFOV      = float(par2estimate[0])
    center_optim = bool(int(par2estimate[1]))
    # est_dist from par2estimate has 4 entries (6th kc coeff always 0)
    est_dist_4   = par2estimate[2:6].astype(int)
    est_dist     = np.append(est_dist_4, 0)  # pad to 5 to match kc layout

    selfcalib = {
        'goradproblem': 0,
        'results': {},
        'KK_list': {},
        'kc_list': {},
    }

    idxcams = config.cameras_2_use
    # Recover per-camera resolution from principal_point (set in GoCal.py):
    #   config.principal_point[i, :2] = [res_x/2, res_y/2]
    # so nx = pp_x * 2, ny = pp_y * 2

    # ------------------------------------------------------------------ #
    # Loop over cameras
    # ------------------------------------------------------------------ #
    for count, i in enumerate(idxcams):
        log(f"\n{'='*54}")
        log(f"***** camera {i} {'*'*38}")
        log(f"{'='*54}")

        # ---- Step 1: Load 3D/2D correspondences ---------------------- #

        points_file = os.path.join(config.data_path, f"points4cal{i}.dat")
        log(f"Loading data from: {points_file}")
        X_1, x_1 = preparedata(points_file)

        nx = int(config.principal_point[count, 0] * 2)  # pp_x = res_x/2
        ny = int(config.principal_point[count, 1] * 2)  # pp_y = res_y/2

        # ---- Step 2: Run the full calibration pipeline --------------- #

        results = go_calib_optim_iter(
            x_list=[x_1],
            X_list=[X_1],
            nx=nx, ny=ny,
            init_fov=initFOV,
            center_optim=center_optim,
            est_dist=est_dist,
            verbose=verbose,)

        KK  = results['KK']           # (3,3) camera matrix
        kc  = results['kc']           # (5,)  distortion vector
        err_std = results['err_std']  # (2,)  per-axis reprojection error std
        param_has_nan = bool(np.any(np.isnan(results['solution'])))

        # ---- Step 3: Validate — fall back to safe defaults if failed - #

        if param_has_nan or np.any(err_std > 2 * INL_TOL):
            log(f"\nWARNING: Calibration failed for camera {i} "
                f"(err_std={err_std}, INL_TOL={INL_TOL}). "
                f"Inserting null/default camera model.")
            # Safe default: 700px focal length, 320x240 principal point
            KK = np.array([[700., 0., 320.],
                           [0., 700., 240.],
                           [0.,   0.,   1.]])
            kc = np.zeros(5)
            selfcalib['goradproblem'] = 1
        else:
            # ---- Step 4: Save distortion visualisation --- #

            save_prefix = os.path.join(
                os.path.join(config.data_path, f'NonLinModel.cam{i}'))
            try:
                visualize_distortions(
                    nx, ny,
                    results['fc'], results['cc'], kc,
                    alpha_c=results['alpha_c'],
                    fc_error=results['fc_error'],
                    cc_error=results['cc_error'],
                    kc_error=results['kc_error'],
                    err_std=err_std,
                    save_path=save_prefix,)
                log(f"Distortion plots saved to {save_prefix}_*.png")
            except Exception as e:
                log(f"Warning: could not save distortion plots: {e}")

        # ---- Step 5: Write BlueC config file ------------------------- #

        outputfile = os.path.join(config.data_path, f"{config.basename}{i}.rad")
        log(f"\nExport of intrinsic calibration data to: {outputfile}")
        _write_bluec_config(outputfile, KK, kc)
        log(f"Written: {outputfile}")

        selfcalib['results'][i] = results
        selfcalib['KK_list'][i] = KK
        selfcalib['kc_list'][i] = kc[:4]   # BlueC uses only 4 coefficients

    return selfcalib


def _write_bluec_config(filepath, KK, kc):
    """
    Write camera matrix KK and distortion vector kc to a BlueC-format
    ASCII config file.

    The BlueC format stores the full 3x3 camera matrix element-by-element
    (K11..K33) followed by the first four distortion coefficients kc1..kc4.
    The fifth radial coefficient (6th-order) is not part of the BlueC spec.
    """
    with open(filepath, 'w') as fid:
        # Camera matrix K (row-major, 1-indexed naming)
        for r in range(3):
            for c in range(3):
                label = f"K{r+1}{c+1}"
                # Extra newline after the last K entry
                ending = '\n\n' if (r == 2 and c == 2) else '\n'
                fid.write(f"{label} = {KK[r, c]:.16f}{ending}")
        # Distortion coefficients (only first 4 used by BlueC)
        for j in range(4):
            fid.write(f"kc{j+1} = {kc[j]:.16f}\n")


# ------------------------------------------------------------------ #
# Self-test — uses synthetic data so no real .rad files needed
# ------------------------------------------------------------------ #
if __name__ == '__main__':
    import tempfile, types
    from project_points2 import project_points2

    np.random.seed(3)

    # Synthetic ground-truth camera
    nx, ny   = 640, 480
    fc_true  = np.array([800.0, 800.0])
    cc_true  = np.array([318.0, 242.0])
    kc_true  = np.array([-0.2, 0.08, 0.0, 0.0, 0.0])

    grid_x, grid_y = np.meshgrid(np.linspace(-0.07, 0.07, 6),
                                  np.linspace(-0.05, 0.05, 5))
    X = np.zeros((3, 30))
    X[0] = grid_x.ravel();  X[1] = grid_y.ravel()
    xp, *_ = project_points2(X, np.array([0.05, -0.02, 0.01]),
                              np.array([0.0, 0.0, 0.5]),
                              fc_true, cc_true, kc_true, 0.0)
    xp += np.random.randn(*xp.shape) * 0.3

    # Write a temporary .rad file (BlueC format: X Y Z skip x y per row)
    tmpdir = tempfile.mkdtemp()
    rad_in  = os.path.join(tmpdir, 'cam1.rad')
    rad_out = os.path.join(tmpdir, 'NonLinModel.cam1.cfg')
    data    = np.column_stack([X.T,
                               np.zeros(30),      # column 3 skipped
                               xp.T])
    np.savetxt(rad_in, data)

    # Build a minimal config dict (mirrors MATLAB struct field access)
    config = {
        'cal': {
            'cams2use': [1],
            'Res':      np.array([[nx, ny]]),
        },
        'files': {
            'points4cal': os.path.join(tmpdir, 'cam%d.rad'),
            'rad':        os.path.join(tmpdir, 'NonLinModel.cam%d.cfg'),
        },
        'paths': {
            'data': tmpdir,
        },
    }

    par2estimate = [70,     # initFOV
                    1,      # center_optim
                    1, 1, 0, 0]  # est_dist (4 coefficients)

    result = goradf(config, par2estimate, INL_TOL=1.0)

    print(f"\ngoradproblem flag: {result['goradproblem']}  (0=success)")
    KK_out = result['KK_list'][1]
    kc_out = result['kc_list'][1]
    print(f"Recovered KK diagonal: [{KK_out[0,0]:.1f}, {KK_out[1,1]:.1f}]"
          f"  (true fc={fc_true})")
    print(f"Recovered kc: {np.round(kc_out, 4)}  (true kc[:4]={kc_true[:4]})")

    # Verify output file was written and is parseable
    assert os.path.exists(rad_out), "Config file not written"
    with open(rad_out) as f:
        content = f.read()
    assert 'K11' in content and 'kc1' in content, "Config file malformed"
    print(f"Config file written OK:\n{content}")

    assert result['goradproblem'] == 0, "Should succeed on clean synthetic data"
    print("Self-test passed ✓")
