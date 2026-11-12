import os
import json
import numpy as np
from copy import deepcopy

# Import calibration modules
from MartinecPajdla.fill_mm_bundle import fill_mm_bundle

from CoreFunctions.Euclidize import euclidize
from CoreFunctions.FindInliers import findinl
from CoreFunctions.FindOutliers import findoutl
from CoreFunctions.EstimateReprojectionError import reprerror

from OutputFunctions.Dispcam_Stats import disp_cam_stats
from OutputFunctions.EvalReprerror import evalreprerror, plot_reproj_history
from OutputFunctions.SaveCalPar import savecalpar
from OutputFunctions.alignment_plot import run_alignment

from CommonCfgAndIO.CalibrationSystem import CalibrationSystem, build_camera_params

from RadialDistortions.Undo_Radial import undoradial

# Multimedia modules — guarded so the file loads normally when not installed
from Multimedia.Multimedia_lut import MultiCameraLUT
from Multimedia.Multimedia_params import CameraParams
from Multimedia.multimedia_bundle import multimedia_bundle_adjustment, multimedia_point_residuals

# === DEREFRACT EDIT 1 — Method B bridge (Svoboda <-> Maas) ===
from CoreFunctions.Derefraction import derefraction_bootstrap, simple_metric_anchor
from Multimedia.Frame_convention import FrameConvention
# dispatch_anchor('z'/'xyz') needs a known-camera prior file (validation only);
# the production path is pure self-cal via simple_metric_anchor.
from CoreFunctions.Apply_anchor import dispatch_anchor  # optional: ground-truth A/B tests



# ---------------------------------------------------------------------------
# Ground-truth comparison helpers (diagnostics only — never used by the solver)
# ---------------------------------------------------------------------------
def _similarity_fit(src, dst):
    """Proper rotation R, scale s, translation t with dst ~ s * R @ src + t
    (rows are points). Free rotation/translation/scale: compares SHAPE only."""
    a = src - src.mean(0); b = dst - dst.mean(0)
    U, S, Vt = np.linalg.svd(a.T @ b)
    D = np.eye(3); D[2, 2] = 1.0 if np.linalg.det(Vt.T @ U.T) >= 0 else -1.0
    R = Vt.T @ D @ U.T
    s = float((S * np.diag(D)).sum() / (a ** 2).sum()) if (a ** 2).sum() > 0 else 1.0
    t = dst.mean(0) - s * (R @ src.mean(0))
    return R, s, t


def _glass_frame_fit(src, dst):
    """Rotation about the vertical + horizontal translation only: the freedoms
    that leave the glass plane (z = const) unchanged, i.e. a pure choice of world
    frame. No tilt, no vertical shift, no scale - so height and tilt errors of
    the rig relative to the glass remain in the residual. dst ~ Rz @ src + t."""
    a = src[:, :2] - src[:, :2].mean(0); b = dst[:, :2] - dst[:, :2].mean(0)
    th = np.arctan2(np.sum(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]),
                    np.sum(a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1]))
    c, sn = np.cos(th), np.sin(th)
    Rz = np.array([[c, -sn, 0.0], [sn, c, 0.0], [0.0, 0.0, 1.0]])
    t = np.zeros(3)
    t[:2] = dst[:, :2].mean(0) - Rz[:2, :2] @ src[:, :2].mean(0)
    return Rz, t


def _axis_down(rot):
    """Optical axis (third row of a world->camera rotation), signed to point
    DOWN toward the glass (cameras look down at the water). Independent of the
    rotation convention (Maas omega-phi-kappa or the generator's look-at)."""
    a = np.asarray(rot, float)[2].copy()
    return -a if a[2] > 0 else a


def _angle_deg(u, v):
    return float(np.degrees(np.arccos(np.clip(
        np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)), -1.0, 1.0))))


def main():
    """Main calibration function."""

    import time as _time
    _T0 = _time.perf_counter()          # wall-clock start of the whole run
    _timing = {'setup': 0.0, 'iterations': [], 'polish': 0.0, 'total': 0.0}
    # ---- per-stage timers (EXCLUSIVE: a nested stage pauses its parent, so the
    #      stages add up to the total) -------------------------------------------
    from contextlib import contextmanager
    _stage_t, _stage_n, _stage_stack = {}, {}, []
    _phase = ['']                       # 'polish: ' while the polish runs

    @contextmanager
    def _timed(name):
        now = _time.perf_counter()
        if _stage_stack:                                   # pause the parent
            _stage_t[_stage_stack[-1][0]] = (_stage_t.get(_stage_stack[-1][0], 0.0)
                                             + now - _stage_stack[-1][1])
        _stage_stack.append([name, now])
        _stage_n[name] = _stage_n.get(name, 0) + 1
        try:
            yield
        finally:
            end = _time.perf_counter()
            nm, t_res = _stage_stack.pop()
            _stage_t[nm] = _stage_t.get(nm, 0.0) + end - t_res
            if _stage_stack:                               # resume the parent
                _stage_stack[-1][1] = end

    def _tw(name, fn):
        def _wrapped(*a, **k):
            with _timed(_phase[0] + name):
                return fn(*a, **k)
        return _wrapped

    def _ba_timed(*a, **k):
        _opt = k.get('opt', {}) or {}
        if list(_opt.get('cam_params', [0])) == []:
            _lbl = 'BA: cloud re-triangulation'
        elif not _opt.get('fix_points', True):
            _lbl = 'BA: joint'
        else:
            _lbl = 'BA: cameras'
        with _timed(_phase[0] + _lbl):
            return _mmba_untimed(*a, **k)

    def _lut_build(L):
        with _timed(_phase[0] + 'LUT builds'):
            _r_max = float(getattr(config, 'lut_r_max', 0.0) or 0.0)
            L.build_all(R_max=_r_max if _r_max > 0 else max(data.resolutions[:, 0]) * 0.5,
                        Zp_max=getattr(config, 'lut_zp_max', 300.0),
                        n_R=getattr(config, 'lut_n_r', 80),
                        n_Zp=getattr(config, 'lut_n_zp', 80), verbose=False)

    # time the heavy library calls by wrapping them under their own names
    # (originals read from the module namespace: these names are local to main
    #  from here on, so reading them directly would raise UnboundLocalError)
    _g = globals()
    findinl = _tw('RANSAC (findinl)', _g['findinl'])
    fill_mm_bundle = _tw('projective factorisation', _g['fill_mm_bundle'])
    euclidize = _tw('metric upgrade (euclidize)', _g['euclidize'])
    derefraction_bootstrap = _tw('seed search: de-refraction', _g['derefraction_bootstrap'])
    reprerror = _tw('pinhole reprojection', _g['reprerror'])
    run_alignment = _tw('alignment report + plots', _g['run_alignment'])
    _mmba_untimed = _g['multimedia_bundle_adjustment']
    multimedia_bundle_adjustment = _ba_timed

    _reproj_history = []                 # (stage_label, reprojection RMS px) across the run
    # Convergence thresholds (in-situ, no ground truth). Single source of truth:
    # used by the early-stop in the main loop AND by the polish gate at the end.
    EARLY_STOP_MISFIT_MM = 4.0
    EARLY_STOP_REPROJ_PX = 15.0           # raised from 10 px (user decision)
    _calib_converged = None               # set after the loop: did any iteration pass?
    _gt_cache = None                      # ground truth for the per-iteration diagnostic (loaded once)
    _final_cams = None                    # cameras of the returned solution
    _phase_boundaries = []              # (index, name) where a new phase starts (for the plot)

    # Load configuration and data
    global Pe, Xe, Ce, Re, cam, in_result
    system = CalibrationSystem()
    config = system.load_config_from_file('config_paths.txt')
    data = system.load_data(config)

    # === DEREFRACT EDIT 6 — propagate loaded .rad intrinsics onto config ===
    # build_camera_params() reads config.camera_parameters, but load_data attaches
    # the .rad list to `data`. Copy it across so the real fx (~616) reaches the
    # multimedia camera model instead of the c=376 image-size fallback.
    if getattr(data, 'camera_parameters', None):
        config.camera_parameters = data.camera_parameters

    print('Multi-Camera Self-Calibration, Python version')
    print('************************************************************')
    print(f'Experiment name: {config.basename}')

    # Configuration parameters
    NUM_CAMS_FILL = config.num_cameras_fill
    INL_TOL = config.initial_tolerance
    DO_BA = config.do_bundle_adjustment
    UNDO_RADIAL = config.undo_radial
    # Intrinsics: ESTIMATE mode uses no .rad values anywhere in the pipeline. A copy is
    # kept ONLY for diagnostics (comparison lines in the log), never for any decision.
    _est_intr = getattr(config, 'intrinsics_mode', 'rad') == 'estimate'
    _rad_reference = None
    if _est_intr:
        _rad_reference = getattr(data, 'camera_parameters', None) or None
        data.camera_parameters = []
        config.camera_parameters = None
        data.INT_CAM_PARAMS = False
        print("[intrinsics] ESTIMATE mode: .rad values are NOT used (focal length from the seed"
              + (f" / Focal-Start-Px {config.focal_start_px:g}" if getattr(config, 'focal_start_px', 0) else "")
              + ", principal point at the image centre, distortion assumed zero; freed in the final stages)"
              + (" - .rad kept for comparison in the log only" if _rad_reference else ""))
    INT_RADIAL_PARAM = data.INT_CAM_PARAMS

    # Multimedia setup — mm is None when [Multimedia] section is absent from .cfg
    if config.use_multimedia:
        mm = config.to_multimedia_params()
        print(f'Multimedia geometry ENABLED: n1={mm.n1}, n2={mm.n2}, d={mm.d} mm, n3={mm.n3}')
    else:
        mm = None

    lut        = None   # built after first euclidize() when camera positions are metric
    cameras_mm = None   # built alongside lut

    # === DEREFRACT EDIT 2 — pure self-cal anchor (camera-glass distance only) ===
    # No camera-position prior is used in production. simple_metric_anchor sets
    # scale/orientation from config.camera_glass_distance_mm alone; the absolute
    # scale is finally pinned by the multimedia BA via the glass-thickness signal.
    # (De-refraction tolerates large seed-scale error: ~0.1 mm residual at +/-25%.)

    # Self-calibration iteration control
    selfcal = {
        'par2estimate': config.nonlinear_parameters.copy(),
        'iterate': True,
        'count': 0}

    # Best-result selector: keep the lowest-alignment iteration that also had a
    # converged de-refraction and a valid frame, so a good result (e.g. 15 mm) is
    # never overwritten by a later warp-limited one.
    best = {'misfit': float('inf'), 'rms': float('inf'), 'align_mm': float('nan'),
            'iter': None, 'cam': None, 'in_result': None, 'cameras_mm': None,
            'inlier_idx': None, 'inliers': None}

    def _gt_align_rms(inres):
        """Ground-truth alignment RMS (mm) for a reconstruction, or None if no
        ground truth is configured. Mirrors the selector's extraction."""
        try:
            _al = run_alignment(inres, config)
        except Exception:
            return None
        if isinstance(_al, dict):
            for _k in ('rms_after', 'rms', 'alignment_rms', 'rms_after_mm'):
                if _k in _al and _al[_k] is not None:
                    return float(_al[_k])
        try:
            _rep = os.path.join(config.data_path, 'alignment_report.txt')
            with open(_rep) as _f:
                for _line in _f:
                    if 'after' in _line.lower() and 'align' in _line.lower():
                        import re as _re
                        _m = _re.search(r'([0-9]+\.[0-9]+)', _line)
                        if _m:
                            return float(_m.group(1))
        except Exception:
            pass
        return None


    def _save_mm_camera_files(cams):
        """Write multimedia_params.json and camera{N}.json (all 16 Maas params)
        for the given cameras. Single implementation, used in the loop and once
        more at the end for the returned solution."""
        mm_dict = {'n1': mm.n1, 'n2': mm.n2, 'd': mm.d,
                   'n3': mm.n3, 'z_glass_top': config.z_glass_top}
        mm_path = os.path.join(config.data_path, 'multimedia_params.json')
        with open(mm_path, 'w') as f:
            json.dump(mm_dict, f, indent=2)
        print(f'Multimedia params saved to: {mm_path}')
        for _ci, cam_obj in enumerate(cams):
            cam_out = {
                'cam_id':   cam_obj.cam_id,
                'position': cam_obj.position.tolist(),
                'angles':   cam_obj.angles.tolist(),
                'c':  cam_obj.c,  'xh': cam_obj.xh, 'yh': cam_obj.yh,
                'k1': cam_obj.k1, 'k2': cam_obj.k2, 'k3': cam_obj.k3,
                'p1': cam_obj.p1, 'p2': cam_obj.p2,
                'a1': cam_obj.a1, 'a2': cam_obj.a2,
                'multimedia': {'n1': mm.n1, 'n2': mm.n2, 'd': mm.d, 'n3': mm.n3},
            }
            if INT_RADIAL_PARAM and _ci < len(data.camera_parameters):
                _cp = data.camera_parameters[_ci]
                cam_out['lens_distortion_precorrected'] = {
                    'convention': 'Caltech/Bouguet kc = [k1, k2, p1, p2, k3], '
                                  'normalised image coordinates',
                    'kc': [float(v) for v in np.asarray(_cp['distortion'])[:5]],
                    'a1': float(_cp.get('raw_params', {}).get('a1', 0.0)),
                    'a2': float(_cp.get('raw_params', {}).get('a2', 0.0)),
                    'applied_to': 'observations, once at load; the k1..a2 fields '
                                  'above are 0 because the multimedia model is a '
                                  'pinhole on corrected coordinates',
                }
            with open(os.path.join(config.data_path, f'camera{cam_obj.cam_id}.json'), 'w') as f:
                json.dump(cam_out, f, indent=2)
        print(f'Per-camera parameters saved: {len(cams)} camera{{N}}.json files')

    def _save_correspondences(cam_, in_res, align_=None):
        """Write points4cal{N}.dat: [X Y Z W x y w] for each camera.
        Single implementation, used in the loop and at the end of the run."""
        for i in range(CAMS):
            xe_ = cam_[i]['xdist'][:3, cam_[i]['visandrec']]
            if isinstance(align_, dict) and 'X' in align_:
                Xs = align_['X'][:4, cam_[i]['recandvis']]
            else:
                Xs = in_res['Xe'][:4, cam_[i]['recandvis']]
            np.savetxt(os.path.join(config.data_path,
                       f"points4cal{config.cameras_2_use[i]}.dat"),
                       np.hstack([Xs.T, xe_.T]), fmt='%.6f')


    def _correct_observations(data_, n_cams):
        """Lens-distortion + A/D correction of the measured image coordinates,
        computed ONCE and used identically in every global iteration (incl. 0).
        This is the ONLY place distortion acts: the multimedia model
        (observation_equation_221 -> collinearity_condition) is a pinhole on
        corrected coordinates. Switched on by the data — any non-zero .rad kc
        (or a1/a2) — or by the legacy Undo-Radial flag.
        Coefficients: Caltech/Bouguet kc = [k1, k2, p1, p2, k3], normalised coords."""
        Ws = data_.Ws.copy()
        if not INT_RADIAL_PARAM:
            return Ws, "[distortion] no .rad intrinsics loaded - observations used as measured"
        done = []
        for i in range(n_cams):
            cp = data_.camera_parameters[i]
            kc = np.asarray(cp['distortion'], float)
            raw = cp.get('raw_params', {})
            a1 = float(raw.get('a1', 0.0)); a2 = float(raw.get('a2', 0.0))
            has_kc = bool(np.any(np.abs(kc) > 0)) or bool(config.undo_radial)
            has_ad = (mm is not None) and (abs(a1) > 1e-10 or abs(a2) > 1e-10)
            if has_kc:
                Ws[3*i:3*i+3, :] = undoradial(data_.Ws[3*i:3*i+3, :],
                                              cp['intrinsic_matrix'], kc)
            if has_ad:                               # Maas eq.(1) A/D affine correction
                x_row = Ws[3*i, :].copy(); y_row = Ws[3*i+1, :]
                Ws[3*i, :] = (1.0 + a1) * x_row + a2 * y_row
            if has_kc or has_ad:
                done.append(f"cam{config.cameras_2_use[i]} kc={np.round(kc[:5], 5).tolist()}"
                            + (f" a1={a1} a2={a2}" if has_ad else ""))
        if done:
            return Ws, ("[distortion] observations corrected ONCE at load, used in every "
                        "iteration: " + "; ".join(done))
        return Ws, "[distortion] all .rad kc = 0 - observations used as measured"


    def _gt_diag(label, cams, Xe_, idx_):
        """Ground-truth diagnostic line for one candidate solution (glass-frame tilt,
        height errors, glass-frame and shape particle errors). Ground truth is used
        for this print ONLY - never for selection or any decision."""
        nonlocal _gt_cache
        try:
            if _gt_cache is None:
                _gtp = os.path.join(config.data_path, 'ground_truth.json')
                _gt_cache = json.load(open(_gtp)) if os.path.exists(_gtp) else False
            if not _gt_cache:
                return
            _gtc_d = {int(c['id']): c for c in _gt_cache.get('cameras', [])}
            _pr = [(c, _gtc_d[int(c.cam_id)]) for c in cams if int(c.cam_id) in _gtc_d]
            if len(_pr) < 3:
                return
            _Cr = np.array([p[0].position for p in _pr], float)
            _Cg = np.array([p[1]['position_mm'] for p in _pr], float)
            _R, _s, _t = _similarity_fit(_Cr, _Cg)
            _tilt = _angle_deg(_R @ np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 1.0]))
            _Rz, _tz = _glass_frame_fit(_Cr, _Cg)
            _dh = (_Cr @ _Rz.T + _tz)[:, 2] - _Cg[:, 2]
            _txt = f"tilt {_tilt:.2f} deg | height error " + " ".join(f"{d:+.1f}" for d in _dh) + " mm"
            _gp = np.asarray(_gt_cache.get('particles_mm', []), float)
            _ix = np.asarray(idx_, int)
            _cl = (Xe_[:3, :] / Xe_[3:4, :]).T
            if _gp.ndim == 2 and _cl.shape[0] == _ix.size and _ix.size and _ix.max() < _gp.shape[0]:
                _sub = _gp[_ix]
                _ew = np.linalg.norm(_cl @ _Rz.T + _tz - _sub, axis=1)
                _Rp, _sp, _tp = _similarity_fit(_cl, _sub)
                _es = np.linalg.norm(_sp * _cl @ _Rp.T + _tp - _sub, axis=1)
                _txt += (f" | particles ({_ix.size}): glass frame {np.sqrt(np.mean(_ew**2)):.1f} mm, "
                         f"shape {np.sqrt(np.mean(_es**2)):.2f} mm")
            print(f"  [gt-diag] {label} (ground truth, diagnostic only - never used for selection): {_txt}")
        except Exception as _e:
            print(f"  [gt-diag] skipped ({type(_e).__name__}: {_e})")

    # camera-glass height for the seed frame: the configured value, or None = derive
    # it from the line-of-sight distances in each seed frame (Camera-Glass-Distance: auto)
    _cgd_cfg = None if getattr(config, 'camera_glass_auto', False) else config.camera_glass_distance_mm
    # Staged distance weight: the search keeps the strong weight; the final stages
    # (handover joint BAs, polish) use Dist-Weight-Final. A lighter final weight lets
    # the images, not the hand-measured distances, set the final geometry - and then
    # selection uses reprojection and the misfit is only bounded (Dist-Misfit-Max).
    _w_search = float(getattr(config, 'camera_distance_weight', 40.0))
    _w_final = float(getattr(config, 'dist_weight_final', _w_search))
    _light_final = _w_final < _w_search - 1e-9
    EARLY_STOP_MISFIT_MM = float(getattr(config, 'dist_misfit_max', EARLY_STOP_MISFIT_MM))
    # Final stages (handover joint BAs, polish): free the intrinsics in ESTIMATE mode
    # (with a soft principal-point prior at the image centre) and, in both modes, use
    # the measured line of sight as a weak constraint if LOS-Sigma-Final-Mm > 0.
    # Weights are (image noise / measurement sigma)^2 with an image noise of 0.5 px.
    _los_sigma_final = float(getattr(config, 'los_sigma_final_mm', 0.0) or 0.0)
    _pp_sigma = float(getattr(config, 'pp_prior_sigma_px', 8.0) or 8.0)

    def _final_opt(cam_params_base):
        cp = sorted(set(list(cam_params_base) + ([6, 7, 8] if _est_intr else [])))
        extra = {'cam_params': cp}
        if _est_intr:
            _res = np.atleast_2d(np.asarray(data.resolutions, float))
            _w = (0.5 / _pp_sigma) ** 2
            extra['intr_prior'] = ([(k, 'xh', float(_res[min(k, len(_res) - 1)][0]) / 2.0, _w) for k in range(CAMS)]
                                   + [(k, 'yh', float(_res[min(k, len(_res) - 1)][1]) / 2.0, _w) for k in range(CAMS)])
        if _los_sigma_final > 0 and getattr(config, 'camera_los', None):
            extra['cam_los'] = list(config.camera_los)
            extra['cam_los_weight'] = (0.5 / _los_sigma_final) ** 2
        return extra
    _Ws_obs = None                       # corrected observations (computed once)

    # ---- Seed-Max-Radius (stage-1 test of the wide-coverage seed problem) ------
    # Keep only particles seen within R px of the image centre in EVERY camera that
    # sees them; the rest are marked unobserved, so RANSAC never selects them. Frame
    # indices are unchanged, so the ground-truth comparison needs no remapping.
    _rmax = float(getattr(config, 'seed_max_radius', 0.0) or 0.0)
    _Ws_full_raw, _idm_full, _Ws_obs_full = None, None, None   # unfiltered data (stage 2)
    if _rmax > 0:
        _Ws_full_raw = np.asarray(data.Ws, float).copy()
        _idm_full = np.asarray(data.id_matrix).copy()
        _Kr = data.Ws.shape[0] // 3
        _ppr = np.asarray(data.resolutions, float)[:_Kr] / 2.0
        _seen = np.asarray(data.id_matrix, float)[:_Kr] > 0
        with np.errstate(all='ignore'):
            _rad = np.vstack([np.hypot(data.Ws[3 * k] - _ppr[k, 0], data.Ws[3 * k + 1] - _ppr[k, 1])
                              for k in range(_Kr)])
            _rad[~_seen] = np.nan
            _rworst = np.nanmax(_rad, axis=0)
        _drop = ~(np.isfinite(_rworst) & (_rworst <= _rmax))
        data.Ws = data.Ws.astype(float)
        data.Ws[:, _drop] = np.nan
        data.id_matrix = np.asarray(data.id_matrix).copy()
        data.id_matrix[:, _drop] = 0
        _all_p = bool(getattr(config, 'ba_all_particles', False))
        print(f"[seed-radius] keeping {int((~_drop).sum())} of {_drop.size} particles seen within "
              f"{_rmax:g} px of the image centre in every camera "
              + ("for the SEED; the bundle adjustment gets all particles" if _all_p
                 else "(applies to the whole run)"))
        if _all_p and getattr(config, 'ba_mode', 'alternate') != 'camera-then-joint':
            print("[seed-radius] NOTE: BA-All-Particles only acts in BA-Mode = camera-then-joint")
        if int((~_drop).sum()) < 40:
            print("[seed-radius] WARNING: fewer than 40 particles left - the seed will be weak")

    ###########################################################################
    # Main global cycle begins
    ###########################################################################
    _timing['setup'] = _time.perf_counter() - _T0    # config + data load + geometry setup
    while selfcal['iterate'] and selfcal['count'] < config.global_iteration_max:

        _t_iter = _time.perf_counter()               # start of this global iteration

        # Create working copy of data
        linear = deepcopy(data)

        CAMS = len(config.cameras_2_use)
        FRAMES = data.id_matrix.shape[1]

        if CAMS < 3 or FRAMES < 20:
            raise ValueError('Not enough cameras or images -> Problem in loading data?')

        # Correct the required amount of cameras to be filled
        if CAMS - NUM_CAMS_FILL < 3:
            NUM_CAMS_FILL = CAMS - 3

        # Set resolution and principal point in config
        config.resolution = data.resolutions[0]
        # Principal point for the PROJECTIVE / METRIC SEED stage (centring of the
        # observations, fill_mm, euclidize): the image centre, as in the original
        # MultiCamSelfCal. Using the true (.rad) principal point here was tested and
        # made every seed draw land on the same warped metric solution (shape error
        # 22%, ~31% of particles below the glass), so no seed passed the gatekeeper;
        # with the image centre the draws vary and good seeds are found.
        # The multimedia camera model (build_camera_params) still uses the TRUE .rad
        # principal points, so the final calibration is not affected by this choice.
        pp = np.hstack([data.resolutions / 2, np.zeros((CAMS, 1))])
        if selfcal['count'] == 0:
            _pp_rad = ('; multimedia model uses .rad: '
                       + '; '.join(f"cam{config.cameras_2_use[i]} "
                                   f"({cp['intrinsic_matrix'][0, 2]:.2f}, {cp['intrinsic_matrix'][1, 2]:.2f})"
                                   for i, cp in enumerate(data.camera_parameters[:CAMS]))
                       if INT_RADIAL_PARAM else '')
            print('[principal point] seed stage uses the image centre '
                  + '; '.join(f'cam{config.cameras_2_use[i]} ({pp[i,0]:.0f}, {pp[i,1]:.0f})'
                              for i in range(CAMS)) + _pp_rad)
        principal_point = pp.reshape(CAMS * 3, 1)
        config.principal_point = pp  # Store as (CAMS, 3) for later use

        #######################################################################
        # Distortion + A/D correction: computed once, applied in EVERY iteration
        #######################################################################
        if _Ws_obs is None:
            _Ws_obs, _dist_report = _correct_observations(data, CAMS)
            print(_dist_report)
        linear.Ws = _Ws_obs.copy() - np.tile(principal_point, (1, FRAMES))

        #######################################################################
        # Detection of outliers - RANSAC is pairwise applied
        #######################################################################
        print()
        print(f'********** After {selfcal["count"]} iteration *******************************************')
        print(f'RANSAC validation step running with tolerance threshold: {INL_TOL:.2f} ...')

        inliers = { 'reprerr': [9e9],
                    'IdMat': np.zeros_like(linear.id_matrix) }

        inliers['IdMat'] = findinl(linear.Ws, linear.id_matrix, INL_TOL)

        #######################################################################
        # Create packed representation
        # Remove zero-columns or columns with insufficient cameras
        #######################################################################
        packed = {}
        packed['idx'] = np.where(np.sum(inliers['IdMat'], axis=0) >= CAMS - NUM_CAMS_FILL)[0]
        packed['IdMat'] = inliers['IdMat'][:, packed['idx']]
        packed['Ws'] = linear.Ws[:, packed['idx']]

        if packed['Ws'].shape[1] < 20:
            raise ValueError(
                f"Only {packed['Ws'].shape[1]} points survived RANSAC validation and packing: "
                f"probably not enough points for reliable self calibration")

        #######################################################################
        # Fill cam structures  (factored so a fresh RANSAC draw can rebuild it)
        #######################################################################
        def _make_cam():
            _cam = []
            for i in range(CAMS):
                cam_dict = {
                    'camId': config.cameras_2_use[i],
                    'idlin': np.where(linear.id_matrix[i, :] > 0)[0],
                    'idin': np.where(inliers['IdMat'][i, :] > 0)[0]
                }
                cam_dict['xdist'] = data.Ws[3*i:3*i+3, cam_dict['idlin']]
                cam_dict['xgt'] = linear.Ws[3*i:3*i+3, cam_dict['idlin']].copy()
                cam_dict['xgtin'] = linear.Ws[3*i:3*i+3, cam_dict['idin']].copy()
                cam_dict['xgt'] += config.principal_point[i, :].reshape(3, 1)
                cam_dict['xgtin'] += config.principal_point[i, :].reshape(3, 1)
                _cam.append(cam_dict)
            return _cam

        cam = _make_cam()

        #######################################################################
        # Options for the Martinec-Pajdla filling procedure
        #######################################################################
        options = {
            'verbose': False,
            'no_BA': True,         # BA skipped in main loop; enabled only for final BA step
            'no_factorization': False,
            'tol': 1e-6,
            'metric': 1,
            'strategy': -1,
            'create_nullspace': {
                'trial_coef': 10,  # MATLAB default; 10x more samples = much better null space
                'threshold': 0.01,
                'verbose': False
            }
        }

        outliers = 1
        inliers['idx'] = packed['idx'].copy()

        # ------------------------------------------------------------------
        # Re-euclidize the de-refracted observations into the SAME camera-based
        # physical frame the BA optimises. Factored out so the seed gatekeeper can
        # check the cloud the BA WILL ACTUALLY USE (not boot's frozen cloud, which
        # drifts above the glass), and so the accepted result is reused rather than
        # recomputed. Passing camera_distances is the key fix: without it this
        # frame silently fell back to the flaky cloud-top gauge, so none of the
        # camera-based scale / camera-glass-distance origin reached the BA's cloud.
        # ------------------------------------------------------------------
        def _reeuclidize(ws_corrected):
            opts = dict(options); opts['no_BA'] = 0
            P, X, u1, u2, info = fill_mm_bundle(
                ws_corrected[:, inliers['idx']],
                config.principal_point[:, :2].T, opts,
                cameras_mm=None, z_glass_top=config.z_glass_top)
            Rmat = P @ X
            Lambda = Rmat[2::3, :]
            Pe_, Xe_, Ce_, Re_ = euclidize(Rmat, Lambda, P, X, config)
            res = {'Pe': Pe_, 'Xe': Xe_, 'Ce': Ce_, 'Re': Re_}
            if mm is not None:
                _cc = (Ce_[:3, :] / Ce_[3:4, :]).T
                _xe3 = Xe_[:3, :] / Xe_[3:4, :]
                _t = float(getattr(mm, 't_total', 0.0))
                _frame = FrameConvention.from_euclidize(
                    _cc, _xe3, _cgd_cfg, _t,
                    z_glass_top=config.z_glass_top,
                    camera_distances=getattr(config, 'camera_distances', None),
                    camera_los=getattr(config, 'camera_los', None),
                    verbose=False)
                if _frame is not None:
                    res['Pe'] = _frame.to_frame_Pe(Pe_)
                    _Ce3 = _frame.to_frame_points(Ce_[:3, :] / Ce_[3:4, :])
                    _Xe3p = _frame.to_frame_points(_xe3)
                    res['Ce'] = np.vstack([_Ce3, np.ones(_Ce3.shape[1])])
                    res['Xe'] = np.vstack([_Xe3p, np.ones(_Xe3p.shape[1])])
            return res

        #######################################################################
        # Iterative outlier removal loop
        #######################################################################
        while outliers:
            print(f"{len(inliers['idx'])} points/frames have survived validations so far")
            print('Filling of missing points is running ...')

            # Perform reconstruction with missing point filling
            P, X, u1, u2, info = fill_mm_bundle(  # (2,CAMS) to match MATLAB pp(:,1:2)'
                linear.Ws[:, inliers['idx']],
                config.principal_point[:, :2].T,
                options,
                cameras_mm=cameras_mm,
                z_glass_top=config.z_glass_top)

            # Compute reprojection matrix
            Rmat = P @ X
            Lambda = Rmat[2::3, :]  # Every 3rd row starting from index 2

            # Euclidean reconstruction
            Pe, Xe, Ce, Re = euclidize(Rmat, Lambda, P, X, config)

            # Build per-camera geometry and LUT — first valid moment for metric positions
            if mm is not None:
                cameras_mm = build_camera_params(Ce, Re, config, mm, CAMS, Pe=Pe)
                lut = MultiCameraLUT(cameras_mm, z_glass_top=config.z_glass_top)
                _lut_build(lut)

            print('************************************************************')

            # Compute reprojection errors — multimedia model used when lut is not None
            cam = reprerror(cam, Pe, Xe, FRAMES, inliers)  # pure-pinhole reprerror (installed version)

            # Detect outliers in cameras
            outliers, inliers = findoutl(cam, inliers, INL_TOL, NUM_CAMS_FILL)

            print(f'Number of detected outliers: {outliers:3d}')
            print('About cameras (Id, 2D reprojection error, #inliers):')
            disp_cam_stats(cam, inliers)
            print('***************************************************************')

            # Track reprojection error history
            inliers['reprerr'].append(np.mean([c['mean2Derr'] for c in cam]))

            # Match MATLAB: only set no_BA option, never break — loop runs until outliers==0
            if (inliers['reprerr'][-1] < 5 * INL_TOL or
                    inliers['reprerr'][-2] < inliers['reprerr'][-1]):
                try:
                    options['no_BA'] = 0 if config.do_bundle_adjustment else 1
                except:
                    options['no_BA'] = 1

        #######################################################################
        # === DEREFRACT EDIT 3 — Method B: iterative de-refraction bootstrap ===
        # Re-feed Svoboda with progressively de-refracted image points until the
        # pixel-space refraction correction converges. Produces a metric, pinhole-
        # consistent dataset (Ws_for_ba) + anchored geometry that lands the
        # multimedia BA inside its convergence basin.
        #######################################################################
        Ws_for_ba = linear.Ws            # air-only path: unchanged
        derefract_ran = False            # True when the bootstrap produced a usable frozen reconstruction
        reeuc_accepted = None            # gated re-euclidize (derefract path); None on air-only path
        seed_ok = True                   # gatekeeper verdict; True by default (air-only / no known geometry)
        if mm is not None:
            _opts_df = dict(options); _opts_df['no_BA'] = True   # fast factorisation per pass

            def _svoboda_step(Ws_cols):
                P_, X_, _u1, _u2, _info = fill_mm_bundle(
                    Ws_cols, config.principal_point[:, :2].T, _opts_df,
                    cameras_mm=None,                     # corrected pts are pinhole-consistent
                    z_glass_top=config.z_glass_top)
                Rmat_ = P_ @ X_; Lambda_ = Rmat_[2::3, :]
                _out = euclidize(Rmat_, Lambda_, P_, X_, config)
                _last_seed_Pe[0] = _out[0]          # the camera builder's focal source (estimate mode)
                return _out

            _last_seed_Pe = [None]

            def _build_cams(Ce_, Re_):
                return build_camera_params(Ce_, Re_, config, mm, CAMS, Pe=_last_seed_Pe[0])

            def _build_lut(cams_):
                L = MultiCameraLUT(cams_, z_glass_top=config.z_glass_top)
                _lut_build(L)
                return L

            # ---- seed gatekeeper + retry -------------------------------------
            # Each retry draws a FRESH RANSAC inlier set (findinl is stochastic),
            # which changes the euclidize reconstruction -> a genuinely different
            # seed. We accept the first seed the gatekeeper judges physically
            # reasonable (finite/positive scale, cameras above glass, >=90%
            # particles below glass with a 10 mm margin; constellation shape is
            # advisory, corrected later by the BA distance penalty). If a draw is
            # deterministic (identical to the previous one) we stop early rather
            # than grind through all attempts for nothing. Tolerances live in
            # seed_gatekeeper.py.
            from seed_gatekeeper import seed_is_reasonable
            MAX_SEED_RETRIES = int(getattr(config, 'max_seed_retries', 10))
            _seed_frac = float(getattr(config, 'seed_subset_frac', 1.0) or 1.0)
            _seed_rng = np.random.default_rng()
            _known = getattr(config, 'camera_distances', None)
            boot = None
            _prev_sig = None
            for _attempt in range(MAX_SEED_RETRIES):
                if _attempt > 0:
                    # fresh RANSAC draw + rebuild the inlier-dependent structures
                    inliers['IdMat'] = findinl(linear.Ws, linear.id_matrix, INL_TOL)
                    packed['idx'] = np.where(
                        np.sum(inliers['IdMat'], axis=0) >= CAMS - NUM_CAMS_FILL)[0]
                    if len(packed['idx']) < 20:
                        print(f"  [seed] attempt {_attempt + 1}: only {len(packed['idx'])} "
                              f"points survived RANSAC -> redrawing")
                        continue
                    packed['IdMat'] = inliers['IdMat'][:, packed['idx']]
                    packed['Ws'] = linear.Ws[:, packed['idx']]
                    inliers['idx'] = packed['idx'].copy()
                    cam = _make_cam()          # keep cam consistent with this draw

                # random subset of this attempt's inliers, so retries differ even when
                # RANSAC keeps every point (clean data -> identical, deterministic seeds)
                if _seed_frac < 1.0 and len(inliers['idx']) >= 40:
                    _n_keep = max(30, int(round(_seed_frac * len(inliers['idx']))))
                    _keep = np.sort(_seed_rng.choice(len(inliers['idx']), size=_n_keep, replace=False))
                    packed['idx'] = np.asarray(inliers['idx'])[_keep]
                    packed['IdMat'] = inliers['IdMat'][:, packed['idx']]
                    packed['Ws'] = linear.Ws[:, packed['idx']]
                    inliers['idx'] = packed['idx'].copy()
                    cam = _make_cam()
                    print(f"  [seed] attempt {_attempt + 1}: random subset of {_n_keep} inliers")

                boot = derefraction_bootstrap(
                    linear.Ws, inliers['idx'], inliers['IdMat'],
                    _svoboda_step, _build_cams, _build_lut,
                    camera_glass_distance=_cgd_cfg,
                    anchor=None,                      # -> simple_metric_anchor (pure self-cal)
                    z_glass_top=config.z_glass_top,
                    camera_distances=_known,
                    camera_los=getattr(config, 'camera_los', None),
                    max_shift_px=float(getattr(config, 'derefract_max_shift_px', 0.0) or 0.0),
                    max_iter=8, tol=1e-2, c_hook=None, verbose=True)

                if not _known:
                    break   # no known geometry -> nothing to gate on, accept

                # Gate on the RE-EUCLIDIZED cloud — the exact reconstruction the BA
                # will optimise — not boot's frozen cloud (which drifts above glass).
                try:
                    _re = _reeuclidize(boot['Ws_corrected'])
                except Exception as _e:
                    print(f"  [seed] attempt {_attempt + 1}/{MAX_SEED_RETRIES} rejected: "
                          f"re-euclidize failed ({type(_e).__name__}) -> redrawing")
                    continue
                _cams_phys = (_re['Ce'][:3, :] / _re['Ce'][3:4, :]).T
                _cloud_phys = (_re['Xe'][:3, :] / _re['Xe'][3:4, :]).T
                _ok, _reasons, _diag = seed_is_reasonable(
                    _cams_phys, _cloud_phys, config.z_glass_top, _known,
                    scale=None,
                    cyclic_order=getattr(config, 'camera_cyclic_order', None))
                if _ok:
                    reeuc_accepted = _re
                    if _attempt > 0:
                        print(f"  [seed] accepted on attempt {_attempt + 1}/{MAX_SEED_RETRIES} "
                              f"(re-euclidized cloud: "
                              f"{100*_diag['frac_particles_below_glass']:.0f}% below glass)")
                    break

                _sig = (int(len(inliers['idx'])),
                        round(float(_diag.get('frac_particles_below_glass', -1)), 3))
                print(f"  [seed] attempt {_attempt + 1}/{MAX_SEED_RETRIES} rejected: "
                      f"{'; '.join(_reasons)}")
                if _sig == _prev_sig:
                    print(f"  [seed] reconstruction did not change between draws "
                          f"(deterministic) -> stopping retries, proceeding with this seed")
                    break
                _prev_sig = _sig
            else:
                print(f"  [seed] no reasonable seed after {MAX_SEED_RETRIES} attempts; "
                      f"proceeding with the last (BA may be skipped)")

            Ws_for_ba  = boot['Ws_corrected']
            cameras_mm = boot['cameras_mm']
            lut        = boot['lut']
            Pe, Xe, Ce, Re = boot['Pe'], boot['Xe'], boot['Ce'], boot['Re']

            # ---- SINGLE SOURCE OF TRUTH for geometry validity --------------------
            # The seed gatekeeper (seed_is_reasonable, evaluated on the re-euclidized
            # cloud the BA actually uses) is the ONLY authority on whether this
            # iteration's geometry is usable. Everyone downstream reads `seed_ok`;
            # nothing re-decides. boot's own frame flag is kept only as a printed
            # diagnostic and no longer gates anything.
            if not _known:
                seed_ok = True                       # no known geometry -> no gate (legacy behaviour)
            else:
                seed_ok = (reeuc_accepted is not None)
            derefract_ran = True                     # de-refraction always produced Ws_for_ba here

            _boot_frame_ok = boot.get('geometry_ok', True)   # diagnostic only
            _boot_head = (f"De-refraction stopped by the divergence guard after {boot['n_iter']} bounded pass(es); "
                          if boot.get('guard_stopped')
                          else f"De-refraction converged in {boot['n_iter']} pass(es); ")
            print(_boot_head
                  + f"frame scale={boot['anchor_info'].get('scale')}, "
                  f"seed_ok={seed_ok} (gatekeeper verdict), boot_frame_ok={_boot_frame_ok}")
            if not seed_ok:
                print("  [gatekeeper] no reasonable seed for this outer iteration -> "
                      "BA will be SKIPPED and the iteration marked invalid.")

        #######################################################################
        # Final refinement through Bundle Adjustment if required
        #######################################################################
        in_result = {}

        start_ba = getattr(config, 'start_ba', False)
        # NOTE: the de-refraction (above) still runs and produces Ws_for_ba (the
        # corrected, pinhole-consistent points). We feed those through a FRESH
        # euclidize here — this is the path that previously reached 15 mm, because
        # a fresh euclidize of the de-refracted points can land the cloud BELOW the
        # glass (in the water), where the multimedia BA has a real refraction
        # residual to descend. Boot's frozen reconstruction, by contrast, lands the
        # cloud ABOVE the glass and leaves the BA inert. The selector + below-glass
        # diagnostic below let us see how often this lands in the BA's valid domain.
        if (DO_BA and not start_ba) or (DO_BA and start_ba and len(inliers['reprerr']) < 3):
            print('**************************************************************')
            print('Refinement by using Bundle Adjustment')
            options['no_BA'] = 0

            if reeuc_accepted is not None:
                # reuse the gated re-euclidize (already in the camera-based frame)
                in_result['Pe'] = reeuc_accepted['Pe']
                in_result['Xe'] = reeuc_accepted['Xe']
                in_result['Ce'] = reeuc_accepted['Ce']
                in_result['Re'] = reeuc_accepted['Re']
            else:
                _re = _reeuclidize(Ws_for_ba)
                in_result['Pe'] = _re['Pe']
                in_result['Xe'] = _re['Xe']
                in_result['Ce'] = _re['Ce']
                in_result['Re'] = _re['Re']

            # DIAGNOSTIC (log only): the seed's OWN intrinsics, from its metric upgrade
            # (RQ of each seed camera matrix), next to the calibration file's values if
            # one is loaded. Measures how good a starting point the seed gives when the
            # intrinsics are to be estimated without a .rad file.
            try:
                from CommonCfgAndIO.CalibrationSystem import _intrinsics_from_Pe
                _parts = []
                _resol = np.atleast_2d(np.asarray(data.resolutions, float))
                _cps = getattr(config, 'camera_parameters', None) or _rad_reference or []
                for _k in range(CAMS):
                    _wh = _resol[min(_k, len(_resol) - 1)]
                    _c, _x, _y = _intrinsics_from_Pe(in_result['Pe'][3*_k:3*_k+3, :], imsize=(_wh[0], _wh[1]))
                    if _c is None:
                        _parts.append(f"cam{_k+1} n/a"); continue
                    _t = f"cam{_k+1} f={_c:.1f} pp=({_x:.1f},{_y:.1f})"
                    _cp = _cps[_k] if _k < len(_cps) and isinstance(_cps[_k], dict) else None
                    if _cp is not None and _cp.get('intrinsic_matrix') is not None:
                        _K = np.asarray(_cp['intrinsic_matrix'], float)
                        _t += (f" [.rad f={_K[0,0]:.1f} ({100*(_c/_K[0,0]-1):+.1f}%), "
                               f"pp=({_K[0,2]:.1f},{_K[1,2]:.1f})]")
                    _parts.append(_t)
                print("  [seed-intrinsics] " + " | ".join(_parts))
            except Exception as _e:
                print(f"  [seed-intrinsics] skipped ({type(_e).__name__}: {_e})")

            # Which side of the glass did the re-euclidized cloud land on? This is
            # the predictor of whether the BA will engage (below glass -> real
            # refraction residual -> BA can descend; above -> inert).
            if mm is not None:
                _Xb = in_result['Xe'][:3, :] / in_result['Xe'][3:4, :]
                _gb = config.z_glass_top - float(getattr(mm, 't_total', 0.0))
                _below = float(np.nanmean(_Xb[2, :] < _gb))
                print(f"  [reEuc] cloud Z range=[{np.nanmin(_Xb[2,:]):.1f},{np.nanmax(_Xb[2,:]):.1f}]mm "
                      f"particles below glass={100*_below:.0f}% "
                      f"-> BA {'SHOULD engage' if _below > 0.5 else 'likely inert'}")

            # Rebuild LUT from final refined camera positions before final reprerror call
            if mm is not None:
                cameras_mm = build_camera_params(in_result['Ce'], in_result['Re'], config, mm, CAMS,
                                                 Pe=in_result.get('Pe'))
                lut = MultiCameraLUT(cameras_mm, z_glass_top=config.z_glass_top)
                _lut_build(lut)

            cam = reprerror(cam, in_result['Pe'], in_result['Xe'], FRAMES, inliers)
        else:
            in_result['Pe'] = Pe
            in_result['Xe'] = Xe
            in_result['Ce'] = Ce
            in_result['Re'] = Re

        #######################################################################
        # Multimedia bundle adjustment — corrects bias from standard pinhole BA
        #######################################################################
        _mm_rms_after = float('nan')   # in-situ reprojection metric (px) for selection
        _mm_misfit = float('nan')      # in-situ camera-distance misfit (mm) — primary selection metric
        _n_failed_last = 0                   # failed rays of this iteration's final BA
        if mm is not None and seed_ok:
            import warnings as _warnings
            _ba_mode = getattr(config, 'ba_mode', 'alternate')
            _skip_joint = False
            _cam_params = getattr(config, 'ba_cam_params', [0, 1, 2, 3, 4, 5])
            _dist_opt = {'cam_dist': getattr(config, 'camera_distances', []),
                         'cam_dist_weight': getattr(config, 'camera_distance_weight', 1.0)}

            def _log_ba(tag, inf):
                print(f"  [{tag}] rms_before={inf.get('rms_before', float('nan')):.2f}px "
                      f"rms_after={inf.get('rms_after', float('nan')):.2f}px "
                      f"nfev={inf.get('n_iter', 0)} "
                      f"max cam shift={np.max(inf.get('cam_shifts', [0.0])):.1f}mm"
                      + (f" cam-dist misfit={inf.get('cam_dist_misfit', float('nan')):.1f}mm"
                         if inf.get('cam_dist_pairs', 0) else "")
                      + (f" FAILED rays={inf.get('n_failed_rays', 0)}"
                         if inf.get('n_failed_rays', 0) else ""))

            # (1) camera BA, cloud frozen  — modes 'alternate' and 'camera-then-joint'
            if _ba_mode in ('alternate', 'camera-then-joint'):
                with _warnings.catch_warnings():
                    _warnings.simplefilter('ignore')   # mute the inner radial-shift non-convergence flood
                    cameras_mm, in_result['Xe'], info_mm_ba = multimedia_bundle_adjustment(
                        cameras_mm  = cameras_mm,
                        Xe          = in_result['Xe'],
                        cam         = cam,
                        z_glass_top = config.z_glass_top,
                        opt         = {
                            'fix_points'      : True,
                            'cam_params'      : _cam_params,
                            'max_nfev'        : 80,
                            'ftol'            : 1e-4,
                            'xtol'            : 1e-4,
                            'gtol'            : 1e-4,
                            'max_pos_step_mm' : 250.0,
                            # Soft known-geometry constraint: pins the camera scale that
                            # reprojection alone cannot, closing the baseline ambiguity.
                            **_dist_opt,
                        },
                    )
                _log_ba('mmBA', info_mm_ba)

            # ---- early-out after the camera BA (mode 'camera-then-joint') ----------
            # Evidence from 5 runs (19 bundle adjustments): when the camera BA left
            # failed rays or cameras facing away from the particles, the joint BA never
            # recovered (it ended at 100-360 px, or in the facing-away attractor at
            # ~3.4 px). The exact facing-away mirror twin is repaired first so the
            # joint BA starts in the right basin; any other lost state skips the joint
            # BA - the iteration is invalid anyway, and the time goes to the next one.
            _skip_joint = False
            if _ba_mode == 'camera-then-joint':
                from seed_gatekeeper import is_facing_away_mirror, mirror_twin, front_facing_check
                _pts_c = (in_result['Xe'][:3, :] / in_result['Xe'][3:4, :]).T
                _cyc_c = getattr(config, 'camera_cyclic_order', None)
                if _cyc_c and is_facing_away_mirror(cameras_mm, _pts_c, _cyc_c):
                    cameras_mm, _pts_c = mirror_twin(cameras_mm, _pts_c)
                    in_result['Xe'][:3, :] = _pts_c.T * in_result['Xe'][3:4, :]
                    print("  [twin] camera BA ended in the facing-away mirror twin -> mapped to "
                          "its exact correct-handed twin before the joint BA")
                _nf_c = int(info_mm_ba.get('n_failed_rays', 0))
                _ffc_ok, _ffc, _ = front_facing_check(cameras_mm, _pts_c)
                if _nf_c > 0 or not _ffc_ok:
                    _skip_joint = True
                    _why = (f"{_nf_c} failed rays" if _nf_c > 0 else
                            "particles behind camera(s): in front "
                            + ", ".join(f"{100*f:.0f}%" for f in _ffc))
                    print(f"  [early-out] camera BA left a lost state ({_why}) -> joint BA "
                          f"skipped; this iteration will be invalid")

            # (2) JOINT camera + point BA — modes 'joint' and 'camera-then-joint'.
            # Cameras and cloud move together, so the rig's height and tilt relative
            # to the glass can be corrected; the alternation cannot do that (each
            # half-step holds the other half fixed).
            if _ba_mode in ('joint', 'camera-then-joint') and not _skip_joint:
                with _warnings.catch_warnings():
                    _warnings.simplefilter('ignore')
                    cameras_mm, in_result['Xe'], info_mm_ba = multimedia_bundle_adjustment(
                        cameras_mm  = cameras_mm,
                        Xe          = in_result['Xe'],
                        cam         = cam,
                        z_glass_top = config.z_glass_top,
                        opt         = {
                            'fix_points'      : False,
                            'cam_params'      : _cam_params,
                            'max_nfev'        : 80,
                            'ftol'            : 1e-4,
                            'xtol'            : 1e-4,
                            'gtol'            : 1e-4,
                            'max_pos_step_mm' : 250.0,
                            'max_pt_step_mm'  : 200.0,
                            'allow_invalid_geometry': True,
                            **_dist_opt,
                        },
                    )
                _log_ba('jointBA', info_mm_ba)

            # ---- STAGE 2: hand ALL particles to the bundle adjustment ---------------
            # The seed (pinhole method) used only the central particles, where the
            # pinhole approximation of refraction holds. Only AFTER the joint BA on
            # those particles are the cameras in the good basin (run evidence: right
            # after the camera BA they were still at ~66 px / 20 mm misfit, so rays
            # intersected from them misplaced every outer particle). So: only from a
            # successful central joint BA (< Handover-Max-Reproj-Px, default 1 px; no failed rays, all cameras facing
            # the particles), add the outer particles - straight-line ray intersection
            # from the cameras, then a cameras-held refinement with the exact
            # refraction model (tested: 69 mm -> 0.76 mm median) - prune added ones
            # that still do not fit (> 5 px or a failed ray), and run a second joint
            # BA on everything. Any failure restores the central-particle state.
            _do_handover = False
            if (_ba_mode == 'camera-then-joint' and not _skip_joint and _Ws_full_raw is not None
                    and bool(getattr(config, 'ba_all_particles', False))):
                from seed_gatekeeper import front_facing_check
                _ff_h, _, _ = front_facing_check(
                    cameras_mm, (in_result['Xe'][:3, :] / in_result['Xe'][3:4, :]).T)
                _h_max = float(getattr(config, 'handover_max_reproj_px', 1.0) or 1.0)
                _do_handover = (float(info_mm_ba.get('rms_after', np.inf)) < _h_max
                                and int(info_mm_ba.get('n_failed_rays', 0)) == 0 and _ff_h)
                if not _do_handover:
                    print(f"  [all-particles] handover skipped: central joint BA not good enough (limit {_h_max:g} px) "
                          f"(reproj {float(info_mm_ba.get('rms_after', np.nan)):.2f} px, failed rays "
                          f"{int(info_mm_ba.get('n_failed_rays', 0))}, front-facing {_ff_h})")
            if _do_handover:
                _gt_diag(f"iter={selfcal['count']} central particles, before the handover",
                         cameras_mm, in_result['Xe'], inliers['idx'])
                _central = (deepcopy(cam), deepcopy(inliers), in_result['Xe'].copy(),
                            deepcopy(cameras_mm), info_mm_ba)
                try:
                    if _Ws_obs_full is None:
                        class _FullData:
                            pass
                        _fd = _FullData(); _fd.Ws = _Ws_full_raw
                        _fd.camera_parameters = data.camera_parameters
                        _Ws_obs_full, _ = _correct_observations(_fd, CAMS)
                    _Wc = _Ws_obs_full - np.tile(principal_point, (1, FRAMES))
                    _idx_all = np.where(np.sum(_idm_full[:CAMS] > 0, axis=0) >= CAMS - NUM_CAMS_FILL)[0]
                    _n_seed = int(np.asarray(inliers['idx']).size)

                    def _build_full_cam(idx_keep):
                        _ca = []
                        for i in range(CAMS):
                            _il = np.where(_idm_full[i, :] > 0)[0]
                            _d = {'camId': config.cameras_2_use[i], 'idlin': _il, 'idin': _il,
                                  'xdist': _Ws_full_raw[3*i:3*i+3, _il]}
                            _d['xgt'] = _Wc[3*i:3*i+3, _il].copy() + config.principal_point[i, :].reshape(3, 1)
                            _d['xgtin'] = _d['xgt'].copy()
                            _ca.append(_d)
                        _inl = dict(inliers); _inl['idx'] = np.asarray(idx_keep, int)
                        _inl['IdMat'] = np.asarray(_idm_full[:CAMS])
                        return _ca, _inl

                    # ITERATIVE handover: each round places the remaining particles with the
                    # CURRENT cameras and runs the joint BA on everything that fits. Run
                    # evidence: one round from a 20-deg central solution placed only half of
                    # the outer particles, yet brought the tilt to 2.4 deg - cameras good
                    # enough to place more of them in a second round.
                    _MAX_ROUNDS = 3
                    for _round in range(1, _MAX_ROUNDS + 1):
                        _idx_cur = np.asarray(inliers['idx'], int)
                        _n_new = int(np.setdiff1d(_idx_all, _idx_cur).size)
                        if _n_new == 0:
                            break
                        _before = (deepcopy(cam), deepcopy(inliers), in_result['Xe'].copy(),
                                   deepcopy(cameras_mm), info_mm_ba)
                        # initial cloud: particles in use keep their positions; the others get
                        # a straight-line intersection of their rays from the current cameras
                        _col_cur = {int(f): c for c, f in enumerate(_idx_cur)}
                        _Xs = in_result['Xe'][:3, :] / in_result['Xe'][3:4, :]
                        _idx_try = _idx_all.copy()
                        _Xa = np.ones((4, _idx_try.size))
                        for _c, _f in enumerate(_idx_try):
                            if int(_f) in _col_cur:
                                _Xa[:3, _c] = _Xs[:, _col_cur[int(_f)]]
                                continue
                            _A = np.zeros((3, 3)); _b = np.zeros(3)
                            for _k, _cm in enumerate(cameras_mm):
                                if _idm_full[_k, _f] <= 0:
                                    continue
                                _x = _Wc[3*_k, _f] + config.principal_point[_k, 0]
                                _y = _Wc[3*_k+1, _f] + config.principal_point[_k, 1]
                                _v = np.array([-(_x - _cm.xh) / _cm.c, -(_y - _cm.yh) / _cm.c, 1.0])
                                _dv = np.asarray(_cm.rotation_matrix).T @ _v; _dv /= np.linalg.norm(_dv)
                                _M = np.eye(3) - np.outer(_dv, _dv); _A += _M; _b += _M @ np.asarray(_cm.position)
                            _Xa[:3, _c] = np.linalg.lstsq(_A, _b, rcond=None)[0]
                        _cam_all, _inl_all = _build_full_cam(_idx_try)
                        _cam_all = reprerror(_cam_all, in_result['Pe'], _Xa, FRAMES, _inl_all)
                        with _warnings.catch_warnings():
                            _warnings.simplefilter('ignore')
                            _, _Xa, _iref = multimedia_bundle_adjustment(
                                cameras_mm=cameras_mm, Xe=_Xa, cam=_cam_all, z_glass_top=config.z_glass_top,
                                opt={'cam_params': [], 'fix_points': False, 'max_nfev': 60,
                                     'ftol': 1e-6, 'xtol': 1e-6, 'gtol': 1e-6,
                                     'max_pt_step_mm': 400.0, 'allow_invalid_geometry': True,
                                     'verbose': False})
                        # prune NEW particles that still do not fit the current cameras
                        _chk = multimedia_point_residuals(deepcopy(_cam_all), cameras_mm, _Xa)
                        _worst = np.zeros(_idx_try.size)
                        for _k in range(CAMS):
                            for _col, _e in zip(_chk[_k]['recandvis'], _chk[_k]['err2d']):
                                _worst[_col] = max(_worst[_col], _e if np.isfinite(_e) else np.inf)
                        _is_new = np.array([int(f) not in _col_cur for f in _idx_try])
                        _bad = _is_new & (_worst > 5.0)
                        _n_added = _n_new - int(_bad.sum())
                        if _n_added == 0:
                            print(f"  [all-particles] round {_round}: none of the {_n_new} remaining particles "
                                  f"fits the current cameras -> handover complete")
                            break
                        if _bad.any():
                            _keep = ~_bad
                            _idx_try = _idx_try[_keep]; _Xa = _Xa[:, _keep]
                            _cam_all, _inl_all = _build_full_cam(_idx_try)
                            _cam_all = reprerror(_cam_all, in_result['Pe'], _Xa, FRAMES, _inl_all)
                        cam, inliers = _cam_all, _inl_all
                        in_result['Xe'] = _Xa
                        print(f"  [all-particles] round {_round}: {_idx_cur.size} -> {_idx_try.size} particles "
                              f"(added {_n_added}, {int(_bad.sum())} still do not fit"
                              + (f"; seed used {_n_seed} central particles" if _round == 1 else "")
                              + f"); cameras-held refinement {_iref.get('rms_before', float('nan')):.2f} -> "
                              f"{_iref.get('rms_after', float('nan')):.2f} px")
                        with _warnings.catch_warnings():
                            _warnings.simplefilter('ignore')
                            cameras_mm, in_result['Xe'], info_mm_ba = multimedia_bundle_adjustment(
                                cameras_mm=cameras_mm, Xe=in_result['Xe'], cam=cam,
                                z_glass_top=config.z_glass_top,
                                opt={'fix_points': False, 'cam_params': _cam_params, 'max_nfev': 80,
                                     'ftol': 1e-4, 'xtol': 1e-4, 'gtol': 1e-4,
                                     'max_pos_step_mm': 250.0, 'max_pt_step_mm': 200.0,
                                     'allow_invalid_geometry': True, **_dist_opt,
                                     'cam_dist_weight': _w_final, **_final_opt(_cam_params)})
                        _log_ba(f'jointBA round {_round}', info_mm_ba)
                        # a round must not make the solution worse
                        _ff_r, _, _ = front_facing_check(
                            cameras_mm, (in_result['Xe'][:3, :] / in_result['Xe'][3:4, :]).T)
                        if (int(info_mm_ba.get('n_failed_rays', 0)) > 0 or not _ff_r
                                or float(info_mm_ba.get('rms_after', np.inf)) >= 5.0):
                            cam, inliers, in_result['Xe'], cameras_mm, info_mm_ba = _before
                            print(f"  [all-particles] round {_round} made the solution worse -> "
                                  f"reverted to the state before it; handover stops")
                            break
                except Exception as _e:
                    cam, inliers, in_result['Xe'], cameras_mm, info_mm_ba = _central
                    print(f"  [all-particles] handover failed ({type(_e).__name__}: {_e}); "
                          f"restored the central-particle solution")

            _mm_rms_after = float(info_mm_ba.get('rms_after', float('nan')))
            _n_failed_last = int(info_mm_ba.get('n_failed_rays', 0))
            if info_mm_ba.get('cam_dist_pairs', 0):
                _mm_misfit = float(info_mm_ba.get('cam_dist_misfit', float('nan')))
            try:
                _Xb = in_result['Xe'][:3, :] / in_result['Xe'][3:4, :]
                _gb = config.z_glass_top - float(getattr(mm, 't_total', 0.0))
                _below = float(np.nanmean(_Xb[2, :] < _gb))
                print(f"  [mmBA] cloud Z range=[{np.nanmin(_Xb[2,:]):.1f},{np.nanmax(_Xb[2,:]):.1f}]mm "
                      f"particles below glass={100*_below:.0f}% "
                      f"(glass bottom z={_gb:.1f})")
            except Exception:
                pass

            # (3) cloud re-triangulation, cameras frozen — mode 'alternate' only
            if _ba_mode == 'alternate':
                try:
                    with _warnings.catch_warnings():
                        _warnings.simplefilter('ignore')
                        _cams_ret, in_result['Xe'], info_ret = multimedia_bundle_adjustment(
                            cameras_mm  = cameras_mm,
                            Xe          = in_result['Xe'],
                            cam         = cam,
                            z_glass_top = config.z_glass_top,
                            opt         = {
                                'cam_params'      : [],       # freeze cameras
                                'fix_points'      : False,    # optimise the cloud
                                'max_nfev'        : 60,
                                'ftol'            : 1e-4,
                                'xtol'            : 1e-4,
                                'gtol'            : 1e-4,
                                'max_pt_step_mm'  : 200.0,    # bound each point's move
                                'allow_invalid_geometry': True,
                            },
                        )
                    print(f"  [reTri] cloud re-triangulated vs corrected cameras: "
                          f"rms {info_ret.get('rms_before', float('nan')):.2f} -> "
                          f"{info_ret.get('rms_after', float('nan')):.2f}px")
                except Exception as _e:
                    print(f"  [reTri] skipped ({type(_e).__name__})")

            # ---- exact mirror-twin repair ---------------------------------------
            # The Maas projection cannot tell a solution from its mirror twin (world
            # mirrored through a vertical plane, every camera facing away): same
            # images, distances, heights and line-of-sight distances. If the BA ended
            # in that twin (wrong chirality AND every camera facing away from the
            # particles), map it to its exact correct-handed twin. The repaired
            # solution must still pass every validity check at selection - a mirror
            # is never accepted as a mirror.
            _cyc_t = getattr(config, 'camera_cyclic_order', None)
            if _cyc_t:
                from seed_gatekeeper import is_facing_away_mirror, mirror_twin
                _pts_t = (in_result['Xe'][:3, :] / in_result['Xe'][3:4, :]).T
                if is_facing_away_mirror(cameras_mm, _pts_t, _cyc_t):
                    cameras_mm, _pts_new = mirror_twin(cameras_mm, _pts_t)
                    in_result['Xe'][:3, :] = _pts_new.T * in_result['Xe'][3:4, :]
                    print("  [twin] BA ended in the facing-away mirror twin -> mapped to its exact "
                          "correct-handed twin (same images, distances, heights); re-validated at selection")

            # Sync Ce: camera centres in in_result must match corrected cameras_mm
            for _k in range(CAMS):
                in_result['Ce'][:3, _k] = cameras_mm[_k].position

            # Reconstruct Re from corrected angles so Re.dat is consistent with cameras_mm
            try:
                from scipy.spatial.transform import Rotation as _Rot
                Re_corrected = np.zeros_like(in_result['Re'])
                for _k, _cam in enumerate(cameras_mm):
                    Re_corrected[3*_k:3*_k+3, :] = _Rot.from_euler('xyz', _cam.angles).as_matrix()
                in_result['Re'] = Re_corrected
            except ImportError:
                pass  # scipy unavailable — Re.dat stays as euclidize() output

            # Rebuild LUT with corrected camera positions
            lut = MultiCameraLUT(cameras_mm, z_glass_top=config.z_glass_top)
            _lut_build(lut)

            # Re-run reprerror so final error metrics use corrected cameras
            cam = reprerror(cam, in_result['Pe'], in_result['Xe'], FRAMES, inliers)

        #######################################################################
        # Save camera matrices
        #######################################################################
        P = in_result['Pe']
        X = in_result['Xe']
        R = in_result['Re']
        C = in_result['Ce']

        np.savetxt(os.path.join(config.data_path, "Pmats.dat"), P.flatten(), fmt='%.6f')
        np.savetxt(os.path.join(config.data_path, "Xe.dat"), X, fmt='%.6f')
        np.savetxt(os.path.join(config.data_path, "Re.dat"), R.flatten(), fmt='%.6f')
        np.savetxt(os.path.join(config.data_path, "Ce.dat"), C, fmt='%.6f')

        # Save calibration parameters
        in_result['Cst'], in_result['Rot'] = savecalpar(in_result['Pe'], config)

        # Save multimedia params + per-camera files (rewritten at the end of the run
        # from the RETURNED solution — see FINALISE OUTPUTS)
        if mm is not None:
            _save_mm_camera_files(cameras_mm)

        #######################################################################
        # Alignment routines — compare results to ground truth
        #######################################################################
        # Triggered when config.align_existing is a non-empty filename string.
        # Accepts two ground truth formats:
        #   .dat  — camera centres only (e.g. original_cam_centers.dat)
        #   .json — full ground truth with cameras + particles (e.g. ground_truth.json)
        # Produces alignment_comparison.png and alignment_report.txt in data_path.
        align: dict = run_alignment(in_result, config)

        # --- best-result selection ---------------------------------------------
        # Read the post-alignment RMS (mm). Prefer the align dict; fall back to the
        # report file that run_alignment writes; otherwise skip selection this pass.
        _align_rms = None
        if isinstance(align, dict):
            for _k in ('rms_after', 'rms', 'alignment_rms', 'rms_after_mm'):
                if _k in align and align[_k] is not None:
                    _align_rms = float(align[_k]); break
        if _align_rms is None:
            try:
                _rep = os.path.join(config.data_path, 'alignment_report.txt')
                with open(_rep) as _f:
                    for _line in _f:
                        if 'after' in _line.lower() and 'align' in _line.lower():
                            import re as _re
                            _m = _re.search(r'([0-9]+\.[0-9]+)', _line)
                            if _m: _align_rms = float(_m.group(1)); break
            except Exception:
                _align_rms = None

        # VALIDITY: the single gatekeeper verdict (seed_ok) decides geometry; the
        # only extra requirement is that the BA actually produced a result to
        # select (ba_engaged). No separate geometry_ok / deref_converged gates.
        # SELECTION: the camera-distance MISFIT (mm) is the primary in-situ metric
        # — it tracks ground-truth alignment far more tightly than reprojection —
        # and DOMINATES the choice. Reprojection error (px) is only a tie-breaker
        # between iterations whose misfit is within MISFIT_TIE_MM of each other.
        # Both are available in deployment (no ground truth); alignment is printed
        # only to expose the misfit->truth relationship.
        MISFIT_TIE_MM = 2.0
        _ba_engaged = np.isfinite(_mm_rms_after)
        _reproj_history.append((f"iter {selfcal['count']}", _mm_rms_after))
        # Chirality AFTER the BA: the gatekeeper vetted the seed's handedness, but
        # the camera BA (trust region 250 mm) can still carry the constellation
        # into its mirror image, which scores perfectly on misfit AND reprojection.
        # Same function as the gatekeeper (single authority for chirality logic).
        _chir_ok = True
        _chir_str = 'n/a'
        _cyc = getattr(config, 'camera_cyclic_order', None)
        if _ba_engaged and _cyc and cameras_mm is not None:
            from seed_gatekeeper import camera_chirality_sign
            _chir = camera_chirality_sign(
                np.array([c.position for c in cameras_mm]), _cyc)
            _chir_ok = (_chir >= 0)
            _chir_str = {1: '+1', -1: '-1 MIRROR', 0: '0 degenerate'}[_chir]
        # Camera-glass line of sight (in-situ, hand-measured): a solution whose rig
        # sits at the wrong height/tilt relative to the glass fails here even when
        # misfit and reprojection look good. Rule + tolerance live in seed_gatekeeper.
        _los_ok, _los_str = True, 'n/a'
        if _ba_engaged and getattr(config, 'camera_los', None) and cameras_mm is not None:
            from seed_gatekeeper import los_check
            _los_ok, _los_devs, _los_reason = los_check(cameras_mm, config.camera_los,
                                                        config.z_glass_top)
            if _los_devs:
                _los_str = (f"max {max(abs(v) for v in _los_devs.values()):.1f}mm"
                            + ('' if _los_ok else ' FAIL'))
        # Front-facing ("cheirality"): the particles must lie in front of every camera.
        # The projection model does not enforce it, so the facing-away mirror twin
        # would otherwise pass with perfect in-situ numbers. Rule in seed_gatekeeper.
        _ff_ok, _ff_str = True, 'n/a'
        if _ba_engaged and cameras_mm is not None:
            from seed_gatekeeper import front_facing_check
            _ff_ok, _ff_fr, _ = front_facing_check(
                cameras_mm, (in_result['Xe'][:3, :] / in_result['Xe'][3:4, :]).T)
            _ff_str = f"min {100*np.nanmin(_ff_fr):.0f}%" + ('' if _ff_ok else ' FAIL')
        # Failed rays: the model could not compute some observations (geometry on
        # the wrong side of the glass) - never a valid calibration.
        _rays_ok = (_n_failed_last == 0)
        _valid = bool(seed_ok) and _ba_engaged and _chir_ok and _los_ok and _ff_ok and _rays_ok
        # if the constraint is off (no known distances) misfit is nan -> fall back
        # to reprojection-only selection so legacy behaviour is preserved.
        _has_misfit = np.isfinite(_mm_misfit)

        _align_str = f"{_align_rms:.2f}mm" if _align_rms is not None else "n/a"
        _best_str = (f"misfit={best['misfit']:.2f}mm" if np.isfinite(best['misfit'])
                     else f"reproj={best['rms']:.2f}px")
        print(f"  [select] iter={selfcal['count']} "
              f"misfit={_mm_misfit:.2f}mm reproj={_mm_rms_after:.2f}px "
              f"alignment={_align_str} valid={_valid} "
              f"(seed_ok={seed_ok}, ba_engaged={_ba_engaged}, chirality={_chir_str}, los={_los_str}, front={_ff_str}"
              f"{'' if _rays_ok else f', FAILED rays={_n_failed_last}'})  "
              f"best so far: {_best_str}")

        # ---- per-iteration ground-truth diagnostic (NEVER used for selection) ------
        if _ba_engaged and cameras_mm is not None:
            _gt_diag(f"iter={selfcal['count']}", cameras_mm, in_result['Xe'], inliers['idx'])

        if _valid:
            _take = False
            if _light_final:
                # light final distance weight: the misfit reflects the distance-
                # measurement errors, so the images decide (misfit only bounded)
                _take = _mm_rms_after < best['rms']
            elif _has_misfit and np.isfinite(best['misfit']):
                # misfit dominates; reprojection breaks near-ties
                if _mm_misfit < best['misfit'] - MISFIT_TIE_MM:
                    _take = True
                elif abs(_mm_misfit - best['misfit']) <= MISFIT_TIE_MM:
                    _take = _mm_rms_after < best['rms']
            elif _has_misfit:            # first misfit-bearing result
                _take = True
            else:                        # no known distances -> reprojection-only
                _take = _mm_rms_after < best['rms']
            if _take:
                best.update(misfit=(_mm_misfit if _has_misfit else float('inf')),
                            rms=_mm_rms_after, iter=selfcal['count'],
                            align_mm=(_align_rms if _align_rms is not None else float('nan')),
                            cam=deepcopy(cam), in_result=deepcopy(in_result),
                            cameras_mm=deepcopy(cameras_mm),
                            inlier_idx=np.array(inliers['idx']).copy(),
                            inliers=deepcopy(inliers))

        # ---- EARLY STOP -----------------------------------------------------
        # A valid iteration whose in-situ metrics are already good enough — misfit
        # below EARLY_STOP_MISFIT_MM and reprojection below EARLY_STOP_REPROJ_PX —
        # is accepted immediately: further global iterations would only draw fresh
        # RANSAC sets in the hope of a marginally better one, whereas the polish
        # pass below will refine THIS solution further. Both thresholds are in-situ
        # (no ground truth), so this is a deployment-safe stopping rule.
        if (_valid and _has_misfit
                and _mm_misfit < EARLY_STOP_MISFIT_MM
                and _mm_rms_after < EARLY_STOP_REPROJ_PX):
            print(f"  [early-stop] iteration {selfcal['count']} meets thresholds "
                  f"(misfit {_mm_misfit:.2f} < {EARLY_STOP_MISFIT_MM} mm, "
                  f"reproj {_mm_rms_after:.2f} < {EARLY_STOP_REPROJ_PX} px) -> "
                  f"stopping the main loop and proceeding to the polish pass.")
            selfcal['iterate'] = False

        #######################################################################
        # Evaluate reprojection error (stats only; graphs are drawn once at the end)
        #######################################################################
        # NOTE: these are PINHOLE-model residuals (reprerror). In the multimedia
        # pipeline they are not the calibration metric (the BA's multimedia
        # reprojection on the [select] line is), so they are computed silently and
        # only feed the legacy global-iteration control below.
        cam = evalreprerror(cam, config, do_plot=False,
                            model='pinhole', verbose=(mm is None))

        #######################################################################
        # Save 2D-3D correspondences for further processing
        #######################################################################
        _save_correspondences(cam, in_result, align)

        #######################################################################
        # Update iteration control
        #######################################################################
        _timing['iterations'].append((int(selfcal['count']), _time.perf_counter() - _t_iter))
        selfcal['count'] += 1

        # Check if global iteration should continue
        max_mean_err = max([c['mean2Derr'] for c in cam])

        if (not selfcal['iterate']):
            # early-stop (or a prior decision) already ended the loop — honour it
            pass
        elif (max_mean_err > config.global_iteration_threshold and config.do_global_iterations and
            selfcal['count'] < config.global_iteration_max):

            # Estimate radial distortion and iterate further
            # selfcalib = goradf(config, selfcal['par2estimate'], INL_TOL)
            selfcal['iterate'] = True
            # (distortion is corrected once at load — no per-iteration switch)

            # Reduce tolerance threshold if estimation was successful.
            # STANDARD (pinhole) pipeline: progressively tighten the inlier
            # tolerance each global iteration — this refines the inlier set and is
            # important for convergence there.
            # MULTIMEDIA pipeline: keep the tolerance FIXED. Tightening starves the
            # later de-refraction iterations (too few surviving points -> sparse,
            # unstable reconstruction -> gatekeeper rejects every seed), which
            # produced the degraded final iterations. Holding it steady keeps the
            # later iterations healthy and contributing.
            if mm is None:
                INL_TOL = max([(2/3) * INL_TOL, config.global_iteration_threshold])
            else:
                print(f"  [tolerance] multimedia on -> holding INL_TOL fixed at "
                      f"{INL_TOL:.2f} (no tightening)")

            # Add additional parameters to estimate based on iteration count
            if config.nonlinear_update[3]:
                selfcal['par2estimate'][3] = 1  # Second radial parameter
            if selfcal['count'] > 1 and config.nonlinear_update[1]:
                selfcal['par2estimate'][1] = 1  # Principal point
            if selfcal['count'] > 3 and all(config.nonlinear_update[4:6]):
                selfcal['par2estimate'][4] = 1  # Tangential distortion
                selfcal['par2estimate'][5] = 1
        else:
            # End the iteration
            selfcal['iterate'] = False

    ###########################################################################
    # End of main global cycle
    ###########################################################################
    print()
    print(f"Calibration completed after {selfcal['count']} iteration(s)")
    if best['iter'] is not None:
        _misfit_str = (f"{best['misfit']:.2f}mm" if np.isfinite(best['misfit']) else "n/a")
        print(f"BEST result by CAMERA-DISTANCE MISFIT (in-situ, misfit-dominant with "
              f"reprojection tie-break): iteration {best['iter']} "
              f"with misfit={_misfit_str} reproj={best['rms']:.2f}px  ->  "
              f"ground-truth alignment={best['align_mm']:.2f}mm (returned).")
        print("  (Selection used only in-situ metrics — misfit then reprojection, no ground "
              "truth — to mirror deployment.)")
        cam, in_result = best['cam'], best['in_result']
        _final_cams = best.get('cameras_mm')   # replaced by the polished cameras if accepted

        # ---- POLISH PASS ---------------------------------------------------------
        # Take the selected best reconstruction and settle it: a final, tightly-
        # converged bundle adjustment started FROM the good solution. The trust
        # region is released (we are already near truth, so there is no risk of a
        # wild excursion, and the ±250 mm leash would otherwise cap the last small
        # corrections), tolerances are tightened by four orders of magnitude, and
        # the distance penalty stays active so the pinned scale is preserved. Two
        # tightly-converged half-steps — cameras, then cloud — are alternated a few
        # times; each is bounded only loosely. The polished result is kept only if
        # it does not worsen the in-situ misfit, so the pass can never degrade the
        # returned calibration.
        # CONVERGENCE VERDICT (in-situ): the run counts as converged only if the
        # selected solution itself meets the early-stop thresholds. Polishing a
        # solution that never converged can drive its misfit and reprojection to
        # good-looking values while it stays far from truth (a wrong basin or a
        # mirror), which would be indistinguishable from success in deployment.
        # So a non-converged run is reported as such and NOT polished.
        _calib_converged = bool(np.isfinite(best['misfit'])
                                and best['misfit'] < EARLY_STOP_MISFIT_MM
                                and best['rms'] < EARLY_STOP_REPROJ_PX)
        if (mm is not None and getattr(config, 'camera_distances', None)
                and not _calib_converged):
            print('************************************************************')
            print(f"CALIBRATION DID NOT CONVERGE: best iteration {best['iter']} has "
                  f"misfit={best['misfit']:.2f} mm, reproj={best['rms']:.2f} px "
                  f"(thresholds: misfit < {EARLY_STOP_MISFIT_MM} mm and reproj < "
                  f"{EARLY_STOP_REPROJ_PX} px). Polish SKIPPED; returning the "
                  f"unpolished best — treat this calibration as UNRELIABLE and re-run.")
        if (mm is not None and best.get('cameras_mm') is not None
                and getattr(config, 'camera_distances', None)
                and _calib_converged):
            _t_polish = _time.perf_counter()
            _phase[0] = 'polish: '
            import warnings as _warnings
            _known = config.camera_distances
            _w = _w_final
            _cams_p = deepcopy(best['cameras_mm'])
            _Xe_p = in_result['Xe'].copy()
            _cam_obs = best['cam']

            def _misfit_of(cams):
                import numpy as _np
                errs = []
                for (i, j, d) in _known:
                    errs.append(abs(float(_np.linalg.norm(cams[i].position - cams[j].position)) - d))
                return float(_np.sqrt(_np.mean(_np.square(errs)))) if errs else float('nan')

            _m0 = _misfit_of(_cams_p)
            _a0 = _gt_align_rms(in_result)
            _phase_boundaries.append((len(_reproj_history), 'polish'))
            _reproj_history.append(('polish start', best['rms']))
            print('************************************************************')
            print(f"POLISH PASS (final tight refinement from the best solution; "
                  f"starting misfit={_m0:.2f}mm reproj={best['rms']:.2f}px"
                  + (f" alignment={_a0:.2f}mm" if _a0 is not None else "") + ")")
            _tight = dict(ftol=1e-8, xtol=1e-8, gtol=1e-8, max_nfev=200,
                          allow_invalid_geometry=True,
                          cam_dist=_known, cam_dist_weight=_w)
            try:
                with _warnings.catch_warnings():
                    _warnings.simplefilter('ignore')
                    _pmode = getattr(config, 'ba_mode', 'alternate')
                    _pcp = getattr(config, 'ba_cam_params', [0, 1, 2, 3, 4, 5])
                    for _cyc in range(3 if _pmode == 'alternate' else 1):
                        if _pmode == 'alternate':
                            # cameras (penalty on), released trust region
                            _cams_p, _Xe_p, _iC = multimedia_bundle_adjustment(
                                cameras_mm=_cams_p, Xe=_Xe_p, cam=_cam_obs,
                                z_glass_top=config.z_glass_top,
                                opt={**_tight, 'fix_points': True, 'max_pos_step_mm': 1e4,
                                     'cam_params': _pcp, **_final_opt(_pcp)})
                            # cloud (cameras frozen), released bound
                            _cams_p, _Xe_p, _iP = multimedia_bundle_adjustment(
                                cameras_mm=_cams_p, Xe=_Xe_p, cam=_cam_obs,
                                z_glass_top=config.z_glass_top,
                                opt={**_tight, 'cam_params': [], 'fix_points': False,
                                     'max_pt_step_mm': 1e4})
                        else:
                            # one tight JOINT BA (cameras + points), released bounds
                            _cams_p, _Xe_p, _iP = multimedia_bundle_adjustment(
                                cameras_mm=_cams_p, Xe=_Xe_p, cam=_cam_obs,
                                z_glass_top=config.z_glass_top,
                                opt={**_tight, 'fix_points': False, 'cam_params': _pcp,
                                     'max_pos_step_mm': 1e4, 'max_pt_step_mm': 1e4, **_final_opt(_pcp)})
                        # ground-truth alignment for this cycle (if GT available):
                        # build a temporary in_result from the polished state.
                        _tmp = {k: (v.copy() if hasattr(v, 'copy') else v)
                                for k, v in in_result.items()}
                        for _k in range(CAMS):
                            _tmp['Ce'][:3, _k] = _cams_p[_k].position
                        _tmp['Xe'] = _Xe_p
                        _acyc = _gt_align_rms(_tmp)
                        print(f"  [polish] {'cycle ' + str(_cyc+1) if _pmode == 'alternate' else 'joint'}: "
                              f"reproj={_iP.get('rms_after', float('nan')):.3f}px "
                              f"misfit={_misfit_of(_cams_p):.2f}mm"
                              + (f" alignment={_acyc:.2f}mm" if _acyc is not None else ""))
                        _reproj_history.append((f"polish {_cyc+1}",
                                                float(_iP.get('rms_after', float('nan')))))
                _m1 = _misfit_of(_cams_p)
                _r1 = float(_iP.get('rms_after', float('nan')))
                # accept if misfit does not degrade beyond a sub-noise tolerance
                # (misfit already sits at the ~mm measurement-noise floor, so a
                # fraction-of-a-mm wiggle is meaningless; this lets the polish keep
                # a tightly-converged reprojection improvement).
                _POLISH_MISFIT_TOL = 0.3
                _cyc_p = getattr(config, 'camera_cyclic_order', None)
                # exact mirror-twin repair (same rule as after the main BA step)
                if _cyc_p:
                    from seed_gatekeeper import is_facing_away_mirror, mirror_twin
                    _pts_p = (_Xe_p[:3, :] / _Xe_p[3:4, :]).T
                    if is_facing_away_mirror(_cams_p, _pts_p, _cyc_p):
                        _cams_p, _pts_pn = mirror_twin(_cams_p, _pts_p)
                        _Xe_p = _Xe_p.copy()
                        _Xe_p[:3, :] = _pts_pn.T * _Xe_p[3:4, :]
                        print("  [polish] ended in the facing-away mirror twin -> mapped to its "
                              "exact correct-handed twin before the checks")
                # chirality re-check: the polish runs with the trust region
                # released, so it could carry the constellation into its mirror.
                _chir_p = 1
                if _cyc_p:
                    from seed_gatekeeper import camera_chirality_sign
                    _chir_p = camera_chirality_sign(
                        np.array([c.position for c in _cams_p]), _cyc_p)
                print(f"  [polish] chirality of polished cameras: "
                      f"{ {1: '+1 (ok)', -1: '-1 (MIRROR)', 0: '0 (degenerate)'}[_chir_p] }")
                _los_p_ok, _los_p_reason = True, ''
                if getattr(config, 'camera_los', None):
                    from seed_gatekeeper import los_check
                    _los_p_ok, _, _los_p_reason = los_check(_cams_p, config.camera_los,
                                                            config.z_glass_top)
                from seed_gatekeeper import front_facing_check
                _ff_p_ok, _, _ff_p_reason = front_facing_check(_cams_p, (_Xe_p[:3, :] / _Xe_p[3:4, :]).T)
                _nf_p = int(_iP.get('n_failed_rays', 0)) if isinstance(_iP, dict) else 0
                if _nf_p > 0:
                    print(f"  [polish] rejected ({_nf_p} failed rays); keeping the pre-polish best")
                elif _chir_p < 0:
                    print("  [polish] rejected (polished constellation is the MIRROR image); "
                          "keeping the pre-polish best")
                elif not _ff_p_ok:
                    print(f"  [polish] rejected ({_ff_p_reason}); keeping the pre-polish best")
                elif not _los_p_ok:
                    print(f"  [polish] rejected ({_los_p_reason}); keeping the pre-polish best")
                elif np.isfinite(_m1) and ((not _light_final and _m1 <= _m0 + _POLISH_MISFIT_TOL)
                                           or (_light_final and _m1 <= EARLY_STOP_MISFIT_MM
                                               and np.isfinite(_r1) and _r1 <= best['rms'] + 0.02)):
                    for _k in range(CAMS):
                        in_result['Ce'][:3, _k] = _cams_p[_k].position
                    in_result['Xe'] = _Xe_p
                    # The returned solution is now the POLISHED one. Reprojection
                    # statistics, graphs and all output files are produced once,
                    # from this solution, in FINALISE OUTPUTS below.
                    _final_cams = _cams_p
                    _a1 = _gt_align_rms(in_result)
                    if _a1 is not None:
                        best['align_mm'] = _a1     # report the POLISHED camera alignment
                    print(f"  [polish] ACCEPTED: misfit {_m0:.2f} -> {_m1:.2f} mm, "
                          f"reproj {best['rms']:.2f} -> {_r1:.3f} px"
                          + (f", alignment {_a0:.2f} -> {_a1:.2f} mm"
                             if (_a0 is not None and _a1 is not None) else "")
                          + " (outputs re-written from the polished solution at the end)")
                else:
                    print(f"  [polish] rejected (misfit would rise {_m0:.2f} -> {_m1:.2f} mm); "
                          f"keeping the pre-polish best")
            except Exception as _e:
                print(f"  [polish] skipped ({type(_e).__name__}: {_e})")
            _timing['polish'] = _time.perf_counter() - _t_polish
            _phase[0] = ''
    else:
        print("NO VALID ITERATION: every iteration failed a validity check (seed, chirality "
              "or camera-glass line of sight - see the [select] lines). Returning the LAST "
              "iteration's result, which is UNRELIABLE.")
        _calib_converged = False
    print('************************************************************')

    # ---- GROUND-TRUTH DIAGNOSTICS (ground truth used ONLY here) ---------------
    # Two views of the same returned solution:
    #   SHAPE ONLY  - similarity fit (free rotation, translation, scale). Measures
    #                 the shape of the camera constellation / particle cloud. It is
    #                 BLIND to the rig's height and tilt relative to the glass.
    #   GLASS FRAME - only rotation about the vertical + horizontal shift allowed
    #                 (the freedoms that leave the glass plane unchanged). Height,
    #                 tilt and orientation errors relative to the refractive
    #                 interface stay in the residual: this is the honest accuracy.
    # The glass-frame transform is fitted on the CAMERA POSITIONS and then applied
    # to cameras, optical axes and particles alike.
    # Particle j of the cloud corresponds to ground-truth particle
    # best['inlier_idx'][j] (one particle per frame).
    try:
        import json as _json
        _gt_path = os.path.join(config.data_path, 'ground_truth.json')
        if os.path.exists(_gt_path) and best.get('inlier_idx') is not None:
            _gt = _json.load(open(_gt_path))
            _gt_particles = np.asarray(_gt.get('particles_mm', []), float)
            _cloud = (in_result['Xe'][:3, :] / in_result['Xe'][3:4, :]).T
            _idx = np.asarray(best['inlier_idx'], int)
            _pts_ok = (_gt_particles.ndim == 2 and _gt_particles.shape[0] > 0
                       and _cloud.shape[0] == _idx.shape[0]
                       and _idx.max() < _gt_particles.shape[0])
            _gt_sub = _gt_particles[_idx] if _pts_ok else None
            _gtc = {int(c['id']): c for c in _gt.get('cameras', [])}
            _pairs = [(_co, _gtc[int(_co.cam_id)]) for _co in (_final_cams or [])
                      if int(_co.cam_id) in _gtc]

            def _rms(e):
                return float(np.sqrt(np.mean(np.sum(e ** 2, axis=1))))

            print("====================  GROUND-TRUTH DIAGNOSTICS  ====================")
            print("  (ground truth is used for these numbers only; the calibration never sees it)")
            print("  SHAPE ONLY  (similarity fit: free rotation, translation, scale)")
            if len(_pairs) >= 3:
                _Cr = np.array([p[0].position for p in _pairs], float)
                _Cg = np.array([p[1]['position_mm'] for p in _pairs], float)
                _R, _s, _t = _similarity_fit(_Cr, _Cg)
                print(f"    cameras   : RMS {_rms(_s * _Cr @ _R.T + _t - _Cg):8.3f} mm")
            if _pts_ok:
                _Rp, _sp, _tp = _similarity_fit(_cloud, _gt_sub)
                _ep = np.linalg.norm(_sp * _cloud @ _Rp.T + _tp - _gt_sub, axis=1)
                print(f"    particles : RMS {np.sqrt(np.mean(_ep ** 2)):8.3f} mm   "
                      f"(median {np.median(_ep):.3f}, 90th pct {np.percentile(_ep, 90):.3f} mm)")

            if len(_pairs) >= 3:
                _Rz, _tz = _glass_frame_fit(_Cr, _Cg)
                _Cw = _Cr @ _Rz.T + _tz
                _dh = _Cw[:, 2] - _Cg[:, 2]
                _tilt = _angle_deg(_R @ np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 1.0]))
                print("  GLASS FRAME (only horizontal shift + rotation about the vertical allowed)")
                print(f"    cameras   : RMS {_rms(_Cw - _Cg):8.3f} mm   | height error: "
                      + "  ".join(f"cam{int(p[0].cam_id)} {d:+.1f}" for p, d in zip(_pairs, _dh)) + " mm")
                print(f"    rig tilt relative to the glass : {_tilt:.2f} deg")
                print("    camera pointing error          : "
                      + "  ".join(f"cam{int(p[0].cam_id)} "
                                  f"{_angle_deg(_Rz @ _axis_down(p[0].rotation_matrix), _axis_down(p[1]['rotation_matrix'])):.2f}"
                                  for p in _pairs) + " deg")
                if _pts_ok:
                    _Pw = _cloud @ _Rz.T + _tz
                    _ew = np.linalg.norm(_Pw - _gt_sub, axis=1)
                    print(f"    particles : RMS {np.sqrt(np.mean(_ew ** 2)):8.3f} mm   "
                          f"(median {np.median(_ew):.3f}, 90th pct {np.percentile(_ew, 90):.3f} mm)  "
                          f"| mean depth error {np.mean(_Pw[:, 2] - _gt_sub[:, 2]):+.1f} mm")
            else:
                print("  GLASS FRAME : skipped (needs >= 3 cameras of the returned solution)")
            if _calib_converged is False:
                print("  NOTE: run did NOT converge - these numbers describe an unreliable calibration.")
            print("=====================================================================")
    except Exception as _e:
        print(f"  [gt-diagnostics] skipped ({type(_e).__name__}: {_e})")

    # ---- FINALISE OUTPUTS (once, from the RETURNED solution) -----------------
    # Every reported reprojection number and every output file must describe the
    # solution the run returns (polished if the polish was accepted, otherwise the
    # selected best iteration) — not the last iteration and not a stale pinhole
    # projection. In the multimedia pipeline the reprojection is evaluated with
    # the BA's OWN model and observations (multimedia_point_residuals), so the
    # 'RMS per axis' printed here equals the BA's 'reproj' value.
    if mm is not None and best.get('iter') is not None and _final_cams is not None:
        print('************************************************************')
        print(f"FINAL OUTPUTS for the returned solution (iteration {best['iter']}, "
              f"{'polished' if _final_cams is not best.get('cameras_mm') else 'unpolished'}):")
        try:
            cam = multimedia_point_residuals(deepcopy(best['cam']), _final_cams,
                                             in_result['Xe'])
            cam = evalreprerror(cam, config, do_plot=True, model='multimedia')
        except Exception as _e:
            print(f"  [warn] multimedia reprojection failed ({type(_e).__name__}: {_e}); "
                  f"graph not produced.")
        if getattr(config, 'camera_los', None):
            try:
                from seed_gatekeeper import los_check, camera_los_distance, LOS_TOLERANCE_MM
                _lok, _ldevs, _ = los_check(_final_cams, config.camera_los, config.z_glass_top)
                _meas = dict(config.camera_los)
                print("  [los] camera-glass line of sight (in-situ, hand-measured vs recovered): "
                      + "  ".join(f"cam{config.cameras_2_use[i]} {_meas[i]:.1f} vs "
                                  f"{camera_los_distance(_final_cams[i], config.z_glass_top):.1f} mm "
                                  f"({v:+.1f})" for i, v in sorted(_ldevs.items()))
                      + f"  -> {'OK' if _lok else 'FAIL'} (tolerance ±{LOS_TOLERANCE_MM:g} mm)")
            except Exception as _e:
                print(f"  [los] check skipped ({type(_e).__name__}: {_e})")
        try:
            _save_mm_camera_files(_final_cams)
        except Exception as _e:
            print(f"  [warn] camera files not written ({type(_e).__name__}: {_e})")
        try:
            _final_align = run_alignment(in_result, config)   # alignment report + plot
        except Exception:
            _final_align = None
        try:
            _save_correspondences(cam, in_result, _final_align)
            print("  Correspondences (points4cal*.dat) saved for the returned solution.")
        except Exception as _e:
            print(f"  [warn] correspondences not written ({type(_e).__name__}: {_e})")
    else:
        # standard (pinhole) pipeline: pinhole residuals ARE the calibration metric
        try:
            cam = evalreprerror(cam, config, do_plot=True,
                                model=('pinhole' if mm is None else None))
        except Exception as _e:
            print(f"  [warn] final reprojection plot skipped ({type(_e).__name__}: {_e})")
    plot_reproj_history(_reproj_history, config, _phase_boundaries)

    # ---- FINAL INTRINSICS (estimate mode: what the data determined) ---------
    if _est_intr and _final_cams:
        _parts = []
        for _k, _co in enumerate(_final_cams):
            _t = f"cam{_k+1} f={_co.c:.1f} pp=({_co.xh:.1f},{_co.yh:.1f})"
            if _rad_reference and _k < len(_rad_reference):
                _K = np.asarray(_rad_reference[_k]['intrinsic_matrix'], float)
                _t += (f" [.rad f={_K[0,0]:.1f} ({100*(_co.c/_K[0,0]-1):+.2f}%), "
                       f"pp=({_K[0,2]:.1f},{_K[1,2]:.1f})]")
            _parts.append(_t)
        print("[intrinsics] ESTIMATED (final solution): " + " | ".join(_parts))

    # ---- RUN VERDICT ---------------------------------------------------------
    if _calib_converged is not None:
        print()
        print('RUN VERDICT: ' + ('CONVERGED — calibration usable' if _calib_converged
              else 'NOT CONVERGED — calibration unreliable, re-run'))

    # ---- TIMING REPORT ------------------------------------------------------
    _timing['total'] = _time.perf_counter() - _T0
    _iter_total = sum(t for _, t in _timing['iterations'])
    print()
    print('====================  TIMING  ====================')
    print(f"  Setup (config + data load)   : {_timing['setup']:8.2f} s")
    for _n, _t in _timing['iterations']:
        print(f"  Global iteration {_n:<2d}          : {_t:8.2f} s")
    print(f"  All iterations (subtotal)    : {_iter_total:8.2f} s"
          + (f"   [{_iter_total/len(_timing['iterations']):.2f} s/iter avg]"
             if _timing['iterations'] else ""))
    print(f"  Polish pass                  : {_timing['polish']:8.2f} s")
    print(f"  --------------------------------------------")
    print(f"  TOTAL RUNTIME                : {_timing['total']:8.2f} s"
          f"   ({_timing['total']/60.0:.2f} min)")
    _covered = sum(_stage_t.values())
    print('  -- by stage (exclusive; nested stages not double-counted) --')
    for _nm, _tt in sorted(_stage_t.items(), key=lambda kv: -kv[1]):
        print(f"  {_nm:<37s}: {_tt:8.1f} s  {100*_tt/_timing['total']:5.1f}%  "
              f"({_stage_n[_nm]} call{'s' if _stage_n[_nm] != 1 else ''})")
    print(f"  {'other (not instrumented)':<37s}: {_timing['total']-_covered:8.1f} s  "
          f"{100*(_timing['total']-_covered)/_timing['total']:5.1f}%")
    print('==================================================')

    return cam, in_result, config


if __name__ == '__main__':
    cam, results, config = main()
