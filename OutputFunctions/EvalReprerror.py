import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend: renders to file, never opens a GUI window
import matplotlib.pyplot as plt
import os
import subprocess
import sys


def evalreprerror(cam, config, do_plot=True, model=None, verbose=True):
    """
    Compute (and optionally plot) final 2D reprojection error statistics.

    The reprojection statistics (mean2Derr / std2Derr per camera) are computed and
    written onto `cam` BEFORE any plotting, so they are always available to the
    caller even if plotting or file-saving fails. Pass do_plot=False to compute the
    statistics WITHOUT rendering/saving any figure — used for the per-iteration and
    polish calls so that graphs are produced only once, at the end of the run.
    Plotting and saving are best-effort: a rendering error, non-finite data on a
    poor run, or a locked output file must never crash the calibration.

    model   : label of the projection model that produced err2d ('multimedia'
              or 'pinhole'); printed and shown on the graph so numbers from
              different models can never be confused.
    verbose : print the statistics (False = compute silently).
    The per-axis RMS printed here, sqrt(mean(err2d^2)/2), is the SAME
    definition as the bundle adjustment's 'reproj' value.
    """
    # --- Overall 2D reprojection error (finite values only) ------------------
    all_err2d = np.concatenate([np.asarray(c['err2d']).ravel() for c in cam])
    finite = all_err2d[np.isfinite(all_err2d)]
    n_bad = int(all_err2d.size - finite.size)
    mean_err2d = float(np.mean(finite)) if finite.size else float('nan')
    std_err2d = float(np.std(finite)) if finite.size else float('nan')

    rms_axis = float(np.sqrt(np.mean(finite ** 2) / 2.0)) if finite.size else float('nan')
    if verbose:
        print('2D reprojection error' + (f' ({model} model)' if model else ''))
        print(f'All points: mean  {mean_err2d:.2f} pixels, std is {std_err2d:.2f}'
              f'  |  RMS per axis {rms_axis:.3f} px (same definition as the BA reproj)'
              + (f'   ({n_bad} non-finite point(s) excluded)' if n_bad else ''))

        if (not np.isfinite(mean_err2d)) or mean_err2d > 1.5 or std_err2d > 1.5:
            print('***************************************************')
            print('W A R N I N G: the reprojection error is relatively high !')

    # --- Per-camera statistics (computed and stored BEFORE any plotting) ------
    for i in range(len(config.cameras_2_use)):
        e = np.asarray(cam[i]['err2d']).ravel()
        ef = e[np.isfinite(e)]
        cam[i]['mean2Derr'] = float(np.mean(ef)) if ef.size else float('nan')
        cam[i]['std2Derr'] = float(np.std(ef)) if ef.size else float('nan')

    # --- Plotting + saving is best-effort: never let it crash the pipeline ----
    if do_plot:
        try:
            _plot_reprojection(cam, config, model=model, rms_axis=rms_axis)
        except Exception as e:
            print(f'  [warn] reprojection-error plotting/saving skipped '
                  f'({type(e).__name__}: {e}); calibration result is unaffected.')

    return cam


def _plot_reprojection(cam, config, model=None, rms_axis=None):
    """Render and save the two reprojection-error figures. Non-finite per-camera
    values are shown as zero so a bad run cannot produce an invalid figure size."""
    cam_ids = config.cameras_2_use
    mean_errors = [cam[i]['mean2Derr'] for i in range(len(cam))]
    std_errors  = [cam[i]['std2Derr']  for i in range(len(cam))]
    mean_plot = [m if np.isfinite(m) else 0.0 for m in mean_errors]
    std_plot  = [s if np.isfinite(s) else 0.0 for s in std_errors]

    # --- Figure 30: Line plot ---
    fig30, ax30 = plt.subplots()
    ax30.plot(cam_ids, mean_plot, 'bd', label='Mean 2D error')
    ax30.plot(cam_ids, mean_plot, 'b-')
    ax30.plot(cam_ids, std_plot,  'rd', label='STD 2D error')
    ax30.plot(cam_ids, std_plot,  'r-')
    ax30.grid(True)
    ax30.set_xlabel('Id of the camera')
    ax30.set_ylabel('pixels')
    _ttl = '2D error: mean (blue), std (red)'
    if model:
        _ttl += f'  [{model} model'
        _ttl += (f', RMS/axis {rms_axis:.3f} px]' if rms_axis is not None
                 and np.isfinite(rms_axis) else ']')
    ax30.set_title(_ttl, fontsize=10)
    ax30.legend()

    # --- Figure 31: Bar chart ---
    fig31, ax31 = plt.subplots()
    x = np.arange(len(cam_ids))
    width = 0.35
    ax31.bar(x - width/2, mean_plot, width, label='Mean 2D error', color='blue')
    ax31.bar(x + width/2, std_plot,  width, label='STD 2D error',  color='red')
    ax31.set_xticks(x)
    ax31.set_xticklabels(cam_ids)
    ax31.grid(True)
    ax31.set_xlabel('Id of the camera')
    ax31.set_ylabel('pixels')
    ax31.set_title(_ttl, fontsize=10)
    ax31.legend()

    path_line = os.path.join(config.data_path, 'reprerrors.png')
    path_bars = os.path.join(config.data_path, 'reprerrors_bars.png')
    # NOTE: no bbox_inches='tight' — a tight bbox computed from unusual data can
    # occasionally evaluate to an invalid image size; a fixed canvas is robust.
    try:
        fig30.savefig(path_line, dpi=150)
        fig31.savefig(path_bars, dpi=150)
        print('Reprojection error plots saved to:')
        print(f'  {path_line}')
        print(f'  {path_bars}')
    finally:
        plt.close('all')  # Free memory; no GUI windows opened

    # Auto-opening the saved images in a GUI viewer is OFF by default. On Windows
    # the viewer (Photos) holds a lock on the file, so the NEXT run cannot
    # overwrite it and the pipeline crashes with 'Invalid argument'. The plots are
    # always written to disk regardless; open them manually, or set
    # config.auto_open_plots = True to restore auto-opening.
    if getattr(config, 'auto_open_plots', False):
        _open_image(path_line)
        _open_image(path_bars)


def _open_image(path):
    """Open an image file with the OS default viewer, non-blocking."""
    try:
        if sys.platform == 'win32':
            os.startfile(path)                          # Windows: Photos / Paint
        elif sys.platform == 'darwin':
            subprocess.Popen(['open', path])            # macOS: Preview
        else:
            subprocess.Popen(['xdg-open', path])        # Linux: default viewer
    except Exception as e:
        print(f'  (Could not auto-open {path}: {e})')


def plot_reproj_history(history, config, phase_boundaries=None):
    """
    Plot the reprojection RMS over the whole run, with the stages marked.

    Parameters
    ----------
    history : list of (label:str, value:float)
        The reprojection RMS (px) at each stage of the run, in order — one entry
        per global iteration and per polish cycle.
    config : CalibrationConfig
        Provides data_path for the output file.
    phase_boundaries : list of (index:int, name:str), optional
        Positions at which a new phase begins (e.g. the first polish cycle), drawn
        as labelled vertical lines so the reader can see which stage is which.

    Best-effort: any failure is caught so it can never crash the pipeline.
    """
    try:
        if not history:
            return
        labels = [str(l) for l, _ in history]
        vals = [float(v) if np.isfinite(v) else np.nan for _, v in history]
        xs = np.arange(len(vals))

        fig, ax = plt.subplots(figsize=(max(6, 0.7 * len(vals) + 2), 4.6))
        ax.plot(xs, vals, 'o-', color='tab:blue', label='Reprojection RMS')
        ax.set_yscale('log')  # errors span orders of magnitude across a run
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Reprojection RMS (px, log scale)')
        ax.set_title('Reprojection error across the calibration run')
        ax.grid(True, which='both', alpha=0.3)

        # label each point with its value (the change spans orders of magnitude,
        # so the exact number next to each marker is easier to read than the axis)
        for _x, _v in zip(xs, vals):
            if np.isfinite(_v):
                ax.annotate(f'{_v:.2f}', xy=(_x, _v), xytext=(0, 7),
                            textcoords='offset points', ha='center', va='bottom',
                            fontsize=8, color='tab:blue',
                            bbox=dict(boxstyle='round,pad=0.15', fc='white',
                                      ec='none', alpha=0.7))

        # mark phase transitions (e.g. iterations -> polish) with vertical lines
        for idx, name in (phase_boundaries or []):
            if 0 <= idx < len(vals):
                ax.axvline(idx - 0.5, color='tab:red', linestyle='--', alpha=0.8)
                ax.text(idx - 0.5, ax.get_ylim()[1], f' {name}', color='tab:red',
                        va='top', ha='left', fontsize=8, rotation=0)

        # annotate the final value (placed lower-left, away from the legend)
        _final = next((v for v in reversed(vals) if np.isfinite(v)), None)
        if _final is not None:
            ax.text(0.02, 0.06, f'final: {_final:.3f} px',
                    transform=ax.transAxes, ha='left', va='bottom', fontsize=9,
                    bbox=dict(boxstyle='round', fc='white', ec='tab:blue', alpha=0.85))
        ax.legend(loc='upper right', fontsize=8)

        # leave room for the rotated x-axis labels so none are clipped
        fig.subplots_adjust(bottom=0.28, left=0.12, right=0.97, top=0.9)

        path = os.path.join(config.data_path, 'reproj_history.png')
        try:
            fig.savefig(path, dpi=150)
            print(f'  Reprojection-history plot saved to: {path}')
        finally:
            plt.close('all')
        if getattr(config, 'auto_open_plots', False):
            _open_image(path)
    except Exception as e:
        print(f'  [warn] reprojection-history plot skipped '
              f'({type(e).__name__}: {e}); calibration result is unaffected.')
