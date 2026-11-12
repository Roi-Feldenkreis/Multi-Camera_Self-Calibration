import numpy as np
import matplotlib.pyplot as plt
from CalTechToolBoxPython.apply_distortion import apply_distortion
from CalTechToolBoxPython.quiver import quiver as calib_quiver


def visualize_distortions(nx, ny, fc, cc, kc, alpha_c=0.0,
                          fc_error=None, cc_error=None,
                          kc_error=None, alpha_c_error=0.0,
                          err_std=None, save_path=None):
    """
    Plot the lens distortion model in three figures:
      Figure 1 — Complete distortion (radial + tangential)
      Figure 2 — Tangential component only
      Figure 3 — Radial component only

    Each figure shows:
      - A quiver plot of the distortion displacement field
      - Contour lines of displacement magnitude (in pixels)
      - Principal point (o) and image centre (x) markers
      - Parameter annotation text along the bottom

    Parameters
    ----------
    nx, ny        : int   — image width and height in pixels
    fc            : (2,)  — focal lengths [fx, fy]
    cc            : (2,)  — principal point [cx, cy]
    kc            : (5,)  — distortion coefficients [k1, k2, p1, p2, k3]
    alpha_c       : float — skew coefficient (default 0)
    fc_error      : (2,)  — uncertainty on fc   (default zeros)
    cc_error      : (2,)  — uncertainty on cc   (default zeros)
    kc_error      : (5,)  — uncertainty on kc   (default zeros)
    alpha_c_error : float — uncertainty on alpha_c (default 0)
    err_std       : (2,)  — reprojection error std [x, y] (default zeros)
    save_path     : str   — if given, save each figure as a PNG with this prefix
    """

    fc = np.asarray(fc, dtype=float)
    cc = np.asarray(cc, dtype=float)
    kc = np.asarray(kc, dtype=float)
    if fc_error    is None: fc_error    = np.zeros(2)
    if cc_error    is None: cc_error    = np.zeros(2)
    if kc_error    is None: kc_error    = np.zeros(5)
    if err_std     is None: err_std     = np.zeros(2)
    fc_error  = np.asarray(fc_error,  dtype=float)
    cc_error  = np.asarray(cc_error,  dtype=float)
    kc_error  = np.asarray(kc_error,  dtype=float)

    # ------------------------------------------------------------------ #
    # Step 1 — Build normalised ray grid
    # Sample a 21x21 grid of pixel positions, back-project through inv(K)
    # to get normalised (undistorted) image coordinates.
    # ------------------------------------------------------------------ #
    gx = np.linspace(0, nx - 1, 21)
    gy = np.linspace(0, ny - 1, 21)
    mx, my = np.meshgrid(gx, gy)    # (21, 21) each — image pixel grid
    nnx, nny = mx.shape             # 21, 21

    # Flatten column-major to match reshape(mx', nnx*nny, 1)
    px = mx.T.ravel()               # (441,) — pixel x positions
    py = my.T.ravel()               # (441,) — pixel y positions

    # Back-project: normalised coords = inv(K) * [px; py; 1]
    KK = np.array([
        [fc[0], alpha_c * fc[0], cc[0]],
        [0,     fc[1],           cc[1]],
        [0,     0,               1.0  ]
    ])
    rays = np.linalg.inv(KK) @ np.vstack([px, py, np.ones_like(px)])
    x = rays[:2] / rays[2]         # normalised coords (2, 441)

    # ------------------------------------------------------------------ #
    # Step 2 — Build annotation text (reused across all three figures)
    # ------------------------------------------------------------------ #
    left_text = (
        f"Pixel error     = [{err_std[0]:.4g}, {err_std[1]:.4g}]\n"
        f"Focal Length    = ({fc[0]:.6g}, {fc[1]:.6g})\n"
        f"Principal Point = ({cc[0]:.6g}, {cc[1]:.6g})\n"
        f"Skew            = {alpha_c:.4g}\n"
        f"Radial coeff    = ({kc[0]:.4g}, {kc[1]:.4g}, {kc[4]:.4g})\n"
        f"Tangential      = ({kc[2]:.4g}, {kc[3]:.4g})")
    right_text = (
        f"+/- [{fc_error[0]:.4g}, {fc_error[1]:.4g}]\n"
        f"+/- [{cc_error[0]:.4g}, {cc_error[1]:.4g}]\n"
        f"+/- {alpha_c_error:.4g}\n"
        f"+/- [{kc_error[0]:.4g}, {kc_error[1]:.4g}, {kc_error[4]:.4g}]\n"
        f"+/- [{kc_error[2]:.4g}, {kc_error[3]:.4g}]")

    # ------------------------------------------------------------------ #
    # Helper: draw one distortion figure
    # ------------------------------------------------------------------ #
    def _draw_figure(title_str, kc_subset, label_contours=True):
        """
        kc_subset : (5,) — distortion coefficients for this panel
                    (zero out whichever components are not being shown)
        """
        # Apply distortion using our converted apply_distortion.py
        xd, _ = apply_distortion(x, kc_subset)

        # Re-project distorted normalised coords back to pixel space
        px2 = fc[0] * (xd[0, :] + alpha_c * xd[1, :]) + cc[0]
        py2 = fc[1] * xd[1, :] + cc[1]

        # Pixel-space displacement: how far each point moved due to distortion
        dx = px2 - px
        dy = py2 - py

        # Displacement magnitude reshaped back onto the 2D grid
        # reshape order matches reshape(sqrt(...), nny, nnx)'
        dr = np.sqrt(dx**2 + dy**2).reshape(nnx, nny).T   # (21, 21)

        fig, ax_main = plt.subplots(figsize=(8, 7))
        fig.patch.set_facecolor('white')

        # Quiver: our calib_quiver.py works on grid arrays (mx/my shaped).
        # Here px/py are flat 1D scatter points, so we use matplotlib's
        # native ax.quiver which handles both scattered and grid inputs.
        ax_main.quiver(px + 1, py + 1, dx, dy,
                       angles='xy', scale_units='xy', scale=1,
                       width=0.002, headwidth=4)

        # Principal point and image centre markers
        ax_main.plot(cc[0] + 1, cc[1] + 1, 'o', markersize=6,
                     label='Principal point')
        ax_main.plot((nx - 1) / 2 + 1, (ny - 1) / 2 + 1, 'x',
                     markersize=8, markeredgewidth=2, label='Image centre')

        # Contour lines of displacement magnitude over the grid
        CS = ax_main.contour(mx, my, dr, colors='k', linewidths=0.8)
        if label_contours:
            ax_main.clabel(CS, inline=True, fontsize=7, fmt='%.3g')

        ax_main.set_title(title_str, fontsize=10)
        ax_main.invert_yaxis()          # axis ij: y increases downward
        ax_main.set_xlim(1, nx)
        ax_main.set_ylim(ny, 1)
        ax_main.set_aspect('equal')
        ax_main.tick_params(labelsize=8)
        ax_main.legend(fontsize=7, loc='upper right')

        # Shrink axes to leave space for annotation text
        shr = 0.9
        pos = ax_main.get_position()
        ax_main.set_position([
            pos.x0 + pos.width  * (1 - shr) / 2,
            pos.y0 + pos.height * (1 - shr) + 0.08,
            pos.width  * shr,
            pos.height * shr])

        # Annotation text at the bottom (invisible overlay axes)
        ax_txt = fig.add_axes((0, 0, 1, 1), visible=False)
        ax_txt.text(0.02, 0.01, left_text,
                    ha='left', va='bottom', fontsize=7.5,
                    transform=ax_txt.transAxes, family='monospace')
        ax_txt.text(0.98, 0.01, right_text,
                    ha='right', va='bottom', fontsize=7.5,
                    transform=ax_txt.transAxes, family='monospace')

        return fig

    # ------------------------------------------------------------------ #
    # Step 3 — Draw the three figures with different kc subsets
    # ------------------------------------------------------------------ #
    kc_full       = kc.copy()
    kc_tangential = np.array([0,     0,     kc[2], kc[3], 0    ])
    kc_radial     = np.array([kc[0], kc[1], 0,     0,     kc[4]])

    fig1 = _draw_figure('Complete Distortion Model',           kc_full,       label_contours=True)
    fig2 = _draw_figure('Tangential Component of Distortion',  kc_tangential, label_contours=False)
    fig3 = _draw_figure('Radial Component of Distortion',      kc_radial,     label_contours=True)

    if save_path:
        fig1.savefig(f'{save_path}_complete.png',   dpi=120, bbox_inches='tight')
        fig2.savefig(f'{save_path}_tangential.png', dpi=120, bbox_inches='tight')
        fig3.savefig(f'{save_path}_radial.png',     dpi=120, bbox_inches='tight')
        print(f"Saved: {save_path}_complete.png")
        print(f"Saved: {save_path}_tangential.png")
        print(f"Saved: {save_path}_radial.png")

    plt.show()
    return fig1, fig2, fig3


# ------------------------------------------------------------------ #
# Self-test with realistic camera parameters
# ------------------------------------------------------------------ #
if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')   # non-interactive backend for testing

    fig1, fig2, fig3 = visualize_distortions(
        nx=640, ny=480,
        fc=np.array([800.0, 800.0]),
        cc=np.array([320.0, 240.0]),
        kc=np.array([-0.3, 0.15, 0.001, -0.0005, 0.0]),
        alpha_c=0.002,
        fc_error=np.array([1.2, 1.3]),
        cc_error=np.array([0.5, 0.4]),
        kc_error=np.array([0.01, 0.005, 0.0002, 0.0001, 0.001]),
        alpha_c_error=0.0003,
        err_std=np.array([0.35, 0.28]),
        save_path='/mnt/user-data/outputs/distortion_vis'
    )
    print("All three figures generated ✓")
