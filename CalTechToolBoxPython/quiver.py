import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Arrow head shape constants (match original MATLAB defaults)
ALPHA = 0.33   # arrowhead length as fraction of vector length
BETA  = 0.33   # arrowhead width as fraction of vector length


def quiver(*args, filled=False, ax=None):
    """
    Plot velocity vectors as arrows — direct port of MATLAB's quiver.m.

    Call signatures
    ---------------
    quiver(u, v)
    quiver(u, v, s)
    quiver(x, y, u, v)
    quiver(x, y, u, v, s)

    All signatures accept optional keyword arguments:
        color    : matplotlib colour string, e.g. 'r', 'blue'
        linestyle: matplotlib linestyle string, e.g. '-', '--'
        marker   : marker at vector base, e.g. 'o', 's'.
                   If marker is given, arrowheads are suppressed.
        filled   : bool — fill the base marker (default False)
        ax       : matplotlib Axes to draw on (default: current axes)

    Parameters
    ----------
    x, y : array-like — grid positions (optional; defaults to index grid)
    u, v : array-like — vector components
    s    : float      — scale factor; 0 disables autoscaling (default 1)

    Returns
    -------
    lines : list of Line2D — [shaft_line, arrowhead_line, marker_line]
            Elements may be None if that component was not drawn.
    """

    # ------------------------------------------------------------------ #
    # Parse positional arguments: separate numeric arrays from the
    # optional scale factor, then determine the call signature used.
    # ------------------------------------------------------------------ #
    color     = ''
    linestyle = '-'
    marker    = ''
    plot_arrows = True
    autoscale   = 1.0

    # Separate keyword-style string args passed positionally (MATLAB style)
    args = list(args)

    # Pull recognised string args off the end (mimics MATLAB's while-loop)
    while args and isinstance(args[-1], str):
        val = args.pop()
        if val.lower().startswith('f'):       # 'filled'
            filled = True
        else:
            # Try to interpret as a matplotlib format string
            # e.g. 'r--', 'bo', 'k-.'
            import re
            col_map = {'b':'b','g':'g','r':'r','c':'c',
                       'm':'m','y':'y','k':'k','w':'w'}
            ls_map  = {'--':'--', '-.':'-.', ':':':', '-':'-'}
            mk_chars = set('os^v<>dph+x*')
            c = re.findall(r'[bgrcmykw]', val)
            if c:
                color = c[0]
            for ls_str in ['--', '-.', ':', '-']:
                if ls_str in val:
                    linestyle = ls_str
                    break
            mk = re.findall(r'[os\^v<>dph\+x\*]', val)
            if mk and mk[0] != '.':
                marker = mk[0]
                plot_arrows = False

    # Remaining args are numeric: (u,v), (u,v,s), (x,y,u,v), (x,y,u,v,s)
    n = len(args)
    if n < 2 or n > 5:
        raise ValueError("Expected 2–5 numeric arguments.")

    if n in (2, 3):
        # quiver(u, v) or quiver(u, v, s)
        u = np.asarray(args[0], dtype=float)
        v = np.asarray(args[1], dtype=float)
        rows, cols = u.shape if u.ndim == 2 else (u.size, 1)
        x, y = np.meshgrid(np.arange(cols), np.arange(rows))
        if n == 3:
            autoscale = float(args[2])
    else:
        # quiver(x, y, u, v) or quiver(x, y, u, v, s)
        x = np.asarray(args[0], dtype=float)
        y = np.asarray(args[1], dtype=float)
        u = np.asarray(args[2], dtype=float)
        v = np.asarray(args[3], dtype=float)
        if n == 5:
            autoscale = float(args[4])

    # Scalar-expand u, v to match x shape
    if u.size == 1:
        u = np.full(x.shape, u.flat[0])
    if v.size == 1:
        v = np.full(u.shape, v.flat[0])

    # ------------------------------------------------------------------ #
    # Autoscale: normalise arrow lengths so the longest vector fits in
    # one average grid cell, then stretch by the scale factor s.
    # ------------------------------------------------------------------ #
    if autoscale:
        if min(x.shape) == 1:                     # vector input → square grid
            sq = np.sqrt(x.size)
            m, nc = sq, sq
        else:
            m, nc = x.shape

        delx = np.ptp(x) / nc   # average x cell width
        dely = np.ptp(y) / m    # average y cell height
        del2 = delx**2 + dely**2

        if del2 > 0:
            lengths = np.sqrt((u**2 + v**2) / del2)
            maxlen  = lengths.max()
        else:
            maxlen = 0

        scale = (autoscale * 0.9 / maxlen) if maxlen > 0 else autoscale * 0.9
        u = u * scale
        v = v * scale

    # ------------------------------------------------------------------ #
    # Draw arrow shafts.
    # Build NaN-separated coordinate arrays so all shafts are one Line2D.
    # Layout per vector: [tail, tip, NaN]
    # ------------------------------------------------------------------ #
    xf = x.ravel()
    yf = y.ravel()
    uf = u.ravel()
    vf = v.ravel()

    shaft_x = np.empty(3 * xf.size); shaft_x[0::3] = xf
    shaft_x[1::3] = xf + uf;         shaft_x[2::3] = np.nan
    shaft_y = np.empty(3 * yf.size); shaft_y[0::3] = yf
    shaft_y[1::3] = yf + vf;         shaft_y[2::3] = np.nan

    if ax is None:
        ax = plt.gca()

    fmt = color + linestyle
    h1, = ax.plot(shaft_x, shaft_y, fmt if fmt.strip('-') else linestyle,
                  color=color if color else None)

    # ------------------------------------------------------------------ #
    # Draw arrowheads as two-line V-shapes at the tip of each vector.
    # Each wing of the V is: [wing-point, tip, NaN]
    # The wing points are computed using alpha (length) and beta (width).
    # eps prevents division by zero for zero-length vectors.
    # ------------------------------------------------------------------ #
    h2 = None
    if plot_arrows:
        eps = np.finfo(float).eps
        tip_x = xf + uf
        tip_y = yf + vf

        # Left wing then right wing, separated by NaN
        # (matches the 4-row layout in the original MATLAB)
        aw_x = np.empty(4 * xf.size)
        aw_y = np.empty(4 * yf.size)

        aw_x[0::4] = tip_x - ALPHA * (uf + BETA * (vf + eps))
        aw_x[1::4] = tip_x
        aw_x[2::4] = tip_x - ALPHA * (uf - BETA * (vf + eps))
        aw_x[3::4] = np.nan

        aw_y[0::4] = tip_y - ALPHA * (vf - BETA * (uf + eps))
        aw_y[1::4] = tip_y
        aw_y[2::4] = tip_y - ALPHA * (vf + BETA * (uf + eps))
        aw_y[3::4] = np.nan

        h2, = ax.plot(aw_x, aw_y, fmt if fmt.strip('-') else linestyle,
                      color=h1.get_color())

    # ------------------------------------------------------------------ #
    # Optionally draw a marker at the base (tail) of each vector.
    # ------------------------------------------------------------------ #
    h3 = None
    if marker:
        h3, = ax.plot(xf, yf, marker=marker, linestyle='none',
                      color=h1.get_color())
        if filled:
            h3.set_markerfacecolor(h1.get_color())

    return [h1, h2, h3]


# ------------------------------------------------------------------ #
# Demo — mirrors the example in the MATLAB docstring
# ------------------------------------------------------------------ #
if __name__ == '__main__':
    x, y = np.meshgrid(np.arange(-2, 2.2, 0.2), np.arange(-1, 1.15, 0.15))
    z  = x * np.exp(-x**2 - y**2)
    py, px = np.gradient(z, 0.15, 0.2)   # note: np.gradient returns (dy,dx)

    fig, ax = plt.subplots()
    ax.contour(x, y, z)
    quiver(x, y, px, py, ax=ax)
    ax.set_aspect('equal')
    ax.set_title('quiver demo')
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/quiver_demo.png', dpi=120)
    plt.show()
    print("Demo saved to quiver_demo.png")
