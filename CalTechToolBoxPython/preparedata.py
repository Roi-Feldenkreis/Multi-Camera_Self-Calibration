import numpy as np


def preparedata(filename: str, normalise_scale: bool = True):
    """
    Read a BlueC .rad correspondence file and extract 3D/2D point pairs
    for use with a camera calibration toolbox.

    Parameters
    ----------
    filename : str
        Path to the .rad ASCII data file.
    normalise_scale : bool, default True
        If True, the returned 3D points are uniformly scaled so that their
        RMS magnitude is ~1.  This makes the calibration optimiser numerically
        stable regardless of whether the input is in metric mm (used for
        multimedia calibration) or projective units (used for standard
        Svoboda self-calibration).

        K (intrinsic matrix) is INVARIANT under this scaling — when the
        3D points are scaled by s, the optimiser will simply recover an
        extrinsic translation that is also scaled by s.  The K matrix it
        returns is independent of the scale of X. (Verified mathematically:
        x_pixel = K(RX + t)/[third row] is invariant when (X, t) both scale
        by 1/s, and the optimiser is free to absorb the scale into t.)

        Set to False to preserve the legacy MATLAB-style behaviour.

    Returns
    -------
    X : (3, N) — 3D world points  (columns 0-2 of the file), possibly normalised
    x : (2, N) — 2D image projections (columns 4-5 of the file), unchanged
    """

    # Load whitespace-delimited ASCII file — mirrors MATLAB's load(filename,'-ASCII')
    datamat = np.loadtxt(filename)

    X = datamat[:, 0:3].T   # columns 1-3 in MATLAB (0-2 in Python) → (3, N)
    x = datamat[:, 4:6].T   # columns 5-6 in MATLAB (4-5 in Python) → (2, N)

    if normalise_scale:
        # RMS magnitude of the (X, Y, Z) coordinates.  Use ddof=0 for
        # population RMS — matches the convention typically used in
        # photogrammetry point normalisation.
        rms = float(np.sqrt(np.mean(X ** 2)))
        if rms > 1e-9:
            X = X / rms
        # If rms is essentially zero the points are degenerate — leave
        # them unscaled so the caller sees the real failure mode.

    return X, x


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python preparedata.py <path_to_file.rad>")
        sys.exit(0)

    X, x = preparedata(sys.argv[1])
    print(f"Loaded {X.shape[1]} point correspondences.")
    print(f"X shape: {X.shape}  (3D world points)")
    print(f"x shape: {x.shape}  (2D image projections)")
