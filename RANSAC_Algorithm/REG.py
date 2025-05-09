import numpy as np
from u2fdlt import u2fdlt
from fsampson import fsampson
from nsamples import nsamples


def reg(u, th, th4=None, conf=0.99, ss=8):
    """
    Robust estimation of the epipolar geometry via RANSAC

    Parameters:
    -----------
    u : numpy.ndarray (6xN)
        Point pairs in homogeneous coordinates
        First 3 rows are points in first image
        Last 3 rows are points in second image
    th : float
        Inlier tolerance threshold in pixels
    th4 : float, optional
        Currently not used (kept for compatibility)
    conf : float, optional (default=0.99)
        Confidence level (higher values mean more samples will be taken)
    ss : int, optional (default=8)
        Sample size - number of point pairs for each model hypothesis

    Returns:
    --------
    F : numpy.ndarray (3x3)
        Estimated fundamental matrix
    inls : numpy.ndarray (N,) of bool
        Boolean mask indicating inliers (True) and outliers (False)

    Notes:
    ------
    This implements the RANSAC algorithm to robustly estimate the
    fundamental matrix in the presence of outliers.
    """

    MAX_SAM = 100000  # Maximum number of random samples

    len_pts = u.shape[1]

    # If th4 not provided, use th
    if th4 is None:
        th4 = th

    # Initialize variables
    ptr = np.arange(len_pts)  # Point indices
    max_i = 5  # Minimum number of inliers to consider
    max_sam = MAX_SAM  # Maximum number of samples

    no_sam = 0  # Current number of samples
    no_mod = 0  # Current number of models (not used in this implementation)

    # Square the threshold and double it
    th = 2 * th ** 2

    # RANSAC main loop
    inls = None
    F = None

    while no_sam < max_sam:
        # Randomly select ss points
        for pos in range(ss):
            idx = pos + int(np.ceil(np.random.rand() * (len_pts - pos)))
            # Swap positions to avoid resampling
            ptr[pos], ptr[idx] = ptr[idx], ptr[pos]

        no_sam += 1

        # Estimate model from the sample
        sF = u2fdlt(u[:, ptr[0:ss]], False)

        # Calculate errors using the Sampson distance
        errs = fsampson(sF, u)

        # Determine inliers based on threshold
        v = errs < th
        no_i = np.sum(v)

        # If this is the best model so far, update
        if max_i < no_i:
            inls = v
            F = sF
            max_i = no_i
            # Update the maximum number of samples based on inlier ratio
            max_sam = min(max_sam, nsamples(max_i, len_pts, ss, conf))

    # Refine the fundamental matrix using all inliers and with point normalization
    F = u2fdlt(u[:, inls], True)

    # Display results
    if no_sam == MAX_SAM:
        print(f"WARNING: RANSAC - termination forced after {no_sam} samples")
    else:
        print(f"RANSAC: {no_sam} samples, {np.sum(inls)} inliers out of {len_pts} points")

    return F, inls