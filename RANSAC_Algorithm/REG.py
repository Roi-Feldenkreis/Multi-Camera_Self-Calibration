import numpy as np
from RANSAC_Algorithm.U2fdlt import u2fdlt
from RANSAC_Algorithm.Fsampson import fsampson
from RANSAC_Algorithm.Nsamples import nsamples


def reg(u, th, th4=None, conf=0.99, ss=8):
    """
    Robust estimation of the epipolar geometry via RANSAC

    Parameters:
    -----------
    u : numpy.ndarray (6xN)
        Point pairs in homogeneous coordinates
    th : float
        Inlier tolerance threshold in pixels
    th4 : float, optional
        Currently not used (kept for compatibility)
    conf : float, optional (default=0.99)
        Confidence level
    ss : int, optional (default=8)
        Sample size
    """

    MAX_SAM = 100000

    len_pts = u.shape[1]

    if th4 is None:
        th4 = th

    max_i = 5
    max_sam = MAX_SAM
    no_sam = 0
    th = 2 * th ** 2

    inls = np.zeros(len_pts, dtype=bool)  # safe default: all outliers
    F = None

    while no_sam < max_sam:
        # np.random.choice is faster than manual Fisher-Yates for small ss
        sample_idx = np.random.choice(len_pts, ss, replace=False)
        no_sam += 1

        sF = u2fdlt(u[:, sample_idx], False)
        errs = fsampson(sF, u)
        v = errs < th
        no_i = np.sum(v)

        if max_i < no_i:
            inls = v
            F = sF
            max_i = no_i
            max_sam = min(max_sam, nsamples(max_i, len_pts, ss, conf))

    # Safety check: if no model was ever found, raise clearly
    if F is None or np.sum(inls) < ss:
        if F is None:
            raise ValueError("RANSAC failed to find any valid model (all iterations had ≤5 inliers)")
        print(f"WARNING: No valid model found with more than {max_i} inliers")
        return F, inls

    # Refine using all inliers with normalization
    F = u2fdlt(u[:, inls], True)

    if no_sam == MAX_SAM:
        print(f"WARNING: RANSAC - termination forced after {no_sam} samples")
    else:
        print(f"RANSAC: {no_sam} samples, {np.sum(inls)} inliers out of {len_pts} points")

    return F, inls
