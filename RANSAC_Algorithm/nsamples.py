import numpy as np


def nsamples(no_i, pt_num, s, conf=0.99):
    """
    Calculate the number of samples needed for RANSAC

    Parameters:
    -----------
    no_i : int
        Current number of inliers
    pt_num : int
        Total number of points
    s : int
        Sample size (number of points needed for model estimation)
    conf : float
        Confidence value (typically 0.95 or 0.99)
        by default the Confidence value is 0.99

    Returns:
    --------
    N : float
        Number of samples needed

    Notes:
    ------
    This implements the standard RANSAC formula for determining
    the number of iterations needed to achieve a given confidence
    of finding the correct model, based on the current inlier ratio.
    """

    # Calculate outlier ratio
    outl = 1 - no_i / pt_num

    # Calculate required number of samples
    # The formula is derived from: (1 - (1-outl)^s)^N = (1-conf)
    # where N is the number of samples we're solving for

    N = np.log(1 - conf) / np.log(1 - (1 - outl) ** s + np.finfo(float).eps)

    return N