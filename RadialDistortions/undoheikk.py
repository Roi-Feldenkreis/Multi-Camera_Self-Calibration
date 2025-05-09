import numpy as np


def imcorr(sys, par, dp):
    """
    Corrects image coordinates contaminated by radial and tangential distortion.

    This function implements the Heikkilä camera model to correct distorted image
    coordinates by removing both radial and tangential distortion effects.

    Parameters:
    -----------
    sys : list or numpy.ndarray (4,)
        System parameters:
        - sys[0] = NDX: X-dimension of the image
        - sys[1] = NDY: Y-dimension of the image
        - sys[2] = Sx: Scale factor in X-direction
        - sys[3] = Sy: Scale factor in Y-direction

    par : list or numpy.ndarray (8,)
        Camera intrinsic parameters:
        - par[0] = Asp: Aspect ratio
        - par[1] = Foc: Focal length (not used directly in this function)
        - par[2] = Cpx: Principal point x-coordinate
        - par[3] = Cpy: Principal point y-coordinate
        - par[4] = Rad1: First radial distortion coefficient
        - par[5] = Rad2: Second radial distortion coefficient
        - par[6] = Tan1: First tangential distortion coefficient
        - par[7] = Tan2: Second tangential distortion coefficient

    dp : numpy.ndarray (n x 2)
        Distorted image coordinates in pixels

    Returns:
    --------
    p : numpy.ndarray (n x 2)
        Corrected (undistorted) image coordinates in pixels

    Notes:
    ------
    This implementation follows the camera model by Janne Heikkilä from
    the University of Oulu, Finland.

    The correction process:
    1. Normalize the coordinates
    2. Apply distortion correction
    3. Convert back to pixel coordinates
    """

    # Extract system parameters
    NDX = sys[0]  # X-dimension of the image
    NDY = sys[1]  # Y-dimension of the image
    Sx = sys[2]  # Scale factor in X-direction
    Sy = sys[3]  # Scale factor in Y-direction

    # Extract camera parameters
    Asp = par[0]  # Aspect ratio
    # Foc = par[1]  # Focal length (not used in this function)
    Cpx = par[2]  # Principal point x-coordinate
    Cpy = par[3]  # Principal point y-coordinate
    Rad1 = par[4]  # First radial distortion coefficient
    Rad2 = par[5]  # Second radial distortion coefficient
    Tan1 = par[6]  # First tangential distortion coefficient
    Tan2 = par[7]  # Second tangential distortion coefficient

    # Convert distorted coordinates to normalized coordinates
    dx = (dp[:, 0] - Cpx) * Sx / (NDX * Asp)
    dy = (dp[:, 1] - Cpy) * Sy / NDY

    # Calculate squared radial distance from the center
    r2 = dx * dx + dy * dy

    # Calculate radial distortion factor
    delta = Rad1 * r2 + Rad2 * r2 * r2

    # Apply radial and tangential distortion correction
    cx = dx * (1 + delta) + 2 * Tan1 * dx * dy + Tan2 * (r2 + 2 * dx * dx)
    cy = dy * (1 + delta) + Tan1 * (r2 + 2 * dy * dy) + 2 * Tan2 * dx * dy

    # Convert corrected normalized coordinates back to pixel coordinates
    p = np.zeros((dp.shape[0], 2))
    p[:, 0] = NDX * Asp * cx / Sx + Cpx
    p[:, 1] = NDY * cy / Sy + Cpy

    return p