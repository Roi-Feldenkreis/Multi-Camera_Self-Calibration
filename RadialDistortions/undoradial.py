import numpy as np
from comp_distortion_oulu import comp_distortion_oulu


def undoradial(x_kk, K, kc):
    """
    Remove radial distortion from pixel coordinates.

    This function converts distorted pixel coordinates to linearized coordinates
    that follow the pinhole camera model, by removing the effects of radial
    and tangential distortion.

    Parameters:
    -----------
    x_kk : numpy.ndarray (3xN)
        Coordinates of the distorted pixel points in homogeneous form
        First 2 rows are x, y pixel coordinates, third row is typically ones
    K : numpy.ndarray (3x3)
        Camera calibration matrix containing:
        - Focal lengths in x, y directions (K[0,0], K[1,1])
        - Principal point coordinates (K[0,2], K[1,2])
    kc : numpy.ndarray (4x1 or 5x1)
        Vector of distortion parameters:
        - kc[0], kc[1], kc[4] (if present): radial distortion coefficients
        - kc[2], kc[3]: tangential distortion coefficients

    Returns:
    --------
    xl : numpy.ndarray (3xN)
        Linearized pixel coordinates (undistorted) in homogeneous form

    Notes:
    ------
    The process involves:
    1. Converting pixel coordinates to normalized coordinates
    2. Applying distortion compensation using the Oulu model
    3. Converting back to pixel coordinates

    This is based on the Camera Calibration Toolbox from Caltech.
    """

    # Extract camera parameters from the calibration matrix
    cc = np.zeros(2)  # Principal point
    fc = np.zeros(2)  # Focal lengths

    cc[0] = K[0, 2]  # Principal point x-coordinate
    cc[1] = K[1, 2]  # Principal point y-coordinate
    fc[0] = K[0, 0]  # Focal length in x-direction
    fc[1] = K[1, 1]  # Focal length in y-direction

    # Step 1: Subtract principal point and divide by focal length
    # This converts pixel coordinates to normalized coordinates
    x_distort = np.zeros((2, x_kk.shape[1]))
    x_distort[0, :] = (x_kk[0, :] - cc[0]) / fc[0]
    x_distort[1, :] = (x_kk[1, :] - cc[1]) / fc[1]

    # Step 2: Apply distortion compensation if needed
    if np.linalg.norm(kc) != 0:
        # Apply the distortion compensation model from Oulu University
        xn = comp_distortion_oulu(x_distort, kc)
    else:
        # No distortion to correct
        xn = x_distort

    # Step 3: Convert back to pixel coordinates (linearized)
    # Create homogeneous coordinates by adding a row of ones
    xn_homo = np.vstack((xn, np.ones((1, xn.shape[1]))))

    # Apply the calibration matrix to get back to pixel coordinates
    xl = K @ xn_homo

    return xl