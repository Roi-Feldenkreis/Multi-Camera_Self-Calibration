import numpy as np


def comp_distortion_oulu(xd, k):
    """
    Compensates for radial and tangential distortion using the Oulu University model.

    This function removes distortion from normalized image coordinates. It's an essential
    step in camera calibration and 3D reconstruction pipelines.

    Parameters:
    -----------
    xd : numpy.ndarray (2xN)
        Distorted (normalized) point coordinates in the image plane
    k : numpy.ndarray or list
        Distortion coefficients (radial and tangential)
        If k has length 1: Only k1 radial distortion parameter is used
        If k has length 5: [k1, k2, p1, p2, k3] where:
            - k1, k2, k3 are radial distortion coefficients
            - p1, p2 are tangential distortion coefficients

    Returns:
    --------
    x : numpy.ndarray (2xN)
        Undistorted (normalized) point coordinates in the image plane

    Notes:
    ------
    - This compensation must be done after subtracting the principal point and
      dividing by the focal length (i.e., on normalized image coordinates).
    - Uses an iterative method for compensation, running for 20 iterations.
    - The distortion model includes both radial distortion (k1, k2, k3) and
      tangential distortion (p1, p2).
    """

    if len(k) == 1:
        # If only one parameter is provided, use simplified model
        # Note: comp_distortion function not provided in original files,
        # implementing simple radial distortion compensation
        x = comp_distortion(xd, k)

    else:
        # Extract distortion parameters
        k1 = k[0]  # First radial distortion coefficient
        k2 = k[1]  # Second radial distortion coefficient
        k3 = k[4] if len(k) >= 5 else 0  # Third radial distortion coefficient
        p1 = k[2]  # First tangential distortion coefficient
        p2 = k[3]  # Second tangential distortion coefficient

        # Initial guess is the distorted coordinates
        x = xd.copy()

        # Iterative refinement (20 iterations)
        for _ in range(20):
            # Calculate squared radial distance from center
            r_2 = np.sum(x ** 2, axis=0)

            # Compute radial distortion factor
            k_radial = 1 + k1 * r_2 + k2 * r_2 ** 2 + k3 * r_2 ** 3

            # Compute tangential distortion
            delta_x = np.vstack([
                2 * p1 * x[0, :] * x[1, :] + p2 * (r_2 + 2 * x[0, :] ** 2),
                p1 * (r_2 + 2 * x[1, :] ** 2) + 2 * p2 * x[0, :] * x[1, :]
            ])

            # Update undistorted coordinates
            x = (xd - delta_x) / (np.ones((2, 1)) * k_radial)

    return x


def comp_distortion(xd, k):
    """
    Simple radial distortion compensation with only one parameter.

    This is a helper function used when only k1 is provided.

    Parameters:
    -----------
    xd : numpy.ndarray (2xN)
        Distorted (normalized) point coordinates
    k : float or list with one element
        First radial distortion coefficient (k1)

    Returns:
    --------
    x : numpy.ndarray (2xN)
        Undistorted (normalized) point coordinates
    """
    if isinstance(k, list) or isinstance(k, np.ndarray):
        k = k[0]

    # Initial guess
    x = xd.copy()

    # Iterative refinement (15 iterations is usually sufficient)
    for _ in range(15):
        # Calculate squared radial distance
        r_2 = np.sum(x ** 2, axis=0)

        # Compute radial distortion factor
        k_radial = 1 + k * r_2

        # Update undistorted coordinates (no tangential distortion)
        x = xd / (np.ones((2, 1)) * k_radial)

    return x