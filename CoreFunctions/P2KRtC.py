import numpy as np
from CoreFunctions.RQ_decomposition import rq


def P2KRtC(P):
    """
    Decompose the Euclidean 3x4 projection matrix P into its components.

    Parameters:
    -----------
    P : numpy.ndarray
        3x4 projection matrix

    Returns:
    --------
    K : numpy.ndarray
        3x3 upper triangular calibration matrix
        call the intrinsic matrix
    R : numpy.ndarray
        3x3 rotation matrix
    t : numpy.ndarray
        3x1 translation vector
    C : numpy.ndarray
        3x1 position of the camera center
    """
    # Normalize the projection matrix
    P = P / np.linalg.norm(P[2, 0:3])

    # Decompose the matrix using RQ decomposition
    K, R = rq(P[:, 0:3])

    # Calculate translation vector
    t = np.linalg.inv(K) @ P[:, 3]

    # Calculate camera center
    C = -R.T @ t

    return K, R, t, C
