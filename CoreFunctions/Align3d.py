import numpy as np


def align3d(inP, inX, simT):
    """
    Aligns 3D points and projection matrices using a similarity transformation.

    Parameters
    ----------
    inP : numpy.ndarray
        Input projection matrices (3*N_cams x 4)
    inX : numpy.ndarray
        Input 3D points in homogeneous coordinates (4 x N_points)
    simT : dict
        Similarity transformation parameters:
        - 's': scale factor (float)
        - 'R': 3x3 rotation matrix
        - 't': 3-element translation vector

    Returns
    -------
    P : numpy.ndarray
        Transformed projection matrices
    X : numpy.ndarray
        Transformed 3D points
    """

    # Build 4x4 transformation matrix:
    # T = | s*R  t |
    #     | 0    1 |
    T = np.vstack((simT['s'] * simT['R'], [0, 0, 0]))
    T = np.column_stack((T, np.append(simT['t'], 1)))

    # Transform 3D points
    X = T @ inX

    # Transform projection matrices
    P = inP @ np.linalg.inv(T)

    return P, X