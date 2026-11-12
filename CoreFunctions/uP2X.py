import numpy as np


def uP2X(Umat, Ps):
    """
    Linear reconstruction of 3D points from N-perspective views.

    This function implements the linear triangulation method described in:
    Hartley and Zisserman, Multiple View Geometry, 2000, pp 297-298

    Parameters:
    -----------
    Umat : numpy.ndarray
        3*N x n matrix of n homogeneous points across N views
    Ps : list
        List of N projection matrices, each 3x4

    Returns:
    --------
    X : numpy.ndarray
        4 x n matrix of homogeneous 3D points
    """
    N = Umat.shape[0] // 3  # Number of cameras
    n = Umat.shape[1]  # Number of points

    # Initialize 3D points matrix
    X = np.zeros((4, n))

    # For each point, solve the linear system
    for i in range(n):
        A = np.zeros((2 * N, 4))

        # For all cameras, fill in constraints
        for j in range(N):
            # Extract image coordinates for this point in this camera
            u = Umat[j * 3, i]  # x coordinate
            v = Umat[j * 3 + 1, i]  # y coordinate

            # Extract the projection matrix for this camera
            P = Ps[j]

            # These equations come from the projection constraint: u = PX/PZ, v = PY/PZ
            A[j * 2, :] = u * P[2, :] - P[0, :]  # x constraint
            A[j * 2 + 1, :] = v * P[2, :] - P[1, :]  # y constraint

        # corresponding to the smallest singular value
        _, _, V = np.linalg.svd(A)
        X[:, i] = V[-1, :]  # Last row of V

    # Normalize the homogeneous coordinates
    X = X / np.tile(X[3, :], (4, 1))

    return X