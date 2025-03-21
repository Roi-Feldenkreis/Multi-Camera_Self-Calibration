import numpy as np
from scipy .linalg import qr
from scipy .linalg import norm


def rq(X):
    """
    Performs RQ decomposition on a matrix X.

    Parameters:
    X (np.ndarray): Input matrix.

    Returns:
    R (np.ndarray): Upper triangular matrix.
    Q (np.ndarray): Unitary matrix.
    """
    # Perform QR decomposition on the transpose of X
    Qt, Rt = qr(X.T)

    # Transpose the results to get the initial R and Q matrices
    Rt = Rt.T
    Qt = Qt.T

    # Initialize Qu as an orthogonal matrix derived from Rt
    Qu = np.zeros((3, 3))

    # Compute the first row of Qu
    Qu[0, :] = np.cross(Rt[1, :], Rt[2, :])
    Qu[0, :] = Qu[0, :] / norm(Qu[0, :])

    # Compute the second row of Qu
    Qu[1, :] = np.cross(Qu[0, :], Rt[2, :])
    Qu[1, :] = Qu[1, :] / norm(Qu[1, :])

    # Compute the third row of Qu
    Qu[2, :] = np.cross(Qu[0, :], Qu[1, :])

    # Calculate the upper triangular matrix R and the unitary matrix Q
    R = Rt @ Qu.T
    Q = Qu @ Qt
    if(R[1][0] != 0.0 or R[2][0] != 0.0 or R[2][1] != 0.0):
        R[1][0] = 0.0
        R[2][0] = 0.0
        R[2][1] = 0.0

    return R, Q


# Example usage
# X = np.random.rand(3, 3)  # Example input matrix
# print('X =')
# print(X)
# R, Q = rq(X)
# print('R =')
# print(R)
# print('Q =')
# print(Q)
