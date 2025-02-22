import numpy as np
from p2e import p2e


def lin_fm(u):
    """
    Constructs the coefficient matrix A for the 7-point fundamental matrix estimation.

    Parameters:
    -----------
    u : numpy.ndarray a 6x7 or 4x7 matrix containing 7 point correspondences.
        - If `u` is 6x7, the first three rows represent points from the first image
          in homogeneous coordinates, and the last three rows represent the corresponding points from the second image.
        - If `u` is 4x7, it contains 2D (Euclidean) coordinates directly.

    Returns:
    --------
    At : numpy.ndarray a 7x9 coefficient matrix used to solve for the fundamental matrix.

    Explanation:
    ------------
    This function constructs the linear system used in the 7-point algorithm
    to estimate the fundamental matrix (F).
    """

    # Ensure that exactly 7 point correspondences are provided
    if u.shape[1] != 7:
        raise ValueError("Wrong size of input points.")

    # Convert homogeneous coordinates to Euclidean if needed
    elif u.shape[0] == 6:
        x1 = p2e(u[0:3, :])  # First image points
        x2 = p2e(u[3:6, :])  # Second image points
    elif u.shape[0] != 4:
        raise ValueError("Wrong size of input points.")

    # If points are already Euclidean (2x7), use directly
    x1 = u[0:2, :]
    x2 = u[2:4, :]

    # Construct the A matrix for Af = 0
    A = np.vstack([
        x2[0] * x1[0],  # x' * x
        x2[0] * x1[1],  # x' * y
        x2[0],  # x'
        x2[1] * x1[0],  # y' * x
        x2[1] * x1[1],  # y' * y
        x2[1],  # y'
        x1[0],  # x
        x1[1],  # y
        np.ones(7)  # Constant term
    ])  # Shape: (9, 7)

    # Transpose A to get (7,9) shape
    At = A.T
    return At


# Example usage
"""
u = np.array([
        [1, 2, 3, 4, 5, 6, 7],   # x-coordinates in first image
        [8, 9, 10, 11, 12, 13, 14],  # y-coordinates in first image
        [1, 1, 1, 1, 1, 1, 1],   # Homogeneous scale
        [2, 3, 4, 5, 6, 7, 8],   # x-coordinates in second image
        [9, 10, 11, 12, 13, 14, 15],  # y-coordinates in second image
        [1, 1, 1, 1, 1, 1, 1]    # Homogeneous scale
    ])
A_matrix = lin_fm(u)
print("A matrix:")
print(A_matrix)
"""
