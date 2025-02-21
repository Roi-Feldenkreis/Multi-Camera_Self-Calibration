import numpy as np


def mfFDs(F, u):
    """
    Computes an error metric based on the epipolar constraint.

    This function evaluates the error associated with a fundamental matrix F and a set of point correspondences u.
    The error is computed based on a normalized epipolar constraint formulation.

    Parameters:
    F (numpy.ndarray): A 3x3 fundamental matrix.
    u (numpy.ndarray): A 6xN matrix, where:
        - The first three rows (0:3) correspond to homogeneous coordinates of image points.
        - The last three rows (3:6) correspond to their corresponding points in another image.

    Returns:
    numpy.ndarray: A 1D array of length N containing the computed error values for each correspondence.
    """
    Fu1 = F @ u[3:6, :]
    Fu2 = (F.T @ u[0:3, :]) ** 2
    Fu1pow = Fu1 ** 2
    denom = Fu1pow[0, :] + Fu1pow[1, :] + Fu2[0, :] + Fu2[1, :]
    errvec = np.zeros(u.shape[1])

    for i in range(u.shape[1]):
        xFx = u[0:3, i].T @ Fu1[:, i]
        errvec[i] = xFx ** 2 / denom[i]

    err = errvec
    return err


# Example usage
# F = np.random.rand(3, 3)  # Example F matrix
# u = np.random.rand(6, 10)  # Example u matrix with 10 correspondences
# err = mfFDs(F, u)
# print(err)
