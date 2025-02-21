import numpy as np
import p2e

def normu(u):
    """
    Computes a normalization transformation matrix for a set of 2D points to improve numerical stability.

    Parameters:
    -----------
    u : numpy.ndarray a 3xN array of points in homogeneous coordinates. (If u is already in 2D (2xN), it is used directly).

    Returns:
    --------
    A : numpy.ndarray a 3x3 normalization matrix that translates and scales the points to have a mean of zero 
        and an average distance of sqrt(2) from the origin.
    """
    # If u is 3xN, convert it to 2xN by using p2e equivalent
    if u.shape[0] == 3:
        u = p2e(u)

    # Calculate the mean of each row
    m = np.mean(u, axis=1).reshape(-1, 1)

    # Subtract the mean from all points
    u = u - m @ np.ones((1, u.shape[1]))

    # Calculate the distances
    distu = np.sqrt(np.sum(u ** 2, axis=0))

    # Calculate the scaling factor
    r = np.mean(distu) / np.sqrt(2)

    # Create the normalization matrix
    A = np.diag([1 / r, 1 / r, 1])
    A[0:2, 2] = -m.flatten() / r

    return A

# Example usage
# u = np.random.rand(3, 10)  # Example u matrix with 10 points in homogeneous coordinates
# A = normu(u)
# print(A)
