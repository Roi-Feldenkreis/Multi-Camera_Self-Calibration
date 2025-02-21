import numpy as np

def p2e(u):
    """
    Converts homogeneous coordinates to Euclidean coordinates.

    Parameters:
    -----------
    u : numpy.ndarray a 3xN array where each column represents a point in homogeneous coordinates.

    Returns:
    --------
    e : numpy.ndarray a 2xN array where each column represents the point in Euclidean coordinates.
    """
    e = u[:2, :] / u[2, :]
    return e

# Example usage
# u = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# e = p2e(u)
# print(e)
