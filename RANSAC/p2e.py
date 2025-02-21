import numpy as np

def p2e(u):
    """
    Converts homogeneous coordinates to Euclidean coordinates.

    Parameters:
    -----------
    u : numpy.ndarray a 3xN array where each column represents a point in homogeneous coordinates.

    Returns:
    --------
    E : numpy.ndarray a 2xN array where each column represents the point in Euclidean coordinates.
    """
    E2 = []
    for i in range(len(u)):
        e = [u[i][0] / u[i][2] ,u[i][1] / u[i][2]]
        E2.append(e)
    E = np.array(E2)
    return E

# Example usage
# u = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# e = p2e(u)
# print(e)
