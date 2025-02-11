import numpy as np

def seig(M):
    """
    seig - Sorted eigenvalues and eigenvectors

    Parameters:
    M (numpy.ndarray): A square matrix.

    Returns:
    tuple: A tuple containing:
           - V (numpy.ndarray): The eigenvectors sorted by eigenvalue.
           - d (numpy.ndarray): The sorted eigenvalues.
    """
    # Compute eigenvalues and eigenvectors
    D, V = np.linalg.eig(M)
    # Sort eigenvalues and corresponding eigenvectors
    s = np.argsort(D)
    d = D[s]
    V = V[:, s]
    return V, d

# Example usage
# M = np.array(...)  # Define your matrix M here
# V, d = seig(M)
# print("Sorted eigenvalues:", d)
# print("Sorted eigenvectors:", V)
