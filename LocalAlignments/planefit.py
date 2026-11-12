import numpy as np
from OutputFunctions.SaveCalPar import savecalpar
from CoreFunctions.Align3d import align3d
from CoreFunctions.Estimate import estimate


def planefit(X):
    """
    Fit a plane to a point cloud using algebraic minimization.
    
    Parameters:
    X (np.ndarray): Nx3 matrix with 3D points.
    
    Returns:
    n (np.ndarray): 3x1 normal of the fitted plane.
    """
    
    # Subtract the mean of the points
    X = X - np.mean(X, axis=0)
    
    # Perform singular value decomposition
    u, s, v = np.linalg.svd(X)
    
    # The normal is the last column of V (or the last row of V transposed)
    n = v[2, :]  # v[:, 2] would also work, just transposed indexing
    
    # Normalize the normal vector
    n = n / np.linalg.norm(n)
    
    return n
