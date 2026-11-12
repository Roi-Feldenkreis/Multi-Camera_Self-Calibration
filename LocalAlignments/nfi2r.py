import numpy as np


def nfi2r(n, fi):
    """
    Computes the rotation matrix given the axis of rotation and the angle.
    
    Parameters:
    n (np.ndarray): 3x1 axis of rotation (direction vector).
    fi (float): Angle of rotation in radians (counterclockwise).
    
    Returns:
    np.ndarray: 3x3 rotation matrix.
    """
    
    # Normalize the axis of rotation
    n = n / np.linalg.norm(n)
    
    # Calculate cosine and sine of the angle
    cfi = np.cos(fi)
    sfi = np.sin(fi)
    
    # Compute the rotation matrix
    R = np.zeros((3, 3))
    R[0, 0:3] = [cfi + n[0]**2 * (1 - cfi), n[0] * n[1] * (1 - cfi) - n[2] * sfi, n[0] * n[2] * (1 - cfi) + n[1] * sfi]
    R[1, 0:3] = [n[0] * n[1] * (1 - cfi) + n[2] * sfi, cfi + n[1]**2 * (1 - cfi), n[1] * n[2] * (1 - cfi) - n[0] * sfi]
    R[2, 0:3] = [n[2] * n[0] * (1 - cfi) - n[1] * sfi, n[2] * n[1] * (1 - cfi) + n[0] * sfi, cfi + n[2]**2 * (1 - cfi)]
    
    return R.T  # Transpose due to reverse notation
