import numpy as np

def FDs(P):
    """
    Extract focal lengths from all the cameras.

    Parameters:
    P (list of numpy arrays): List of camera projection matrices.

    Returns:
    numpy array: Array of focal lengths for each camera.
    """
    fx = []
    fy = []

    for i in range(len(P)):
        A = P[i][:3, :3]
        fx_i = np.sqrt(A[0, 0]**2 + A[0, 1]**2 + A[0, 2]**2)
        fy_i = np.sqrt(A[1, 0]**2 + A[1, 1]**2 + A[1, 2]**2)
        fx.append(fx_i)
        fy.append(fy_i)

    F = np.array([fx, fy])
    return F

# Example usage:
# P = [np.array([[...], [...], [...]]), ...]
# F = FDs(P)
# print(F)
