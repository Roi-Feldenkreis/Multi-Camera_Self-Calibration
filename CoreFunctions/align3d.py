import numpy as np


def align3d(inP, inX, simT):
    """
    Aligns 3D points and projection matrices using a similarity transformation.

    Parameters:
    inP (numpy.ndarray): Input 3x4 projection matrices.
    inX (numpy.ndarray): Input 4xn matrix containing 3D points.
    simT (dict): Dictionary containing similarity transformation parameters:
                 - 's': scale factor
                 - 'R': 3x3 rotation matrix
                 - 't': 3-element translation vector

    Returns:
    P (numpy.ndarray): Transformed 3x4 projection matrices.
    X (numpy.ndarray): Transformed 4xn matrix containing 3D points.
    """

    # Create a 4x4 transformation matrix T from the scale, rotation, and translation
    T = np.vstack((simT['s'] * simT['R'], [0, 0, 0]))  # Add a row [0, 0, 0] at the bottom
    T = np.column_stack((T, np.append(simT['t'], 1)))  # Add a column [t, 1] on the right

    # Transform the 3D points using the transformation matrix T
    X = T @ inX

    # Transform the projection matrices using the inverse of the transformation matrix T
    P = inP @ np.linalg.inv(T)

    return P, X

# Example usage:
# Assuming simT is a dictionary with keys 's', 'R', and 't' and proper values for scaling, rotation, and translation
# P, X = align3d(inP, inX, simT)
