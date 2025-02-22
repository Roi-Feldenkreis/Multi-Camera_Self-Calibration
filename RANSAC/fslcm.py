import numpy as np


def fslcm(F1, F2):
    """
    Computes the coefficients of the cubic polynomial det(λ * F1 + (1 - λ) * F2) = 0.

    Parameters:
    -----------
    F1 : numpy.ndarray a 3x3 fundamental matrix obtained from the null space of the linear system.
    F2 : numpy.ndarray a 3x3 fundamental matrix obtained from the null space of the linear system.

    Returns:
    --------
    p : numpy.ndarray a vector containing the 4 coefficients of the cubic polynomial in λ.

    Explanation:
    ------------
    This function enforces the fundamental matrix constraint:

        det(F) = det(λ * F1 + (1 - λ) * F2) = 0

    The determinant expands into a cubic polynomial:

        det(F) = aλ³ + bλ² + cλ + d = 0

    The function extracts the coefficients `a, b, c, d` and returns them as a numpy array.

    The roots of this polynomial determine the valid λ values for constructing the fundamental matrix.

    """

    # Compute determinant coefficients for λ^3, λ^2, λ, and constant
    A = np.linalg.det(F1)  # Coefficient for λ^3
    B = np.linalg.det(F2)  # Coefficient for constant term
    C = np.linalg.det(F1 + F2) - A - B  # Coefficient for λ^2
    D = np.linalg.det(2 * F1 + F2) - 2 * C - A  # Coefficient for λ

    p = np.array([A, C, D, B])

    return p

# Example usage
"""
    F1 = np.random.rand(3, 3)
    F2 = np.random.rand(3, 3)
    p = fslcm(F1, F2)
    print(p)  
"""