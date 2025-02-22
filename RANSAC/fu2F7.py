import numpy as np
from scipy.linalg import null_space
from numpy.polynomial import Polynomial as P
from Lin_Fm import lin_fm
from fslcm import fslcm


def fu2F7(u):
     """
    Computes the fundamental matrix using the 7-point algorithm.

    Parameters:
    -----------
    u : numpy.ndarray a 6x7 matrix containing 7 point correspondences in homogeneous coordinates.
        - The first three rows represent points from the first image.
        - The last three rows represent the corresponding points from the second image.

    Returns:
    --------
    Fs : numpy.ndarray a 3x3xN array containing up to 3 possible fundamental matrices.

    Explanation:
    ------------
    This function implements the **7-point algorithm** for estimating the fundamental matrix (F):
    1. Calls `lin_fm(u)`, which constructs a coefficient matrix `Z` based on the input points.
    2. Computes the **null space** of `Z`, which should have **two basis vectors** (F1 and F2) if the sample is non-degenerate.
    3. If the null space has more than two vectors, the sample is degenerate, and the function returns an empty list.
    4. Reshapes the null space vectors into two 3×3 matrices: `F1` and `F2`.
    5. Solves for the scalar **λ** by computing the determinant of a linear combination of `F1` and `F2`, leading to a cubic polynomial.
    6. Finds the **roots** of this polynomial, which correspond to valid `λ` values.
    7. Computes the possible fundamental matrices as `F = λ * F1 + (1 - λ) * F2`.

    The output is a set of possible fundamental matrices that satisfy the 7-point constraint.

    """
    Z = lin_fm(u)
    NullSp = null_space(Z)

    if NullSp.shape[1] > 2:
        return []  # degenerated sample

    F1 = NullSp[:, 0].reshape(3, 3)
    F2 = NullSp[:, 1].reshape(3, 3)
    p = fslcm(F1, F2)
    aroots = P.roots(p)

    Fs = np.zeros((3, 3, len(aroots)))

    for i, l in enumerate(aroots):
        Ft = F1 * l + F2 * (1 - l)
        Fs[:, :, i] = Ft

    return Fs

# Example usage
"""
    u = np.random.rand(9, 9)  # Example input
    Fs = fu2F7(u)
    print(Fs)
"""
