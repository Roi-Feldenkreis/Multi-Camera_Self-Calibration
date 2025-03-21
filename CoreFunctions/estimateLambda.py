import numpy as np
from scipy.linalg import svd
from scipy.linalg import norm


def estimateLambda(Ws, pair):
    """
    Estimates initial projective depths given the measurement matrix.

    This algorithm is based on a paper by P. Sturm and B. Triggs called
    "A Factorization Based Algorithm for Multi-Image Projective Structure
    and Motion" (1996). Projective depths are estimated using fundamental matrices.

    Parameters:
    Ws (numpy.ndarray): The 3*nxm measurement matrix (normalized before calling this function).
    pair (list): List of dictionaries containing Fundamental matrices ('F') and
                 indexes of points used for their computations ('idxin').

    Returns:
    Lambda (numpy.ndarray): nxm matrix containing the estimated projective depths.
    """

    n = Ws.shape[0] // 3  # Number of cameras
    m = Ws.shape[1]  # Number of frames

    # Initialize the Lambda matrix with ones
    Lambda = np.ones((n, m))

    for i in range(n - 1):
        j = i + 1
        F_ij = pair[i]['F']

        # Compute the epipole from F_ij * e_ij == 0
        U, S, Vt = svd(F_ij, full_matrices=False)
        e_ij = Vt[-1, :]  # Epipole is the last column of V

        for p in pair[i]['idxin']:
            q_ip = Ws[i * 3:i * 3 + 3, p]
            q_jp = Ws[j * 3:j * 3 + 3, p]

            # Calculate the projective depth for point p in camera j
            cross_product = np.cross(e_ij, q_ip)
            Lambda[j, p] = Lambda[i, p] * (norm(cross_product) ** 2 / np.sum(cross_product * (F_ij.T @ q_jp)))

    return Lambda

# Example usage:
# Assuming Ws is a 3*nxm matrix and pair is a list of dictionaries with keys 'F' and 'idxin'
# Lambda = estimateLambda(Ws, pair)
