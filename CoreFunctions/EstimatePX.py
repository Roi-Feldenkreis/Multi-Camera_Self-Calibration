import numpy as np
from scipy.linalg import svd


def estimatePX(Ws, Lambda):
    """
    Estimate the projective shape and motion

    Parameters:
    Ws (ndarray): 3*nxm measurement matrix
    Lambda (ndarray): nxm matrix containing some initial projective depths

    Returns:
    P (ndarray): n x 3 x 4 matrix containing the projective motion (NEW FORMAT)
    X (ndarray): 4xm matrix containing the projective shape
    Lambda (ndarray): the new estimation of the projective depths
    """
    n = Ws.shape[0] // 3  # Number of cameras (CAMS)
    m = Ws.shape[1]  # Number of points

    # Compute the first updated Ws
    Ws_updated = np.zeros_like(Ws)
    for i in range(n):
        for j in range(m):
            Ws_updated[3 * i, j] = Ws[3 * i, j] * Lambda[i, j]
            Ws_updated[3 * i + 1, j] = Ws[3 * i + 1, j] * Lambda[i, j]
            Ws_updated[3 * i + 2, j] = Ws[3 * i + 2, j] * Lambda[i, j]

    Lambda_new = Lambda.copy()
    iterations = 0
    errs = [1e10 * 99.9, 1e10 * 99]
    tol = 1e-3

    while (errs[iterations] - errs[iterations + 1]) > tol:
        U, D, Vt = svd(Ws_updated)
        V = Vt.T

        # Set elements of D beyond the 4th to 0
        D[4:] = 0
        D = np.diag(D)

        # Projective shape X and motion P (P still in stacked format during computation)
        P_stacked = U @ D[:U.shape[1], :4]  # Stacked format during iterations
        X = V[:, :4].T
        # U @ D @ V.T == P @ X

        # Correct projective depths using stacked format
        # Extract every 3rd row starting from index 2 (camera Z-directions)
        P_z_rows = P_stacked[2::3, :]  # Shape: (n, 4) - all cameras' 3rd rows
        normfact = np.sum(P_z_rows ** 2, axis=1)  # Shape: (n,)
        Lambda_old = Lambda_new.copy()
        Lambda_new = (P_z_rows / np.sqrt(normfact[:, None])) @ X

        # Normalize lambdas
        lambnfr = np.sum(Lambda_new ** 2, axis=0)
        Lambda_new = np.sqrt(n) * Lambda_new / np.sqrt(lambnfr)

        lambnfc = np.sum(Lambda_new ** 2, axis=1)
        Lambda_new = np.sqrt(m) * Lambda_new / np.sqrt(lambnfc[:, None])

        # Update Ws with new lambdas
        for i in range(n):
            for j in range(m):
                Ws_updated[3 * i, j] = Ws[3 * i, j] * Lambda_new[i, j]
                Ws_updated[3 * i + 1, j] = Ws[3 * i + 1, j] * Lambda_new[i, j]
                Ws_updated[3 * i + 2, j] = Ws[3 * i + 2, j] * Lambda_new[i, j]

        iterations += 1
        errs.append(np.sum(np.abs(Lambda_old - Lambda_new)))

    # Final SVD computation
    U, D, Vt = svd(Ws_updated)
    V = Vt.T

    # Compute final projective shape and motion in stacked format
    P_stacked = U @ D[:U.shape[1], :4]  # Shape: (3*n, 4)
    X = V[:, :4].T
    X = X / X[3, :]  # Normalize homogeneous coordinates

    # Convert P from stacked format (3*n × 4) to 3D array format (n × 3 × 4)
    P = np.zeros((n, 3, 4))
    for i in range(n):
        P[i, :, :] = P_stacked[3 * i:3 * (i + 1), :]

    Lambda = Lambda_new

    return P, X, Lambda
