import numpy as np
from scipy.linalg import svd


def estimatePX(Ws, Lambda):
    """
    Estimate the projective shape and motion

    Parameters:
    Ws (ndarray): 3*nxm measurement matrix
    Lambda (ndarray): nxm matrix containing some initial projective depths

    Returns:
    P (ndarray): 3*nx4 matrix containing the projective motion
    X (ndarray): 4xm matrix containing the projective shape
    Lambda (ndarray): the new estimation of the projective depths
    """
    n = Ws.shape[0] // 3
    m = Ws.shape[1]

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

        # Projective shape X and motion P
        P = U @ D[:U.shape[1], :4]
        X = V[:, :4].T
        # U @ D @ V.T == P @ X

        # Correct projective depths
        normfact = np.sum(P[2::3, :].T ** 2, axis=0)
        Lambda_old = Lambda_new.copy()
        Lambda_new = P[2::3, :] / np.sqrt(normfact[:, None]) @ X

        # Normalize lambdas
        lambnfr = np.sum(Lambda_new ** 2, axis=0)
        Lambda_new = np.sqrt(n) * Lambda_new / np.sqrt(lambnfr)

        lambnfc = np.sum(Lambda_new.T ** 2, axis=0)
        Lambda_new = np.sqrt(m) * Lambda_new / np.sqrt(lambnfc[:, None])

        for i in range(n):
            for j in range(m):
                Ws_updated[3 * i, j] = Ws[3 * i, j] * Lambda_new[i, j]
                Ws_updated[3 * i + 1, j] = Ws[3 * i + 1, j] * Lambda_new[i, j]
                Ws_updated[3 * i + 2, j] = Ws[3 * i + 2, j] * Lambda_new[i, j]

        iterations += 1
        errs.append(np.sum(np.abs(Lambda_old - Lambda_new)))

    U, D, Vt = svd(Ws_updated)
    V = Vt.T

    # Compute new projective shape and motion
    P = U @ D[:U.shape[1], :4]
    X = V[:, :4].T
    X = X / X[3, :]

    Lambda = Lambda_new

    return P, X, Lambda
