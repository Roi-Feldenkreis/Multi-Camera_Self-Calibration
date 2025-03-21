import numpy as np
from IsotropicPointNormalization import isptnorm
import estimateLambda
import estimatePX
import euclidize


def selfcalib(Ws, IdMat):
    """
    Performs the self-calibration algorithm on a measurement matrix.

    Parameters:
    Ws (np.ndarray): The 3*n x m measurement matrix.
    IdMat (np.ndarray): The matrix indicating detected points.

    Returns:
    Xe (np.ndarray): 4 x m matrix containing reconstructed calibration points.
    Pe (np.ndarray): 3*CAMS x 4 matrix containing estimated camera matrices.
    C (np.ndarray): 4 x n matrix containing the camera centers.
    R (np.ndarray): 3*CAMS x 3 matrix containing estimated camera rotation matrices.
    T (np.ndarray): 3 x n matrix containing the camera translation vectors.
    foc (np.ndarray): CAMS x 1 vector containing the focal lengths of the cameras.
    """

    POINTS = Ws.shape[1]
    CAMS = Ws.shape[0] // 3

    if True:
        # Normalize image data (see Hartley, p.91)
        T = []  # The CAMS*3 x 3 normalization transformations
        for i in range(CAMS):
            X_i, T_i = isptnorm(Ws[i*3-2:i*3, IdMat[i, :] > 0].T)
            Ws[i*3-2:i*3, IdMat[i, :] > 0] = np.vstack([X_i.T, np.ones((1, np.sum(IdMat[i, :] > 0)))])
            T.append(T_i)
        T = np.vstack(T)
    else:
        T = np.tile(np.eye(3), (CAMS, 1))

    # Estimate projective depths
    Lambda_est = estimateLambda(Ws, IdMat)
    # Lambda_est = np.ones((CAMS, POINTS))

    if True:
        # Normalize estimated lambdas
        lambnfr = np.sum(Lambda_est ** 2, axis=0)
        Lambda_est = np.sqrt(CAMS) * Lambda_est / np.sqrt(lambnfr)
        lambnfc = np.sum(Lambda_est.T ** 2, axis=1)
        Lambda_est = np.sqrt(POINTS) * Lambda_est / np.sqrt(lambnfc).reshape(-1, 1)

    # No need for negative lambdas
    Lambda_est = np.abs(Lambda_est)

    # Lambda check
    # Employing lambdas, the Ws should have rank 4
    if False:
        lambdaMat = []
        for i in range(CAMS):
            lambdaMat.append(np.tile(Lambda_est[i, :], (3, 1)))
        lambdaMat = np.vstack(lambdaMat)
        Ws_rankcheck = lambdaMat * Ws
        print(np.linalg.svd(Ws_rankcheck), np.linalg.svd(Ws))

    # Compute projective shape and motion
    P, X, Lambda = estimatePX(Ws, Lambda_est)

    # Undo normalization
    for i in range(CAMS):
        P[3*i:3*(i+1), :] = np.linalg.inv(T[3*i:3*(i+1), :]) @ P[3*i:3*(i+1), :]

    # Euclidean reconstruction
    Pe, Xe, C, R, T, foc, warn = euclidize(Ws, Lambda, P, X)

    return Xe, Pe, C, R, T, foc

