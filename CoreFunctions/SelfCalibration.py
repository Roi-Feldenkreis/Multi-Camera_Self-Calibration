import numpy as np
from CoreFunctions.IsotropicPointNormalization import isptnorm
from CoreFunctions.EstimateLambda import estimateLambda
from CoreFunctions.EstimatePX import estimatePX
from CoreFunctions.Euclidize import euclidize


def selfcalib(Ws, IdMat, valid=False):
    """
    Performs the self-calibration algorithm on a measurement matrix.

    Parameters:
    Ws (np.ndarray): The 3*n x m measurement matrix.
    IdMat (np.ndarray): The matrix indicating detected points.
    valid (bool): Enable lambda validation check (default: False).

    Returns:
    Xe (np.ndarray): 4 x m matrix containing reconstructed calibration points.
    Pe (np.ndarray): CAMS x 3 x 4 matrix containing estimated camera matrices.
    C (np.ndarray): 4 x n matrix containing the camera centers.
    R (np.ndarray): CAMS x 3 x 3 matrix containing estimated camera rotation matrices.
    T (np.ndarray): CAMS x 3 x 3 matrix containing the normalization transformations.
    foc (np.ndarray): CAMS x 1 vector containing the focal lengths of the cameras.
    """

    POINTS = Ws.shape[1]
    CAMS = Ws.shape[0] // 3

    # Normalize image data - T in CAMS×3×3 format
    T = np.zeros((CAMS, 3, 3))  # CAMS x 3 x 3 normalization transformations
    for i in range(CAMS):
        X_i, T_i = isptnorm(Ws[i * 3 - 2:i * 3, IdMat[i, :] > 0].T)
        Ws[i * 3 - 2:i * 3, IdMat[i, :] > 0] = np.vstack([X_i.T, np.ones((1, np.sum(IdMat[i, :] > 0)))])
        T[i, :, :] = T_i  # Store 3×3 matrix directly

    # Estimate projective depths
    Lambda_est = estimateLambda(Ws, IdMat)
    # Lambda_est = np.ones((CAMS, POINTS))

    # Normalize estimated lambdas
    lambnfr = np.sum(Lambda_est ** 2, axis=0)
    Lambda_est = np.sqrt(CAMS) * Lambda_est / np.sqrt(lambnfr)
    lambnfc = np.sum(Lambda_est.T ** 2, axis=1)
    Lambda_est = np.sqrt(POINTS) * Lambda_est / np.sqrt(lambnfc).reshape(-1, 1)

    # No need for negative lambdas
    Lambda_est = np.abs(Lambda_est)

    # Lambda check
    # Employing lambdas, the Ws should have rank 4
    if valid:
        lambdaMat = []
        for i in range(CAMS):
            lambdaMat.append(np.tile(Lambda_est[i, :], (3, 1)))
        lambdaMat = np.vstack(lambdaMat)
        Ws_rankcheck = lambdaMat * Ws
        print(np.linalg.svd(Ws_rankcheck), np.linalg.svd(Ws))

    # Compute projective shape and motion - P in CAMS×3×4 format
    P, X, Lambda = estimatePX(Ws, Lambda_est)

    # Validate that estimatePX returns P in CAMS×3×4 format
    if len(P.shape) != 3 or P.shape != (CAMS, 3, 4):
        raise ValueError(f"estimatePX should return P in CAMS×3×4 format, got {P.shape}")

    # Undo normalization using CAMS×3×4 formats
    for i in range(CAMS):
        P[i, :, :] = np.linalg.inv(T[i, :, :]) @ P[i, :, :]

    # Euclidean reconstruction
    Pe, Xe, C, R, T, foc, warn = euclidize(Ws, Lambda, P, X)

    # Validate that euclidize returns the expected CAMS×3×4 format
    if len(Pe.shape) != 3 or Pe.shape != (CAMS, 3, 4):
        raise ValueError(f"euclidize should return Pe in CAMS×3×4 format, got {Pe.shape}")

    if len(R.shape) != 3 or R.shape != (CAMS, 3, 3):
        raise ValueError(f"euclidize should return R in CAMS×3×3 format, got {R.shape}")

    return Xe, Pe, C, R, T, foc


# Example usage demonstrating the 3D array format
if __name__ == "__main__":
    """
    Example demonstrating the unified new 3D array format for all matrices.
    """
    print("=== Unified 3D Array Format Example ===")

    # Simulate some example data
    CAMS = 3

    print(f"\n=== All Matrices Now Use 3D Array Format ===")
    print(f"T: {(CAMS, 3, 3)} - Normalization transformations")
    print(f"P: {(CAMS, 3, 4)} - Projection matrices")
    print(f"Pe: {(CAMS, 3, 4)} - Estimated projection matrices")
    print(f"R: {(CAMS, 3, 3)} - Rotation matrices")

    print(f"\n=== Access Patterns ===")
    print(f"Camera i normalization: T[i, :, :]")
    print(f"Camera i projection: P[i, :, :] or Pe[i, :, :]")
    print(f"Camera i rotation: R[i, :, :]")

    print(f"\n=== Benefits ===")
    print(f"✓ Consistent format across all matrices")
    print(f"✓ Intuitive indexing: matrix[camera_id, :, :]")
    print(f"✓ No more complex slicing calculations")
    print(f"✓ Compatible with modern computer vision libraries")