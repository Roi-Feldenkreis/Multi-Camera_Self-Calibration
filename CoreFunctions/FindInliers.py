import numpy as np
from RANSAC_Algorithm.REG import reg


def findinl(Ws, IdMat, tol):
    """
    Find inliers in joint image matrix by pairwise epipolar geometry.

    Parameters
    ----------
    Ws : numpy.ndarray
        Joint image matrix (3*M x N), M = number of cameras, N = number of points
    IdMat : numpy.ndarray
        Indicator matrix (M x N), 0 = no point, 1 = point detected
    tol : float
        Tolerance in pixels for epipolar geometry

    Returns
    -------
    IdMatIn : numpy.ndarray
        Inlier indicator matrix (same shape as IdMat)
    """

    NoCams = IdMat.shape[0]

    # Track camera usage: -1 means processed/used
    not_used = np.array([np.sum(IdMat[i, :]) for i in range(NoCams)])
    used = np.full(NoCams, -1, dtype=int)

    IdMatIn = np.zeros_like(IdMat)
    IdMat_work = IdMat.copy()

    while np.sum(not_used) > (1 - NoCams):
        # Find camera with most detected points
        id_cam_max = np.argmax(not_used)
        used[id_cam_max] = not_used[id_cam_max]
        not_used[id_cam_max] = -1

        # Find correspondences with other cameras
        Mask = np.tile(IdMat_work[id_cam_max, :], (NoCams, 1))
        Corresp = np.logical_and(Mask, IdMat_work.astype(bool))
        Corresp[id_cam_max, :] = False

        # Find camera with most correspondences
        id_cam_to_pair = np.argmax(np.sum(Corresp, axis=1))

        # Get indices of points visible in both cameras
        combined = IdMat_work[id_cam_max, :] + IdMat_work[id_cam_to_pair, :]
        idx_corr_to_pair = np.where(combined == 2)[0]

        if len(idx_corr_to_pair) < 8:
            raise ValueError("Not enough points to compute epipolar geometry in RANSAC validation")

        # Extract corresponding points from both cameras
        Wspair = np.vstack([
            Ws[id_cam_max * 3: id_cam_max * 3 + 3, idx_corr_to_pair],
            Ws[id_cam_to_pair * 3: id_cam_to_pair * 3 + 3, idx_corr_to_pair]])

        # Compute epipolar geometry and find inliers
        F, inls = reg(Wspair, tol, tol, 0.99)

        # Update inlier matrix
        IdMatIn[id_cam_max, idx_corr_to_pair[inls]] = 1

        # Update working IdMat
        IdMat_work[id_cam_max, :] = 0
        IdMat_work[id_cam_max, idx_corr_to_pair[inls]] = 1

    return IdMatIn
