import numpy as np


def findoutl(cam, inliers, INL_TOL, NUM_CAMS_FILL):
    """
    Finds outliers in cameras based on 2D reprojection errors.

    Args:
    cam (list of dicts): A list of dictionaries where each dictionary represents a camera and contains:
                         - 'std2Derr': Standard deviation of 2D errors.
                         - 'mean2Derr': Mean of 2D errors.
                         - 'err2d': Array of 2D errors for each point.
                         - 'idlin': Array of linear indices for the points.
                         - 'visandrec': Array of visible and reconstructed point indices.
    inliers (dict): A dictionary containing:
                    - 'IdMat': A matrix indicating detected points (MxN where M is number of cameras, N is number of points).
                    - 'idx': Array of inlier indices.
    INL_TOL (float): Tolerance for identifying outliers based on 2D errors.
    NUM_CAMS_FILL (int): Number of cameras that should detect a point to consider it an inlier.

    Returns:
    tuple: A tuple containing:
           - outliers (int): The number of outlier points.
           - inliers (dict): Updated inliers dictionary.
    """

    CAMS = len(cam)  # Number of cameras

    # Initialize a matrix to mark outliers
    idxoutMat = np.zeros_like(inliers['IdMat'])

    for i in range(CAMS):
        # Determine if the camera has outliers based on mean and standard deviation of 2D errors
        if (cam[i]['std2Derr'] > cam[i]['mean2Derr']) or (cam[i]['mean2Derr'] > INL_TOL):
            reprerrs = cam[i]['err2d'] - cam[i]['mean2Derr']  # Compute reprojection errors relative to the mean error
            # Identify indices of points with significant reprojection errors
            idxout = np.where((reprerrs > 3 * cam[i]['std2Derr']) & (reprerrs > INL_TOL))[0]
        else:
            idxout = np.array([], dtype=int)  # No outliers if the conditions are not met

        # Mark the identified outliers in the matrix
        idxoutMat[i, cam[i]['idlin'][cam[i]['visandrec'][idxout]]] = 1

    # Zero out columns in IdMat with at least one outlier
    inliers['IdMat'][:, np.sum(idxoutMat, axis=0) > 0] = 0

    # Find indices of inliers that are detected by enough cameras
    inliers['idx'] = np.where(np.sum(inliers['IdMat'], axis=0) >= inliers['IdMat'].shape[0] - NUM_CAMS_FILL)[0]

    # Count the number of outlier points
    outliers = np.sum(np.sum(idxoutMat, axis=0) > 0)

    return outliers, inliers


# Example usage
"""
cam = [
    {
        'std2Derr': 0.5,
        'mean2Derr': 0.3,
        'err2d': np.random.rand(10),
        'idlin': np.arange(10),
        'visandrec': np.arange(10)
    } for _ in range(5)
]

inliers = {
    'IdMat': (np.random.rand(5, 10) > 0.5).astype(int),
    'idx': []
}

INL_TOL = 0.2
NUM_CAMS_FILL = 2

outliers, inliers = findoutl(cam, inliers, INL_TOL, NUM_CAMS_FILL)
print("Outliers:", outliers)
print("Inliers IdMat:", inliers['IdMat'])
print("Inliers idx:", inliers['idx'])
"""
