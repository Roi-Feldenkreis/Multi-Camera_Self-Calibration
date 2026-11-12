import numpy as np


def reprerror(cam, Pe, Xe, FRAMES, inliers):
    """
    Estimate reprojection error for each camera.

    Parameters
    ----------
    cam : list of dict
        List of camera dictionaries containing 'idlin' and 'xgt'
    Pe : numpy.ndarray
        (3*CAMS x 4) Euclidean projection matrices
    Xe : numpy.ndarray
        (4 x m) Euclidean 3D points
    FRAMES : int
        Total number of frames
    inliers : dict
        Dictionary containing 'idx' array of inlier indices
    Returns
    -------
    cam : list of dict
        Updated camera list with reprojection errors
    """

    CAMS = Pe.shape[0] // 3

    for i in range(CAMS):
        # Project 3D points to 2D using camera matrix
        xe = Pe[3*i:3*i+3, :] @ Xe
        cam[i]['xe'] = xe / np.tile(xe[2, :], (3, 1))  # Normalize homogeneous coordinates

        # Create masks for reconstructed and visible points
        mask_rec = np.zeros(FRAMES, dtype=int)  # Points that survived validation
        mask_vis = np.zeros(FRAMES, dtype=int)  # Points visible in this camera

        mask_rec[inliers['idx']] = 1
        mask_vis[cam[i]['idlin']] = 1
        mask_both = mask_vis & mask_rec  # Points both visible AND reconstructed

        # Create cumulative index mappings
        unmask_rec = np.cumsum(mask_rec)
        unmask_vis = np.cumsum(mask_vis)

        # Find points that are both visible and reconstructed
        # Logic: ~xor(mask_rec, mask_both) & mask_rec
        # This selects points where mask_rec=1 AND mask_both=1 (i.e., just mask_both)
        selection = np.logical_and(~np.logical_xor(mask_rec, mask_both), mask_rec).astype(bool)

        # Get indices (subtract 1 to convert from MATLAB 1-based to Python 0-based)
        cam[i]['recandvis'] = unmask_rec[selection] - 1
        cam[i]['visandrec'] = unmask_vis[selection] - 1

        # Ensure indices are integers
        cam[i]['recandvis'] = cam[i]['recandvis'].astype(int)
        cam[i]['visandrec'] = cam[i]['visandrec'].astype(int)

        # Calculate 2D reprojection error
        xe_pts = cam[i]['xe'][0:2, cam[i]['recandvis']]
        xgt_pts = cam[i]['xgt'][0:2, cam[i]['visandrec']]

        cam[i]['err2d'] = np.sqrt(np.sum((xe_pts - xgt_pts) ** 2, axis=0))
        cam[i]['mean2Derr'] = np.mean(cam[i]['err2d'])
        cam[i]['std2Derr'] = np.std(cam[i]['err2d'])

    return cam
