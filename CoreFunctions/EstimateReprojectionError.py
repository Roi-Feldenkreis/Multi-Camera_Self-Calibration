import numpy as np


def reprerror(cam, Pe, Xe, FRAMES, inliers):
    """
    Estimate reprojection error for each camera.

    Args:
    cam (list of dicts): A list of dictionaries where each dictionary represents a camera and contains:
                         - 'idlin': Array of linear indices for the points.
                         - 'xgt': Ground truth 2D coordinates of the points.
    Pe (numpy.ndarray): 3nx4 Euclidean motion matrix (n is the number of cameras).
    Xe (numpy.ndarray): 4xm Euclidean shape matrix (m is the number of points).
    FRAMES (int): Number of frames.
    inliers (dict): A dictionary containing:
                    - 'idx': Array of inlier indices.

    Returns:
    list of dicts: Updated list of camera dictionaries with reprojection errors and statistics.
    """

    CAMS = Pe.shape[0] // 3  # Number of cameras

    for i in range(CAMS):
        # Project 3D points to 2D using the camera matrix
        xe = Pe[(3 * i):(3 * i + 3), :] @ Xe
        cam[i]['xe'] = xe / xe[2, :]  # Normalize homogeneous coordinates

        # Create masks for reconstructed and visible points
        mask_rec = np.zeros(FRAMES, dtype=int)
        mask_vis = np.zeros(FRAMES, dtype=int)
        mask_rec[inliers['idx']] = 1
        mask_vis[cam[i]['idlin']] = 1
        mask_both = mask_vis & mask_rec

        # Create unmasked cumulative sums
        unmask_rec = np.cumsum(mask_rec)
        unmask_vis = np.cumsum(mask_vis)

        # Identify points that are both visible and reconstructed
        cam[i]['recandvis'] = unmask_rec[np.logical_and(~np.logical_xor(mask_rec, mask_both), mask_rec)]
        cam[i]['visandrec'] = unmask_vis[np.logical_and(~np.logical_xor(mask_rec, mask_both), mask_rec)]

        # Debug statements to check shapes and indices
        print(f"Camera {i}:")
        print(f"xe shape: {cam[i]['xe'].shape}")
        print(f"xgt shape: {cam[i]['xgt'].shape}")
        print(f"recandvis indices: {cam[i]['recandvis']}")
        print(f"visandrec indices: {cam[i]['visandrec']}")

        recandvis_indices = cam[i]['recandvis'].astype(int)
        visandrec_indices = cam[i]['visandrec'].astype(int)

        # Ensure indices are within bounds
        recandvis_indices = recandvis_indices[recandvis_indices < cam[i]['xe'].shape[1]]
        visandrec_indices = visandrec_indices[visandrec_indices < cam[i]['xgt'].shape[1]]

        if len(recandvis_indices) == 0 or len(visandrec_indices) == 0:
            cam[i]['err2d'] = np.array([])
            cam[i]['mean2Derr'] = np.nan
            cam[i]['std2Derr'] = np.nan
            continue

        cam[i]['err2d'] = np.sqrt(
            np.sum((cam[i]['xe'][0:2, recandvis_indices] - cam[i]['xgt'][0:2, visandrec_indices]) ** 2, axis=0))
        cam[i]['mean2Derr'] = np.mean(cam[i]['err2d'])
        cam[i]['std2Derr'] = np.std(cam[i]['err2d'])

    return cam


# Example usage
'''
    cam = [
        {
            'idlin': np.array([0, 1, 2, 3, 4]),
            'xgt': np.random.rand(2, 5)
        } for _ in range(3)
    ]
    
    Pe = np.random.rand(9, 4)
    Xe = np.random.rand(4, 5)
    FRAMES = 5
    inliers = {'idx': np.array([0, 1, 2, 3, 4])}
    
    cam = reprerror(cam, Pe, Xe, FRAMES, inliers)
    for c in cam:
        print("Reprojection Errors:", c['err2d'])
        print("Mean 2D Error:", c['mean2Derr'])
        print("Std 2D Error:", c['std2Derr'])
'''

