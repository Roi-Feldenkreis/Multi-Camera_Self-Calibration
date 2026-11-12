import numpy as np


def workvolume(cam, room=None, imres=None, idxcams=None):
    """
    Calculate the working volume for a set of cameras.

    Parameters:
    -----------
    cam : list
        List of camera structures, each containing a projection matrix P
    room : list, optional
        Room dimensions as [x_min, x_max, y_min, y_max, z_min, z_max]
    imres : numpy.ndarray, optional
        Image resolutions for each camera as [width, height]
    idxcams : list, optional
        Indices of cameras to use

    Returns:
    --------
    Xmat : numpy.ndarray
        Matrix of 3D points in the working volume
    idxisa : numpy.ndarray
        Indices of points that are visible in all cameras
    """
    STEP = 0.1  # Step size for generating 3D points

    # Set default camera indices if not provided
    if idxcams is None:
        idxcams = list(range(len(cam)))

    # Set default image resolution if not provided
    if imres is None:
        imres = np.tile([640, 480], (len(idxcams), 1))
    else:
        imres = imres[idxcams, :]

    # Set default room dimensions if not provided
    if room is None:
        room = [-3, 3, -3, 3, 0, 3]  # [x_min, x_max, y_min, y_max, z_min, z_max]

    # Compose Pmat containing all P matrices
    Pmat = np.vstack([cam[i]['P'] for i in idxcams])

    # Create 3D points grid
    zcoor = np.arange(room[4], room[5] + STEP, STEP)
    znum = len(zcoor)

    ycoor = np.arange(room[2], room[3] + STEP, STEP)
    ynum = len(ycoor)

    xcoor = np.arange(room[0], room[1] + STEP, STEP)
    xnum = len(xcoor)

    # Create coordinate vectors
    yvec = np.tile(ycoor, znum).reshape(-1)
    zvec = np.repeat(zcoor, ynum)
    xvec = np.repeat(np.tile(xcoor, znum * ynum), 1).reshape(-1)

    # Create homogeneous coordinate matrix
    Xmat = np.column_stack((xvec, np.tile(np.column_stack((yvec, zvec)), (xnum, 1))))
    Xmat = np.column_stack((Xmat, np.ones(Xmat.shape[0])))

    print(f"Size of Xmat: {Xmat.shape}")

    # Project points into all cameras
    umat = Pmat @ Xmat.T

    # Normalize projected points
    scalemat = np.vstack([np.tile(umat[3 * i, :], (3, 1)) for i in range(len(idxcams))])
    umat = umat / scalemat

    # Create mask for points in the image boundaries
    mask = np.zeros_like(umat)
    for i in range(len(idxcams)):
        mask[3 * i, :] = umat[3 * i, :] < imres[i, 0]
        mask[3 * i + 1, :] = umat[3 * i + 1, :] < imres[i, 1]
        mask[3 * i + 2, :] = 1

    # Points must also be in front of the camera (positive z-coordinate)
    mask = mask * (umat > 0)

    # Find points visible in all cameras
    idxisa = np.where(np.sum(mask, axis=0) == mask.shape[0])[0]

    return Xmat, idxisa