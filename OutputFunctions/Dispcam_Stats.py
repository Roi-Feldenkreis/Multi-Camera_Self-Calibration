import numpy as np


def disp_cam_stats(cam, Inliers):
    """
    DISP_CAM_STATS display statistics about cameras to the command window
    auxiliary function for the main GoCal script
    with a minor modification is may also write the statistics to a file
    I never needed it.

    Parameters:
    cam - list of dictionaries or objects with attributes like: camId, std2Derr, and mean2Derr
    Inliers - dictionar numpy arrays corresponding to IdMat
    """

    print('CamId    std       mean     #inliers ')
    for i in range(len(cam)):
        print(f'{cam[i]["camId"]:3d}  {cam[i]["std2Derr"]:8.2f}  {cam[i]["mean2Derr"]:8.2f} {int(np.sum(Inliers['IdMat'][i, :])):6d}')

