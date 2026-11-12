import numpy as np
import matplotlib.pyplot as plt

from OutputFunctions.Draw_Scene import draw_scene
from CoreFunctions.Align3d import align3d
from CoreFunctions.Estimate import estimate
from OutputFunctions.SaveCalPar import savecalpar


def align_existing_camera_centers(in_data, config):
    """
    Aligns camera centers using existing reference camera positions.

    Parameters
    ----------
    in_data : dict
        Contains camera calibration data:
        - 'Cst': Camera centers (3 x N array)
        - 'Rot': Camera rotation matrices
        - 'Pe': Camera projection matrices
        - 'Xe': 3D point coordinates
    config : CalibrationConfig
        Configuration object with data_path and cameras_2_use attributes

    Returns
    -------
    align : dict
        Alignment results containing simT, P, X, Cst, Rot
    """

    Cst = in_data['Cst']
    Rot = in_data['Rot']

    # Load reference camera centers
    cam_file = f"{config.data_path}original_cam_centers.dat"
    Cam = np.loadtxt(cam_file)

    # Visualize original data before alignment
    draw_scene(in_data['Xe'], Cst.T, Rot, 41, 'cloud',
              'Graphical Output Validation: View from top or bottom (no sRt)',
              config.cameras_2_use)

    # Estimate similarity transformation
    align = {'simT': {}}
    align['simT']['s'], align['simT']['R'], align['simT']['t'] = estimate(Cst.T, Cam.T)

    # Apply transformation to align 3D data
    align['P'], align['X'] = align3d(in_data['Pe'], in_data['Xe'], align['simT'])

    # Save aligned camera parameters
    align['Cst'], align['Rot'] = savecalpar(align['P'], config)

    # Visualize aligned data - Top view
    draw_scene(align['X'], align['Cst'].T, align['Rot'], 61, 'cloud',
              'Graphical Output Validation: Aligned data',
              config.cameras_2_use)

    ax = plt.gca()
    ax.set_xlim(ax.get_xlim())  # Lock current limits
    ax.set_ylim(ax.get_ylim())
    ax.set_zlim(ax.get_zlim())
    # Set camera view (equivalent to MATLAB's CameraTarget/CameraPosition)
    ax.view_init(elev=90, azim=-90)  # Top view looking down

    plt.savefig(f"{config.data_path}topview.png", format='png')

    # Visualize aligned data - Side view
    draw_scene(align['X'], align['Cst'].T, align['Rot'], 62, 'cloud',
              'Graphical Output Validation: Aligned data',
              config.cameras_2_use)

    ax = plt.gca()
    ax.view_init(elev=0, azim=0)  # Side view

    plt.savefig(f"{config.data_path}sideview.png", format='png')

    return align