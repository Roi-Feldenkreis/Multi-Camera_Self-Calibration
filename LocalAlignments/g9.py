import os
import numpy as np
import matplotlib.pyplot as plt
from OutputFunctions.Draw_Scene import draw_scene
from OutputFunctions.SaveCalPar import savecalpar
from CoreFunctions.Align3d import align3d
from CoreFunctions.Estimate import estimate


def g9(in_data, config):
    """
    Python version of the g9 function for local alignment in room G9.

    Parameters:
    - in_data: input data containing Xe, Cst, and Rot
    - config: configuration for visualization and cameras

    Returns:
    - align: a dictionary containing alignment information
    """

    Cst = in_data['Cst']
    Rot = in_data['Rot']

    # Draw the initial scene from top or bottom view
    draw_scene(in_data['Xe'], Cst.T, Rot, 41, 'cloud', 'Graphical Output Validation: View from top or bottom', config['cal']['cams2use'])

    # Define absolute world frame camera positions
    cam = {}
    cam[1] = {'C': np.array([3, 1.8, 0.2])}
    cam[2] = {'C': np.array([-2.2, 2.05, 0.1])}
    cam[3] = {'C': np.array([-2.2, 2.05, 3.2])}

    # Specify the indices of cameras to use in alignment
    idx = [1, 2, 3]

    # Similarity transformation
    Cst_selected = Cst[idx, :].T
    cam_selected = np.column_stack([cam[i]['C'] for i in idx])

    align = {}
    align['simT'] = {}
    align['simT']['s'], align['simT']['R'], align['simT']['t'] = estimate(Cst_selected, cam_selected)

    # Align the 3D points
    align['P'], align['X'] = align3d(in_data['Pe'], in_data['Xe'], align['simT'])

    # Save the aligned data
    if True:  # Equivalent to the SAVE_STEPHI | SAVE_PGUHA condition
        align['Cst'], align['Rot'] = savecalpar(align['P'], config)

    # Visualize the aligned data
    draw_scene(align['X'], align['Cst'].T, align['Rot'], 61, 'cloud', 'Graphical Output Validation: Aligned data', config['cal']['cams2use'])
    ax = plt.gca()
    ax.view_init()

    # Save top view
    plt.figure(61)
    plt.savefig(os.path.join(config['paths']['data'], 'topview.eps'))

    # Draw side view
    draw_scene(align['X'], align['Cst'].T, align['Rot'], 62, 'cloud', 'Graphical Output Validation: Aligned data', config['cal']['cams2use'])
    ax.view_init()
    plt.figure(61)
    plt.savefig(os.path.join(config['paths']['data'], 'topview.eps'), format='eps')

    # Save side view
    plt.figure(62)
    plt.savefig(os.path.join(config['paths']['data'], 'sideview.eps'))

    return align
