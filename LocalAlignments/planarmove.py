import numpy as np
from LocalAlignments.nfi2r import nfi2r
from LocalAlignments.planefit import planefit
from OutputFunctions.SaveCalPar import savecalpar
from CoreFunctions.Align3d import align3d


def planarmove(in_data, config):
    """
    Alignment under assumption of planar motion.
    
    Parameters:
    in_data (dict): A dictionary containing camera and 3D point data (like in.Xe, in.Pe).
    config (dict): Configuration parameters, such as file paths and camera usage info.
    
    Returns:
    align (dict): The aligned structure with the specified world frame.
    """
    
    # Fit a plane to the reconstructed points and estimate normal
    plane_n = planefit(in_data['Xe'][0:3, :].T)
    new_n = np.array([0, 0, 1])  # Align to the xy-plane horizontally

    # Calculate rotation axis and angle
    rotaxis = np.cross(plane_n, new_n)
    rotangle = np.arccos(np.dot(plane_n, new_n) / (np.linalg.norm(plane_n) * np.linalg.norm(new_n)))

    # Create the rotation matrix
    R_matrix = nfi2r(rotaxis, rotangle)

    # Define scale and translation
    s = 3
    t = np.array([0, 0, 1]) - s * np.dot(R_matrix, np.mean(in_data['Xe'][0:3, :], axis=1))

    # Build the alignment dictionary
    align = {}
    align['simT'] = {
        's': s,
        'R': R_matrix,
        't': t}

    # Align 3D points and cameras using 3D transformation
    align['P'], align['X'] = align3d(in_data['Pe'], in_data['Xe'], align['simT'])

    # Save aligned data (implementing saving of calibration parameters based on config)
    if True:  # SAVE_STEPHI | SAVE_PGUHA equivalent
        align['Cst'], align['Rot'] = savecalpar(align['P'], config)

    # Drawing scenes (TopView)
    draw_scene(align['X'], align['Cst'].T, align['Rot'], 61, 'cloud',
              'Graphical Output Validation: Aligned data, TopView', config['cal']['cams2use'])

    # Set camera view for top view
    set_camera_target([0, 0, 1])
    set_camera_position([0, 0, 2])

    # Drawing scenes (SideView)
    draw_scene(align['X'], align['Cst'].T, align['Rot'], 62, 'cloud',
              'Graphical Output Validation: Aligned data, SideView', config['cal']['cams2use'])

    # Set camera view for side view
    set_camera_target([0, 0, 0.9])
    set_camera_position([2, 0, 0.9])

    # Save output
    save_figure(61, config['paths']['data'] + 'topview.eps')
    save_figure(62, config['paths']['data'] + 'sideview.eps')

    return align



