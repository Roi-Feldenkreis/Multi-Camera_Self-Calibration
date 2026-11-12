import numpy as np
from math import acos
import matplotlib.pyplot as plt
from LocalAlignments.nfi2r import nfi2r
from LocalAlignments.planefit import planefit
from OutputFunctions.SaveCalPar import savecalpar
from CoreFunctions.Align3d import align3d



def planarcams(in_data, cam, config, idx):
    """
    Aligns cameras under the assumption of planar arrangements.
    """

    # Fit a plane to the reconstructed points and estimate the normal
    plane_normal = planefit(in_data['Ce'][:, idx].T)

    # Align the xy-plane horizontally
    new_normal = np.array([0, 0, 1])
    rot_axis = np.cross(new_normal, plane_normal)
    rot_angle = acos(np.dot(plane_normal, new_normal) / (np.linalg.norm(plane_normal) * np.linalg.norm(new_normal)))

    # Compute the rotation matrix
    R = nfi2r(rot_axis, rot_angle)

    # Scaling factor and translation
    s = 3
    t = np.array([0, 0, 1]) - s * R @ np.mean(in_data['Xe'][0:3, :], axis=1)

    # Store the similarity transformation
    align = {}
    align['simT'] = {'s': s, 'R': R, 't': t}

    # Apply the alignment transformation
    align['P'], align['X'] = align3d(in_data['Pe'], in_data['Xe'], align['simT'])

    # Save aligned data
    align['Cst'], align['Rot'] = savecalpar(align['P'], config)

    # Draw the top view
    draw_scene(align['X'], align['Cst'].T, align['Rot'], 61, 'cloud', 'Graphical Output Validation: Aligned data, TopView', config['cal']['cams2use'])
    plt.figure(61)
    plt.gca().set_title('Top View')
    plt.gca().set_xlabel('X')
    plt.gca().set_ylabel('Y')

    # Draw the side view
    draw_scene(align['X'], align['Cst'].T, align['Rot'], 62, 'cloud', 'Graphical Output Validation: Aligned data, SideView', config['cal']['cams2use'])
    plt.figure(62)
    plt.gca().set_title('Side View')
    plt.gca().set_xlabel('X')
    plt.gca().set_ylabel('Y')

    return align, cam

