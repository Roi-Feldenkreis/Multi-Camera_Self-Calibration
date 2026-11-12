import numpy as np
import matplotlib.pyplot as plt
from OutputFunctions.Draw_Scene import draw_scene
from OutputFunctions.SaveCalPar import savecalpar
from CoreFunctions.Align3d import align3d
from CoreFunctions.Estimate import estimate


def erlangen(in_data, config, REALVIZ = False):


    Cst = in_data['Cst']
    Rot = in_data['Rot']

    draw_scene(in_data['Xe'], Cst.T, Rot, 41, 'cloud', 'Graphical Output Validation: View from top or bottom',
               config['cal']['cams2use'])

    if REALVIZ:
        cam = {
            1: np.array([1.20055, -1.85769, 4.02702]),
            2: np.array([0.382772, -4.49652, 5.28274]),
            3: np.array([-2.99133, -3.90767, 6.1968]),
            4: np.array([-2.92944, -2.40276, -7.78579]),
            5: np.array([-9.00505, -5.61218, -9.73132]),
            6: np.array([-7.76965, -2.94546, -1.63609])}
    else:
        cam = {
            1: np.array([1.16, 0.3, 1.8]),
            2: np.array([1.1, 3.0, 2.5]),
            3: np.array([-0.9, 2.4, 1.56]),
            4: np.array([0.15, -2.5, 2.2]),
            5: np.array([-1.95, -2.65, 2.45]),
            6: np.array([-1.75, -2.2, 1.42])}

    cam_centers = np.column_stack([cam[i] for i in sorted(cam.keys())])
    align = {}

    align['simT'] = estimate(Cst.T, cam_centers)
    align['P'], align['X'] = align3d(in_data['Pe'], in_data['Xe'], align['simT'])

    align['Cst'], align['Rot'] = savecalpar(align['P'], config)

    draw_scene(align['X'], align['Cst'].T, align['Rot'], 61, 'cloud', 'Graphical Output Validation: Aligned data',
               config['cal']['cams2use'])
    ax = plt.gca()
    ax.set_title('Graphical Output Validation: Aligned data')
    ax.view_init(azim=0, elev=0)

    plt.savefig(f"{config['paths']['data']}topview.eps", format='eps')

    draw_scene(align['X'], align['Cst'].T, align['Rot'], 62, 'cloud', 'Graphical Output Validation: Aligned data',
               config['cal']['cams2use'])
    ax = plt.gca()
    ax.set_title('Graphical Output Validation: Aligned data')
    ax.view_init(azim=90, elev=0)

    plt.savefig(f"{config['paths']['data']}sideview.eps", format='eps')

    return align
