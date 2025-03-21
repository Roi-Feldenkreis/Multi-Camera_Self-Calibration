import numpy as np
from RANSAC import rEG


def add(list, id, value):
    #TODO: creat this function
    return 0


def remove(list, id):
    #TODO: creat this function
    return 0



def findinl(Ws, IdMat, tol):
    '''
     FindInl    find inliers in joint image matrix by pairwise epipolar geometry function IdMatIn = findinl(Ws,IdMat,tol)

    :param Ws:  3MxN joint image matrix
    :param IdMat:  MxN ... 0 -> no point detected
                           1 -> point detected
    :param tol:  [pixels] tolerance for the epipolar geometry
                 the point are accpted as outliers only if they
                 are closer to the epipolar line than tol
    :return:
                IdMat ... MxN ... 0 -> no point detected
                                  1 -> point detected
    '''
    NoCams = IdMat.shape[0]

    # fill the array with structures not_used denoted as 0
    not_used = [{'pts': np.sum(IdMat[i, :])} for i in range(NoCams)]
    used = [{'pts': -1} for _ in range(NoCams)]

    # allocate IdMat for outliers
    IdMatIn = np.zeros_like(IdMat)

    while np.sum([x['pts'] for x in not_used]) > 1 - NoCams:
        buff, id_cam_max = max((x['pts'], i) for i, x in enumerate(not_used))
        used = add(used, id_cam_max, not_used[id_cam_max]['pts'])
        not_used = remove(not_used, id_cam_max)
        Mask = np.tile(IdMat[id_cam_max, :], (NoCams, 1))
        Corresp = Mask & IdMat
        Corresp[id_cam_max, :] = 0
        buff, id_cam_to_pair = max((np.sum(Corresp[i, :]), i) for i in range(NoCams))
        idx_corr_to_pair = np.where(np.sum(IdMat[[id_cam_max, id_cam_to_pair], :], axis=0) == 2)[0]

        if idx_corr_to_pair.size < 8:
            raise ValueError('Not enough points to compute epipolar geometry in RANSAC validation')

        Wspair = np.vstack([Ws[id_cam_max * 3 - 2:id_cam_max * 3 + 1, idx_corr_to_pair],
                            Ws[id_cam_to_pair * 3 - 2:id_cam_to_pair * 3 + 1, idx_corr_to_pair]])

        F, inls = rEG(Wspair, tol, tol, 0.99)
        IdMatIn[id_cam_max, idx_corr_to_pair[inls]] = 1
        IdMat[id_cam_max, :] = 0
        IdMat[id_cam_max, idx_corr_to_pair[inls]] = 1

    return IdMatIn


# Example usage of the findinl function
"""
    Ws = np.random.rand(3 * 5, 10)  # Example 5 cameras, 10 points
    IdMat = (np.random.rand(5, 10) > 0.5).astype(int)  # Random detection matrix
    tol = 1.0  # Example tolerance
    
    IdMatIn = findinl(Ws, IdMat, tol)
    print(IdMatIn)
"""