import os
import numpy as np
from CoreFunctions.RQ_decomposition import rq


def savecalpar(P, config):
    """
    Save calibration parameters in different formats.

    Parameters
    ----------
    P : numpy.ndarray
        (3*CAMS x 4) matrix containing projection matrices from selfcalibration
    config : CalibrationConfig
        Configuration object with data_path, basename, cameras_2_use

    Returns
    -------
    Cst : numpy.ndarray
        (CAMS x 3) matrix containing camera centers in world coordinates
    Rot : numpy.ndarray
        (3*CAMS x 3) matrix containing camera rotation matrices
    """

    idxused = config.cameras_2_use
    CAMS = P.shape[0] // 3

    Cst = np.zeros((CAMS, 3))
    Pst = np.zeros((3 * CAMS, 3))
    Rot = []

    # Make a copy to avoid modifying original P
    P = P.copy()

    for i in range(CAMS):
        # Extract camera projection matrix
        Pmat = P[i*3:i*3+3, :]

        # Save individual P matrix
        cal_pmat_path = os.path.join(str(config.data_path), f"{config.basename}{idxused[i]}.Pmat.cal")
        np.savetxt(cal_pmat_path, Pmat, fmt='%f')

        # Normalize projection matrix
        sc = np.linalg.norm(P[i*3+2, 0:3])
        P[i*3:i*3+3, :] = P[i*3:i*3+3, :] / sc

        # RQ decomposition
        K, R = rq(P[i*3:i*3+3, 0:3])

        # Translation vector and camera center
        tvec = np.linalg.inv(K) @ P[i*3:i*3+3, 3]
        C = -R.T @ tvec

        # Stephi calibration parameters
        Pstephi = R.T @ np.linalg.inv(K)
        Pst[i*3:i*3+3, :] = Pstephi
        Cst[i, :] = C.T

        # Save Stephi format calibration file
        st_cal_par_path = os.path.join(str(config.data_path), f"{config.basename}{idxused[i]}.cal")
        with open(st_cal_par_path, 'w') as f:
            f.write(f'C1 = {C[0]:f} \n')
            f.write(f'C2 = {C[1]:f} \n')
            f.write(f'C3 = {C[2]:f} \n')
            f.write('\n')
            f.write(f'P11 = {Pstephi[0, 0]:f} \n')
            f.write(f'P12 = {Pstephi[0, 1]:f} \n')
            f.write(f'P13 = {Pstephi[0, 2]:f} \n')
            f.write(f'P21 = {Pstephi[1, 0]:f} \n')
            f.write(f'P22 = {Pstephi[1, 1]:f} \n')
            f.write(f'P23 = {Pstephi[1, 2]:f} \n')
            f.write(f'P31 = {Pstephi[2, 0]:f} \n')
            f.write(f'P32 = {Pstephi[2, 1]:f} \n')
            f.write(f'P33 = {Pstephi[2, 2]:f} \n')

        Rot.append(R)

    Rot = np.vstack(Rot)

    # Save combined files
    np.savetxt(os.path.join(str(config.data_path), "Pst.dat"), Pst, fmt='%f')
    np.savetxt(os.path.join(str(config.data_path), "Cst.dat"), Cst, fmt='%f')

    return Cst, Rot