import numpy as np
from CoreFunctions.uP2X import uP2X


def estimateX(data, IdMat, cam, config):
    """
    Estimate 3D points robustly by finding the best camera tuple for reconstruction.

    Parameters:
    -----------
    data : dict
        Data structure containing Ws and Pmat
    IdMat : numpy.ndarray
        Current point identification matrix
    cam : list
        Array of camera structures
    config : dict
        Configuration parameters

    Returns:
    --------
    reconstructed : dict
        Dictionary containing:
        - ptsIdx: indexes of points used for the reconstruction
        - X: reconstructed 3D points
        - CamIdx: indexes of cameras used for the reconstruction
    """
    # Get sample size from config
    SS = config['cal']['NTUPLES']  # sample size
    MIN_POINTS = config['cal']['MIN_PTS_VAL']  # minimal number of correspondences in the sample

    Ws = data['Ws']
    Pmat = data['Pmat']

    CAMS = IdMat.shape[0]
    FRAMES = IdMat.shape[1]

    # Create indexes for all possible SS-tuples
    samples = []

    # Generate all possible camera combination tuples based on sample size
    if SS == 2:
        for i in range(CAMS):
            for j in range(i + 1, CAMS):
                samples.append({'CamIds': [i, j]})
    elif SS == 3:
        for i in range(CAMS):
            for j in range(i + 1, CAMS):
                for k in range(j + 1, CAMS):
                    samples.append({'CamIds': [i, j, k]})
    elif SS == 4:
        for i in range(CAMS):
            for j in range(i + 1, CAMS):
                for k in range(j + 1, CAMS):
                    for l in range(k + 1, CAMS):
                        samples.append({'CamIds': [i, j, k, l]})
    elif SS == 5:
        for i in range(CAMS):
            for j in range(i + 1, CAMS):
                for k in range(j + 1, CAMS):
                    for l in range(k + 1, CAMS):
                        for m in range(l + 1, CAMS):
                            samples.append({'CamIds': [i, j, k, l, m]})

    print(f'Computing reconstruction from all {len(samples)} camera {SS}-tuples')

    # Create triple indexes
    tripleIdx = []
    for i in range(CAMS):
        tripleIdx.append([i * 3 - 2, i * 3 - 1, i * 3])

    # For all possible combinations of SS-tuples of cameras
    # do the linear 3D reconstruction if enough points available
    for i in range(len(samples)):
        cam_indices = samples[i]['CamIds']
        # Find points that are visible in all cameras in the tuple
        ptsIdx = np.where(np.sum(IdMat[cam_indices, :], axis=0) == SS)[0]

        if len(ptsIdx) > MIN_POINTS:
            # Extract relevant camera triple indices
            triple_indices = []
            for cam_idx in cam_indices:
                triple_indices.extend(tripleIdx[cam_idx])

            # Get corresponding projection matrices
            cam_Pmats = [Pmat[cam_idx] for cam_idx in cam_indices]

            # Reconstruct 3D points using uP2X
            X = uP2X(Ws[triple_indices][:, ptsIdx], cam_Pmats)

            # Compute reprojections for error calculation
            for j in range(CAMS):
                # Project 3D points to image coordinates
                xe = np.dot(Pmat[j], X)
                cam[j]['xe'] = xe / np.tile(xe[2, :], (3, 1))

                # Create masks for validation
                mask = {}
                mask['rec'] = np.zeros(FRAMES)  # mask of points that survived validation so far
                mask['vis'] = np.zeros(FRAMES)  # mask of visible points

                mask['rec'][ptsIdx] = 1
                mask['vis'][cam[j]['ptsLoaded']] = 1
                mask['both'] = mask['vis'] & mask['rec']  # points both visible and reconstructed

                unmask = {}
                unmask['rec'] = np.cumsum(mask['rec'])
                unmask['vis'] = np.cumsum(mask['vis'])

                # Find indices of points that are both reconstructed and visible
                cam[j]['recandvis'] = unmask['rec'][
                    np.logical_and(~np.logical_xor(mask['rec'], mask['both']), mask['rec']).astype(bool)]
                cam[j]['visandrec'] = unmask['vis'][
                    np.logical_and(~np.logical_xor(mask['rec'], mask['both']), mask['rec']).astype(bool)]

                # Calculate reprojection errors
                cam[j]['err2d'] = np.sum(
                    (cam[j]['xe'][0:2, cam[j]['recandvis']] - cam[j]['xgt'][0:2, cam[j]['visandrec']]) ** 2)
                cam[j]['mean2Derr'] = np.mean(cam[j]['err2d'])
                cam[j]['std2Derr'] = np.std(cam[j]['err2d'])

            # Store error metrics for this sample
            samples[i]['mean2Derrs'] = [cam[j]['mean2Derr'] for j in range(CAMS)]
            samples[i]['std2Derrs'] = sum([cam[j]['std2Derr'] for j in range(CAMS)])
            samples[i]['mean2Derr'] = sum(samples[i]['mean2Derrs'])
        else:
            # Not enough points for reconstruction
            samples[i]['mean2Derr'] = 9e99
            samples[i]['std2Derrs'] = 9e99

    # Find the best sample (minimum error)
    mean_errors = [sample.get('mean2Derr', 9e99) for sample in samples]
    std_errors = [sample.get('std2Derrs', 9e99) for sample in samples]
    total_errors = [mean + std for mean, std in zip(mean_errors, std_errors)]
    idxBest = np.argmin(total_errors)

    # Recompute with the best values
    reconstructed = {}
    reconstructed['ptsIdx'] = np.where(np.sum(IdMat[samples[idxBest]['CamIds'], :], axis=0) == SS)[0]

    # Extract indices for the best cameras
    triple_indices = []
    for cam_idx in samples[idxBest]['CamIds']:
        triple_indices.extend(tripleIdx[cam_idx])

    # Get projection matrices for best cameras
    best_Pmats = [Pmat[cam_idx] for cam_idx in samples[idxBest]['CamIds']]

    # Compute final 3D reconstruction
    reconstructed['X'] = uP2X(Ws[triple_indices][:, reconstructed['ptsIdx']], best_Pmats)
    reconstructed['CamIdx'] = samples[idxBest]['CamIds']

    return reconstructed