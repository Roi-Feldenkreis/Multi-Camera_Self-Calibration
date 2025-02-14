import numpy as np
import warnings
import fu2F7
import mfFDs
import nsamples
import u2F

def rEG(u, th, th7=None, conf=0.95):
    """
        The function rEG is a robust computation of epipolar geometry based on RANSAC
    :param u: point correspondences (6xn), where n is the number of corrs.
    :param th: threshold value for the Sampson's distance (see FDs)
    :param th7: threshold for inliers to iterate on F (default = th)
    :param conf: confidence level of self-termination  (default = .95)

    :return:
        F: The best estimated fundamental matrix.
        inls: An array indicating which correspondences are inliers.
    """
    MAX_SAM = 100000
    iter_amount = 0.5

    if th7 is None:
        th7 = th

    len_u = u.shape[1]
    ptr = np.arange(len_u)
    max_i = 8
    max_sam = MAX_SAM

    no_sam = 0
    no_mod = 0

    F = None
    inls = None

    while no_sam < max_sam:
        for pos in range(7):
            idx = pos + int(np.ceil(np.random.rand() * (len_u - pos)))
            ptr[[pos, idx]] = ptr[[idx, pos]]

        no_sam += 1

        aFs = fu2F7(u[:, ptr[:7]])

        for i in range(aFs.shape[2]):
            no_mod += 1
            aF = aFs[:, :, i]
            Ds = mfFDs(aF, u)
            v = Ds < th
            v7 = Ds < th7
            no_i = np.sum(v)

            if max_i < no_i:
                inls = v
                F = aF
                max_i = no_i
                max_sam = min(max_sam, nsamples(max_i, len_u, 7, conf))

            if np.sum(v7) >= 8 + iter_amount * (max_i - 8):
                aF = u2F(u[:, v7])
                Ds = mfFDs(aF, u)
                v = Ds < th
                no_i = np.sum(v)
                if max_i < no_i:
                    inls = v
                    F = aF
                    max_i = no_i
                    exp_sam = nsamples(max_i, len_u, 7, 0.95)
                    max_sam = min(max_sam, exp_sam)

    if no_sam == MAX_SAM:
        warnings.warn(f'RANSAC - termination forced after {no_sam} samples expected number of samples is {exp_sam}')

    return F, inls

# Example usage
# u = np.array(...)  # Define your u array here
# th = 0.1  # Define your threshold here
# F, inls = rEG(u, th)
# print(F, inls)
