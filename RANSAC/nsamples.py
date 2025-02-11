import numpy as np


def nsamples(ni, ptNum, pf, conf):
    q = np.prod([(ni - pf + 1 + i) / (ptNum - pf + 1 + i) for i in range(pf)])

    if (1 - q) < np.finfo(float).eps:
        SampleCnt = 1
    else:
        SampleCnt = np.log(1 - conf) / np.log(1 - q)

    if SampleCnt < 1:
        SampleCnt = 1

    return SampleCnt


# Example usage
ni = 10  # Number of inliers
ptNum = 100  # Total number of points
pf = 8  # Number of points needed for the model
conf = 0.99  # Desired confidence level

sample_count = nsamples(ni, ptNum, pf, conf)
print(sample_count)
