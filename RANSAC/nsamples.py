import numpy as np


def nsamples(ni, ptNum, pf, conf=):
     """
    Computes the required number of random samples to achieve a desired confidence level
    in robust estimation.

    Parameters:
    -----------
    ni : (nt) The number of inliers in the data.
    ptNum : (int) The total number of points (both inliers and outliers).
    pf : (int) The number of points needed to estimate the model.
    conf : (float) The desired confidence level (typically between 0 and 1).

    Returns:
    --------
    SampleCnt : (float) The estimated number of random samples needed to achieve the given confidence level.

    Explanation:  
    ------------
    This function calculates the probability `q` that a randomly selected sample consists entirely of inliers,
    Based on `q`, the function determines the minimum number of samples required.
    This calculation is commonly used in RANSAC to determine how many iterations 
    are needed to find a valid model with high confidence.
    """

    q = np.prod([(ni - pf + 1 + i) / (ptNum - pf + 1 + i) for i in range(pf)])

    if (1 - q) < np.finfo(float).eps:
        SampleCnt = 1
    else:
        SampleCnt = np.log(1 - conf) / np.log(1 - q)

    if SampleCnt < 1:
        SampleCnt = 1

    return SampleCnt


# Example usage
"""
ni = 10  # Number of inliers
ptNum = 100  # Total number of points
pf = 8  # Number of points needed for the model
conf = 0.99  # Desired confidence level

sample_count = nsamples(ni, ptNum, pf, conf)
print(sample_count)
"""
