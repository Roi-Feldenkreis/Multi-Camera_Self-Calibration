import numpy as np

def isptnorm(x):
    """
    Isotropic Point Normalization.

    Parameters:
    x (np.ndarray): [N x dim] coordinates.

    Returns:
    xnorm (np.ndarray): Normalized coordinates.
    T (np.ndarray): Transformation matrix used.
    """

    # Data dimension
    dim = x.shape[1]
    N = x.shape[0]

    # Make homogeneous coordinates
    x = np.hstack((x, np.ones((N, 1))))

    # Compute sum of square differences
    ssd = np.zeros((N, dim))
    for i in range(dim):
        ssd[:, i] = (x[:, i] - np.mean(x[:, i])) ** 2

    scale = (np.sqrt(dim) * N) / (np.sum(np.sqrt(np.sum(ssd, axis=1))))

    T = np.zeros((dim + 1, dim + 1))
    for i in range(dim):
        T[i, i] = scale
        T[i, dim + 1 - 1] = -scale * np.mean(x[:, i])
    T[dim, dim] = 1

    xnorm = (T @ x.T).T

    # Return non-homogeneous part of the points coordinates
    xnorm = xnorm[:, :dim]

    return xnorm, T

# Example usage
# x = np.array([[...], [...], ...])  # Define your input array
# xnorm, T = isptnorm(x)
