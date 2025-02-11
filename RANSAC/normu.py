import numpy as np
import p2e

def normu(u):
    # If u is 3xN, convert it to 2xN by using p2e equivalent
    if u.shape[0] == 3:
        u = p2e(u)

    # Calculate the mean of each row
    m = np.mean(u, axis=1).reshape(-1, 1)

    # Subtract the mean from all points
    u = u - m @ np.ones((1, u.shape[1]))

    # Calculate the distances
    distu = np.sqrt(np.sum(u ** 2, axis=0))

    # Calculate the scaling factor
    r = np.mean(distu) / np.sqrt(2)

    # Create the normalization matrix
    A = np.diag([1 / r, 1 / r, 1])
    A[0:2, 2] = -m.flatten() / r

    return A

# Example usage
u = np.random.rand(3, 10)  # Example u matrix with 10 points in homogeneous coordinates
A = normu(u)
print(A)
