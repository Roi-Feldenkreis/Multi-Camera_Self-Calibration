import numpy as np

def p2e(u):
    e = u[:2, :] / u[2, :]
    return e

# Example usage
u = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
e = p2e(u)
print(e)
