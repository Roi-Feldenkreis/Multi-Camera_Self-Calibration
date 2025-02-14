import numpy as np
import p2e
import seig

def u2F(u, str=''):
    """
        This function estimates fundamental matrix using ortogonal LS regression
        u2F(u) estimates F from u using NORMU
        u2F(u,'nonorm') disables normalization
    """
    if str == 'nonorm':
        donorm = False
    else:
        donorm = True

    ptNum = u.shape[1]

    if donorm:
        A1 = normu(u[:3, :])
        A2 = normu(u[3:6, :])

        u1 = A1 @ u[:3, :]
        u2 = A2 @ u[3:6, :]
    else:
        u1 = u[:3, :]
        u2 = u[3:6, :]

    Z = np.zeros((ptNum, 9))
    for i in range(ptNum):
        Z[i, :] = np.reshape(np.outer(u1[:, i], u2[:, i]), 9)

    M = Z.T @ Z
    V = seig(M)
    F = np.reshape(V[:, 0], (3, 3))

    uu, us, uv = np.linalg.svd(F)
    us[np.argmin(np.abs(np.diag(us)))] = 0
    F = uu @ np.diag(us) @ uv

    if donorm:
        F = A1.T @ F @ A2

    F = F / np.linalg.norm(F, 2)
    return F

def normu(u):
    if u.shape[0] == 3:
        u = p2e(u)

    m = np.mean(u, axis=1)
    u = u - m[:, np.newaxis]
    distu = np.sqrt(np.sum(u**2, axis=0))
    r = np.mean(distu) / np.sqrt(2)
    A = np.diag([1/r, 1/r, 1])
    A[:2, 2] = -m / r

    return A

# Example usage
# u = np.array(...)  # Define your u array here
# F = u2F(u)
# print(F)
