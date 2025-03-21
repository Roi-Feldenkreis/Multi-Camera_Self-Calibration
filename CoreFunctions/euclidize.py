import numpy as np
from scipy.linalg import svd
from CoreFunctions.RQ_decomposition import rq


def euclidize(Ws, Lambda, P, X, config):
    """
    Perform Euclidean reconstruction under the assumption of unknown focal lengths,
    constant principal points = 0, and aspect ratio = 1.

    Parameters:
    Ws (ndarray): 3*nxm measurement matrix
    Lambda (ndarray): nxm matrix containing the projective depths
    P (ndarray): 3*nx4 projective motion matrix
    X (ndarray): 4xm projective shape matrix
    config (dict): Configuration data with `cal.pp` and `cal.SQUARE_PIX` expected

    Returns:
    Pe (ndarray): 3*nx4 Euclidean motion matrix
    Xe (ndarray): 4xm Euclidean shape matrix
    C (ndarray): 4xn matrix containing the camera centers
    Rot (ndarray): 3*nx3 matrix containing the camera rotation matrices
    """
    n = Ws.shape[0] // 3  # number of cameras
    m = Ws.shape[1]  # number of points

    # Compute B
    a = np.sum(Ws[::3, :] * Lambda, axis=1)
    b = np.sum(Ws[1::3, :] * Lambda, axis=1)
    c = np.sum(Lambda, axis=1)

    TempA = -P[2::3, :] * a[:, None] / c[:, None] + P[0::3, :]
    TempB = -P[2::3, :] * b[:, None] / c[:, None] + P[1::3, :]

    Temp = np.vstack((TempA, TempB))
    U, S, Vt = svd(Temp, full_matrices=False)
    V = Vt.T
    B = V[:, 3]

    # Compute A

    # M * M ^ T == P * Q * P ^ T, thus
    #
    # (m_x)(P1)
    # (m_y) * (m_x m_y m_z) == (P2) * Q * (P1 P2 P3)(let Pi denote the i - th row of P), thus
    # (m_z)(P3)
    #
    # (| m_x |^2    m_x * m_y   m_x * m_z )      (P1 * Q * P1 ^ T  P1 * Q * P2 ^ T  P1 * Q * P3 ^ T )
    # (.           | m_y |^2    m_y * m_z )  ==  ( .               P2 * Q * P2 ^ T  P2 * Q * P3 ^ T )
    # (........                | m_z |^ 2 )      ( ..........                       P3 * Q * P3 ^ T )


    Temp = []
    for i in range(n):
        P1 = P[3 * i, :]        # 1st row of i-th camera
        P2 = P[3 * i + 1, :]    # 2nd row of i-th camera
        P3 = P[3 * i + 2, :]    # 3rd row of i-th camera
        u, v = P1, P2

        Temp.append([u[0] * v[0], u[0] * v[1] + u[1] * v[0], u[2] * v[0] + u[0] * v[2], u[0] * v[3] + u[3] * v[0],
                     u[1] * v[1], u[1] * v[2] + u[2] * v[1], u[1] * v[3] + u[3] * v[1], u[2] * v[2],
                     u[2] * v[3] + u[3] * v[2], u[3] * v[3]])

        if config['cal']['SQUARE_PIX']:
            Temp.append([u[0] ** 2 - v[0] ** 2, 2 * (u[0] * u[1] - v[0] * v[1]), 2 * (u[0] * u[2] - v[0] * v[2]),
                         2 * (u[0] * u[3] - v[0] * v[3]), u[1] ** 2 - v[1] ** 2, 2 * (u[1] * u[2] - v[1] * v[2]),
                         2 * (u[1] * u[3] - v[1] * v[3]), u[2] ** 2 - v[2] ** 2, 2 * (u[2] * u[3] - v[2] * v[3]),
                         u[3] ** 2 - v[3] ** 2])

        u, v = P1, P3
        Temp.append([u[0] * v[0], u[0] * v[1] + u[1] * v[0], u[2] * v[0] + u[0] * v[2], u[0] * v[3] + u[3] * v[0],
                     u[1] * v[1], u[1] * v[2] + u[2] * v[1], u[1] * v[3] + u[3] * v[1], u[2] * v[2],
                     u[2] * v[3] + u[3] * v[2], u[3] * v[3]])

        u, v = P2, P3
        Temp.append([u[0] * v[0], u[0] * v[1] + u[1] * v[0], u[2] * v[0] + u[0] * v[2], u[0] * v[3] + u[3] * v[0],
                     u[1] * v[1], u[1] * v[2] + u[2] * v[1], u[1] * v[3] + u[3] * v[1], u[2] * v[2],
                     u[2] * v[3] + u[3] * v[2], u[3] * v[3]])
    # one additional equation only if needed
    if n < 4 and not config['cal']['SQUARE_PIX']:
        u = P[2, :]
        Temp.append(
            [u[0] ** 2, 2 * u[0] * u[1], 2 * u[0] * u[2], 2 * u[0] * u[3], u[1] ** 2, 2 * u[1] * u[2], 2 * u[1] * u[3],
             u[2] ** 2, 2 * u[2] * u[3], u[3] ** 2])
        b = np.zeros(len(Temp))
        b[-1] = 1
        # TLS solution of Temp*q=b
        U, S, Vt = svd(np.column_stack((Temp, b)), full_matrices=False)
        V = Vt.T
        q = -V[:-1, -1] / V[-1, -1]
    else:
        U, S, Vt = svd(Temp, full_matrices=False)
        q = -Vt.T[:, -1]

    Q = np.array([
        [q[0], q[1], q[2], q[3]],
        [q[1], q[4], q[5], q[6]],
        [q[2], q[5], q[7], q[8]],
        [q[3], q[6], q[8], q[9]]
    ])

    # Test which solution to take for q (-V or V)
    M_M = P[0:3, :] @ Q @ P[0:3, :].T
    if M_M[0, 0] <= 0:
        q = -q
        Q = np.array([
            [q[0], q[1], q[2], q[3]],
            [q[1], q[4], q[5], q[6]],
            [q[2], q[5], q[7], q[8]],
            [q[3], q[6], q[8], q[9]]
        ])

    U, S, Vt = svd(Q, full_matrices=False)
    A = U[:, :3] @ np.sqrt(np.diag(S[:3]))

    H = np.column_stack((A, B))

    # Euclidean motion and shape
    Pe = P @ H
    Xe = np.linalg.inv(H) @ X

    # Normalize coordinates
    Xe /= Xe[3, :]

    PeRT = []
    Rot = []
    for i in range(n):
        sc = np.linalg.norm(Pe[3 * i + 2, :3], 'fro')
        Pe[3 * i:3 * i + 3, :] /= sc

        xe = Pe[3 * i:3 * i + 3, :] @ Xe
        if np.sum(xe[2, :] < 0):
            Pe[3 * i:3 * i + 3, :] = -Pe[3 * i:3 * i + 3, :]

        K, R = rq(Pe[3 * i:3 * i + 3, :3])
        Cc = -R.T @ np.linalg.inv(K) @ Pe[3 * i:3 * i + 3, 3]

        K[0, 2] -= config['cal']['pp'][i, 0]
        K[1, 2] -= config['cal']['pp'][i, 1]
        PeRT.append(K @ np.hstack((R, -R @ Cc[:, None])))
        Rot.append(R)

    Pe = np.vstack(PeRT)
    C = np.array([Cc for _ in range(n)]).T
    Rot = np.vstack(Rot)

    return Pe, Xe, C, Rot
