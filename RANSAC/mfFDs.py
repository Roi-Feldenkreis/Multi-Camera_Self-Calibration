import numpy as np


def mfFDs(F, u):
    # MATLAB: Fu1 = F*u(4:6,:)
    Fu1 = F @ u[3:6, :]

    # MATLAB: Fu2 = (F'*u(1:3,:)).^2
    Fu2 = (F.T @ u[0:3, :]) ** 2

    # MATLAB: Fu1pow = Fu1.^2
    Fu1pow = Fu1 ** 2

    # MATLAB: denom = Fu1pow(1,:)+Fu1pow(2,:)+Fu2(1,:)+Fu2(2,:)
    denom = Fu1pow[0, :] + Fu1pow[1, :] + Fu2[0, :] + Fu2[1, :]

    # MATLAB: errvec = zeros(1,size(u,2))
    errvec = np.zeros(u.shape[1])

    for i in range(u.shape[1]):
        # MATLAB: xFx = u(1:3,i)'*Fu1(:,i)
        xFx = u[0:3, i].T @ Fu1[:, i]

        # MATLAB: errvec(i) = xFx^2/denom(i)
        errvec[i] = xFx ** 2 / denom[i]

    err = errvec
    return err


# Example usage
F = np.random.rand(3, 3)  # Example F matrix
u = np.random.rand(6, 10)  # Example u matrix with 10 correspondences
err = mfFDs(F, u)
print(err)
