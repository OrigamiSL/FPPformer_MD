import numpy as np
import matplotlib.pyplot as plt


def DMD_reconstruct(X, duration, dt):  # V, L
    X1 = X[:, :duration]
    X2 = X[:, 1:]
    U, S, V = np.linalg.svd(X1)
    S = np.diag(S)
    V = np.transpose(V).conjugate()
    Ur = U[:, :duration]
    Sr = S[:duration, :duration]
    Vr = V[:, :duration]
    a1 = np.dot(np.dot(np.transpose(Ur).conjugate(), X2), Vr)
    inv_Sr = np.linalg.inv(Sr)
    Atilde = np.dot(a1, inv_Sr)

    D, W = np.linalg.eig(Atilde)
    omega = np.log(D) / dt

    a2 = np.dot(X2, Vr)
    Phi = np.dot(np.dot(a2, inv_Sr), W)

    b = np.dot(np.linalg.pinv(Phi), X1[:, 0])
    tx = np.linspace(0, duration - 1, duration)
    b_R = b.reshape(duration, 1).repeat(duration, axis=1)
    omega_R = omega.reshape(duration, 1).repeat(duration, axis=1)
    tx_R = np.array(tx.reshape(1, duration)).repeat(duration, axis=0)

    x_recon = b_R * np.exp(omega_R * tx_R)
    x_recon = np.dot(Phi, x_recon).real

    # t_plot = [i for i in range(duration)]
    # for k in range(X.shape[0]):
    #     plt.figure(figsize=(24, 16))
    #     plt.plot(t_plot, x_recon[k, :], color='r')
    #     plt.plot(t_plot, X1[k, :], color='b')
    #     plt.show()
    #     plt.close()
    error = (np.mean(np.abs(x_recon - X1) / (np.abs(x_recon) + np.abs(X1)), axis=1, keepdims=True).
             repeat(duration, axis=1))
    x_recon = np.where(error < 0.2, x_recon, X1)

    return x_recon
