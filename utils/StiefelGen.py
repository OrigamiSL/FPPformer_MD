import os
import warnings

import numpy as np
from geomstats.geometry.stiefel import Stiefel, StiefelCanonicalMetric
import matplotlib.pyplot as plt


def StiefelT(U, perc):
    INJ_RADIUS = 0.89 * np.pi

    dim1, dim2 = U.shape
    st = Stiefel(dim1, dim2)
    print(dim1, dim2)
    st_metric = StiefelCanonicalMetric(dim1, dim2)

    tan_plane_vec = st.random_tangent_vec(U, 1)

    canonical_ip = st_metric.inner_product(tan_plane_vec, tan_plane_vec, U)
    # print('canonical_ip', canonical_ip)
    scaled_tan = tan_plane_vec / np.sqrt(canonical_ip) * perc * INJ_RADIUS

    St_mat = st_metric.exp(scaled_tan, U)

    return St_mat


def StiefelGen(X, perc=None):
    U, S, V = np.linalg.svd(X)
    S = np.diag(S)
    pad = np.zeros([U.shape[1] - S.shape[0], S.shape[1]])
    S = np.concatenate([S, pad], axis=0)

    if perc is None:
        perc = 0.5
    U1 = StiefelT(U, perc)
    V1 = StiefelT(V, perc)
    x_recon = np.dot(np.dot(U1, S), V1)
    # t_plot = [i for i in range(X.shape[1])]
    # for k in range(X.shape[0]):
    #     plt.figure(figsize=(24, 16))
    #     plt.plot(t_plot, x_recon[k, :], color='r')
    #     plt.plot(t_plot, X[k, :], color='b')
    #     plt.show()
    #     plt.close()
    error = (np.mean(np.abs(x_recon - X) / (np.abs(x_recon) + np.abs(X)), axis=1, keepdims=True).
             repeat(X.shape[1], axis=1))
    x_recon = np.where(error < 0.2, x_recon, X)

    return x_recon
