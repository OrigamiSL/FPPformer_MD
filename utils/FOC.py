import os
import warnings

import torch
import numpy as np
import matplotlib.pyplot as plt


def phase_mix_2(phase_fft, inds):
    phase_difference = phase_fft - phase_fft[inds]
    dtheta = phase_difference % (2 * torch.pi)

    dtheta[dtheta > torch.pi] -= 2 * torch.pi
    clockwise = dtheta > 0
    sign = torch.where(clockwise, -1, 1)
    return dtheta, sign


def FOC(X):  # Apply proposed mixup but use random coeffs instead of similarity based
    sample = torch.tensor(X).unsqueeze(-1)
    with torch.no_grad():
        fftsamples = torch.fft.rfft(sample, dim=1, norm='ortho')
        inds = torch.randperm(sample.size(0))
        coeffs = torch.ones(sample.shape[0])
        coeffs = torch.nn.init.trunc_normal_(coeffs, 1, 0.1, 0.9, 1)

        abs_fft = torch.abs(fftsamples)
        phase_fft = torch.angle(fftsamples)
        mixed_abs = abs_fft * coeffs[:, None, None] + (1 - coeffs[:, None, None]) * abs_fft[inds]
        dtheta, sign = phase_mix_2(phase_fft, inds)
        mixed_phase = phase_fft + (1 - coeffs[:, None, None]) * torch.abs(dtheta) * sign
        z = torch.polar(mixed_abs, mixed_phase)
        mixed_samples_time = torch.fft.irfft(z, dim=1, norm='ortho')
        x_recon = mixed_samples_time.squeeze(-1).detach().cpu().numpy()

        # t_plot = [i for i in range(sample.shape[1])]
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
