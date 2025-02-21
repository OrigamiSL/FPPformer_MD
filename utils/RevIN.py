import torch
import torch.nn as nn


class RevIN(nn.Module):
    def __init__(self, eps=1e-5):
        """
        :param eps: a value added for numerical stability
        """
        super(RevIN, self).__init__()
        self.eps = eps

    def forward(self, x, mode: str):
        if mode == 'stats':
            self._get_statistics(x)
        elif mode == 'norm':
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _get_statistics(self, x):
        B, L, V = x.shape
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean[..., :x.shape[-1]]
        x = x / self.stdev[..., :x.shape[-1]]
        return x

    def _denormalize(self, x):
        x = x * self.stdev[..., :x.shape[-1]]
        x = x + self.mean[..., :x.shape[-1]]
        return x
