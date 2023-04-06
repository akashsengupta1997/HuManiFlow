"""
Code adapted from https://github.com/pimdh/relie.
"""
import torch

from torch.distributions import Transform


class ToTransform(Transform):
    """
    Transform dtype or device.
    """

    event_dim = 0
    sign = 1

    def __init__(self, options_in, options_out):
        super().__init__(1)
        self.options_in = options_in
        self.options_out = options_out

    def _call(self, x):
        return x.to(**self.options_out)

    def _inverse(self, y):
        return y.to(**self.options_in)

    def log_abs_det_jacobian(self, x, y):
        return torch.zeros_like(x).float()
