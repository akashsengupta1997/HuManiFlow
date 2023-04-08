import math
import torch
from torch.distributions import Transform, constraints


class RadialTanhTransform(Transform):
    r"""
    Code from https://github.com/pimdh/relie.

    RadialTanhTransform used to restrict support of distribution on so(3) ~= R^3
    to ball of given radius.

    y = (x / norm(x)) * radius * tanh(norm(x)).
    i.e. if norm(x) is large, x gets mapped to point on ball surface with given radius.
         if norm(x) is small, x = radius * x

    i.e. "Transform R^d of radius (0, inf) to (0, R)"

    Uses the fact that tanh is linear near 0.
    """
    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True
    event_dim = 1

    def __init__(self, radius):
        super().__init__(cache_size=1)
        self.radius = radius

    def _call(self, x):
        x_norm = x.norm(dim=-1, keepdim=True)
        mask = x_norm > 1e-8
        x_norm = torch.where(mask, x_norm, torch.ones_like(x_norm))

        return torch.where(
            mask, torch.tanh(x_norm) * x / x_norm * self.radius, x * self.radius  # tanh(z) = z for small z
        )

    def _inverse(self, y):
        org_dtype = y.dtype
        y = y.double()
        y_norm = y.norm(dim=-1, keepdim=True)
        mask = y_norm > 1e-8
        y_norm = torch.where(mask, y_norm, torch.ones_like(y_norm))

        return torch.where(
            mask, torch.atanh(y_norm / self.radius) * y / y_norm, y / self.radius
        ).to(org_dtype)

    # def log_abs_det_jacobian(self, x, y):  Original code from ReLie - may be an error here**
    #     """
    #     Uses d tanh /dx = 1-tanh^2
    #     :param x: Tensor
    #     :param y: Tensor
    #     :return: Tensor
    #     """
    #     y_norm = y.norm(dim=-1)
    #     d = y.shape[-1]
    #     tanh = y_norm / self.radius
    #     log_dtanh = torch.log1p(-tanh ** 2)
    #
    #     log_radius = torch.full_like(log_dtanh, math.log(self.radius))
    #     return d * torch.where(y_norm > 1e-8, log_dtanh + log_radius, log_radius)  # **shouldn't there be a log(y_norm / x_norm) term here?

    def log_abs_det_jacobian(self, x, y):
        """
        Uses d tanh /dx = 1-tanh^2
        :param x: Tensor
        :param y: Tensor
        :return: Tensor
        """
        x_norm = x.norm(dim=-1)
        y_norm = y.norm(dim=-1)
        tanh = y_norm / self.radius

        log_dtanh = torch.log1p(-tanh ** 2)
        log_radius = torch.full_like(log_dtanh, math.log(self.radius))
        log_norm_ratio_squared = 2 * (torch.log(y_norm) - torch.log(x_norm))

        return torch.where(y_norm > 1e-8, log_dtanh + log_radius + log_norm_ratio_squared, log_radius)

