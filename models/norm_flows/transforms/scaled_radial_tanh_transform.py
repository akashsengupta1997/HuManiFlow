import torch
from torch.distributions import Transform, constraints


class ScaledRadialTanhTransform(Transform):
    """
    RadialTanhTransform used to restrict support of distribution on so(3) ~= R^3
    to ball of given radius.

    y = (x / norm(x)) * radius * tanh(norm(x) / radius).
    i.e. if norm(x) is large, x gets mapped to point near ball surface with given radius.
         if norm(x) is small, x = radius * x / radius = x

    i.e. "Transform R^d of radius (0, inf) to (0, R)"

    Uses the fact that tanh is linear near 0.
    """
    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True
    event_dim = 1

    def __init__(self,
                 radius):
        super().__init__(cache_size=1)
        self.radius = radius

    def _call(self, x):
        x_norm = x.norm(dim=-1, keepdim=True)
        mask = x_norm > 1e-7
        x_norm = torch.where(mask, x_norm, torch.ones_like(x_norm))

        return torch.where(
            mask, torch.tanh(x_norm / self.radius) * (x / x_norm) * self.radius, x  # tanh(z) = z for small z
        )

    def _inverse(self, y):
        org_dtype = y.dtype
        y = y.double()
        y_norm = y.norm(dim=-1, keepdim=True)
        mask = y_norm > 1e-7
        y_norm = torch.where(mask, y_norm, torch.ones_like(y_norm))

        return torch.where(
            mask, torch.atanh(y_norm / self.radius) * (y / y_norm) * self.radius, y
        ).to(org_dtype)

    def log_abs_det_jacobian(self, x, y):
        """
        :param x: Tensor
        :param y: Tensor
        :return: Tensor
        """
        x_norm = x.norm(dim=-1)
        y_norm = y.norm(dim=-1)

        log_det = 2 * (torch.log(y_norm) - torch.log(x_norm)) + torch.log1p(-((y_norm / self.radius)**2))

        return torch.where(y_norm > 1e-7, log_det, torch.zeros_like(log_det))
