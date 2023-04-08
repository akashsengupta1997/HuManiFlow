import math
import torch
from torch.distributions import constraints, Transform

from models.norm_flows.transforms import LocalDiffeoTransform
from utils.rigid_transform_utils import (
    so3_exp,
    so3_log,
    so3_vee,
    so3_xset,
    so3_log_abs_det_jacobian,
)


class SO3ExpCompactTransform(LocalDiffeoTransform):
    """Assumes underlying distribution has compact support only in the <2pi open ball."""

    domain = constraints.real_vector
    codomain = constraints.real_vector

    event_dim = 1

    def __init__(self, support_radius=2 * math.pi):
        """
        :param support_radius: radius of inverse set.
        """
        super().__init__()
        self.support_radius = support_radius

    def _call(self, x):
        return so3_exp(x)

    def _inverse_set(self, y):
        return self._xset(so3_vee(so3_log(y)))

    def _xset(self, x):
        xset = so3_xset(x, 1)
        norms = xset.norm(dim=-1)
        mask = norms < self.support_radius  # Needed if support_radius < 2pi, since ||x|| +/- 2pi may be out of support.
        xset.masked_fill_(torch.logical_not(mask[..., None]), 0)
        return x, xset, mask

    def log_abs_det_jacobian(self, x, y):
        """
        Log abs det of forward jacobian of exp map.
        :param x: Algebra element shape (..., 3)
        :param y: Group element (..., 3, 3)
        :return: Jacobian of exp shape (...)
        """
        return so3_log_abs_det_jacobian(x).float()
