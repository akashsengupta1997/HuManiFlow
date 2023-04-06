import operator
from functools import partial, reduce

import torch
from torch.distributions import constraints

from pyro.distributions.conditional import ConditionalTransformModule
from pyro.distributions.torch_transform import TransformModule
from pyro.nn import ConditionalDenseNN, DenseNN


class AdditiveCoupling(TransformModule):

    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True

    def __init__(self, split_dim, hypernet, *, dim=-1):
        super().__init__(cache_size=1)
        if dim >= 0:
            raise ValueError("'dim' keyword argument must be negative")

        self.split_dim = split_dim
        self.nn = hypernet
        self.dim = dim
        self.event_dim = -dim

    def _call(self, x):
        """
        :param x: the input into the bijection
        :type x: torch.Tensor

        Invokes the bijection x=>y; in the prototypical context of a
        :class:`~pyro.distributions.TransformedDistribution` `x` is a sample from
        the base distribution (or the output of a previous transform)
        """
        x1, x2 = x.split([self.split_dim, x.size(self.dim) - self.split_dim], dim=self.dim)

        # Now that we can split on an arbitrary dimension, we have do a bit of reshaping...
        mean = self.nn(x1.reshape(x1.shape[:-self.event_dim] + (-1,)))
        mean = mean.reshape(mean.shape[:-1] + x2.shape[-self.event_dim:])

        y1 = x1
        y2 = x2 + mean
        return torch.cat([y1, y2], dim=self.dim)

    def _inverse(self, y):
        """
        :param y: the output of the bijection
        :type y: torch.Tensor
        """
        y1, y2 = y.split([self.split_dim, y.size(self.dim) - self.split_dim], dim=self.dim)
        x1 = y1

        # Now that we can split on an arbitrary dimension, we have do a bit of reshaping...
        mean = self.nn(x1.reshape(x1.shape[:-self.event_dim] + (-1,)))
        mean = mean.reshape(mean.shape[:-1] + y2.shape[-self.event_dim:])

        x2 = (y2 - mean)
        return torch.cat([x1, x2], dim=self.dim)

    def log_abs_det_jacobian(self, x, y):
        """
        Calculates the elementwise determinant of the log jacobian
        """
        return torch.zeros(x.shape[:-1], device=x.device, dtype=x.dtype, requires_grad=True)


class ConditionalAdditiveCoupling(ConditionalTransformModule):

    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True
    event_dim = 1

    def __init__(self, split_dim, hypernet, **kwargs):
        super().__init__()
        self.split_dim = split_dim
        self.nn = hypernet
        self.kwargs = kwargs

    def condition(self, context):
        cond_nn = partial(self.nn, context=context)
        return AdditiveCoupling(self.split_dim, cond_nn, **self.kwargs)


def additive_coupling(input_dim, hidden_dims=None, split_dim=None, dim=-1, **kwargs):

    if not isinstance(input_dim, int):
        if len(input_dim) != -dim:
            raise ValueError('event shape {} must have same length as event_dim {}'.format(input_dim, -dim))
        event_shape = input_dim
        extra_dims = reduce(operator.mul, event_shape[(dim + 1):], 1)
    else:
        event_shape = [input_dim]
        extra_dims = 1
    event_shape = list(event_shape)

    if split_dim is None:
        split_dim = event_shape[dim] // 2
    if hidden_dims is None:
        hidden_dims = [10 * event_shape[dim] * extra_dims]

    hypernet = DenseNN(split_dim * extra_dims,
                       hidden_dims,
                       [(event_shape[dim] - split_dim) * extra_dims])
    return AdditiveCoupling(split_dim, hypernet, dim=dim, **kwargs)


def conditional_additive_coupling(input_dim, context_dim, hidden_dims=None, split_dim=None, dim=-1, **kwargs):
    if not isinstance(input_dim, int):
        if len(input_dim) != -dim:
            raise ValueError('event shape {} must have same length as event_dim {}'.format(input_dim, -dim))
        event_shape = input_dim
        extra_dims = reduce(operator.mul, event_shape[(dim + 1):], 1)
    else:
        event_shape = [input_dim]
        extra_dims = 1
    event_shape = list(event_shape)

    if split_dim is None:
        split_dim = event_shape[dim] // 2
    if hidden_dims is None:
        hidden_dims = [10 * event_shape[dim] * extra_dims]

    nn = ConditionalDenseNN(split_dim * extra_dims, context_dim, hidden_dims,
                            [(event_shape[dim] - split_dim) * extra_dims])
    return ConditionalAdditiveCoupling(split_dim, nn, dim=dim, **kwargs)
