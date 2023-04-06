from functools import partial

from torch.distributions import constraints

from pyro.distributions.conditional import ConditionalTransformModule
from pyro.distributions.transforms.spline_coupling import SplineCoupling
from pyro.nn import ConditionalDenseNN


class ConditionalSplineCoupling(ConditionalTransformModule):
    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True
    event_dim = 1

    def __init__(self,
                 input_dim,
                 split_dim,
                 hypernet,
                 count_bins=8,
                 bound=3.,
                 order='linear',
                 identity=False):
        super().__init__()
        self.input_dim = input_dim
        self.split_dim = split_dim

        self.nn = hypernet

        self.count_bins = count_bins
        self.bound = bound
        self.order = order
        self.identity = identity

    def condition(self, context):
        """
        Conditions on a context variable, returning a non-conditional transform of
        of type :class:`~pyro.distributions.transforms.SplineCoupling`.
        """
        cond_nn = partial(self.nn, context=context)

        return SplineCoupling(input_dim=self.input_dim,
                              split_dim=self.split_dim,
                              hypernet=cond_nn,
                              count_bins=self.count_bins,
                              bound=self.bound,
                              order=self.order,
                              identity=self.identity)


def conditional_spline_coupling(input_dim,
                                context_dim,
                                hidden_dims=None,
                                split_dim=None,
                                count_bins=8,
                                bound=3.,
                                order='linear'):
    if split_dim is None:
        split_dim = input_dim // 2

    if hidden_dims is None:
        hidden_dims = [input_dim * 10, input_dim * 10]

    nn = ConditionalDenseNN(split_dim,
                            context_dim,
                            hidden_dims,
                            param_dims=[(input_dim - split_dim) * count_bins,
                                        (input_dim - split_dim) * count_bins,
                                        (input_dim - split_dim) * (count_bins - 1),
                                        (input_dim - split_dim) * count_bins])

    return ConditionalSplineCoupling(input_dim,
                                     split_dim,
                                     nn,
                                     count_bins=count_bins,
                                     bound=bound,
                                     order=order,
                                     identity=True)

