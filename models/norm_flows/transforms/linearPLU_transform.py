import torch
import torch.nn.functional as F
from torch.distributions import Transform

from pyro.nn import DenseNN
from pyro.distributions import constraints

from pyro.distributions.conditional import ConditionalTransformModule
from pyro.distributions.torch_transform import TransformModule


class ConditionedLinearPLUTransform(Transform):
    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True
    event_dim = 1

    def __init__(self, permutation=None, LU=None):
        super(ConditionedLinearPLUTransform, self).__init__(cache_size=1)

        self.permutation = permutation
        self.LU = LU

    @property
    def U_diag(self):
        return self.LU.diagonal(dim1=-2, dim2=-1)  # (bs, input_dim)

    @property
    def L(self):
        return self.LU.tril(diagonal=-1) + torch.eye(self.LU.size(-1), dtype=self.LU.dtype, device=self.LU.device)  # (bs, input_dim, input_dim)

    @property
    def U(self):
        return self.LU.triu()  # (bs, input_dim, input_dim)

    def _call(self, x):
        """
        :param x: (*sample_shape, input_dim) input into the bijection
        :type x: torch.Tensor
        """
        # y = PLUx
        weight = (self.permutation @ self.L @ self.U)
        y = torch.matmul(weight, x[..., None])[..., 0]
        # print('\nIN CALL')
        # print('x max min', x[..., 0].max().item(), x[..., 0].min().item())
        # print('y max min', y[..., 0].max().item(), y[..., 0].min().item())
        # print('determinant', (self.permutation @ self.L @ self.U).det())
        # print('determinant inverse', (self.permutation @ self.L @ self.U).inverse().det())
        # print('2-norm', (self.permutation @ self.L @ self.U).norm(dim=(-1, -2), p=2))
        # print('2-norm inverse', (self.permutation @ self.L @ self.U).inverse().norm(p=2, dim=(-1, -2)))
        return y

    def _inverse(self, y):
        """
        Inverts y => x.
        :param y: (*sample_shape, input_dim) output of the bijection
        :type y: torch.Tensor
        """
        # PLUx = y
        # LUx = P^T y
        y = y[..., None]  # (*sample_shape, input_dim, 1)
        LUx = torch.matmul(self.permutation.T, y)  # (*sample_shape, input_dim, 1)

        # Solve L(Ux) = P^T y
        Ux, _ = torch.triangular_solve(LUx, self.L, upper=False)  # (*sample_shape, input_dim, 1)

        # Solve Ux = (PL)^-1y
        x, _ = torch.triangular_solve(Ux, self.U)  # (*sample_shape, input_dim, 1)
        # print('\nIN INVERSE')
        # print('y max min', y[..., 0].max().item(), y[..., 0].min().item())
        # print('x max min', x[..., 0].max().item(), x[..., 0].min().item())
        # print('determinant', (self.permutation @ self.L @ self.U).det())
        # print('determinant inverse', (self.permutation @ self.L @ self.U).inverse().det())
        # print('2-norm', (self.permutation @ self.L @ self.U).norm(dim=(-1, -2), p=2))
        # print('2-norm inverse', (self.permutation @ self.L @ self.U).inverse().norm(p=2, dim=(-1, -2)))
        return x[..., 0]  # (*sample_shape, input_dim)

    def log_abs_det_jacobian(self, x, y):
        """
        Calculates the elementwise determinant of the log Jacobian, i.e.
        log(abs(det(dy/dx))).
        """
        log_det = self.U_diag.abs().log().sum(dim=-1).expand(x.shape[:-1])
        return log_det


class LinearPLUTransform(ConditionedLinearPLUTransform, TransformModule):
    """
    Adapted from pyro's GeneralizedChannelPermute.
    """
    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True
    event_dim = 1

    def __init__(self, input_dim, permutation=None):
        super(LinearPLUTransform, self).__init__()
        self.__delattr__('permutation')
        self._cache_size = 0
        # Sample a random orthogonal matrix
        W, _ = torch.qr(torch.randn(input_dim, input_dim))
        # Construct the partially pivoted LU-form and the pivots
        LU, pivots = W.lu()

        # Convert the pivots into the permutation matrix
        if permutation is None:
            P, _, _ = torch.lu_unpack(LU, pivots)
        else:
            if len(permutation) != input_dim:
                raise ValueError(
                    'Keyword argument "permutation" expected to have {} elements but {} found.'.format(
                        input_dim, len(permutation)))
            P = torch.eye(input_dim, input_dim)[permutation.type(dtype=torch.int64)]

        # We register the permutation matrix so that the model can be serialized
        self.register_buffer('permutation', P)

        # NOTE: For this implementation I have chosen to store the parameters densely, rather than
        # storing L, U, and s separately
        self.LU = torch.nn.Parameter(LU)


class ConditionalLinearPLUTransform(ConditionalTransformModule):
    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True
    event_dim = 1

    def __init__(self, nn, input_dim, permutation=None):
        super().__init__()
        self.cond_plu_nn = nn
        self.input_dim = input_dim
        if permutation is None:
            permutation = torch.randperm(input_dim, device='cpu').to(torch.Tensor().device)
        P = torch.eye(len(permutation), len(permutation))[permutation.type(dtype=torch.int64)]
        self.register_buffer('permutation', P)

    def condition(self, context):
        LU = self.cond_plu_nn(context)  # (bs, input_dim * input_dim)
        LU = LU.view(LU.shape[:-1] + (self.input_dim, self.input_dim)) # (bs, input_dim, input_dim)
        # Note that LU matrix is NOT L @ U:
        # Upper triangular part of LU (including main diagonal) is U matrix
        # Lower triangular part of LU (below main diagonal) is L matrix (main diagonal of L matrix set to be ones)
        # Constrain diagonal of U (i.e. diagonal of LU) to be positive
        LU[..., range(self.input_dim), range(self.input_dim)] = F.softplus(LU[..., range(self.input_dim), range(self.input_dim)],
                                                                           beta=0.75)  # For faster convergence
        # If softplus beta is bigger, 2-norm of inverse PLU matrix is big (or equivalent 2-norm of PLU is small)
        # i.e. PLU is very contractive transformation
        # --> Causes very large -ve log_prob values
        return ConditionedLinearPLUTransform(self.permutation, LU)


def conditional_linear_plu_transform(input_dim,
                                     context_dim,
                                     hidden_dims=None,
                                     permutation=None):
    """
    A helper function to create a
    :class:`~pyro.distributions.transforms.ConditionalGeneralizedChannelPermute`
    object for consistency with other helpers.

    :param input_dim: Number of channel dimensions in the input.
    :type input_dim: int

    """

    if hidden_dims is None:
        hidden_dims = [input_dim * 10, input_dim * 10]
    nn = DenseNN(context_dim, hidden_dims, param_dims=[input_dim * input_dim])
    return ConditionalLinearPLUTransform(nn, input_dim, permutation=permutation)