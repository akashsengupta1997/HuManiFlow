from itertools import cycle
import torch

import pyro.distributions.transforms as trans
from pyro.distributions import (
    Normal,
    Independent,
    ConditionalTransformedDistribution,
    ConditionalTransform
)

from models.norm_flows.transforms import (
    conditional_spline_coupling,
    conditional_linear_plu_transform,
    conditional_additive_coupling,
    LinearPLUTransform,
    ScaledRadialTanhTransform
)


def create_conditional_norm_flow(device,
                                 event_dim,
                                 context_dim,
                                 num_transforms,
                                 transform_type,
                                 transform_hidden_dims,
                                 permute_type=None,
                                 permute_hidden_dims=None,
                                 batch_norm=False,
                                 radial_tanh_radius=None,
                                 base_dist_std=1.,
                                 **kwargs):
    """
    """
    assert transform_type in ['affine_coupling', 'affine_masked',
                              'additive_coupling',
                              'spline_coupling', 'spline_masked']

    if permute_type is not None:
        assert permute_type in ['permute', 'linear_plu', 'conditional_linear_plu']
        if permute_type == 'conditional_linear_plu':
            assert permute_hidden_dims is not None
        if permute_type != 'permute' and not batch_norm:
            print('Warning: should use batch_norm with linear PLU transformations!')

        idx = list(range(event_dim))
        perm_cycle = cycle([torch.tensor(idx[i:] + idx[:i], dtype=torch.long, device=device) for i in range(event_dim)])

    base_dist = Independent(Normal(loc=torch.zeros(event_dim, device=device),
                                   scale=torch.ones(event_dim, device=device) * base_dist_std,
                                   validate_args=False),
                            reinterpreted_batch_ndims=1)

    transforms = []
    for i in range(num_transforms):

        # Permutation layer - i.e. linear transformation with permutation matrix P.
        # Permutations may be "generalised" - i.e. linear transformations with matrix W = PLU
        if permute_type is not None:
            permutation_tensor = next(perm_cycle)
            if permute_type == 'permute':
                permute_transform = trans.Permute(permutation=permutation_tensor)
            elif permute_type == 'linear_plu':
                permute_transform = LinearPLUTransform(input_dim=event_dim,
                                                       permutation=permutation_tensor).to(device)
            elif permute_type == 'conditional_linear_plu':
                permute_transform = conditional_linear_plu_transform(input_dim=event_dim,
                                                                     context_dim=context_dim,
                                                                     hidden_dims=permute_hidden_dims,
                                                                     permutation=permutation_tensor).to(device)
            transforms.append(permute_transform)

        # Batch normalisation layer
        # Note that batch_norm is needed during training, when we use INVERSE transforms to compute log_prob.
        # Thus batch_norm's _call is actually inverse batch_norm and batch_norm's _inverse is actually the usual forward batch_norm.
        # _inverse is only needed if we are sampling during training.
        if batch_norm:
            transforms.append(trans.BatchNorm(input_dim=event_dim).to(device))

        # Conditional bijective transformation layer
        # Transformation is autoregressive - i.e. triangular Jacobian
        if transform_type == 'affine_coupling':
            transform = trans.conditional_affine_coupling(input_dim=event_dim,
                                                          context_dim=context_dim,
                                                          hidden_dims=transform_hidden_dims,
                                                          **kwargs)
        elif transform_type == 'affine_masked':
            transform = trans.conditional_affine_autoregressive(input_dim=event_dim,
                                                                context_dim=context_dim,
                                                                hidden_dims=transform_hidden_dims,
                                                                **kwargs)
        elif transform_type == 'additive_coupling':
            transform = conditional_additive_coupling(input_dim=event_dim,
                                                      context_dim=context_dim,
                                                      hidden_dims=transform_hidden_dims,
                                                      **kwargs)
        elif transform_type == 'spline_coupling':
            transform = conditional_spline_coupling(input_dim=event_dim,
                                                    context_dim=context_dim,
                                                    hidden_dims=transform_hidden_dims,
                                                    **kwargs)
        elif transform_type == 'spline_masked':
            transform = trans.conditional_spline_autoregressive(input_dim=event_dim,
                                                                context_dim=context_dim,
                                                                hidden_dims=transform_hidden_dims,
                                                                **kwargs)
        transforms.append(transform.to(device))

    if radial_tanh_radius is not None:
        transforms.append(ScaledRadialTanhTransform(radius=radial_tanh_radius))

    conditional_flow_dist = ConditionalTransformedDistribution(base_dist=base_dist,
                                                               transforms=transforms)

    transform_modules = torch.nn.ModuleList([t for t in transforms if isinstance(t, torch.nn.Module)])

    return conditional_flow_dist, transform_modules, transforms


def forward_trans_conditional_norm_flow(transforms,
                                        base_sample,
                                        context):
    for t in transforms:
        if isinstance(t, ConditionalTransform):
            base_sample = t.condition(context)(base_sample)
        else:
            base_sample = t(base_sample)

    return base_sample

