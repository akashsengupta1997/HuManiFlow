import torch
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.transforms import Transform
from torch.distributions.utils import _sum_rightmost

from pyro.distributions.conditional import ConditionalDistribution, ConstantConditionalDistribution

from models.norm_flows.transforms import LocalDiffeoTransform


class LocalDiffeoTransformedDistribution(Distribution):
    """
    Code from https://github.com/pimdh/relie.

    Version of TransformedDistribution that allows for non-injective maps
    with a finite (countable) inverse set/pre-image.
    """
    arg_constraints = {}

    def __init__(self, base_distribution, transforms, validate_args=None):
        self.base_dist = base_distribution
        if isinstance(transforms, Transform) or isinstance(
            transforms, LocalDiffeoTransform
        ):
            self.transforms = [transforms]
        elif isinstance(transforms, list):
            if not all(
                isinstance(t, Transform) or isinstance(t, LocalDiffeoTransform)
                for t in transforms
            ):
                raise ValueError(
                    "transforms must be a Transform or a list of Transforms"
                )
            self.transforms = transforms
        else:
            raise ValueError(
                "transforms must be a Transform or list, but was {}".format(transforms)
            )
        # TODO: Accommodate changes in shape
        shape = self.base_dist.batch_shape + self.base_dist.event_shape
        event_dim = max(
            [len(self.base_dist.event_shape)] + [t.event_dim for t in self.transforms]
        )
        batch_shape = shape[: len(shape) - event_dim]
        event_shape = shape[len(shape) - event_dim :]
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    @constraints.dependent_property
    def support(self):
        return (
            self.transforms[-1].codomain if self.transforms else self.base_dist.support
        )

    @property
    def has_rsample(self):
        return self.base_dist.has_rsample

    def sample(self, sample_shape=torch.Size()):
        """
        Generates a sample_shape shaped sample or sample_shape shaped batch of
        samples if the distribution parameters are batched. Samples first from
        base distribution and applies `transform()` for every transform in the
        list.
        """
        with torch.no_grad():
            x = self.base_dist.sample(sample_shape)
            for transform in self.transforms:
                x = transform(x)
            return x

    def rsample(self, sample_shape=torch.Size()):
        """
        Generates a sample_shape shaped reparameterized sample or sample_shape
        shaped batch of reparameterized samples if the distribution parameters
        are batched. Samples first from base distribution and applies
        `transform()` for every transform in the list.
        """
        x = self.base_dist.rsample(sample_shape)
        for transform in self.transforms:
            x = transform(x)
        return x

    def log_prob(self, value):
        """
        Scores the sample by inverting the transform(s) and computing the score
        using the score of the base distribution and the log abs det jacobian.
        """
        return self._log_prob(value, reversed(self.transforms))

    def _log_prob(self, y, transforms):
        # TODO: fix dtypes
        event_dim = len(self.event_shape)
        assert torch.isnan(y).sum() == 0, f'y Shape: {y.shape}, Num NaNs: {torch.isnan(y).sum()}'
        assert (y.abs() == float("inf")).any() == 0
        if not transforms:
            log_prob = _sum_rightmost(
                self.base_dist.log_prob(y), event_dim - len(self.base_dist.event_shape)
            ).float()
            assert torch.isnan(log_prob).sum() == 0, f'log_prob Shape: {log_prob.shape}, Num NaNs: {torch.isnan(y).sum()}'
            # nan_mask = torch.isnan(log_prob)
            # if nan_mask.sum() != 0:
            #     print(f'{nan_mask.sum()} NaNs found in so3_dist.log_prob!')
            #     log_prob[nan_mask] = 0.
            return log_prob

        transform, *transforms = transforms

        if isinstance(transform, Transform):
            x = transform.inv(y)
            log_prob = -_sum_rightmost(
                transform.log_abs_det_jacobian(x, y), event_dim - transform.event_dim
            )
            next_log_prob = self._log_prob(x, transforms)
            assert torch.isnan(log_prob).sum() == 0
            assert torch.isnan(next_log_prob).sum() == 0
            sum_log_prob = log_prob.float() + next_log_prob.float()
            assert torch.isnan(sum_log_prob).sum() == 0
            return sum_log_prob
        else:
            x, xset, mask = transform.inverse_set(y)
            # First propate back x to use caching
            x_log_prob = -_sum_rightmost(
                transform.log_abs_det_jacobian(x, y), event_dim - transform.event_dim
            )
            x_next_log_prob = self._log_prob(x, transforms)
            x_term = x_log_prob.float() + x_next_log_prob.float()

            # Now propagate others
            xset_log_prob = -_sum_rightmost(
                transform.log_abs_det_jacobian(xset, y), event_dim - transform.event_dim
            )
            xset_next_log_prob = self._log_prob(xset, transforms)
            xset_terms = torch.where(
                mask,
                xset_log_prob.float() + xset_next_log_prob.float(),
                torch.tensor([float("-inf")], device=xset_log_prob.device),
            )

            terms = torch.cat([x_term[None], xset_terms])
            assert torch.isnan(terms).sum() == 0
            return torch.logsumexp(terms, dim=0)


class ConditionalLocalDiffeoTransformedDistribution:
    def __init__(self, base_dist, transforms):
        self.base_dist = base_dist if isinstance(
            base_dist, ConditionalDistribution) else ConstantConditionalDistribution(base_dist)
        self.transforms = transforms

    def condition(self, context):
        base_dist = self.base_dist.condition(context)
        # Should be conditioning transforms too, in case they are
        # ConditionalTransforms or ConditionalTransformModules and not just Transforms or TransformModules?
        # Not needed currently, because self.base_dist is a ConditionalTransformedDistribution
        # with its own ConditionalTransformModules. Thus an ConditionalTransform/Module can be made part of
        # self.base_dist, instead of passing as a transform to here.
        # Upon calling self.base_dist.condition(context), base_dist becomes a TransformedDistribution.
        return LocalDiffeoTransformedDistribution(base_distribution=base_dist,
                                                  transforms=self.transforms,
                                                  validate_args=False)

    def clear_cache(self):
        pass
