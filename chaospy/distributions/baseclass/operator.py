"""Operator transformation."""
import numpy
import chaospy

from ..baseclass import Distribution


class OperatorDistribution(Distribution):
    """Operator transformation."""

    def __init__(self, left, right, exclusion=None, repr_args=None):
        if not isinstance(left, Distribution):
            left = numpy.atleast_1d(left)
            if left.ndim > 1:
                raise chaospy.UnsupportedFeature(
                    "distribution operators limited to at-most 1D arrays.")
        if not isinstance(right, Distribution):
            right = numpy.atleast_1d(right)
            if right.ndim > 1:
                raise chaospy.UnsupportedFeature(
                    "distribution operators limited to at-most 1D arrays.")
        dependencies, parameters, rotation = chaospy.declare_dependencies(
            distribution=self,
            parameters=dict(left=left, right=right),
            is_operator=True,
        )

        super(OperatorDistribution, self).__init__(
            parameters=parameters,
            dependencies=dependencies,
            exclusion=exclusion,
            repr_args=repr_args,
        )
        self._cache_copy = {}
        self._lower_cache = {}
        self._upper_cache = {}

    def _get_left_right(self, idx, cache):
        if cache is not self._cache_copy:
            self._cache_copy = cache
            self._lower_cache = {}
            self._upper_cache = {}
        parameters = super(OperatorDistribution, self).get_parameters(idx, cache)
        assert set(parameters) == {"cache", "left", "right", "idx"}

        left = parameters["left"]
        if isinstance(left, Distribution):
            left = left._get_cache(idx, cache=cache, get=0)
        elif len(left) > 1 and idx is not None:
            left = left[idx]
        right = parameters["right"]
        if isinstance(right, Distribution):
            right = right._get_cache(idx, cache=cache, get=0)
        elif len(right) > 1 and idx is not None:
            right = right[idx]
        return left, right

    def get_parameters(self, idx, cache):
        left, right = self._get_left_right(idx, cache)
        assert not isinstance(left, Distribution) or not isinstance(right, Distribution)
        return dict(idx=idx, left=left, right=right, cache=cache)

    def get_lower_parameters(self, idx, cache):
        left, right = self._get_left_right(idx, cache)
        return dict(idx=idx, left=left, right=right, cache=cache)

    def get_upper_parameters(self, idx, cache):
        left, right = self._get_left_right(idx, cache)
        return dict(idx=idx, left=left, right=right, cache=cache)

    def get_mom_parameters(self):
        left, right = self._get_left_right(idx=None, cache={})
        return dict(left=left, right=right)

    def get_ttr_parameters(self, idx):
        left, right = self._get_left_right(idx=idx, cache={})
        assert not isinstance(left, Distribution) or not isinstance(right, Distribution)
        left, right = self._get_left_right(idx=idx, cache={})
        return dict(idx=idx, left=left, right=right)
