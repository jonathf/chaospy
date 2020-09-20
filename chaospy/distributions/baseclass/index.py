"""Baseclass for all conditional distributions."""
import numpy
import chaospy

from .distribution import Distribution


class Index(Distribution):
    """Baseclass for an index of multivariate distribution."""

    def __init__(self, parent, conditions=()):
        idx = parent._rotation[len(conditions)]
        if conditions:
            assert isinstance(conditions, chaospy.J)
            parameters = dict(parent=parent, idx=idx, conditions=conditions)
            repr_args = [parent, conditions]
        else:
            parameters = dict(parent=parent, idx=idx)
            repr_args = [parent]
        super(Index, self).__init__(
            parameters=parameters,
            dependencies=[parent._dependencies[idx].copy()],
            rotation=[0],
            repr_args=repr_args,
        )

    def get_parameters(self, cache, assert_numerical=True):
        """Get distribution parameters."""
        parameters = super(Index, self).get_parameters(cache, assert_numerical=assert_numerical)
        if "conditions" not in parameters:
            parameters["conditions"] = []
        if assert_numerical and not all([condition in cache for condition in parameters["conditions"]]):
            raise chaospy.UnsupportedFeature("Conditions not resolved")
        return parameters

    def _lower(self, idx, parent, conditions, cache):
        out = parent._get_cache_1(cache)
        if isinstance(out, Distribution):
            out = out.lower
        return out[idx]

    def _upper(self, idx, parent, conditions, cache):
        out = parent._get_cache_1(cache)
        if isinstance(out, Distribution):
            out = out.upper
        return out[idx]

    def __repr__(self):
        return "Index(%(idx)d, %(parent)s)" % self._parameters
