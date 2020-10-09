import numpy
import chaospy

from ..baseclass import Distribution


class OperatorDistribution(Distribution):

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

    def get_parameters(self, idx, cache, assert_numerical=True):
        parameters = super(OperatorDistribution, self).get_parameters(
            idx, cache, assert_numerical=assert_numerical)
        assert set(parameters) == {"cache", "left", "right", "idx"}

        if isinstance(parameters["left"], Distribution):
            parameters["left"] = parameters["left"]._get_cache(idx, cache=parameters["cache"], get=0)
        elif len(parameters["left"]) > 1 and idx is not None:
            parameters["left"] = parameters["left"][idx]
        if isinstance(parameters["right"], Distribution):
            parameters["right"] = parameters["right"]._get_cache(idx, cache=parameters["cache"], get=0)
        elif len(parameters["right"]) > 1 and idx is not None:
            parameters["right"] = parameters["right"][idx]

        if assert_numerical:
            assert (not isinstance(parameters["left"], Distribution) or
                    not isinstance(parameters["right"], Distribution))
        if cache is not self._cache_copy:
            self._cache_copy = cache
            self._lower_cache = {}
            self._upper_cache = {}
        if idx is None:
            del parameters["idx"]
        return parameters
