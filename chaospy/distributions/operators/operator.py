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
        length = max(len(left), len(right))
        dependencies = [set() for _ in range(length)]
        if isinstance(left, Distribution):
            if len(left) == 1:
                dependencies = [
                    dep.union(left._dependencies[0])
                    for dep in dependencies
                ]
            else:
                dependencies = [
                    dep.union(other)
                    for dep, other in zip(dependencies, left._dependencies)
                ]
        if isinstance(right, Distribution):
            if len(right) == 1:
                dependencies = [
                    dep.union(right._dependencies[0])
                    for dep in dependencies
                ]
            else:
                dependencies = [
                    dep.union(other)
                    for dep, other in zip(dependencies, right._dependencies)
                ]
        super(OperatorDistribution, self).__init__(
            parameters=dict(left=left, right=right),
            dependencies=dependencies,
            exclusion=exclusion,
            repr_args=repr_args,
        )

    def get_parameters(self, **kwargs):
        parameters = super(OperatorDistribution, self).get_parameters(**kwargs)
        assert set(parameters) == {"cache", "left", "right"}
        if isinstance(parameters["left"], Distribution):
            parameters["left"] = parameters["left"]._get_cache_1(cache=parameters["cache"])
        if isinstance(parameters["right"], Distribution):
            parameters["right"] = parameters["right"]._get_cache_1(cache=parameters["cache"])
        return parameters
