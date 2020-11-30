"""Distribution for the core probability distribution."""
import itertools
import numpy
import chaospy

from .distribution import Distribution


class SimpleDistribution(Distribution):
    """Distribution for the core probability distribution."""

    def __init__(
            self,
            parameters=None,
            rotation=None,
            exclusion=None,
            repr_args=None,
    ):
        """
        Args:
            parameters (Optional[Distribution[str, Union[ndarray, Distribution]]]):
                Collection of model parameters.
            rotation (None, Sequence[int], Sequence[Sequence[bool]]):
                The order of which to resolve conditionals. Either as
                a sequence of column rotations, or as a permutation
                matrix. Defaults to `range(len(distribution))` which
                is the same as `p(x0), p(x1|x0), p(x2|x0,x1), ...`.
            repr_args (Optional[Sequence[str]]):
                Positional arguments to place in the object string
                representation. The repr output will then be:
                `<class name>(<arg1>, <arg2>, ...)`.

        """
        if parameters is None:
            parameters = {}
        dependencies, parameters, rotation = chaospy.declare_dependencies(
            distribution=self,
            parameters=parameters,
            rotation=rotation,
        )
        super(SimpleDistribution, self).__init__(
            parameters=parameters,
            dependencies=dependencies,
            rotation=rotation,
            exclusion=exclusion,
            repr_args=repr_args,
        )


    def get_parameters(self, idx, cache, assert_numerical=True):
        """
        Get distribution parameters.

        Uses the cache to replace parameters that are distribution with cached
        values.

        Args:
            cache (Dict[Distribution, numpy.ndarray]):
                Collection of already calculated values.

        Returns:
            (Dict[str, numpy.ndarray]):
                Collection of parameters. Probability distributions are
                replaced with cached values.

        Raise:
            UnsupportedFeature:
                If a parameter is a probability distribution without cache, it
                means the dependency is unresolved.

        """
        parameters = super(SimpleDistribution, self).get_parameters(
            idx, cache, assert_numerical=assert_numerical)
        del parameters["cache"]
        del parameters["idx"]
        for key, value in parameters.items():
            if isinstance(value, Distribution):
                value = value._get_cache(idx, cache, get=0)
                if isinstance(value, Distribution) and assert_numerical:
                    raise chaospy.UnsupportedFeature(
                        "operation not supported for %s with dependencies" % self)
                parameters[key] = value
        return parameters

    def get_upper_parameters(self, idx, cache):
        parameters = self.get_parameters(idx=idx, cache=cache, assert_numerical=False)
        for key, value in parameters.items():
            if isinstance(value, Distribution):
                parameters[key] = value._get_upper(idx, cache)
        return parameters

    def get_lower_parameters(self, idx, cache):
        parameters = self.get_parameters(idx=idx, cache=cache, assert_numerical=False)
        for key, value in parameters.items():
            if isinstance(value, Distribution):
                parameters[key] = value._get_lower(idx, cache)
        return parameters

    def get_mom_parameters(self):
        parameters = self.get_parameters(
            idx=None, cache={}, assert_numerical=False)
        if any([isinstance(value, Distribution) for value in parameters.values()]):
            raise chaospy.UnsupportedFeature(
                "operation not supported for %s with dependencies" % self)
        return parameters

    def _mom(self, kloc, **kwargs):
        """Default moment generator, throws error."""
        raise chaospy.UnsupportedFeature(
            "%s: does not support analytical raw moments." % self)

    def _ttr(self, kloc, **kwargs):
        """Default TTR generator, throws error."""
        raise chaospy.UnsupportedFeature(
            "%s: does not support analytical recurrence coefficients." % self)
