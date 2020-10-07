"""
Constructing custom probability distributions is done by subclassing the
distribution :class:`~chaospy.distributions.baseclass.simple.SimpleDistribution`::

    >>> class Uniform(chaospy.SimpleDistribution):
    ...     def __init__(self, lo=0, up=1):
    ...         '''Initializer.'''
    ...         super().__init__(parameters=dict(lo=lo, up=up))
    ...     def _cdf(self, x_data, lo, up):
    ...         '''Cumulative distribution function.'''
    ...         return (x_data-lo)/(up-lo)
    ...     def _lower(self, lo, up):
    ...         '''Lower bound.'''
    ...         return lo
    ...     def _upper(self, lo, up):
    ...         '''Upper bound.'''
    ...         return up
    ...     def _pdf(self, x_data, lo, up):
    ...         '''Probability density function.'''
    ...         return 1./(up-lo)
    ...     def _ppf(self, q_data, lo, up):
    ...         '''Point percentile function.'''
    ...         return q_data*(up-lo)+lo

Custom distributions needs to be initialized and wrapped. For example::

    >>> dist = Uniform(-3, 3)
    >>> dist.fwd([-3, 0, 3])  # Forward Rosenblatt transformation
    array([0. , 0.5, 1. ])

Here the method ``_cdf`` is an absolute requirement. In addition, either
``_ppf``, or the couple ``_lower`` and ``_upper`` should be provided. The
others are not required, but may increase speed and or accuracy of
calculations. In addition to the once listed, it is also
possible to define the following methods:

``_mom``
    Method for creating raw statistical moments, used by the ``mom`` method.
``_ttr``
    Method for creating coefficients from three terms recurrence method, used to
    perform "analytical" Stiltjes' method.
"""
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
                Collection of parameters. Probability distributions
                are replaced with cached values.

        Raise:
            UnsupportedFeature:
                If a parameter is a probability distribution
                without cache, it means the dependency is
                unresolved.

        """
        parameters = super(SimpleDistribution, self).get_parameters(
            idx, cache, assert_numerical=assert_numerical)
        del parameters["cache"]
        del parameters["idx"]
        for key, value in parameters.items():
            if isinstance(value, Distribution):
                value = value._get_cache(idx, cache, get=0)
                assert not assert_numerical or not isinstance(value, Distribution)
                parameters[key] = value
            if idx is not None and len(value) > 1:
                parameters[key] = value[idx]
        return parameters

    def _mom(self, kloc, **kwargs):
        """Default moment generator, throws error."""
        raise chaospy.UnsupportedFeature(
            "%s: does not support analytical raw moments." % self)

    def _ttr(self, kloc, **kwargs):
        """Default TTR generator, throws error."""
        raise chaospy.UnsupportedFeature(
            "%s: does not support analytical recurrence coefficients." % self)
