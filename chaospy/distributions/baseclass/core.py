"""
Constructing custom probability distributions is done by subclassing the
distribution :class:`~chaospy.distributions.baseclass.core.DistributionCore`::

    >>> class Uniform(chaospy.DistributionCore):
    ...     def __init__(self, lo=0, up=1):
    ...         '''Initializer.'''
    ...         super().__init__(lo=lo, up=up)
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

Usage is then straight forward::

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


class DistributionCore(Distribution):
    """Distribution for the core probability distribution."""

    def __init__(self, **parameters):
        """
        Args:
            parameters (Optional[Distribution[str, Union[ndarray, Distribution]]]):
                Collection of model parameters.
            rotation (Optional[Sequence[int]]):
                The order of which to evaluate dependencies.
            repr_args (Optional[Sequence[str]]):
                Positional arguments to place in the object string
                representation. The repr output will then be:
                `<class name>(<arg1>, <arg2>, ...)`.

        """
        repr_args = parameters.pop("repr_args", None)
        rotation = parameters.pop("rotation", None)
        dependencies = parameters.pop("dependencies", None)
        if dependencies is None:
            length = max([(len(param) if isinstance(param, Distribution)
                           else len(numpy.atleast_1d(param)))
                          for param in parameters.values()]+[1])
            dependencies = [set([idx]) for idx in self._declare_dependencies(length)]
        for param in parameters.values():
            if isinstance(param, Distribution):
                assert len(param) in (1, len(dependencies))
                for idx in range(len(dependencies)):
                    dependencies[idx].update(
                        param._dependencies[min(idx, len(param._dependencies)-1)])
        super(DistributionCore, self).__init__(
            parameters=parameters,
            dependencies=dependencies,
            rotation=rotation,
            repr_args=repr_args,
        )

    def get_parameters(self, cache):
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
        parameters = super(DistributionCore, self).get_parameters(cache)
        parameters.pop("cache")
        for key, value in parameters.items():
            if isinstance(value, Distribution):
                if value in cache:
                    parameters[key] = cache[value]
        return parameters

    def _check_parameters(self, parameters):
        """
        Check validity of distribution parameters.

        Args:
            parameters (Dict[str, Union[Distribution, numpy.ndarray]]):
                Collection of parameters to check if valid.

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
        for key, value in parameters.items():
            if isinstance(value, Distribution):
                raise chaospy.UnsupportedFeature(
                    "%s: Dependency %s=%s unresolved" % (self, key, value))

    def _mom(self, kloc, **kwargs):
        """Default moment generator, throws error."""
        raise chaospy.UnsupportedFeature(
            "%s: does not support analytical raw moments." % self)

    def _ttr(self, kloc, **kwargs):
        """Default TTR generator, throws error."""
        raise chaospy.UnsupportedFeature(
            "%s: does not support analytical recurrence coefficients." % self)

    def _value(self, **kwargs):
        return self
