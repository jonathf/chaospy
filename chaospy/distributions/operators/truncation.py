"""
Truncation.

Example usage
-------------

Simple distribution to start with::

    >>> distribution = chaospy.Normal(0, 1)
    >>> distribution.inv([0.9, 0.99, 0.999]).round(4)
    array([1.2816, 2.3263, 3.0902])

Same distribution, but with a upper truncation::

    >>> right_trunc = chaospy.Trunc(chaospy.Normal(0, 1), upper=1)
    >>> right_trunc
    Trunc(Normal(mu=0, sigma=1), upper=1)
    >>> right_trunc.inv([0.9, 0.99, 0.999]).round(4)
    array([0.6974, 0.9658, 0.9965])

Same, but with lower truncation::

    >>> left_trunc = chaospy.Trunc(chaospy.Normal(0, 1), lower=1)
    >>> left_trunc
    Trunc(Normal(mu=0, sigma=1), lower=1)
    >>> left_trunc.inv([0.001, 0.01, 0.1]).round(4)
    array([1.0007, 1.0066, 1.0679])

"""
import numpy
import chaospy

from ..baseclass import Distribution, OperatorDistribution


class Trunc(Distribution):
    """Truncation."""

    def __init__(self, dist, lower=None, upper=None):
        """
        Constructor.

        Args:
            dist (Distribution):
                Distribution to be truncated.
            lower (Distribution, numpy.ndarray):
                Lower truncation bound.
            upper (Distribution, numpy.ndarray):
                Upper truncation bound.
        """
        assert isinstance(dist, Distribution)
        repr_args = [dist]
        repr_args += chaospy.format_repr_kwargs(lower=(lower, None))
        repr_args += chaospy.format_repr_kwargs(upper=(upper, None))
        exclusion = set()
        for deps in dist._dependencies:
            exclusion.update(deps)
        if isinstance(lower, Distribution):
            if lower.stochastic_dependent:
                raise chaospy.StochasticallyDependentError(
                    "Joint distribution with dependencies not supported.")
            assert len(dist) == len(lower)
            lower_ = lower.lower
        elif lower is None:
            lower = lower_ = dist.lower
        else:
            lower = lower_ = numpy.atleast_1d(lower)
        if isinstance(upper, Distribution):
            if upper.stochastic_dependent:
                raise chaospy.StochasticallyDependentError(
                    "Joint distribution with dependencies not supported.")
            assert len(dist) == len(upper)
            upper_ = upper.upper
        elif upper is None:
            upper = upper_ = dist.upper
        else:
            upper = upper_ = numpy.atleast_1d(upper)
        assert numpy.all(upper_ > lower_), (
            "condition `upper > lower` not satisfied: %s <= %s" % (upper_, lower_))

        dependencies, parameters, rotation = chaospy.declare_dependencies(
            distribution=self,
            parameters=dict(lower=lower, upper=upper),
            length=len(dist),
        )
        super(Trunc, self).__init__(
            parameters=parameters,
            dependencies=dependencies,
            exclusion=exclusion,
            repr_args=repr_args,
        )
        self._dist = dist

    def get_parameters(self, idx, cache, assert_numerical=True):
        parameters = super(Trunc, self).get_parameters(
            idx, cache, assert_numerical=assert_numerical)
        assert set(parameters) == {"cache", "lower", "upper", "idx"}

        if isinstance(parameters["lower"], Distribution):
            parameters["lower"] = parameters["lower"]._get_cache(idx, cache=parameters["cache"], get=0)
        elif len(parameters["lower"]) > 1 and idx is not None:
            parameters["lower"] = parameters["lower"][idx]
        if isinstance(parameters["upper"], Distribution):
            parameters["upper"] = parameters["upper"]._get_cache(idx, cache=parameters["cache"], get=0)
        elif len(parameters["upper"]) > 1 and idx is not None:
            parameters["upper"] = parameters["upper"][idx]
        if assert_numerical:
            assert (not isinstance(parameters["lower"], Distribution) or
                    not isinstance(parameters["upper"], Distribution))
        if idx is None:
            del parameters["idx"]
        return parameters

    def _lower(self, idx, lower, upper, cache):
        """
        Distribution lower bound.

        Examples:
            >>> chaospy.Trunc(chaospy.Uniform(), upper=0.6).lower
            array([0.])
            >>> chaospy.Trunc(chaospy.Uniform(), lower=0.6).lower
            array([0.6])
        """
        del upper
        if isinstance(lower, Distribution):
            lower = lower._get_lower(idx, cache=cache)
        return lower

    def _upper(self, idx, lower, upper, cache):
        """
        Distribution lower bound.

        Examples:
            >>> chaospy.Trunc(chaospy.Uniform(), upper=0.6).upper
            array([0.6])
            >>> chaospy.Trunc(chaospy.Uniform(), lower=0.6).upper
            array([1.])
        """
        del lower
        if isinstance(upper, Distribution):
            upper = upper._get_upper(idx, cache=cache)
        return upper

    def _cdf(self, xloc, idx, lower, upper, cache):
        """
        Cumulative distribution function.

        Example:
            >>> chaospy.Uniform().fwd([-0.5, 0.3, 0.7, 1.2])
            array([0. , 0.3, 0.7, 1. ])
            >>> chaospy.Trunc(chaospy.Uniform(), upper=0.4).fwd([-0.5, 0.2, 0.8, 1.2])
            array([0. , 0.5, 1. , 1. ])
            >>> chaospy.Trunc(chaospy.Uniform(), lower=0.6).fwd([-0.5, 0.2, 0.8, 1.2])
            array([0. , 0. , 0.5, 1. ])
        """
        assert not isinstance(lower, Distribution)
        assert not isinstance(upper, Distribution)
        lower = numpy.broadcast_to(lower, xloc.shape)
        upper = numpy.broadcast_to(upper, xloc.shape)
        lower = self._dist._get_fwd(lower, idx, cache=cache.copy())
        upper = self._dist._get_fwd(upper, idx, cache=cache.copy())
        uloc = self._dist._get_fwd(xloc, idx, cache)
        return (uloc-lower)/(1-lower)/upper

    def _pdf(self, xloc, idx, lower, upper, cache):
        """
        Probability density function.

        Example:
            >>> dist = chaospy.Trunc(chaospy.Uniform(), upper=0.6)
            >>> dist.pdf([-0.25, 0.25, 0.5, 0.75, 1.25])
            array([0.        , 1.66666667, 1.66666667, 0.        , 0.        ])
            >>> dist = chaospy.Trunc(chaospy.Uniform(), upper=0.4)
            >>> dist.pdf([-0.25, 0.25, 0.5, 0.75, 1.25])
            array([0. , 2.5, 0. , 0. , 0. ])
            >>> dist = chaospy.Trunc(chaospy.Uniform(), lower=0.4)
            >>> dist.pdf([-0.25, 0.25, 0.5, 0.75, 1.25])
            array([0.        , 0.        , 1.66666667, 1.66666667, 0.        ])
            >>> dist = chaospy.Trunc(chaospy.Uniform(), lower=0.6)
            >>> dist.pdf([-0.25, 0.25, 0.5, 0.75, 1.25])
            array([0. , 0. , 0. , 2.5, 0. ])
        """
        assert not isinstance(lower, Distribution)
        assert not isinstance(upper, Distribution)
        lower = numpy.broadcast_to(lower, xloc.shape)
        upper = numpy.broadcast_to(upper, xloc.shape)
        lower = self._dist._get_fwd(lower, idx, cache=cache.copy())
        upper = self._dist._get_fwd(upper, idx, cache=cache.copy())
        uloc = self._dist._get_pdf(xloc, idx, cache=cache)
        return uloc/(1-lower)/upper

    def _ppf(self, qloc, idx, lower, upper, cache):
        """
        Point percentile function.

        Example:
            >>> chaospy.Uniform().inv([0.1, 0.2, 0.9])
            array([0.1, 0.2, 0.9])
            >>> chaospy.Trunc(chaospy.Uniform(), upper=0.4).inv([0.1, 0.2, 0.9])
            array([0.04, 0.08, 0.36])
            >>> chaospy.Trunc(chaospy.Uniform(), lower=0.6).inv([0.1, 0.2, 0.9])
            array([0.64, 0.68, 0.96])
        """
        assert not isinstance(lower, Distribution)
        assert not isinstance(upper, Distribution)
        lower = numpy.broadcast_to(lower, qloc.shape)
        upper = numpy.broadcast_to(upper, qloc.shape)
        lower = self._dist._get_fwd(lower, idx, cache=cache.copy())
        upper = self._dist._get_fwd(upper, idx, cache=cache.copy())
        return self._dist._get_inv(qloc*upper*(1-lower)+lower, idx, cache=cache)
