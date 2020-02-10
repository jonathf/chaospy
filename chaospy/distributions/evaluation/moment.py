"""
Evaluate raw statistical moments.

Example usage
-------------

Define a simple distribution and data::

    >>> class Exponential(chaospy.Dist):
    ...     def _mom(self, k_data, alpha): return alpha**-k_data
    >>> dist = Exponential(alpha=2.)
    >>> k_data = numpy.array([1])

Normal usage::

    >>> print(evaluate_moment(dist, k_data))
    0.5
    >>> print(evaluate_moment(dist, k_data, parameters={"alpha": 1.}))
    1.0

0-th order is always precomputed::

    >>> print(evaluate_moment(dist, numpy.array([0])))
    1.0

The use of cache::

    >>> print(evaluate_moment(dist, k_data, cache={((1,), dist): 5.}))
    5.0

Approximate with the use of density function if moment is missing::

    >>> class Exponential(chaospy.Dist):
    ...     _cdf = lambda self, x_data, alpha: 1-numpy.e**(-alpha*x_data)
    ...     _lower = lambda self, alpha: 0
    ...     _upper = lambda self, alpha: 50
    ...     _pdf = lambda self, x_data, alpha: alpha*numpy.e**(-alpha*x_data)
    >>> dist = Exponential(alpha=2)
    >>> print(numpy.around(evaluate_moment(dist, k_data), 4))
    0.5
"""
import logging

import numpy
from .parameters import load_parameters


def evaluate_moment(
        distribution,
        k_data,
        parameters=None,
        cache=None,
):
    """
    Evaluate raw statistical moments.

    Args:
        distribution (Dist):
            Distribution to evaluate.
        x_data (numpy.ndarray):
            Locations for where evaluate moment of.
        parameters (:py:data:typing.Any):
            Collection of parameters to override the default ones in the
            distribution.
        cache (:py:data:typing.Any):
            A collection of previous calculations in case the same distribution
            turns up on more than one occasion.

    Returns:
        The raw statistical moment of ``distribution`` at location ``x_data``
        using parameters ``parameters``.
    """
    logger = logging.getLogger(__name__)
    assert len(k_data) == len(distribution), (
        "distribution %s is not of length %d" % (distribution, len(k_data)))
    assert len(k_data.shape) == 1

    if numpy.all(k_data == 0):
        return 1.

    def cache_key(distribution):
        return (tuple(k_data), distribution)

    if cache is None:
        cache = {}
    else:
        if cache_key(distribution) in cache:
            return cache[cache_key(distribution)]

    from .. import baseclass
    try:
        parameters = load_parameters(
            distribution, "_mom", parameters, cache, cache_key)
        out = distribution._mom(k_data, **parameters)

    except baseclass.StochasticallyDependentError:

        logger.info(
            "Distribution %s has stochastic dependencies; "
            "Approximating moments with quadrature.", distribution)
        from .. import approximation
        out = approximation.approximate_moment(distribution, k_data)

    if isinstance(out, numpy.ndarray):
        out = out.item()

    cache[cache_key(distribution)] = out

    return out
