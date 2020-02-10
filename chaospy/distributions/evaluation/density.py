"""
Evaluate probability density function (PDF).

Example usage
-------------

Define a simple distribution and data::

    >>> class Exponential(chaospy.Dist):
    ...     _cdf = lambda self, x_data, alpha: 1-numpy.e**(-alpha*x_data)
    ...     _pdf = lambda self, x_data, alpha: alpha*numpy.e**(-alpha*x_data)
    ...     _lower = lambda self, alpha: 0.
    ...     _upper = lambda self, alpha: 100.
    >>> dist = Exponential(alpha=2)
    >>> x_data = numpy.array([[0.1, 0.2, 0.3]])

Normal usage for calculating density::

    >>> print(numpy.around(evaluate_density(dist, x_data), 4))
    [[1.6375 1.3406 1.0976]]
    >>> print(numpy.around(evaluate_density(dist, x_data, parameters={"alpha": 1}), 4))
    [[0.9048 0.8187 0.7408]]

Cache get triggered if the same distribution is evaluated twice in the same
dependency resolution. The correct behavior for densities is to only return
non-zero values for coordinates that matches. E.g.::

    >>> cache_value = numpy.array([[0.1, 0.2, 0.7]])
    >>> print(numpy.around(evaluate_density(dist, x_data, cache={dist: cache_value}), 4))
    [[1.6375 1.3406 0.    ]]

If a distribution is missing the definition of the density function, it is
instead estimated from cumulative distribution function and boundary function::

    >>> dist = Exponential(alpha=1)
    >>> print(numpy.around(evaluate_density(dist, x_data), 4))
    [[0.9048 0.8187 0.7408]]
"""
import numpy

from .parameters import load_parameters
from .bound import evaluate_lower, evaluate_upper


def evaluate_density(
        distribution,
        x_data,
        parameters=None,
        cache=None,
):
    """
    Evaluate probability density function (PDF).

    Args:
        distribution (Dist):
            Distribution to evaluate.
        x_data (numpy.ndarray):
            Locations for where evaluate density at.
        parameters (:py:data:typing.Any):
            Collection of parameters to override the default ones in the
            distribution.
        cache (:py:data:typing.Any):
            A collection of previous calculations in case the same distribution
            turns up on more than one occasion.

    Returns:
        The probability density values of ``distribution`` at location
        ``x_data`` using parameters ``parameters``.
    """
    assert len(x_data) == len(distribution)
    assert len(x_data.shape) == 2

    cache = cache if cache is not None else {}
    out = numpy.zeros(x_data.shape)

    lower = evaluate_lower(distribution, cache=cache.copy())
    upper = evaluate_upper(distribution, cache=cache.copy())
    index = numpy.all((x_data.T >= lower.T) & (x_data.T <= upper.T), axis=1)

    # Distribution self know how to handle density evaluation.
    if hasattr(distribution, "_pdf"):
        parameters = load_parameters(
            distribution, "_pdf", parameters=parameters, cache=cache)
        ret_val = distribution._pdf(x_data, **parameters)
        ret_val = numpy.array(ret_val).reshape(len(distribution), -1)
        assert ret_val.shape[-1] in [1, len(index)], (
            "%s._pdf is returning something funky" % distribution)
        if ret_val.shape[-1] == 1:
            out[:, index] = ret_val
        else:
            out[:, index] = ret_val[:, index]

    # Approximate density evaluation based on cumulative distribution function.
    else:
        from .. import approximation
        parameters = load_parameters(
            distribution, "_cdf", parameters=parameters, cache=cache)
        out[:, index] = approximation.approximate_density(
            distribution, x_data, parameters=parameters, cache=cache)[:, index]

    # dependency handling.
    if distribution in cache:
        out = numpy.where(x_data == cache[distribution], out, 0)
    else:
        cache[distribution] = x_data

    return out
