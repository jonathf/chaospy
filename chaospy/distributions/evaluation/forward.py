"""
Evaluate forward Rosenblatt transformation.

In the case of univariate distributions, this is equivalent to the cumulative
distribution function (CDF).

Example usage
-------------

Define a simple distribution and data::

    >>> class Exponential(chaospy.Dist):
    ...     _cdf = lambda self, x_data, alpha: 1-numpy.e**(-alpha*x_data)
    ...     _lower = lambda self, alpha: 0.
    ...     _upper = lambda self, alpha: 100.
    >>> dist = Exponential(alpha=2)
    >>> x_data = numpy.array([[0.1, 0.2, 0.3]])

Normal usage::

    >>> print(numpy.around(evaluate_forward(dist, x_data), 4))
    [[0.1813 0.3297 0.4512]]

Use non-default parameters::

    >>> print(numpy.around(evaluate_forward(
    ...     dist, x_data, parameters={"alpha": 1}), 4))
    [[0.0952 0.1813 0.2592]]
"""
import numpy

from .parameters import load_parameters
from .bound import evaluate_lower, evaluate_upper


def evaluate_forward(
        distribution,
        x_data,
        parameters=None,
        cache=None,
):
    """
    Evaluate forward Rosenblatt transformation.

    Args:
        distribution (Dist):
            Distribution to evaluate.
        x_data (numpy.ndarray):
            Locations for where evaluate forward transformation at.
        parameters (:py:data:typing.Any):
            Collection of parameters to override the default ones in the
            distribution.
        cache (:py:data:typing.Any):
            A collection of previous calculations in case the same distribution
            turns up on more than one occasion.

    Returns:
        The cumulative distribution values of ``distribution`` at location
        ``x_data`` using parameters ``parameters``.
    """
    assert len(x_data) == len(distribution), (
        "distribution %s is not of length %d" % (distribution, len(x_data)))
    assert hasattr(distribution, "_cdf"), (
        "distribution require the `_cdf` method to function.")

    cache = cache if cache is not None else {}

    lower = evaluate_lower(distribution, cache=cache.copy())
    upper = evaluate_upper(distribution, cache=cache.copy())

    parameters = load_parameters(
        distribution, "_cdf", parameters=parameters, cache=cache)

    # Store cache.
    cache[distribution] = x_data

    # Evaluate forward function.
    out = numpy.zeros(x_data.shape)
    out[:] = distribution._cdf(x_data, **parameters)
    out[(x_data.T < lower.T).T] = 0
    out[(x_data.T > upper.T).T] = 1

    return out
