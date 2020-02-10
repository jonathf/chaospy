"""
Evaluate inverse Rosenblatt transformation.

In the case of univariate distributions, this is equivalent to the point
percentile function (PPF).

Example usage
-------------

Define a simple distribution and data::

    >>> class Exponential(chaospy.Dist):
    ...     def _ppf(self, u_data, alpha): return -numpy.log(1-u_data)/alpha
    >>> dist = Exponential(alpha=2)
    >>> u_data = numpy.array([[0.1, 0.2, 0.3]])

Normal usage::

    >>> print(numpy.around(evaluate_inverse(dist, u_data), 4))
    [[0.0527 0.1116 0.1783]]

Use non-default parameters::

    >>> print(numpy.around(evaluate_inverse(
    ...     dist, u_data, parameters={"alpha": 1}), 4))
    [[0.1054 0.2231 0.3567]]

If a distribution is missing the definition of the density function, it is
instead estimated from cumulative distribution function and boundary function::

    >>> class Exponential(chaospy.Dist):
    ...     _cdf = lambda self, x_data, alpha: 1-numpy.e**(-alpha*x_data)
    ...     _lower = lambda self, alpha: 0.
    ...     _upper = lambda self, alpha: 100.
    >>> dist = Exponential(alpha=1)
    >>> print(numpy.around(evaluate_inverse(dist, u_data), 4))
    [[0.1054 0.2231 0.3567]]
"""
import numpy
from .parameters import load_parameters


def evaluate_inverse(
        distribution,
        u_data,
        cache=None,
        parameters=None
):
    """
    Evaluate inverse Rosenblatt transformation.

    Args:
        distribution (Dist):
            Distribution to evaluate.
        u_data (numpy.ndarray):
            Locations for where evaluate inverse transformation distribution at.
        parameters (:py:data:typing.Any):
            Collection of parameters to override the default ones in the
            distribution.
        cache (:py:data:typing.Any):
            A collection of previous calculations in case the same distribution
            turns up on more than one occasion.

    Returns:
        The cumulative distribution values of ``distribution`` at location
        ``u_data`` using parameters ``parameters``.
    """
    cache = cache if cache is not None else {}
    dtype = int if distribution.interpret_as_integer else float
    out = numpy.zeros(u_data.shape, dtype=dtype)

    # Distribution self know how to handle inverse Rosenblatt.
    if hasattr(distribution, "_ppf"):
        parameters = load_parameters(
            distribution, "_ppf", parameters=parameters, cache=cache)
        out[:] = distribution._ppf(u_data.copy(), **parameters)

    # Approximate inverse Rosenblatt based on cumulative distribution function.
    else:
        from .. import approximation
        parameters = load_parameters(
            distribution, "_cdf", parameters=parameters, cache=cache)
        out[:] = approximation.approximate_inverse(
            distribution, u_data.copy(), cache=cache.copy(), parameters=parameters)

    # Store cache.
    cache[distribution] = out

    return out
