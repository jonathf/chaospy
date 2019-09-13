"""
Evaluate distribution bounds: The location where it is reasonable to define the
distribution density is located between.

Example usage
-------------

Define a simple distribution and data::

    >>> class Uniform(chaospy.Dist):
    ...     def _bnd(self, x_data, lo, up): return lo, up
    >>> dist = Uniform(lo=1, up=3)
    >>> x_data = numpy.array([[0.1, 0.2, 0.3]])

Normal usage::

    >>> lower, upper = evaluate_bound(dist, x_data)
    >>> print(numpy.around(lower, 4))
    [[1. 1. 1.]]
    >>> print(numpy.around(upper, 4))
    [[3. 3. 3.]]
    >>> lower, upper = evaluate_bound(dist, x_data, parameters={"lo": -1})
    >>> print(numpy.around(lower, 4))
    [[-1. -1. -1.]]
    >>> print(numpy.around(upper, 4))
    [[3. 3. 3.]]
"""
import numpy
from .parameters import load_parameters


def evaluate_bound(
        distribution,
        x_data,
        parameters=None,
        cache=None,
):
    """
    Evaluate lower and upper bounds.

    Args:
        distribution (Dist):
            Distribution to evaluate.
        x_data (numpy.ndarray):
            Locations for where evaluate bounds at. Relevant in the case of
            multivariate distributions where the bounds are affected by the
            output of other distributions.
        parameters (:py:data:typing.Any):
            Collection of parameters to override the default ones in the
            distribution.
        cache (:py:data:typing.Any):
            A collection of previous calculations in case the same distribution
            turns up on more than one occasion.

    Returns:
        The lower and upper bounds of ``distribution`` at location
        ``x_data`` using parameters ``parameters``.
    """
    assert len(x_data) == len(distribution)
    assert len(x_data.shape) == 2

    cache = cache if cache is not None else {}

    parameters = load_parameters(
        distribution, "_bnd", parameters=parameters, cache=cache)

    out = numpy.zeros((2,) + x_data.shape)

    lower, upper = distribution._bnd(x_data.copy(), **parameters)
    out.T[:, :, 0] = numpy.asfarray(lower).T
    out.T[:, :, 1] = numpy.asfarray(upper).T

    cache[distribution] = out

    return out
