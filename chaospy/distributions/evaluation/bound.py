"""
Evaluate distribution bounds: The location where it is reasonable to define the
distribution density is located between.

Example usage
-------------

Define a simple distribution and data::

    >>> class Uniform(chaospy.Dist):
    ...     _lower = lambda self, lo, up: lo
    ...     _upper = lambda self, lo, up: up
    >>> dist = Uniform(lo=1, up=3)

Normal usage::

    >>> evaluate_lower(dist)
    array([1.])
    >>> evaluate_upper(dist)
    array([3.])
    >>> evaluate_lower(dist, parameters={"lo": -1.})
    array([-1.])
"""
import numpy
from .parameters import load_parameters


def evaluate_lower(
        distribution,
        parameters=None,
        cache=None,
):
    """
    Evaluate lower and upper bounds.

    Args:
        distribution (Dist):
            Distribution to evaluate.
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
    cache = cache if cache is not None else {}

    parameters = load_parameters(
        distribution, "_lower", parameters=parameters)

    lower = distribution._lower(**parameters)
    lower = numpy.asfarray(lower)+numpy.zeros(len(distribution))

    cache[distribution] = lower
    return lower


def evaluate_upper(
        distribution,
        parameters=None,
        cache=None,
):
    """
    Evaluate lower and upper bounds.

    Args:
        distribution (Dist):
            Distribution to evaluate.
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
    cache = cache if cache is not None else {}

    parameters = load_parameters(
        distribution, "_upper", parameters=parameters)

    upper = distribution._upper(**parameters)
    upper = numpy.asfarray(upper)+numpy.zeros(len(distribution))
    cache[distribution] = upper
    return upper
