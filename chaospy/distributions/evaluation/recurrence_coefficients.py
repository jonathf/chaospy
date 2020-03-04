"""
Evaluate three terms recurrence coefficients (TTR).

Example usage
-------------

Define a simple distribution and data::

    >>> class Exponential(chaospy.Dist):
    ...     def _ttr(self, k_data, alpha): return (2*k_data+1)/alpha, (k_data/alpha)**2
    >>> dist = Exponential(alpha=2.)
    >>> k_data = numpy.array([2])

Normal usage::

    >>> evaluate_recurrence_coefficients(dist, k_data)
    array([2.5, 1. ])
    >>> evaluate_recurrence_coefficients(dist, k_data, parameters={"alpha": 1.})
    array([5., 4.])

The use of cache::

    >>> evaluate_recurrence_coefficients(dist, k_data, cache={((2,), dist): (3., 4.)})
    (3.0, 4.0)
"""
import logging

import numpy
from .parameters import load_parameters


def evaluate_recurrence_coefficients(
        distribution,
        k_data,
        parameters=None,
        cache=None,
):
    """
    Evaluate three terms recurrence coefficients (TTR).

    Args:
        distribution (Dist):
            Distribution to evaluate.
        x_data (numpy.ndarray):
            Locations for where evaluate recurrence coefficients for.
        parameters (:py:data:typing.Any):
            Collection of parameters to override the default ones in the
            distribution.
        cache (:py:data:typing.Any):
            A collection of previous calculations in case the same distribution
            turns up on more than one occasion.

    Returns:
        The recurrence coefficients ``A`` and ``B`` of ``distribution`` at
        location ``x_data`` using parameters ``parameters``.
    """
    assert len(k_data) == len(distribution), (
        "distribution %s is not of length %d" % (distribution, len(k_data)))
    assert len(k_data.shape) == 1

    def cache_key(distribution):
        return (tuple(k_data), distribution)

    if cache is None:
        cache = {}
    else:
        if cache_key(distribution) in cache:
            return cache[cache_key(distribution)]

    parameters = load_parameters(
        distribution, "_ttr", parameters, cache, cache_key)
    coeff1, coeff2 = distribution._ttr(k_data, **parameters)

    out = numpy.zeros((2,) + k_data.shape)
    out.T[:, 0] = numpy.asarray(coeff1).T
    out.T[:, 1] = numpy.asarray(coeff2).T
    if len(distribution) == 1:
        out = out[:, 0]

    return out
