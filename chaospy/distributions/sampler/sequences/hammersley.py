"""
Create samples from the `Hammersley set`_.

The Hammersley set is equivalent to the Halton sequence, except for one
dimension is replaced with a regular grid.

Example usage
-------------

Standard usage::

    >>> distribution = chaospy.J(chaospy.Uniform(0, 1), chaospy.Uniform(0, 1))
    >>> samples = distribution.sample(3, rule="hammersley")
    >>> print(numpy.around(samples, 4))
    [[0.75  0.125 0.625]
     [0.25  0.5   0.75 ]]
    >>> samples = distribution.sample(4, rule="hammersley")
    >>> print(numpy.around(samples, 4))
    [[0.75  0.125 0.625 0.375]
     [0.2   0.4   0.6   0.8  ]]

.. _Hammersley set: https://en.wikipedia.org/wiki/Low-discrepancy_sequence#Hammersley_set
"""
import numpy

from .halton import create_halton_samples


def create_hammersley_samples(order, dim=1, burnin=-1, primes=()):
    """
    Create samples from the Hammersley set.

    For ``dim == 1`` the sequence falls back to Van Der Corput sequence.

    Args:
        order (int):
            The order of the Hammersley sequence. Defines the number of samples.
        dim (int):
            The number of dimensions in the Hammersley sequence.
        burnin (int):
            Skip the first ``burnin`` samples. If negative, the maximum of
            ``primes`` is used.
        primes (tuple):
            The (non-)prime base to calculate values along each axis. If
            empty, growing prime values starting from 2 will be used.

    Returns:
        (numpy.ndarray):
            Hammersley set with ``shape == (dim, order)``.
    """
    if dim == 1:
        return create_halton_samples(
            order=order, dim=1, burnin=burnin, primes=primes)
    out = numpy.empty((dim, order), dtype=float)
    out[:dim-1] = create_halton_samples(
        order=order, dim=dim-1, burnin=burnin, primes=primes)
    out[dim-1] = numpy.linspace(0, 1, order+2)[1:-1]
    return out
