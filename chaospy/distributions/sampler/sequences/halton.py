"""
Create samples from the `Halton sequence`_.

In statistics, Halton sequences are sequences used to generate points in space
for numerical methods such as Monte Carlo simulations. Although these sequences
are deterministic, they are of low discrepancy, that is, appear to be random
for many purposes. They were first introduced in 1960 and are an example of
a quasi-random number sequence. They generalise the one-dimensional van der
Corput sequences.

Example usage
-------------

Standard usage::

    >>> distribution = chaospy.J(chaospy.Uniform(0, 1), chaospy.Uniform(0, 1))
    >>> samples = distribution.sample(3, rule="halton")
    >>> print(numpy.around(samples, 4))
    [[0.125  0.625  0.375 ]
     [0.4444 0.7778 0.2222]]
    >>> samples = distribution.sample(4, rule="halton")
    >>> print(numpy.around(samples, 4))
    [[0.125  0.625  0.375  0.875 ]
     [0.4444 0.7778 0.2222 0.5556]]

.. _Halton sequence: https://en.wikipedia.org/wiki/Halton_sequence
"""
import numpy

from .van_der_corput import create_van_der_corput_samples
from .primes import create_primes


def create_halton_samples(order, dim=1, burnin=-1, primes=()):
    """
    Create Halton sequence.

    For ``dim == 1`` the sequence falls back to Van Der Corput sequence.

    Args:
        order (int):
            The order of the Halton sequence. Defines the number of samples.
        dim (int):
            The number of dimensions in the Halton sequence.
        burnin (int):
            Skip the first ``burnin`` samples. If negative, the maximum of
            ``primes`` is used.
        primes (tuple):
            The (non-)prime base to calculate values along each axis. If
            empty, growing prime values starting from 2 will be used.

    Returns (numpy.ndarray):
        Halton sequence with ``shape == (dim, order)``.
    """
    primes = list(primes)
    if not primes:
        prime_order = 10*dim
        while len(primes) < dim:
            primes = create_primes(prime_order)
            prime_order *= 2
    primes = primes[:dim]
    assert len(primes) == dim, "not enough primes"

    if burnin < 0:
        burnin = max(primes)

    out = numpy.empty((dim, order))
    indices = [idx+burnin for idx in range(order)]
    for dim_ in range(dim):
        out[dim_] = create_van_der_corput_samples(
            indices, number_base=primes[dim_])
    return out
