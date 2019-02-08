"""
Create `Van Der Corput` low discrepancy sequence samples.

A van der Corput sequence is an example of the simplest one-dimensional
low-discrepancy sequence over the unit interval; it was first described in 1935
by the Dutch mathematician J. G. van der Corput. It is constructed by reversing
the base-n representation of the sequence of natural numbers (1, 2, 3, ...).

In practice, use Halton sequence instead of Van Der Corput, as it is the
same, but generalized to work in multiple dimensions.

Example usage
-------------

Using base 10::

    >>> print(create_van_der_corput_samples(range(11), number_base=10))
    [0.1  0.2  0.3  0.4  0.5  0.6  0.7  0.8  0.9  0.01 0.11]

Using base 2::

    >>> print(create_van_der_corput_samples(range(8), number_base=2))
    [0.5    0.25   0.75   0.125  0.625  0.375  0.875  0.0625]

.. Van Der Corput: https://en.wikipedia.org/wiki/Van_der_Corput_sequence
"""
from __future__ import division
import numpy


def create_van_der_corput_samples(idx, number_base=2):
    """
    Van der Corput samples.

    Args:
        idx (int, numpy.ndarray):
            The index of the sequence. If array is provided, all values in
            array is returned.
        number_base (int):
            The numerical base from where to create the samples from.

    Returns (float, numpy.ndarray):
        Van der Corput samples.
    """
    assert number_base > 1

    idx = numpy.asarray(idx).flatten() + 1
    out = numpy.zeros(len(idx), dtype=float)

    base = float(number_base)
    active = numpy.ones(len(idx), dtype=bool)
    while numpy.any(active):
        out[active] += (idx[active] % number_base)/base
        idx //= number_base
        base *= number_base
        active = idx > 0
    return out
