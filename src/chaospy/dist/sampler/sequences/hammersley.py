"""
Create samples from the `Hammersley set`_.

The Hammersley set is equivalent to the Halton sequence, except for one
dimension is replaced with a regular grid.

Example usage
-------------

Standard usage::

    >>> print(create_hammersley_samples(order=3, dim=2))
    [[ 0.75   0.125  0.625]
     [ 0.25   0.5    0.75 ]]
    >>> print(create_hammersley_samples(order=3, dim=3))
    [[ 0.125       0.625       0.375     ]
     [ 0.44444444  0.77777778  0.22222222]
     [ 0.25        0.5         0.75      ]]

Custom burn-ins::

    >>> print(create_hammersley_samples(order=3, dim=3, burnin=0))
    [[ 0.5         0.25        0.75      ]
     [ 0.33333333  0.66666667  0.11111111]
     [ 0.25        0.5         0.75      ]]
    >>> print(create_hammersley_samples(order=3, dim=3, burnin=1))
    [[ 0.25        0.75        0.125     ]
     [ 0.66666667  0.11111111  0.44444444]
     [ 0.25        0.5         0.75      ]]
    >>> print(create_hammersley_samples(order=3, dim=3, burnin=2))
    [[ 0.75        0.125       0.625     ]
     [ 0.11111111  0.44444444  0.77777778]
     [ 0.25        0.5         0.75      ]]

Using custom prime bases::

    >>> print(create_hammersley_samples(order=3, dim=2, primes=[7]))
    [[ 0.16326531  0.30612245  0.44897959]
     [ 0.25        0.5         0.75      ]]
    >>> print(create_hammersley_samples(order=3, dim=3, primes=[7, 5]))
    [[ 0.16326531  0.30612245  0.44897959]
     [ 0.64        0.84        0.08      ]
     [ 0.25        0.5         0.75      ]]

.. Hammersley set: https://en.wikipedia.org/wiki/Low-discrepancy_sequence#Hammersley_set
"""
import numpy

from .halton import create_halton_samples


def create_hammersley_samples(order, dim=1, burnin=None, primes=None):
    """
    Create samples from the Hammersley set.

    For ``dim == 1`` the sequence falls back to Van Der Corput sequence.

    Args:
        order (int):
            The order of the Hammersley sequence. Defines the number of samples.
        dim (int):
            The number of dimensions in the Hammersley sequence.
        burnin (int, optional):
            Skip the first ``burnin`` samples. If omitted, the maximum of
            ``primes`` is used.
        primes (array_like, optional):
            The (non-)prime base to calculate values along each axis. If
            omitted, growing prime values starting from 2 will be used.

    Returns (numpy.ndarray):
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
