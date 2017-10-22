"""
Generate Chebyshev pseudo-random samples.

Example usage
-------------

Basic usage::

    >>> print(create_chebyshev_samples(order=1))
    [[ 0.5]]
    >>> print(create_chebyshev_samples(order=2))
    [[ 0.25  0.75]]
    >>> print(create_chebyshev_samples(order=5))
    [[ 0.0669873  0.25       0.5        0.75       0.9330127]]

Certain orders are nested::

    >>> print(create_chebyshev_samples(order=3))
    [[ 0.14644661  0.5         0.85355339]]
    >>> print(create_chebyshev_samples(order=7))
    [[ 0.03806023  0.14644661  0.30865828  0.5         0.69134172  0.85355339
       0.96193977]]

Create nested samples directly with the dedicated function::

    >>> print(create_nested_chebyshev_samples(order=1))
    [[ 0.5]]
    >>> print(create_nested_chebyshev_samples(order=2))
    [[ 0.14644661  0.5         0.85355339]]
    >>> print(create_nested_chebyshev_samples(order=3))
    [[ 0.03806023  0.14644661  0.30865828  0.5         0.69134172  0.85355339
       0.96193977]]

Multivariate usage::

    >>> print(create_chebyshev_samples(order=2, dim=2))
    [[ 0.25  0.25  0.75  0.75]
     [ 0.25  0.75  0.25  0.75]]
"""
import numpy

import chaospy.quad


def create_chebyshev_samples(order, dim=1):
    """
    Chebyshev sampling function.

    Args:
        order (int):
            The number of samples to create along each axis.
        dim (int):
            The number of dimensions to create samples for.

    Returns:
        samples following Chebyshev sampling scheme mapped to the
        ``[0, 1]^dim`` hyper-cube and ``shape == (dim, order)``.
    """
    x_data = .5*numpy.cos(numpy.arange(order, 0, -1)*numpy.pi/(order+1)) + .5
    x_data = chaospy.quad.combine([x_data]*dim)
    return x_data.T


def create_nested_chebyshev_samples(order, dim=1):
    """
    Nested Chebyshev sampling function.

    Args:
        order (int):
            The number of samples to create along each axis.
        dim (int):
            The number of dimensions to create samples for.

    Returns:
        samples following Chebyshev sampling scheme mapped to the
        ``[0, 1]^dim`` hyper-cube and ``shape == (dim, 2**order-1)``.
    """
    return create_chebyshev_samples(order=2**order-1, dim=dim)
