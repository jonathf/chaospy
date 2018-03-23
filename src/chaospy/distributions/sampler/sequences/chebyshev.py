"""
Generate Chebyshev pseudo-random samples.

Example usage
-------------

Basic usage::

    >>> print(create_chebyshev_samples(order=2))
    [[0.25 0.75]]
    >>> print(numpy.around(create_chebyshev_samples(order=5), 4))
    [[0.067 0.25  0.5   0.75  0.933]]

Certain orders are nested::

    >>> print(numpy.around(create_chebyshev_samples(order=3), 4))
    [[0.1464 0.5    0.8536]]
    >>> print(numpy.around(create_chebyshev_samples(order=7), 4))
    [[0.0381 0.1464 0.3087 0.5    0.6913 0.8536 0.9619]]

Create nested samples directly with the dedicated function::

    >>> print(numpy.around(create_nested_chebyshev_samples(order=2), 4))
    [[0.1464 0.5    0.8536]]
    >>> print(numpy.around(create_nested_chebyshev_samples(order=3), 4))
    [[0.0381 0.1464 0.3087 0.5    0.6913 0.8536 0.9619]]

Multivariate usage::

    >>> print(numpy.around(create_chebyshev_samples(order=2, dim=2), 4))
    [[0.25 0.25 0.75 0.75]
     [0.25 0.75 0.25 0.75]]
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
