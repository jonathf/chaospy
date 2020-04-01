"""
Generate Chebyshev pseudo-random samples.

Example usage
-------------

Basic usage::

    >>> distribution = chaospy.Uniform(0, 1)
    >>> samples = distribution.sample(2, rule="chebyshev")
    >>> print(numpy.around(samples, 4))
    [0.25 0.75]
    >>> samples = distribution.sample(5, rule="chebyshev")
    >>> print(numpy.around(samples, 4))
    [0.067 0.25  0.5   0.75  0.933]

Certain orders are nested::

    >>> samples = distribution.sample(3, rule="chebyshev")
    >>> print(numpy.around(samples, 4))
    [0.1464 0.5    0.8536]
    >>> samples = distribution.sample(7, rule="chebyshev")
    >>> print(numpy.around(samples, 4))
    [0.0381 0.1464 0.3087 0.5    0.6913 0.8536 0.9619]

Create nested samples directly with the dedicated function::

    >>> samples = distribution.sample(2, rule="nested_chebyshev")
    >>> print(numpy.around(samples, 4))
    [0.1464 0.5    0.8536]
    >>> samples = distribution.sample(3, rule="nested_chebyshev")
    >>> print(numpy.around(samples, 4))
    [0.0381 0.1464 0.3087 0.5    0.6913 0.8536 0.9619]

Multivariate usage::

    >>> distribution = chaospy.J(chaospy.Uniform(0, 1), chaospy.Uniform(0, 1))
    >>> samples = distribution.sample(2, rule="chebyshev")
    >>> print(numpy.around(samples, 4))
    [[0.25 0.25 0.75 0.75]
     [0.25 0.75 0.25 0.75]]
"""
import numpy
import chaospy


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
    x_data = chaospy.quadrature.combine([x_data]*dim)
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
