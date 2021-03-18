"""
Generate Chebyshev pseudo-random samples.

Example usage
-------------

Basic usage::

    >>> distribution = chaospy.Uniform(0, 1)
    >>> samples = distribution.sample(2, rule="chebyshev")
    >>> samples.round(4)
    array([0.25, 0.75])
    >>> samples = distribution.sample(5, rule="chebyshev")
    >>> samples.round(4)
    array([0.067, 0.25 , 0.5  , 0.75 , 0.933])

Certain orders are nested::

    >>> samples = distribution.sample(3, rule="chebyshev")
    >>> samples.round(4)
    array([0.1464, 0.5   , 0.8536])
    >>> samples = distribution.sample(7, rule="chebyshev")
    >>> samples.round(4)
    array([0.0381, 0.1464, 0.3087, 0.5   , 0.6913, 0.8536, 0.9619])

Create nested samples directly with the dedicated function::

    >>> samples = distribution.sample(2, rule="nested_chebyshev")
    >>> samples.round(4)
    array([0.1464, 0.5   , 0.8536])
    >>> samples = distribution.sample(3, rule="nested_chebyshev")
    >>> samples.round(4)
    array([0.0381, 0.1464, 0.3087, 0.5   , 0.6913, 0.8536, 0.9619])

Multivariate usage::

    >>> distribution = chaospy.J(chaospy.Uniform(0, 1), chaospy.Uniform(0, 1))
    >>> samples = distribution.sample(2, rule="chebyshev")
    >>> samples.round(4)
    array([[0.25, 0.25, 0.75, 0.75],
           [0.25, 0.75, 0.25, 0.75]])

"""
import numpy

import chaospy
from chaospy.quadrature import utils


def create_chebyshev_samples(order, dim=1):
    """
    Generate Chebyshev pseudo-random samples.

    Args:
        order (int):
            The number of samples to create along each axis.
        dim (int):
            The number of dimensions to create samples for.

    Returns:
        samples following Chebyshev sampling scheme mapped to the
        ``[0, 1]^dim`` hyper-cube and ``shape == (dim, order)``.

    Examples:
        >>> samples = chaospy.create_chebyshev_samples(6, 1)
        >>> samples.round(4)
        array([[0.0495, 0.1883, 0.3887, 0.6113, 0.8117, 0.9505]])
        >>> samples = chaospy.create_chebyshev_samples(3, 2)
        >>> samples.round(3)
        array([[0.146, 0.146, 0.146, 0.5  , 0.5  , 0.5  , 0.854, 0.854, 0.854],
               [0.146, 0.5  , 0.854, 0.146, 0.5  , 0.854, 0.146, 0.5  , 0.854]])

    """
    x_data = .5*numpy.cos(numpy.arange(order, 0, -1)*numpy.pi/(order+1)) + .5
    x_data = utils.combine([x_data]*dim)
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
