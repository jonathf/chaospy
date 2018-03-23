"""
Generate samples from a regular grid.

Example usage
-------------

Basic usage::

    >>> print(numpy.around(create_grid_samples(order=2), 4))
    [[0.3333 0.6667]]
    >>> print(numpy.around(create_grid_samples(order=5), 4))
    [[0.1667 0.3333 0.5    0.6667 0.8333]]

Certain orders are nested::

    >>> print(numpy.around(create_grid_samples(order=3), 4))
    [[0.25 0.5  0.75]]
    >>> print(numpy.around(create_grid_samples(order=7), 4))
    [[0.125 0.25  0.375 0.5   0.625 0.75  0.875]]

Create nested samples directly with the dedicated function::

    >>> print(numpy.around(create_nested_grid_samples(order=2), 4))
    [[0.25 0.5  0.75]]
    >>> print(numpy.around(create_nested_grid_samples(order=3), 4))
    [[0.125 0.25  0.375 0.5   0.625 0.75  0.875]]

Multivariate usage::

    >>> print(numpy.around(create_grid_samples(order=2, dim=2), 4))
    [[0.3333 0.3333 0.6667 0.6667]
     [0.3333 0.6667 0.3333 0.6667]]
"""
import numpy

import chaospy.quad


def create_grid_samples(order, dim=1):
    """
    Create samples from a regular grid.

    Args:
        order (int):
            The order of the grid. Defines the number of samples.
        dim (int):
            The number of dimensions in the grid

    Returns (numpy.ndarray):
        Regular grid with ``shape == (dim, order)``.
    """
    x_data = numpy.arange(1, order+1)/(order+1.)
    x_data = chaospy.quad.combine([x_data]*dim)
    return x_data.T


def create_nested_grid_samples(order, dim=1):
    """
    Create samples from a nested grid.

    Args:
        order (int):
            The order of the grid. Defines the number of samples.
        dim (int):
            The number of dimensions in the grid

    Returns (numpy.ndarray):
        Regular grid with ``shape == (dim, 2**order-1)``.
    """
    return create_grid_samples(order=2**order-1, dim=dim)
