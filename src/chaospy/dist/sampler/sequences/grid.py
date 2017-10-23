"""
Generate samples from a regular grid.

Example usage
-------------

Basic usage::

    >>> print(create_grid_samples(order=1))
    [[ 0.5]]
    >>> print(create_grid_samples(order=2))
    [[ 0.33333333  0.66666667]]
    >>> print(create_grid_samples(order=5))
    [[ 0.16666667  0.33333333  0.5         0.66666667  0.83333333]]

Certain orders are nested::

    >>> print(create_grid_samples(order=3))
    [[ 0.25  0.5   0.75]]
    >>> print(create_grid_samples(order=7))
    [[ 0.125  0.25   0.375  0.5    0.625  0.75   0.875]]

Create nested samples directly with the dedicated function::

    >>> print(create_nested_grid_samples(order=1))
    [[ 0.5]]
    >>> print(create_nested_grid_samples(order=2))
    [[ 0.25  0.5   0.75]]
    >>> print(create_nested_grid_samples(order=3))
    [[ 0.125  0.25   0.375  0.5    0.625  0.75   0.875]]

Multivariate usage::

    >>> print(create_grid_samples(order=2, dim=2))
    [[ 0.33333333  0.33333333  0.66666667  0.66666667]
     [ 0.33333333  0.66666667  0.33333333  0.66666667]]
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
