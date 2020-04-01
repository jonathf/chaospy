"""
Generate samples from a regular grid.

Example usage
-------------

Basic usage::

    >>> distribution = chaospy.Uniform(0, 1)
    >>> samples = distribution.sample(2, rule="grid")
    >>> print(numpy.around(samples, 4))
    [0.3333 0.6667]
    >>> samples = distribution.sample(5, rule="grid")
    >>> print(numpy.around(samples, 4))
    [0.1667 0.3333 0.5    0.6667 0.8333]

Certain orders are nested::

    >>> samples = distribution.sample(3, rule="grid")
    >>> print(numpy.around(samples, 4))
    [0.25 0.5  0.75]
    >>> samples = distribution.sample(7, rule="grid")
    >>> print(numpy.around(samples, 4))
    [0.125 0.25  0.375 0.5   0.625 0.75  0.875]

Create nested samples directly with the dedicated function::

    >>> samples = distribution.sample(2, rule="nested_grid")
    >>> print(numpy.around(samples, 4))
    [0.25 0.5  0.75]
    >>> samples = distribution.sample(3, rule="nested_grid")
    >>> print(numpy.around(samples, 4))
    [0.125 0.25  0.375 0.5   0.625 0.75  0.875]

Multivariate usage::

    >>> distribution = chaospy.J(chaospy.Uniform(0, 1), chaospy.Uniform(0, 1))
    >>> samples = distribution.sample(2, rule="grid")
    >>> print(numpy.around(samples, 4))
    [[0.3333 0.3333 0.6667 0.6667]
     [0.3333 0.6667 0.3333 0.6667]]
"""
import numpy
import chaospy


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
    x_data = chaospy.quadrature.combine([x_data]*dim)
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
