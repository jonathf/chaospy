"""
Create samples from a Korobov lattice.

Examples usage
--------------

Normal usage::

    >>> print(create_korobov_samples(order=4, dim=2))
    [[0.2 0.4 0.6 0.8]
     [0.4 0.8 0.2 0.6]]

With custom number base::

    >>> print(create_korobov_samples(order=4, dim=2, base=3))
    [[0.2 0.4 0.6 0.8]
     [0.6 0.2 0.8 0.4]]
"""
import numpy


def create_korobov_samples(order, dim, base=17797):
    """
    Create Korobov lattice samples.

    Args:
        order (int):
            The order of the Korobov latice. Defines the number of
            samples.
        dim (int):
            The number of dimensions in the output.
        base (int):
            The number based used to calculate the distribution of values.

    Returns (numpy.ndarray):
        Korobov lattice with ``shape == (dim, order)``
    """
    values = numpy.empty(dim)
    values[0] = 1
    for idx in range(1, dim):
        values[idx] = base*values[idx-1] % (order+1)

    grid = numpy.mgrid[:dim, :order+1]
    out = values[grid[0]] * (grid[1]+1) / (order+1.) % 1.
    return out[:, :order]
