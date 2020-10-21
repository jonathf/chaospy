"""
Create samples from a Korobov lattice.

Examples usage
--------------

Normal usage::

    >>> distribution = chaospy.J(chaospy.Uniform(0, 1), chaospy.Uniform(0, 1))
    >>> samples = distribution.sample(4, rule="korobov")
    >>> samples.round(4)
    array([[0.2, 0.4, 0.6, 0.8],
           [0.4, 0.8, 0.2, 0.6]])
    >>> samples = distribution.sample(6, rule="korobov")
    >>> samples.round(4)
    array([[0.1429, 0.2857, 0.4286, 0.5714, 0.7143, 0.8571],
           [0.4286, 0.8571, 0.2857, 0.7143, 0.1429, 0.5714]])

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
