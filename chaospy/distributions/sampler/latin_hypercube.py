"""
Create Latin Hyper-cube samples.

Example usage
-------------

Normal usage::
    >>> chaospy.create_latin_hypercube_samples(order=4, dim=2).round(4)
    array([[0.6634, 0.2788, 0.9876, 0.1205],
           [0.4681, 0.0531, 0.5102, 0.8493]])
"""
import numpy


def create_latin_hypercube_samples(order, dim=1):
    """
    Latin Hypercube sampling.

    Args:
        order (int):
            The order of the latin hyper-cube. Defines the number of samples.
        dim (int):
            The number of dimensions in the latin hyper-cube.

    Returns (numpy.ndarray):
        Latin hyper-cube with ``shape == (dim, order)``.
    """
    randoms = numpy.random.random(order*dim).reshape((dim, order))
    for dim_ in range(dim):
        perm = numpy.random.permutation(order)  # pylint: disable=no-member
        randoms[dim_] = (perm + randoms[dim_])/order
    return randoms
