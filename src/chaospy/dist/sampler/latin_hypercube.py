"""
Create Latin Hyper-cube samples.

Example usage
-------------

Normal usage::
    >>> print(chaospy.create_latin_hypercube_samples(order=4, dim=2))
    [[ 0.6633974   0.27875174  0.98757072  0.12054785]
     [ 0.46811863  0.05308317  0.51017741  0.84929862]]
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
