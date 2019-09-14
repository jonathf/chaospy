"""
Constructing a multivariate probability distribution consisting of identical
independent distributed marginals can be done using the
:func:`~chaospy.distributions.operators.iid.Iid`. E.g.::

    >>> X = chaospy.Normal()
    >>> Y = chaospy.Iid(X, 4)
    >>> print(Y.sample())
    [ 0.39502989 -1.20032309  1.64760248 -0.04465437]
"""
from copy import deepcopy
import numpy

from .joint import J


class Iid(J):
    """
    Opaque method for creating independent identical distributed random
    variables from an univariate variable.

    Args:
        dist (Dist):
            Distribution to make into i.i.d. vector.
        length (int):
            The number of samples.
    """

    def __init__(self, dist, length):
        self._repr = {"dist": dist, "length": length}
        J.__init__(self, *[deepcopy(dist) for _ in range(length)])
