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
from .joint import J


class Iid(J):
    """
    Opaque method for creating independent identical distributed random
    variables from an univariate variable.

    Args:
        dist (Dist):
            Distribution to make into i.i.d. vector.
    """

    def __init__(self, dist, length):
        assert len(dist) == 1 and length >= 1
        J.__init__(self, *[deepcopy(dist) for _ in range(length)])

    def __str__(self):
        return (self.__class__.__name__ + "(" + str(self.prm["_000"]) +
                ", " + str(len(self)) + ")")
