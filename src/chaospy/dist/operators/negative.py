"""
Negative of a distribution.

Example usage
-------------

Invert sign of a distribution::

    >>> distribution = -chaospy.Uniform(0, 1)
    >>> print(distribution)
    (-Uniform(0,1))
    >>> print(distribution.sample(5))
    [-0.34641041 -0.88499306 -0.04971714 -0.5178086  -0.12752546]
    >>> print(distribution.fwd([-0.3, -0.2, -0.1]))
    [ 0.7  0.8  0.9]
    >>> print(distribution.inv(distribution.fwd([-0.3, -0.2, -0.1])))
    [-0.3 -0.2 -0.1]
    >>> print(distribution.pdf([-0.3, -0.2, -0.1]))
    [ 1.  1.  1.]
    >>> print(distribution.mom([1, 2, 3]))
    [-0.5         0.33333333 -0.25      ]
    >>> print(distribution.ttr([1, 2, 3]))
    [[-0.5        -0.5        -0.5       ]
     [ 0.08333333  0.06666667  0.06428571]]

"""
import numpy
from chaospy.dist import Dist

class Neg(Dist):
    """Negative of a distribution."""

    def __init__(self, left):
        """
        Constructor.

        Args:
            left (Dist) : distribution.
        """
        Dist.__init__(self, left=left,
                _length=len(left), _advance=True)

    def _bnd(self, xloc, graph):
        """Distribution bounds."""
        return -graph(-xloc, graph.dists["left"])[::-1]

    def _pdf(self, xloc, graph):
        """Probability density function."""
        return graph(-xloc, graph.dists["left"])

    def _cdf(self, xloc, graph):
        """Cumulative distribution function."""
        return 1-graph(-xloc, graph.dists["left"])

    def _ppf(self, q, graph):
        """Point percentile function."""
        return -graph(1-q, graph.dists["left"])

    def _mom(self, k, graph):
        """Statistical moments."""
        return (-1)**numpy.sum(k, 0)*graph(k, graph.dists["left"])

    def _ttr(self, k, graph):
        """Three terms recursion coefficients."""
        a,b = graph(k, graph.dists["left"])
        return -a,b

    def _str(self, left):
        """String representation."""
        return "(-" + str(left) + ")"

    def _val(self, graph):
        """Value extraction."""
        if "left" in graph.keys:
            return -graph.keys["left"]
        return self

def neg(left):
    """
    Negative of a distribution.

    Args:
        dist (Dist) : distribution.
    """
    if not isinstance(left, Dist):
        return -left
    return Neg(left)

