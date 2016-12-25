"""
Division.
"""
import numpy

from chaospy.dist import Dist
import chaospy.dist.operators

class Div(Dist):
    """Division."""

    def __init__(self, left, right):
        """
        Constructor.

        Args:
            left (Dist, array_like) : Left hand side.
            right (Dist, array_like) : Right hand side.
        """
        left_ = not isinstance(left, Dist) or 1 and len(left)
        right_ = not isinstance(right, Dist) or 1 and len(right)
        if isinstance(left, Dist):
            assert not numpy.any(numpy.prod(left.range(), 0) < 0)
        if isinstance(right, Dist):
            assert not numpy.any(numpy.prod(right.range(), 0) < 0)
        length = max(left_,right_)
        Dist.__init__(self, left=left, right=right,
                _length=length, _advance=True)

    def _pdf(self, xloc, graph):
        """Probability density function."""
        if "left" in graph.keys and "right" in graph.dists:
            num, dist = graph.keys["left"], graph.dists["right"]
            num = (num.T*numpy.ones(xloc.shape[::-1])).T

            p0 = graph.fwd_as_pdf(xloc*0, dist)
            xloc, num = numpy.where(xloc, xloc, 1), \
                    numpy.where(xloc, num, numpy.inf*numpy.sign(num))
            p1 = graph(num/xloc, dist)
            out = numpy.abs((1-2*p0)*p1*num/xloc**2)
        else:
            num, dist = graph.keys["right"], graph.dists["left"]
            num = (num.T*numpy.ones(xloc.T.shape)).T
            num = numpy.where(num, num, numpy.inf)
            out = numpy.abs(graph(xloc*1./num, dist)/num)
        return out

    def _cdf(self, xloc, graph):
        """Cumulative distribution function."""
        if "left" in graph.keys and "right" in graph.dists:
            num, dist = graph.keys["left"], graph.dists["right"]
            num = (num.T*numpy.ones(xloc.shape[::-1])).T
            p0 = graph.copy()(xloc*0, dist)
            xloc, num = numpy.where(xloc, xloc, 1), \
                    numpy.where(xloc, num, numpy.inf*numpy.sign(num))
            p1 = graph(num/xloc, dist)
            out = (1-p0)*(1-p1) + p0*p1
        else:
            num, dist = graph.keys["right"], graph.dists["left"]
            xloc = (xloc.T/(num.T + 1.*(num.T==0))).T
            out = graph(xloc, dist)
            out = (out.T*(num.T>0) + (1.-out.T)*(num.T<0) + \
                    1.*(xloc.T>num.T)*(num.T==0)).T
        return out

    def _ppf(self, q, graph):
        """Point percentile function."""
        if "left" in graph.keys and "right" in graph.dists:
            num, dist = graph.keys["left"], graph.dists["right"]
        else:
            num, dist = graph.keys["right"], graph.dists["left"]
            num = (num.T*numpy.ones(q.shape[::-1])).T
            q = (q*(num>0) + (1.-q)*(num<=0))
        return graph(q, dist)*num

    def _bnd(self, xloc, graph):
        """Distribution bounds."""
        if "left" in graph.keys and "right" in graph.dists:
            num, dist = graph.keys["left"], graph.dists["right"]
        else:
            num, dist = graph.keys["right"], graph.dists["left"]

        num = (num + 1.*(num==0))
        xloc = (xloc.T/num.T).T
        lo_,up_ = graph(xloc, dist)
        lower = (num.T*(lo_.T*(num.T>0) + up_.T*(num.T<0))).T
        upper = (num.T*(up_.T*(num.T>0) + lo_.T*(num.T<0))).T

        return lower,upper

    def _mom(self, k, graph):
        """Statistical moments."""
        if len(graph.dists)==2 and graph.dists["left"].dependent(graph.dists["right"]):
            raise NotImplementedError("dependency")

        left = []
        if "left" in graph.dists:  left.append(graph(k, graph.dists["left"]))
        else:           left.append((graph.keys["left"].T**k.T).T)
        if "right" in graph.dists:  left.append(graph(k, graph.dists["right"]))
        else:           left.append((graph.keys["right"].T**k.T).T)

        return numpy.prod(left[0]*left[1], 0)


    def _ttr(self, n, graph):
        """Three terms recursion coefficients."""
        if "left" in graph.keys and "right" in graph.dists:
            num, dist = graph.keys["left"], graph.dists["right"]
        else:
            num, dist = graph.keys["right"], graph.dists["left"]

        a,b = graph(n, dist)
        return a*num, b*num*num


    def _val(self, graph):
        """Value extraction."""
        if len(graph.keys)==2:
            return graph.keys["left"]*graph.keys["right"]
        return self

    def _str(self, left, right):
        """String representation."""
        return str(left) + "*" + str(right)


def div(left, right):
    """
    Distribution division.

    Args:
        left (Dist, array_like) : left hand side.
        right (Dist, array_like) : right hand side.
    """
    if left is right:
        return 1.

    if isinstance(left, Dist):
        if not isinstance(right, Dist):
            return chaospy.dist.operators.mul(left, 1./right)

    elif not isinstance(right, Dist):
        return left/right

    return Div(left, right)
