"""
Power operator.
"""
import numpy

from chaospy.dist import Dist


class Pow(Dist):
    """Power operator."""

    def __init__(self, left, right):
        """
        Constructor.

        Args:
            left (Dist, array_like) : Left hand side.
            right (Dist, array_like) : Right hand side.
        """
        left_ = 1 if not isinstance(left, Dist) else len(left)
        right_ = 1 if not isinstance(right, Dist) else len(right)
        length = max(left_,right_)
        Dist.__init__(self, left=left, right=right,
                _length=length, _advance=True)

    def _bnd(self, xloc, graph):
        """Distribution bounds."""
        if "left" in graph.keys and "right" in graph.dists:

            num, dist = graph.keys["left"], graph.dists["right"]

            assert numpy.all(num>=0), "root of negative number"

            y = numpy.where(xloc<0, -numpy.inf,
                    numpy.log(numpy.where(xloc>0, xloc, 1)))/\
                        numpy.log(num*(1.-(num==1)))

            bnd = num**graph(y, dist)
            correct = bnd[0]<bnd[1]
            bnd_ = numpy.empty(bnd.shape)
            bnd_[0] = numpy.where(correct, bnd[0], bnd[1])
            bnd_[1] = numpy.where(correct, bnd[1], bnd[0])

        else:

            num, dist = graph.keys["right"], graph.dists["left"]
            y = numpy.sign(xloc)*numpy.abs(xloc)**(1./num)
            y[(xloc == 0.)*(num < 0)] = numpy.inf

            bnd = graph(y, dist)
            assert numpy.all((num % 1 == 0) + (bnd[0] >= 0)), \
                    "root of negative number"

            pair = num % 2 == 0
            bnd_ = numpy.empty(bnd.shape)
            bnd_[0] = numpy.where(pair*(bnd[0]*bnd[1]<0), 0, bnd[0])
            bnd_[0] = numpy.where(pair*(bnd[0]*bnd[1]>0), \
                    numpy.min(numpy.abs(bnd), 0), bnd_[0])**num
            bnd_[1] = numpy.where(pair, numpy.max(numpy.abs(bnd), 0),
                    bnd[1])**num

            bnd_[0], bnd_[1] = numpy.where(
                bnd_[0] < bnd_[1], bnd_[0], bnd_[1]
            ), numpy.where(
                bnd_[0] < bnd_[1], bnd_[1], bnd_[0]
            )

        return bnd_


    def _cdf(self, xloc, graph):
        """Cumulative distribution function."""
        if "left" in graph.keys and "right" in graph.dists:

            num, dist = graph.keys["left"], graph.dists["right"]
            assert numpy.all(num>0), "imaginary result"

            y = numpy.log(numpy.abs(xloc) + 1.*(xloc<=0))/\
                    numpy.log(numpy.abs(num)+1.*(num == 1))

            out = graph(y, dist)
            out = numpy.where(xloc<=0, 0., out)

        else:

            num, dist = graph.keys["right"], graph.dists["left"]
            y = numpy.sign(xloc)*numpy.abs(xloc)**(1./num)
            pairs = numpy.sign(xloc**num) != -1

            _1 = graph.copy()(-y, dist)
            out = graph(y, dist)
            out = numpy.where(num < 0, 1-out, out - pairs*_1)

        return out

    def _ppf(self, q, graph):
        """Point percentile function."""
        if "left" in graph.keys and "right" in graph.dists:
            num, dist = graph.keys["left"], graph.dists["right"]
            out = num**graph(q, dist)
        else:
            num, dist = graph.keys["right"], graph.dists["left"]
            out = graph(q, dist)**num
        return out

    def _pdf(self, xloc, graph):
        """Probability density function."""
        if "left" in graph.keys and "right" in graph.dists:

            num, dist = graph.keys["left"], graph.dists["right"]
            assert numpy.all(num>0), "imaginary result"
            x_ = numpy.where(xloc<=0, -numpy.inf,
                    numpy.log(xloc + 1.*(xloc<=0))/numpy.log(num+1.*(num==1)))
            num_ = numpy.log(num+1.*(num==1))*xloc
            num_ = num_ + 1.*(num_==0)

            out = graph(x_, dist)/num_

        else:

            num, dist = graph.keys["right"], graph.dists["left"]
            x_ = numpy.sign(xloc)*numpy.abs(xloc)**(1./num -1)
            xloc = numpy.sign(xloc)*numpy.abs(xloc)**(1./num)
            pairs = numpy.sign(xloc**num) == 1

            G_ = graph.copy()
            out = graph(xloc, dist)
            if numpy.any(pairs):
                out = out + pairs*G_(-xloc, dist)
            out = numpy.sign(num)*out * x_ / num
            out[numpy.isnan(out)] = numpy.inf

        return out

    def _mom(self, k, graph):
        """Statistical moments."""
        if "right" in graph.keys and not numpy.any(graph.keys["right"] % 1):
            out = graph(k*numpy.array(graph.keys["right"], dtype=int), graph.dists["left"])
        else:
            raise NotImplementedError()
        return out

    def _val(self, graph):
        """Value extraction."""
        if len(graph.keys)==2:
            return graph.keys["left"]**graph.keys["right"]
        return self

    def _str(self, left, right):
        """String representation."""
        return "(%s)**(%s)" % (left,right)


def pow(left, right):
    """
    Power operator.

    Args:
        left (Dist, array_like) : Left hand side.
        right (Dist, array_like) : Right hand side.
    """
    if isinstance(left, Dist):

        if not isinstance(right, Dist):
            right = numpy.array(right)
            if right.size==1 and right==1:
                return left

    elif not isinstance(right, Dist):
        return left+right

    return Pow(left=left, right=right)
