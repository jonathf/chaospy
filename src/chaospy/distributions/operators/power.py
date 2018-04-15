"""Power operator."""
import numpy

from ..baseclass import Dist
from .. import evaluation


class Pow(Dist):
    """Power operator."""

    def __init__(self, left, right):
        """
        Constructor.

        Args:
            left (Dist, array_like) : Left hand side.
            right (Dist, array_like) : Right hand side.
        """
        Dist.__init__(self, left=left, right=right)

    def _bnd(self, xloc, left, right, cache):
        """
        Distribution bounds.

        Example:
            >>> print(chaospy.Uniform().range([-2, 0, 2, 4]))
            [[0. 0. 0. 0.]
             [1. 1. 1. 1.]]
            >>> print(chaospy.Pow(chaospy.Uniform(), 2).range([-2, 0, 2, 4]))
            [[0. 0. 0. 0.]
             [1. 1. 1. 1.]]
            >>> print(chaospy.Pow(chaospy.Uniform(1, 2), -1).range([-2, 0, 2, 4]))
            [[0.5 0.5 0.5 0.5]
             [1.  1.  1.  1. ]]
            >>> print(chaospy.Pow(2, chaospy.Uniform()).range([-2, 0, 2, 4]))
            [[1. 1. 1. 1.]
             [2. 2. 2. 2.]]
            >>> print(chaospy.Pow(2, chaospy.Uniform(-1, 0)).range([-2, 0, 2, 4]))
            [[0.5 0.5 0.5 0.5]
             [1.  1.  1.  1. ]]
            >>> print(chaospy.Pow(2, 3).range([-2, 0, 2, 4]))
            [[8. 8. 8. 8.]
             [8. 8. 8. 8.]]
        """
        if isinstance(left, Dist) and left in cache:
            left = cache[left]
        if isinstance(right, Dist) and right in cache:
            right = cache[right]

        if isinstance(left, Dist):
            if isinstance(right, Dist):
                raise evaluation.DependencyError(
                    "under-defined distribution {} or {}".format(left, right))
        elif not isinstance(right, Dist):
            return left**right, left**right
        else:
            output = numpy.ones(xloc.shape)
            left = left * output
            assert numpy.all(left >= 0), "root of negative number"

            indices = xloc > 0
            output[indices] = numpy.log(xloc[indices])
            output[~indices] = -numpy.inf

            indices = left != 1
            output[indices] /= numpy.log(left[indices])

            output = evaluation.evaluate_bound(right, output, cache=cache)
            output = left**output
            output[:] = (
                numpy.where(output[0] < output[1], output[0], output[1]),
                numpy.where(output[0] < output[1], output[1], output[0]),
            )
            return output

        output = numpy.zeros(xloc.shape)
        right = right + output

        indices = right > 0
        output[indices] = numpy.abs(xloc[indices])**(1/right[indices])
        output[indices] *= numpy.sign(xloc[indices])
        output[right == 0] = 1
        output[(xloc == 0) & (right < 0)] = numpy.inf

        output = evaluation.evaluate_bound(left, output, cache=cache)

        pair = right % 2 == 0
        bnd_ = numpy.empty(output.shape)
        bnd_[0] = numpy.where(pair*(output[0]*output[1] < 0), 0, output[0])
        bnd_[0] = numpy.where(pair*(output[0]*output[1] > 0), \
                numpy.min(numpy.abs(output), 0), bnd_[0])**right
        bnd_[1] = numpy.where(pair, numpy.max(numpy.abs(output), 0),
                output[1])**right

        bnd_[0], bnd_[1] = (
            numpy.where(bnd_[0] < bnd_[1], bnd_[0], bnd_[1]),
            numpy.where(bnd_[0] < bnd_[1], bnd_[1], bnd_[0]),
        )
        return bnd_


    def _cdf(self, xloc, left, right, cache):
        """
        Cumulative distribution function.

        Example:
            >>> print(chaospy.Uniform().fwd([-0.5, 0.5, 1.5, 2.5]))
            [0.  0.5 1.  1. ]
            >>> print(chaospy.Pow(chaospy.Uniform(), 2).fwd([-0.5, 0.5, 1.5, 2.5]))
            [0.         0.70710678 1.         1.        ]
            >>> print(chaospy.Pow(chaospy.Uniform(1, 2), -1).fwd([0.4, 0.6, 0.8, 1.2]))
            [0.         0.33333333 0.75       1.        ]
            >>> print(chaospy.Pow(2, chaospy.Uniform()).fwd([-0.5, 0.5, 1.5, 2.5]))
            [0.        0.        0.5849625 1.       ]
            >>> print(chaospy.Pow(2, chaospy.Uniform(-1, 0)).fwd([0.4, 0.6, 0.8, 1.2]))
            [0.         0.26303441 0.67807191 1.        ]
            >>> print(chaospy.Pow(2, 3).fwd([7, 8, 9]))
            [0. 1. 1.]
        """
        if isinstance(left, Dist) and left in cache:
            left = cache[left]
        if isinstance(right, Dist) and right in cache:
            right = cache[right]

        if isinstance(left, Dist):
            if isinstance(right, Dist):
                raise evaluation.DependencyError(
                    "under-defined distribution {} or {}".format(left, right))

        elif not isinstance(right, Dist):
            return numpy.inf

        else:
            assert numpy.all(left > 0), "imaginary result"

            y = (numpy.log(numpy.abs(xloc) + 1.*(xloc <= 0)) /
                 numpy.log(numpy.abs(left)+1.*(left == 1)))

            out = evaluation.evaluate_forward(right, y)
            out = numpy.where(xloc <= 0, 0., out)
            return out

        y = numpy.sign(xloc)*numpy.abs(xloc)**(1./right)
        pairs = numpy.sign(xloc**right) != -1

        out1, out2 = (
            evaluation.evaluate_forward(left, y, cache=cache),
            evaluation.evaluate_forward(left, -y, cache=cache),
        )
        out = numpy.where(right < 0, 1-out1, out1-pairs*out2)
        return out

    def _ppf(self, q, left, right, cache):
        """
        Point percentile function.

        Example:
            >>> print(chaospy.Uniform().inv([0.1, 0.2, 0.9]))
            [0.1 0.2 0.9]
            >>> print(chaospy.Pow(chaospy.Uniform(), 2).inv([0.1, 0.2, 0.9]))
            [0.01 0.04 0.81]
            >>> print(chaospy.Pow(chaospy.Uniform(1, 2), -1).inv([0.1, 0.2, 0.9]))
            [0.52631579 0.55555556 0.90909091]
            >>> print(chaospy.Pow(2, chaospy.Uniform()).inv([0.1, 0.2, 0.9]))
            [1.07177346 1.14869835 1.86606598]
            >>> print(chaospy.Pow(2, chaospy.Uniform(-1, 0)).inv([0.1, 0.2, 0.9]))
            [0.53588673 0.57434918 0.93303299]
            >>> print(chaospy.Pow(2, 3).inv([0.1, 0.2, 0.9]))
            [8. 8. 8.]
        """
        if isinstance(left, Dist) and left in cache:
            left = cache[left]
        if isinstance(right, Dist) and right in cache:
            right = cache[right]

        if isinstance(left, Dist):
            if isinstance(right, Dist):
                raise evaluation.DependencyError(
                    "under-defined distribution {} or {}".format(left, right))
        elif not isinstance(right, Dist):
            return left**right

        else:
            out = evaluation.evaluate_inverse(right, q, cache=cache)
            out = numpy.where(left < 0, 1-out, out)
            out = left**out
            return out

        right = right + numpy.zeros(q.shape)
        q = numpy.where(right < 0, 1-q, q)
        out = evaluation.evaluate_inverse(left, q, cache=cache)**right
        return out


    def _pdf(self, xloc, left, right, cache):
        """
        Probability density function.

        Example:
            >>> print(chaospy.Uniform().pdf([-0.5, 0.5, 1.5, 2.5]))
            [0. 1. 0. 0.]
            >>> print(chaospy.Pow(chaospy.Uniform(), 2).pdf([-0.5, 0.5, 1.5, 2.5]))
            [0.         0.70710678 0.         0.        ]
            >>> print(chaospy.Pow(chaospy.Uniform(1, 2), -1).pdf([0.4, 0.6, 0.8, 1.2]))
            [0.         2.77777778 1.5625     0.        ]
            >>> print(chaospy.Pow(2, chaospy.Uniform()).pdf([-0.5, 0.5, 1.5, 2.5]))
            [0.         0.         0.96179669 0.        ]
            >>> print(chaospy.Pow(2, chaospy.Uniform(-1, 0)).pdf([0.4, 0.6, 0.8, 1.2]))
            [0.         2.40449173 1.8033688  0.        ]
            >>> print(chaospy.Pow(2, 3).pdf([7, 8, 9]))
            [ 0. inf  0.]
        """
        if isinstance(left, Dist) and left in cache:
            left = cache[left]
        if isinstance(right, Dist) and right in cache:
            right = cache[right]

        if isinstance(left, Dist):
            if isinstance(right, Dist):
                raise evaluation.DependencyError(
                    "under-defined distribution {} or {}".format(left, right))

        elif not isinstance(right, Dist):
            return numpy.inf

        else:

            assert numpy.all(left > 0), "imaginary result"
            x_ = numpy.where(xloc <= 0, -numpy.inf,
                    numpy.log(xloc + 1.*(xloc<=0))/numpy.log(left+1.*(left == 1)))
            num_ = numpy.log(left+1.*(left == 1))*xloc
            num_ = num_ + 1.*(num_==0)

            out = evaluation.evaluate_density(right, x_, cache=cache)/num_
            return out

        x_ = numpy.sign(xloc)*numpy.abs(xloc)**(1./right -1)
        xloc = numpy.sign(xloc)*numpy.abs(xloc)**(1./right)
        pairs = numpy.sign(xloc**right) == 1

        out = evaluation.evaluate_density(left, xloc, cache=cache)
        if numpy.any(pairs):
            out = out + pairs*evaluation.evaluate_density(left, -xloc, cache=cache)

        out = numpy.sign(right)*out * x_ / right
        out[numpy.isnan(out)] = numpy.inf

        return out

    def _mom(self, k, left, right, cache):
        """
        Statistical moments.

        Example:
            >>> print(numpy.around(chaospy.Uniform().mom([0, 1, 2, 3]), 4))
            [1.     0.5    0.3333 0.25  ]
            >>> print(numpy.around(chaospy.Pow(chaospy.Uniform(), 2).mom([0, 1, 2, 3]), 4))
            [1.     0.3333 0.2    0.1429]
            >>> print(numpy.around(chaospy.Pow(chaospy.Uniform(1, 2), -1).mom([0, 1, 2, 3]), 4))
            [1.     0.75   0.5833 0.4688]
            >>> print(numpy.around(chaospy.Pow(2, chaospy.Uniform()).mom([0, 1, 2, 3]), 4))
            [1.     1.5    2.3333 3.75  ]
            >>> print(numpy.around(chaospy.Pow(2, chaospy.Uniform(-1, 0)).mom([0, 1, 2, 3]), 4))
            [1.     0.75   0.5833 0.4688]
            >>> print(numpy.around(chaospy.Pow(2, 1).mom([0, 1, 2, 3]), 4))
            [1. 2. 4. 8.]
        """
        if isinstance(right, Dist):
            raise evaluation.DependencyError(
                "distribution as exponent not supported.")
        if not isinstance(left, Dist):
            return left**(right*k)
        if numpy.any(right < 0):
            raise evaluation.DependencyError(
                "distribution to negative power not supported.")
        if not numpy.allclose(right, numpy.array(right, dtype=int)):
            raise evaluation.DependencyError(
                "distribution to fractional power not supported.")
        return evaluation.evaluate_moment(left, k*right, cache=cache)

    def __str__(self):
        """
        Example:
            >>> print(chaospy.Pow(chaospy.Uniform(), 2))
            Pow(Uniform(lower=0, upper=1), 2)
        """
        return (self.__class__.__name__ + "(" + str(self.prm["left"]) +
                ", " + str(self.prm["right"]) + ")")
