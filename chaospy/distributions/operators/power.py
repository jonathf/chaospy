"""Power operator."""
import numpy

from ..baseclass import Dist, StochasticallyDependentError
from .. import evaluation


class Pow(Dist):
    """Power operator."""

    def __init__(self, left, right):
        """
        Constructor.

        Args:
            left (Dist, numpy.ndarray) : Left hand side.
            right (Dist, numpy.ndarray) : Right hand side.
        """
        Dist.__init__(self, left=left, right=right)

    def _lower(self, left, right, cache):
        """
        Distribution lower bounds.

        Example:
            >>> print(chaospy.Uniform().lower)
            [0.]
            >>> print(chaospy.Pow(chaospy.Uniform(), 2).lower)
            [0.]
            >>> print(chaospy.Pow(chaospy.Uniform(1, 2), -1).lower)
            [0.5]
            >>> print(chaospy.Pow(2, chaospy.Uniform()).lower)
            [1.]
            >>> print(chaospy.Pow(2, chaospy.Uniform(-1, 0)).lower)
            [0.5]
            >>> print(chaospy.Pow(2, 3).lower)
            [8.]
        """
        if isinstance(left, Dist):
            left_lower = evaluation.evaluate_lower(left, cache=cache)
            left_upper = evaluation.evaluate_upper(left, cache=cache)
            assert left_lower >= 0, "root of negative number"

            if isinstance(right, Dist):
                right_lower = evaluation.evaluate_lower(right, cache=cache)
                right_upper = evaluation.evaluate_upper(right, cache=cache)

                return numpy.min(numpy.broadcast_arrays(
                    left_lower**right_lower,
                    left_lower**right_upper,
                    left_upper**right_lower,
                    left_upper**right_upper,
                ), axis=0)

            return numpy.min([left_lower**right, left_upper**right], axis=0)

        elif not isinstance(right, Dist):
            return left**right

        right_lower = evaluation.evaluate_lower(right, cache=cache)
        right_upper = evaluation.evaluate_upper(right, cache=cache)
        return numpy.min([left**right_lower, left**right_upper], axis=0)

    def _upper(self, left, right, cache):
        """
        Distribution bounds.

        Example:
            >>> print(chaospy.Uniform().upper)
            [1.]
            >>> print(chaospy.Pow(chaospy.Uniform(), 2).upper)
            [1.]
            >>> print(chaospy.Pow(chaospy.Uniform(1, 2), -1).upper)
            [1.]
            >>> print(chaospy.Pow(2, chaospy.Uniform()).upper)
            [2.]
            >>> print(chaospy.Pow(2, chaospy.Uniform(-1, 0)).upper)
            [1.]
            >>> print(chaospy.Pow(2, 3).upper)
            [8.]
        """
        if isinstance(left, Dist):
            left_lower = evaluation.evaluate_lower(left, cache=cache)
            left_upper = evaluation.evaluate_upper(left, cache=cache)
            assert left_lower >= 0, "root of negative number"

            if isinstance(right, Dist):
                right_lower = evaluation.evaluate_lower(right, cache=cache)
                right_upper = evaluation.evaluate_upper(right, cache=cache)

                return numpy.max(numpy.broadcast_arrays(
                    left_lower**right_lower,
                    left_lower**right_upper,
                    left_upper**right_lower,
                    left_upper**right_upper,
                ), axis=0)

            return numpy.max([left_lower**right, left_upper**right], axis=0)

        elif not isinstance(right, Dist):
            return left**right

        right_lower = evaluation.evaluate_lower(right, cache=cache)
        right_upper = evaluation.evaluate_upper(right, cache=cache)
        return numpy.max([left**right_lower, left**right_upper], axis=0)

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
        left = evaluation.get_forward_cache(left, cache)
        right = evaluation.get_forward_cache(right, cache)

        if isinstance(left, Dist):
            if isinstance(right, Dist):
                raise StochasticallyDependentError(
                    "under-defined distribution {} or {}".format(left, right))

        elif not isinstance(right, Dist):
            return numpy.inf

        else:
            assert numpy.all(left > 0), "imaginary result"

            y = (numpy.log(numpy.abs(xloc) + 1.*(xloc <= 0)) /
                 numpy.log(numpy.abs(left)+1.*(left == 1)))

            out = evaluation.evaluate_forward(right, y, cache=cache.copy())
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
        left = evaluation.get_inverse_cache(left, cache)
        right = evaluation.get_inverse_cache(right, cache)

        if isinstance(left, Dist):
            if isinstance(right, Dist):
                raise StochasticallyDependentError(
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
        left = evaluation.get_forward_cache(left, cache)
        right = evaluation.get_forward_cache(right, cache)

        if isinstance(left, Dist):
            if isinstance(right, Dist):
                raise StochasticallyDependentError(
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
            [1.     0.6931 0.5    0.375 ]
            >>> print(numpy.around(chaospy.Pow(2, chaospy.Uniform()).mom([0, 1, 2, 3]), 4))
            [1.     1.4427 2.164  3.3663]
            >>> print(numpy.around(chaospy.Pow(2, chaospy.Uniform(-1, 0)).mom([0, 1, 2, 3]), 4))
            [1.     0.7213 0.541  0.4208]
            >>> print(numpy.around(chaospy.Pow(2, 1).mom([0, 1, 2, 3]), 4))
            [1. 2. 4. 8.]
        """
        if isinstance(right, Dist):
            raise StochasticallyDependentError(
                "distribution as exponent not supported.")
        if not isinstance(left, Dist):
            return left**(right*k)
        if numpy.any(right < 0):
            raise StochasticallyDependentError(
                "distribution to negative power not supported.")
        if not numpy.allclose(right, numpy.array(right, dtype=int)):
            raise StochasticallyDependentError(
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

    def _fwd_cache(self, cache):
        left = evaluation.get_forward_cache(self.prm["left"], cache)
        right = evaluation.get_forward_cache(self.prm["right"], cache)
        if not isinstance(left, Dist) and not isinstance(right, Dist):
            return left**right
        return self

    def _inv_cache(self, cache):
        left = evaluation.get_inverse_cache(self.prm["left"], cache)
        right = evaluation.get_inverse_cache(self.prm["right"], cache)
        if not isinstance(left, Dist) and not isinstance(right, Dist):
            return left**right
        return self
