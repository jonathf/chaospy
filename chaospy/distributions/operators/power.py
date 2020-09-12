"""Power operator."""
import numpy
import chaospy

from ..baseclass import Distribution
from .operator import OperatorDistribution


class Pow(OperatorDistribution):
    """Power operator."""

    def __init__(self, left, right):
        """
        Constructor.

        Args:
            left (Distribution, numpy.ndarray) : Left hand side.
            right (Distribution, numpy.ndarray) : Right hand side.
        """
        super(Pow, self).__init__(
            left=left,
            right=right,
            repr_args=[left, right],
        )

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

        """
        # small hack.
        del cache
        left = self._parameters["left"]
        right = self._parameters["right"]
        if isinstance(left, Distribution):
            left_lower = numpy.asfarray(left._get_lower(cache={}))
            left_upper = numpy.asfarray(left._get_upper(cache={}))
            assert left_lower >= 0, "root of negative number"

            if isinstance(right, Distribution):
                right_lower = right._get_lower(cache={})
                right_upper = right._get_upper(cache={})

                return numpy.min(numpy.broadcast_arrays(
                    left_lower.T**right_lower.T,
                    left_lower.T**right_upper.T,
                    left_upper.T**right_lower.T,
                    left_upper.T**right_upper.T,
                ), axis=0).T

            return numpy.min([left_lower.T**right.T,
                              left_upper.T**right.T], axis=0).T

        assert isinstance(right, Distribution)
        right_lower = right._get_lower(cache={})
        right_upper = right._get_upper(cache={})
        return numpy.min([numpy.asfarray(left).T**right_lower.T,
                          numpy.asfarray(left).T**right_upper.T], axis=0).T

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

        """
        # small hack.
        del cache
        left = self._parameters["left"]
        right = self._parameters["right"]
        if isinstance(left, Distribution):
            left_lower = numpy.asfarray(left._get_lower(cache={}))
            left_upper = numpy.asfarray(left._get_upper(cache={}))
            assert left_lower >= 0, "root of negative number"

            if isinstance(right, Distribution):
                right_lower = right._get_lower(cache={})
                right_upper = right._get_upper(cache={})

                return numpy.max(numpy.broadcast_arrays(
                    left_lower.T**right_lower.T,
                    left_lower.T**right_upper.T,
                    left_upper.T**right_lower.T,
                    left_upper.T**right_upper.T,
                ), axis=0).T

            return numpy.max([left_lower.T**right.T,
                              left_upper.T**right.T], axis=0).T

        assert isinstance(right, Distribution)
        right_lower = right._get_lower(cache={})
        right_upper = right._get_upper(cache={})
        return numpy.max([numpy.asfarray(left).T**right_lower.T,
                          numpy.asfarray(left).T**right_upper.T], axis=0).T

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

        """
        if isinstance(left, Distribution):
            x_ = numpy.sign(xloc)*numpy.abs(xloc)**(1./right-1)
            xloc = numpy.sign(xloc)*numpy.abs(xloc)**(1./right)
            pairs = numpy.sign(xloc**right) == 1
            out = left._get_pdf(xloc, cache=cache.copy())
            if numpy.any(pairs):
                out = out+pairs*left._get_pdf(-xloc, cache=cache)
            out = numpy.sign(right)*out*x_/right
            out[numpy.isnan(out)] = numpy.inf

        else:
            assert numpy.all(left > 0), "imaginary result"
            x_ = numpy.where(xloc <= 0, -numpy.inf,
                             numpy.log(xloc + 1.*(xloc<=0))/numpy.log(left+1.*(left == 1)))
            num_ = numpy.log(left+1.*(left == 1))*xloc
            num_ = num_ + 1.*(num_==0)
            out = right._get_pdf(x_, cache=cache)/num_

        return out

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

        """
        if isinstance(left, Distribution):
            y = numpy.sign(xloc)*numpy.abs(xloc)**(1./right)
            pairs = numpy.sign(xloc**right) != -1
            out2 = left._get_fwd(-y, cache=cache.copy())
            out1 = left._get_fwd(y, cache=cache)
            out = numpy.where(right < 0, 1-out1, out1-pairs*out2)
        else:
            y = (numpy.log(numpy.abs(xloc)+1.*(xloc <= 0))/
                 numpy.log(numpy.abs(left)+1.*(left == 1)))
            out = right._get_fwd(y, cache=cache)
            out = numpy.where(xloc <= 0, 0., out)
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

        """
        if isinstance(left, Distribution):
            q = numpy.where(right.T < 0, 1-q.T, q.T).T
            out = (left._get_inv(q, cache=cache).T**right.T).T
        else:
            out = right._get_inv(q, cache=cache)
            out = numpy.where(left < 0, 1-out, out)
            out = (left.T**out.T).T
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

        """
        del cache
        if isinstance(right, Distribution):
            raise chaospy.UnsupportedFeature(
                "distribution as exponent not supported.")
        if numpy.any(right < 0):
            raise chaospy.UnsupportedFeature(
                "distribution to negative power not supported.")
        if not numpy.allclose(right, numpy.array(right, dtype=int)):
            raise chaospy.UnsupportedFeature(
                "distribution to fractional power not supported.")
        return left._get_mom(k*right)

    def _value(self, left, right, cache):
        if isinstance(left, Distribution) or isinstance(right, Distribution):
            return self
        return left**right
