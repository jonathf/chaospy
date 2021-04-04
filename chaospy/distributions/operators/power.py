"""Power operator."""
import numpy
import chaospy

from ..baseclass import Distribution, OperatorDistribution


class Power(OperatorDistribution):
    """Power operator."""

    _operator = lambda self, left, right: left**right

    def __init__(self, left, right):
        """
        Constructor.

        Args:
            left (Distribution, numpy.ndarray) : Left hand side.
            right (Distribution, numpy.ndarray) : Right hand side.
        """
        super(Power, self).__init__(
            left=left,
            right=right,
            repr_args=[left, right],
        )

    def _lower(self, idx, left, right, cache):
        """
        Distribution lower bounds.

        Example:
            >>> chaospy.Uniform().lower
            array([0.])
            >>> chaospy.Power(chaospy.Uniform(), 2).lower
            array([0.])
            >>> chaospy.Power(chaospy.Uniform(1, 2), -1).lower
            array([0.5])
            >>> chaospy.Power(2, chaospy.Uniform()).lower
            array([1.])
            >>> chaospy.Power(2, chaospy.Uniform(-1, 0)).lower
            array([0.5])

        """
        left = self._parameters["left"]
        right = self._parameters["right"]

        if isinstance(left, Distribution):
            left_upper = left._get_upper(idx, cache=self._upper_cache)
            left_lower = left._get_lower(idx, cache=self._lower_cache)

            if isinstance(right, Distribution):
                right_upper = right._get_upper(idx, cache=self._upper_cache)
                right_lower = right._get_lower(idx, cache=self._lower_cache)

                out = numpy.min(numpy.broadcast_arrays(
                    left_lower.T**right_lower.T,
                    left_lower.T**right_upper.T,
                    left_upper.T**right_lower.T,
                    left_upper.T**right_upper.T,
                ), axis=0).T

            else:
                # assert 0, (idx, left_lower, left_upper, right)
                out = numpy.min([left_lower**right[idx],
                                 left_upper**right[idx]], axis=0).T

        else:
            assert isinstance(right, Distribution)
            right_upper = right._get_upper(idx, cache=self._upper_cache)
            right_lower = right._get_lower(idx, cache=self._lower_cache)
            out = numpy.min([left[idx]**right_lower,
                             left[idx]**right_upper], axis=0).T

        return out

    def _upper(self, idx, left, right, cache):
        """
        Distribution bounds.

        Example:
            >>> chaospy.Uniform().upper
            array([1.])
            >>> chaospy.Power(chaospy.Uniform(), 2).upper
            array([1.])
            >>> chaospy.Power(chaospy.Uniform(1, 2), -1).upper
            array([1.])
            >>> chaospy.Power(2, chaospy.Uniform()).upper
            array([2.])
            >>> chaospy.Power(2, chaospy.Uniform(-1, 0)).upper
            array([1.])

        """
        left = self._parameters["left"]
        right = self._parameters["right"]
        if isinstance(left, Distribution):
            left_lower = left._get_lower(idx, cache=self._lower_cache)
            left_upper = left._get_upper(idx, cache=self._upper_cache)

            if isinstance(right, Distribution):
                right_lower = right._get_lower(idx, cache=self._lower_cache)
                right_upper = right._get_upper(idx, cache=self._upper_cache)

                out = numpy.max(numpy.broadcast_arrays(
                    (left_lower.T**right_lower.T).T,
                    (left_lower.T**right_upper.T).T,
                    (left_upper.T**right_lower.T).T,
                    (left_upper.T**right_upper.T).T,
                ), axis=0)

            else:
                out = numpy.max([left_lower**right[idx],
                                 left_upper**right[idx]], axis=0)

        else:
            assert isinstance(right, Distribution)
            right_lower = right._get_lower(idx, cache=self._lower_cache)
            right_upper = right._get_upper(idx, cache=self._upper_cache)
            out = numpy.max([left[idx]**right_lower,
                             left[idx]**right_upper], axis=0)

        return out

    def _pdf(self, xloc, idx, left, right, cache):
        """
        Probability density function.

        Example:
            >>> chaospy.Uniform().pdf([-0.5, 0.5, 1.5, 2.5])
            array([0., 1., 0., 0.])
            >>> chaospy.Power(chaospy.Uniform(), 2).pdf([-0.5, 0.5, 1.5, 2.5])
            array([0.        , 0.70710678, 0.        , 0.        ])
            >>> chaospy.Power(chaospy.Uniform(1, 2), -1).pdf([0.4, 0.6, 0.8, 1.2])
            array([0.        , 2.77777778, 1.5625    , 0.        ])
            >>> chaospy.Power(2, chaospy.Uniform()).pdf([-0.5, 0.5, 1.5, 2.5])
            array([0.        , 0.        , 0.96179669, 0.        ])
            >>> chaospy.Power(2, chaospy.Uniform(-1, 0)).pdf([0.4, 0.6, 0.8, 1.2])
            array([0.        , 2.40449173, 1.8033688 , 0.        ])

        """
        if isinstance(left, Distribution):
            x_ = numpy.sign(xloc)*numpy.abs(xloc)**(1./right-1)
            xloc = numpy.sign(xloc)*numpy.abs(xloc)**(1./right)
            pairs = numpy.sign(xloc**right) == 1
            out = left._get_pdf(xloc, idx, cache=cache.copy())
            if numpy.any(pairs):
                out = out+pairs*left._get_pdf(-xloc, idx, cache=cache)
            out = numpy.sign(right)*out*x_/right
            out[numpy.isnan(out)] = numpy.inf

        else:
            assert numpy.all(left > 0), "imaginary result"
            x_ = numpy.where(xloc <= 0, -numpy.inf,
                             numpy.log(xloc + 1.*(xloc<=0))/numpy.log(left+1.*(left == 1)))
            num_ = numpy.log(left+1.*(left == 1))*xloc
            num_ = num_ + 1.*(num_==0)
            out = right._get_pdf(x_, idx, cache=cache)/num_

        return out

    def _cdf(self, xloc, idx, left, right, cache):
        """
        Cumulative distribution function.

        Example:
            >>> chaospy.Uniform().fwd([-0.5, 0.5, 1.5, 2.5])
            array([0. , 0.5, 1. , 1. ])
            >>> chaospy.Power(chaospy.Uniform(), 2).fwd([-0.5, 0.5, 1.5, 2.5])
            array([0.        , 0.70710678, 1.        , 1.        ])
            >>> chaospy.Power(chaospy.Uniform(1, 2), -1).fwd([0.4, 0.6, 0.8, 1.2])
            array([0.        , 0.33333333, 0.75      , 1.        ])
            >>> chaospy.Power(2, chaospy.Uniform()).fwd([-0.5, 0.5, 1.5, 2.5])
            array([0.       , 0.       , 0.5849625, 1.       ])
            >>> chaospy.Power(2, chaospy.Uniform(-1, 0)).fwd([0.4, 0.6, 0.8, 1.2])
            array([0.        , 0.26303441, 0.67807191, 1.        ])

        """
        if isinstance(left, Distribution):
            y = numpy.sign(xloc)*numpy.abs(xloc)**(1./right)
            pairs = numpy.sign(xloc**right) != -1
            out2 = left._get_fwd(-y, idx, cache=cache.copy())
            out1 = left._get_fwd(y, idx, cache=cache)
            out = numpy.where(right < 0, 1-out1, out1-pairs*out2)
        else:
            y = (numpy.log(numpy.abs(xloc)+1.*(xloc <= 0))/
                 numpy.log(numpy.abs(left)+1.*(left == 1)))
            out = right._get_fwd(y, idx, cache=cache)
            out = numpy.where(xloc <= 0, 0., out)
        return out

    def _ppf(self, q, idx, left, right, cache):
        """
        Point percentile function.

        Example:
            >>> chaospy.Uniform().inv([0.1, 0.2, 0.9])
            array([0.1, 0.2, 0.9])
            >>> chaospy.Power(chaospy.Uniform(), 2).inv([0.1, 0.2, 0.9])
            array([0.01, 0.04, 0.81])
            >>> chaospy.Power(chaospy.Uniform(1, 2), -1).inv([0.1, 0.2, 0.9])
            array([0.52631579, 0.55555556, 0.90909091])
            >>> chaospy.Power(2, chaospy.Uniform()).inv([0.1, 0.2, 0.9])
            array([1.07177346, 1.14869835, 1.86606598])
            >>> chaospy.Power(2, chaospy.Uniform(-1, 0)).inv([0.1, 0.2, 0.9])
            array([0.53588673, 0.57434918, 0.93303299])

        """
        if isinstance(left, Distribution):
            q = numpy.where(right.T < 0, 1-q.T, q.T).T
            out = (left._get_inv(q, idx, cache=cache).T**right.T).T
        else:
            out = right._get_inv(q, idx, cache=cache)
            out = numpy.where(left < 0, 1-out, out)
            out = (left.T**out.T).T
        return out

    def _mom(self, k, left, right, cache):
        """
        Statistical moments.

        Example:
            >>> numpy.around(chaospy.Uniform().mom([0, 1, 2, 3]), 4)
            array([1.    , 0.5   , 0.3333, 0.25  ])
            >>> numpy.around(chaospy.Power(chaospy.Uniform(), 2).mom([0, 1, 2, 3]), 4)
            array([1.    , 0.3333, 0.2   , 0.1429])
            >>> numpy.around(chaospy.Power(chaospy.Uniform(1, 2), -1).mom([0, 1, 2, 3]), 4)
            array([1.    , 0.6931, 0.5   , 0.375 ])
            >>> numpy.around(chaospy.Power(2, chaospy.Uniform()).mom([0, 1, 2, 3]), 4)
            array([1.    , 1.4427, 2.164 , 3.3663])
            >>> numpy.around(chaospy.Power(2, chaospy.Uniform(-1, 0)).mom([0, 1, 2, 3]), 4)
            array([1.    , 0.7213, 0.541 , 0.4208])

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
