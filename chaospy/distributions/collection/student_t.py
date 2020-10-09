"""Student-T distribution."""
import numpy
from scipy import special
import chaospy

from ..baseclass import SimpleDistribution, ShiftScaleDistribution


class student_t(SimpleDistribution):
    """Student-T distribution."""

    def __init__(self, a=1):
        super(student_t, self).__init__(dict(a=a))

    def _pdf(self, x, a):
        return special.gamma(.5*a+.5)*(1+x*x/a)**(-.5*a-.5) /\
                (numpy.sqrt(a*numpy.pi)*special.gamma(.5*a))

    def _cdf(self, x, a):
        return special.stdtr(a, x)

    def _ppf(self, q, a):
        return special.stdtrit(a, numpy.clip(q, 1e-12, 1-1e-12))

    def _mom(self, k, a):
        if numpy.any(a < k):
            raise ValueError("too high mom for student-t")
        out = special.gamma(.5*k+.5)* \
                special.gamma(.5*a-.5*k)*a**(.5*k)
        return numpy.where(k%2==0, out/(numpy.pi**.5*special.gamma(.5*a)), 0)

    def _ttr(self, k, a):
        return 0., k*a*(a-k+1.)/ ((a-2*k)*(a-2*k+2))

    def _lower(self, a):
        return special.stdtrit(a, 1e-12)

    def _upper(self, a):
        return special.stdtrit(a, 1-1e-12)


class StudentT(ShiftScaleDistribution):
    """
    (Non-central) Student-t distribution.

    Args:
        df (float, Distribution):
            Degrees of freedom.
        loc (float, Distribution):
            Location parameter.
        scale (float, Distribution):
            Scale parameter.

    Examples:
        >>> distribution = chaospy.StudentT(10)
        >>> distribution
        StudentT(10)
        >>> uloc = numpy.linspace(0, 1, 6)
        >>> uloc
        array([0. , 0.2, 0.4, 0.6, 0.8, 1. ])
        >>> xloc = distribution.inv(uloc)
        >>> xloc.round(3)
        array([-40.532,  -0.879,  -0.26 ,   0.26 ,   0.879,  40.532])
        >>> numpy.allclose(distribution.fwd(xloc), uloc)
        True
        >>> distribution.pdf(xloc).round(3)
        array([0.   , 0.258, 0.375, 0.375, 0.258, 0.   ])
        >>> distribution.sample(4).round(3)
        array([ 0.407, -1.278,  1.816, -0.046])
        >>> distribution.mom(1).round(3)
        0.0
        >>> distribution.ttr([0, 1, 2, 3]).round(3)
        array([[ 0.  ,  0.  ,  0.  ,  0.  ],
               [ 0.  ,  1.25,  3.75, 10.  ]])

    """

    def __init__(self, df=1, mu=0, sigma=1):
        super(StudentT, self).__init__(
            dist=student_t(df),
            scale=sigma,
            shift=mu,
        )
        self._repr_args = [df]
        self._repr_args += chaospy.format_repr_kwargs(mu=(mu, 0))
        self._repr_args += chaospy.format_repr_kwargs(sigma=(sigma, 1))
