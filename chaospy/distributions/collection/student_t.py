"""Student-T distribution."""
import numpy
from scipy import special

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
        return special.stdtrit(a, q)

    def _mom(self, k, a):
        if numpy.any(a < k):
            raise ValueError("too high mom for student-t")
        out = special.gamma(.5*k+.5)* \
                special.gamma(.5*a-.5*k)*a**(.5*k)
        return numpy.where(k%2==0, out/(numpy.pi**.5*special.gamma(.5*a)), 0)

    def _ttr(self, k, a):
        return 0., k*a*(a-k+1.)/ ((a-2*k)*(a-2*k+2))


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
        >>> distribution = chaospy.StudentT(2, 2, 2)
        >>> distribution
        StudentT(2, mu=2, sigma=2)
        >>> q = numpy.linspace(0, 1, 6)[1:-1]
        >>> distribution.inv(q).round(4)
        array([-0.1213,  1.4226,  2.5774,  4.1213])
        >>> distribution.fwd(distribution.inv(q)).round(4)
        array([0.2, 0.4, 0.6, 0.8])
        >>> distribution.pdf(distribution.inv(q)).round(4)
        array([0.0905, 0.1663, 0.1663, 0.0905])
        >>> distribution.sample(4).round(4)
        array([ 2.913 , -1.4132,  7.8594,  1.8992])
        >>> distribution.mom(1)
        array(2.)
    """

    def __init__(self, df=1, mu=0, sigma=1):
        super(StudentT, self).__init__(
            dist=student_t(df),
            scale=sigma,
            shift=mu,
        )
        self._repr_args = [df, "mu=%s" % mu, "sigma=%s" % sigma]
