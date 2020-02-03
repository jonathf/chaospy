"""Student-T distribution."""
import numpy
from scipy import special

from ..baseclass import Dist
from ..operators.addition import Add
from .deprecate import deprecation_warning


class student_t(Dist):
    """Student-T distribution."""

    def __init__(self, a=1):
        Dist.__init__(self, a=a)

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


class StudentT(Add):
    """
    (Non-central) Student-t distribution.

    Args:
        df (float, Dist):
            Degrees of freedom
        loc (float, Dist):
            Location parameter
        scale (float, Dist):
            Scale parameter

    Examples:
        >>> distribution = chaospy.StudentT(2, 2, 2)
        >>> distribution
        StudentT(df=2, loc=2, scale=2)
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

    def __init__(self, df=1, loc=0, scale=1):
        self._repr = {"df": df, "loc": loc, "scale": scale}
        Add.__init__(self, left=student_t(df)*scale, right=loc)


Student_t = deprecation_warning(StudentT, "Student_t")
