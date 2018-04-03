"""Log-uniform distribution."""
import numpy

from ..baseclass import Dist
from ..operators.addition import Add


class log_uniform(Dist):
    """Log-uniform distribution."""

    def __init__(self, lo=0, up=1):
        Dist.__init__(self, lo=lo, up=up)

    def _pdf(self, x, lo, up):
        return 1./(x*(up-lo))

    def _cdf(self, x, lo, up):
        return (numpy.log(x)-lo)/(up-lo)

    def _ppf(self, q, lo, up):
        return numpy.e**(q*(up-lo) + lo)

    def _bnd(self, x, lo, up):
        return numpy.e**lo, numpy.e**up

    def _mom(self, k, lo, up):
        return ((numpy.e**(up*k)-numpy.e**(lo*k))/((up-lo)*(k+(k==0))))**(k!=0)


class LogUniform(Add):
    """
    Log-uniform distribution

    Args:
        lower (float, Dist): Location of lower threshold of uniform distribution.
        upper (float, Dist): Location of upper threshold of uniform distribution.
        scale (float, Dist): Scaling parameter
        shift (float, Dist): Location parameter

    Examples:
        >>> f = chaospy.LogUniform(2, 3, 2, 3)
        >>> print(f)
        LogUniform(lower=2, scale=2, shift=3, upper=3)
        >>> q = numpy.linspace(0,1,6)[1:-1]
        >>> print(numpy.around(f.inv(q), 4))
        [21.05   25.0464 29.9275 35.8893]
        >>> print(numpy.around(f.fwd(f.inv(q)), 4))
        [0.2 0.4 0.6 0.8]
        >>> print(numpy.around(f.pdf(f.inv(q)), 4))
        [0.0554 0.0454 0.0371 0.0304]
        >>> print(numpy.around(f.sample(4), 4))
        [31.4099 19.5793 41.2227 26.9349]
        >>> print(numpy.around(f.mom(1), 4))
        28.393
    """
    def __init__(self, lower=0, upper=1, scale=1, shift=0):
        self._repr = {"lower": lower, "upper": upper, "scale": scale, "shift": shift}
        Add.__init__(self, left=log_uniform(lower, upper)*scale, right=shift)
