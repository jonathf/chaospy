"""Multivariate Log-Normal Distribution."""
import numpy
from scipy import special

from ..baseclass import Dist


class MvLogNormal(Dist):
    """
    Multivariate Log-Normal Distribution.

    Args:
        loc (float, Dist): Mean vector
        scale (float, Dist): Covariance matrix or variance vector if scale
            is a 1-d vector.

    Examples:
        >>> distribution = chaospy.MvLogNormal([1, 2], [[1, 0.6], [0.6, 1]])
        >>> print(distribution)
        MvLogNormal(loc=[1.0, 2.0], scale=[[1.0, 0.6], [0.6, 1.0]])
        >>> mesh = numpy.meshgrid(*[numpy.linspace(0, 1, 5)[1:-1]]*2)
        >>> print(numpy.around(distribution.inv(mesh), 4))
        [[[ 1.3847  2.7183  5.3361]
          [ 1.3847  2.7183  5.3361]
          [ 1.3847  2.7183  5.3361]]
        <BLANKLINE>
         [[ 2.874   4.3077  6.4566]
          [ 4.9298  7.3891 11.075 ]
          [ 8.4562 12.6745 18.9971]]]
        >>> print(numpy.around(distribution.fwd(distribution.inv(mesh)), 4))
        [[[0.25 0.5  0.75]
          [0.25 0.5  0.75]
          [0.25 0.5  0.75]]
        <BLANKLINE>
         [[0.25 0.25 0.25]
          [0.5  0.5  0.5 ]
          [0.75 0.75 0.75]]]
        >>> print(numpy.around(distribution.pdf(distribution.inv(mesh)), 4))
        [[0.0317 0.0135 0.0037]
         [0.0232 0.0099 0.0027]
         [0.0108 0.0046 0.0012]]
        >>> print(numpy.around(distribution.sample(4), 4))
        [[ 4.0351  0.8185 14.1201  2.5996]
         [23.279   1.8986  4.9261  5.8399]]
        >>> print(numpy.around(distribution.mom((1, 2)), 4))
        6002.9122
    """

    def __init__(self, loc=[0, 0], scale=[[1, 0.5], [0.5, 1]]):
        loc = numpy.asfarray(loc)
        scale = numpy.asfarray(scale)
        assert len(loc) == len(scale)
        self._repr = {"loc": str(loc.tolist()), "scale": str(scale.tolist())}

        C = numpy.linalg.cholesky(scale)
        Ci = numpy.linalg.inv(C)
        Dist.__init__(self, loc=loc, C=C, Ci=Ci, scale=scale)

    def _cdf(self, x, loc, C, Ci, scale):
        y = numpy.log(numpy.abs(x) + 1.*(x<=0))
        out = special.ndtr(numpy.dot(Ci, (y.T-loc.T).T))
        return numpy.where(x <= 0, 0., out)

    def _ppf(self, q, loc, C, Ci, scale):
        return numpy.e**(numpy.dot(C, special.ndtri(q)).T+loc.T).T

    def _mom(self, k, loc, C, Ci, scale):
        output =  numpy.dot(k, loc)
        output += .5*numpy.dot(numpy.dot(k, scale), k)
        output =  numpy.e**(output)
        return output

    def _bnd(self, x, loc, C, Ci, scale):
        up = (7.1*numpy.sqrt(numpy.diag(scale))*x.T**0 + loc.T).T
        return 0*up, numpy.e**up

    def __len__(self):
        return len(self.prm["C"])
