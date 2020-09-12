"""Multivariate Log-Normal Distribution."""
import numpy
from scipy import special

from ..baseclass import Distribution



class MvLogNormal(Distribution):
    """
    Multivariate Log-Normal Distribution.

    Args:
        loc (float, Distribution):
            Mean vector
        scale (float, Distribution):
            Covariance matrix or variance vector if scale is a 1-d vector.

    Examples:
        >>> distribution = chaospy.MvLogNormal([1, 2], [[1, 0.6], [0.6, 1]])
        >>> distribution
        MvLogNormal(loc=[1.0, 2.0], scale=[[1.0, 0.6], [0.6, 1.0]])
        >>> mesh = numpy.meshgrid(*[numpy.linspace(0, 1, 5)[1:-1]]*2)
        >>> distribution.inv(mesh).round(4)
        array([[[ 1.3847,  2.7183,  5.3361],
                [ 1.3847,  2.7183,  5.3361],
                [ 1.3847,  2.7183,  5.3361]],
        <BLANKLINE>
               [[ 2.874 ,  4.3077,  6.4566],
                [ 4.9298,  7.3891, 11.075 ],
                [ 8.4562, 12.6745, 18.9971]]])
        >>> distribution.fwd(distribution.inv(mesh)).round(4)
        array([[[0.25, 0.5 , 0.75],
                [0.25, 0.5 , 0.75],
                [0.25, 0.5 , 0.75]],
        <BLANKLINE>
               [[0.25, 0.25, 0.25],
                [0.5 , 0.5 , 0.5 ],
                [0.75, 0.75, 0.75]]])
        >>> distribution.pdf(distribution.inv(mesh)).round(4)
        array([[0.0317, 0.0135, 0.0037],
               [0.0232, 0.0099, 0.0027],
               [0.0108, 0.0046, 0.0012]])
        >>> distribution.sample(4).round(4)
        array([[ 4.0351,  0.8185, 14.1201,  2.5996],
               [23.279 ,  1.8986,  4.9261,  5.8399]])
        >>> distribution.mom((1, 2)).round(4)
        6002.9122
    """

    def __init__(self, loc=[0, 0], scale=[[1, 0.5], [0.5, 1]]):
        loc = numpy.asfarray(loc)
        scale = numpy.asfarray(scale)
        assert len(loc) == len(scale)

        C = numpy.linalg.cholesky(scale)
        Ci = numpy.linalg.inv(C)

        indinces = self._declare_dependencies(len(C))
        dependencies = [set(indinces[:idx+1]) for idx in range(len(C))]

        super(MvLogNormal, self).__init__(
            parameters=dict(loc=loc, C=C, Ci=Ci, scale=scale),
            dependencies=dependencies,
            repr_args=["loc=%s" % loc.tolist(), "scale=%s" % scale.tolist()],
        )

    def _cdf(self, x, loc, C, Ci, scale, cache):
        y = numpy.log(numpy.abs(x) + 1.*(x<=0))
        out = special.ndtr(numpy.dot(Ci, (y.T-loc.T).T))
        return numpy.where(x <= 0, 0., out)

    def _ppf(self, q, loc, C, Ci, scale, cache):
        return numpy.e**(numpy.dot(C, special.ndtri(q)).T+loc.T).T

    def _mom(self, k, loc, C, Ci, scale, cache):
        output =  numpy.dot(k, loc)
        output += .5*numpy.dot(numpy.dot(k, scale), k)
        output =  numpy.e**(output)
        return output

    def _lower(self, loc, C, Ci, scale, cache):
        return numpy.zeros(len(self))

    def _upper(self, loc, C, Ci, scale, cache):
        return numpy.exp(7.1*numpy.sqrt(numpy.diag(scale)) + loc.T).T

    def _value(self, loc, C, Ci, scale, cache):
        return self
