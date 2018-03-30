"""Multivariate Normal Distribution."""
import numpy
from scipy import special, misc

from .normal import normal

from ..baseclass import Dist


class MvNormal(Dist):
    """
    Multivariate Normal Distribution

    Args:
        loc (float, Dist) : Mean vector
        scale (float, Dist) : Covariance matrix or variance vector if scale is a 1-d vector.

    Examples:
        >>> distribution = chaospy.MvNormal([1, 2], [[1, 0.6], [0.6, 1]])
        >>> print(distribution)
        MvNormal(loc=[1.0, 2.0], scale=[[1.0, 0.6], [0.6, 1.0]])
        >>> mesh = numpy.meshgrid(*[numpy.linspace(0, 1, 5)[1:-1]]*2)
        >>> print(numpy.around(distribution.inv(mesh), 4))
        [[[0.3255 1.     1.6745]
          [0.3255 1.     1.6745]
          [0.3255 1.     1.6745]]
        <BLANKLINE>
         [[1.0557 1.4604 1.8651]
          [1.5953 2.     2.4047]
          [2.1349 2.5396 2.9443]]]
        >>> print(numpy.around(distribution.fwd(distribution.inv(mesh)), 4))
        [[[0.25 0.5  0.75]
          [0.25 0.5  0.75]
          [0.25 0.5  0.75]]
        <BLANKLINE>
         [[0.25 0.25 0.25]
          [0.5  0.5  0.5 ]
          [0.75 0.75 0.75]]]
        >>> print(numpy.around(distribution.pdf(distribution.inv(mesh)), 4))
        [[0.0991 0.146  0.1452]
         [0.1634 0.1989 0.1634]
         [0.1452 0.146  0.0991]]
        >>> print(numpy.around(distribution.sample(4), 4))
        [[ 1.395  -0.2003  2.6476  0.9553]
         [ 3.1476  0.6411  1.5946  1.7647]]
        >>> print(numpy.around(distribution.mom((1, 2)), 4))
        7.4
    """

    def __init__(self, loc=[0, 0], scale=[[1, .5], [.5, 1]]):
        loc = numpy.asfarray(loc)
        scale = numpy.asfarray(scale)
        assert len(loc) == len(scale)
        self._repr = {"loc": str(loc.tolist()), "scale": str(scale.tolist())}

        C = numpy.linalg.cholesky(scale)
        Ci = numpy.linalg.inv(C)
        Dist.__init__(self, C=C, Ci=Ci, loc=loc,
                _advance=True, _length=len(C))

    def _cdf(self, x, graph):
        Ci, loc = graph.keys["Ci"], graph.keys["loc"]
        return special.ndtr(numpy.dot(Ci, (x.T-loc.T).T))

    def _ppf(self, q, graph):
        return (numpy.dot(graph.keys["C"], special.ndtri(q)).T+graph.keys["loc"].T).T

    def _pdf(self, x, graph):

        loc, C, Ci = graph.keys["loc"], graph.keys["C"], graph.keys["Ci"]
        det = numpy.linalg.det(numpy.dot(C,C.T))

        x_ = numpy.dot(Ci.T, (x.T-loc.T).T)
        out = numpy.ones(x.shape)
        out[0] =  numpy.e**(-.5*numpy.sum(x_*x_, 0))/numpy.sqrt((2*numpy.pi)**len(Ci)*det)
        return out

    def _bnd(self, x, graph):

        C, loc = graph.keys["C"], graph.keys["loc"]
        scale = numpy.sqrt(numpy.diag(numpy.dot(C,C.T)))
        lo,up = numpy.zeros((2,)+x.shape)
        lo.T[:] = (-7.5*scale+loc)
        up.T[:] = (7.5*scale+loc)
        return lo,up

    def _mom(self, k, graph):

        C, loc = graph.keys["C"], graph.keys["loc"]
        scale = numpy.dot(C, C.T)

        def mom(k):

            zeros = (numpy.sum(k,0)%2==1)+numpy.any(numpy.array(k)<0, 0)
            if numpy.all(zeros, 0):
                return 0.

            dim, K = k.shape
            ra = numpy.arange(dim).repeat(K).reshape(dim,K)

            i = numpy.argmax(k!=0, 0)

            out = numpy.zeros(k.shape[1:])
            out[:] = numpy.where(numpy.choose(i,k),
                    (numpy.choose(i,k)-1)*scale[i,i]*mom(k-2*(ra==i)), 1)
            for x in range(1, dim):
                out += \
                (numpy.choose(i,k)!=0)*(x>i)*k[x]*scale[i,x]*mom(k-(ra==i)-(ra==x))

            return out

        dim = len(loc)
        K = numpy.mgrid[[slice(0,_+1,1) for _ in numpy.max(k, 1)]]
        K = K.reshape(dim, int(K.size/dim))
        M = mom(K)

        out = numpy.zeros(k.shape[1])
        for i in range(len(M)):
            coef = numpy.prod(misc.comb(k.T, K[:,i]).T, 0)
            diff = k.T - K[:,i]
            pos = diff>=0
            diff = diff*pos
            pos = numpy.all(pos, 1)
            loc_ = numpy.prod(loc**diff, 1)
            out += pos*coef*loc_*M[i]

        return out

    def _dep(self, graph):
        n = normal()
        out = [set([n]) for _ in range(len(self))]
        return out
