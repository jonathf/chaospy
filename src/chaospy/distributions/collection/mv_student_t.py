"""Multivariate Student-T Distribution."""
import numpy
from scipy import special, misc

from .student_t import student_t

from ..baseclass import Dist


class MvStudentT(Dist):
    """
    Multivariate Student-T Distribution.

    Args:
        df (float, Dist) : Degree of freedom
        loc (array_like, Dist) : Location parameter
        scale (array_like) : Covariance matrix

    Examples:
        >>> distribution = chaospy.MvStudentT(4, [1, 2], [[1, 0.6], [0.6, 1]])
        >>> print(distribution)
        MvStudentT(df=4, loc=[1.0, 2.0], scale=[[1.0, 0.6], [0.6, 1.0]])
        >>> mesh = numpy.meshgrid(*[numpy.linspace(0, 1, 5)[1:-1]]*2)
        >>> print(numpy.around(distribution.inv(mesh), 4))
        [[[0.2593 1.     1.7407]
          [0.2593 1.     1.7407]
          [0.2593 1.     1.7407]]
        <BLANKLINE>
         [[0.963  1.4074 1.8519]
          [1.5556 2.     2.4444]
          [2.1481 2.5926 3.037 ]]]
        >>> print(numpy.around(distribution.fwd(distribution.inv(mesh)), 4))
        [[[0.25 0.5  0.75]
          [0.25 0.5  0.75]
          [0.25 0.5  0.75]]
        <BLANKLINE>
         [[0.25 0.25 0.25]
          [0.5  0.5  0.5 ]
          [0.75 0.75 0.75]]]
        >>> print(numpy.around(distribution.pdf(distribution.inv(mesh)), 4))
        [[0.1401 0.1677 0.1672]
         [0.1778 0.1989 0.1778]
         [0.1672 0.1677 0.1401]]
        >>> print(numpy.around(distribution.sample(4), 4))
        [[ 1.4248 -0.4149  3.1369  0.9525]
         [ 3.3169  0.4406  1.4287  1.7486]]
    """

    def __init__(self, df=1, loc=[0, 0], scale=[[1, .5], [.5, 1]]):
        loc = numpy.asfarray(loc)
        scale = numpy.asfarray(scale)
        assert len(loc) == len(scale)
        self._repr = {"df": df, "loc": loc.tolist(), "scale": scale.tolist()}

        C = numpy.linalg.cholesky(scale)
        Ci = numpy.linalg.inv(C)
        Dist.__init__(self, a=df, C=C, Ci=Ci, loc=loc)

    def _cdf(self, x, a, C, Ci, loc):
        x = numpy.dot(Ci, (x.T-loc.T).T)
        return special.stdtr(a, x)

    def _ppf(self, q, a, C, Ci, loc):
        z = special.stdtrit(a, q)
        out = (numpy.dot(C, z).T + loc.T).T
        return out

    def _pdf(self, x, a, C, Ci, loc):

        det = numpy.linalg.det(numpy.dot(C,C.T))
        k = len(C)

        x_ = numpy.dot(Ci.T, (x.T-loc.T).T)
        out = numpy.ones(x.shape)
        out[0] = special.gamma(.5*(a+k))/(special.gamma(.5*a)* \
                a**(.5*k)*numpy.pi**(.5*k)*det**.5*\
                (1+numpy.sum(x_*x_,0)/a))
        return out

    def _bnd(self, x, a, C, Ci, loc):
        output = numpy.zeros((2,)+x.shape)
        scale = numpy.sqrt(numpy.diag(numpy.dot(C, C.T)))
        output.T[:, :, 0] = -10**3*scale + loc
        output.T[:, :, 1] = 10**3*scale + loc
        return output

    def __len__(self):
        return len(self.prm["C"])

    # def _mom(self, k, a, C, Ci, loc):

    #     scale = numpy.dot(C, C.T)

    #     def mom(k):

    #         zeros = (numpy.sum(k,0)%2==1)+numpy.any(numpy.array(k)<0, 0)
    #         if numpy.all(zeros, 0):
    #             return 0.

    #         dim, K = k.shape
    #         ra = numpy.arange(dim).repeat(K).reshape(dim,K)

    #         i = numpy.argmax(k!=0, 0)

    #         out = numpy.zeros(k.shape[1:])
    #         out[:] = numpy.where(numpy.choose(i,k),
    #                 (numpy.choose(i,k)-1)*scale[i,i]*mom(k-2*(ra==i)), 1)
    #         for x in range(1, dim):
    #             out += \
    #             (numpy.choose(i,k)!=0)*(x>i)*k[x]*scale[i,x]*mom(k-(ra==i)-(ra==x))

    #         return out

    #     dim = len(loc)
    #     K = numpy.mgrid[[slice(0,_+1,1) for _ in numpy.max(k, 1)]]
    #     K = K.reshape(dim, K.size//dim)
    #     M = mom(K)

    #     out = numpy.zeros(k.shape[1])
    #     for i in range(len(M)):
    #         coef = numpy.prod(misc.comb(k.T, K[:,i]).T, 0)
    #         diff = k.T - K[:,i]
    #         pos = diff>=0
    #         diff = diff*pos
    #         pos = numpy.all(pos, 1)
    #         loc_ = numpy.prod(loc**diff, 1)
    #         out += pos*coef*loc_*M[i]

    #     return out
