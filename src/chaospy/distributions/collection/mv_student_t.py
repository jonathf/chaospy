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
        >>> distribution = chaospy.MvStudentT(40, [1, 2], [[1, 0.6], [0.6, 1]])
        >>> print(distribution)
        MvStudentT(df=40, loc=[1.0, 2.0], scale=[[1.0, 0.6], [0.6, 1.0]])
        >>> mesh = numpy.meshgrid(*[numpy.linspace(0, 1, 5)[1:-1]]*2)
        >>> print(numpy.around(distribution.inv(mesh), 4))
        [[[0.3193 1.     1.6807]
          [0.3193 1.     1.6807]
          [0.3193 1.     1.6807]]
        <BLANKLINE>
         [[1.0471 1.4555 1.8639]
          [1.5916 2.     2.4084]
          [2.1361 2.5445 2.9529]]]
        >>> print(numpy.around(distribution.fwd(distribution.inv(mesh)), 4))
        [[[0.25 0.5  0.75]
          [0.25 0.5  0.75]
          [0.25 0.5  0.75]]
        <BLANKLINE>
         [[0.25 0.25 0.25]
          [0.5  0.5  0.5 ]
          [0.75 0.75 0.75]]]
        >>> print(numpy.around(distribution.pdf(distribution.inv(mesh)), 4))
        [[0.1921 0.1959 0.1958]
         [0.197  0.1989 0.197 ]
         [0.1958 0.1959 0.1921]]
        >>> print(numpy.around(distribution.sample(4), 4))
        [[ 1.3979 -0.2189  2.6868  0.9551]
         [ 3.1625  0.6234  1.582   1.7631]]
        >>> print(numpy.around(distribution.ttr([[1, 1, 2], [1, 2, 2]]), 4))
        [[[-1.79359200e+02 -1.79359200e+02  1.24720000e+00]
          [ 5.86980774e+07  5.86980774e+07  2.07213043e+10]]
        <BLANKLINE>
         [[-7.85283000e+01  2.20170000e+00  2.20170000e+00]
          [ 1.85199191e+08  6.53502729e+10  6.53502729e+10]]]
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
