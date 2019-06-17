"""Multivariate Normal Distribution."""
import numpy
from scipy import special

from chaospy.bertran import bindex
from .normal import normal

from ..baseclass import Dist


class MvNormal(Dist):
    """
    Multivariate Normal Distribution.

    Args:
        loc (float, Dist): Mean vector
        scale (float, Dist): Covariance matrix or variance vector if scale is a 1-d vector.

    Examples:
        >>> distribution = chaospy.MvNormal([1, 2], [[1, 0.6], [0.6, 1]])
        >>> print(distribution)
        MvNormal(loc=[1.0, 2.0], scale=[[1.0, 0.6], [0.6, 1.0]])
        >>> print(chaospy.Cov(distribution))
        [[1.  0.6]
         [0.6 1. ]]
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
        >>> print(numpy.around(distribution.ttr([[1, 1, 2], [1, 2, 2]]), 4))
        [[[1.   1.   1.  ]
          [1.36 1.36 2.72]]
        <BLANKLINE>
         [[2.   2.   2.  ]
          [0.64 1.28 1.28]]]
    """

    def __init__(self, loc=[0, 0], scale=[[1, .5], [.5, 1]]):
        loc = numpy.asfarray(loc)
        scale = numpy.asfarray(scale)
        assert len(loc) == len(scale)
        self._repr = {"loc": loc.tolist(), "scale": scale.tolist()}

        C = numpy.linalg.cholesky(scale)
        Ci = numpy.linalg.inv(C)
        Dist.__init__(self, C=C, Ci=Ci, loc=loc)

    def _cdf(self, x, C, Ci, loc):
        return special.ndtr(numpy.dot(Ci, (x.T-loc.T).T))

    def _ppf(self, q, C, Ci, loc):
        return (numpy.dot(C, special.ndtri(q)).T+loc.T).T

    def _pdf(self, x, C, Ci, loc):
        det = numpy.linalg.det(numpy.dot(C,C.T))
        x_ = numpy.dot(Ci.T, (x.T-loc.T).T)
        out = numpy.ones(x.shape)
        out[0] =  numpy.e**(-.5*numpy.sum(x_*x_, 0))/numpy.sqrt((2*numpy.pi)**len(Ci)*det)
        return out

    def _bnd(self, x, C, Ci, loc):
        scale = numpy.sqrt(numpy.diag(numpy.dot(C,C.T)))
        lo,up = numpy.zeros((2,)+x.shape)
        lo.T[:] = (-7.5*scale+loc)
        up.T[:] = (7.5*scale+loc)
        return lo, up

    def _mom(self, k, C, Ci, loc):
        scale = numpy.dot(C, C.T)
        out = 0.
        for idx, kdx in enumerate(bindex(k, dim=len(C), sort="G")):
            coef = numpy.prod(special.comb(k.T, kdx).T, 0)
            diff = k.T - kdx
            pos = diff >= 0
            diff = diff*pos
            pos = numpy.all(pos)
            loc_ = numpy.prod(loc**diff)

            out += pos*coef*loc_*isserlis_moment(tuple(kdx), scale)

        return float(out)

    def __len__(self):
        return len(self.prm["C"])


def isserlis_moment(k, scale):
    """
    Raw statistical moments centralized Normal distribution using Isserlis'
    theorem.

    Args:
        k (Tuple[int, ...]):
            Moment orders.
        scale (ndarray):
            Covariance matrix defining dependencies between variables.

    Returns:
        Raw statistical moment of order ``k`` given covariance ``scale``.

    Examples:
        >>> scale = 0.5*numpy.eye(3)+0.5
        >>> isserlis_moment((2, 2, 2), scale)
        3.5
        >>> isserlis_moment((0, 0, 0), scale)
        1.0
        >>> isserlis_moment((1, 0, 0), scale)
        0.0
        >>> isserlis_moment((0, 1, 1), scale)
        0.5
        >>> isserlis_moment((0, 0, 2), scale)
        1.0
    """
    if not isinstance(k, numpy.ndarray):
        k = numpy.asarray(k)
    assert len(k.shape) == 1

    # Recursive exit condition
    if (numpy.sum(k) % 2 == 1) or numpy.any(k < 0):
        return 0.

    # Choose a pivot index as first non-zero entry
    idx = numpy.nonzero(k)[0]
    if not idx.size:
        # Skip if no pivot found
        return 1.
    idx = idx[0]

    eye = numpy.eye(len(k), dtype=int)
    out = (k[idx]-1)*scale[idx, idx]*isserlis_moment(k-2*eye[idx], scale)
    for idj in range(idx+1, len(k)):
        out += k[idj]*scale[idx, idj]*isserlis_moment(
            k-eye[idx]-eye[idj], scale)

    return float(out)
