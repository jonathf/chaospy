"""Multivariate Normal Distribution."""
import numpy
from scipy import special

from .normal import normal

from ..baseclass import Dist


class MvNormal(Dist):
    r"""
    Multivariate Normal Distribution.

    Args:
        loc (float, Dist):
            Mean vector
        scale (float, Dist):
            Covariance matrix or variance vector if scale is a 1-d vector.
        rotation (Sequence[int]):
            The order of which to resolve conditionals.
            Defaults to `range(len(distribution))` which is the same as
            `p(x0), p(x1|x0), p(x2|x0,x1), ...`.

    Examples:
        >>> distribution = chaospy.MvNormal([10, 20, 30],
        ...     [[1, 0.2, 0.3], [0.2, 2, 0.4], [0.3, 0.4, 1]], rotation=[1, 2, 0])
        >>> distribution  # doctest: +NORMALIZE_WHITESPACE
        MvNormal(loc=[10.0, 20.0, 30.0], rotation=[1, 2, 0],
                 scale=[[1.0, 0.2, 0.3], [0.2, 2.0, 0.4], [0.3, 0.4, 1.0]])
        >>> chaospy.E(distribution)
        array([10., 20., 30.])
        >>> chaospy.Cov(distribution)
        array([[1. , 0.2, 0.3],
               [0.2, 2. , 0.4],
               [0.3, 0.4, 1. ]])
        >>> mesh = numpy.mgrid[:2, :2, :2].reshape(3, -1)*.5+.1
        >>> mesh
        array([[0.1, 0.1, 0.1, 0.1, 0.6, 0.6, 0.6, 0.6],
               [0.1, 0.1, 0.6, 0.6, 0.1, 0.1, 0.6, 0.6],
               [0.1, 0.6, 0.1, 0.6, 0.1, 0.6, 0.1, 0.6]])
        >>> mapped_samples = distribution.inv(mesh)
        >>> mapped_samples.round(2)
        array([[ 7.64,  7.77,  8.26,  8.39,  9.72,  9.85, 10.34, 10.47],
               [18.72, 18.72, 20.25, 20.25, 18.72, 18.72, 20.25, 20.25],
               [28.39, 29.86, 28.85, 30.32, 28.39, 29.86, 28.85, 30.32]])
        >>> numpy.allclose(distribution.fwd(mapped_samples), mesh)
        True
        >>> distribution.pdf(mapped_samples).round(4)
        array([0.0042, 0.0092, 0.0092, 0.0203, 0.0092, 0.0203, 0.0203, 0.0446])
        >>> distribution.sample(4).round(4)
        array([[10.929 ,  8.1396, 11.4652,  9.8899],
               [21.1382, 19.2016, 18.2575, 19.7394],
               [29.6464, 30.716 , 28.6983, 30.5428]])

    """

    def __init__(self, loc=[0, 0], scale=[[1, .5], [.5, 1]], rotation=None):
        loc = numpy.asfarray(loc)
        self.scale = numpy.asfarray(scale)
        assert len(loc) == len(scale)
        self._repr = {"loc": loc.tolist(), "scale": self.scale.tolist()}
        if rotation is not None:
            self.rotation = numpy.asarray(rotation, dtype=int)
            self._repr["rotation"] = self.rotation.tolist()
        else:
            self.rotation = numpy.arange(len(self.scale), dtype=int)

        self.P = numpy.zeros((len(self.scale), len(self.scale)), dtype=int)
        self.P[numpy.arange(len(self.scale), dtype=int), self.rotation] = 1
        self.C = numpy.linalg.cholesky(self.P.T.dot(self.scale).dot(self.P))
        self.Ci = numpy.linalg.inv(self.C)
        self.loc = self.P.dot(loc)
        Dist.__init__(self)

    def _ppf(self, q):
        z = numpy.clip(special.ndtri(q).T, -7.5, 7.5).T
        z = self.P.dot(z)
        x = (self.C.dot(z).T+self.loc).T
        x = self.P.T.dot(x)
        return x

    def _cdf(self, x):
        x = self.P.dot(x)
        z = self.Ci.dot((x.T-self.loc).T)
        z = self.P.T.dot(z)
        x = special.ndtr(z)
        return x

    def _pdf(self, x):
        x_ = self.P.dot(x)
        scale = self.P.T.dot(self.scale).dot(self.P)

        out = numpy.ones(x.shape)
        alpha = 1/numpy.sqrt(2*numpy.pi*scale[0, 0])
        out[0] = alpha*numpy.e**(-0.5*(x_[0]-self.loc[0])**2/scale[0, 0])

        for dim in range(1, len(self)):
            s11 = scale[dim, dim]
            s22inv = numpy.linalg.inv(scale[:dim, :dim])
            s12 = scale[dim, :dim]
            s21 = scale[:dim, dim]
            loc = self.loc[dim]+s12.dot(s22inv).dot((x_[:dim].T-self.loc[:dim]).T)
            sigma = s11-s12.dot(s22inv).dot(s21)
            alpha = 1/numpy.sqrt(2*numpy.pi*sigma)
            out[dim] = alpha*numpy.e**(-0.5*(x_[dim]-loc)**2/sigma)

        out = self.P.dot(out)
        return out

    def _lower(self):
        return -7.5*numpy.sqrt(numpy.diag(self.scale))+self.loc.T.dot(self.P).T

    def _upper(self):
        return 7.5*numpy.sqrt(numpy.diag(self.scale))+self.loc.T.dot(self.P).T

    def _mom(self, k):
        loc = self.loc.T.dot(self.P)
        out = 0.
        for idx, kdx in enumerate(numpy.ndindex(*[_+1 for _ in k])):
            coef = numpy.prod(special.comb(k.T, kdx).T, 0)
            diff = k.T - kdx
            pos = diff >= 0
            diff = diff*pos
            pos = numpy.all(pos)
            loc_ = numpy.prod(loc**diff)

            out += pos*coef*loc_*isserlis_moment(tuple(kdx), self.scale)

        return float(out)

    def __len__(self):
        return len(self.C)


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
