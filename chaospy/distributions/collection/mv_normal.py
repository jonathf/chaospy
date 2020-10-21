"""Multivariate Normal Distribution."""
import logging
import numpy
from scipy import special
import chaospy

from .normal import normal
from ..baseclass import MeanCovarianceDistribution


class MvNormal(MeanCovarianceDistribution):
    r"""
    Multivariate Normal Distribution.

    Args:
        mu (float, numpy.ndarray):
            Mean vector
        scale (float, numpy.ndarray):
            Covariance matrix or variance vector if scale is a 1-d vector.
        rotation (Sequence[int]):
            The order of which to resolve conditionals.
            Defaults to `range(len(distribution))` which is the same as
            `p(x0), p(x1|x0), p(x2|x0,x1), ...`.

    Examples:
        >>> distribution = chaospy.MvNormal([10, 20, 30],
        ...     [[1, 0.2, 0.3], [0.2, 2, 0.4], [0.3, 0.4, 1]], rotation=[1, 2, 0])
        >>> distribution  # doctest: +NORMALIZE_WHITESPACE
        MvNormal(mu=[10, 20, 30],
                 sigma=[[1, 0.2, 0.3], [0.2, 2, 0.4], [0.3, 0.4, 1]])
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
        array([[ 8.25,  8.67,  8.47,  8.88,  9.71, 10.13,  9.93, 10.35],
               [18.19, 18.19, 20.36, 20.36, 18.19, 18.19, 20.36, 20.36],
               [28.41, 29.88, 28.84, 30.31, 28.41, 29.88, 28.84, 30.31]])
        >>> numpy.allclose(distribution.fwd(mapped_samples), mesh)
        True
        >>> distribution.pdf(mapped_samples).round(4)
        array([0.0042, 0.0092, 0.0092, 0.0203, 0.0092, 0.0203, 0.0203, 0.0446])
        >>> distribution.sample(4).round(4)
        array([[10.3396,  9.0158, 11.1009, 10.0971],
               [21.6096, 18.871 , 17.5357, 19.6314],
               [29.6231, 30.7349, 28.7239, 30.5507]])

    """

    def __init__(
            self,
            mu,
            sigma=None,
            rotation=None,
    ):
        super(MvNormal, self).__init__(
            dist=normal(),
            mean=mu,
            covariance=sigma,
            rotation=rotation,
            repr_args=chaospy.format_repr_kwargs(mu=(mu, None))+
                      chaospy.format_repr_kwargs(sigma=(sigma, None)),
        )

    def _mom(self, k, mean, sigma, cache):
        if isinstance(mean, chaospy.Distribution):
            mean = mean._get_cache(None, cache=cache, get=0)
            if isinstance(mean, chaospy.Distribution):
                raise chaospy.UnsupportedFeature(
                    "Analytical moment of a conditional not supported")
        out = 0.
        for idx, kdx in enumerate(numpy.ndindex(*[_+1 for _ in k])):
            coef = numpy.prod(special.comb(k.T, kdx).T, 0)
            diff = k.T-kdx
            pos = diff >= 0
            diff = diff*pos
            pos = numpy.all(pos)
            location_ = numpy.prod(mean**diff, -1)

            out = out+pos*coef*location_*isserlis_moment(tuple(kdx), sigma)

        return out


def isserlis_moment(k, scale):
    """
    Centralized statistical moments using Isserlis' theorem.

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
    if scale.ndim == 2:
        scale = scale[numpy.newaxis]
        return isserlis_moment(k, scale)[0]

    if not isinstance(k, numpy.ndarray):
        k = numpy.asarray(k)
    assert len(k.shape) == 1

    # Recursive exit condition
    if (numpy.sum(k) % 2 == 1) or numpy.any(k < 0):
        return numpy.zeros(len(scale))

    # Choose a pivot index as first non-zero entry
    idx = numpy.nonzero(k)[0]
    if not idx.size:
        # Skip if no pivot found
        return numpy.ones(len(scale))
    idx = idx[0]

    eye = numpy.eye(len(k), dtype=int)
    out = (k[idx]-1)*scale[:, idx, idx]*isserlis_moment(k-2*eye[idx], scale)
    for idj in range(idx+1, len(k)):
        out += k[idj]*scale[:, idx, idj]*isserlis_moment(k-eye[idx]-eye[idj], scale)
    return out
