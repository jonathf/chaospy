"""Multivariate Normal Distribution."""
import logging
import numpy
from scipy import special

from .normal import normal

from ..baseclass import Dist
from ..operators import LocScaling


class MvNormal(LocScaling):
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
                 sigma=[[1.0, 0.2, 0.3], [0.2, 2.0, 0.4], [0.3, 0.4, 1.0]])
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
        array([[ 8.39,  8.85,  8.39,  8.85,  9.86, 10.32,  9.86, 10.32],
               [17.64, 18.26, 19.72, 20.34, 17.77, 18.39, 19.85, 20.47],
               [28.72, 30.25, 28.72, 30.25, 28.72, 30.25, 28.72, 30.25]])
        >>> numpy.allclose(distribution.fwd(mapped_samples), mesh)
        True
        >>> distribution.pdf(mapped_samples).round(4)
        array([0.0042, 0.0092, 0.0092, 0.0203, 0.0092, 0.0203, 0.0203, 0.0446])
        >>> distribution.sample(4).round(4)
        array([[10.1583,  9.1555, 11.3267, 10.1527],
               [21.2826, 19.2191, 17.4524, 19.9038],
               [29.2714, 31.0016, 29.1834, 30.651 ]])

    """

    def __init__(
            self,
            mu,
            sigma=None,
            rotation=None,
            scale=None,
    ):
        logger = logging.getLogger(__name__)
        assert scale is None or sigma is None, (
            "Parameters 'scale' and 'sigma' can not be provided at the same time")
        self._repr = {"mu": numpy.array(mu).tolist()}
        if scale is not None:
            logger.warning("Argument `scale` is to be deprecated. "
                           "Use argument `sigma=scale**2` instead.")
            sigma = scale**2
            self._repr["scale"] = numpy.array(scale).tolist()
        elif sigma is not None:
            self._repr["sigma"] = numpy.array(sigma).tolist()

        LocScaling.__init__(
            self, dist=normal(), mean=mu, covariance=sigma, rotation=rotation)

    def _mom(self, k, cache):
        out = 0.
        for idx, kdx in enumerate(numpy.ndindex(*[_+1 for _ in k])):
            coef = numpy.prod(special.comb(k.T, kdx).T, 0)
            diff = k.T-kdx
            pos = diff >= 0
            diff = diff*pos
            pos = numpy.all(pos)
            location_ = numpy.prod(self.mean**diff)

            out += pos*coef*location_*isserlis_moment(tuple(kdx), self.covariance)

        return float(out)


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
        out += k[idj]*scale[idx, idj]*isserlis_moment(k-eye[idx]-eye[idj], scale)

    return float(out)
