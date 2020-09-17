import numpy
from scipy.special import comb
import numpoly
import chaospy

from .distribution import Distribution
from .core import DistributionCore


class LowerUpper(DistributionCore):
    """
    Lower-Upper transformation.

    Linear transforms any distribution on any interval `[lower, upper]`.

    Args:
        dist (Distribution):
            The underlying distribution to be scaled.
        lower (float, Sequence[float], Distribution):
            Lower bounds.
        upper (float, Sequence[float], Distribution):
            Upper bounds.
        rotation (Sequence[int], Sequence[Sequence[bool]]):
            The order of which to resolve conditionals. Either as a sequence of
            column rotations, or as a permutation matrix.
            Defaults to `range(len(distribution))` which is the same as
            `p(x0), p(x1|x0), p(x2|x0,x1), ...`.

    Attributes:
        shift (numpy.ndarray):
            The shift of the distribution.
        scale (numpy.ndarray):
            The scale of the distribution.

    """

    def __init__(
            self,
            dist,
            lower=0.,
            upper=1.,
            rotation=None,
            repr_args=None,
    ):
        assert isinstance(dist, Distribution), "'dist' should be a distribution"
        if repr_args is None:
            repr_args = dist._repr_args[:]
        if not ((isinstance(lower, (int, float)) and lower == 0) and
                (isinstance(upper, (int, float)) and upper == 1)):
            repr_args += ["lower=%s" % lower, "upper=%s" % upper]
        if rotation is not None:
            repr_args += ["rotation=%s" % list(rotation)]

        super(LowerUpper, self).__init__(
            lower=lower,
            upper=upper,
            repr_args=repr_args,
        )
        if len(dist) == 1 and len(self) > 1:
            dist = chaospy.Iid(dist, len(self))
        if rotation is not None:
            rotation = list(rotation)
        else:
            rotation = [key for key, _ in sorted(enumerate(dist._dependencies), key=lambda x: len(x[1]))]
        permute = numpy.zeros((len(rotation), len(rotation)), dtype=int)
        permute[numpy.arange(len(rotation), dtype=int), rotation] = 1
        self._rotation = rotation
        self._permute = permute
        self._dist = dist

    def _lower(self, lower, upper):
        del upper
        if isinstance(lower, Distribution):
            lower = lower._get_lower(cache={})
        return numpy.asfarray(lower)

    def _upper(self, lower, upper):
        del lower
        if isinstance(upper, Distribution):
            upper = upper._get_upper(cache={})
        return numpy.asfarray(upper)

    def _ppf(self, qloc, lower, upper):
        scale = (upper-lower)/(self._dist.upper-self._dist.lower)
        shift = lower-self._dist.lower*(upper-lower)/(self._dist.upper-self._dist.lower)
        zloc = self._dist.inv(qloc)
        out = (scale.T*zloc.T+shift.T).T
        return out

    def _cdf(self, xloc, lower, upper):
        scale = (upper-lower)/(self._dist.upper-self._dist.lower)
        shift = lower-self._dist.lower*(upper-lower)/(self._dist.upper-self._dist.lower)
        zloc = ((xloc.T-shift.T)/scale.T).T
        out = self._dist.fwd(zloc)
        return out

    def _pdf(self, xloc, lower, upper):
        scale = (upper-lower)/(self._dist.upper-self._dist.lower)
        shift = lower-self._dist.lower*(upper-lower)/(self._dist.upper-self._dist.lower)
        zloc = ((xloc.T-shift.T)/scale.T).T
        out = (self._dist.pdf(zloc, decompose=True).T/scale.T).T
        assert xloc.shape == out.shape
        return out

    def _mom(self, kloc, lower, upper):
        scale = (upper-lower)/(self._dist.upper-self._dist.lower)
        shift = lower-self._dist.lower*(upper-lower)/(self._dist.upper-self._dist.lower)
        poly = numpoly.variable(len(self))
        poly = numpoly.sum(scale*poly, axis=-1)+shift
        poly = numpoly.set_dimensions(numpoly.prod(poly**kloc), len(self))
        out = sum(self._dist.mom(key)*coeff
                  for key, coeff in zip(poly.exponents, poly.coefficients))
        return out

    def _ttr(self, kloc, lower, upper):
        scale = (upper-lower)/(self._dist.upper-self._dist.lower)
        shift = lower-self._dist.lower*(upper-lower)/(self._dist.upper-self._dist.lower)
        coeff0, coeff1 = self._dist._get_ttr(kloc)
        coeff0 = coeff0*scale+shift
        coeff1 = coeff1*scale*scale
        return coeff0, coeff1

    def _cache(self, **kwargs):
        return self
