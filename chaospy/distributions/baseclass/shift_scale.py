import numpy
from scipy.special import comb
import numpoly
import chaospy

from .distribution import Distribution
from .core import DistributionCore


class ShiftScale(DistributionCore):
    """
    Shift-Scaling transformation.

    Linear transforms any distribution of the form `A*X+b` where A is a scaling
    matrix and `b` is a shift vector.

    Args:
        dist (Distribution):
            The underlying distribution to be scaled.
        shift (float, Sequence[float], Distribution):
            Mean vector.
        scale (float, Sequence[float], Distribution):
            Covariance matrix or variance vector if scale is a 1-d vector.
            If omitted, assumed to be 1.
        rotation (Sequence[int], Sequence[Sequence[bool]]):
            The order of which to resolve conditionals. Either as a sequence of
            column rotations, or as a permutation matrix.
            Defaults to `range(len(distribution))` which is the same as
            `p(x0), p(x1|x0), p(x2|x0,x1), ...`.

    """

    def __init__(
            self,
            dist,
            shift=0,
            scale=1,
            rotation=None,
            repr_args=None,
    ):
        assert isinstance(dist, Distribution), "'dist' should be a distribution"
        if repr_args is None:
            repr_args = dist._repr_args[:]
        if not isinstance(scale, (int, float)) or scale != 1:
            repr_args += ["scale=%s" % scale]
        if not isinstance(shift, (int, float)) or shift != 0:
            repr_args += ["shift=%s" % shift]
        if rotation is not None:
            repr_args += ["rotation=%s" % list(rotation)]
        length = max([(len(param) if isinstance(param, Distribution)
                        else len(numpy.atleast_1d(param)))
                        for param in [dist, shift, scale]]+[1])
        dependencies = [set([idx]) for idx in self._declare_dependencies(length)]

        super(ShiftScale, self).__init__(
            shift=shift,
            scale=scale,
            rotation=rotation,
            dependencies=dependencies,
            repr_args=repr_args,
        )
        if len(dist) == 1 and len(self) > 1:
            dist = chaospy.Iid(dist, len(self))
        permute = numpy.zeros((len(self._rotation), len(self._rotation)), dtype=int)
        permute[numpy.arange(len(self._rotation), dtype=int), self._rotation] = 1
        self._permute = permute
        self._dist = dist

    def _ppf(self, qloc, shift, scale):
        zloc = self._dist.inv(qloc)
        out = (scale.T*zloc.T+shift.T).T
        return out

    def _cdf(self, xloc, shift, scale):
        zloc = ((xloc.T-shift.T)/scale.T).T
        out = self._dist.fwd(zloc)
        return out

    def _pdf(self, xloc, shift, scale):
        zloc = ((xloc.T-shift.T)/scale.T).T
        out = (self._dist.pdf(zloc, decompose=True).T/scale.T).T
        assert xloc.shape == out.shape
        return out

    def _mom(self, kloc, shift, scale):
        assert not isinstance(scale, Distribution)
        poly = numpoly.variable(len(self))
        poly = numpoly.sum(scale*poly, axis=-1)+shift
        poly = numpoly.set_dimensions(numpoly.prod(poly**kloc), len(self))
        out = sum(self._dist.mom(key)*coeff
                  for key, coeff in zip(poly.exponents, poly.coefficients))
        return out

    def _ttr(self, kloc, shift, scale):
        coeff0, coeff1 = self._dist._get_ttr(kloc)
        coeff0 = coeff0*scale+shift
        coeff1 = coeff1*scale*scale
        return coeff0, coeff1

    def _lower(self, shift, scale):
        if isinstance(shift, Distribution):
            shift = shift._get_lower(cache={})
        if isinstance(scale, Distribution):
            scale = scale._get_lower(cache={})
        return scale.dot(self._dist.lower)+shift

    def _upper(self, shift, scale):
        if isinstance(shift, Distribution):
            shift = shift._get_upper(cache={})
        if isinstance(scale, Distribution):
            scale = scale._get_upper(cache={})
        return scale.dot(self._dist.upper)+shift
