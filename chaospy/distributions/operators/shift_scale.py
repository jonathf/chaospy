import numpy
from scipy.special import comb
import numpoly
import chaospy

from ..baseclass import Dist
from .. import evaluation


class ShiftScale(Dist):
    """
    Shift-Scaling transformation.

    Linear transforms any distribution of the form `A*X+b` where A is a scaling
    matrix and `b` is a shift vector.

    Args:
        dist (Dist):
            The underlying distribution to be scaled.
        shift (float, Sequence[float], Dist):
            Mean vector.
        scale (float, Sequence[Sequence[float]], Dist):
            Covariance matrix or variance vector if scale is a 1-d vector.
            If omitted, assumed to be 1.
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

    def __init__(self, dist, shift=0, scale=1, rotation=None):
        assert isinstance(dist, Dist), "'dist' should be a distribution"
        shift = shift if isinstance(shift, Dist) else numpy.atleast_1d(shift)
        self._scale_is_dist = isinstance(scale, Dist)
        scale = scale if self._scale_is_dist else numpy.atleast_1d(scale)
        length = max(len(dist), len(shift), len(scale))

        if isinstance(shift, Dist):
            if len(shift) == 1 and length > 1:
                shift = chaospy.Iid(shift, length)
        else:
            assert shift.ndim == 1, "Parameter 'shift' have too many dimensions"
            if len(shift) == 1 and length > 1:
                shift = numpy.repeat(shift.item(), length)
        assert len(shift) == length

        if len(dist) == 1 and length > 1:
            dist = chaospy.Iid(dist, length)

        if self._scale_is_dist:
            if len(scale) == 1 and length > 1:
                scale = chaospy.Iid(scale, length)
        else:
            scale = numpy.asarray(scale)
            assert scale.ndim <= 1, (
                "'scale' must either be scalar or vector")
            if scale.size == 1:
                scale = numpy.ones(length)*scale
            assert scale.shape == (length,), (
                "Parameters 'shift' and 'scale' have shape mismatch.")
        assert len(scale) == length

        self._dist = dist
        self._rotate(rotation)
        super(ShiftScale, self).__init__(shift=shift, scale=scale)

    def _rotate(self, rotation=None):
        if rotation is not None:
            self._rotation = list(rotation)
        else:
            self._rotation = [key for key, _ in sorted(enumerate(self._dist._dependencies), key=lambda x: len(x[1]))]
        accumulant = set()
        self._dependencies = [deps.copy() for deps in self._dist._dependencies]
        for idx in self._rotation:
            accumulant.update(self._dist._dependencies[idx])
            self._dependencies[idx] = accumulant.copy()

        self._permute = numpy.zeros((len(self._rotation), len(self._rotation)), dtype=int)
        self._permute[numpy.arange(len(self._rotation), dtype=int), self._rotation] = 1

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
        out = (self._dist.pdf(zloc, decompose=True).T/scale.T)
        return out

    def _mom(self, kloc, shift, scale):
        poly = numpoly.variable(len(self))
        if not self._scale_is_dist:
            poly = numpoly.sum(scale*poly, axis=-1)+shift
        poly = numpoly.set_dimensions(numpoly.prod(poly**kloc), len(self))
        out = sum(self._dist.mom(key)*coeff
                  for key, coeff in zip(poly.exponents, poly.coefficients))
        return out

    def _ttr(self, kloc, shift, scale):
        assert not self._scale_is_dist
        assert numpy.allclose(numpy.diag(numpy.diag(scale)), scale), (
            "TTR require stochastically independent components.")
        coeff0, coeff1 = evaluation.evaluate_recurrence_coefficients(self._dist, kloc)
        coeff0 = coeff0*scale+shift
        coeff1 = coeff1*scale*scale
        return coeff0, coeff1

    def _lower(self, shift, scale):
        return scale.dot(self._dist.lower)+shift

    def _upper(self, shift, scale):
        return scale.dot(self._dist.upper)+shift
