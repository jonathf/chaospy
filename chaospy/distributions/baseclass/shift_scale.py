"""Shift-Scale transformation"""
import numpy
from scipy.special import comb
import numpoly
import chaospy

from .distribution import Distribution


class ShiftScaleDistribution(Distribution):
    """
    Shift-Scale transformation.

    Linear transforms any distribution of the form `A*X+b` where A is a
    scaling matrix and `b` is a shift vector.

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
        repr_args += chaospy.format_repr_kwargs(scale=(scale, 1))
        repr_args += chaospy.format_repr_kwargs(shift=(shift, 0))
        length = len(dist) if len(dist) > 1 else None
        dependencies, parameters, rotation = chaospy.declare_dependencies(
            distribution=self,
            parameters=dict(shift=shift, scale=scale),
            rotation=rotation,
            is_operator=True,
            length=length,
            extra_parameters=dict(dist=dist),
        )
        super(ShiftScaleDistribution, self).__init__(
            parameters=parameters,
            rotation=rotation,
            dependencies=dependencies,
            repr_args=repr_args,
        )
        self._dist = dist
        permute = numpy.zeros((len(self._rotation), len(self._rotation)), dtype=int)
        permute[numpy.arange(len(self._rotation), dtype=int), self._rotation] = 1
        self._permute = permute

    def get_parameters(self, idx, cache, assert_numerical=True):

        shift = self._parameters["shift"]
        if isinstance(shift, Distribution):
            shift = shift._get_cache(idx, cache, get=0)
        elif idx is not None and len(shift) > 1:
            shift = shift[idx]
        assert not isinstance(shift, Distribution), shift

        scale = self._parameters["scale"]
        if isinstance(scale, Distribution):
            scale = scale._get_cache(idx, cache, get=0)
        elif idx is not None and len(scale) > 1:
            scale = scale[idx]
        assert not isinstance(scale, Distribution), scale
        assert numpy.all([scale]) > 0, "condition not satisfied: `scale > 0`"

        assert not assert_numerical or not (isinstance(shift, Distribution) or
                                            isinstance(scale, Distribution))

        return dict(idx=idx, dist=self._dist, shift=shift, scale=scale, cache=cache)

    def _ppf(self, qloc, idx, dist, shift, scale, cache):
        return dist._get_inv(qloc, idx, cache=cache)*scale+shift

    def _cdf(self, xloc, idx, dist, shift, scale, cache):
        return dist._get_fwd((xloc-shift)/scale, idx, cache=cache)

    def _pdf(self, xloc, idx, dist, shift, scale, cache):
        return dist._get_pdf((xloc-shift)/scale, idx, cache=cache)/scale

    def get_mom_parameters(self):
        parameters = self.get_parameters(
            idx=None, cache={}, assert_numerical=False)
        del parameters["idx"]
        del parameters["cache"]
        return parameters

    def _mom(self, kloc, dist, shift, scale):
        poly = numpoly.variable(len(self))
        poly = numpoly.sum(scale*poly, axis=-1)+shift
        poly = numpoly.set_dimensions(numpoly.prod(poly**kloc), len(self))
        out = sum([dist._get_mom(key)*coeff
                   for key, coeff in zip(poly.exponents, poly.coefficients)])
        return out

    def get_ttr_parameters(self, idx):
        parameters = self.get_parameters(
            idx=idx, cache={}, assert_numerical=False)
        del parameters["cache"]
        return parameters

    def _ttr(self, kloc, idx, dist, shift, scale):
        coeff0, coeff1 = dist._get_ttr(kloc, idx)
        coeff0 = coeff0*scale+shift
        coeff1 = coeff1*scale*scale
        return coeff0, coeff1

    def _lower(self, idx, dist, shift, scale, cache):
        return dist._get_lower(idx, cache=cache)*scale+shift

    def _upper(self, idx, dist, shift, scale, cache):
        return dist._get_upper(idx, cache=cache)*scale+shift
