"""Lower-Upper transformation."""
import numpy
from scipy.special import comb
import numpoly
import chaospy

from .distribution import Distribution


class LowerUpperDistribution(Distribution):
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
        repr_args += chaospy.format_repr_kwargs(lower=(lower, 0), upper=(upper, 1))

        dependencies, parameters, rotation, = chaospy.declare_dependencies(
            distribution=self,
            parameters=dict(lower=lower, upper=upper),
            is_operator=True,
            rotation=rotation,
            extra_parameters=dict(dist=dist),
        )
        super(LowerUpperDistribution, self).__init__(
            parameters=parameters,
            dependencies=dependencies,
            rotation=rotation,
            repr_args=repr_args,
        )
        self._dist = dist

    def _get_lower_upper(self, idx, cache):
        parameters = super(LowerUpperDistribution, self).get_parameters(idx, cache)
        lower = parameters["lower"]
        if isinstance(lower, Distribution):
            lower = lower._get_cache(idx, cache, get=0)
        upper = parameters["upper"]
        if isinstance(upper, Distribution):
            upper = upper._get_cache(idx, cache, get=0)
        return lower, upper

    @staticmethod
    def _lower_upper_to_shift_scale(lower, upper, lower0, upper0):
        scale = (upper-lower)/(upper0-lower0)
        shift = lower-lower0*(upper-lower)/(upper0-lower0)
        return shift, scale

    def get_parameters(self, idx, cache):
        lower, upper = self._get_lower_upper(idx, cache)
        assert not isinstance(lower, Distribution) and not isinstance(upper, Distribution)
        [lower], [upper] = lower, upper
        assert upper > lower, "condition not satisfied: `upper > lower`"
        lower0 = self._dist._get_lower(idx, cache.copy())
        upper0 = self._dist._get_upper(idx, cache.copy())
        shift, scale = self._lower_upper_to_shift_scale(lower, upper, lower0, upper0)
        scale = (upper-lower)/(upper0-lower0)
        shift = lower-lower0*(upper-lower)/(upper0-lower0)
        return dict(idx=idx, scale=scale, shift=shift, cache=cache)

    def _ppf(self, qloc, idx, scale, shift, cache):
        return self._dist._get_inv(qloc, idx, cache)*scale+shift

    def _cdf(self, xloc, idx, scale, shift, cache):
        return self._dist._get_fwd((xloc-shift)/scale, idx, cache)

    def _pdf(self, xloc, idx, scale, shift, cache):
        return self._dist._get_pdf((xloc-shift)/scale, idx, cache)/scale

    def get_lower_parameters(self, idx, cache):
        lower, _ = self._get_lower_upper(idx, cache)
        if isinstance(lower, Distribution):
            lower = lower._get_lower(idx, cache)
        return dict(lower=lower[idx])

    def _lower(self, lower):
        return lower

    def get_upper_parameters(self, idx, cache):
        _, upper = self._get_lower_upper(idx, cache)
        if isinstance(upper, Distribution):
            upper = upper._get_upper(idx, cache)
        return dict(upper=upper[idx])

    def _upper(self, upper):
        return upper

    def get_mom_parameters(self):
        lower, upper = self._get_lower_upper(idx=None, cache={})
        assert not isinstance(lower, Distribution) and not isinstance(upper, Distribution)
        shift, scale = self._lower_upper_to_shift_scale(
            lower, upper, lower0=self._dist.lower, upper0=self._dist.upper)
        return dict(shift=shift, scale=scale)

    def _mom(self, kloc, scale, shift):
        poly = numpoly.variable(len(self))
        poly = numpoly.sum(scale*poly, axis=-1)+shift
        poly = numpoly.set_dimensions(numpoly.prod(poly**kloc), len(self))
        out = sum(self._dist._get_mom(key)*coeff
                  for key, coeff in zip(poly.exponents, poly.coefficients))
        return out

    def get_ttr_parameters(self, idx):
        lower, upper = self._get_lower_upper(idx=idx, cache={})
        assert not isinstance(lower, Distribution) and not isinstance(upper, Distribution)
        shift, scale = self._lower_upper_to_shift_scale(
            lower, upper, lower0=self._dist.lower, upper0=self._dist.upper)
        return dict(idx=idx, shift=shift, scale=scale)

    def _ttr(self, kloc, idx, scale, shift):
        coeff0, coeff1 = self._dist._get_ttr(kloc, idx)
        coeff0 = coeff0*scale+shift
        coeff1 = coeff1*scale*scale
        return coeff0, coeff1
