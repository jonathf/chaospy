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
        repr_args += chaospy.format_repr_kwargs(lower=(lower, 0), upper=(upper, 1))

        dependencies, parameters, rotation, = chaospy.declare_dependencies(
            distribution=self,
            parameters=dict(lower=lower, upper=upper),
            rotation=rotation,
        )
        super(LowerUpperDistribution, self).__init__(
            parameters=parameters,
            dependencies=dependencies,
            repr_args=repr_args,
        )
        self._dist = dist

    def get_parameters(self, idx, cache, assert_numerical=True):
        parameters = super(LowerUpperDistribution, self).get_parameters(
            idx, cache, assert_numerical=assert_numerical)
        lower = parameters["lower"]
        if isinstance(lower, Distribution):
            lower = lower._get_cache(idx, cache, get=0)
        if idx is not None and len(lower) > 1:
            lower = lower[idx]
        upper = parameters["upper"]
        if isinstance(upper, Distribution):
            upper = upper._get_cache(idx, cache, get=0)
        if idx is not None and len(upper) > 1:
            upper = upper[idx]
        assert not assert_numerical or not (isinstance(lower, Distribution) or
                                            isinstance(upper, Distribution))

        lower0 = self._dist._get_lower(idx, cache.copy())
        upper0 = self._dist._get_upper(idx, cache.copy())
        scale = (upper-lower)/(upper0-lower0)
        shift = lower-lower0*(upper-lower)/(upper0-lower0)
        parameters = self._dist.get_parameters(idx, cache, assert_numerical=assert_numerical)
        return dict(dist=self._dist, scale=scale, shift=shift, parameters=parameters)

    def _lower(self, dist, scale, shift, parameters):
        return dist._lower(**parameters)*scale+shift

    def _upper(self, dist, scale, shift, parameters):
        return dist._upper(**parameters)*scale+shift

    def _ppf(self, qloc, dist, scale, shift, parameters):
        return dist._ppf(qloc, **parameters)*scale+shift

    def _cdf(self, xloc, dist, scale, shift, parameters):
        return dist._cdf((xloc-shift)/scale, **parameters)

    def _pdf(self, xloc, dist, scale, shift, parameters):
        return dist._pdf((xloc-shift)/scale, **parameters)/scale

    def _mom(self, kloc, dist, scale, shift, parameters):
        poly = numpoly.variable(len(self))
        poly = numpoly.sum(scale*poly, axis=-1)+shift
        poly = numpoly.set_dimensions(numpoly.prod(poly**kloc), len(self))
        out = sum(dist._get_mom(key)*coeff
                  for key, coeff in zip(poly.exponents, poly.coefficients))
        return out

    def _ttr(self, kloc, dist, scale, shift, parameters):
        coeff0, coeff1 = dist._ttr(kloc, **parameters)
        coeff0 = coeff0*scale+shift
        coeff1 = coeff1*scale*scale
        return coeff0, coeff1
