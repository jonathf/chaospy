"""Baseclass for all conditional distributions."""
import numpy
import chaospy

from .distribution import Distribution


class ItemDistribution(Distribution):
    """Baseclass for an index of multivariate distribution."""

    def __init__(self, index, parent):
        super(ItemDistribution, self).__init__(
            parameters=dict(index=index, parent=parent),
            dependencies=[parent._dependencies[index].copy()],
            rotation=[0],
            repr_args=[index, parent],
        )

    def get_parameters(self, idx, cache, assert_numerical=True):
        """Get distribution parameters."""
        assert idx == 0 or idx is None, "Indexes only have a single component"
        if idx == 0:
            idx = int(self._parameters["index"])
        parent = self._parameters["parent"]
        parameters = parent.get_parameters(
            idx, cache, assert_numerical=assert_numerical)
        return dict(parent=parent, parameters=parameters)

    def __repr__(self):
        return "ItemDistribution(%(index)d, %(parent)s)" % self._parameters

    def _lower(self, parent, parameters):
        return parent._lower(**parameters)

    def _upper(self, parent, parameters):
        return parent._upper(**parameters)

    def _ppf(self, xloc, parent, parameters):
        return parent._ppf(xloc, **parameters)

    def _cdf(self, xloc, parent, parameters):
        return parent._cdf(xloc, **parameters)

    def _pdf(self, xloc, parent, parameters):
        return parent._pdf(xloc, **parameters)

    def _mom(self, kloc, parent, parameters):
        idx = int(self._parameters["index"])
        kloc = kloc*numpy.eye(len(parent), dtype=int)[idx]
        return parent._mom(kloc, **parameters)

    def _ttr(self, kloc, parent, parameters):
        raise chaospy.StochasticallyDependentError("TTR not supported")

    def _cache(self, idx, cache, get):
        if idx is None:
            return self
        assert idx == 0
        idx = int(self._parameters["index"])
        parent = self._parameters["parent"]
        return parent._get_cache(idx, cache, get)
