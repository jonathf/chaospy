"""Baseclass for all Copulas."""
import numpy
import chaospy

from .distribution import Distribution


class CopulaDistribution(Distribution):
    """Baseclass for all Copulas."""

    def __init__(self, dist, trans, rotation=None, repr_args=None):
        r"""
        Args:
            dist (Distribution):
                Distribution to wrap the copula around.
            trans (Distribution):
                The copula wrapper `[0,1]^D \into [0,1]^D`.

        """
        assert len(dist) == len(trans), "Copula length missmatch"
        accumulant = set()
        dependencies = [deps.copy() for deps in dist._dependencies]
        for idx, _ in sorted(enumerate(trans._dependencies), key=lambda x: len(x[1])):
            accumulant.update(dist._dependencies[idx])
            dependencies[idx] = accumulant.copy()

        super(CopulaDistribution, self).__init__(
            parameters=dict(dist=dist, trans=trans),
            dependencies=dependencies,
            rotation=rotation,
            repr_args=repr_args,
        )

    def get_parameters(self, idx, cache, assert_numerical=True):
        parameters = super(CopulaDistribution, self).get_parameters(
            idx, cache, assert_numerical=assert_numerical)
        if idx is None:
            del parameters["idx"]
        return parameters

    def _lower(self, idx, dist, trans, cache):
        return dist._get_lower(idx, cache=cache)

    def _upper(self, idx, dist, trans, cache):
        return dist._get_upper(idx, cache=cache)

    def _cdf(self, xloc, idx, dist, trans, cache):
        output = dist._get_fwd(xloc, idx, cache=cache)
        output = trans._get_fwd(output, idx, cache=cache)
        return output

    def _ppf(self, qloc, idx, dist, trans, cache):
        qloc = trans._get_inv(qloc, idx, cache=cache)
        xloc = dist._get_inv(qloc, idx, cache=cache)
        return xloc

    def _pdf(self, xloc, idx, dist, trans, cache):
        density = dist._get_pdf(xloc, idx, cache=cache.copy())
        return trans._get_pdf(
            dist._get_fwd(xloc, idx, cache=cache), idx, cache=cache)*density
