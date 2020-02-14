"""Baseclass for unary operators."""
from .. import evaluation, approximation
from ..baseclass import Dist


class UnaryOperator(Dist):

    def _precedence_order(self):
        """Precedence order of the various dimensions."""
        dist = self.prm["dist"]
        if isinstance(dist, Dist):
            indices = dist._precedence_order()
        else:
            indices = list(range(len(self)))
        return indices

    def _pdf(self, xloc, dist, cache, **kwargs):
        """Probability density function."""
        xloc_ = self._pre_fwd(xloc, **kwargs)
        qloc = evaluation.evaluate_density(dist, xloc_, cache=cache)
        qloc *= self._post_pdf(xloc, **kwargs)
        return qloc

    def _cdf(self, xloc, dist, cache, **kwargs):
        """Cumulative distribution function."""
        xloc = self._pre_fwd(xloc, **kwargs)
        uloc = evaluation.evaluate_forward(dist, xloc, cache=cache)
        return self._post_fwd(uloc, **kwargs)

    def _ppf(self, q, dist, cache, **kwargs):
        """Point percentile function."""
        q = self._pre_inv(q, **kwargs)
        uloc = evaluation.evaluate_inverse(dist, q, cache=cache)
        return self._post_inv(uloc, **kwargs)

    def _lower(self, dist, cache, **kwargs):
        """Distribution bounds."""
        uloc = evaluation.evaluate_lower(dist, cache=cache)
        return self._post_inv(uloc, **kwargs)

    def _upper(self, dist, cache, **kwargs):
        """Distribution bounds."""
        uloc = evaluation.evaluate_upper(dist, cache=cache)
        return self._post_inv(uloc, **kwargs)

    def _mom(self, x, dist, cache, **kwargs):
        return approximation.approximate_moment(self, x)

    def __len__(self):
        return len(self.prm["dist"])

    def _fwd_cache(self, cache):
        kwargs = {key: evaluation.get_forward_cache(value, cache)
                  for key, value in self.prm.items()}
        if not any(isinstance(value, Dist) for value in kwargs.values()):
            return self._post_inv(kwargs.pop("dist"), **kwargs)
        return self

    def _inv_cache(self, cache):
        kwargs = {key: evaluation.get_forward_cache(value, cache)
                  for key, value in self.prm.items()}
        if not any(isinstance(value, Dist) for value in kwargs.values()):
            return self._pre_fwd(kwargs.pop("dist"), **kwargs)
        return self
