import numpy

from ..baseclass import Dist
from .. import evaluation


class BinaryOperator(Dist):

    matrix = False

    def __init__(self, left, right):
        """
        Args:
            left (Dist, numpy.ndarray):
                Left hand side.
            right (Dist, numpy.ndarray):
                Right hand side.
        """
        Dist.__init__(self, left=left, right=right)

    def _precedence_order(self):
        """Precedence order of the various dimensions."""
        left = self.prm["left"]
        right = self.prm["right"]
        if isinstance(left, Dist):
            indices = left._precedence_order()
            if isinstance(right, Dist):
                assert indices == right._precedence_order()
        elif isinstance(right, Dist):
            indices = right._precedence_order()
        else:
            indices = list(range(len(self)))
        return indices


    def _cdf(self, xloc, left, right, cache):
        """Cumulative distribution function."""
        left = evaluation.get_forward_cache(left, cache)
        right = evaluation.get_forward_cache(right, cache)

        if isinstance(left, Dist):
            if isinstance(right, Dist):
                raise evaluation.DependencyError(
                    "under-defined distribution {} or {}".format(left, right))

            if not self.matrix:
                right = (numpy.asfarray(right).T+numpy.zeros(xloc.shape).T).T
            xloc = self._pre_fwd_right(xloc, right)
            uloc = evaluation.evaluate_forward(left, xloc, cache=cache)
            uloc = self._post_fwd(uloc, right)

        elif not isinstance(right, Dist):
            uloc = self._alt_fwd(xloc, left, right)

        else:
            if not self.matrix:
                left = (numpy.asfarray(left).T+numpy.zeros(xloc.shape).T).T
            xloc = self._pre_fwd_left(xloc, left)
            uloc = evaluation.evaluate_forward(right, xloc, cache=cache)
            uloc = self._post_fwd(uloc, left)

        assert uloc.shape == xloc.shape
        return uloc
