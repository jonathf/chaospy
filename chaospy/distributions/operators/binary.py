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
        if not isinstance(left, Dist):
            left = numpy.atleast_1d(left)
        if not isinstance(right, Dist):
            right = numpy.atleast_1d(right)
        length = max(len(left), len(right))
        self._dependencies = [set() for _ in range(length)]
        if isinstance(left, Dist):
            if len(left) == 1:
                self._dependencies = [
                    dep.union(left._dependencies[0])
                    for dep in self._dependencies
                ]
            else:
                self._dependencies = [
                    dep.union(other)
                    for dep, other in zip(self._dependencies, left._dependencies)
                ]
        if isinstance(right, Dist):
            if len(right) == 1:
                self._dependencies = [
                    dep.union(right._dependencies[0])
                    for dep in self._dependencies
                ]
            else:
                self._dependencies = [
                    dep.union(other)
                    for dep, other in zip(self._dependencies, right._dependencies)
                ]
        Dist.__init__(self, left=left, right=right)

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

    def __len__(self):
        """Length of binary operator."""
        left = self.prm["left"]
        if not isinstance(left, Dist):
            left = numpy.atleast_1d(left)
        right = self.prm["right"]
        if not isinstance(right, Dist):
            right = numpy.atleast_1d(right)
        return max(len(left), len(right))
