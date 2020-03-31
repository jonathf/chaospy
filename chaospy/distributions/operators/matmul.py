import chaospy
import numpy

from ..baseclass import Dist
from .. import evaluation
from .binary import BinaryOperator


class Matmul(BinaryOperator):
    """Multiplication."""

    matrix = True

    def __init__(self, left, right):
        """
        Args:
            left (Dist, numpy.ndarray):
                Left hand side.
            right (Dist, numpy.ndarray):
                Right hand side.
        """

        if isinstance(left, Dist) and not isinstance(right, Dist):
            right = numpy.asarray(right)
            if not right.shape:
                raise ValueError("matmul: right side does not have enough dimensions")
            if len(right.shape) == 1:
                right = numpy.diag(right)

            if len(left) != right.shape[0]:
                raise ValueError("matmul: shapes incompatible (%d,) and %s" %
                                 (len(left), right.shape))

        elif not isinstance(left, Dist) and isinstance(right, Dist):
            left = numpy.asarray(left)
            if not left.shape:
                raise ValueError(
                    "matmul: left side does not have enough dimensions")
            if len(left.shape) == 1:
                left = numpy.diag(left)

            if left.shape[-1] != len(right):
                raise ValueError("matmul: shapes incompatible %s and (%d,)" %
                                 (left.shape, len(right)))

        elif not isinstance(left, Dist) and not isinstance(right, Dist):
            raise ValueError(
                "matmul: at least one argument must be a distribution")

        Dist.__init__(self, left=left, right=right)


    def _lower(self, left, right, cache):
        """Distribution lower bounds."""
        if isinstance(left, Dist):
            left_upper = evaluation.evaluate_upper(left, cache=cache)
            left_lower = evaluation.evaluate_lower(left, cache=cache)

            if isinstance(right, Dist):
                right_upper = evaluation.evaluate_upper(right, cache=cache)
                right_lower = evaluation.evaluate_lower(right, cache=cache)

                out = numpy.min(numpy.broadcast_arrays(
                    left_lower*right_lower,
                    left_lower*right_upper,
                    left_upper*right_lower,
                    left_upper*right_upper,
                ), axis=0)

            else:
                out1 = numpy.dot(left_lower.T, right).T
                out2 = numpy.dot(left_upper.T, right).T
                out = numpy.min([out1, out2], axis=0)

        elif not isinstance(right, Dist):
            out = left*right

        else:
            right_upper = evaluation.evaluate_upper(right, cache=cache)
            right_lower = evaluation.evaluate_lower(right, cache=cache)
            out = numpy.min([numpy.dot(left, right_lower),
                             numpy.dot(left, right_upper)], axis=0)

        return out

    def _upper(self, left, right, cache):
        """Distribution upper bounds."""
        if isinstance(left, Dist):
            left_lower = evaluation.evaluate_lower(left, cache=cache)
            left_upper = evaluation.evaluate_upper(left, cache=cache)

            if isinstance(right, Dist):
                right_lower = evaluation.evaluate_lower(right, cache=cache)
                right_upper = evaluation.evaluate_upper(right, cache=cache)

                out = numpy.max(numpy.broadcast_arrays(
                    (left_lower.T*right_lower.T).T,
                    (left_lower.T*right_upper.T).T,
                    (left_upper.T*right_lower.T).T,
                    (left_upper.T*right_upper.T).T,
                ), axis=0)

            else:
                out = numpy.max([numpy.dot(left_lower, right),
                                 numpy.dot(left_upper, right)], axis=0)

        elif not isinstance(right, Dist):
            out = left*right

        else:
            right_lower = evaluation.evaluate_lower(right, cache=cache)
            right_upper = evaluation.evaluate_upper(right, cache=cache)
            out = numpy.max([numpy.dot(left, right_lower),
                             numpy.dot(left, right_upper)], axis=0)

        return out

    def _pre_fwd_left(self, xloc, other):
        Ci = numpy.linalg.inv(other)
        return numpy.dot(Ci, xloc)

    def _pre_fwd_right(self, xloc, other):
        Ci = numpy.linalg.inv(other)
        return numpy.dot(xloc.T, Ci).T

    def _post_fwd(self, uloc, other):
        return uloc

    def _alt_fwd(self, xloc, left, right):
        return 0.5*(numpy.dot(left, right) == xloc)

    def _ppf(self, uloc, left, right, cache):
        """Point percentile function."""
        left = evaluation.get_inverse_cache(left, cache)
        right = evaluation.get_inverse_cache(right, cache)

        if isinstance(left, Dist):
            if isinstance(right, Dist):
                raise evaluation.DependencyError(
                    "under-defined distribution {} or {}".format(left, right))

            xloc = evaluation.evaluate_inverse(left, uloc, cache=cache)
            xloc = numpy.dot(xloc.T, right).T
            assert uloc.shape == xloc.shape

        elif not isinstance(right, Dist):
            xloc = numpy.dot(left, right)

        else:
            xloc = evaluation.evaluate_inverse(right, uloc, cache=cache)
            xloc = numpy.dot(left, xloc)

        return xloc

    def _pdf(self, xloc, left, right, cache):
        """Probability density function."""
        left = evaluation.get_forward_cache(left, cache)
        right = evaluation.get_forward_cache(right, cache)

        if isinstance(left, Dist):
            if isinstance(right, Dist):
                raise evaluation.DependencyError(
                    "under-defined distribution {} or {}".format(left, right))

            Ci = numpy.linalg.inv(right)
            xloc = numpy.dot(xloc.T, Ci).T

            pdf = evaluation.evaluate_density(left, xloc, cache=cache)
            pdf = numpy.dot(pdf.T, Ci).T
            assert pdf.shape == xloc.shape

        elif not isinstance(right, Dist):
            pdf = numpy.inf

        else:
            Ci = numpy.linalg.inv(left)
            xloc = numpy.dot(Ci, xloc)

            pdf = evaluation.evaluate_density(right, xloc, cache=cache)
            pdf = numpy.dot(Ci, pdf)

        return numpy.abs(pdf)

    def _mom(self, key, left, right, cache):
        """Statistical moments."""
        if evaluation.get_dependencies(left, right):
            raise evaluation.DependencyError(
                "product of dependent distributions not feasible: "
                "{} and {}".format(left, right)
            )

    def _ttr(self, kloc, left, right, cache):
        """Three terms recursion coefficients."""
        raise evaluation.DependencyError("matmul causes dependencies.")

    def __str__(self):
        if self._repr is not None:
            return super(Mul, self).__str__()
        return (self.__class__.__name__ + "(" + str(self.prm["left"]) +
                ", " + str(self.prm["right"]) + ")")

    def __len__(self):
        out1 = out2 = 1
        try:
            out1 = len(self.prm["left"])
        except TypeError:
            pass
        try:
            out2 = len(self.prm["right"])
        except TypeError:
            pass
        return max(out1, out2)

    def _fwd_cache(self, cache):
        left = evaluation.get_forward_cache(self.prm["left"], cache)
        right = evaluation.get_forward_cache(self.prm["right"], cache)
        if not isinstance(left, Dist) and not isinstance(right, Dist):
            return left*right
        return self

    def _inv_cache(self, cache):
        left = evaluation.get_inverse_cache(self.prm["left"], cache)
        right = evaluation.get_inverse_cache(self.prm["right"], cache)
        if not isinstance(left, Dist) and not isinstance(right, Dist):
            return left*right
        return self


def mul(left, right):
    """
    Distribution multiplication.

    Args:
        left (Dist, numpy.ndarray) : left hand side.
        right (Dist, numpy.ndarray) : right hand side.
    """
    from .mv_mul import MvMul
    length = max(left, right)
    if length == 1:
        return Mul(left, right)
    return MvMul(left, right)
