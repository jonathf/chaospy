r"""
Archimedean copula definition.

All Copulas can be converted into Cumulative Distribution Functions (CDF) using
the following formula:

.. math::
  \begin{eqnarray}
    F(x_1) &= \frac{C(x_1, 1, 1, ...)}{C(1, 1, 1)} \\
    F(x_2\mid x_1) &= \frac{d/dx_1 C(x_1, 1, 1, ...)}{d/dx_1 C(1, 1, 1, ...)} \\
    F(x_3\mid x_1,x_2) &=
        \frac{d/dx_1 d/dx_2 C(x_1, x_2, x_3, 1, ...) }{
              d/dx_1 d/dx_2 C(x_1, x_2, 1, 1, ...)}
  \end{eqnarray}

And thereby density functions:

.. math::
  \begin{eqnarray}
    P(x_1) &= \frac{d/dx_1 C(x_1, 1, 1, ...)}{C(1, 1, 1)} \\
    P(x_2\mid x_1) &= \frac{d/dx_1 d/dx_2 C(x_1, 1, 1, ...)}{d/dx_1 C(1, 1, 1, ...)} \\
    P(x_3\mid x_1,x_2) &=
        \frac{d/dx_1 d/dx_2 d/dx_3 C(x_1, x_2, x_3, 1, ...) }{
              d/dx_1 d/dx_2 C(x_1, x_2, 1, 1, ...)}
  \end{eqnarray}

The general definition of an Archimedean is:

.. math::
    C(x_1, x_2, x_3, ...) = iphi( phi(x_1) + phi(x_2) + phi(x_3) + ... )

Applying the Copula to CDF formula on the Archimedean, we need various partial
derivatives:

.. math::
  \begin{eqnarray}
    C &= iphi(phi(x_1)+phi(x_2)+phi(x_3)+...) \\
    d/dx_1 C &= iphi'( phi(x_1)+phi(x_2)+phi(x_3)+...) phi'(x_1) \\
    d/dx_1 d/dx_2 C &= iphi''(phi(x_1)+phi(x_2)+phi(x_3)+...) phi'(x_1) phi'(x_2) \\
    d/dx_1 d/dx_2 d/dx_3 C &= iphi'''(phi(x_1)+phi(x_2)+phi(x_3)+...) phi'(x_1) phi'(x_2) phi'(x_3) \\
    & \vdots \\
    d/dx_1 \cdots d/dx_n C &= iphi^{(n)}(phi(x_1)+phi(x_2)+phi(x_3)+...) phi'(x_1) \cdots phi'(x_n)
  \end{eqnarray}

As a helper function, we introduce sigma:

.. math::
    \sigma(r, n, \theta) = \prod_{d=0}^{n-1}(-1/\theta-d)*r^{-1/theta-n}

It has the convenient property:

.. math::
    \sigma'(r, n, \theta) = \prod_{d=0}^{n}(-1/\theta-d)*r^{-1/theta-n-1} = \sigma(r, n+1, \theta)
"""
import numpy

from ..baseclass import Distribution


class Archimedean(Distribution):

    def __init__(self, length, theta=1., rotation=None):
        dependencies = [{idx} for idx in self._declare_dependencies(length)]
        super(Archimedean, self).__init__(
            parameters=dict(theta=float(theta)),
            dependencies=dependencies,
            rotation=rotation,
        )

    def _lower(self, theta, cache):
        return numpy.zeros(len(self))

    def _upper(self, theta, cache):
        return numpy.ones(len(self))

    def _cdf(self, x_loc, theta, cache):
        x_loc = x_loc[self._rotation]
        out = numpy.zeros(x_loc.shape)
        loc = numpy.ones(x_loc.shape)
        for order in range(len(self)):
            out[order] = self._copula(loc, theta, order)
            loc[order] = x_loc[order]
            ret_val = self._copula(loc, theta, order)
            out[order] = numpy.where(out[order] != 0, ret_val/numpy.where(out[order] != 0, out[order], 1), 1)
        return out

    def _pdf(self, x_loc, theta, cache):
        x_loc = x_loc[self._rotation]
        out = numpy.zeros(x_loc.shape)
        loc = numpy.ones(x_loc.shape)
        for idx in range(len(self)):
            out[idx] = self._copula(loc, theta, idx)
            loc[idx] = x_loc[idx]
            index = out[idx] != 0
            ret_val = self._copula(loc, theta, idx+1)
            out[idx] = numpy.where(index, ret_val/out[idx], 1)
        return out

    def _copula(self, x_loc, theta, order=0):
        out = numpy.sum(self._phi(x_loc, theta), 0)
        out = self._inverse_phi(out, theta, order)
        if order:
            out *= numpy.where(
                out, numpy.prod(self._delta_phi(x_loc[:order], theta), 0), 0)
        else:
            out = numpy.clip(out, 0, 1)
        return out

    @staticmethod
    def _sigma(u_loc, theta, order):
        out = 1.
        for dim in range(order):
            out *= (-1/theta-dim)
        out = out*u_loc**(-1/theta-order)
        return out

    def _cache(self, dist, trans, cache):
        return self
