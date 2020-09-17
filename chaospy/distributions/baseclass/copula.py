r"""
A cumulative distribution function of an independent multivariate random
variable can be made dependent through a copula as follows:

.. math::
    F_{Q_0,\dots,Q_{D-1}} (q_0,\dots,q_{D-1}) =
    C(F_{Q_0}(q_0), \dots, F_{Q_{D-1}}(q_{D-1}))

where :math:`C` is the copula function, and :math:`F_{Q_i}` are marginal
distribution functions.  One of the more popular classes of copulas is the
Archimedean copulas.
.. \cite{sklar_random_1996}.
They are defined as follows:

.. math::
    C(u_1,\dots,u_n) =
    \phi^{[-1]} (\phi(u_1)+\dots+\phi(u_n)),

where :math:`\phi` is a generator and :math:`\phi^{[-1]}` is its
pseudo-inverse. Support for Archimedean copulas in `chaospy` is possible
through reformulation of the Rosenblatt transformation.  In two dimension, this
reformulation is as follows:

.. math::

    F_{U_0}(u_0) = \frac{C(u_0,1)}{C(1,1)}

    F_{U_1\mid U_0}(u_1\mid u_0) =
    \frac{\tfrac{\partial}{\partial u_0}
    C(u_0,u_1)}{\tfrac{\partial}{\partial u_0} C(u_0,1)}

This definition can also be generalized in to multiple variables using the
formula provided by Nelsen 1999.
.. cite:: nelsen_introduction_1999

The definition of the Rosenblatt transform can require multiple
differentiations.  An analytical formulation is usually not feasible, so the
expressions are estimated using difference scheme similar to the one outlined
for probability density function defined in :ref:`distributions`. The accurate
might therefore be affected.

Since copulas are meant as a replacement for Rosenblatt
transformation, it is usually assumed that the distribution it is
used on is stochastically independent.
However in the definition of a copula does not actually require it, and sine
the Rosenblatt transformation allows for it, multiple copulas can be stacked
together in `chaospy`.
"""
import numpy
import chaospy

from .distribution import Distribution


class Copula(Distribution):

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

        super(Copula, self).__init__(
            parameters=dict(dist=dist, trans=trans),
            dependencies=dependencies,
            rotation=rotation,
            repr_args=repr_args,
        )

    def _cdf(self, x, dist, trans, cache):
        output = dist._get_fwd(x, cache=cache)
        output = trans._get_fwd(output, cache=cache)
        return output

    def _lower(self, dist, trans, cache):
        return dist._get_lower(cache=cache)

    def _upper(self, dist, trans, cache):
        return dist._get_upper(cache=cache)

    def _ppf(self, qloc, dist, trans, cache):
        qloc = trans._get_inv(qloc, cache=cache)
        xloc = dist._get_inv(qloc, cache=cache)
        return xloc

    def _pdf(self, x, dist, trans, cache):
        density = dist._get_pdf(x, cache=cache.copy())
        return trans._get_pdf(dist._get_fwd(x, cache=cache), cache=cache)*density

    def _mom(self, x, dist, trans, cache):
        raise chaospy.UnsupportedFeature(
            "Joint distribution with dependencies not supported.")

    def _cache(self, dist, trans, cache):
        return self
