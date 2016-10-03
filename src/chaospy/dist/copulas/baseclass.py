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
import numpy as np

from ..backend import Dist

class Copula(Dist):

    def __init__(self, dist, trans):
        """
        Args:
            dist (Dist) : Distribution to wrap the copula around.
            trans (Dist) : The copula wrapper `[0,1]^D \into [0,1]^D`.
        """
        Dist.__init__(self, dist=dist, trans=trans,
                _advance=True, _length=len(trans))

    def _cdf(self, x, G):
        dist, trans = G.D["dist"], G.D["trans"]
        q = G(G(x, dist), trans)
        return q

    def _bnd(self, x, G):
        return G(x, G.D["dist"])

    def _ppf(self, q, G):
        dist, trans = G.D["dist"], G.D["trans"]
        return G(G(q, trans), dist)

    def _pdf(self, x, G):
        dist, trans = G.D["dist"], G.D["trans"]
        return G(G.fwd_as_pdf(x, dist), trans)*G(x, dist)


class Archimedean(Dist):
    """
    Archimedean copula superclass.

    Subset this to generate an archimedean.
    """

    def _ppf(self, x, th, eps):

        for i in range(1, len(x)):

            q = x[:i+1].copy()
            lo, up = 0,1
            dq = np.zeros(i+1)
            dq[i] = eps
            flo, fup = -q[i],1-q[i]

            for iteration in range(1, 10):
                fq = self._diff(q[:i+1], th, eps)
                dfq = self._diff((q[:i+1].T+dq).T, th, eps)
                dfq = (dfq-fq)/eps
                dfq = np.where(dfq==0, np.inf, dfq)

                fq = fq-x[i]
                if not np.any(np.abs(fq)>eps):
                    break

                # reduce boundaries
                flo = np.where(fq<=0, fq, flo)
                lo = np.where(fq<=0, q[i], lo)

                fup = np.where(fq>=0, fq, fup)
                up = np.where(fq>=0, q[i], up)

                # Newton increment
                qdq = q[i]-fq/dfq

                # if new val on interior use Newton
                # else binary search
                q[i] = np.where((qdq<up)*(qdq>lo),
                        qdq, .5*(up+lo))

            x[i] = q[i]
        return x


    def _cdf(self, x, th, eps):
        out = np.zeros(x.shape)
        out[0] = x[0]
        for i in range(1,len(x)):
            out[i][x[i]==1] = 1
            out[i] = self._diff(x[:i+1], th, eps)

        return out

    def _pdf(self, x, th, eps):
        out = np.ones(x.shape)
        sign = 1-2*(x>.5)
        for i in range(1,len(x)):
            x[i] += eps*sign[i]
            out[i] = self._diff(x[:i+1], th, eps)
            x[i] -= eps*sign[i]
            out[i] -= self._diff(x[:i+1], th, eps)
            out[i] /= eps

        out = abs(out)
        return out

    def _diff(self, x, th, eps):
        """
        Differentiation function.

        Numerical approximation of a Rosenblatt transformation created from
        copula formulation.
        """
        foo = lambda y: self.igen(np.sum(self.gen(y, th), 0), th)

        out1 = out2 = 0.
        sign = 1 - 2*(x>.5).T
        for I in np.ndindex(*((2,)*(len(x)-1)+(1,))):

            eps_ = np.array(I)*eps
            x_ = (x.T + sign*eps_).T
            out1 += (-1)**sum(I)*foo(x_)

            x_[-1] = 1
            out2 += (-1)**sum(I)*foo(x_)

        out = out1/out2
        return out


    def _bnd(self, **prm):
        return 0,1
