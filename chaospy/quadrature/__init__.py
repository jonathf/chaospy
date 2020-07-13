r"""
Quadrature methods, or numerical integration, is broad class of algorithm for
performing integration of any function :math:`g` that are defined without
requiring an analytical definition. In the scope of ``chaospy`` we limit this
scope to focus on methods that can be reduced to the following approximation:

.. math::
    \int p(x) g(x) dx \approx \sum_{n=1}^N W_n g(X_n)

Here :math:`p(x)` is an weight function, which is assumed to be an probability
distribution, and :math:`W_n` and :math:`X_n` are respectively quadrature
weights and abscissas used to define the approximation.

The simplest example applying such an approximation is Monte Carlo integration.
In such a method, you only need to select :math:`W_n=1/N` and :math:`X_n` to be
independent identical distributed samples drawn from the distribution of
:math:`p(x)`. In practice::

    >>> distribution = chaospy.Uniform(-1, 1)
    >>> abscissas = distribution.sample(1000)
    >>> weights = 1./len(abscissas)

However, except for very high dimensional problems, Monte Carlo is quite an
inefficient way to perform numerical integration, and there exist quite a few
methods that performs better in most low-dimensional settings. If however,
Monto Carlo is your best choice, it might be worth taking a look at
:ref:`sampling`.

.. note:
    Most quadrature rules optimized to a given weight function is referred to
    as the :ref:`gaussian` rules. It does the embedding of the weight function
    automatically as that is what it is designed for. For most other quadrature
    rules, including a weight function is typically not canonical. This however
    isn't very compatible with the Gaussian quadrature rules which take
    probability density functions into account as part of their implementation.
    It also does not match well with ``chaospy`` which assumes the density
    weight function to be defined and incorporated implicit.

    To address this issue, the weight functions are incorporated into the
    weight terms by substituting :math:`W^*_i \leftarrao W_i p(X_i)`, giving
    us:

    .. math::
        \int p(x) g(x) dx \approx
        \sum_i W_i p(X_i) g(X_i) = \sum_i W^{*}_i g(X_i)

    Which is the same format as the Gaussian quadrature rules.

    The consequence of this is that non-Gaussian quadrature rules only produces
    the canonical weights for the probability distribution
    ``chaospy.Uniform(0, 1)``, everything else is custom. To get around this
    limitation, there are few workarounds:

    * Use a uniform distribution on an arbitrary interval ``Uniform(a, b)``,
      and multiply the weight terms with the interval length: ``W *= (b-a)``
    * Use the quadrature rules directly from ``chaospy.quadrature.collection``.
    * Adjust weights afterwards: ``W /= dist.pdf(X)``

To create quadrature abscissas and weights, use the
:func:`~chaospy.quadrature.frontend.generate_quadrature` function. Which type
of quadrature to use is defined by the flag ``rule``. This argument can either
be the full name, or a single letter representing the rule. These are as
follows.

"""
from .frontend import generate_quadrature
from .sparse_grid import construct_sparse_grid
from .combine import combine

from .clenshaw_curtis import quad_clenshaw_curtis
from .discrete import quad_discrete
from .fejer import quad_fejer
from .gaussian import quad_gaussian
from .gauss_patterson import quad_gauss_patterson
from .gauss_legendre import quad_gauss_legendre
from .gauss_lobatto import quad_gauss_lobatto
from .gauss_kronrod import quad_gauss_kronrod
from .gauss_radau import quad_gauss_radau
from .genz_keister import quad_genz_keister
from .grid import quad_grid
from .leja import quad_leja
from .newton_cotes import quad_newton_cotes

from .recurrence import (
    analytical_stieljes,
    modified_chebyshev,
    coefficients_to_quadrature,
    construct_recurrence_coefficients,
    discretized_stieltjes,
    lanczos,
    RECURRENCE_ALGORITHMS,
)
