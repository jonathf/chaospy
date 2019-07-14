r"""
Quadrature methods, or numerical integration, is broad class of algorithm for
performing integration of any function :math:`g` that are defined without
requiring an analytical definition. In the scope of ``chaospy`` we limit this
scope to focus on methods that can be reduced to the following approximation:

.. math::
    \int p(x) g(x) dx = \sum_{n=1}^N W_n g(X_n)

Here :math:`p(x)` is an weight function, which for our use would be an
probability distribution, and :math:`W_n` and :math:`X_n` are respectively
quadrature weights and abscissas used to define the approximation.

This simplest example of such an approximation is Monte Carlo integration. In
such a method, you only need to select :math:`W_n=1/N` and :math:`X_n` to be
independent identical distributed samples drawn from the distribution of
:math:`p(x)`.

However, except for very high dimensional problems, Monte Carlo is quite an
inefficient way to perform numerical integration, and there exist quite a few
methods that performs better in most low-dimensional settings.

Gaussian Quadrature
-------------------

One of the more popular integration schemes when dealing with orthogonal
polynomials are known as Gaussian quadrature. These are specially tailored
integration schemes each for different weighting schemes. Traditionally the
weights are given a form that does not adhere to the probability density
function rule of being normalized to 1, however the different is only scaling.

For example, consider the Gauss-Legendre which is optimized to perform the
integration:

.. math::
    \int_{-1}^1 g(x) dx \approx \sum_i W_i g(X_i)

The corresponding probability distribution that matches this contant weight
function on the :math:`(-1, 1)` interval, is ``chaospy.Uniform(-1, 1)``.
However, this distribution has a density of 0.5, instead of 1 in our example
here.

.. math::
    \int_{-1}^1 0.5 g(x) dx \approx \sum_i W_i g(X_i)

So to use ``chaospy`` to create a true Gaussian quadrature rule, one often has
to multiply the weights :math:`W_i` with some adjustment scalar. For example::

    >>> distribution = chaospy.Uniform(-1, 1)
    >>> N = 3
    >>> adjust_scalar = 2
    >>> X, W = chaospy.generate_quadrature(N, distribution, rule="G")
    >>> W *= adjust_scalar
    >>> print(X)
    [[-0.86113631 -0.33998104  0.33998104  0.86113631]]
    >>> print(W)
    [0.34785485 0.65214515 0.65214515 0.34785485]

Here ``rule="G"`` refers to using :ref:`gaussian_quadrature`.

The various constants and distributions to achieve the various quadrature rules
are as follows.

==================== ======================= ========================= ===================
Scheme               Weight function         Distribution              Adjustment
==================== ======================= ========================= ===================
Hermite              :math:`e^{-x^2}`        ``Normal(0, 2**-0.5)``    :math:`\sqrt{\pi}`
Legendre             :math:`1`               ``Uniform(-1, 1)``        :math:`2`
Jakobi               :math:`(1-x)^a(1+x)^b`  ``Beta(a+1, b+1, -1, 1)`` :math:`2^{a+b}`
1. order Chebyshev   :math:`1/\sqrt{1-x^2}`  ``Beta(0.5, 0.5, -1, 1)`` :math:`1/2`
2. order Chebyshev   :math:`\sqrt{1-x^2}`    ``Beta(1.5, 1.5, -1, 1)`` :math:`2`
Laguerre             :math:`e^{-x}`          ``Exponential()``         :math:`1`
Generalized Laguerre :math:`x^a e^{-x}`      ``Gamma(a+1)``            :math:`\Gamma(a+1)`
Gegenbaur            :math:`(1-x^2)^{a-0.5}` ``Beta(a+.5,a+.5,-1,1)``  :math:`2^{2a-1}`
==================== ======================= ========================= ===================

However, the list is not limited to these cases. Any and all probability
function, and with scaling, and weight functions are supported in this manor.
However, not all weight functions does not work very well. E.g. using the
log-normal probability density function as a weight function is known to scale
horribly bad. Which one works or not, depends on context, so any non-standard
use has to be done with some care.

Non-Gaussian Quadrature
-----------------------

Quadrature outside the Gaussian quadrature typically does not include an weight
function implying that they are on the general form:

.. math::
    \int g(x) dx \approx \sum_i W_i g(X_i)

This however isn't very compatible with the Gaussian quadrature rules which
take probability density functions into account as part of their
implementation. It also does not match well with ``chaospy`` which assumes the
density weight function to be defined and incorporated implicit.

To address this issue, the weight functions are incorporated into the weight
terms by defining :math:`W^*_i=W_i p(X_i)`, giving us:

.. math::
    \int p(x) g(x) dx \approx \sum_i W_i p(X_i) g(X_i) = \sum_i W^{*}_i g(X_i)

Which is the same format as the Gaussian quadrature rules.

The consequence of this is that non-Gaussian quadrature rules only produces the
canonical weights for the probability distribution ``Uniform(0, 1)``,
everything else is custom. To get around this limitation, there are few
workarounds:

* Use a uniform distribution on an arbitrary interval ``Uniform(a, b)``, and
  multiply the weight terms with the interval length: ``W *= (b-a)``
* Use the quadrature rules directly from ``chaospy.quad.collection``.
* Adjust weights afterwards: ``W /= dist.pdf(X)``
"""
from .combine import combine
from .interface import generate_quadrature

from .stieltjes import generate_stieltjes
from .collection import *
from .sparse_grid import sparse_grid
from .generator import rule_generator
