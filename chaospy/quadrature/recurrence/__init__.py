r"""
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
    >>> abscissas, weights = chaospy.generate_quadrature(
    ...     N, distribution, rule="gaussian")
    >>> weights *= adjust_scalar
    >>> abscissas
    array([[-0.86113631, -0.33998104,  0.33998104,  0.86113631]])
    >>> weights
    array([0.34785485, 0.65214515, 0.65214515, 0.34785485])

Here ``rule="gaussian"`` is the flag that indicate that Gaussian quadrature
should be used.

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

However, the list is not limited to these cases. Any and all valid weight
function are supported this way. However, not all weight functions does not
work very well. E.g. using the log-normal probability density function as
a weight function is known to scale badly. Which one works or not, depends on
context, so any non-standard use has to be done with some care.
"""
from .frontend import (
    construct_recurrence_coefficients, RECURRENCE_ALGORITHMS)
from .jacobi import coefficients_to_quadrature

from .chebyshev import modified_chebyshev
from .lanczos import lanczos
from .stieltjes import discretized_stieltjes, analytical_stieljes
