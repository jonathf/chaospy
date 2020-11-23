.. _quadrature:

Quadrature Integration
======================

Introduction
------------

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

.. note::
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
:func:`chaospy.generate_quadrature` function. Which type of quadrature to use
is defined by the flag ``rule``. This argument can either be the full name, or
a single letter representing the rule. These are as follows.

.. _sparsegrid:

Smolyak Sparse-Grid
-------------------

As the number of dimensions increases linear, the number of samples increases
exponentially. This is known as the curse of dimensionality. Except for
switching to Monte Carlo integration, the is no way to completely guard against
this problem. However, there are some possibility to mitigate the problem
personally. One such strategy is to employ Smolyak sparse-grid quadrature. This
method uses a quadrature rule over a combination of different orders to tailor
a scheme that uses fewer abscissas points than a full tensor-product approach.

To use Smolyak sparse-grid in ``chaospy``, just pass the flag ``sparse=True``
to the :func:`chaospy.generate_quadrature` function. For example::

    >>> distribution = chaospy.J(
    ...     chaospy.Uniform(0, 4), chaospy.Uniform(0, 4))
    >>> abscissas, weights = chaospy.generate_quadrature(
    ...     3, distribution, sparse=True)
    >>> abscissas.round(4)
    array([[0., 0., 0., 1., 2., 2., 2., 2., 2., 3., 4., 4., 4.],
           [0., 2., 4., 2., 0., 1., 2., 3., 4., 2., 0., 2., 4.]])
    >>> weights.round(4)
    array([-0.0833,  0.2222, -0.0833,  0.4444,  0.2222,  0.4444, -1.3333,
            0.4444,  0.2222,  0.4444, -0.0833,  0.2222, -0.0833])

This compared to the full tensor-product grid::

    >>> abscissas, weights = chaospy.generate_quadrature(3, distribution, sparse=False)
    >>> abscissas.round(4)
    array([[0., 0., 0., 0., 1., 1., 1., 1., 3., 3., 3., 3., 4., 4., 4., 4.],
           [0., 1., 3., 4., 0., 1., 3., 4., 0., 1., 3., 4., 0., 1., 3., 4.]])
    >>> weights.round(4)
    array([0.0031, 0.0247, 0.0247, 0.0031, 0.0247, 0.1975, 0.1975, 0.0247,
           0.0247, 0.1975, 0.1975, 0.0247, 0.0031, 0.0247, 0.0247, 0.0031])

The method works with all quadrature rules, but is known to be quite
inefficient when applied to rules that can not be nested. For example using
Gauss-Legendre samples::

    >>> abscissas, weights = chaospy.generate_quadrature(
    ...     6, distribution, rule="gauss_legendre", sparse=True)
    >>> len(weights)
    140
    >>> abscissas, weights = chaospy.generate_quadrature(
    ...     6, distribution, rule="gauss_legendre", sparse=False)
    >>> len(weights)
    49

.. note::
    Some quadrature rules are only partially nested at certain orders. These
    include e.g. :func:`chaospy.quad_clenshaw_curtis`,
    :func:`chaospy.quad_fejer` and :func:`chaospy.quad_newton_cotes`. To
    exploit this nested-nes, the default behavior is to only include orders
    that are properly nested. This implies that flipping the flag ``sparse``
    will result in a somewhat different scheme. To fix the scheme one way or
    the other, explicitly include the flag ``growth=False`` or ``growth=True``
    respectively.

.. _gaussian:

Gaussian Quadrature
-------------------

Most integration problems when dealing with polynomial chaos expansion comes
with a weight function :math:`p(x)` which happens to be the probability density
function. Gaussian quadrature creates weights and abscissas that are tailored
to be optimal with the inclusion of a weight function. It is therefore not one
method, but a collection of methods, each tailored to different probability
density functions.

In ``chaospy`` Gaussian quadrature is a functionality attached to each
probability distribution. This means that instead of explicitly supporting
a list of quadrature rules, all rules are supported through the capability of
the distribution implementation. For common distribution, this means that the
quadrature rules are calculated analytically using Stieltjes method on known
three terms recursion coefficients, and using those to create quadrature node
using the e.g. discretized Stieltjes algorithm.

For example for the tailored quadrature rules defined above:

* Gauss-Hermit quadrature is tailored to the normal (Gaussian) distribution::

    >>> distribution = chaospy.Normal(0, 1)
    >>> abscissas, weights = chaospy.generate_quadrature(
    ...     5, distribution, rule="gaussian")
    >>> abscissas.round(4)
    array([[-3.3243, -1.8892, -0.6167,  0.6167,  1.8892,  3.3243]])
    >>> weights.round(4)
    array([0.0026, 0.0886, 0.4088, 0.4088, 0.0886, 0.0026])

* Gauss-Legendre quadrature is tailored to the Uniform distributions::

    >>> distribution = chaospy.Uniform(-1, 1)
    >>> abscissas, weights = chaospy.generate_quadrature(
    ...     5, distribution, rule="gaussian")
    >>> abscissas.round(4)
    array([[-0.9325, -0.6612, -0.2386,  0.2386,  0.6612,  0.9325]])
    >>> weights.round(4)
    array([0.0857, 0.1804, 0.234 , 0.234 , 0.1804, 0.0857])

* Gauss-Jacobi quadrature is tailored to the Beta distribution::

    >>> distribution = chaospy.Beta(2, 4, lower=-1, upper=1)
    >>> abscissas, weights = chaospy.generate_quadrature(
    ...     5, distribution, rule="gaussian")
    >>> abscissas.round(4)
    array([[-0.8969, -0.6679, -0.3448,  0.0289,  0.4029,  0.7279]])
    >>> weights.round(4)
    array([0.0749, 0.272 , 0.355 , 0.2253, 0.0667, 0.0062])

* Gauss-Laguerre quadrature is tailored to the Exponential distribution::

    >>> distribution = chaospy.Exponential()
    >>> abscissas, weights = chaospy.generate_quadrature(
    ...     5, distribution, rule="gaussian")
    >>> abscissas.round(4)
    array([[ 0.2228,  1.1889,  2.9927,  5.7751,  9.8375, 15.9829]])
    >>> weights.round(4)
    array([4.590e-01, 4.170e-01, 1.134e-01, 1.040e-02, 3.000e-04, 0.000e+00])

* Generalized Gauss-Laguerre quadrature is tailored to the Gamma distribution::

    >>> distribution = chaospy.Gamma(2, 4)
    >>> abscissas, weights = chaospy.generate_quadrature(
    ...     5, distribution, rule="gaussian")
    >>> abscissas.round(4)
    array([[ 2.1107,  7.1852, 15.5066, 27.6753, 44.9384, 70.5839]])
    >>> weights.round(4)
    array([0.2777, 0.4939, 0.203 , 0.0247, 0.0008, 0.    ])

For uncommon distributions an analytical Stieltjes method can not be performed
as the distribution does not provide three terms recursion coefficients. In
this scenario, the discretized counterpart is used instead as an approximation.
For example, to mention a few:

* The Triangle distribution::

    >>> distribution = chaospy.Triangle(-1, 0, 1)
    >>> abscissas, weights = chaospy.generate_quadrature(
    ...     5, distribution, rule="gaussian")
    >>> abscissas.round(4)
    array([[-0.8657, -0.5766, -0.1943,  0.1943,  0.5766,  0.8657]])
    >>> weights.round(4)
    array([0.0295, 0.1475, 0.323 , 0.323 , 0.1475, 0.0295])

* The Laplace distribution::

    >>> distribution = chaospy.Laplace(0, 1)
    >>> abscissas, weights = chaospy.generate_quadrature(
    ...     5, distribution, rule="gaussian")
    >>> abscissas.round(4)
    array([[-10.4917,  -4.6469,  -1.0404,   1.0404,   4.6469,  10.4917]])
    >>> weights.round(4)
    array([1.000e-04, 2.180e-02, 4.781e-01, 4.781e-01, 2.180e-02, 1.000e-04])

* The Weibull distribution::

    >>> distribution = chaospy.Weibull()
    >>> abscissas, weights = chaospy.generate_quadrature(
    ...     5, distribution, rule="gaussian")
    >>> abscissas.round(4)
    array([[ 0.2228,  1.1886,  2.9918,  5.7731,  9.8334, 15.9737]])
    >>> weights.round(4)
    array([4.589e-01, 4.170e-01, 1.134e-01, 1.040e-02, 3.000e-04, 0.000e+00])

* The Rayleigh distribution::

    >>> distribution = chaospy.Rayleigh()
    >>> abscissas, weights = chaospy.generate_quadrature(
    ...     5, distribution, rule="gaussian")
    >>> abscissas.round(4)
    array([[0.2474, 0.7688, 1.4797, 2.3318, 3.3233, 4.5304]])
    >>> weights.round(4)
    array([9.600e-02, 3.592e-01, 3.891e-01, 1.412e-01, 1.430e-02, 2.000e-04])

Statistician vs physicists
--------------------------

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
However, this distribution has a density of 0.5, instead of 1 as in the
example.

.. math::
    \int_{-1}^1 0.5 g(x) dx \approx \sum_i W_i g(X_i)

So to use ``chaospy`` to create a "true" Gaussian quadrature rule, one often has
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
