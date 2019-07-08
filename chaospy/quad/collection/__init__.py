# pylint: disable=wildcard-import
r"""
To create quadrature abscissas and weights, use the
:func:`~chaospy.quad.interface.generate_quadrature` function. Which type of
quadrature to use is defined by the flag ``rule``. This argument can either be
the full name, or a single letter representing the rule. These are as follows.

Gaussian Quadratures Rules
--------------------------

:ref:`gaussian_quadrature`
    Optimal Gaussian quadrature using the Golub-Welsch algorithm.
:ref:`gauss_legendre`
    Same as :ref:`gaussian_quadrature` for uniform distribution, but applicable
    to other distribution by incorporating the probability density as part of
    the function to be integrated.
:ref:`gauss_patterson`
    Extension of Gauss-Legendre rule. Valid to order 8.
:ref:`gauss_kronrod`
    Extension to the Gauss-Patterson rule to include most distribution and any
    order.
:ref:`gauss_lobatto`
    Gaussian quadrature rule that enforces the endpoints to be included in the
    rule.
:ref:`gauss_radau`
    Gaussian quadrature rule that enforces that a single fixed point to be
    included in the rule.

Non-Gaussian Quadrature Rules
-----------------------------

:ref:`clenshaw_curtis`
    Chebyshev nodes with endpoints included.
:ref:`fejer`
    Chebyshev nodes without endpoints included.
:ref:`leja`
    Fully nested quadrature method.
:ref:`genz_keister`
    Genz-Keizter 16 rule. Nested. Valid to order 8.
:ref:`newton_cotes`
    Numerical integration rule based on fixed width abscissas.
"""
from .frontend import *
from .gauss_kronrod import kronrod_jacobi
