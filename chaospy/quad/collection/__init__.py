# pylint: disable=wildcard-import
r"""
To create quadrature abscissas and weights, use the
:func:`~chaospy.quad.generate_quadrature` function. Which type of quadrature to
use is defined by the flag ``rule``. This argument can either be the full name,
or a single letter representing the rule. These are as follows.

``Gaussian``, ``G``
    Optimal Gaussian quadrature from Golub-Welsch. A slow method for higher
    order.
``Legendre``, ``E``
    Gauss-Legendre quadrature
``Clenshaw``, ``C``
    Clenshaw-Curtis quadrature. Exponential growth rule is used when sparse is
    True to make the rule nested.
``Leja``, J``
    Leja quadrature. Linear growth rule is nested.
``Genz``, ``Z``
    Hermite Genz-Keizter 16 rule. Nested. Valid to order 8.
``Patterson``, ``P``
    Gauss-Patterson quadrature rule. Nested. Valid to order 8.
``Fejer``, ``F``
    Fejer quadrature. Same as Clenshaw-Curtis, but without the endpoints.
"""
from .frontend import *
