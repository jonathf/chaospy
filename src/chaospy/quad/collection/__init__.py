"""
Collection of quadrature rules.

"Gaussian", "G"     Optimal Gaussian quadrature from Golub-Welsch
                    Slow for high order.
"Legendre", "E"     Gauss-Legendre quadrature
"Clenshaw", "C"     Clenshaw-Curtis quadrature. Exponential growth rule is
                    used when sparse is True to make the rule nested.
"Leja", J"          Leja quadrature. Linear growth rule is nested.
"Genz", "Z"         Hermite Genz-Keizter 16 rule. Nested. Valid to order 8.
"Patterson", "P"    Gauss-Patterson quadrature rule. Nested. Valid to order 8.
"""

from .interface import get_function

from .clenshaw_curtis import quad_clenshaw_curtis
from .gauss_patterson import quad_gauss_patterson
from .gauss_legendre import quad_gauss_legendre
from .genz_keister import quad_genz_keister
from .golub_welsch import quad_golub_welsch
from .leja import quad_leja

QUAD_FUNCTIONS = {
    "clenshaw_curtis": quad_clenshaw_curtis,
    "gauss_legendre": quad_gauss_legendre,
    "gauss_patterson": quad_gauss_patterson,
    "genz_keister": quad_genz_keister,
    "golub_welsch": quad_golub_welsch,
    "leja": quad_leja,
}

QUAD_SHORT_NAMES = {
    "c": "clenshaw_curtis",
    "e": "gauss_legendre",
    "p": "gauss_patterson",
    "z": "genz_keister",
    "g": "golub_welsch",
    "j": "leja",
}
