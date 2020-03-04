"""
Gaussian Quadratures Rules
--------------------------

:ref:`gaussian`
    The classical Gaussian quadrature scheme applied on any probability
    distribution.
:ref:`gauss_legendre`
    Same as :ref:`gaussian` for uniform distribution, but applicable to other
    distribution by incorporating the probability density as part of the
    function to be integrated.
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
import numpy

from .combine import combine

from .clenshaw_curtis import quad_clenshaw_curtis
from .fejer import quad_fejer
from .gaussian import quad_gaussian
from .gauss_patterson import quad_gauss_patterson
from .gauss_legendre import quad_gauss_legendre
from .gauss_lobatto import quad_gauss_lobatto
from .gauss_kronrod import quad_gauss_kronrod
from .gauss_radau import quad_gauss_radau
from .genz_keister import quad_genz_keister
from .leja import quad_leja
from .newton_cotes import quad_newton_cotes

QUAD_NAMES = {
    "c": "clenshaw_curtis", "clenshaw_curtis": "clenshaw_curtis",
    "f": "fejer", "fejer": "fejer",
    "g": "gaussian", "gaussian": "gaussian",
    "e": "gauss_legendre", "gauss_legendre": "gauss_legendre",
    "l": "gauss_lobatto", "gauss_lobatto": "gauss_lobatto",
    "k": "gauss_kronrod", "gauss_kronrod": "gauss_kronrod",
    "p": "gauss_patterson", "gauss_patterson": "gauss_patterson",
    "r": "gauss_radau", "gauss_radau": "gauss_radau",
    "z": "genz_keister", "genz_keister": "genz_keister",
    "j": "leja", "leja": "leja",
    "n": "newton_cotes", "newton_cotes": "newton_cotes",
}
QUAD_FUNCTIONS = {
    "clenshaw_curtis": quad_clenshaw_curtis,
    "fejer": quad_fejer,
    "gaussian": quad_gaussian,
    "gauss_kronrod": quad_gauss_kronrod,
    "gauss_legendre": quad_gauss_legendre,
    "gauss_lobatto": quad_gauss_lobatto,
    "gauss_patterson": quad_gauss_patterson,
    "gauss_radau": quad_gauss_radau,
    "genz_keister": quad_genz_keister,
    "leja": quad_leja,
    "newton_cotes": quad_newton_cotes,
}


def generate_quadrature(
        order,
        dist,
        rule="clenshaw_curtis",
        sparse=False,
        accuracy=100,
        growth=None,
        segments=1,
        recurrence_algorithm="",
):
    """
    Numerical quadrature node and weight generator.

    Args:
        order (int):
            The order of the quadrature.
        dist (chaospy.distributions.baseclass.Dist):
            The distribution which density will be used as weight function.
        rule (str):
            Rule for generating abscissas and weights. Either done with
            quadrature rules, or with random samples with constant weights.
        sparse (bool):
            If True used Smolyak's sparse grid instead of normal tensor product
            grid.
        accuracy (int):
            If gaussian is set, but the Dist provieded in domain does not
            provide an analytical TTR, ac sets the approximation order for the
            descitized Stieltje's method.
        growth (bool, None):
            If True sets the growth rule for the quadrature rule to only
            include orders that enhances nested samples. Defaults to the same
            value as ``sparse`` if omitted.
        segments (int):
            Split intervals into N subintervals and create a patched
            quadrature based on the segmented quadrature. Can not be lower than
            `order`. If 0 is provided, default to square root of `order`.
            Nested samples only exist when the number of segments are fixed.
        recurrence_algorithm (str):
            Name of the algorithm used to generate abscissas and weights in
            case of Gaussian quadrature scheme. If omitted, ``analytical`` will
            be tried first, and ``stieltjes`` used if that fails.

    Returns:
        (numpy.ndarray, numpy.ndarray):
            Abscissas and weights created from full tensor grid rule. Flatten
            such that ``abscissas.shape == (len(dist), len(weights))``.

    Examples:
        >>> distribution = chaospy.Iid(chaospy.Normal(0, 1), 3)
        >>> abscissas, weights = generate_quadrature(
        ...     1, distribution, rule=("gaussian", "fejer"))
        >>> abscissas
        array([[-1.  , -1.  ,  1.  ,  1.  ],
               [-3.75,  3.75, -3.75,  3.75]])
        >>> weights
        array([0.25, 0.25, 0.25, 0.25])
    """
    if sparse:
        from . import sparse_grid
        return sparse_grid.construct_sparse_grid(
            order, dist, rule=rule, accuracy=accuracy, growth=growth)

    if not isinstance(rule, str):
        order = numpy.ones(len(dist), dtype=int)*order
        abscissas, weights = zip(*[
            generate_quadrature(order_, dist_, rule_, growth)
            for order_, dist_, rule_ in zip(order, dist, rule)
        ])
        abscissas = combine([abscissa.T for abscissa in abscissas]).T
        weights = numpy.prod(combine([abscissa.T for abscissa in weights]), -1)
        return abscissas, weights

    rule = QUAD_NAMES[rule.lower()]
    kwargs = {}

    if rule in ("clenshaw_curtis", "fejer", "newton_cotes"):
        kwargs.update(growth=growth, segments=segments)

    if rule in ("gaussian", "gauss_kronrod", "gauss_radau", "gauss_lobatto"):
        kwargs.update(accuracy=accuracy,
                      recurrence_algorithm=recurrence_algorithm)

    quad_function = QUAD_FUNCTIONS[rule]
    abscissas, weights = quad_function(order, dist, **kwargs)

    assert len(weights) == abscissas.shape[1]
    assert len(abscissas.shape) == 2

    from ..distributions.operators.joint import J
    from ..distributions.evaluation import sorted_dependencies
    if dist.interpret_as_integer:
        abscissas = abscissas.astype(int)
    elif isinstance(dist, J):
        for dist_ in sorted_dependencies(dist):
            if dist_ in dist.inverse_map and dist_.interpret_as_integer:
                idx = dist.inverse_map[dist_]
                abscissas[idx:idx+len(dist_)] = numpy.around(
                    abscissas[idx:idx+len(dist_)])

    return abscissas, weights
