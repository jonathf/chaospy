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
import chaospy

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
    "d": "discrete", "discrete": "discrete",
    "i": "grid", "grid": "grid",
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
    "discrete": quad_discrete,
    "grid": quad_grid,
}


def generate_quadrature(
        order,
        dist,
        rule="",
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
        dist (chaospy.distributions.baseclass.Distribution):
            The distribution which density will be used as weight function.
        rule (str, Sequence[str]):
            Rule for generating abscissas and weights. If one name is provided,
            that rule is applied to all dimensions. If multiple names, each
            rule is positionally applied to each dimension. If omitted,
            ``clenshaw_curtis`` is applied to all continuous dimensions, and
            ``discrete`` to all discrete ones.
        sparse (bool):
            If True used Smolyak's sparse grid instead of normal tensor product
            grid.
        accuracy (int):
            If gaussian is set, but the dist provided in domain does not
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
        >>> distribution = chaospy.Iid(chaospy.Normal(0, 1), 2)
        >>> abscissas, weights = generate_quadrature(
        ...     1, distribution, rule=("gaussian", "fejer"))
        >>> abscissas.round(3)
        array([[-1.  , -1.  ,  1.  ,  1.  ],
               [-4.11,  4.11, -4.11,  4.11]])
        >>> weights.round(3)
        array([0.25, 0.25, 0.25, 0.25])

    """
    if not rule:
        if isinstance(dist, chaospy.J):
            rule = [
                ("discrete" if dist_.interpret_as_integer else "clenshaw_curtis")
                for dist_ in dist
            ]
        else:
            rule = ("discrete" if dist.interpret_as_integer
                    else "clenshaw_curtis")
    if sparse:
        from . import sparse_grid
        return sparse_grid.construct_sparse_grid(
            order, dist, rule=rule, accuracy=accuracy, growth=growth)

    if len(dist) == 1 or dist.stochastic_dependent:
        assert isinstance(rule, str), "dependencies require rule consistency"
        abscissas, weights = _generate_quadrature(
            order=order,
            dist=dist,
            rule=rule,
            growth=growth,
            segments=segments,
            accuracy=accuracy,
            recurrence_algorithm=recurrence_algorithm,
        )

    else:
        if isinstance(rule, str):
            rule = [rule]*len(dist)
        assert len(rule) == len(dist), (
            "rules and distribution length does not match.")
        assert all(isinstance(rule_, str) for rule_ in rule)

        order = numpy.ones(len(dist), dtype=int)*order
        abscissas, weights = zip(*[
            _generate_quadrature(
                order=order_,
                dist=dist_,
                rule=rule_,
                growth=growth,
                segments=segments,
                accuracy=accuracy,
                recurrence_algorithm=recurrence_algorithm,
            )
            for order_, dist_, rule_ in zip(order, dist, rule)
        ])
        abscissas = combine([abscissa.T for abscissa in abscissas]).T
        weights = numpy.prod(combine([weight.T for weight in weights]), -1)

    assert abscissas.shape == (len(dist), len(weights))
    if dist.interpret_as_integer:
        abscissas = numpy.round(abscissas).astype(int)
    return abscissas, weights


def _generate_quadrature(order, dist, rule, **kwargs):

    if isinstance(dist, chaospy.OperatorDistribution):

        args = ("left", "right")
        right_dist = isinstance(dist._parameters["right"], chaospy.Distribution)
        args = args if right_dist else args[::-1]
        assert not isinstance(dist._parameters[args[0]], chaospy.Distribution)
        const = numpy.asarray(dist._parameters[args[0]])

        if isinstance(dist, chaospy.Add):
            dist = dist._parameters[args[1]]
            abscissas, weights = _generate_quadrature(
                order=order, dist=dist, rule=rule, **kwargs)
            abscissas = (abscissas.T+const.T).T
            return abscissas, weights

        elif isinstance(dist, chaospy.Multiply):
            dist = dist._parameters[args[1]]
            abscissas, weights = _generate_quadrature(
                order=order, dist=dist, rule=rule, **kwargs)
            abscissas = (abscissas.T*const.T).T
            return abscissas, weights

    rule = QUAD_NAMES[rule.lower()]
    parameters = {}

    if rule in ("clenshaw_curtis", "fejer", "newton_cotes", "discrete"):
        parameters.update(growth=kwargs["growth"], segments=kwargs["segments"])

    if rule in ("gaussian", "gauss_kronrod", "gauss_radau", "gauss_lobatto"):
        parameters.update(accuracy=kwargs["accuracy"],
                          recurrence_algorithm=kwargs["recurrence_algorithm"])

    quad_function = QUAD_FUNCTIONS[rule]
    abscissas, weights = quad_function(order, dist, **parameters)
    return abscissas, weights
