"""Numerical quadrature node and weight generator."""
import numpy
import chaospy

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
        rule=None,
        sparse=False,
        growth=None,
        segments=1,
        recurrence_algorithm="stieltjes",
        tolerance=1e-10,
        scaling=3,
        n_max=5000,
):
    """
    Numerical quadrature node and weight generator.

    Args:
        order (int):
            The order of the quadrature.
        dist (chaospy.distributions.baseclass.Distribution):
            The distribution which density will be used as weight function.
        rule (str, Sequence[str], None):
            Rule for generating abscissas and weights. If one name is provided,
            that rule is applied to all dimensions. If multiple names, each
            rule is positionally applied to each dimension. If omitted,
            ``clenshaw_curtis`` is applied to all continuous dimensions, and
            ``discrete`` to all discrete ones.
        sparse (bool):
            If True used Smolyak's sparse grid instead of normal tensor product
            grid.
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
        tolerance (float):
            The allowed relative error in norm between two quadrature orders
            before method assumes convergence.
        scaling (float):
            A multiplier the adaptive order increases with for each step
            quadrature order is not converged. Use 0 to indicate unit
            increments.
        n_max (int):
            The allowed number of quadrature points to use in approximation.

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
        rule = [
            ("discrete" if dist_.interpret_as_integer else "clenshaw_curtis")
            for dist_ in dist
        ]

    if sparse:
        return chaospy.sparse_grid.construct_sparse_grid(
            order=order,
            dist=dist,
            growth=growth,
            recurrence_algorithm=recurrence_algorithm,
            rule=rule,
            tolerance=tolerance,
            scaling=scaling,
            n_max=n_max,
        )

    if len(dist) == 1 or dist.stochastic_dependent:
        if not isinstance(rule, str) and len(rule) == 1:
            rule = rule[0]
        assert isinstance(rule, str), "dependencies require rule consistency"
        abscissas, weights = _generate_quadrature(
            order=order,
            dist=dist,
            rule=rule,
            growth=growth,
            segments=segments,
            recurrence_algorithm=recurrence_algorithm,
            tolerance=tolerance,
            scaling=scaling,
            n_max=n_max,
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
                recurrence_algorithm=recurrence_algorithm,
                tolerance=tolerance,
                scaling=scaling,
                n_max=n_max,
            )
            for order_, dist_, rule_ in zip(order, dist, rule)
        ])
        abscissas = chaospy.combine([abscissa.T for abscissa in abscissas]).T
        weights = numpy.prod(chaospy.combine([weight.T for weight in weights]), -1)

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
        parameters.update(tolerance=kwargs["tolerance"], scaling=kwargs["scaling"],
                          n_max=kwargs["n_max"], recurrence_algorithm=kwargs["recurrence_algorithm"])

    quad_function = QUAD_FUNCTIONS[rule]
    abscissas, weights = quad_function(order, dist, **parameters)
    return abscissas, weights
