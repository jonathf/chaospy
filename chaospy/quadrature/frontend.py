"""Numerical quadrature node and weight generator."""
import logging
import numpy
import chaospy

from .utils import combine
from .sparse_grid import sparse_grid

SHORT_NAME_TABLE = {
    "c": "clenshaw_curtis", "clenshaw_curtis": "clenshaw_curtis",
    "f1": "fejer_1", "fejer_1": "fejer_1",
    "f2": "fejer_2", "fejer_2": "fejer_2",
    "g": "gaussian", "gaussian": "gaussian",
    "e": "legendre", "legendre": "legendre",
    "l": "lobatto", "lobatto": "lobatto",
    "k": "kronrod", "kronrod": "kronrod",
    "p": "patterson", "patterson": "patterson",
    "r": "radau", "radau": "radau",
    "j": "leja", "leja": "leja",
    "n": "newton_cotes", "newton_cotes": "newton_cotes",
    "d": "discrete", "discrete": "discrete",
    "i": "grid", "grid": "grid",
    "z16": "genz_keister_16", "genz_keister_16": "genz_keister_16",
    "z18": "genz_keister_18", "genz_keister_18": "genz_keister_18",
    "z22": "genz_keister_22", "genz_keister_22": "genz_keister_22",
    "z24": "genz_keister_24", "genz_keister_24": "genz_keister_24",
}
DEPRECATED_SHORT_NAMES = {
    "f": "f2",
    "fejer": "fejer_2",
    "gauss_kronrod": "kronrod",
    "gauss_lobatto": "lobatto",
    "gauss_patterson": "patterson",
    "gauss_radau": "radau",
    "gauss_legendre": "legendre",
    "z": "genz_keister_24",
    "genz_keister": "genz_keister_24",
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
        ...     1, distribution, rule=("gaussian", "fejer_2"))
        >>> abscissas.round(3)
        array([[-1.  , -1.  ,  1.  ,  1.  ],
               [-4.11,  4.11, -4.11,  4.11]])
        >>> weights.round(3)
        array([0.222, 0.222, 0.222, 0.222])

    See also:
        :func:`chaospy.quadrature.clenshaw_curtis`
        :func:`chaospy.quadrature.fejer_1`
        :func:`chaospy.quadrature.fejer_2`
        :func:`chaospy.quadrature.gaussian`
        :func:`chaospy.quadrature.legendre_proxy`
        :func:`chaospy.quadrature.lobatto`
        :func:`chaospy.quadrature.kronrod`
        :func:`chaospy.quadrature.patterson`
        :func:`chaospy.quadrature.radau`
        :func:`chaospy.quadrature.leja`
        :func:`chaospy.quadrature.newton_cotes`
        :func:`chaospy.quadrature.discrete`
        :func:`chaospy.quadrature.grid`
        :func:`chaospy.quadrature.genz_keister_16`
        :func:`chaospy.quadrature.genz_keister_18`
        :func:`chaospy.quadrature.genz_keister_22`
        :func:`chaospy.quadrature.genz_keister_24`

    """
    if not rule:
        rule = [
            ("discrete" if dist_.interpret_as_integer else "clenshaw_curtis")
            for dist_ in dist
        ]

    if sparse:
        return sparse_grid(
            order=order,
            dist=dist,
            growth=growth,
            recurrence_algorithm=recurrence_algorithm,
            rule=rule,
            tolerance=tolerance,
            scaling=scaling,
            n_max=n_max,
        )

    if (not isinstance(dist, chaospy.Distribution) or
            (len(dist) == 1 or dist.stochastic_dependent)):
        if not isinstance(rule, str) and len(set(rule)) == 1:
            rule = rule[0]
        assert isinstance(rule, str), (
            "dependencies require rule consistency; %s provided" % rule)
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
        abscissas = combine([abscissa.T for abscissa in abscissas]).T
        weights = numpy.prod(combine([weight.T for weight in weights]), -1)

    if isinstance(dist, chaospy.Distribution):
        assert abscissas.shape == (len(dist), len(weights))
        if dist.interpret_as_integer:
            abscissas = numpy.round(abscissas).astype(int)
    else:
        assert abscissas.shape[-1] == len(weights)
    return abscissas, weights


def _generate_quadrature(order, dist, rule, **kwargs):

    logger = logging.getLogger(__name__)
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

    rule = rule.lower()
    if rule in DEPRECATED_SHORT_NAMES:
        logger.warning("quadrature rule '%s' is renamed to '%s'; "
                       "error will be raised in the future",
                       rule, DEPRECATED_SHORT_NAMES[rule])
        rule = DEPRECATED_SHORT_NAMES[rule]
    rule = SHORT_NAME_TABLE[rule]

    parameters = {}

    if rule in ("clenshaw_curtis", "fejer_1", "fejer_2", "newton_cotes", "discrete", "grid"):
        parameters["growth"] = kwargs["growth"]

    if rule in ("clenshaw_curtis", "fejer_1", "fejer_2", "newton_cotes", "grid", "legendre"):
        parameters["segments"] = kwargs["segments"]

    if rule in ("gaussian", "kronrod", "radau", "lobatto"):
        parameters.update(
            n_max=kwargs["n_max"],
            tolerance=kwargs["tolerance"],
            scaling=kwargs["scaling"],
            recurrence_algorithm=kwargs["recurrence_algorithm"],
        )

    quad_function = chaospy.quadrature.INTEGRATION_COLLECTION[rule]
    abscissas, weights = quad_function(order, dist, **parameters)
    return abscissas, weights
