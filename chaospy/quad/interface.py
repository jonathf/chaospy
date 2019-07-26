"""Frontend collection."""
import numpy


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


def construct_quadrature(
        order,
        dist,
        rule,
        growth=False,
        accuracy=100,
        recurrence_algorithm="",
):
    """
    Create a quadrature function and set default parameter values.

    Args:
        rule (str):
            Name of quadrature rule defined in ``QUAD_FUNCTIONS``.
        domain (Dist, numpy.ndarray):
            Defines ``lower`` and ``upper`` that is passed quadrature rule. If
            ``Dist``, ``domain`` is renamed to ``dist`` and also
            passed.
        parameters (:py:data:typing.Any):
            Redefining of the parameter defaults. Only add parameters that the
            quadrature rule expect.
    """
    if not isinstance(rule, str):
        order = np.ones(len(dist), dtype=int)*order
        abscissas, weights = zip(*[
            construct_quadrature(order_, dist_, rule_, growth)
            for order_, dist_, rule_ in zip(order, dist, rule)
        ])
        abscissas = combine(abscissas).T.reshape(len(dist), -1)
        weights = numpy.prod(combine(weights), -1)
        return abscissas, weights

    rule = QUAD_NAMES[rule.lower()]
    kwargs = {}

    if rule in ("clenshaw_curtis", "fejer", "newton_cotes"):
        kwargs.update(growth=growth)

    if rule in ("gaussian", "gauss_kronrod", "gauss_radau", "gauss_lobatto"):
        kwargs.update(accuracy=accuracy,
                      recurrence_algorithm=recurrence_algorithm)

    from . import collection
    quad_function = getattr(collection, "quad_" + rule)
    abscissas, weights = quad_function(order, dist, **kwargs)

    assert len(weights) == abscissas.shape[1]
    assert len(abscissas.shape) == 2
    return abscissas, weights
