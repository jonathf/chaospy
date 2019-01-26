"""
Front end for generating psuedo-random samples.

Example usage
-------------

Generating simple samples::

    >>> print(numpy.around(generate_samples(order=4), 4))
    [[0.6536 0.115  0.9503 0.4822]]

Custom domain::

    >>> print(numpy.around(generate_samples(order=4, domain=[-1, 1]), 4))
    [[ 0.7449 -0.5753 -0.9186 -0.2056]]
    >>> print(numpy.around(generate_samples(order=4, domain=chaospy.Normal(0, 1)), 4))
    [[-0.7286  1.0016 -0.8166  0.651 ]]

Use a custom sampling scheme::

    >>> print(numpy.around(generate_samples(order=4, rule="H"), 4))
    [[0.75  0.125 0.625 0.375]]

Multivariate case::

    >>> print(numpy.around(generate_samples(order=4, domain=[[-1, 0], [0, 1]]), 4))
    [[-0.6078 -0.8177 -0.2565 -0.9304]
     [ 0.8853  0.9526  0.9311  0.4154]]
    >>> print(numpy.around(generate_samples(
    ...     order=4, domain=chaospy.J(chaospy.Normal(), chaospy.Uniform())), 4))
    [[-1.896   2.0975 -0.4135  0.5437]
     [ 0.3619  0.0351  0.8551  0.6573]]

Antithetic variates::

    >>> print(numpy.around(generate_samples(order=8, rule="H", antithetic=True), 4))
    [[0.75  0.25  0.125 0.875 0.625 0.375 0.375 0.625]]

Multivariate antithetic variates::

    >>> print(numpy.around(generate_samples(order=8, domain=2, rule="M", antithetic=True), 4))
    [[0.75  0.25  0.75  0.25  0.125 0.875 0.125 0.875]
     [0.25  0.25  0.75  0.75  0.5   0.5   0.5   0.5  ]]
"""
import logging
import numpy
from . import sequences, latin_hypercube

SAMPLERS = {
    "C": sequences.create_chebyshev_samples,
    "NC": sequences.create_nested_chebyshev_samples,
    "K": sequences.create_korobov_samples,
    "RG": sequences.create_grid_samples,
    "NG": sequences.create_nested_grid_samples,
    "S": sequences.create_sobol_samples,
    "H": sequences.create_halton_samples,
    "M": sequences.create_hammersley_samples,
    "L": latin_hypercube.create_latin_hypercube_samples,
    "R": lambda order, dim: numpy.random.random((dim, order)),
}


def generate_samples(order, domain=1, rule="R", antithetic=None):
    """
    Sample generator.

    Args:
        order (int):
            Sample order. Determines the number of samples to create.
        domain (Dist, int, array_like):
            Defines the space where the samples are generated. If integer is
            provided, the space ``[0, 1]^domain`` will be used. If array-like
            object is provided, a hypercube it defines will be used. If
            distribution, the domain it spans will be used.
        rule (str):
            rule for generating samples. The various rules are listed in
            :mod:`chaospy.distributions.sampler.generator`.
        antithetic (array_like, optional):
            List of bool. Represents the axes to mirror using antithetic
            variable.
    """
    logger = logging.getLogger(__name__)
    logger.debug("generating random samples using rule %s", rule)

    rule = rule.upper()

    if isinstance(domain, int):
        dim = domain
        trans = lambda x_data: x_data

    elif isinstance(domain, (tuple, list, numpy.ndarray)):
        domain = numpy.asfarray(domain)
        if len(domain.shape) < 2:
            dim = 1
        else:
            dim = len(domain[0])
        trans = lambda x_data: ((domain[1]-domain[0])*x_data.T + domain[0]).T

    else:
        dist = domain
        dim = len(dist)
        trans = dist.inv

    if antithetic is not None:

        from .antithetic import create_antithetic_variates
        antithetic = numpy.array(antithetic, dtype=bool).flatten()
        if antithetic.size == 1 and dim > 1:
            antithetic = numpy.repeat(antithetic, dim)

        size = numpy.sum(1*numpy.array(antithetic))
        order_saved = order
        order = int(numpy.log(order - dim))
        order = order if order > 1 else 1
        while order**dim < order_saved:
            order += 1
        trans_ = trans
        trans = lambda x_data: trans_(
            create_antithetic_variates(x_data, antithetic)[:, :order_saved])

    assert rule in SAMPLERS, "rule not recognised"
    sampler = SAMPLERS[rule]
    x_data = trans(sampler(order=order, dim=dim))

    logger.debug("order: %d, dim: %d -> shape: %s", order, dim, x_data.shape)
    return x_data
