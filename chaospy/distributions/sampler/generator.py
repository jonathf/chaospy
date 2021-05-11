"""Sample generator."""
import logging
import numpy
from . import sequences, latin_hypercube

SAMPLER_NAMES = {
    "a": "additive_recursion", "additive_recursion": "additive_recursion",
    "c": "chebyshev", "chebyshev": "chebyshev",
    "nc": "nested_chebyshev", "nested_chebyshev": "nested_chebyshev",
    "k": "korobov", "korobov": "korobov",
    "g": "grid", "grid": "grid",
    "ng": "nested_grid", "nested_grid": "nested_grid",
    "s": "sobol", "sobol": "sobol",
    "h": "halton", "halton": "halton",
    "m": "hammersley", "hammersley": "hammersley",
    "l": "latin_hypercube", "latin_hypercube": "latin_hypercube",
    "r": "random", "random": "random",
}
SAMPLER_FUNCTIONS = {
    "additive_recursion": sequences.create_additive_recursion_samples,
    "chebyshev": sequences.create_chebyshev_samples,
    "nested_chebyshev": sequences.create_nested_chebyshev_samples,
    "korobov": sequences.create_korobov_samples,
    "grid": sequences.create_grid_samples,
    "nested_grid": sequences.create_nested_grid_samples,
    "sobol": sequences.create_sobol_samples,
    "halton": sequences.create_halton_samples,
    "hammersley": sequences.create_hammersley_samples,
    "latin_hypercube": latin_hypercube.create_latin_hypercube_samples,
    "random": lambda order, dim: numpy.random.random((dim, order)),
}


def generate_samples(order, domain=1, rule="random", antithetic=None):
    """
    Sample generator.

    Args:
        order (int):
            Sample order. Determines the number of samples to create.
        domain (Distribution, int, numpy.ndarray):
            Defines the space where the samples are generated. If integer is
            provided, the space ``[0, 1]^domain`` will be used. If array-like
            object is provided, a hypercube it defines will be used. If
            distribution, the domain it spans will be used.
        rule (str):
            rule for generating samples.
        antithetic (tuple):
            Sequence of boolean values. Represents the axes to mirror using
            antithetic variable.
    """
    logger = logging.getLogger(__name__)

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
        order = int(numpy.log(order-dim))
        order = order if order > 1 else 1
        while (order-1)*2**dim < order_saved:
            order += 1
        trans_ = trans
        trans = lambda x_data: trans_(
            create_antithetic_variates(x_data, antithetic)[:, :order_saved])

    rule = SAMPLER_NAMES[rule.lower()]
    logger.debug("generating random samples using %s rule", rule)
    sampler = SAMPLER_FUNCTIONS[rule]
    x_data = trans(sampler(order=order, dim=dim))

    logger.debug("order: %d, dim: %d -> shape: %s", order, dim, x_data.shape)
    return x_data
