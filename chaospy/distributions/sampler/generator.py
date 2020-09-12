"""
Each sampling scheme can be accessed through the ``sample`` method on each
distribution. But in addition, they can also be created on the unit hyper-cube
using direct sampling functions. The frontend for all these functions is the
:func:`~chaospy.distributions.sampler.generator.generate_samples` function. It
allows for the same functionality as the ``sample`` method, but also support
some extra functionality by not being associated with a specific distribution.
For example::

    >>> samples = generate_samples(order=4)
    >>> samples.round(4)
    array([[0.6536, 0.115 , 0.9503, 0.4822]])

Custom domain::

    >>> samples = generate_samples(order=4, domain=[-1, 1])
    >>> samples.round(4)
    array([[ 0.7449, -0.5753, -0.9186, -0.2056]])
    >>> samples = generate_samples(order=4, domain=chaospy.Normal(0, 1))
    >>> samples.round(4)
    array([[-0.7286,  1.0016, -0.8166,  0.651 ]])

Use a custom sampling scheme::

    >>> generate_samples(order=4, rule="halton").round(4)
    array([[0.75 , 0.125, 0.625, 0.375]])

Multivariate case::

    >>> samples = generate_samples(order=4, domain=[[-1, 0], [0, 1]])
    >>> samples.round(4)
    array([[-0.6078, -0.8177, -0.2565, -0.9304],
           [ 0.8853,  0.9526,  0.9311,  0.4154]])
    >>> distribution = chaospy.J(chaospy.Normal(0, 1), chaospy.Uniform(0, 1))
    >>> samples = generate_samples(order=4, domain=distribution)
    >>> samples.round(4)
    array([[-1.896 ,  2.0975, -0.4135,  0.5437],
           [ 0.3619,  0.0351,  0.8551,  0.6573]])

Antithetic variates::

    >>> samples = generate_samples(order=8, rule="halton", antithetic=True)
    >>> samples.round(4)
    array([[0.75 , 0.25 , 0.125, 0.875, 0.625, 0.375, 0.375, 0.625]])

Multivariate antithetic variates::

    >>> samples = generate_samples(
    ...     order=8, domain=2, rule="hammersley", antithetic=True)
    >>> samples.round(4)
    array([[0.75 , 0.25 , 0.75 , 0.25 , 0.125, 0.875, 0.125, 0.875],
           [0.25 , 0.25 , 0.75 , 0.75 , 0.5  , 0.5  , 0.5  , 0.5  ]])

Here as with the ``sample`` method, the flag ``rule`` is used to determine
sampling scheme. The default ``rule="random"`` uses classical psuedo-random
samples created using ``numpy.random``.
"""
import logging
import numpy
from . import sequences, latin_hypercube

SAMPLER_NAMES = {
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
            rule for generating samples. The various rules are listed in
            :mod:`chaospy.distributions.sampler.generator`.
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
        order = int(numpy.log(order - dim))
        order = order if order > 1 else 1
        while order**dim < order_saved:
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
