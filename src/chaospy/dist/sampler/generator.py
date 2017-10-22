"""
Front end for generating psuedo-random samples.

The samples are as follows:

``C``
    Chebyshev nodes
``NC``
    Nested Chebyshev
``K``
    Korobov
``R``
    (Pseudo-)Random
``RG``
    Regular grid
``NG``
    Nested grid
``L``
    Latin hypercube
``S``
    Sobol
``H``
    Halton
``M``
    Hammersley

Example usage
-------------

Generating simple samples::

    >>> print(generate_samples(order=4))
    [[ 0.65358959  0.11500694  0.95028286  0.4821914 ]]

Custom domain::

    >>> print(generate_samples(order=4, domain=[-1, 1]))
    [[ 0.74494907 -0.57533464 -0.91858075 -0.20561108]]
    >>> print(generate_samples(order=4, domain=chaospy.Normal(0, 1)))
    [[-0.72857056  1.00163781 -0.81658665  0.65097762]]

Use a custom sampling scheme::

    >>> print(generate_samples(order=4, rule="H"))
    [[ 0.75   0.125  0.625  0.375  0.875]]

Multivariate case::

    >>> print(generate_samples(order=4, domain=[[-1, 0], [0, 1]]))
    [[-0.60784587 -0.81774348 -0.25646059 -0.93041792]
     [ 0.8853372   0.9526444   0.93114343  0.41543095]]
    >>> print(generate_samples(
    ...     order=4, domain=chaospy.J(chaospy.Normal(), chaospy.Uniform())))
    [[-1.89597524  2.0975487  -0.41345216  0.54373243]
     [ 0.36187707  0.0351059   0.85505825  0.65725351]]

Antithetic variates::
    >>> print(generate_samples(order=8, rule="H", antithetic=True))
    [[ 0.75   0.125  0.25   0.875]]
"""
import numpy

from . import collection


SAMPLERS = {
    "C": collection.create_chebyshev_samples,
    "NC": collection.create_nested_chebyshev_samples,
    "K": collection.create_korobov_samples,
    "R": lambda order, dim: numpy.random.random((dim, order)),
    "RG": collection.create_grid_samples,
    "NG": collection.create_nested_grid_samples,
    "L": collection.create_latin_hypercube_samples,
    "S": collection.create_sobol_samples,
    "H": collection.create_halton_samples,
    "M": collection.create_hammersley_samples,
}


def generate_samples(order, domain=(0, 1), rule="R", antithetic=None, verbose=False):
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
            rule for generating samples, where d is the number of dimensions.
        antithetic (array_like, optional):
            List of bool. Represents the axes to mirror using antithetic
            variable.
    """
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
        order_, order = order, int(order*2.**-size+1*(order % 2 != 0))
        trans_ = trans
        trans = lambda x_data: trans_(
            create_antithetic_variates(x_data, antithetic)[:, :order_])

    assert rule in SAMPLERS, "rule not recognised"
    sampler = SAMPLERS[rule]
    x_data = trans(sampler(order=order, dim=dim))

    return x_data
