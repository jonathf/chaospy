r"""
Most integration problems when dealing with polynomial chaos expansion comes
with a weight function :math:`p(x)` which happens to be the probability density
function. Gaussian quadrature creates weights and abscissas that are tailored
to be optimal with the inclusion of a weight function. It is therefore not one
method, but a collection of methods, each tailored to different probability
density functions.

In ``chaospy`` Gaussian quadrature is a functionality attached to each
probability distribution. This means that instead of explicitly supporting
a list of quadrature rules, all rules are supported through the capability of
the distribution implementation. For common distribution, this means that the
quadrature rules are calculated analytically using Stieltjes method on known
three terms recursion coefficients, and using those to create quadrature node
using the Golub-Welsch algorithm.

For example for the tailored quadrature rules defined above:

* Gauss-Hermit quadrature is tailored to the normal (Gaussian) distribution::

    >>> X, W = chaospy.generate_quadrature(4, chaospy.Normal(0, 1), rule="G")
    >>> print("{} {}".format(numpy.around(X, 3), numpy.around(W, 3)))
    [[-2.857 -1.356 -0.     1.356  2.857]] [0.011 0.222 0.533 0.222 0.011]

* Gauss-Legendre quadrature is tailored to the Uniform distributions::

    >>> X, W = chaospy.generate_quadrature(4, chaospy.Uniform(0, 1), rule="G")
    >>> print("{} {}".format(numpy.around(X, 3), numpy.around(W, 3)))
    [[0.047 0.231 0.5   0.769 0.953]] [0.118 0.239 0.284 0.239 0.118]

* Gauss-Jacobi quadrature is tailored to the Beta distribution::

    >>> X, W = chaospy.generate_quadrature(4, chaospy.Beta(2, 4), rule="G")
    >>> print("{} {}".format(numpy.around(X, 3), numpy.around(W, 3)))
    [[0.067 0.212 0.41  0.627 0.827]] [0.118 0.367 0.36  0.139 0.015]

* Gauss-Laguerre quadrature is tailored to the Exponential distribution::

    >>> X, W = chaospy.generate_quadrature(4, chaospy.Exponential(), rule="G")
    >>> print("{} {}".format(numpy.around(X, 3), numpy.around(W, 3)))
    [[ 0.264  1.413  3.596  7.086 12.641]] [0.522 0.399 0.076 0.004 0.   ]

* Generalized Gauss-Laguerre quadrature is tailored to the Gamma distribution::

    >>> X, W = chaospy.generate_quadrature(4, chaospy.Gamma(2, 4), rule="G")
    >>> print("{} {}".format(numpy.around(X, 3), numpy.around(W, 3)))
    [[ 2.468  8.452 18.443 33.596 57.04 ]] [0.348 0.502 0.141 0.009 0.   ]

For uncommon distributions an analytical Stieltjes method can not be performed
as the distribution does not provide three terms recursion coefficients. In
this scenario, the discretized counterpart is used instead as an approximation.
For example, to mention a few:

* The triangle distribution::

    >>> X, W = chaospy.generate_quadrature(4, chaospy.Triangle(), rule="G")
    >>> print("{} {}".format(numpy.around(X, 3), numpy.around(W, 3)))
    [[-0.821 -0.45  -0.     0.45   0.821]] [0.052 0.239 0.418 0.239 0.052]

* The Laplace distribution::

    >>> X, W = chaospy.generate_quadrature(4, chaospy.Laplace(), rule="G")
    >>> print("{} {}".format(numpy.around(X, 3), numpy.around(W, 3)))
    [[-8.091 -2.855 -0.     2.855  8.091]] [0.001 0.114 0.771 0.114 0.001]

* The Weibull distribution::

    >>> X, W = chaospy.generate_quadrature(4, chaospy.Weibull(), rule="G")
    >>> print("{} {}".format(numpy.around(X, 3), numpy.around(W, 3)))
    [[ 0.264  1.413  3.596  7.086 12.64 ]] [0.522 0.399 0.076 0.004 0.   ]

* The Rayleigh distribution::

    >>> X, W = chaospy.generate_quadrature(4, chaospy.Rayleigh(), rule="G")
    >>> print("{} {}".format(numpy.around(X, 3), numpy.around(W, 3)))
    [[0.308 0.938 1.779 2.79  4.032]] [0.144 0.453 0.339 0.063 0.002]

As a small side note, it is worth noting that since the weight function is
assumed to be a probability density function, we here only focuses on
probabilistic Gaussian quadrature. This means that we assume the constraint:

.. math::

    \int p(x) dx = 1

There is also another version, often named as the physicist version of Gaussian
quadrature. They are for the most part the same, but the latter have other
weighting constraints. To jump between the two variants, most often one has to
multiply the weights with an appropriate constant to achieve the correct
values.
"""
import numpy
import scipy.linalg

import chaospy.quad


def quad_golub_welsch(order, dist, accuracy=100, **kws):
    """
    Golub-Welsch algorithm for creating quadrature nodes and weights.

    Args:
        order (int):
            Quadrature order
        dist (Dist):
            Distribution nodes and weights are found for with `dim=len(dist)`
        accuracy (int):
            Accuracy used in discretized Stieltjes procedure. Will
            be increased by one for each iteration.

    Returns:
        (numpy.ndarray, numpy.ndarray):
            Optimal collocation nodes with `x.shape=(dim, order+1)` and weights
            with `w.shape=(order+1,)`.

    Examples:
        >>> Z = chaospy.Normal()
        >>> x, w = chaospy.quad_golub_welsch(3, Z)
        >>> print(numpy.around(x, 4))
        [[-2.3344 -0.742   0.742   2.3344]]
        >>> print(numpy.around(w, 4))
        [0.0459 0.4541 0.4541 0.0459]
        >>> Z = chaospy.J(chaospy.Uniform(), chaospy.Uniform())
        >>> x, w = chaospy.quad_golub_welsch(1, Z)
        >>> print(numpy.around(x, 4))
        [[0.2113 0.2113 0.7887 0.7887]
         [0.2113 0.7887 0.2113 0.7887]]
        >>> print(numpy.around(w, 4))
        [0.25 0.25 0.25 0.25]
    """
    order = numpy.array(order)*numpy.ones(len(dist), dtype=int)+1
    _, _, coeff1, coeff2 = chaospy.quad.generate_stieltjes(
        dist, numpy.max(order), accuracy=accuracy, retall=True, **kws)

    dimensions = len(dist)
    abscisas, weights = _golbub_welsch(order, coeff1, coeff2)

    if dimensions == 1:
        abscisa = numpy.reshape(abscisas, (1, order[0]))
        weight = numpy.reshape(weights, (order[0],))
    else:
        abscisa = chaospy.quad.combine(abscisas).T
        weight = numpy.prod(chaospy.quad.combine(weights), -1)

    assert len(abscisa) == dimensions
    assert len(weight) == len(abscisa.T)
    return abscisa, weight


def _golbub_welsch(orders, coeff1, coeff2):
    """Recurrence coefficients to abscisas and weights."""
    abscisas, weights = [], []

    for dim, order in enumerate(orders):
        if order:
            bands = numpy.zeros((2, order))
            bands[0] = coeff1[dim, :order]
            bands[1, :-1] = numpy.sqrt(coeff2[dim, 1:order])
            vals, vecs = scipy.linalg.eig_banded(bands, lower=True)

            abscisa, weight = vals.real, vecs[0, :]**2
            indices = numpy.argsort(abscisa)
            abscisa, weight = abscisa[indices], weight[indices]

        else:
            abscisa, weight = numpy.array([coeff1[dim, 0]]), numpy.array([1.])

        abscisas.append(abscisa)
        weights.append(weight)
    return abscisas, weights
