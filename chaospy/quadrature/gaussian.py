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
using the e.g. discretized Stieltjes algorithm.

For example for the tailored quadrature rules defined above:

* Gauss-Hermit quadrature is tailored to the normal (Gaussian) distribution::

    >>> distribution = chaospy.Normal(0, 1)
    >>> abscissas, weights = chaospy.generate_quadrature(
    ...     5, distribution, rule="gaussian")
    >>> abscissas.round(4)
    array([[-3.3243, -1.8892, -0.6167,  0.6167,  1.8892,  3.3243]])
    >>> weights.round(4)
    array([0.0026, 0.0886, 0.4088, 0.4088, 0.0886, 0.0026])

* Gauss-Legendre quadrature is tailored to the Uniform distributions::

    >>> distribution = chaospy.Uniform(-1, 1)
    >>> abscissas, weights = chaospy.generate_quadrature(
    ...     5, distribution, rule="gaussian")
    >>> abscissas.round(4)
    array([[-0.9325, -0.6612, -0.2386,  0.2386,  0.6612,  0.9325]])
    >>> weights.round(4)
    array([0.0857, 0.1804, 0.234 , 0.234 , 0.1804, 0.0857])

* Gauss-Jacobi quadrature is tailored to the Beta distribution::

    >>> distribution = chaospy.Beta(2, 4, lower=-1, upper=1)
    >>> abscissas, weights = chaospy.generate_quadrature(
    ...     5, distribution, rule="gaussian")
    >>> abscissas.round(4)
    array([[-0.8969, -0.6679, -0.3448,  0.0289,  0.4029,  0.7279]])
    >>> weights.round(4)
    array([0.0749, 0.272 , 0.355 , 0.2253, 0.0667, 0.0062])

* Gauss-Laguerre quadrature is tailored to the Exponential distribution::

    >>> distribution = chaospy.Exponential()
    >>> abscissas, weights = chaospy.generate_quadrature(
    ...     5, distribution, rule="gaussian")
    >>> abscissas.round(4)
    array([[ 0.2228,  1.1889,  2.9927,  5.7751,  9.8375, 15.9829]])
    >>> weights.round(4)
    array([4.590e-01, 4.170e-01, 1.134e-01, 1.040e-02, 3.000e-04, 0.000e+00])

* Generalized Gauss-Laguerre quadrature is tailored to the Gamma distribution::

    >>> distribution = chaospy.Gamma(2, 4)
    >>> abscissas, weights = chaospy.generate_quadrature(
    ...     5, distribution, rule="gaussian")
    >>> abscissas.round(4)
    array([[ 2.1107,  7.1852, 15.5066, 27.6753, 44.9384, 70.5839]])
    >>> weights.round(4)
    array([0.2777, 0.4939, 0.203 , 0.0247, 0.0008, 0.    ])

For uncommon distributions an analytical Stieltjes method can not be performed
as the distribution does not provide three terms recursion coefficients. In
this scenario, the discretized counterpart is used instead as an approximation.
For example, to mention a few:

* The Triangle distribution::

    >>> distribution = chaospy.Triangle(-1, 0, 1)
    >>> abscissas, weights = chaospy.generate_quadrature(
    ...     5, distribution, rule="gaussian")
    >>> abscissas.round(4)
    array([[-0.8657, -0.5766, -0.1943,  0.1943,  0.5766,  0.8657]])
    >>> weights.round(4)
    array([0.0295, 0.1475, 0.323 , 0.323 , 0.1475, 0.0295])

* The Laplace distribution::

    >>> distribution = chaospy.Laplace(0, 1)
    >>> abscissas, weights = chaospy.generate_quadrature(
    ...     5, distribution, rule="gaussian")
    >>> abscissas.round(4)
    array([[-10.4917,  -4.6469,  -1.0404,   1.0404,   4.6469,  10.4917]])
    >>> weights.round(4)
    array([1.000e-04, 2.180e-02, 4.781e-01, 4.781e-01, 2.180e-02, 1.000e-04])

* The Weibull distribution::

    >>> distribution = chaospy.Weibull()
    >>> abscissas, weights = chaospy.generate_quadrature(
    ...     5, distribution, rule="gaussian")
    >>> abscissas.round(4)
    array([[ 0.2242,  1.19  ,  2.9933,  5.7746,  9.8349, 15.9752]])
    >>> weights.round(4)
    array([4.589e-01, 4.170e-01, 1.134e-01, 1.040e-02, 3.000e-04, 0.000e+00])

* The Rayleigh distribution::

    >>> distribution = chaospy.Rayleigh()
    >>> abscissas, weights = chaospy.generate_quadrature(
    ...     5, distribution, rule="gaussian")
    >>> abscissas.round(4)
    array([[0.2474, 0.7688, 1.4797, 2.3318, 3.3233, 4.5304]])
    >>> weights.round(4)
    array([9.600e-02, 3.592e-01, 3.891e-01, 1.412e-01, 1.430e-02, 2.000e-04])

"""
from .recurrence import (
    construct_recurrence_coefficients, coefficients_to_quadrature)
from .combine import combine_quadrature


def quad_gaussian(
        order,
        dist,
        rule="fejer",
        accuracy=200,
        recurrence_algorithm="",
):
    """
    Generating Gaussian quadrature by first generating so called *three terms
    recurrence* coefficients using one various different algorithms. The
    coefficients are them converted to abscissas and weights by constructing
    lower banded Jakobi matrix and calculating eigenvalues and eigenvectors,
    which can be directly translated to abscissas and weights. Construction of
    the coefficients is potentially numerically unstable, there for multiple
    algorithms exists.

    Args:
        dist (chaospy.distributions.baseclass.Distribution):
            The distribution which density will be used as weight function.
        order (int):
            The order of the quadrature.
        rule (str):
            In the case of ``lanczos`` or ``stieltjes``, defines the
            proxy-integration scheme.
        accuracy (int):
            In the case ``rule`` is used, defines the quadrature order of the
            scheme used. In practice, must be at least as large as ``order``.
        recurrence_algorithm (str):
            Name of the algorithm used to generate abscissas and weights. If
            omitted, ``analytical`` will be tried first, and ``stieltjes`` used
            if that fails.

    Returns:
        (numpy.ndarray, numpy.ndarray):
            Gaussian quadrature abscissas and weights with shapes
            ``(len(dist), order+1)`` and ``(order+1,)`` respectively.

    Raises:
        NotImplementedError:
            In the case of recurrence algorithm ``analytical``, error is raised
            if the distribution does not implement the three terms recurrence
            algorithm analytically.
        numpy.linalg.LinAlgError:
            For non-canonical random variables, the construction might fail
            because of illegal numerical operations.

    Examples:
        >>> distribution = chaospy.Normal(0, 1)
        >>> abscissas, weights = chaospy.quad_gaussian(
        ...     5, distribution, recurrence_algorithm="stieltjes")
        >>> abscissas.round(4)
        array([[-3.3243, -1.8892, -0.6167,  0.6167,  1.8892,  3.3243]])
        >>> weights.round(4)
        array([0.0026, 0.0886, 0.4088, 0.4088, 0.0886, 0.0026])
        >>> distribution = chaospy.J(chaospy.Uniform(), chaospy.Normal())
        >>> abscissas, weights = chaospy.quad_gaussian(
        ...     2, distribution, recurrence_algorithm="chebyshev")
        >>> abscissas.round(2)
        array([[ 0.11,  0.11,  0.11,  0.5 ,  0.5 ,  0.5 ,  0.89,  0.89,  0.89],
               [-1.73,  0.  ,  1.73, -1.73,  0.  ,  1.73, -1.73,  0.  ,  1.73]])
        >>> weights.round(3)
        array([0.046, 0.185, 0.046, 0.074, 0.296, 0.074, 0.046, 0.185, 0.046])
    """
    coefficients = construct_recurrence_coefficients(
        order, dist, rule, accuracy, recurrence_algorithm)
    abscissas, weights = coefficients_to_quadrature(coefficients)
    return combine_quadrature(abscissas, weights)
