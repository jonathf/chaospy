"""
Descriptives are a collection of statistical analysis tools that can be used to
analyse distributions and polynomials, both as an expansion (see
:ref:`orthogonality`) and as results. For example, the expected value operator
:func:`~chaospy.descriptives.expected.E` can be applied on distributions
directly as follows::

    >>> distribution = chaospy.Uniform(0, 1)
    >>> expected = chaospy.E(distribution)
    >>> expected
    array(0.5)

For multivariate distributions::

    >>> distribution = chaospy.J(
    ...     chaospy.Uniform(0, 1),
    ...     chaospy.Normal(0, 1)
    ... )
    >>> expected = chaospy.E(distribution)
    >>> expected
    array([0.5, 0. ])


For simple polynomials, distribution goes as the second argument. In other
words, it calculates the expected value of the unit variable with respect to
distribution. For example::

    >>> distribution = chaospy.J(
    ...     chaospy.Uniform(0, 1),
    ...     chaospy.Normal(0, 1)
    ... )
    >>> q0, q1 = chaospy.variable(2)
    >>> expected = chaospy.E(q1, distribution)
    >>> expected
    array(0.)
"""
from .expected import E
from .conditional import E_cond
from .covariance import Cov
from .kurtosis import Kurt
from .percentile import Perc
from .quantity_of_interest import QoI_Dist
from .skewness import Skew
from .standard_deviation import Std
from .variance import Var

from .correlation import *
from .sensitivity import *
