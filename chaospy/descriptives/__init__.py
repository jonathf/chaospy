"""
Descriptives are a collection of statistical analysis tools that can be used to
analyse :ref:`distributions` and :ref`polynomials`, both as an expansion (see
:ref:`orthogonality`) and as results (see :ref:`regression` and
:ref:`spectral`).  For example, the expected value operator
:func:`~chaospy.descriptives.E` can be applied on distributions directly as
follows::

    >>> distribution = cp.Uniform(0, 1)
    >>> expected = cp.E(distribution)
    >>> print(expected)
    0.5

For multivariate distributions::

    >>> distribution = cp.J(
    ...     cp.Uniform(0, 1),
    ...     cp.Normal(0, 1)
    ... )
    >>> expected = cp.E(distribution)
    >>> print(expected)
    [ 0.5  0. ]


For simple polynomials, distribution goes as the second argument. In other
words, it calculates the expected value of the unit variable with respect to
distribution. For example::

    >>> distribution = cp.J(
    ...     cp.Uniform(0, 1),
    ...     cp.Normal(0, 1)
    ... )
    >>> q0, q1 = cp.variable(2)
    >>> expected = cp.E(q1, distribution)
    >>> print(expected)
    0.0
"""
import chaospy as cp

import chaospy.descriptives.first
import chaospy.descriptives.second1d
import chaospy.descriptives.second2d
import chaospy.descriptives.higher
import chaospy.descriptives.sensitivity
import chaospy.descriptives.misc

from chaospy.descriptives.first import E, E_cond
from chaospy.descriptives.second1d import Var, Std
from chaospy.descriptives.second2d import Cov, Corr
from chaospy.descriptives.higher import Skew, Kurt

from chaospy.descriptives.sensitivity import (
    Sens_m, Sens_m2, Sens_m_nataf, Sens_nataf, Sens_t, Sens_t_nataf,
)
from chaospy.descriptives.misc import Acf, Spearman, Perc, QoI_Dist
