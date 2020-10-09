# pylint: disable=wildcard-import
"""
One of the backbone of any uncertainty quantification is a collection of
probability distributions, and ``chaospy`` is no exception. For example, to
create a Gaussian random variable::

    >>> distribution = chaospy.Normal(mu=2, sigma=2)

The syntax for using distribution is here very similar to the syntax used in
``scipy.dist``. For example, to create values from the *probability density
function*::

    >>> t = numpy.linspace(-3, 3, 9)
    >>> distribution.pdf(t).round(3)
    array([0.009, 0.021, 0.043, 0.078, 0.121, 0.164, 0.193, 0.198, 0.176])

Similarly to create values from the *cumulative distribution function*::

    >>> distribution.cdf(t).round(3)
    array([0.006, 0.017, 0.04 , 0.085, 0.159, 0.266, 0.401, 0.55 , 0.691])

To be able to perform any Monte Carlo method, each distribution contains
*random number generator*::

    >>> distribution.sample(6).round(4)
    array([ 2.7901, -0.4006,  5.2952,  1.9107,  4.2763,  0.4033])

The sample scheme also has a few advanced options. For example, to create
low-discrepancy Hammersley sequences samples combined with antithetic variates::

    >>> distribution.sample(size=6, rule="halton", antithetic=True).round(4)
    array([ 3.349 ,  0.651 , -0.3007,  4.3007,  2.6373,  1.3627])

For a full overview of these options, see :ref:`sampling`

In addition a function for extracting raw-statistical moments is available::

    >>> distribution.mom([0, 1, 2, 3, 4])
    array([  1.,   2.,   8.,  32., 160.])

Note that these are raw moments, not the classical moments with adjustments.
For example, the variance is defined as follows::

    >>> distribution.mom(2) - distribution.mom(1)**2
    4.0

However, if the adjusted moments are of interest, the can be retrieved using
the tools described in :ref:`descriptives`::

    >>> chaospy.Var(distribution)
    array(4.)

Random Seed
-----------

To be able to reproduce results it is possible to fix the random seed in
``chaospy``. For simplicity, The library respect ``numpy.random.seed``. E.g.::

    >>> numpy.random.seed(1234)
    >>> distribution.sample(5).round(4)
    array([0.2554, 2.622 , 1.6865, 3.5808, 3.5442])
    >>> numpy.random.seed(1234)
    >>> distribution.sample(5).round(4)
    array([0.2554, 2.622 , 1.6865, 3.5808, 3.5442])
    >>> distribution.sample(5).round(4)
    array([0.79  , 0.8132, 3.6967, 5.459 , 4.3098])

"""
from .baseclass import *
from .sampler import *
from .collection import *
from .copulas import *
from .operators import *
from .constructor import construct
from .approximation import *

from . import (
    baseclass, sampler, approximation,
    copulas, collection, operators, constructor
)
