# pylint: disable=wildcard-import
r"""
As noted throughout the documentation, known distributions are created easily
by calling their name of the function of interest. For example to create
a Gaussian random variable::

   >>> distribution = chaospy.Normal(0,1)

To construct simple multivariate random variables with stochastically
independent components, either all the same using
:func:`~chaospy.distributions.joint.Iid`::

   >>> distribution = chaospy.Iid(chaospy.Normal(0, 1), 2)

Or with more detailed control through :func:`~chaospy.distributions.joint.J`::

   >>> distribution = chaospy.J(chaospy.Normal(0, 1), chaospy.Normal(0, 1))

The functionality of the distributions are covered in various other sections:

* To generate random samples, see :ref:`montecarlo`.
* To create transformations, see :ref:`rosenblatt`.
* To generate raw statistical moments, see :ref:`moments`.
* To generate three terms recurrence coefficients, see :ref:`orthogonality`.
* To analyse statistical properies, see :ref:`descriptives`.
"""
from . import baseclass
from .baseclass import Dist

from .constructor import construct

from . import (
    graph, sampler, approx, joint, cores,
    copulas, collection, operators, rosenblatt,
)

from .graph import *
from .sampler import *
from .approx import *
from .joint import *
from .cores import *
from .copulas import *
from .collection import *
from .operators import *

from numpy.random import seed
