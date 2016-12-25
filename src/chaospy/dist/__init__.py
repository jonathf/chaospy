r"""
As noted throughout the documentation, known distributions are created easily
by calling their name of the function of interest. For example to create
a Gaussian random variable::

   >>> distribution = cp.Normal(0,1)

To construct simple multivariate random variables with stochastically
independent components, either all the same using :func:`~chaospy.dist.Iid`::

   >>> distribution = cp.Iid(cp.Normal(0, 1), 2)

Or with more detailed control through :func:`~chaospy.dist.J`::

   >>> distribution = cp.J(cp.Normal(0, 1), cp.Normal(0, 1))

The functionality of the distributions are covered in various other sections:

* To generate random samples, see :ref:`montecarlo`.
* To create transformations, see :ref:`rosenblatt`.
* To generate raw statistical moments, see :ref:`moments`.
* To generate three terms recurrence coefficients, see :ref:`orthogonality`.
* To analyse statistical properies, see :ref:`descriptives`.

"""

import chaospy.dist.baseclass
from chaospy.dist.baseclass import Dist

import chaospy.dist.graph
import chaospy.dist.sampler
import chaospy.dist.approx
import chaospy.dist.joint
import chaospy.dist.cores
import chaospy.dist.copulas
import chaospy.dist.collection
import chaospy.dist.operators
import chaospy.dist.rosenblatt

from chaospy.dist.graph import *
from chaospy.dist.sampler import *
from chaospy.dist.approx import *
from chaospy.dist.joint import *
from chaospy.dist.cores import *
from chaospy.dist.copulas import *
from chaospy.dist.collection import *
from chaospy.dist.operators import *

from numpy.random import seed


if __name__ == "__main__":
    seed(1000)
    import doctest
    import chaospy as cp
    doctest.testmod()
