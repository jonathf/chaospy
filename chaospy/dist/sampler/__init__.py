"""
Monte Carlo simulation is by nature a very slow converging method.  The error
in convergence is proportional to :math:`1/\sqrt{K}` where :math:`K` is the
number of samples.  It is somewhat better with variance reduction techniques
that often reaches errors proportional to :math:`1/K`. For a
full overview of the convergence rate of the various methods, see for
example :cite:`kroese_handbook_2011`. However as the number
of dimensions grows, Monte Carlo convergence rate stays the same, making it
immune to the curse of dimensionality.

Generating random samples can be done from the distribution instance
method ``sample`` as discussed in the :ref:`tutorial`.
For example, to generate nodes from the Korobov latice::

   >>> distribution = pc.Iid(pc.Beta(2, 2), 2)
   >>> samples = distribution.sample(4, rule="K")
   >>> print(nodes)
   [[ 0.28714073  0.43293108  0.56706892  0.71285927]
    [ 0.43293108  0.71285927  0.28714073  0.56706892]]


At the core of `chaospy`, all samples are generated to the unit hyper-cube
through the function :func:`~chaospy.dist.samplegen`. These samples are then
mapped using a :ref:`rosenblatt` to map the values on the domain respective to
the distribution in question. This way, all variance reduction techniques are
supported by all distributions.
"""

import numpy as np

from .generator import samplegen
from .samplers import *
from .sobol_lib import sobol
from .antithetic import antithetic_gen


if __name__=="__main__":
    import __init__ as cp
    import doctest
    doctest.testmod()
