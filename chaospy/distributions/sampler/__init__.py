r"""
Monte Carlo simulation is by nature a very slow converging method.  The error
in convergence is proportional to :math:`1/\sqrt{K}` where :math:`K` is the
number of samples.  It is somewhat better with variance reduction techniques
that often reaches errors proportional to :math:`1/K`. For a full overview of
the convergence rate of the various methods, see for example the excellent book
`handbook of Monte Carlo methods` by Kroese, Taimre and Botev. However as the
number of dimensions grows, Monte Carlo convergence rate stays the same, making
it immune to the curse of dimensionality.

Generating random samples can be done from the distribution instance method
``sample`` as discussed in the :ref:`tutorial`. For example, to generate nodes
from the Korobov lattice::

    >>> distribution = chaospy.Iid(chaospy.Beta(2, 2), 2)
    >>> samples = distribution.sample(4, rule="korobov")
    >>> samples.round(4)
    array([[0.2871, 0.4329, 0.5671, 0.7129],
           [0.4329, 0.7129, 0.2871, 0.5671]])

.. _handbook of Monte Carlo methods: https://onlinelibrary.wiley.com/doi/book/10.1002/9781118014967
"""
from .generator import generate_samples

from .sequences import *  # pylint: disable=wildcard-import
from .latin_hypercube import create_latin_hypercube_samples
from .antithetic import create_antithetic_variates
