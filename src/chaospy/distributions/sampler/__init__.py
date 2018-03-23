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
For example, to generate nodes from the Korobov lattice::

    >>> distribution = chaospy.Iid(chaospy.Beta(2, 2), 2)
    >>> samples = distribution.sample(4, rule="K")
    >>> print(numpy.around(samples, 4))
    [[0.2871 0.4329 0.5671 0.7129]
     [0.4329 0.7129 0.2871 0.5671]]

At the core of `chaospy`, all samples are generated to the unit hyper-cube
through the function :func:`~chaospy.distributions.samplers.generator.generate_samples`.
These samples are then mapped using a :ref:`rosenblatt` to map the values on
the domain respective to the distribution in question. This way, all variance
reduction techniques are supported by all distributions.

If one wants even more control over the sample generation using low-discrepancy
sequencecs, it is also possible to take a look at
:mod:`chaospy.distributions.sampler.generator.sequences`. This module will contain all
the individual sequence schemes with extra flags for controlling each scheme.
In the case of the Krobov lattice::

    >>> from chaospy.distributions.sampler import create_korobov_samples
    >>> print(create_korobov_samples(order=4, dim=2))
    [[0.2 0.4 0.6 0.8]
     [0.4 0.8 0.2 0.6]]
    >>> print(create_korobov_samples(order=4, dim=2, base=1234))
    [[0.2 0.4 0.6 0.8]
     [0.8 0.6 0.4 0.2]]
"""
from .generator import generate_samples

from .sequences import *  # pylint: disable=wildcard-import
from .latin_hypercube import create_latin_hypercube_samples
from .antithetic import create_antithetic_variates
