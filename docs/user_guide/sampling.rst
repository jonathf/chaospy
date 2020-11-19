.. _sampling:

Variance Reduction
==================

Introduction
------------

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

.. _generator:

Generator function
------------------

Each sampling scheme can be accessed through the ``sample`` method on each
distribution. But in addition, they can also be created on the unit hyper-cube
using direct sampling functions. The frontend for all these functions is the
:func:`chaospy.generate_samples` function. It allows for the same functionality
as the ``sample`` method, but also support some extra functionality by not
being associated with a specific distribution. For example::

    >>> samples = generate_samples(order=4)
    >>> samples.round(4)
    array([[0.6536, 0.115 , 0.9503, 0.4822]])

Custom domain::

    >>> samples = generate_samples(order=4, domain=[-1, 1])
    >>> samples.round(4)
    array([[ 0.7449, -0.5753, -0.9186, -0.2056]])
    >>> samples = generate_samples(order=4, domain=chaospy.Normal(0, 1))
    >>> samples.round(4)
    array([[-0.7286,  1.0016, -0.8166,  0.651 ]])

Use a custom sampling scheme::

    >>> generate_samples(order=4, rule="halton").round(4)
    array([[0.75 , 0.125, 0.625, 0.375]])

Multivariate case::

    >>> samples = generate_samples(order=4, domain=[[-1, 0], [0, 1]])
    >>> samples.round(4)
    array([[-0.6078, -0.8177, -0.2565, -0.9304],
           [ 0.8853,  0.9526,  0.9311,  0.4154]])
    >>> distribution = chaospy.J(chaospy.Normal(0, 1), chaospy.Uniform(0, 1))
    >>> samples = generate_samples(order=4, domain=distribution)
    >>> samples.round(4)
    array([[-1.896 ,  2.0975, -0.4135,  0.5437],
           [ 0.3619,  0.0351,  0.8551,  0.6573]])

Antithetic variates::

    >>> samples = generate_samples(order=8, rule="halton", antithetic=True)
    >>> samples.round(4)
    array([[0.75 , 0.25 , 0.125, 0.875, 0.625, 0.375, 0.375, 0.625]])

Multivariate antithetic variates::

    >>> samples = generate_samples(
    ...     order=8, domain=2, rule="hammersley", antithetic=True)
    >>> samples.round(4)
    array([[0.75 , 0.25 , 0.75 , 0.25 , 0.125, 0.875, 0.125, 0.875],
           [0.25 , 0.25 , 0.75 , 0.75 , 0.5  , 0.5  , 0.5  , 0.5  ]])

Here as with the ``sample`` method, the flag ``rule`` is used to determine
sampling scheme. The default ``rule="random"`` uses classical pseudo-random
samples created using ``numpy.random``.


Low-discrepancy sequences
-------------------------

In mathematics, a `low-discrepancy sequence`_ is a sequence with the property
that for all values of N, its subsequence x1, ..., xN has a low discrepancy.

Roughly speaking, the discrepancy of a sequence is low if the proportion of
points in the sequence falling into an arbitrary set B is close to proportional
to the measure of B, as would happen on average (but not for particular
samples) in the case of an equi-distributed sequence. Specific definitions of
discrepancy differ regarding the choice of B (hyperspheres, hypercubes, etc.)
and how the discrepancy for every B is computed (usually normalized) and
combined (usually by taking the worst value).

Low-discrepancy sequences are also called quasi-random or sub-random sequences,
due to their common use as a replacement of uniformly distributed random
numbers. The "quasi" modifier is used to denote more clearly that the values of
a low-discrepancy sequence are neither random nor pseudorandom, but such
sequences share some properties of random variables and in certain applications
such as the quasi-Monte Carlo method their lower discrepancy is an important
advantage.

.. _low-discrepancy sequence: https://en.wikipedia.org/wiki/Low-discrepancy_sequence

.. _antithetic:

Antithetic Variates
-------------------

Create `antithetic variates`_ from variables on the unit hyper-cube.

In statistics, the antithetic variates method is a variance reduction technique
used in Monte Carlo methods. Considering that the error reduction in the
simulated signal (using Monte Carlo methods) has a square root convergence,
a very large number of sample paths is required to obtain an accurate result.
The antithetic variates method reduces the variance of the simulation results.

Antithetic variate can be accessed as a flag ``antithetic`` in the method
``Distribution.sample`` It can either be set to ``True``, for activation, or as an
array of boolean values, which implies it will be used as the flag ``axes`` in
the examples below.

.. _antithetic variates: https://en.wikipedia.org/wiki/Antithetic_variates

Example usage
-------------

Creating antithetic variates can be done directly from each distribution by
using the ``antithetic`` flag::

    >>> distribution = chaospy.Uniform(0, 1)
    >>> samples = distribution.sample(6, antithetic=True)

Antithetic variates contains compliment values of itself::

    >>> samples.round(4)
    array([0.6536, 0.3464, 0.115 , 0.885 , 0.9503, 0.0497])
    >>> 1-samples.round(4)
    array([0.3464, 0.6536, 0.885 , 0.115 , 0.0497, 0.9503])

Antithetic variates can also be used in multiple dimensions::

    >>> distribution = chaospy.Iid(chaospy.Uniform(0, 1), 2)
    >>> samples = distribution.sample(6, antithetic=True)
    >>> samples.round(4)
    array([[0.8725, 0.1275, 0.8725, 0.1275, 0.2123, 0.7877],
           [0.3972, 0.3972, 0.6028, 0.6028, 0.2331, 0.2331]])
    >>> 1-samples.round(4)
    array([[0.1275, 0.8725, 0.1275, 0.8725, 0.7877, 0.2123],
           [0.6028, 0.6028, 0.3972, 0.3972, 0.7669, 0.7669]])

Lastly, it is also possible to select which axes should be included when
applying the variate by passing a boolean array. For axes that are "false", the
value is frozen in place::

    >>> samples = distribution.sample(6, antithetic=[True, False])
    >>> samples.round(4)
    array([[0.2071, 0.7929, 0.7425, 0.2575, 0.3922, 0.6078],
           [0.1823, 0.1823, 0.7435, 0.7435, 0.0696, 0.0696]])
    >>> 1-samples.round(4)
    array([[0.7929, 0.2071, 0.2575, 0.7425, 0.6078, 0.3922],
           [0.8177, 0.8177, 0.2565, 0.2565, 0.9304, 0.9304]])
    >>> samples = distribution.sample(6, antithetic=[False, True])
    >>> samples.round(4)
    array([[0.8853, 0.8853, 0.9526, 0.9526, 0.9311, 0.9311],
           [0.4154, 0.5846, 0.029 , 0.971 , 0.982 , 0.018 ]])
    >>> 1-samples.round(4)
    array([[0.1147, 0.1147, 0.0474, 0.0474, 0.0689, 0.0689],
           [0.5846, 0.4154, 0.971 , 0.029 , 0.018 , 0.982 ]])
