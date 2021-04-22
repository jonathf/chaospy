.. _sampling:

Sampling and sequences
======================

Introduction
------------

Monte Carlo simulation is by nature a very slow converging method.  The error
in convergence is proportional to :math:`1/\sqrt{K}` where :math:`K` is the
number of samples.  It is somewhat better with variance reduction techniques
that often reaches errors proportional to :math:`1/K`. For a full overview of
the convergence rate of the various methods, see for example the excellent book
"handbook of Monte Carlo methods" by Kroese, Taimre and Botev
:cite:`kroese_handbook_2011`. However as the number of dimensions grows, Monte
Carlo convergence rate stays the same, making it immune to the curse of
dimensionality.

Generating random samples can be done from the distribution instance method
:func:`~chaospy.Distribution.sample` as discussed in the :ref:`tutorial`. For
example, to generate nodes from the Korobov lattice:

.. code-block:: python

    >>> distribution = chaospy.Iid(chaospy.Beta(2, 2), 2)
    >>> samples = distribution.sample(4, rule="korobov")
    >>> samples.round(4)
    array([[0.2871, 0.4329, 0.5671, 0.7129],
           [0.4329, 0.7129, 0.2871, 0.5671]])

See :ref:`low_discrepancy_sequences` for an overview over the available rules.
See also the `Monte Carlo integration example
<../tutorials/monte_carlo_integration.ipynb>`_ for a demonstration of some of
the functionality available.

.. _handbook of Monte Carlo methods: https://onlinelibrary.wiley.com/doi/book/10.1002/9781118014967

.. _generator:

Generator function
------------------

Each sampling scheme can be accessed through the ``sample`` method on each
distribution. But in addition, they can also be created on the unit hyper-cube
using direct sampling functions. The frontend for all these functions is the
:func:`chaospy.generate_samples` function. It allows for the same functionality
as the :func:`~chaospy.Distribution.sample` method, but also support some extra
functionality by not being associated with a specific distribution. For
example:

.. code-block:: python

    >>> samples = chaospy.generate_samples(order=4)
    >>> samples.round(4)
    array([[0.6536, 0.115 , 0.9503, 0.4822]])

Using a custom domain:

.. code-block:: python

    >>> samples = chaospy.generate_samples(order=4, domain=[-1, 1])
    >>> samples.round(4)
    array([[ 0.7449, -0.5753, -0.9186, -0.2056]])
    >>> samples = chaospy.generate_samples(order=4, domain=chaospy.Normal(0, 1))
    >>> samples.round(4)
    array([[-0.7286,  1.0016, -0.8166,  0.651 ]])

Use a custom sampling scheme:

.. code-block:: python

    >>> chaospy.generate_samples(order=4, rule="halton").round(4)
    array([[0.75 , 0.125, 0.625, 0.375]])

Multivariate case:

.. code-block:: python

    >>> samples = chaospy.generate_samples(order=4, domain=[[-1, 0], [0, 1]])
    >>> samples.round(4)
    array([[-0.6078, -0.8177, -0.2565, -0.9304],
           [ 0.8853,  0.9526,  0.9311,  0.4154]])
    >>> distribution = chaospy.J(chaospy.Normal(0, 1), chaospy.Uniform(0, 1))
    >>> samples = chaospy.generate_samples(order=4, domain=distribution)
    >>> samples.round(4)
    array([[-1.896 ,  2.0975, -0.4135,  0.5437],
           [ 0.3619,  0.0351,  0.8551,  0.6573]])

Antithetic variates:

.. code-block:: python

    >>> samples = chaospy.generate_samples(order=8, rule="halton", antithetic=True)
    >>> samples.round(4)
    array([[0.75 , 0.25 , 0.125, 0.875, 0.625, 0.375, 0.375, 0.625]])

Multivariate antithetic variates:

.. code-block:: python

    >>> samples = chaospy.generate_samples(
    ...     order=8, domain=2, rule="halton", antithetic=True)
    >>> samples.round(4)
    array([[0.125 , 0.875 , 0.125 , 0.875 , 0.625 , 0.375 , 0.625 , 0.375 ],
           [0.4444, 0.4444, 0.5556, 0.5556, 0.7778, 0.7778, 0.2222, 0.2222]])

Here as with the :func:`~chaospy.Distribution.sample` method, the flag ``rule``
is used to determine sampling scheme. The default ``rule="random"`` uses
classical pseudo-random samples created using :mod:`numpy.random`.


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
a low-discrepancy sequence are neither random nor pseudo-random, but such
sequences share some properties of random variables and in certain applications
such as the quasi-Monte Carlo method their lower discrepancy is an important
advantage.

.. _low-discrepancy sequence: https://en.wikipedia.org/wiki/Low-discrepancy_sequence

.. _antithetic:

Antithetic variates
-------------------

Create `antithetic variates`_ from variables on the unit hyper-cube.

In statistics, the antithetic variates method is a variance reduction technique
used in Monte Carlo methods. Considering that the error reduction in the
simulated signal (using Monte Carlo methods) has a square root convergence,
a very large number of sample paths is required to obtain an accurate result.
The antithetic variates method reduces the variance of the simulation results.

Antithetic variate can be accessed as a flag ``antithetic`` in the method
:func:`~chaospy.Distribution.sample` It can either be set to ``True``, for
activation, or as an array of boolean values, which implies it will be used as
the flag ``axes`` in the examples below.

Creating antithetic variates can be done directly from each distribution by
using the ``antithetic`` flag:

.. code-block:: python

    >>> distribution = chaospy.Uniform(0, 1)
    >>> samples = distribution.sample(6, antithetic=True)

Antithetic variates contains compliment values of itself:

.. code-block:: python

    >>> samples.round(4)
    array([0.7657, 0.2343, 0.5541, 0.4459, 0.8851, 0.1149])
    >>> 1-samples.round(4)
    array([0.2343, 0.7657, 0.4459, 0.5541, 0.1149, 0.8851])

Antithetic variates can also be used in multiple dimensions:

.. code-block:: python

    >>> distribution = chaospy.Iid(chaospy.Uniform(0, 1), 2)
    >>> samples = distribution.sample(6, antithetic=True)
    >>> samples.round(4)
    array([[0.0104, 0.9896, 0.0104, 0.9896, 0.0746, 0.9254],
           [0.1333, 0.1333, 0.8667, 0.8667, 0.6979, 0.6979]])
    >>> 1-samples.round(4)
    array([[0.9896, 0.0104, 0.9896, 0.0104, 0.9254, 0.0746],
           [0.8667, 0.8667, 0.1333, 0.1333, 0.3021, 0.3021]])

Lastly, it is also possible to select which axes should be included when
applying the variate by passing a boolean array. For axes that are "false", the
value is frozen in place:

.. code-block:: python

    >>> samples = distribution.sample(6, antithetic=[True, False])
    >>> samples.round(4)
    array([[0.8831, 0.1169, 0.181 , 0.819 , 0.4325, 0.5675],
           [0.0181, 0.0181, 0.6914, 0.6914, 0.4697, 0.4697]])
    >>> 1-samples.round(4)
    array([[0.1169, 0.8831, 0.819 , 0.181 , 0.5675, 0.4325],
           [0.9819, 0.9819, 0.3086, 0.3086, 0.5303, 0.5303]])
    >>> samples = distribution.sample(6, antithetic=[False, True])
    >>> samples.round(4)
    array([[0.1282, 0.1282, 0.8913, 0.8913, 0.9182, 0.9182],
           [0.0731, 0.9269, 0.0454, 0.9546, 0.4386, 0.5614]])
    >>> 1-samples.round(4)
    array([[0.8718, 0.8718, 0.1087, 0.1087, 0.0818, 0.0818],
           [0.9269, 0.0731, 0.9546, 0.0454, 0.5614, 0.4386]])

.. _antithetic variates: https://en.wikipedia.org/wiki/Antithetic_variates
