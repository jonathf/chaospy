.. _distributions:

Probability distributions
=========================

One of the backbone of any uncertainty quantification is a collection of
random variables, and ``chaospy`` is no exception. For example, to create a
Gaussian random variable:

.. code-block:: python

    >>> distribution = chaospy.Normal(mu=2, sigma=2)

The syntax for using distribution is here very similar to the syntax used in
:mod:`scipy.stats`. For example, to create values from the *probability density
function* :func:`~chaospy.Distribution.pdf`:

.. code-block:: python

    >>> t = numpy.linspace(-3, 3, 9)
    >>> distribution.pdf(t).round(3)
    array([0.009, 0.021, 0.043, 0.078, 0.121, 0.164, 0.193, 0.198, 0.176])

Similarly to create values from the *cumulative distribution function*
:func:`~chaospy.Distribution.cdf`:

.. code-block:: python

    >>> distribution.cdf(t).round(3)
    array([0.006, 0.017, 0.04 , 0.085, 0.159, 0.266, 0.401, 0.55 , 0.691])

To be able to perform any Monte Carlo method, each distribution contains
*random number generator* :func:`~chaospy.Distribution.sample`:

.. code-block:: python

    >>> distribution.sample(6).round(4)
    array([ 2.7901, -0.4006,  5.2952,  1.9107,  4.2763,  0.4033])

The sample scheme also has a few advanced options. For example, to create
low-discrepancy Halton sequences samples combined with antithetic variates:

.. code-block:: python

    >>> distribution.sample(size=6, rule="halton", antithetic=True).round(4)
    array([ 3.349 ,  0.651 , -0.3007,  4.3007,  2.6373,  1.3627])

For a full overview of these options, see :ref:`sampling`

In addition a function for extracting raw-statistical moments is available
through :func:`~chaospy.Distribution.mom`:

.. code-block:: python

    >>> distribution.mom([0, 1, 2, 3, 4])
    array([  1.,   2.,   8.,  32., 160.])

Note that these are raw moments, not the classical moments with adjustments.
For example, the variance is defined as follows:

.. code-block:: python

    >>> distribution.mom(2) - distribution.mom(1)**2
    4.0

However, if the adjusted moments are of interest, the can be retrieved using
the tools described in :ref:`descriptives`. For example for the variance, there
is :func:`chaospy.Var`:

.. code-block:: python

    >>> chaospy.Var(distribution)
    array(4.)

Joint distributions
-------------------

There are three ways to create a multivariate probability distribution in
``chaospy``: Using the joint constructor
:class:`chaospy.J`, the identical independent
distribution constructor: :class:`chaospy.Iid`,
and to one of the pre-constructed multivariate distribution defined in
:ref:`multivariate_distributions`.

Constructing a multivariate probability distribution can be done using the
:func:`~chaospy.J` constructor. E.g.:

.. code-block:: python

    >>> distribution = chaospy.J(chaospy.Normal(), chaospy.Uniform())
    >>> distribution
    J(Normal(mu=0, sigma=1), Uniform())

The created multivariate distribution behaves much like the univariate case:

.. code-block:: python

    >>> mesh = numpy.mgrid[0.25:0.75:3j, 0.25:0.75:3j]
    >>> mesh
    array([[[0.25, 0.25, 0.25],
            [0.5 , 0.5 , 0.5 ],
            [0.75, 0.75, 0.75]],
    <BLANKLINE>
           [[0.25, 0.5 , 0.75],
            [0.25, 0.5 , 0.75],
            [0.25, 0.5 , 0.75]]])
    >>> distribution.cdf(mesh).round(4)
    array([[0.1497, 0.2994, 0.449 ],
           [0.1729, 0.3457, 0.5186],
           [0.1933, 0.3867, 0.58  ]])
    >>> distribution.pdf(mesh).round(4)
    array([[0.3867, 0.3867, 0.3867],
           [0.3521, 0.3521, 0.3521],
           [0.3011, 0.3011, 0.3011]])
    >>> distribution.sample(6, rule="halton").round(4)
    array([[-1.1503,  0.3186, -0.3186,  1.1503, -1.5341,  0.1573],
           [ 0.4444,  0.7778,  0.2222,  0.5556,  0.8889,  0.037 ]])
    >>> distribution.mom([[2, 4, 6], [1, 2, 3]]).round(10)
    array([0.5 , 1.  , 3.75])

Random Seed
-----------

To be able to reproduce results it is possible to fix the random seed in
``chaospy``. For simplicity, The library respect :func:`numpy.random.seed`.
E.g.:

.. code-block:: python

    >>> numpy.random.seed(1234)
    >>> distribution = chaospy.Normal()
    >>> distribution.sample(5).round(4)
    array([-0.8723,  0.311 , -0.1567,  0.7904,  0.7721])
    >>> numpy.random.seed(1234)
    >>> distribution.sample(5).round(4)
    array([-0.8723,  0.311 , -0.1567,  0.7904,  0.7721])
    >>> distribution.sample(5).round(4)
    array([-0.605 , -0.5934,  0.8483,  1.7295,  1.1549])

Truncation
----------

Note that distributions for which there is no specific truncated variant,
can be truncated using the generic truncation feature, i.e.

.. code-block:: python

    >>> distribution = chaospy.Weibull(1)
    >>> upper_trunc = chaospy.Trunc(distribution, upper=2)
    >>> upper_trunc
    Trunc(Weibull(1), upper=2)
    >>> upper_and_lower_trunc = chaospy.Trunc(
    ...     distribution, lower=0.5, upper=2)
    >>> upper_and_lower_trunc
    Trunc(Weibull(1), lower=0.5, upper=2)

Copula
------

A cumulative distribution function of an independent multivariate random
variable can be made dependent through a copula as follows:

.. math::
    F_{Q_0,\dots,Q_{D-1}} (q_0,\dots,q_{D-1}) =
    C(F_{Q_0}(q_0), \dots, F_{Q_{D-1}}(q_{D-1}))

where :math:`C` is the copula function, and :math:`F_{Q_i}` are marginal
distribution functions.  One of the more popular classes of copulas is the
Archimedean copulas.
.. \cite{sklar_random_1996}.
They are defined as follows:

.. math::
    C(u_1,\dots,u_n) =
    \phi^{[-1]} (\phi(u_1)+\dots+\phi(u_n)),

where :math:`\phi` is a generator and :math:`\phi^{[-1]}` is its
pseudo-inverse. Support for Archimedean copulas in `chaospy` is possible
through reformulation of the Rosenblatt transformation.  In two dimension, this
reformulation is as follows:

.. math::

    F_{U_0}(u_0) = \frac{C(u_0,1)}{C(1,1)}

    F_{U_1\mid U_0}(u_1\mid u_0) =
    \frac{\tfrac{\partial}{\partial u_0}
    C(u_0,u_1)}{\tfrac{\partial}{\partial u_0} C(u_0,1)}

This definition can also be generalized in to multiple variables using the
formula provided by Nelsen 1999.
.. cite:: nelsen_introduction_1999

The definition of the Rosenblatt transform can require multiple
differentiations.  An analytical formulation is usually not feasible, so the
expressions are estimated using difference scheme similar to the one outlined
for probability density function defined in :ref:`distributions`. The accurate
might therefore be affected.

Since copulas are meant as a replacement for Rosenblatt transformation, it is
usually assumed that the distribution it is used on is stochastically
independent. However in the definition of a copula does not actually require
it, and sine the Rosenblatt transformation allows for it, multiple copulas can
be stacked together in `chaospy`.

User defined distributions
--------------------------

Constructing custom probability distributions is done by using the
distribution :class:`chaospy.UserDistribution`. Start by defining

.. code-block:: python

    >>> def cdf(x_loc, lo, up):
    ...     '''Cumulative distribution function.'''
    ...     return (x_loc-lo)/(up-lo)

    >>> def pdf(x_loc, lo, up):
    ...     '''Probability density function.'''
    ...     return 1./(up-lo)

    >>> def lower(lo, up):
    ...     '''Lower bounds function.'''
    ...     return lo

    >>> def upper(lo, up):
    ...     '''Upper bounds function.'''
    ...     return up

Custom distributions can be create with these functions as input:

.. code-block:: python

    >>> distribution = chaospy.UserDistribution(
    ...     cdf=cdf, pdf=pdf, lower=lower,
    ...     upper=upper, parameters=dict(lo=-1, up=1))

The distribution can then be used in the same was as any other
:class:`chaospy.Distribution`:

.. code-block:: python

    >>> distribution.fwd(numpy.linspace(-2, 2, 7)).round(4)
    array([0.    , 0.    , 0.1667, 0.5   , 0.8333, 1.    , 1.    ])
    >>> distribution.pdf(numpy.linspace(-2, 2, 7)).round(4)
    array([0. , 0. , 0.5, 0.5, 0.5, 0. , 0. ])
    >>> distribution.inv(numpy.linspace(0, 1, 7)).round(4)
    array([-1.    , -0.6667, -0.3333,  0.    ,  0.3333,  0.6667,  1.    ])
    >>> distribution.lower, distribution.upper
    (array([-1.]), array([1.]))

Here cumulative density function ``cdf`` is an absolute requirement. In
addition, either ``ppf``, or the couple ``lower`` and ``upper`` should be
provided. The others are not required, but may increase speed and or accuracy
of calculations. In addition to the once listed, it is also possible to define
the following methods:

``mom``
    Method for creating raw statistical moments, used by the
    :func:`~chaospy.Distribution.mom` method.
``ttr``
    Method for creating coefficients from three terms recurrence method, used to
    perform "analytical" Stiltjes' method.
