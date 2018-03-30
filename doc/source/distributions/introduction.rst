
One of the backbone of any uncertainty quantification is a collection of
probability distributions, and ``chaospy`` is no exception. For example, to
create a Gaussian random variable::

    >>> distribution = chaospy.Normal(mu=2, sigma=2)

The syntax for using distribution is here very similar to the syntax used in
``scipy.dist``. For example, to create values from the *probability density
function*::

    >>> t = numpy.linspace(-3, 3, 9)
    >>> print(numpy.around(distribution.pdf(t), 4))
    [0.0088 0.0209 0.0431 0.0775 0.121  0.1641 0.1933 0.1979 0.176 ]

Similarly to create values from the *cumulative distribution function*::

    >>> print(numpy.around(distribution.cdf(t), 4))
    [0.0062 0.0168 0.0401 0.0846 0.1587 0.266  0.4013 0.5497 0.6915]

To be able to perform any Monte Carlo method, each distribution contains
*random number generator*::

    >>> print(numpy.around(distribution.sample(6), 4))
    [ 2.7901 -0.4006  5.2952  1.9107  4.2763  0.4033]

The sample scheme also has a few advanced options. For example, to create
low-discrepancy Hammersley sequences samples combined with antithetic variates::

    >>> print(numpy.around(distribution.sample(
    ...     size=6, rule="H", antithetic=True), 4))
    [ 3.349   0.651  -0.3007  4.3007  2.6373  1.3627]

For a full overview of these options, see :ref:`samples`

In addition a function for extracting raw-statistical moments is available::

    >>> print(distribution.mom([0, 1, 2, 3, 4]))
    [  1.   2.   8.  32. 160.]

Note that these are raw moments, not the classical moments with adjustments.
For example, the variance is defined as follows::

    >>> print(distribution.mom(2) - distribution.mom(1)**2)
    4.0

However, if the adjusted moments are of interest, the can be retrieved using
the tools described in :ref:`descriptives`::

    >>> print(chaospy.Var(distribution))
    4.0

.. autoclass:: chaospy.distributions.baseclass.Dist
    :members: pdf, cdf, sample, mom, fwd, inv
