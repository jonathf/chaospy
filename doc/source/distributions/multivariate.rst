Constructing multivariate probability distributions can be done in one of three
ways. If the distribution consist stochastically independent components, the
function :func:`~chaospy.distributions.joint.Iid` can be used::

    >>> distribution = chaospy.Iid(chaospy.Normal(0, 1), 2)

There are some distributions that are designed to be multivariate, like the
multivariate log-normal distribution::

    >>> distribution = chaospy.MvLognormal(
    ...     loc=[0, 0], scale=[[1, 0.5], [0.5, 1]])

Lastly, for more control the constructor :func:`~chaospy.distributions.joint.J` can be
used::

    >>> distribution = chaospy.J(
    ...     chaospy.Normal(0, 1), chaospy.Uniform(0, 1))


In either case, the multivariate distribution behaves much like the univariate
case::

    >>> mesh = numpy.meshgrid(
    ...     numpy.linspace(0.25, 0.75, 3),
    ...     numpy.linspace(0.25, 0.75, 3),
    ... )
    >>> print(numpy.around(distribution.cdf(mesh), 4))
    [[0.1497 0.1729 0.1933]
     [0.2994 0.3457 0.3867]
     [0.449  0.5186 0.58  ]]
    >>> print(numpy.around(distribution.pdf(mesh), 4))
    [[0.3867 0.3521 0.3011]
     [0.3867 0.3521 0.3011]
     [0.3867 0.3521 0.3011]]
    >>> print(numpy.around(distribution.sample(
    ...     size=6, rule="H", antithetic=True), 4))
    [[-1.1503  1.1503 -1.1503  1.1503  0.3186 -0.3186]
     [ 0.4444  0.4444  0.5556  0.5556  0.7778  0.7778]]
    >>> print(distribution.mom([[2, 4, 6], [1, 2, 3]]))
    [0.5  1.   3.75]

.. autofunction:: chaospy.distributions.joint.J
.. autofunction:: chaospy.distributions.joint.Iid

.. _dependent:

Stocastic Dependent
-------------------

One of ``chaospy``'s most powerful features is possible to construct advance
multivariate variables directly through dependencies. To illustrate this,
consider the following bivariate distribution::

    >>> dist_ind = chaospy.Gamma(1)
    >>> dist_dep = chaospy.Normal(dist_ind**2, dist_ind+1)
    >>> distribution = chaospy.J(dist_ind, dist_dep)

In other words, a distribution dependent upon another distribution was created
simply by inserting it into the constructor of another distribution. The
resulting bivariate distribution if fully valid with dependent components.
For example the probability density function functions will in this case look
as follows::

    >>> x, y = numpy.meshgrid(numpy.linspace(-4, 7), numpy.linspace(0, 3))
    >>> likelihood = distribution.pdf([x, y])

This method also allows for construct any multivariate probability distribution
as long as you can fully construct the distribution's conditional decomposition
as noted in :ref:`rosenblatt`. One only has to construct each univariate
probability distribution and add dependencies in through the parameter
structure.

Now it is worth noting a couple of caveats:

* Since the underlying feature to accomplish this is the :ref:`rosenblatt`, the
  number of unique random variables in the final joint distribution has to be
  constant. In other words, ``dist_dep`` is not a valid distribution in itself,
  since it is univariate, but depends on the results of ``dist_ind``.
* The dependency order does not matter as long as it can defined as an acyclic
  graph. In other words, ``dist_ind`` can not be dependent upon ``dist_dep`` at
  the same time as ``dist_dep`` is dependent upon ``dist_ind``.

.. figure:: ./multivariate.png

   Example of a custom construction with non-trivial dependent random
   variable.

