.. _dependent:

Stochastic Dependencies
-----------------------

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
using Rosenblatt transformations. One only has to construct each univariate
probability distribution and add dependencies in through the parameter
structure.

Now it is worth noting a couple of caveats:

* The underlying feature to accomplish this is using the Rosenblatt
  transformation. Because of this the number of unique random variables in the
  final joint distribution has to be constant. In other words, ``dist_dep`` is
  not a valid distribution in itself, since it is univariate, but depends on
  the results of ``dist_ind``.
* The dependency order does not matter as long as it can defined as an acyclic
  graph. In other words, ``dist_ind`` can not be dependent upon ``dist_dep`` at
  the same time as ``dist_dep`` is dependent upon ``dist_ind``.

.. figure:: ./multivariate.png

   Example of a custom construction with non-trivial dependent random
   variable.
