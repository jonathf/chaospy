Tasks that needs solving
========================

Some undone task that could be be useful to add. Any help is welcome.

Please create a Github issue if you want to claim a task.

Quadrature in unscaled space
----------------------------

Difficulty: easy

Quadrature works best when done on the unscaled random variables. In other
words, instead of making quadrature directly on e.g.
``chaospy.Uniform(1000, 1001)``, create the quadrature on the variable
``chaospy.Uniform(-1, 1)`` and scale the samples to the correct interval
afterwards. This is a much more accurate approach than what is currently
implemented.

In practice there are only three edgecases that needs to be dealt with:
``chaospy.Add``, ``chaospy.Mul`` and ``chaospy.J``. All three can be
intercepted in the ``chaospy.quadrature.frontend:generate_quadrature`` with
``isinstance(dist, chaospy.{Add,Mul,J})``.

Scaling of non-monic polynomials
--------------------------------

Difficulty: easy

Canonical orthogonal polynomial expansion from literature usually come in two
variants: Physisist and Probabilist variant. The difference between the two is
usually only a constant. See e.g.
`https://en.wikipedia.org/wiki/Hermite_polynomials`_.
In ``chaospy`` the latter is supported.

However, there isn't a good reason not to support both. To do so, one could
e.g. add an extra method to each distribution that returns the constant
multiplier at each order. This could by default have the form::

   @staticmethod
   def physisists_constant(order):
      return 1

Or in the case of the Normal distribution (aka Hermite)::

   @staticmethod
   def physisists_constant(order):
      return 2**order

This function can then be read by functions in ``chaospy.orthogonal`` if an
extra argument is provided by the user asking for it.

Optimization of Moment Approximations
-------------------------------------

Difficulty: easy

Currently if a distribution don't implement moment calculations,
in ``chaospy.Distribution._get_mom``, a switch to numerical
integration using the probability density function. If density is missing as
well (which is the case for all Copulas), this estimation will be slow.
In practice, calculating moments usually don't come alone (like ``orth_ttr``,
``orth_chol``, etc.), making the whole procedure computationally prohibitively
expensive.

This can be optimized quite a bit, as the density evaluations are the same for
every moments. So as a simple solution is to let the approximation function
to:
* Cache the moments evaluations with some form `lru_cache` for the `._mom`
  method.
* Store density evaluations related to moment approximations to the density,
  which can be reused for every moment estimation.

The approximation function can be found at
``chaospy.distributions.approximation``.

Cookbook example: Data driven using SampleDist
----------------------------------------------

Difficulty: medium

There is support for converting data into a sample distribution. It would be
nice to display how a good data-driven analysis can be performed.

Cookbook example: Iterative quadrature using Leja
-------------------------------------------------

Difficulty: medium

Leja is completely nested. A example that shows how to do analysis iterative
without re-evaluating the model function would be nice.

Cookbook example: Iterative testing using Kronrod
-------------------------------------------------

Difficulty: medium

Kronrod and Gaussian quadrature is linked in the sense that at each quadrature
order, all Gaussian nodes are nested in the Kronrod samples. A common strategy
to evaluate that polynomial chaos expansion has converged, is to repeat the
analysis done with optimal Gaussian, also holds true when expanded to the
Kronrod example.

Discrete distributions
----------------------

Difficulty: easy

Currently two discrete distributions are supported: ``chaospy.DescreteUniform``
and ``chaospy.Binomial``. Use these as templates and the literature to extend
``chaospy.distribution.collection`` with the thee distributions:
Hyper-geometric, Negative-Binomial and Poisson. If other distributions also
makes sense, add as one sees fit.

Better Lagrange Polynomial Support
----------------------------------

Difficulty: medium

Current Lagrange polynomial implementation is rudimentary and can be improved
upon quite a bit.

This does not require re-inventing the wheel, as there are others who have
solve it before. For examples, it should be possible to get inspired/copy from:
`https://people.sc.fsu.edu/~jburkardt/m_src/lagrange_nd/lagrange_nd.html`_
`https://sandialabs.github.io/pyapprox/tensor_product_lagrange_interpolation.html`_
