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
variants: physisist's and probabilist's variant. The difference between the two
is usually only a constant. See e.g.
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

Splitting quadrature intervals
------------------------------

Difficulty: hard

Quadrature rules works better if it is possible to split the distribution into
chunks and doing individual evaluation on each chunk. In other words, instead
of 100 nodes over one distribution, split the distribution into intervals and
do 10 nodes for each interval. Each sub-quadrature rule is then combined into
a single quadrature rule at the end.

The interface ``chaospy.quadrature.frontend:generate_quadrature`` should be
updated to include interval splitting.

Why this can be a bit tricky:

* Some distributions lend themselves to optimal interval points. A optional
  method could be added to each distribution defining their own interval
  splits.
* This problem should preferably be extended to the multivariate case, where
  intervals are replaced with segments in multivariate space.
* If a quadrature rule includes samples at the ends, sub-rules can overlap.
  For those samples, only one should be kept, and the weight function should be
  updated to include the sum of both.

Descrete distributions
----------------------

Difficulty: easy

Currently two discrete distributions are supported: ``chaospy.DescreteUniform``
and ``chaospy.Binomial``. Use these as templates and the literature to extend
``chaospy.distribution.collection`` with the thee distributions:
Hyper-geometric, Negative-Binomial and Poisson. If other distributions also
makes sense, add as one sees fit.

Replacing internal polynomial module with ``numpoly``
------------------------------------------------------

Difficulty: easy

`Numpoly <https://github.com/jonathf/numpoly>`_ is a new library for handling
polynomials. Remove all instances of ``poly`` and ``Poly`` in ``chaospy`` and
replace them with ``numpoly`` counterparts. Not too complicated, but might
require a bit of tinkering.

Along the way, add ``numpoly`` to the dependencies, and remove ``chaospy.poly``
and its corresponding documentation.
