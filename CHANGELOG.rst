Master Branch
=============

Version 3.3.7 (2020-08-09)
==========================

CHANGED:
  * `numpoly` update to version 1.0.6.

Version 3.3.6 (2020-07-23)
==========================

ADDED:
  * Support for quadrature dispatching for `Mul`, `Add` and (independent) `J`.

CHANGED:
  * Refactor approximate_moment:
    * Remove unused antithetic variate. (Really never used.)
    * Remove redundant support for multiple exponents at once. In practice only
      one is called at the time anyway.
    * Adding buffering for both quadrature and results, so to reduce needed
      computation for recursive methods.
    * New order default: 1000 -> 1000/log2(len(dist)+1)
      About the same for lower dimensions, but scales better with higher
      dimensions.
  * Update requirements to include numpy.

Version 3.3.5 (2020-07-13)
==========================

CHANGED:
  * Refactor discrete distribution:
    * Allowing "offset" (up to 0.5 on each side), making all discrete
      distributions piece-wise constants.
    * Use linear interpolation in `dist.fwd` and `dist.inv` between the edges,
      making them piece linear function.
    * `dist.cdf` adjusted 0.5 to the right to replicate old behavior.
    * Update the two implemented discrete distributions `DiscreteUniform` and
     `Binomial`.

Version 3.3.4 (2020-07-09)
==========================

ADDED:
  * Added changelog (the file you currently are reading).
  * Support for polynomial saving to and loading from disk.

CHANGED:
  * Refactor descrete distributions to work better with quadrature.
  * `numpoly` update to version 1.0.5.

FIXES:
  * Bugfix: Poly-division with large relative error caused infinity-loops.

Version 3.3.3 (2020-06-29)
==========================

CHANGED:
  * Move `chaospy/tutorial -> chaospy/docs/tutorials`.
  * Use nbsphinx to integrate notebooks straight into the RTD docs.
  * Renamed `chaospy/{doc -> docs}`.
  * Include numpoly documentation content directly in toctree.
  * `numpoly` update version 1.0.3
  * Chaospy logger now capture Numpoly as well.
  * Aligning Numpoly properly, making a wrapper redundant.

REMOVED:
  * Announcing deprecation of `chaospy.basis` and `chaospy.prange` in favor
    of `chaospy.monomial`.
  * Deprecating `chaospy.setdim` in favor for `numpoly.set_dimensions`.

Version 3.3.2 (2020-06-16)
==========================

ADDED:
  * Add Joe-copula back into the fold.
  * Add `chaospy.example` to simplify the Jupyter notebook tutorials creation.

CHANGED:
  * Remove CircleCI `build-cache` system in favor of simpler linear builds.
    * Reduce checks to 2.7 and 3.8. Anything between is assumed to be covered
      by the two.
  * Clean up sensitivity analysis tools.
  * Clean up copula docs.
  * Move lots of doc examples from .rst to .ipynb.

REMOVED:
  * Remove `Sens_*_nataf` as they were a one-shot project for a paper and no
    longer work.
  * Deprecate old Archemedean base copula

Version 3.3.1 (2020-06-09)
==========================

CHANGED:
  * Switch `numpoly.bindex` with new `numpoly.glexindex`.

REMOVED:
  * Removing unused Bertran functions.
  * Deprecating old Distribution names (which have been announce for over a year
    through warning messages)

Version 3.3.0 (2020-06-08)
==========================

ADDED:
  * Added `chaospy.orthogonal.frontend:generate_expansion` as an one stop
    expansion generation function.
  * Add tag-check when deploying using tags.
  * Add logging which activates on env `CHAOSPY_DEBUG=1`.
    Log to file with env `CHAOSPY_LOGFILE=/path/to/file`
  * Added *Program Evaluation and Review Technique* (PERT) distribution.
  * Adding support for `Dist.__matmul__`
    (which obviously does nothing in python 2).
  * Adding tests to the *hard-to-get-right* sub-module:
    `chaospy.distributions.operators`.
  * Added LRU cache to some quadrature schemes.
  * Added segments to Newton-Cotes, Fejer and Clenshaw-Curtis
    (as this is recommended to have to discretized Stieltjes).
  * Added experimental Jupyter notebooks with user tutorials/recipes
    `GITROOT/tutorial`
  * Gumbel and Clayton copulas get analytical recursive Rosenblatt
    transformations.

CHANGED:
  * Update `numpoly` to version 0.3.0.
    * Replace explicit numpoly import, with an implicit one with a
      smart-wrapper.
    * Docs updated with new polynomial string representation order.
  * Update to documentation.
  * Replace sample and quadrature scheme name from one letter
    ["G", "E", "C", "H", ...], to new full name strings:
    ["gaussian", "legendre", "clenshaw-curtis", "halton", ...].
    (Old style still works, but is undocumented.)
  * Increase quadrature sample rate 100->200 when doing discretized Stieltjes
    to increase accuracy (at the computational cost).
  * Increased sample rate for approximate inverse (used when inverse is
    missing), increasing accuracy at extra computational cost.
  * New style Archemedean copula.
  * Refactor `chaospy.distributions.operators` to become less messy.
  * Some adjustment to the expansion functions to align with the new frontend.
  * Update lagrange to use `numpoly.bindex` in the backend.
  * Use `graded: bool` and `reverse: bool` as a replacement for `sort: str =
    "GRI"`:
    * The `"I"` in `"GRI"` is deprecated: It can always be achieved with
      `values = values[::-1]`, so it serves little purpose.
    * The `"R"` was implemented backwards. `R` present is equivalent with
      `reverse=False`.
    * `sort` still works, but raises an warning about future deprecation.
    * Using one letter strings is less readable, and needs to be removed.
      Splitting them up, simplifies documentation.

REMOVED:
  * Deprecating copulas Frank, Joe and Ali-Mikhail-Haw, as their accuracy is
    not good enough.
  * Remove really old tutorial stuff not longer in use.

Version 3.2.1 (2020-02-11)
==========================

FIXES:
  * Bugfix for `evaluate_lower` and `evaluate_upper` for operators like
    addition, multiply, power, etc.
  * Fix to `interpret_as_integer` of joint distribution
    (now covering mixed content).

Version 3.2.0 (2020-02-10)
==========================

ADDED:
  * Added `chaospy.__version__`

CHANGED:
  * Upper and lower methods:
    * Replace `Dist.bnd` with `Dist.lower` and `Dist.upper` to have better
      control.
    * Issue future deprecation warning if `Dist._bnd` is used.
    * Deprecate `chaospy.distributions.approximation:find_interior_point` as
      its use falls away with the new methods.
    * Add new `chaospy.distributions.evauation.bound:evaluate_lower` and
      `evaluate_upper`
  * Fix to `interpret_as_integer` of joint distribution with discrete
    components.

REMOVED:
  * Deprecated trigonometric distribution transformations, as the were hard to
    transfer over, undocumented and likely not used.

Version 3.1.1 (2020-01-10)
==========================

CHANGED:
  * `numpoly` version 0.1.6.

Version 3.1.0 (2019-12-29)
==========================

CHANGED:
  * `numpoly` introduced, version 0.1.4:
    * Replacing backend for polynomial handle with `numpoly`, leaving just a
      compatibility wrapper.
    * Refactor descriptive to utilize new backend
    * Update all docstring containing a polynomial as the string representation
      has changed.
    * Declare `chaospy.Poly` as soon-to-be deprecated
  * Replace setuptools+pipenv for installation and development management to
    poetry for both
  * Introduce CircleCI build-cache step.
  * Distribution update:
    * Added `Dist.interpret_as_integer` to better support discrete
      distributions.
    * Update lots of method docs in `chaospy.distributions.collection` to look
      better.

Version 3.0.9 (2019-08-25)
==========================

CHANGED:
  * Making a logger.warning into logger.info (as requested by user).

Version 3.0.8 (2019-08-25)
==========================

ADDED:
  * Added support for `openturns` Distributions (thanks Régis Lebrun).
  * Added "Related Projects" section to root README with thanks and shout-outs.
  * Added discrete distributions: Binomial, DiscreteUniform.
  * Added recipe for stochastic dependent distributions:
    `doc/recipes/dependent.rst`

CHANGED:
  * Moved external interfaces to new submodule: `chaospy.external`:
    SampleDist (KDE), OTDistribution (OpenTURNS), scipy_stats.
  * Update Chaospy logo.

Version 3.0.7 (2019-08-11)
==========================

CHANGED:
  * Replace `chaospy.bertran.operators.bertran_indices` with
    `chaospy.bertran.bindex`:
      * Faster execution by using more `numpy` for heavy lifting
  * Moved `chaospy.{quad -> quadrature}` to finalize the refactor from v3.0.6.
  * Documentation polish to `chaospy.quadrature`.

FIXES:
  * Bugfixes in handling of three-terms-recursion

REMOVED:
  * Remove `chaospy.quad.collection.probabilitic` as it is much easier to
    implement from the user side.

Version 3.0.6 (2019-07-26)
==========================

ADDED:
  * Added license to setup.py
  * New quadrature rules (thanks to Nico Schlömer):
    Gauss-Lobatto, Gauss-Kronrod, Gauss-Radau, Newton-Cotes.

CHANGED:
  * Update CircleCI to test for Python versions 2.7.16, 3.6.8 and 3.7.3
  * Update dependencies.
  * Refactored `chaospy.quadrature`:
    * Standardize quadrature interface.
    * Lots of new docs.
  * Move version number `chaospy.{version -> __init__}`.

REMOVED:
  * Deprecating `chaospy.distributions.collection.raised_cosine` as `hyp1f2` is
    no longer supported by `scipy`.
  * Removing local `set_state` for Sobol indices and instead rely on
    `numpy.random`'s random seed.

Version 3.0.5 (2019-06-17)
==========================

ADDED:
  * Added new method `Dist._range` to override the lower and upper bound
    calculations on some distributions.
  * Added readme to setup.py

CHANGED:
  * Adding caching to some of the functionality in `chaspy.bertran`
  * Use new cached functions to improve on raw statistical moments of
    multivariate Gaussian and multivariate Student-T distributions.
  * Update polynomial output, as update to Bertran changes a few things in str
    handle.

Version 3.0.4 (2019-02-20)
==========================

ADDED:
  * Adding `chaospy.distributions.evauation` submodule to deal with graph
    resolution.
  * Added CircleCI tests for Python 2.7.15

CHANGED:
  * Update CircleCI Python {3.6.2 -> 3.7.1}
  * Some adjustments added to support Python 2.

REMOVED:
  * Remove dependency to `networkx` (as `evaluation` now does this task).
  * Deprecating `chaospy.distributions.cores` (as each distribution are now
    locally defined in `chaospy.distributions.collection`)

Version 3.0.3 (2019-02-10)
==========================

FIXES:
  * Fixes to CircleCI testing.

Version 3.0.2 (2019-02-09)
==========================

ADDED:
  * Sparse segmentation function `chaospy.bertran.sparse:sparse_segment`

CHANGED:
  * Move install source {ROOT/src/chaospy -> ROOT/chaospy}
  * Documentation update (mostly `chaospy.orthogonal`).

REMOVED:
  * Deprecated `cubature` module; Does not work with the chaospy v3, and is hard
    to maintain.

Version 3.0.1 (2019-01-28)
==========================

CHANGED:
  * Update install dependencies to newest version
  * Refactor documentation
    * Update Sphinx configuration to newest version
    * Restructured the documentation a bit to make more sense with the new
      code.
    * Added some extra docs here and there.

Version 3.0.0 (2019-01-16)
==========================

ADDED:
  * Added Fejer quadrature

CHANGED:
  * Full refactor of the `chaospy.dist` submodule:
    * Move: `chaospy.dist -> chaospy.distributions`
    * Deprecate `chaospy.distributions.graph` in favor of new
      `chaospy.distributions.evaluation` which will not depend on `networkx`
      and should be easier to maintain.
    * Move distributions from the two files `distributions.{cores,collection}`
      to the folder `distributions.collection`, where each file now is one core
      and one (or more) wrapper(s).
    * Rename some old distributions; Kept the old ones for now, but they issue
      deprecation warnings.
    * Split `distributions.copulas.collection` into individual components.
    * Tests distribution using black-list instead of current white-list system.
    * Rewritten a lot of documentation.
  * Replace absolute import paths with relative ones.
  * Refactor `chaospy.descriptives` to look better docs and code wise.
  * Adapt to Python 2+3 support.
  * Turn on automatic logging for warnings and upwards
