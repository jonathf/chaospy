.. _tutorial:

Tutorials
=========

Here is a collection of `Jupyter notebooks <https://jupyter.org/>`_
that explore different ways to use ``chaospy``.

If you prefer to test the tutorials without installing any code locally, it is
possible to `run the tutorials using Binder
<https://mybinder.org/v2/gh/jonathf/chaospy/master?filepath=docs%2Ftutorials>`_.

.. toctree::
   :hidden:

   example_introduction
   monte_carlo_integration
   point_collocation
   pseudo_spectral_projection
   scikitlearn_regression
   intrusive_galerkin
   kernel_density_estimation
   gaussian_mixture_model
   wiener_askey_scheme
   truncation_scheme
   lagrange_polynomials
   gstools_kriging
   seir_model_pce

`Example introduction <./example_introduction.ipynb>`_
------------------------------------------------------

The notebooks are all self contained, however many of the examples
addressed is repeated a lot. In this introduction, the common example is
explored in a little more detail, whereas in the other notebooks, the
example is skimmed quickly.

`Monte Carlo integration <./monte_carlo_integration.ipynb>`_
------------------------------------------------------------

Monte Carlo integration is the tried and true method for doing
uncertainty quantification. However, it is in many instances possible to
speed up the convergence rate of the integration, by replacing
traditional (pseudo-)random samples, with more optimized
`low-discrepancy
sequences <https://en.wikipedia.org/wiki/Low-discrepancy_sequence>`_.

`Point collocation <./point_collocation.ipynb>`_
------------------------------------------------

Point collocation method is one of the two classical non-intrusive
polynomial chaos expansion methods. Relies on statistical regression to
estimate the Fourier coefficients needed to create the model
approximation.

`Pseudo-spectral projection <./pseudo_spectral_projection.ipynb>`_
------------------------------------------------------------------

Pseudo-Spectral is the second of the two classical non-intrusive
polynomial chaos expansion methods. Relies on quadrature (and then
typically Gaussian quadrature) integration to estimate the Fourier
coefficients needed to create the model approximation.

`Scikit-learn regression <./scikitlearn_regression.ipynb>`_
-----------------------------------------------------------

The library ``scikit-learn`` is a great machine-learning toolkit that
provides a large collection of regression methods. By default,
``chaospy`` only support traditional least-square regression when doing
`Point Collocation <./point_collocation.ipynb>`_, but ``chaospy`` is
designed to also work together with the various regression functions
provided by the ``scikit-learn`` interface.

`Intrusive Galerkin <./intrusive_galerkin.ipynb>`_
--------------------------------------------------

Intrusive Galerkin method is the classical approach for applying
polynomial chaos expansion on a set of governing equations. An intrusive
method, unlike the non-intrusive method described in `Point
Collocation <./point_collocation.ipynb>`_ and `Pseudo-Spectral
Projection <./pseudo_spectral_projection.ipynb>`_. It requires a bit
more use of the mathematical hand holding to apply properly.

`Kernel density estimation <./kernel_density_estimation.ipynb>`_
----------------------------------------------------------------

Kernel density estimation is a way to estimate the probability density
function of a random variable in a non-parametric way. It works for both
uni-variate and multi-variate data. It includes automatic bandwidth
determination.

`Gaussian mixture model <./gaussian_mixture_model.ipynb>`_
----------------------------------------------------------

A Gaussian mixture model is a probabilistic model constructed from a
mixture of a finite number of Gaussian distributions.

`Wiener-Askey scheme <./wiener_askey_scheme.ipynb>`_
----------------------------------------------------

Classical theory defines the original orthogonal polynomial expansions
back to the Wiener-Askey polynomials. These polynomials have specific
form and have value from a theoretical analysis point of view.

`Truncation scheme <./truncation_scheme.ipynb>`_
------------------------------------------------

Polynomial chaos expansion is classically truncated at fixed polynomial
orders. However, it is possible to refine the truncation rule to better
suite the problem that is being solved using dimension prioritization
and :math:`L_p`-norm truncation rules.

`Lagrange polynomials <./lagrange_polynomials.ipynb>`_
------------------------------------------------------

Lagrange polynomials are also used in Uncertainty quantification as an
alternative strategy to polynomial chaos expansions.

`Polynomial chaos kriging with gstools <./gstools_kriging.ipynb>`_
------------------------------------------------------------------

Use a combination of ``chaospy``, ``scikit-learn`` and ``gstools`` to
create a polynomial chaos kriging model.

`SEIR: Coupled differential equations  <./seir_model_pce.ipnb>`_
----------------------------------------------------------------

A complete epidemological example modelling uncertainty in a set coupled
differential equation, with the use of ``scipy.integration`` and
``multiprocessing``.
