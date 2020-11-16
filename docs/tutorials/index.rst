.. _tutorial:

Tutorials
=========

Here is a collection of `Jupyter notebooks <https://jupyter.org/>`_ that
explore different ways to use ``chaospy``.

`Example Introduction`_
   The notebooks are all self contained, however many of the examples addressed
   is repeated a lot. In this introduction, the common example is explored in a
   little more detail, whereas in the other notebooks, the example is skimmed
   quickly.
`Monte Carlo Integration`_
   Monte Carlo integration is the tried and true method for doing uncertainty
   quantification. However, it is in many instances possible to speed up the
   convergence rate of the integration, by replacing traditional
   (pseudo-)random samples, with more optimized `low-discrepancy sequences
   <https://en.wikipedia.org/wiki/Low-discrepancy_sequence>`_.
`Point Collocation`_
   Point collocation method is one of the two classical non-intrusive
   polynomial chaos expansion methods. Relies on statistical regression to
   estimate the Fourier coefficients needed to create the model approximation.
`Pseudo-Spectral Projection`_
   Psuedo-Spectral is the second of the two classical non-intrusive polynomial
   chaos expansion methods. Relies on quadrature (and then typically Gaussian
   quadrature) integration to estimate the Fourier coefficients needed to
   create the model approximation.
`Scikit-Learn Regression`_
   The library `scikit-learn` is a great machine-learning toolkit that provides
   a large collection of regression methods. By default, ``chaospy`` only
   support traditional least-square regression when doing `Point Collocation`_,
   but ``chaospy`` is designed to also work together with the various
   regression functions provided by the `scikit-learn` interface.
`Intrusive Galerkin`_
   Intrusive Galerkin method is the classical approach for applying polynomial
   chaos expansion on a set of governing equations. An intrusive method, unlike
   the non-intrusive method described in `Point Collocation`_ and
   `Pseudo-Spectral Projection`_. It requires a bit more use of the mathematical
   hand holding to apply properly.
`Kernel Density Estimation`_
   Kernel density estimation is a way to estimate the probability density
   function of a random variable in a non-parametric way. It works for both
   uni-variate and multi-variate data. It includes automatic bandwidth
   determination.
`Gaussian Mixture Model`_
   A Gaussian mixture model is a probabilistic model constructed from a mixture
   of a finite number of Gaussian distributions.
`Expansion Construction`_
   An overview over how to construct both polynomials and polynomial
   expansions.
`Polynomial Evaluation`_
   Polynomials are functions, and functions can be evaluated. In the case of
   the polynomials in ``chaospy``, the polynomial have support for vectorized
   evaluations, and partial evaluations and using ``numpy`` compatibility
   wrapper, to mention a few of the things that can be done.
`Wiener-Askey Scheme`_
   Classical theory defines the original orthogonal polynomial expansions back
   to the Askey-polynomials. These polynomials have specific form and have
   value from a theoretical analysis point of view.
`Truncation Scheme`_
   Polynomial chaos expansion is classically truncated at fixed polynomial
   orders. However, it is possible to refine the truncation rule to better
   suite the problem that is being solved using dimension prioritization and
   :math:`L_p`-norm truncation rules.
`Lagrange Polynomials`_
   Lagrange polynomials are also used in Uncertainty quantification as an
   alternative strategy to polynomial chaos expansions.

.. _Example Introduction: ./01_example_introduction.ipynb
.. _Monte Carlo Integration: ./02_monte_carlo_integration.ipynb
.. _Point Collocation: ./03_point_collocation.ipynb
.. _Pseudo-Spectral Projection: ./04_pseudo_spectral_projection.ipynb
.. _Scikit-Learn Regression: ./05_scikitlearn_regression.ipynb
.. _Intrusive Galerkin: ./06_intrusive_galerkin.ipynb
.. _Kernel Density Estimation: ./07_kernel_density_estimation.ipynb
.. _Gaussian Mixture Model: ./08_gaussian_mixture_model.ipynb
.. _Expansion Construction: ./09_expansion_construction.ipynb
.. _Polynomial Evaluation: ./10_polynomial_evaluation.ipynb
.. _Wiener-Askey Scheme: ./11_wiener_askey_scheme.ipynb
.. _Truncation Scheme: ./12_truncation_scheme.ipynb
.. _Lagrange Polynomials: ./13_lagrange_polynomials.ipynb

.. toctree::
   :hidden:

   01_example_introduction
   02_monte_carlo_integration
   03_point_collocation
   04_pseudo_spectral_projection
   05_scikitlearn_regression
   06_intrusive_galerkin
   07_kernel_density_estimation
   08_gaussian_mixture_model
   09_expansion_construction
   10_polynomial_evaluation
   11_wiener_askey_scheme
   12_truncation_scheme
   13_lagrange_polynomials
