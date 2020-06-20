.. _tutorial:

Tutorials
=========

.. toctree::
   :hidden:

   example_introduction
   introduction/monte_carlo_integration
   introduction/point_collocation
   introduction/pseudo_spectral_projection
   advanced/scikitlearn_regression
   advanced/intrusive_galerkin
   polynomial/expansion_construction
   polynomial/polynomial_evaluation
   polynomial/wiener_askey_scheme
   polynomial/truncation_scheme
   polynomial/lagrange_polynomials

Here is a collection of `Jupyter notebooks <https://jupyter.org/>`_ that
explore different ways to use ``chaospy``.

Precursor
---------

`Example Introduction`_
   The notebooks are all self contained, however many of the examples addressed
   is repeated a lot. In this introduction, the common example is explored in a
   little more detail, whereas in the other notebooks, the example is skimmed
   quickly.

.. _Example Introduction: ./example_introduction.ipynb

Introductory Topics
-------------------

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

.. _Monte Carlo Integration: ./introduction/monte_carlo_integration.ipynb
.. _Point Collocation: ./introduction/point_collocation.ipynb
.. _Pseudo-Spectral Projection: ./introduction/pseudo_spectral_projection.ipynb

Advanced Topics
---------------

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

.. _Scikit-Learn Regression: ./advanced/scikitlearn_regression.ipynb
.. _Intrusive Galerkin: ./advanced/intrusive_galerkin.ipynb

Polynomial Behavior
-------------------

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

.. _Expansion Construction: ./polynomial/expansion_construction.ipynb
.. _Polynomial Evaluation: ./polynomial/polynomial_evaluation.ipynb
.. _Wiener-Askey Scheme: ./polynomial/wiener_askey_scheme.ipynb
.. _Truncation Scheme: ./polynomial/truncation.ipynb
.. _Lagrange Polynomials: ./polynomial/lagrange.ipynb
