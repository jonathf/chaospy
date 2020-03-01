Tutorial
========

Here is a collection of `Jupyter notebooks <https://jupyter.org/>`_ that
explore different ways to use ``chaospy``.

Precursor
---------

`Example Introduction <./example_introduction.ipynb>`_
   The notebooks are all self contained, however many of the examples addressed
   is repeated a lot. In this introduction, the common example is explored in a
   little more detail, whereas in the other notebooks, the example is skimmed
   quickly.

.. _Example Introduction: ./example_introduction.ipynb

Core Topics
-----------

`Low-discrepancy Monte Carlo`_
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

.. _Low-discrepancy Monte Carlo: ./low_discrepancy_monte_carlo.ipynb
.. _Point Collocation: ./point_collocation.ipynb
.. _Pseudo-Spectral Projection: ./pseudo_spectral_projection.ipynb

Advanced Topics
---------------

`Intrusive Galerkin`_
   Intrusive Galerkin method is the classical approach for applying polynomial
   chaos expansion on a set of governing equations. An intrusive method, unlike
   the non-intrusive method described in `Point Collocation`_ and
   `Pseudo-Spectral Projection`_. It requires a bit more use of the mathematical
   hand holding to apply properly.

.. _Intrusive Galerkin: ./intrusive_galerkin.ipynb

Polynomials and Polynomial Expansions
-------------------------------------

`Constructing Polynomial Expansions`_
   An overview over how to construct both polynomials and polynomial
   expansions.
`Polynomial Evaluation`_
   Polynomials are functions, and functions can be evaluated. In the case of
   the polynomials in ``chaospy``, the polynomial have support for vectorized
   evaluations, and partial evaluations and using ``numpy`` compatibility
   wrapper, to mention a few of the things that can be done.
`Lagrange Polynomials`_
   Lagrange polynomials are also used in Uncertainty quantification as an
   alternative strategy to polynomial chaos expansions.
`Truncation of Polynomial Expansion`_
   Polynomial chaos expansion is classically truncated at fixed polynomial
   orders. However, it is possible to refine the truncation rule to better
   suite the problem that is being solved using dimension prioritization and
   :math:`L_p`-norm truncation rules.
`The Wiener-Askey Polynomial Expansions`_
   Classical theory defines the original orthogonal polynomial expansions back
   to the Askey-polynomials. These polynomials have specific form and have
   value from a theoretical analysis point of view.

.. _Constructing Polynomial Expansions: ./polynomial/expansion_construction.ipynb
.. _Polynomial Evaluation: ./polynomial/evaluation.ipynb
.. _Lagrange Polynomials: ./polynomial/lagrange.ipynb
.. _Truncation of Polynomial Expansion: ./polynomial/truncation.ipynb
.. _The Wiener-Askey Polynomial Expansions: ./polynomial/wiener_askey.ipynb
