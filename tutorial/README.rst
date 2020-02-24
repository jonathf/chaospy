Tutorial
========

Here is a collection of `Jupyter notebooks <https://jupyter.org/>`_ that
explore different ways to use ``chaospy``.

Pre-cursor
----------

`Example Introduction <./example_introduction.ipynb>`_
   The notebooks are all self contained, however many of the examples addressed
   is repeated a lot. In this introduction, the common example is explored in a
   little more detail, whereas in the other notebooks, the example is skimmed
   quickly.

Basics Topics
-------------

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


Advanced Topics
---------------

`Intrusive Galerkin`_
   Intrusive Galerkin method is the classical approach for applying polynomial
   chaos expansion on a set of governing equations. An intrusive method, unlike
   the non-intrusive method described in `Point Collocation`_ and
   `Pseudo-Spectral Projection`. It requires a bit more use of the mathematical
   hand holding to apply properly.
`Polynomial Behavior`_
   Both the polynomial expansions and the model approximation created with the
   various polynomial chaos expansions are flexible polynomial vectors that can
   easily be manipulated and transformed.

.. _Example Introduction: ./example_introduction.ipynb
.. _Low-discrepancy Monte Carlo: ./low_discrepancy_monte_carlo.ipynb
.. _Point Collocation: ./point_collocation.ipynb
.. _Pseudo-Spectral Projection: ./pseudo_spectral_projection.ipynb
.. _Intrusive Galerkin: ./intrusive_galerkin.ipynb
.. _Polynomial Behavior: ./polynomial_behavior.ipynb
