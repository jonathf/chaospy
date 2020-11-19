Distributions
=============

.. currentmodule:: chaospy

Baseclass
---------

.. autosummary::
   :template: distribution.rst
   :toctree: api

   Distribution

.. autosummary::
   :toctree: api

   Distribution.pdf
   Distribution.cdf
   Distribution.fwd
   Distribution.inv
   Distribution.sample
   Distribution.mom
   Distribution.ttr

   Distribution.interpret_as_integer
   Distribution.stochastic_dependent
   Distribution.lower
   Distribution.upper


Unbound distributions
---------------------

.. autosummary::
   :template: distribution.rst
   :toctree: api

   Cauchy
   DoubleGamma
   DoubleWeibull
   GeneralizedExtreme
   HyperbolicSecant
   Laplace
   LogGamma
   Logistic
   Normal
   PowerNormal
   StudentT

Partially bound distributions
-----------------------------

.. autosummary::
   :template: distribution.rst
   :toctree: api

   Alpha
   Burr
   Chi
   ChiSquared
   Maxwell
   Exponential
   ExponentialPower
   ExponentialWeibull
   F
   Fisk
   FoldedCauchy
   FoldedNormal
   Frechet
   Gamma
   GeneralizedExponential
   GeneralizedGamma
   GeneralizedHalfLogistic
   Gompertz
   InverseGamma
   Levy
   LogLaplace
   LogNormal
   LogWeibull
   Mielke
   Nakagami
   Pareto1
   Pareto2
   PowerLogNormal
   Wald
   Weibull
   WrappedCauchy

Bound distributions
-------------------

.. autosummary::
   :template: distribution.rst
   :toctree: api

   Anglit
   ArcSinus
   Beta
   Bradford
   FatigueLife
   PowerLaw
   Wigner
   PERT
   Kumaraswamy
   LogUniform
   Reciprocal
   Triangle
   TruncExponential
   TruncNormal
   TukeyLambda
   Uniform

Multivariate distributions
--------------------------

.. autosummary::
   :template: distribution.rst
   :toctree: api

   MvLogNormal
   MvNormal
   MvStudentT

Discrete distributions
----------------------

.. autosummary::
   :template: distribution.rst
   :toctree: api

   Binomial
   DiscreteUniform

Copulas
-------

.. autosummary::
   :template: distribution.rst
   :toctree: api

   Clayton
   Gumbel
   Joe
   Nataf
   TCopula

Operators
---------

.. autosummary::
   :template: distribution.rst
   :toctree: api

   J
   Iid
   Add
   Multiply
   Negative
   Power
   Trunc
   Log
   Log10
   Logn

Kernel Estimation
-----------------

.. autosummary::
   :template: distribution.rst
   :toctree: api

   GaussianKDE

Mixtures
--------

.. autosummary::
   :template: distribution.rst
   :toctree: api

   GaussianMixture
