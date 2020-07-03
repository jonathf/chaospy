.. _listdistributions:

List of Distributions
=====================

.. note::
    Note that distributions for which there is no specific truncated variant,
    can be truncated using the generic truncation feature, i.e.

    .. code-block::

        >>> upper_truncated_weibull = chaospy.Weibull(1, 1) < 2
        >>> print(upper_truncated_weibull)
        Trunc(Weibull(scale=1, shape=1, shift=0), 2)
        >>> upper_and_lower_truncated_weibull = upper_truncated_weibull > 0.5
        >>> print(upper_and_lower_truncated_weibull)
        Trunc(0.5, Trunc(Weibull(scale=1, shape=1, shift=0), 2))

Alpha Distribution
------------------

.. autoclass:: chaospy.distributions.collection.alpha.Alpha

Anglit Distribution
-------------------

.. autoclass:: chaospy.distributions.collection.anglit.Anglit

Arc-Sinus Distribution
----------------------

.. autoclass:: chaospy.distributions.collection.beta.ArcSinus

Beta Distribution
-----------------

.. autoclass:: chaospy.distributions.collection.beta.Beta

Bradford Distribution
---------------------

.. autoclass:: chaospy.distributions.collection.bradford.Bradford

Burr Distribution
-----------------

.. autoclass:: chaospy.distributions.collection.burr.Burr

Cauchy Distribution
-------------------

.. autoclass:: chaospy.distributions.collection.cauchy.Cauchy

Chi Distribution
----------------

.. autoclass:: chaospy.distributions.collection.chi.Chi

Chi-Squared Distribution
------------------------

.. autoclass:: chaospy.distributions.collection.chi_squared.ChiSquared

Double-Gamma Distribution
-------------------------

.. autoclass:: chaospy.distributions.collection.double_gamma.DoubleGamma

Double-Weibull Distribution
---------------------------

.. autoclass:: chaospy.distributions.collection.double_weibull.DoubleWeibull

Exponential Distribution
------------------------

.. autoclass:: chaospy.distributions.collection.gamma.Exponential

Exponential Power Distribution
------------------------------

.. autoclass:: chaospy.distributions.collection.exponential_power.ExponentialPower

Exponential Weibull Distribution
--------------------------------

.. autoclass:: chaospy.distributions.collection.exponential_weibull.ExponentialWeibull

F Distribution
--------------

.. autoclass:: chaospy.distributions.collection.f.F

Fatigue Life Distribution
-------------------------

.. autoclass:: chaospy.distributions.collection.fatigue_life.FatigueLife

Fisk Distribution
-----------------

.. autoclass:: chaospy.distributions.collection.fisk.Fisk

Folded Cauchy Distribution
--------------------------

.. autoclass:: chaospy.distributions.collection.folded_cauchy.FoldedCauchy

Folded Normal Distribution
--------------------------

.. autoclass:: chaospy.distributions.collection.folded_normal.FoldedNormal

Frechet Distribution
--------------------

.. autoclass:: chaospy.distributions.collection.frechet.Frechet

Gamma Distribution
------------------

.. autoclass:: chaospy.distributions.collection.gamma.Gamma

Generalized Exponential Distribution
------------------------------------

.. autoclass:: chaospy.distributions.collection.generalized_exponential.GeneralizedExponential

Generalized Extreme Distribution
--------------------------------

.. autoclass:: chaospy.distributions.collection.generalized_extreme.GeneralizedExtreme

Generalized Gamma Distribution
------------------------------

.. autoclass:: chaospy.distributions.collection.generalized_gamma.GeneralizedGamma

Generalized Half-Logistic Distribution
--------------------------------------

.. autoclass:: chaospy.distributions.collection.generalized_half_logistic.GeneralizedHalfLogistic

Gilbrat Distribution
--------------------

.. autoclass:: chaospy.distributions.collection.log_normal.Gilbrat

Gompertz Distribution
---------------------

.. autoclass:: chaospy.distributions.collection.gompertz.Gompertz

Hyperbolic Secant Distribution
------------------------------

.. autoclass:: chaospy.distributions.collection.hyperbolic_secant.HyperbolicSecant

Kumaraswamy Distribution
------------------------

.. autoclass:: chaospy.distributions.collection.kumaraswamy.Kumaraswamy

Laplace Distribution
--------------------

.. autoclass:: chaospy.distributions.collection.laplace.Laplace

Levy Distribution
-----------------

.. autoclass:: chaospy.distributions.collection.levy.Levy

Log-Gamma Distribution
----------------------

.. autoclass:: chaospy.distributions.collection.log_gamma.LogGamma

Log-Laplace Distribution
------------------------

.. autoclass:: chaospy.distributions.collection.log_laplace.LogLaplace

Log-Normal Distribution
-----------------------

.. autoclass:: chaospy.distributions.collection.log_normal.LogNormal

Log-Uniform Distribution
------------------------

.. autoclass:: chaospy.distributions.collection.log_uniform.LogUniform

Log-Weibull Distribution
------------------------

.. autoclass:: chaospy.distributions.collection.log_weibull.LogWeibull

Logistic Distribution
---------------------

.. autoclass:: chaospy.distributions.collection.logistic.Logistic

Maxwell Distribution
--------------------

.. autoclass:: chaospy.distributions.collection.chi.Maxwell

Mielke Distribution
-------------------

.. autoclass:: chaospy.distributions.collection.mielke.Mielke

Multivariate Log-Normal Distribution
------------------------------------

.. autoclass:: chaospy.distributions.collection.mv_log_normal.MvLogNormal

Multivariate Normal Distribution
--------------------------------

.. autoclass:: chaospy.distributions.collection.mv_normal.MvNormal

Multivariate Student-T Distribution
-----------------------------------

.. autoclass:: chaospy.distributions.collection.mv_student_t.MvStudentT

Nakagami Distribution
---------------------

.. autoclass:: chaospy.distributions.collection.nakagami.Nakagami

Normal Distribution
-------------------

.. autoclass:: chaospy.distributions.collection.normal.Normal

Pareto 1 Distribution
---------------------

.. autoclass:: chaospy.distributions.collection.pareto1.Pareto1

(Generalized) PERT Distribution
-------------------------------

.. autoclass:: chaospy.distributions.collection.beta.PERT

Pareto 2 Distribution
---------------------

.. autoclass:: chaospy.distributions.collection.pareto2.Pareto2

Power-Law Distribution
----------------------

.. autoclass:: chaospy.distributions.collection.beta.PowerLaw

Power-Log-Normal Distribution
-----------------------------

.. autoclass:: chaospy.distributions.collection.power_log_normal.PowerLogNormal

Power-Normal Distribution
-------------------------

.. autoclass:: chaospy.distributions.collection.power_normal.PowerNormal

Rayleigh Distribution
---------------------

.. autoclass:: chaospy.distributions.collection.chi.Rayleigh

Reciprocal Distribution
-----------------------

.. autoclass:: chaospy.distributions.collection.reciprocal.Reciprocal

Student-T Distribution
----------------------

.. autoclass:: chaospy.distributions.collection.student_t.StudentT

Triangle Distribution
---------------------

.. autoclass:: chaospy.distributions.collection.triangle.Triangle

Truncated Normal Distribution
-----------------------------

.. autoclass:: chaospy.distributions.collection.trunc_normal.TruncNormal

Truncated Exponential Distribution
----------------------------------

.. autoclass:: chaospy.distributions.collection.trunc_exponential.TruncExponential

Tukey-Lambda Distribution
-------------------------

.. autoclass:: chaospy.distributions.collection.tukey_lambda.TukeyLambda

Uniform Distribution
--------------------

.. autoclass:: chaospy.distributions.collection.uniform.Uniform

Wald Distribution
-----------------

.. autoclass:: chaospy.distributions.collection.wald.Wald

Weibull Distribution
--------------------

.. autoclass:: chaospy.distributions.collection.weibull.Weibull

Wigner Distribution
-------------------

.. autoclass:: chaospy.distributions.collection.beta.Wigner

Wrapped-Cauchy Distribution
---------------------------

.. autoclass:: chaospy.distributions.collection.wrapped_cauchy.WrappedCauchy
