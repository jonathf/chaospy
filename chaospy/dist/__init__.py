"""
Rosenblatt Distributions (RoseDist)

This module focuses on construction, manipulation and analysis of
random variables, and their underlying probability distribution.
The main component for addressing multivariate distribution is
through the automatical handling of Rosenblatt transformations.

The module contains the following submodules:

    approx          Methods for calculating variable statistics
                    when the approriate function is unavailable.
    backend         The superclass for the distribution object
                    Dist. All distributions are based on this one.
    collection      Collection of various probability
                    distributions.
    copulas         Collection of various copulas.
    cores           Backend module for collection submodule
    graph           The internal module for handling dependencies
                    between variables. Also the core for creating
                    advanced variables.
    hyperbolic      Collection of hyperbolic distributions.
    joint           Tools for creating multivariat distributions.
    operators       Collection of basic operators (+,-,*,/,^,>,<).
    sampler         Sample generator on the unit-hypercube.
    sobol_lib       Library for generating Sobol sequences.
    trignomatric    Collection of trignometric operators.

Examples
--------
>>> X1 = cp.Normal(0,1)
>>> print(X1.mom([1,2,3,4]))
[ 0.  1.  0.  3.]

>>> X2 = cp.Uniform(0,4)
>>> print(X2.fwd([2,3,4]))
[ 0.5   0.75  1.  ]

>>> X = cp.J(X1,X2)
>>> print(X.sample(3))
[[ 0.39502989 -1.20032309  1.64760248]
 [ 1.92876561  3.48989814  0.84933072]]

>>> Y = cp.Iid(X1, 3)
>>> print(cp.Cov(Y))
[[ 1.  0.  0.]
 [ 0.  1.  0.]
 [ 0.  0.  1.]]

Distributions
-------------
Uniform         Uniform
Loguniform      Log-uniform
Normal          Normal (Gaussian)
Lognormal       Log-normal
Gamma           Gamma
Expon           Exponential
Laplace         Laplace
Beta            Beta
Weibull         Weibull
Triangle        Triangle
Wigner          Wigner (semi-circle)
Kumaraswamy     Kumaraswswamy's double bounded
Hypgeosec       hyperbolic secant
Arcsinus        Generalized Arc-sinus
Logistic        Generalized logistic type 1 or Sech squared
Student_t       (Non-central) Student-t
Raised_cosinei  Raised cosine
Alpha           Alpha
MvNormal        Multivariate Normal
MvLognormal     Multivariate Log-Normal
MvStudent_t     Multivariate student-t
Anglit          Anglit
Bradford        Bradford
Burr            Burr Type XII or Singh-Maddala
Fisk            Fisk or Log-logistic
Cauchy          Cauchy
Chi             Chi
Dbl_gamma       Double gamma
Dbl_weibull     Double weibull
Exponweibull    Expontiated Weibull
Exponpow        Expontial power or Generalized normal version 1
Fatiguelife     Fatigue-Life or Birmbaum-Sanders
Foldcauchy      Folded Cauchy
F               (Non-central) F or Fisher-Snedecor
Foldnormal      Folded normal
Frechet         Frechet or Extreme value type 2
Genexpon        Generalized exponential
Genextreme      Generalized extreme value or Fisher-Tippett
Gengamma        Generalized gamma
Genhalflogistic Generalized half-logistic
Gompertz        Gompertz
Gumbel          Gumbel or Log-Weibull
Levy            Levy
Loggamma        Log-gamma
Loglaplace      Log-laplace
Gibrat          Gilbrat or Standard log-normal
Maxwell         Maxwell-Boltzmann
Mielke          Mielke's beta-kappa
Nakagami        Nakagami-m
Chisquard       (Non-central) Chi-squared
Pareto1         Pareto type 1
Pareto2         Pareto type 2
Powerlaw        Powerlaw
Powerlognormal  Power log-normal
Powernorm       Power normal or Box-Cox
Wald            Wald or Reciprocal inverse Gaussian
Rayleigh        Rayleigh
Reciprocal      Reciprocal
Truncexpon      Truncated exponential
Truncnorm       Truncated normal
Tukeylambda     Tukey-lambda
Wrapcauchy      Wraped Cauchy
"""

import chaospy.dist.backend
from chaospy.dist.backend import *

import chaospy.dist.graph
import chaospy.dist.sampler
import chaospy.dist.approx
import chaospy.dist.joint
import chaospy.dist.cores
import chaospy.dist.copulas
import chaospy.dist.collection
import chaospy.dist.operators
import chaospy.dist.hyperbolic
import chaospy.dist.trignometric

from chaospy.dist.graph import *
from chaospy.dist.sampler import *
from chaospy.dist.approx import *
from chaospy.dist.joint import *
from chaospy.dist.cores import *
from chaospy.dist.copulas import *
from chaospy.dist.collection import *
from chaospy.dist.operators import *
from chaospy.dist.hyperbolic import *
from chaospy.dist.trignometric import *

from numpy.random import seed


if __name__ == "__main__":
    seed(1000)
    import doctest
    import chaospy as cp
    doctest.testmod()
