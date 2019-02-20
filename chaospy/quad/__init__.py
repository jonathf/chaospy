r"""
Quadrature methods, or numerical integration, is broad class of algorithm for
performing integration of any function :math:`g` that are defined without
requiring an analytical definition. In the scope of ``chaospy`` we limit this
scope to focus on methods that can be reduced to the following approximation:

.. math::
    \int g(x) p(x) dx = \sum_{n=1}^N W_n g(X_n)

Here :math:`p(x)` is an weight function, which for our use would be an
probability distribution, and :math:`W_n` and :math:`X_n` are respectively
quadrature weights and abscissas used to define the approximation.

This simplest example of such an approximation is Monte Carlo integration. In
such a method, you only need to select :math:`W_n=1/N` and :math:`X_n` to be
independent identical distributed samples drawn from the distribution of
:math:`p(x)`.

However, except for very high dimensional problems, Monte Carlo is quite an
inefficient way to perform numerical integration, and there exist quite a few
methods that performs better in most low-dimensional settings.
"""
from .combine import combine
from .interface import generate_quadrature

from .stieltjes import generate_stieltjes
from .collection import *
from .sparse_grid import sparse_grid
from .generator import rule_generator
