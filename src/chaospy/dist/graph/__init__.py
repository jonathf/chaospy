r"""
Constructing an transitional operator requires to construct the
distributions different from a stand-alone variable. To illustrate how
to construct these types of variables, consider the following example.
Let :math:`Q=A+B`, where one of :math:`A` and :math:`B` is a random
variable, and the other a scalar. Which variable is what dependents on
the user setup of the variable. Assuming that :math:`A` is the random
variable, we have that

.. math::

    F_{Q\mid B}(q\mid b) = \mathbb P {q\leq Q\mid B\!=\!b} =
    \mathbb P {q\leq AB\mid B\!=\!b}

    = \mathbb P {\tfrac qb\leq A\mid B\!=\!b} =
    F_{A\mid B}(\tfrac qb\mid b).

Because of symmetry the distribution will be the same, but with
:math:`A` and :math:`B` substituted.

This is required when trying to use operators on multivariate variables. To
create such a variable with ``construct`` provide an additional ``length``
keyword argument specifying the length of a distribution.
"""
from .baseclass import Graph
