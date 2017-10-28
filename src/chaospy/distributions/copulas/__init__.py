r"""
Copulas are a type dependency structure imposed on independent variables to
achieve to more complex problems without adding too much complexity.

To construct a copula one needs a copula transformation and the
Copula wrapper::

    >>> dist = chaospy.Iid(chaospy.Uniform(), 2)
    >>> copula = chaospy.Gumbel(dist, theta=1.5)

The resulting copula is then ready for use::

    >>> print(copula.sample(5))
    [[ 0.65358959  0.11500694  0.95028286  0.4821914   0.87247454]
     [ 0.24832558  0.33253071  0.1725121   0.32059106  0.27318735]]
"""
from . import baseclass, collection

from .baseclass import Copula, Archimedean
from .collection import (
    Gumbel, Clayton, Ali_mikhail_haq, Frank, Joe, Nataf, T_copula
)
