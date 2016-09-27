r"""
Copulas are a type dependency structure imposed on independent variables to
achieve to more complex problems without adding too much complexity.

To construct a copula one needs a copula transformation and the
Copula wrapper::

    >>> dist = cp.Iid(cp.Uniform(), 2)
    >>> copula = cp.Gumbel(dist, theta=1.5)

The resulting copula is then ready for use::

    >>> cp.seed(1000)
    >>> print(copula.sample(5))
    [[ 0.65358959  0.11500694  0.95028286  0.4821914   0.87247454]
    [ 0.02388273  0.10004972  0.00127477  0.10572619  0.4510529 ]]
"""

import chaospy.dist.copulas.baseclass
import chaospy.dist.copulas.collection

from chaospy.dist.copulas.baseclass import Copula, Archimedean
from chaospy.dist.copulas.collection import (
    Gumbel, Clayton, Ali_mikhail_haq, Frank, Joe, Nataf, T_copula
)


if __name__=="__main__":
    import chaospy as cp
    import doctest
    doctest.testmod()
