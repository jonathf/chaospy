r"""
Copulas are a type dependency structure imposed on independent variables to
achieve to more complex problems without adding too much complexity.

To construct a copula one needs a copula transformation and the
Copula wrapper::

    >>> dist = chaospy.Iid(chaospy.Uniform(), 2)
    >>> copula = chaospy.Gumbel(dist, theta=1.5)

The resulting copula is then ready for use::

    >>> copula.sample(5).round(4)
    array([[0.6536, 0.115 , 0.9503, 0.4822, 0.8725],
           [0.6286, 0.0654, 0.96  , 0.5073, 0.9705]])

"""
from .archimedean import Archimedean

from .gumbel import Gumbel
from .clayton import Clayton
from .joe import Joe
from .nataf import Nataf
from .t_copula import TCopula
