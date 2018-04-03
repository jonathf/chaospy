r"""
Copulas are a type dependency structure imposed on independent variables to
achieve to more complex problems without adding too much complexity.

To construct a copula one needs a copula transformation and the
Copula wrapper::

    >>> dist = chaospy.Iid(chaospy.Uniform(), 2)
    >>> copula = chaospy.Gumbel(dist, theta=1.5)

The resulting copula is then ready for use::

    >>> print(numpy.around(copula.sample(5), 4))
    [[0.6536 0.115  0.9503 0.4822 0.8725]
     [0.2483 0.3325 0.1725 0.3206 0.2732]]
"""
from .baseclass import Copula, Archimedean

from .gumbel import Gumbel
from .clayton import Clayton
from .ali_mikhail_haq import AliMikhailHaq
from .frank import Frank
from .joe import Joe
from .nataf import Nataf
from .t_copula import TCopula
