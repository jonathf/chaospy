r"""Collection of polynomial expansion constructors."""
import logging
from functools import wraps

from .chebyshev import chebyshev_1, chebyshev_2
from .cholesky import cholesky
from .frontend import generate_expansion
from .gegenbauer import gegenbauer
from .gram_schmidt import gram_schmidt
from .hermite import hermite
from .jacobi import jacobi
from .stieltjes import stieltjes
from .lagrange import lagrange
from .laguerre import laguerre
from .legendre import legendre

__all__ = ["generate_expansion"]


def expansion_deprecation_warning(name, func):

    @wraps(func)
    def wrapped(*args, **kwargs):
        """Function wrapper adds warnings."""
        logger = logging.getLogger(__name__)
        logger.warning("chaospy.%s name is to be deprecated; "
                       "Use chaospy.expansion.%s instead",
                       name, func.__name__)
        return func(*args, **kwargs)

    globals()[name] = wrapped
    __all__.append(name)


expansion_deprecation_warning("orth_ttr", stieltjes)
expansion_deprecation_warning("orth_chol", cholesky)
expansion_deprecation_warning("orth_gs", gram_schmidt)
expansion_deprecation_warning("lagrange_polynomial", lagrange)
