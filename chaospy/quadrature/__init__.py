r"""Collection of quadrature methods."""
import logging
from functools import wraps

from .frontend import generate_quadrature
from .sparse_grid import sparse_grid
from .utils import combine

from .chebyshev import chebyshev_1, chebyshev_2
from .clenshaw_curtis import clenshaw_curtis
from .discrete import discrete
from .fejer_1 import fejer_1
from .fejer_2 import fejer_2
from .gaussian import gaussian
from .genz_keister import (
    genz_keister_16, genz_keister_18, genz_keister_22, genz_keister_24)
from .gegenbauer import gegenbauer
from .grid import grid
from .hermite import hermite
from .jacobi import jacobi
from .kronrod import kronrod, kronrod_jacobi
from .laguerre import laguerre
from .legendre import legendre, legendre_proxy
from .leja import leja
from .lobatto import lobatto
from .newton_cotes import newton_cotes
from .patterson import patterson
from .radau import radau

__all__ = ["generate_quadrature", "sparse_grid", "combine"]


INTEGRATION_COLLECTION = {
    "clenshaw_curtis": clenshaw_curtis,
    "discrete": discrete,
    "fejer_1": fejer_1,
    "fejer_2": fejer_2,
    "gaussian": gaussian,
    "genz_keister_16": genz_keister_16,
    "genz_keister_18": genz_keister_18,
    "genz_keister_22": genz_keister_22,
    "genz_keister_24": genz_keister_24,
    "grid": grid,
    "kronrod": kronrod,
    "legendre": legendre_proxy,
    "leja": leja,
    "lobatto": lobatto,
    "newton_cotes": newton_cotes,
    "patterson": patterson,
    "radau": radau,
}


def quadrature_deprecation_warning(name, func):
    """Announce deprecation warning for quad-func."""
    quad_name = "quad_%s" % name

    @wraps(func)
    def wrapped(*args, **kwargs):
        """Function wrapper adds warnings."""
        logger = logging.getLogger(__name__)
        logger.warning("chaospy.%s name is to be deprecated; "
                       "Use chaospy.quadrature.%s instead",
                       quad_name, func.__name__)
        return func(*args, **kwargs)

    globals()[quad_name] = wrapped
    __all__.append(quad_name)

quadrature_deprecation_warning("clenshaw_curtis", clenshaw_curtis)
quadrature_deprecation_warning("discrete", discrete)
quadrature_deprecation_warning("fejer", fejer_2)
quadrature_deprecation_warning("grid", grid)
quadrature_deprecation_warning("gaussian", gaussian)
quadrature_deprecation_warning("newton_cotes", newton_cotes)
quadrature_deprecation_warning("leja", leja)
quadrature_deprecation_warning("gauss_legendre", legendre_proxy)
quadrature_deprecation_warning("gauss_kronrod", kronrod)
quadrature_deprecation_warning("gauss_lobatto", lobatto)
quadrature_deprecation_warning("gauss_patterson", patterson)
quadrature_deprecation_warning("gauss_radau", radau)
quadrature_deprecation_warning("genz_keister", genz_keister_24)
