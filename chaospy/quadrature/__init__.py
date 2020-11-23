r"""Collection of quadrature methods."""
from .frontend import generate_quadrature
from .sparse_grid import construct_sparse_grid
from .combine import combine

from .clenshaw_curtis import quad_clenshaw_curtis
from .discrete import quad_discrete
from .fejer import quad_fejer
from .gaussian import quad_gaussian
from .gauss_patterson import quad_gauss_patterson
from .gauss_legendre import quad_gauss_legendre
from .gauss_lobatto import quad_gauss_lobatto
from .gauss_kronrod import quad_gauss_kronrod, kronrod_jacobi
from .gauss_radau import quad_gauss_radau
from .genz_keister import quad_genz_keister
from .grid import quad_grid
from .leja import quad_leja
from .newton_cotes import quad_newton_cotes
