"""
Hermite Genz-Keister quadrature rules

Adapted from John Burkardt's implementation in Matlab
"""
from .frontend import quad_genz_keister, GENS_KEISTER_FUNCTIONS

from .gk16 import quad_genz_keister_16
from .gk18 import quad_genz_keister_18
from .gk22 import quad_genz_keister_22
from .gk24 import quad_genz_keister_24
