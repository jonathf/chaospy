"""
Hermite Genz-Keister quadrature rules

Adapted from John Burkardt's implementation in Matlab

Licensing
---------
This code is distributed under the GNU LGPL license.
"""

from .gk16 import quad_genz_keister_16
from .gk18 import quad_genz_keister_18
from .gk22 import quad_genz_keister_22
from .gk24 import quad_genz_keister_24

COLLECTION = {
    16: quad_genz_keister_16,
    18: quad_genz_keister_18,
    22: quad_genz_keister_22,
    24: quad_genz_keister_24,
}

from .genz_keister import quad_genz_keister
