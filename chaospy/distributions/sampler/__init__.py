"""Collection of variance reduction techniques."""
from .generator import generate_samples

from .sequences import *  # pylint: disable=wildcard-import
from .latin_hypercube import create_latin_hypercube_samples
from .antithetic import create_antithetic_variates
