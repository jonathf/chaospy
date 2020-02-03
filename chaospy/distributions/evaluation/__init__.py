import logging
import inspect

import numpy

from ... import quadrature

from .common import DependencyError
from .parameters import load_parameters as load_inputs
from .dependencies import sorted_dependencies, get_dependencies

from .density import evaluate_density
from .forward import evaluate_forward
from .inverse import evaluate_inverse
from .bound import evaluate_lower, evaluate_upper
from .moment import evaluate_moment
from .recurrence_coefficients import evaluate_recurrence_coefficients


def get_forward_cache(
        distribution,
        cache,
):
    from .. import baseclass
    if not isinstance(distribution, baseclass.Dist):
        return distribution
    if distribution in cache:
        return cache[distribution]
    if hasattr(distribution, "_fwd_cache"):
        return distribution._fwd_cache(cache)
    return distribution

def get_inverse_cache(
        distribution,
        cache,
):
    from .. import baseclass
    if not isinstance(distribution, baseclass.Dist):
        return distribution
    if distribution in cache:
        return cache[distribution]
    if hasattr(distribution, "_inv_cache"):
        return distribution._inv_cache(cache)
    return distribution
