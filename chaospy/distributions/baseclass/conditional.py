"""Baseclass for all conditional distributions."""
import numpy
from .distribution import Distribution


class Conditional(Distribution):
    """Conditional distribution baseclass."""

    def get_parameters(self, cache):
        """Get distribution parameters."""
        parameters = super(Conditional, self).get_parameters(cache)
        if "conditions" not in parameters:
            parameters["conditions"] = []
        return parameters
