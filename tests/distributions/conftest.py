"""Test configuration for chaospy.distributions."""
import pytest

from chaospy.distributions import collection, Dist

DISTRIBUTIONS = []
for attr in vars(collection).values():
    try:
        if issubclass(attr, Dist) and len(attr()) == 1:
            DISTRIBUTIONS.append(attr)
    except TypeError:
        pass


@pytest.fixture(params=DISTRIBUTIONS)
def distribution(request):
    """Parameterization of distribution."""
    return request.param
