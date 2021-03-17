"""Test for Triangle distribution."""
import pytest
import chaospy as cp

def test_triangle_init():
    """Assert that initialization checks lower and upper bounds."""
    # Should just run
    u = cp.Uniform(0., 1.)
    t = cp.Triangle(u - 1., u, u + 1.)
    cp.J(u, t)
    
    # Overlapping lb and ub
    with pytest.raises(ValueError): 
        cp.Triangle(u - 0.5, u, u + 1.)
    with pytest.raises(ValueError): 
        cp.Triangle(u - 1., u, u + 0.5)
    
    # Should initialize fine
    cp.Triangle(u - 1., 0., u + 0.5)