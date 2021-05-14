import pytest
import numpy
from scipy.integrate import odeint
import chaospy


@pytest.fixture
def galerkin_approx(coordinates, joint, expansion_small, norms_small):
    alpha, beta = chaospy.variable(2)

    e_alpha_phi = chaospy.E(alpha*expansion_small, joint)
    initial_condition = e_alpha_phi/norms_small

    phi_phi = chaospy.outer(expansion_small, expansion_small)
    e_beta_phi_phi = chaospy.E(beta*phi_phi, joint)

    def right_hand_side(c, t):
        return -numpy.sum(c*e_beta_phi_phi, -1)/norms_small

    coefficients = odeint(
        func=right_hand_side,
        y0=initial_condition,
        t=coordinates,
    )
    return chaospy.sum(expansion_small*coefficients, -1)


def test_galerkin_mean(galerkin_approx, joint, true_mean):
    assert numpy.allclose(chaospy.E(galerkin_approx, joint), true_mean, rtol=1e-12)


def test_galerkin_variance(galerkin_approx, joint, true_variance):
    assert numpy.allclose(chaospy.Var(galerkin_approx, joint), true_variance, rtol=1e-12)
