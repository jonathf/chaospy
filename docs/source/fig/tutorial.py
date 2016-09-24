import chaospy as cp
import numpy as np
import matplotlib.pyplot as plt

# Setup:

def foo(coord, param):
    return param[0] * np.e ** (-param[1] * coord)

coord = np.linspace(0, 10, 200)

distribution = cp.J(
    cp.Uniform(1, 2),
    cp.Uniform(0.1, 0.2)
)

# Example:

samples = distribution.sample(50)
evals = [foo(coord, sample) for sample in samples.T]

[plt.plot(coord, eval, "r") for eval in evals]
plt.savefig("intro_demo.png")
plt.clf()

# Monte Carlo:

samples = distribution.sample(1000, "H")
evals = [foo(coord, sample) for sample in samples.T]

expected = np.mean(evals, 0)
deviation = np.std(evals, 0)

plt.fill_between(
    coord, expected-deviation, expected+deviation, color="k", alpha=0.5)
plt.plot(coord, expected, "r")
plt.savefig("intro_montecarlo.png")
plt.clf()

polynomial_expansion = cp.orth_ttr(8, distribution)

foo_approx = cp.fit_regression(polynomial_expansion, samples, evals)

expected = cp.E(foo_approx, distribution)
deviation = cp.Std(foo_approx, distribution)

plt.fill_between(
    coord, expected-deviation, expected+deviation, color="k", alpha=0.5)
plt.plot(coord, expected, "r")
plt.savefig("intro_collocation.png")
plt.clf()


absissas, weights = cp.generate_quadrature(8, distribution, "C")
evals = [foo(coord, val) for val in absissas.T]
foo_approx = cp.fit_quadrature(
    polynomial_expansion, absissas, weights, evals
)

expected = cp.E(foo_approx, distribution)
deviation = cp.Std(foo_approx, distribution)

plt.fill_between(
    coord, expected-deviation, expected+deviation, color="k", alpha=0.5)
plt.plot(coord, expected, "r")
plt.savefig("intro_spectral.png")
plt.clf()
