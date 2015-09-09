import chaospy as cp
import numpy as np


# Defining numerical model
def model(x, u0, c0, c1, c2):
    def c(x):
        if   x < 0.5:           return c0
        elif 0.5 <= x < 0.7:    return c1
        else:                   return c2

    N = len(x)
    u = np.zeros(N)

    u[0] = u0
    for n in xrange(N-1):
        dx = x[n+1] - x[n]
        K1 = -dx*u[n]*c(x[n])
        K2 = -dx*u[n] + K1/2*c(x[n]+dx/2)
        u[n+1] = u[n] + K1 + K2
    return u

# Define distributions of input parameters
c0 = cp.Normal(0.5, 0.15)
c1 = cp.Uniform(0.5, 2.5)
c2 = cp.Uniform(0.03, 0.07)
# Joint probability distribution
distribution = cp.J(c0, c1, c2)

# Create 3rd order quadrature scheme
nodes, weights = cp.generate_quadrature(
    order=3, domain=distribution, rule="Gaussian")

u0 = 0.3
# Evaluate model at the nodes
x = np.linspace(0, 1, 101)
samples = [model(x, u0, node[0], node[1], node[2])
           for node in nodes.T]

# Generate 3rd order orthogonal polynomial expansion
polynomials = cp.orth_ttr(order=3, dist=distribution)

# Create model approximation (surrogate solver)
model_approx = cp.fit_quadrature(
               polynomials, nodes, weights, samples)

# Model analysis
mean = cp.E(model_approx, distribution)
deviation = cp.Std(model_approx, distribution)

# Plot results
from matplotlib import pyplot as plt
plt.rc("figure", figsize=[8,6])
plt.fill_between(x, mean-deviation, mean+deviation, color="k",
        alpha=0.5)
plt.plot(x, mean, "k", lw=2)
plt.xlabel("depth $x$")
plt.ylabel("porosity $u$")
plt.legend(["mean $\pm$ deviation", "mean"])
plt.savefig("ode.pdf")
