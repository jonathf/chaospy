"""
One analysis done with four different approaches:
    1. Monte Carlo integration
    2. Polynomial chaos expansion w/Pseudo-spectral method
    2. Polynomial chaos expansion w/Point collocation method
    2. Polynomial chaos expansion w/Intrusive Galerkin method
"""
import chaospy as cp
import numpy as np
import odespy


# The model solver
def u(x, a, I):
    return I*np.exp(-a*x)
x = np.linspace(0, 10, 1000)


# Defining the random distributions:
a = cp.Uniform(0, 0.1)
I = cp.Uniform(8, 10)
dist = cp.J(a, I)


## Monte Carlo integration
samples = dist.sample(10**5)
u_mc = [u(x, *s) for s in samples.T]

mean = np.mean(u_mc, 1)
var = np.var(u_mc, 1)


## Polynomial chaos expansion
## using Pseudo-spectral method and Gaussian Quadrature
order = 5
P, norms = cp.orth_ttr(order, dist, retall=True)
nodes, weights = cp.generate_quadrature(order+1, dist, rule="G")
solves = [u(x, s[0], s[1]) for s in nodes.T]
U_hat = cp.fit_quadrature(P, nodes, weights, solves, norms=norms)

mean = cp.E(U_hat, dist)
var = cp.Var(U_hat, dist)


## Polynomial chaos expansion
## using Point collocation method and quasi-random samples
order = 5
P = cp.orth_ttr(order, dist)
nodes = dist.sample(2*len(P), "M")
solves = [u(x, s[0], s[1]) for s in nodes.T]
U_hat = cp.fit_regression(P, nodes, solves, rule="T")

mean = cp.E(U_hat, dist)
var = cp.Var(U_hat, dist)


## Polynomial chaos expansion
## using Intrusive Gallerkin method
# :math:
# u' = -a*u
# d/dx sum(c*P) = -a*sum(c*P)
# <d/dx sum(c*P),P[k]> = <-a*sum(c*P), P[k]>
# d/dx c[k]*<P[k],P[k]> = -sum(c*<a*P,P[k]>)
# d/dx c = -E( outer(a*P,P) ) / E( P*P )
#
# u(0) = I
# <sum(c(0)*P), P[k]> = <I, P[k]>
# c[k](0) <P[k],P[k]> = <I, P[k]>
# c(0) = E( I*P ) / E( P*P )
order = 5
P, norm = cp.orth_ttr(order, dist, retall=True, normed=True)

# support structures
q0, q1 = cp.variable(2)
P_nk = cp.outer(P, P)
E_ank = cp.E(q0*P_nk, dist)
E_ik = cp.E(q1*P, dist)
sE_ank = cp.sum(E_ank, 0)

# Right hand side of the ODE
def f(c_k, x):
    return -cp.sum(c_k*E_ank, -1)/norm

solver = odespy.RK4(f)
c_0 = E_ik/norm
solver.set_initial_condition(c_0)
c_n, x = solver.solve(x)
U_hat = cp.sum(P*c_n, -1)

mean = cp.E(U_hat, dist)
var = cp.Var(U_hat, dist)
