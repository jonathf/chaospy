import chaospy as cp
import numpy as np
import pylab as plt

x = np.linspace(0, 10, 100)

def u(x, a, I):
  return I*np.exp(-a*x)


#Create the distributions
dist_a = cp.Uniform(0, 0.1)
dist_I = cp.Uniform(8, 10)
dist = cp.J(dist_a, dist_I)


#Pseudo spectral method
P = cp.orth_ttr(3, dist)
nodes, weights = cp.generate_quadrature(4, dist)
samples_u = [u(x, *node) for node in nodes.T]
u_hat= cp.fit_quadrature(P, nodes, weights, samples_u, rule="G")
mean = cp.E(u_hat, dist)
var = cp.Var(u_hat, dist)


#Point collocation
P = cp.orth_ttr(3, dist)
nodes = dist.sample(2*len(P), "M")
samples_u = [u(x, *node) for node in nodes.T]
u_hat = cp.fit_regression(P, nodes, samples_u, rule="LS")
mean = cp.E(u_hat, dist)
var = cp.Var(u_hat, dist)

