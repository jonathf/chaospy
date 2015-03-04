import chaospy as cp
import numpy as np
import odespy


#Intrusive Galerkin method

dist_a = cp.Uniform(0, 0.1)
dist_I = cp.Uniform(8, 10)
dist = cp.J(dist_a, dist_I) # joint multivariate dist

P, norms = cp.orth_ttr(2, dist, retall=True)
variable_a, variable_I = cp.variable(2)

PP = cp.outer(P, P)
E_aPP = cp.E(variable_a*PP, dist)
E_IP = cp.E(variable_I*P, dist)

def right_hand_side(c, x):            # c' = right_hand_side(c, x)
    return -np.dot(E_aPP, c)/norms    # -M*c

initial_condition = E_IP/norms
solver = odespy.RK4(right_hand_side)
solver.set_initial_condition(initial_condition)
x = np.linspace(0, 10, 1000)
c = solver.solve(x)[0]
u_hat = cp.dot(P, c)



#Rosenblat transformation using point collocation

def u(x,a, I):
    return I*np.exp(-a*x)

dist_R = cp.J(cp.Normal(), cp.Normal())
C = [[1, 0.5], [0.5, 1]]
mu = [0, 0]
dist_Q = cp.MvNormal(mu, C)

P = cp.orth_ttr(2, dist_R)
nodes_R = dist_R.sample(2*len(P), "M")
nodes_Q = dist_Q.inv(dist_R.fwd(nodes_R))

x = np.linspace(0, 1, 100)
samples_u = [u(x, *node) for node in nodes_Q.T]
u_hat = cp.fit_regression(P, nodes_R, samples_u)




#Rosenblat transformation using pseudo spectral

def u(x,a, I):
    return I*np.exp(-a*x)

C = [[1,0.5],[0.5,1]]
mu = np.array([0, 0])
dist_R = cp.J(cp.Normal(), cp.Normal())
dist_Q = cp.MvNormal(mu, C)

P = cp.orth_ttr(2, dist_R)
nodes_R, weights_R = cp.generate_quadrature(3, dist_R)
nodes_Q = dist_Q.inv(dist_R.fwd(nodes_R))
weights_Q = weights_R*dist_Q.pdf(nodes_Q)/dist_R.pdf(nodes_R)

x = np.linspace(0, 1, 100)
samples_u = [u(x, *node) for node in nodes_Q.T]
u_hat = cp.fit_quadrature(P, nodes_R, weights_Q, samples_u)
