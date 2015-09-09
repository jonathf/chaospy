from pylab import *
import current as pc
from flow_in_serial_layers import SerialLayers

rc("figure", figsize=[8.,4.])
rc("figure.subplot", left=.08, top=.95, right=.98)
seed(1001)

q0 = pc.Triangle(0/6., 1/6., 2/6.)
q1 = pc.Triangle(2/6., 3/6., 4/6.)
q2 = pc.Triangle(4/6., 5/6., 6/6.)
Q = pc.J(q0,q1,q2)
#end

z = linspace(0,1,1000)
def model_wrapper(q):
    layers = [0.001, 1., 0.001, 1.]
    bounds = [0, q[0], q[1], q[2], 1]
    U_0, U_L = 0, 1
    sl = SerialLayers(layers, bounds, U_0, U_L)
    return sl(z)
#end

model_wrapper = pc.lazy_eval(model_wrapper, load="model_data.d")
#end
model_wrapper.save("model_data.d")
#end

samples = Q.sample(5)
U = [model_wrapper(q) for q in samples.T]
#end

plot(z, array(U).T, "k")
xlabel(r"Spacial location \verb;x;")
ylabel(r"Flow velocity \verb;U;")
savefig("intro1.pdf"); clf()

approx = pc.pcm(model_wrapper, 2, Q)
#end

sample = Q.sample()
U1 = model_wrapper(sample)
U2 = approx(*sample)
#end

plot(z, U1, "k-")
plot(z, U2, "k--")
xlabel(r"Spacial location \verb;x;")
ylabel(r"Flow velocity \verb;U;")
legend([r"\verb;model_solver;",r"\verb;approx;"], loc="upper left")
savefig("intro2.pdf"); clf()

E = pc.E(approx, Q)
q05, q95 = pc.Perc(approx, [5,95], Q)
#end

plot(z, E, "k-", label="polynomial approx")
plot(z, q05, "k-")
plot(z, q95, "k-")

#  samples = Q.sample(27, "H")
#  U = [model_wrapper(q) for q in samples.T]
#  E = mean(U, 0)
#  q05, q95 = percentile(U, [5, 95], 0)
#  #end
#  
#  plot(z, E, "k--", label="quasi-Monte Carlo")
#  plot(z, q05, "k--")
#  plot(z, q95, "k--")

samples = Q.sample(10**3, "H")
U = [model_wrapper(q) for q in samples.T]
E = mean(U, 0)
q05, q95 = percentile(U, [5, 95], 0)
#end

plot(z, E, "k:", label="quasi-Monte Carlo")
plot(z, q05, "k:")
plot(z, q95, "k:")

xlabel(r"Spacial location \verb;x;")
ylabel(r"Flow velocity")
axis([0,1,0,1])
legend(loc="upper left")

savefig("intro3.pdf"); clf()

mu = [0.001, 0.01, 0.1]
Sigma = [[1,.5,.5],[.5,1,.5],[.5,.5,1]]
#end

N = pc.Iid(pc.Normal(0,1), 3)
L = linalg.cholesky(Sigma)
Q = e**(N0*L + mu)
#end

orth_poly = pc.orth_ttr(2, N)
#end

approx = pc.pcm(model_wrapper, 2, Q, proxy_dist=N)
fail


print len(w)
#  9

print approx
#  [0.0, -0.014499323957q1-0.014499323957q0+0.240519453193,
#      -0.0289986479139q1-0.0289986479139q0+0.481038906386]
#end

print pc.E(approx, Q)
#  [ 0.          0.19270877  0.38541753]

print pc.Cov(approx, Q)
#  [[ 0.          0.          0.        ]
#   [ 0.          0.00196388  0.00392775]
#   [ 0.          0.00392775  0.00785551]]

print approx(*Q.sample(2))
#  [[ 0.          0.        ]
#   [ 0.14367935  0.22228764]
#   [ 0.28735869  0.44457529]]
#end

mu = [-1.0, -2.0, -1.0, -2.0]
Cov = [[1.0, 0.5, 0.5, 0.5],
       [0.5, 1.0, 0.5, 0.5],
       [0.5, 0.5, 1.0, 0.5],
       [0.5, 0.5, 0.5, 1.0]]

Q = pc.MvLognormal(mu, Cov)
#end

Z = pc.Iid(pc.Normal(), 4)
orth_poly = pc.orth_ttr(2, Z)
#end

nodes, weights = pc.quadgen(3, Z, gaussian=True, sparse=True)
print len(weights)
# 137
#end

mapped_nodes = Q.inv(Z.fwd(nodes))

t = linspace(0,10,100)
evaluations = [model_solver(t, q) for q in mapped_nodes.T]
#end

model_approx = pc.fitter_quad(orth_poly, nodes, weights, evaluations)
#end

samples_Z = Z.sample(100)
samples_Y = model_approx(*samples_Z)
E = pc.E(model_approx, Z)
#end

t = linspace(0,10,100)
plot(t, samples_Y, "k")
plot(t, E, "w-", lw=3)
plot(t, E, "k--", lw=3)

axis([0,10,0,4])
xticks(range(11))
xlabel("time $t$")
ylabel(r"Approx model $\hat y(t, q)$")
savefig("intro1.pdf")

samples_Q = Q.sample(100, method="H")
#end

model_solver = pc.lazy_eval(model_solver)
t = linspace(0,10,100)
samples_Y = [model_solver(t,q) for q in samples_Q.T]
model_solver.save("evaluations.data")
#end

model_solver.load("evaluations.data")
#end

model_approx = pc.fitter_lls(orth_poly, samples_Q, samples_Y)
#end

print len(model_solver) # unique evaluations
# 100

samples_Z = Z.sample(200, method="H")
samples_Q = Q.inv(Z.fwd(samples_Z))
t = linspace(0,10,100)
samples_u = [model_solver(t, q) for q in samples_Q.T]
model_approx = pc.fitter_lls(orth_poly, samples_Z, samples_u)

print len(model_solver)
# 200
#end

subplot(121)
xlabel("Time $t$")
ylabel(r"Expected value $\mu(t)$")

subplot(122)
plot(t, pc.Var(model_approx, Z), "k")
xlabel("Time $t$")
ylabel(r"Variance $\sigma^2\!(t)$")

savefig("intro2.pdf")

samples_Q = Q.sample(10**5)
samples_u = model_approx(*samples_Q)
#end
