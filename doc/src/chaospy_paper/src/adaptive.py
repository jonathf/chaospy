from pylab import *
import current as pc

rc("figure", figsize=[8.,4.])
rc("figure.subplot", left=.08, top=.95, right=.98)
seed(1000)

dist = pc.Iid(pc.Uniform(), 5)

def model_solver(q):
    return q[0]*e**-q[1]/(q[2]+1) + sin(q[3])
model_solver = pc.lazy_eval(model_solver)
#end

current = array([1,1,1,1,1])
current_error = inf
#end

for step in range(10):

    for direction in eye(len(dist), dtype=int):
        #end

        orth = pc.orth_ttr(current+direction, dist)
        nodes, weights = pc.quadgen(current+direction, dist,
                rule="C", growth=True)
        vals = [model_solver(q) for q in nodes.T]
        #end

        residuals = pc.cross_validate(orth, nodes, vals, folds=5)
        error = sum(residuals**2)
        #end

        if error < current_error:
            current_error = error
            proposed_dir = current+direction
        #end

    current = proposed_dir
    #end

print current
#  [ 1  3  4  6  1]
#end

orth, norms = pc.orth_ttr(current, dist, retall=True)
nodes, weights = pc.quadgen(current, dist, rule="C", growth=True)
vals = [model_solver(q) for q in nodes.T]
approx = pc.fitter_quad(orth, nodes, weights, vals, norms=norms)

print pc.E(approx, dist)
#  0.678773985695
print pc.Var(approx, dist)
#  0.0854146399504
#end
