from pylab import *
import current as pc

dist_main = pc.MvNormal([0,0], [[1,.5],[.5,1]])
#end

dist_aux = pc.Iid(pc.Normal(), 2)
#end

orth, norms = pc.orth_ttr(2, dist_aux, retall=True)
print orth
#  [1.0, q1, q0, q1^2-1.0, q0q1, q0^2-1.0]
#end

nodes_aux, weights = pc.quadgen(3, dist_aux, rule="G")
#end

function = lambda q: q[0]*e**-q[1]+1
#end

nodes_main = dist_main.inv(dist_aux.fwd( nodes_aux ))
solves = [function(q) for q in nodes_main.T]
#end

approx = pc.fitter_quad(orth, nodes_aux, weights, solves,
        norms=norms)
print pc.E(approx, dist_aux)
#  0.175824752014
#end
