from pylab import *
import current as pc

dist = pc.Iid(pc.Normal(), 2)
#end

nodes, weights = pc.quadgen(2, dist, rule="G")
print nodes
#  [[-1.73205081 -1.73205081 -1.73205081  0.          0.
#     0.          1.73205081  1.73205081  1.73205081]
#   [-1.73205081  0.          1.73205081 -1.73205081  0.
#     1.73205081 -1.73205081  0.          1.73205081]]
print weights
#  [ 0.02777778  0.11111111  0.02777778  0.11111111  0.44444444
#    0.11111111  0.02777778  0.11111111  0.02777778]
#end

orth = pc.orth_ttr(2, dist)
print orth
#  [1.0, q1, q0, q1^2-1.0, q0q1, q0^2-1.0]
#end

def model_solver(q):
    return [q[0]*q[1], q[0]*e**-q[1]+1]
solves = [model_solver(q) for q in nodes.T]
#end

approx = pc.fitter_quad(orth, nodes, weights, solves)
print pc.around(approx, 14)
#  [q0q1, -1.58058656357q0q1+1.63819248006q0+1.0]
#end


orth, norms = pc.orth_ttr(2, dist, retall=True)
approx2 = pc.fitter_quad(orth, nodes, weights, solves, norms=norms)
#end

print np.max(abs(approx-approx2).coeffs(), -1)
#  [  2.44249065e-15   3.77475828e-15]
#end

nodes = dist.sample(12, "M")
print nodes
#  [[ 1.3074948   1.1461811   0.48907479 -0.99729764 -2.04207273  0.3737412
#    -0.17126603  0.40045745  0.25582169 -1.50683751 -0.47799293  1.21890579]
#   [ 0.61939522  1.72367491 -0.55533514 -0.00905152 -0.87071076 -0.04532524
#    -0.95908033 -1.5433918  -1.10189542  1.19303123 -0.85594892 -0.97358421]]
#end

def model_solver(q):
    return [q[0]*q[1], q[0]*e**-q[1]+1]
solves = [model_solver(q) for q in nodes.T]

approx = pc.fitter_lr(orth, nodes, solves)
print pc.around(approx, 14)
#  [q0q1, 0.161037230451q1^2-1.23004736008q0q1+0.152745901925q0^2+
#           0.0439971157359q1+1.21399892993q0+0.86841679007]
#end

approx = pc.fitter_lr(orth, nodes, solves,
        rule="OMP", n_nonzero_coefs=1)
print approx
#  [q0q1, 1.52536467971q0]
#end
