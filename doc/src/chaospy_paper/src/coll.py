from numpy import *
import current as pc

def model_solver(q):
    return q[1]*e**q[0]+1
#end

coll_points = array([[0,0,1],[0,1,1]])
#end

basis = pc.basis(0,1,2)
print basis
#  [1, q0, q1]
#end

solves = model_solver(coll_points)
print solves
#  [ 1.          2.          3.71828183]
#end

approx_solver = pc.fitter_lr(basis, coll_points, solves)
print approx_solver
#  q1+1.71828182846q0+1.0
#end

print approx_solver(*coll_points)
#  [ 1.          2.          3.71828183]
#end
