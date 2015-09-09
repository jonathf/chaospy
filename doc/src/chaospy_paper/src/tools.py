from numpy import *
import current as pc

function = lambda q: q[0]*e**q[1]+1
dist = pc.Iid(pc.Normal(), 2)

approx = pc.pcm(function, 2, dist, rule="G")

print pc.around(approx, 14)
#  1.64234906518q0q1+1.64796896005q0+1.0
print pc.E(approx, dist)
#  1.0
print pc.Var(approx, dist)
#  5.41311214518
#end
