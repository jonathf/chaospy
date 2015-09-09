from pylab import *
import current as pc

dist = pc.Iid(pc.Gamma(1), 3)
x,y,z = pc.variable(3)
poly = x + y + z + x*y*z
print poly
#  q0q1q2+q2+q1+q0
#end

print pc.E_cond(poly, z, dist)
#  2.0q2+2.0
#end

print pc.E_cond(poly, [x,y], dist)
#  q0q1+q1+q0+1.0
#end
