from numpy import *
import current as pc

x,y = pc.variable(2)
print x
# q0
#end

polys = pc.Poly([1, x, x*y])
print polys
#  [1, q0, q0q1]
#end

print pc.basis(4)
#  [1, q0, q0^2, q0^3, q0^4]
#end

print pc.basis(1, 2, dim=2)
#  [q0, q1, q0^2, q0q1, q1^2]
#end

print pc.basis(1, [1, 2])
#  [q0, q1, q0q1, q1^2, q0q1^2]
#end

print pc.basis(1, 2, dim=2, sort="GRI")
#  [q0^2, q0q1, q1^2, q0, q1]
#end

poly = pc.Poly([1, x**2, x*y])
print poly(2, 3)
#  [1 2 6]
print poly(q1=3, q0=2)
#  [1 2 6]
#end

print poly(2, [1,2,3,4])
#  [[1 1 1 1]
#   [4 4 4 4]
#   [2 4 6 8]]
#end

print poly(2)
#  [1, 4, 2q1]
print poly(ma.masked, 2)
#  [1, q0^2, 2q0]
print poly(q1=2)
#  [1, q0^2, 2q0]
#end

print poly(y, x)
#  [1, q1^2, q0q1]
#end

print poly(q1=y**3-1)
#  [1, q0^2, q0q1^3-q0]
#end
