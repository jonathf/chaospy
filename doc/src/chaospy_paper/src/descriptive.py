from numpy import *
import current as pc

dist = pc.Normal()
orths = []
for poly in pc.basis(0, 4, dim=1):

    for orth in orths:

        coef = pc.E(orth*poly, dist)/pc.E(orth**2, dist)
        poly = poly - orth*coef

    orths.append(poly)

orths = pc.Poly(orths)
print orths
#  [1.0, q0, q0^2-1.0, q0^3-3.0q0, q0^4-6.0q0^2+3.0]
#end

orths2 = pc.outer(orths, orths)
print pc.E(orths2, dist)
#  [[  1.   0.   0.   0.   0.]
#   [  0.   1.   0.   0.   0.]
#   [  0.   0.   2.   0.   0.]
#   [  0.   0.   0.   6.   0.]
#   [  0.   0.   0.   0.  24.]]
#end

dist = pc.Gamma(2)
print pc.orth_bert(2, dist)
#  [1.0, q0-2.0, q0^2-6.0q0+6.0]
#end

dist = pc.Uniform(-1,1)
print dist.ttr([0,1,2,3])
#  [[ 0.          0.          0.          0.        ]
#   [-0.          0.33333333  0.26666667  0.25714286]]
#end

dist = pc.Lognormal(0.01)
orths = pc.orth_ttr(2, dist)
print orths
#  [1.0, q0-1.00501252086, q0^2-2.04042818514q0+0.842739860094]
#end

dist = pc.Iid(pc.Gamma(1), 2)
orths = pc.orth_ttr(2, dist)
print orths
#  [1.0, q1-1.0, q0-1.0, q1^2-4.0q1+2.0, q0q1-q1-q0+1.0, q0^2-4.0q0+2.0]
#end

q = pc.variable()
dist = pc.Normal()
print pc.E(q, dist)
#  0.0
#end


