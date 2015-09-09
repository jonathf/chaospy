from pylab import *
import current as pc

rc("figure", figsize=[8.,4.])
rc("figure.subplot", left=.08, top=.95, right=.98)
seed(1000)


nodes, weights = pc.quadgen(2, [0,1])
print nodes
#  [[ 0.   0.5  1. ]]
print weights
#  [ 0.16666667  0.66666667  0.16666667]
#end

dist = pc.Beta(2,2)
nodes, weights = pc.quadgen(3, dist)
print nodes
#  [[ 0.14644661  0.5         0.85355339]]
print weights
#  [ 0.2  0.6  0.2]
#end

nodes, weights = pc.quadgen(2, dist, rule="G")
print nodes
#  [[ 0.17267316  0.5         0.82732684]]
print weights
#  [ 0.23333333  0.53333333  0.23333333]
#end

nodes, weights = pc.quadgen(1, [0,1], rule="E", composite=2)
print nodes
#  [[ 0.10566243  0.39433757  0.60566243  0.89433757]]
print weights
#  [ 0.25  0.25  0.25  0.25]
#end

nodes, weights = pc.quadgen(1, [0,1], rule="E", composite=[0.2])
print nodes
#  [[ 0.04226497  0.15773503  0.36905989  0.83094011]]
print weights
#  [ 0.1  0.1  0.4  0.4]
#end

dist = pc.Uniform(0,1)
print dist.mom([0,1,2])
#  [ 1.          0.5         0.33333333]
#end

def mom(self, k):
    return 1./(1+k)
dist.addattr(mom=mom)
#end


def pdf(self, x):
    out = 9/16.*(x+1)*(x<1/3.)
    out += 3/4.*(x>=1/3.)
    return out

def cdf(self, x):
    out = 9/32.*(x+1)**2*(x<1/3.)
    out += (3*x+1)/4.*(x>=1/3.)
    return out

def bnd(self):
    return -1,1

MyDist = pc.construct(
    pdf=pdf, cdf=cdf, bnd=bnd)
#end

dist = MyDist()
print dist.mom(1, order=100)
#  0.277778240534
#end

print dist.mom(1, order=5, composite=1/3.)
#  0.277777777778
#end

mom = pc.momgen(
    order=100,
    domain=dist,
    composite=1/3.)

dist.addattr(mom=mom)
print dist.mom([1,2,3])
#  [ 0.27777778  0.2962963   0.15925926]
#end
