from pylab import *
import current as pc

rc("figure", figsize=[8.,4.])
rc("figure.subplot", left=.08, top=.93, right=.98)
seed(1000)

dist = pc.Iid(pc.Normal(), 2)
nodes, weights = pc.quadgen(1, dist, rule="G")
print nodes
#  [[-1. -1.  1.  1.]
#   [-1.  1. -1.  1.]]
print weights
#  [ 0.25  0.25  0.25  0.25]
#end

dist = pc.Iid(pc.Uniform(), 2)
nodes1, weights1 = pc.quadgen([3,5], dist, rule="C", sparse=True)
print len(weights1)
#  105

nodes2, weights2 = pc.quadgen([3,5], dist, rule="C", growth=True)
print len(weights2)
#  297
#end

subplot(122)
scatter(nodes2[0], nodes2[1], marker="s",
        s=50*sqrt(weights2),
        alpha=.7, color="k")
xlabel(r"$q_0$")
ylabel(r"$q_1$")
title("Smolyak Sparsegrid")
axis([0,1,0,1])

subplot(121)
pos = weights1>=0
n0, n1 = nodes1[:,pos]
scatter(n0, n1, marker="s",
        s=50*sqrt(weights1[pos]),
        alpha=.7, color="k")
n0, n1 = nodes1[:,True-pos]
scatter(n0, n1, marker="D",
        s=50*sqrt(-weights1[True-pos]),
        alpha=.7, color="r")
xlabel(r"$q_0$")
ylabel(r"$q_1$")
title("Full Tensor product rule")
axis([0,1,0,1])

savefig("mvquad.pdf"); clf()

nodes, weights = pc.quadgen([1,2], dist, rule="G")
print nodes
#  [[-1.         -1.         -1.          1.          1.          1.        ]
#   [-1.73205081  0.          1.73205081 -1.73205081  0.          1.73205081]]
print weights
#  [ 0.08333333  0.33333333  0.08333333  0.08333333  0.33333333  0.08333333]
#end

dist = pc.MvNormal([0,0], [[1,0.5],[0.5,1]])
nodes, weights = pc.quadgen(2, dist, rule="E")
print nodes
#  [[ 0.21132487  0.21132487  0.21132487  0.78867513  0.78867513  0.78867513]
#   [ 0.11270167  0.5         0.88729833  0.11270167  0.5         0.88729833]]
print weights
#  [ 0.13888889  0.22222222  0.13888889  0.13888889  0.22222222  0.13888889]
#end

dist = pc.Iid(pc.Gamma(2), 2)
nodes, weights = pc.quadgen([1,2], dist, rule="G", sparse=True)
print nodes
#  [[ 2.          1.26794919  2.          4.73205081  2.          1.26794919
#     2.          4.73205081  2.        ]
#   [ 0.93582223  1.26794919  1.26794919  1.26794919  3.30540729  4.73205081
#     4.73205081  4.73205081  7.75877048]]
print weights
#  [ 0.58868148  0.62200847 -0.78867513  0.16666667  0.39121606  0.16666667
#   -0.21132487  0.0446582   0.02010246]
#end

dist = pc.Iid(pc.Beta(2, 2), 2)
nodes, weights = pc.quadgen(4, dist, rule="K")
print nodes
#  [[ 0.28714073  0.43293108  0.56706892  0.71285927]
#   [ 0.43293108  0.71285927  0.28714073  0.56706892]]
print weights
#  [ 0.25  0.25  0.25  0.25]
#end

dist = pc.Iid(pc.Uniform(0,1), 2)
nodes, weights = pc.quadgen(2, dist, rule="C", sparse=True)
print nodes
#  [[ 0.14644661  0.85355339  0.          0.5         1.          0.          0.5
#     1.        ]
#   [ 0.          0.          0.14644661  0.5         0.5         0.85355339
#     1.          1.        ]]
print weights
#  [ 0.18860511  0.18860511  0.18860511  0.15717092  0.03929273  0.18860511
#    0.03929273  0.00982318]
#end

dist = pc.Normal()
print dist.sample(2, "G")
#  [-1.73205081  0.          1.73205081]
#end

def trapezoidal(a, b):
    """Trapezoidal generator on interval [a,b]"""

    def integrate(N):
        if N==0: # special case
            return [a], [b-a]
        nodes = np.linspace(a, b, N+1)
        weights = (b-a)*np.ones(N+1)/N
        weights[0] *= 0.5
        weights[-1] *= 0.5
        return nodes, weights

    return integrate

quad = trapezoidal(-1,1)
nodes, weights = quad(3)
print nodes
#  [-1.         -0.33333333  0.33333333  1.        ]
print weights
#  [ 0.33333333  0.66666667  0.66666667  0.33333333]
#end

mv_trapezoidal = pc.rule_generator(
        trapezoidal(-1,1),
        trapezoidal(0,1),
        trapezoidal(0,2))
nodes, weights = mv_trapezoidal(1)
print nodes
#  [[-1. -1. -1. -1.  1.  1.  1.  1.]
#   [ 0.  0.  1.  1.  0.  0.  1.  1.]
#   [ 0.  2.  0.  2.  0.  2.  0.  2.]]
print weights
#  [ 0.5  0.5  0.5  0.5  0.5  0.5  0.5  0.5]
#end

nodes, weights = mv_trapezoidal([1,1,2], sparse=True)
print nodes
#  [[-1.  1. -1. -1. -1.  1. -1.]
#   [ 0.  0.  1.  0.  0.  0.  1.]
#   [ 0.  0.  0.  1.  2.  2.  2.]]
print weights
#  [-1.  1.  1.  2. -1.  1.  1.]
#end

