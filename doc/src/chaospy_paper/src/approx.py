# you mistype something
#end

from pylab import *
import current as pc

rc("figure", figsize=[8.,4.])
rc("figure.subplot", left=.09, top=.93, right=.95, wspace=.24)
seed(1000)

cdf = lambda self, q: e**q/(1+e**q)
bnd = lambda self: (-30,30)
Cauchy = pc.construct(cdf=cdf, bnd=bnd)
Q = Cauchy()
#end

print Q.inv([0.1, 0.2, 0.3])
#  [-2.19722267 -1.38626822 -0.84729574]
print Q.pdf([-1, 0, 1])
#  [ 0.19661217  0.25000002  0.19661273]
print Q.sample(4)
#  [ 0.63485527 -2.04053293  2.95040998 -0.07126454]
#end

print Q.inv([0.1, 0.2, 0.3], tol=1e-3, maxiter=10)
#  [-2.19503823 -1.38626822 -0.8440303 ]
print Q.pdf([-1, 0, 1], step=1e-1)
#  [ 0.20109076  0.24979187  0.20109076]
#end

subplot(121)
hist(Q.sample(10**3), bins=linspace(-10,10,31), normed=1, color="k", alpha=.8)
xlim(-10,10)
xlabel(r"$Q$")
ylabel("Relative Frequency")
title("Histogram")

subplot(122)
t = linspace(-10,10,200)
plot(t,Q.pdf(t), "k", lw=2)
xlabel(r"$Q$")
ylabel(r"Probability")
title(r"Probability Density")

savefig("approx.pdf"); clf()

