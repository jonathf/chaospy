# you mistype something
#end

from pylab import *
import current as pc

rc("figure", figsize=[8.,4.])
rc("figure.subplot", left=.08, top=.95, right=.95)
seed(1000)

Uniform_min = pc.construct(
    cdf=lambda self,q,a,b: (q-a)/(b-a),
    bnd=lambda self,a,b: (a,b))

Q = Uniform_min(a=0,b=1)
#end

Uniform = pc.construct(
    cdf=lambda self,q,a,b: (q-a)/(b-a),
    bnd=lambda self,a,b: (a,b),
    pdf=lambda self,q,a,b: 1./(b-a),
    ppf=lambda self,u,a,b: u*(b-a)+a,
    mom=lambda self,k,a,b: (b**(k+1)-a**(k+1))/(k+1)/(b-a),
    ttr=lambda self,k,a,b: (0.5*(a+b),n*n*(b-a)**2/(16*n*n-4)),
    defaults=dict(a=0., b=1.),
    str=lambda self,a,b: "U(%s,%s)" % (a,b),
    doc="An uniform distribution on the interval [a,b]")

Q = Uniform(a=0, b=2)
#end

U1 = Uniform(a=1, b=2)
U2 = Uniform(a=0, b=U1)
U = pc.J(U1, U2)
#end

print U.fwd([[1.1, 1.5, 1.9],[1, 1, 1]])
#  [[ 0.1         0.5         0.9       ]
#   [ 0.90909091  0.66666667  0.52631579]]
#end

s,t = meshgrid(linspace(1,2,200), linspace(0, 2, 200))

subplot(121)
contourf(s,t,U.fwd([s,t])[0], 50)
xlabel(r"$u_0$")
ylabel(r"$u_1$")
title(r"$F_{U_0}(u_0)$")

subplot(122)
contourf(s,t,U.fwd([s,t])[1], 50)
xlabel(r"$u_0$")
ylabel(r"$u_1$")
title(r"$F_{U_1\mid U_0}(u_1\mid u_0)$")

subplots_adjust(bottom=0.1, right=0.85, top=0.9)
cax = axes([0.9, 0.1, 0.03, 0.8])
colorbar(cax=cax)

savefig("dist.pdf"); clf()
