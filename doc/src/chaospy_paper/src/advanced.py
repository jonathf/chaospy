# you mistype something
#end

from pylab import *
import current as pc

rc("figure", figsize=[8.,4.])
rc("figure.subplot", left=.08, top=.95, right=.95, bottom=.12)
seed(1000)

U1 = pc.Uniform(0,1)
U2 = pc.Uniform(0,1)
add = U1+U2
Q = pc.J(U1, add)
#end

def cdf(self, q, G):

    if "par1" in G.K:
        num = G.K["par1"]
        dist = G.D["par2"]

    else:
        num = G.K["par2"]
        dist = G.D["par1"]

    return G(q-num, dist)
#end

def bnd(self, x, G):

    if "par1" in G.K:
        num = G.K["par1"]
        dist = G.D["par2"]

    else:
        num = G.K["par2"]
        dist = G.D["par1"]

    lo,up = G(x-num, dist)
    out = lo+num, up+num
    return out
#end

def val(self, G):
    if len(G.K)==2:
        return G.K["par1"]+G.K["par2"]
    return self
#end

Addition = pc.construct(
    cdf=cdf,
    bnd=bnd,
    val=val,
    advance=True,
    defaults=dict(par1=0., par2=0.))

add = Addition(par1=U1, par2=U2)
Q = pc.J(U1, add)
#end

subplot(121)
R = Q.sample(1000)
scatter(R[0], R[1], marker="s", color="k", alpha=.7)
xlabel(r"$Q_0$")
ylabel(r"$Q_1$")
axis([0,1,0,2])
title("Scatter")

subplot(122)
s,t = meshgrid(linspace(0,1,200), linspace(0,2,200))
contourf(s,t,Q.pdf([s,t]), 30)
xlabel(r"$q_0$")
ylabel(r"$q_1$")
title("Probability Density")

subplots_adjust(bottom=0.12, right=0.85, top=0.9)
cax = axes([0.9, 0.1, 0.03, 0.8])
colorbar(cax=cax)

savefig("advanced.pdf"); clf()
