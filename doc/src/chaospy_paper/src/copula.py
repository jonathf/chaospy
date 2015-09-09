from pylab import *
import current as pc

rc("figure", figsize=[8.,4.])
rc("figure.subplot", left=.08, top=.95, right=.95, bottom=.12)
pc.seed(1000)

Q = pc.Iid(pc.Uniform(0,4), 2)
C = pc.Frank(Q, 1.5)
#end

subplot(121)
R = C.sample(1000)
scatter(R[0], R[1], marker="s", color="k", alpha=.7)
xlabel(r"$Q_0$")
ylabel(r"$Q_1$")
xticks(range(5))
yticks(range(5))
axis([0,4,0,4])
title("Scatter")

subplot(122)
s,t = meshgrid(linspace(0,4,100), linspace(0,4,100))
contourf(s,t,C.pdf([s,t]), 30)
xlabel(r"$q_0$")
ylabel(r"$q_1$")
xticks(range(5))
yticks(range(5))
title("Probability Density")

subplots_adjust(bottom=0.12, right=0.85, top=0.9)
cax = axes([0.9, 0.1, 0.03, 0.8])
colorbar(cax=cax)

savefig("copula.pdf"); clf()
