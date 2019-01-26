"""Plot figures for multivariate distribution section."""
from pylab import *
import chaospy as cp

def plot_figures():
    """Plot figures for multivariate distribution section."""
    rc("figure", figsize=[8.,4.])
    rc("figure.subplot", left=.08, top=.95, right=.98)
    rc("image", cmap="gray")
    seed(1000)

    Q1 = cp.Gamma(2)
    Q2 = cp.Normal(0, Q1)
    Q = cp.J(Q1, Q2)
    #end

    subplot(121)
    s,t = meshgrid(linspace(0,5,200), linspace(-6,6,200))
    contourf(s,t,Q.pdf([s,t]),50)
    xlabel("$q_1$")
    ylabel("$q_2$")
    subplot(122)
    Qr = Q.sample(500)
    scatter(*Qr, s=10, c="k", marker="s")
    xlabel("$Q_1$")
    ylabel("$Q_2$")
    axis([0,5,-6,6])

    savefig("multivariate.png"); clf()

    Q2 = cp.Gamma(1)
    Q1 = cp.Normal(Q2**2, Q2+1)
    Q = cp.J(Q1, Q2)
    #end

    subplot(121)
    s,t = meshgrid(linspace(-4,7,200), linspace(0,3,200))
    contourf(s,t,Q.pdf([s,t]),30)
    xlabel("$q_1$")
    ylabel("$q_2$")
    subplot(122)
    Qr = Q.sample(500)
    scatter(*Qr)
    xlabel("$Q_1$")
    ylabel("$Q_2$")
    axis([-4,7,0,3])

    savefig("multivariate2.png"); clf()


if __name__ == "__main__":
    plot_figures()
