import numpy
from matplotlib import pyplot
import chaospy

pyplot.rc("figure", figsize=[3, 2])

COLOR1 = "steelblue"
COLOR2 = "slategray"


def save(name):
    pyplot.axis("off")
    pyplot.savefig(
        f"./{name}.png",
        bbox_inches="tight",
        transparent=True,
    )
    pyplot.clf()


def make_distribution():

    t = numpy.linspace(-1, 1, 100)
    dist = chaospy.Normal(0, 0.5)

    pyplot.fill_between(t, 0, dist.pdf(t), alpha=0.3, color=COLOR1)
    pyplot.plot(t, dist.pdf(t), COLOR1, lw=4)
    pyplot.fill_between(t, 0, dist.cdf(t), alpha=0.3, color=COLOR2)
    pyplot.plot(t, dist.cdf(t), COLOR2, lw=4)

    save("distribution")



def make_polynomial():
    q0 = chaospy.variable()
    poly = 1.2*q0*(q0-1.8)*(q0+1.8)
    t = numpy.linspace(-2, 2, 100)

    t0 = numpy.linspace(-2, 0, 100)
    pyplot.fill_between(t0, 0, poly(t0), alpha=0.3, color=COLOR1)
    pyplot.plot(t0, poly(t0), COLOR1, lw=4)

    t0 = numpy.linspace(0, 2, 100)
    pyplot.fill_between(t0, poly(t0), 0, alpha=0.3, color=COLOR2)
    pyplot.plot(t0, poly(t0), COLOR2, lw=4)

    save("polynomial")


def make_sampling():
    dist = chaospy.Iid(chaospy.Uniform(0, 1), 2)
    samples = dist.sample(20, rule="sobol")
    size = 80

    pyplot.scatter(*samples[:, ::2], s=size, lw=3, color="w", edgecolors=COLOR1)
    pyplot.scatter(*samples[:, ::2], s=size, color=COLOR1, alpha=0.6)
    pyplot.scatter(*samples[:, 1::2], s=size, lw=3, color="w", edgecolor=COLOR2)
    pyplot.scatter(*samples[:, 1::2], s=size, color=COLOR2, alpha=0.6)

    save("sampling")


def make_quadrature():
    dist = chaospy.Iid(chaospy.Uniform(0, 1), 2)

    nodes, weights = chaospy.generate_quadrature(2, dist, growth=True, rule="fejer", sparse=True)
    size = (weights*500).astype(int)
    indices = weights < 0

    pyplot.scatter(*nodes[:, indices], s=-size[indices], lw=3, color="w", edgecolors=COLOR2)
    pyplot.scatter(*nodes[:, indices], s=-size[indices], color=COLOR2, alpha=0.6)
    pyplot.scatter(*nodes[:, ~indices], s=size[~indices], lw=3, color="w", edgecolor=COLOR1)
    pyplot.scatter(*nodes[:, ~indices], s=size[~indices], color=COLOR1, alpha=0.6)

    save("quadrature")


def make_orthogonality():

    t = numpy.linspace(-2, 2, 200)
    q0 = chaospy.variable()
    poly1 = (q0-1.2)*(q0+1.2)
    poly2 = -(q0-1.2)*(q0+1.2)

    t0 = numpy.linspace(-2, -1.2)
    pyplot.fill_between(t0, poly1(t0), poly2(t0), color=COLOR1, alpha=0.3)
    t0 = numpy.linspace(1.2, 2)
    pyplot.fill_between(t0, poly1(t0), poly2(t0), color=COLOR1, alpha=0.3)
    pyplot.plot(t, poly1(t), COLOR1, lw=4)
    t0 = numpy.linspace(-1.2, 1.2)
    pyplot.fill_between(t, poly1(t), poly2(t), color=COLOR2, alpha=0.3)
    pyplot.plot(t, poly2(t), COLOR2, lw=4)

    save("orthogonality")


def make_recurrence():
    dist = chaospy.Iid(chaospy.Uniform(0, 1), 2)
    samples1 = numpy.array([[0, 1, 2, 3], [0, 1, 2, 3]])
    samples2 = numpy.array([[0, 0, 0, 1, 1, 2], [1, 2, 3, 2, 3, 3]])
    size = 100

    pyplot.plot([.16, .84], [2, 2], COLOR2, lw=4)
    pyplot.plot([.16, .84], [3, 3], COLOR2, lw=4)
    pyplot.plot([1.16, 1.84], [3, 3], COLOR2, lw=4)
    pyplot.scatter(*samples1, s=size, lw=3, color="w", edgecolors=COLOR1)
    pyplot.scatter(*samples1, s=size, color=COLOR1, alpha=0.6)
    pyplot.scatter(*samples2, s=size, lw=3, color="w", edgecolor=COLOR2)
    pyplot.scatter(*samples2, s=size, color=COLOR2, alpha=0.6)

    save("recurrence")


def make_descriptive():

    numpy.random.seed(1234)
    dist1 = chaospy.Normal(0, 1)
    samples1 = dist1.sample(40)
    dist2 = chaospy.Exponential()
    samples2 = dist2.sample(20)

    x = y = numpy.linspace(0, 2*numpy.pi, 200)
    x, y = numpy.cos(x), numpy.sin(y)

    pyplot.pie([0.5], colors=[COLOR1], radius=1, normalize=False,
               center=(-0.3, 0.3), startangle=45,
               wedgeprops={"width": 0.5, "alpha": 0.5, "lw": 4})
    pyplot.plot(x-0.3, y+0.3, COLOR1, lw=4)
    pyplot.plot(x/2-0.3, y/2+0.3, COLOR1, lw=4)

    pyplot.bar([0, 0.6], [0.5, 1], bottom=[-0.6, -0.6],
               width=0.5, yerr=[0.2, 0.3], color=COLOR2)

    save("descriptive")


if __name__ == "__main__":
    make_distribution()
    make_polynomial()
    make_sampling()
    make_quadrature()
    make_orthogonality()
    make_recurrence()
    make_descriptive()
