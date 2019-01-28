"""Plot figures for tutorial."""
import chaospy as cp
import numpy
import matplotlib.pyplot as plt
import seaborn


def plot_figures():
    """Plot figures for tutorial."""
    numpy.random.seed(1000)


    def foo(coord, param):
        return param[0] * numpy.e ** (-param[1] * coord)

    coord = numpy.linspace(0, 10, 200)
    distribution = cp.J(cp.Uniform(1, 2), cp.Uniform(0.1, 0.2))

    samples = distribution.sample(50)
    evals = numpy.array([foo(coord, sample) for sample in samples.T])

    plt.plot(coord, evals.T, "k-", lw=3, alpha=0.2)
    plt.xlabel(r"\verb;coord;")
    plt.ylabel(r"function evaluations \verb;foo;")
    plt.savefig("demonstration.png")
    plt.clf()


    samples = distribution.sample(1000, "H")
    evals = [foo(coord, sample) for sample in samples.T]
    expected = numpy.mean(evals, 0)
    deviation = numpy.std(evals, 0)

    plt.fill_between(
        coord, expected-deviation, expected+deviation,
        color="k", alpha=0.3
    )
    plt.plot(coord, expected, "k--", lw=3)
    plt.xlabel(r"\verb;coord;")
    plt.ylabel(r"function evaluations \verb;foo;")
    plt.title("Results using Monte Carlo simulation")
    plt.savefig("results_montecarlo.png")
    plt.clf()


    polynomial_expansion = cp.orth_ttr(8, distribution)
    foo_approx = cp.fit_regression(polynomial_expansion, samples, evals)
    expected = cp.E(foo_approx, distribution)
    deviation = cp.Std(foo_approx, distribution)

    plt.fill_between(
        coord, expected-deviation, expected+deviation,
        color="k", alpha=0.3
    )
    plt.plot(coord, expected, "k--", lw=3)
    plt.xlabel(r"\verb;coord;")
    plt.ylabel(r"function evaluations \verb;foo;")
    plt.title("Results using point collocation method")
    plt.savefig("results_collocation.png")
    plt.clf()


    absissas, weights = cp.generate_quadrature(8, distribution, "C")
    evals = [foo(coord, val) for val in absissas.T]
    foo_approx = cp.fit_quadrature(polynomial_expansion, absissas, weights, evals)
    expected = cp.E(foo_approx, distribution)
    deviation = cp.Std(foo_approx, distribution)

    plt.fill_between(
        coord, expected-deviation, expected+deviation,
        color="k", alpha=0.3
    )
    plt.plot(coord, expected, "k--", lw=3)
    plt.xlabel(r"\verb;coord;")
    plt.ylabel(r"function evaluations \verb;foo;")
    plt.title("Results using psuedo-spectral projection method")
    plt.savefig("results_spectral.png")
    plt.clf()


if __name__ == "__main__":
    plot_figures()
