import numpy as np

import chaospy.dist


def samplegen(order, domain, rule="S", antithetic=None,
        verbose=False):
    """
    Sample generator.

    Interpretation of the domain argument:

    +------------+------------------------------------------------------------+
    | Value      | Interpretation                                             |
    +============+============================================================+
    | Dist       | Mapped to distribution domain using inverse Rosenblatt.    |
    +------------+------------------------------------------------------------+
    | int        | No mapping, but sets the number of dimension.              |
    +------------+------------------------------------------------------------+
    | array_like | Stretch samples such that they are in domain[0], domain[1] |
    +------------+------------------------------------------------------------+

    Intepretation of the rule argument:

    +------+---------------------+--------+
    | Key  | Name                | Nested |
    +======+=====================+========+
    | "C"  | Chebyshev nodes     | no     |
    +------+---------------------+--------+
    | "NC" | Nested Chebyshev    | yes    |
    +------+---------------------+--------+
    | "K"  | Korobov             | no     |
    +------+---------------------+--------+
    | "R"  | (Pseudo-)Random     | no     |
    +------+---------------------+--------+
    | "RG" | Regular grid        | no     |
    +------+---------------------+--------+
    | "NG" | Nested grid         | yes    |
    +------+---------------------+--------+
    | "L"  | Latin hypercube     | no     |
    +------+---------------------+--------+
    | "S"  | Sobol               | yes    |
    +------+---------------------+--------+
    | "H"  | Halton              | yes    |
    +------+---------------------+--------+
    | "M"  | Hammersley          | yes    |
    +------+---------------------+--------+

    Args:
        order (int) : Sample order.
        domain (Dist, int, array_like) : Defines the space where the samples
                are generated.
        rule (str) : rule for generating samples, where d is the number of
                dimensions.
        antithetic (array_like, optional) : List of bool. Represents the axes
                to mirror using antithetic variable.

    Examples:
        >>> print(cp.samplegen(3, cp.Normal(), "H"))
        [-2.33441422 -0.74196378  0.74196378  2.33441422]

        >>> cp.seed(1000)
        >>> print(cp.samplegen(3, cp.Normal(), "L"))
        [[ 0.6633974   0.46811863]
         [ 0.27875174  0.05308317]
         [ 0.98757072  0.51017741]
         [ 0.12054785  0.84929862]]
    """
    rule = rule.upper()

    if isinstance(domain, int):
        dim = domain
        trans = lambda x, verbose:x

    elif isinstance(domain, (tuple, list, np.ndarray)):
        domain = np.asfarray(domain)
        if len(domain.shape)<2:
            dim = 1
        else:
            dim = len(domain[0])
        lo,up = domain
        trans = lambda x, verbose: ((up-lo)*x.T + lo).T

    else:
        dist = domain
        dim = len(dist)
        trans = dist.inv

    if not (antithetic is None):

        antithetic = np.array(antithetic, dtype=bool).flatten()
        if antithetic.size==1 and dim>1:
            antithetic = np.repeat(antithetic, dim)

        N = np.sum(1*np.array(antithetic))
        order_,order = order,int(order*2**-N+1*(order%2!=0))
        trans_ = trans
        trans = lambda X, verbose: \
                trans_(antithetic_gen(X, antithetic)[:,:order_])

    if rule=="C":
        X = chaospy.dist.samplers.chebyshev(dim, order)

    elif rule=="NC":
        X = chaospy.dist.samplers.chebyshev_nested(dim, order)

    elif rule=="K":
        X = chaospy.dist.samplers.korobov(dim, order)

    elif rule=="R":
        X = np.random.random((dim,order))

    elif rule=="RG":
        X = chaospy.dist.samplers.regular_grid(dim, order)

    elif rule=="NG":
        X = chaospy.dist.samplers.regular_grid_nested(dim, order)

    elif rule=="L":
        X = chaospy.dist.samplers.latin_hypercube(dim, order)

    elif rule=="S":
        X = chaospy.dist.sobol_lib.sobol(dim, order)

    elif rule=="H":
        X = chaospy.dist.samplers.halton(dim, order)

    elif rule=="M":
        X = chaospy.dist.samplers.hammersley(dim, order)

    else:
        raise KeyError("rule not recognised")

    X = trans(X, verbose=verbose)

    return X
