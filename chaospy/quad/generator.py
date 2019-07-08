"""
Generator for combining one dimensional rules into multivariate versions.
"""
import numpy

import chaospy.quad


def rule_generator(*funcs):
    """
    Constructor for creating multivariate quadrature generator.

    Args:
        funcs (:py:data:typing.Callable):
            One dimensional integration rule where each rule returns
            ``abscissas`` and ``weights`` as one dimensional arrays. They must
            take one positional argument ``order``.

    Returns:
        (:py:data:typing.Callable):
            Multidimensional integration quadrature function that takes the
            arguments ``order`` and ``sparse``. The argument ``sparse`` is used
            to select for if Smolyak sparse grid is used.

    Example:
        >>> clenshaw_curtis = lambda order: chaospy.quad_clenshaw_curtis(
        ...         order, lower=-1, upper=1, growth=True)
        >>> gauss_legendre = lambda order: chaospy.quad_gauss_legendre(
        ...         order, lower=0, upper=1)
        >>> quad_func = chaospy.rule_generator(clenshaw_curtis, gauss_legendre)
        >>> abscissas, weights = quad_func(1)
        >>> print(numpy.around(abscissas, 4))
        [[-1.     -1.      0.      0.      1.      1.    ]
         [ 0.2113  0.7887  0.2113  0.7887  0.2113  0.7887]]
        >>> print(numpy.around(weights, 4))
        [0.1667 0.1667 0.6667 0.6667 0.1667 0.1667]
    """
    dim = len(funcs)
    tensprod_rule = create_tensorprod_function(funcs)
    assert hasattr(tensprod_rule, "__call__")

    mv_rule = create_mv_rule(tensprod_rule, dim)
    assert hasattr(mv_rule, "__call__")
    return mv_rule


def create_tensorprod_function(funcs):
    """Combine 1-D rules into multivariate rule using tensor product."""
    dim = len(funcs)

    def tensprod_rule(order):
        """Tensor product rule."""
        order = order*numpy.ones(dim, int)
        values = [funcs[idx](order[idx]) for idx in range(dim)]

        abscissas = [numpy.array(_[0]).flatten() for _ in values]
        abscissas = chaospy.quad.combine(abscissas).T

        weights = [numpy.array(_[1]).flatten() for _ in values]
        weights = numpy.prod(chaospy.quad.combine(weights), -1)

        return abscissas, weights

    return tensprod_rule


def create_mv_rule(tensorprod_rule, dim):
    """Convert tensor product rule into a multivariate quadrature generator."""
    def mv_rule(order, sparse=False):
        """
        Multidimensional integration rule.

        Args:
            order (int, numpy.ndarray) : order of integration rule. If numpy.ndarray,
                order along each axis.
            sparse (bool) : use Smolyak sparse grid.

        Returns:
            (numpy.ndarray, numpy.ndarray) abscissas and weights.
        """
        if sparse:
            order = numpy.ones(dim, dtype=int)*order
            return chaospy.quad.sparse_grid(tensorprod_rule, order)
        return tensorprod_rule(order)

    return mv_rule
