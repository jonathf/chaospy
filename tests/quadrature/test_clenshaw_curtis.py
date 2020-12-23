"""Test Clenshaw-Curtis quadrature rules."""
import pytest
import numpy
import chaospy

TESTS = [
    # linear 1D tests
    (dict(order=0), numpy.array([[0.5]]), numpy.array([1.])),
    (dict(order=1), numpy.array([[0., 1.]]), numpy.array([0.5, 0.5])),
    (dict(order=2), numpy.array([[0., 0.5, 1. ]]), numpy.array([1/6., 2/3., 1/6.])),
    (dict(order=3), numpy.array([[0.,  0.25, 0.75, 1.  ]]),
     numpy.array([1/18., 4/9., 4/9., 1/18.])),
    (dict(order=10),
     (numpy.cos(numpy.pi*numpy.arange(10, -1, -1)/10.)*.5+.5)[numpy.newaxis],
     numpy.array([1/198., 0.04728953, 0.09281761, 0.12679417, 0.14960664, 0.15688312,
                  0.14960664, 0.12679417, 0.09281761, 0.04728953, 1/198.])),

    # exponential growth
    (dict(order=0, growth=True), numpy.array([[0.5]]), numpy.array([1.])),
    (dict(order=1, growth=True), numpy.array([[0., 0.5, 1.]]), numpy.array([1/6., 2/3., 1/6.])),
    (dict(order=2, growth=True),
     numpy.array([[0., 0.14644661, 0.5, 0.85355339, 1. ]]),
     numpy.array([1/30., 4/15., 2/5., 4/15., 1/30.])),

    # segments
    (dict(order=4, segments=2), numpy.array([[0, 1/4., 1/2., 3/4., 1]]),
     numpy.array([1/12., 1/3., 1/6., 1/3., 1/12.])),

    # # domains
    (dict(order=2, domain=chaospy.Uniform(0, 1)), numpy.array([[0., 0.5, 1.]]), numpy.array([1/6., 2/3., 1/6.])),
    (dict(order=2, domain=chaospy.Uniform(-1, 1)), numpy.array([[-1., 0., 1.]]), numpy.array([1/6., 2/3., 1/6.])),
    (dict(order=2, domain=(-1, 1)), numpy.array([[-1., 0., 1.]]), numpy.array([2/6., 4/3., 2/6.])),
    (dict(order=2, domain=chaospy.Normal(0, 1)),
     numpy.array([[chaospy.Normal().lower.item(), 0., chaospy.Normal().upper.item()]]),
     numpy.array([0., 1., 0.])),
]


@pytest.fixture(params=TESTS)
def test_setup(request):
    return request.param

def test_clenshaw_curtis(test_setup):
    """Test Clenshaw Curtis quadrature."""
    kwargs, nodes0, weights0 = test_setup
    nodes, weights = chaospy.quad_clenshaw_curtis(**kwargs)
    assert nodes.shape == nodes0.shape, nodes
    assert numpy.allclose(nodes, nodes0)
    assert numpy.allclose(weights, weights0)
    assert weights.shape == weights0.shape
    if kwargs.get("domain", None) in (0, 1):
        assert numpy.isclose(numpy.sum(weights), 1.)
