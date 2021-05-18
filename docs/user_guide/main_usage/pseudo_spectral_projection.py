import chaospy
from problem_formulation import joint

gauss_quads = [
    chaospy.generate_quadrature(order, joint, rule="gaussian")
    for order in range(1, 8)
]
