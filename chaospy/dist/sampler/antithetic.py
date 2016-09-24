"""
Antithetic variables.
"""

import numpy as np

def antithetic_gen(U, order):

    order = np.array(order, dtype=bool).flatten()
    if order.size==1 and len(U)>1:
        order = np.repeat(order, len(U))

    U = np.asfarray(U).T
    iU = 1-U
    out = []
    index = np.zeros(len(U.T), dtype=bool)

    def expand(I):
        index[order] = I
        return index

    for I in np.ndindex((2,)*sum(order*1)):
        I = expand(I)
        out.append((U*I + iU*(True-I)).T)

    out = np.concatenate(out[::-1], 1)
    return out
