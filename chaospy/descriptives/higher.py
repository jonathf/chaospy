import numpy as np

import chaospy as cp


def Skew(poly, dist=None, **kws):
    """
    Skewness operator.

    Element by element 3rd order statistics of a distribution or polynomial.

    Args:
        poly (Poly, Dist) : Input to take skewness on.
        dist (Dist) : Defines the space the skewness is taken on.
                It is ignored if `poly` is a distribution.
        **kws (optional) : Extra keywords passed to dist.mom.

    Returns:
        (ndarray) : Element for element variance along `poly`, where
                `skewness.shape==poly.shape`.

    Examples:
        >>> x = cp.variable()
        >>> Z = cp.Gamma()
        >>> print(cp.Skew(Z))
        2.0
    """
    if isinstance(poly, cp.dist.Dist):
        x = cp.poly.variable(len(poly))
        poly, dist = x, poly
    else:
        poly = cp.poly.Poly(poly)

    if poly.dim < len(dist):
        cp.poly.setdim(poly, len(dist))

    shape = poly.shape
    poly = cp.poly.flatten(poly)

    m1 = cp.E(poly, dist)
    m2 = cp.E(poly**2, dist)
    m3 = cp.E(poly**3, dist)
    out = (m3-3*m2*m1+2*m1**3)/(m2-m1**2)**1.5

    out = np.reshape(out, shape)
    return out



def Kurt(poly, dist=None, fisher=True, **kws):
    """
    Kurtosis operator.

    Element by element 4rd order statistics of a distribution or polynomial.

    Args:
        poly (Poly, Dist) : Input to take kurtosis on.
        dist (Dist) : Defines the space the skewness is taken on.
                It is ignored if `poly` is a distribution.
        fisher (bool) : If True, Fisher's definition is used (Normal -> 0.0).
                If False, Pearson's definition is used (normal -> 3.0)
        **kws (optional) : Extra keywords passed to dist.mom.

    Returns:
        (ndarray) : Element for element variance along `poly`, where
                `skewness.shape==poly.shape`.

    Examples:
        >>> x = cp.variable()
        >>> Z = cp.Uniform()
        >>> print(np.around(cp.Kurt(Z), 8))
        -1.2
        >>> Z = cp.Normal()
        >>> print(np.around(cp.Kurt(x, Z), 8))
        0.0
    """
    if isinstance(poly, cp.dist.Dist):
        x = cp.poly.variable(len(poly))
        poly, dist = x, poly
    else:
        poly = cp.poly.Poly(poly)

    if fisher:
        adjust = 3
    else:
        adjust = 0

    shape = poly.shape
    poly = cp.poly.flatten(poly)

    m1 = cp.E(poly, dist)
    m2 = cp.E(poly**2, dist)
    m3 = cp.E(poly**3, dist)
    m4 = cp.E(poly**4, dist)

    out = (m4-4*m3*m1 + 6*m2*m1**2 - 3*m1**4) /\
            (m2**2-2*m2*m1**2+m1**4) - adjust

    out = np.reshape(out, shape)
    return out
