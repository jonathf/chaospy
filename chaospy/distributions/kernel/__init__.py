"""
In some cases a constructed distribution that are first and foremost data
driven. In such scenarios it make sense to make use of
`kernel density estimation`_ (KDE). In ``chaospy`` KDE can be accessed through
the :func:`GaussianKDE` constructor.

Basic usage of the :func:`GaussianKDE` constructor involves just passing the
data as input argument::

    >>> data = [3, 4, 5, 5]
    >>> distribution = chaospy.GaussianKDE(data)

This distribution can be used as any other distributions::

    >>> distribution.cdf([3, 3.5, 4, 4.5, 5]).round(4)
    array([0.1393, 0.2542, 0.3889, 0.5512, 0.7359])
    >>> distribution.mom(1).round(4)
    4.25
    >>> distribution.sample(4).round(4)
    array([4.7784, 2.8769, 5.8109, 4.2995])

In addition multivariate distributions supported::

    >>> data = [[1, 2, 2, 3], [5, 5, 4, 3]]
    >>> distribution = chaospy.GaussianKDE(data)
    >>> distribution.sample(4).round(4)
    array([[2.081 , 3.0304, 3.0882, 0.4872],
           [3.2878, 2.5473, 2.2699, 5.3412]])

.. _kernel density estimation: \
https://en.wikipedia.org/wiki/Kernel_density_estimation
"""
from .gaussian import GaussianKDE
from .mixture import GaussianMixture
