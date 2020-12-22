from functools import partial

import numpy
import chaospy

from .utils import (
    ensure_input, ensure_output, distribution_to_domain,
    univariate_to_multivariate, split_into_segments, scale_samples
)


def hypercube_quadrature(
    quad_func,
    order,
    domain,
    segments,
    auto_scale=True,
):
    quad_func = partial(
        ensure_output,
        quad_func=quad_func,
    )
    quad_func = partial(
        split_into_segments,
        quad_func=quad_func,
        segments=numpy.array(segments)

    )
    quad_func = partial(
        univariate_to_multivariate,
        quad_func=quad_func,
    )
    quad_func = partial(
        ensure_input,
        quad_func=quad_func,
    )
    if auto_scale:
        quad_func = partial(
            scale_samples,
            quad_func=quad_func,
        )

    if isinstance(domain, chaospy.Distribution):
        quad_func = partial(
            distribution_to_domain,
            quad_func=quad_func,
            distribution=domain,
        )
    else:
        quad_func = partial(
            quad_func,
            lower=domain[0],
            upper=domain[1],
        )

    return quad_func(order=order)
