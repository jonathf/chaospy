"""
``chaospy`` is in no means a package that provides absolutely all functionality
for all problems. If another project provides functionality that can work well
in tandem, the best approach, if possible, is to use both at the same time. To
make such an approach more feasible, some compatibility wrappers exists
allowing for using components from other projects as part of ``chaospy``.
"""
from .openturns_ import openturns_dist, OpenTURNSDist
from .scipy_stats import ScipyStatsDist
from .samples import sample_dist, SampleDist
