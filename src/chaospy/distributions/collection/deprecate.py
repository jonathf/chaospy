"""Give warning for using old names."""
import logging
from functools import wraps


def deprecation_warning(func, name):
    """Add a deprecation warning do each distribution."""
    @wraps(func)
    def caller(*args, **kwargs):
        """Docs to be replaced."""
        logger = logging.getLogger(__name__)
        instance = func(*args, **kwargs)
        logger.warning(
            "Distribution `chaospy.{}` has been renamed to ".format(name) +
            "`chaospy.{}` and will be deprecated next release.".format(instance.__class__.__name__))
        return instance
    return caller
