"""SpheroidPy - Python Package for the Analysis of Spheroid Imaging Data."""

__version__ = "0.1.0"

from .spheroid import *
from .experiment import *

__all__ = ['spheroid', 'experiment']

import logging
import warnings

# Silence library logging by default; host applications can opt-in
_pkg_logger = logging.getLogger("SpheroidPy")
_pkg_logger.addHandler(logging.NullHandler())
# Prevent bubbling to root handlers in notebooks/apps
_pkg_logger.propagate = False

# Suppress specific noisy runtime warnings (keep scope narrow)
warnings.filterwarnings(
    "ignore",
    message=r"torch\.meshgrid: in an upcoming release, it will be required to pass the indexing argument\.")

def configure_logging(level: int = logging.WARNING) -> None:
    """Enable logging for the SpheroidPy package (opt-in for host apps).

    Args:
        level: Logging level to set for the package logger.
    """
    logger = logging.getLogger("SpheroidPy")
    logger.setLevel(level)
    logger.propagate = False
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
