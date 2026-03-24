import logging
import os
import warnings

"""SpheroidPy - Python Package for the Analysis of Spheroid Imaging Data."""

__version__ = "0.1.0"

# Set OpenCV log level early so TIFF metadata warnings are suppressed package-wide.
# Users can still override this via OPENCV_LOG_LEVEL in their environment.
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

# Suppress specific noisy runtime warnings (keep scope narrow)
warnings.filterwarnings(
    "ignore",
    message=r"torch\.meshgrid: in an upcoming release, it will be required to pass the indexing argument\.")

# Best-effort OpenCV runtime log suppression (covers existing process env).
try:
    import cv2

    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    except Exception:
        # Fallback for OpenCV variants exposing constants at top-level.
        cv2.setLogLevel(cv2.LOG_LEVEL_ERROR)
except Exception:
    # OpenCV may not be installed in minimal environments.
    pass

from .spheroid import *
from .experiment import *

__all__ = ['spheroid', 'experiment']

# Silence library logging by default; host applications can opt-in
_pkg_logger = logging.getLogger("SpheroidPy")
_pkg_logger.addHandler(logging.NullHandler())
# Prevent bubbling to root handlers in notebooks/apps
_pkg_logger.propagate = False

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
