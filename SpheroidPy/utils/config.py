"""
Central configuration for SpheroidPy.

This module provides a centralized configuration system that allows users to
set global preferences for various components of the library.

Example:
    >>> from SpheroidPy.utils.config import Config, AISegmentationType
    >>> Config.ai_segmentation = AISegmentationType.HRNET
    >>> # Now all 'ai' segmentation calls will use HRNet instead of Detectron2
"""
from enum import Enum
import logging

logger = logging.getLogger("SpheroidPy.utils.config")

# Check availability of AI segmentation methods
try:
    from SpheroidPy.utils.detectron_utils import DETECTRON2_AVAILABLE
except ImportError:
    DETECTRON2_AVAILABLE = False

try:
    from SpheroidPy.utils.hrnet_utils import TORCH_AVAILABLE as HRNET_AVAILABLE
except ImportError:
    HRNET_AVAILABLE = False


class AISegmentationType(Enum):
    """Enumeration of available AI segmentation methods."""
    DETECTRON = "detectron"  # Detectron2 (Mask R-CNN)
    HRNET = "hrnet"          # HRNet segmentation model


def _get_default_ai_segmentation() -> AISegmentationType:
    """Determine default AI segmentation method based on availability.
    
    Priority: Detectron2 (if available) > HRNet (if available)
    
    Returns:
        Default AISegmentationType based on what's available
    """
    if DETECTRON2_AVAILABLE:
        return AISegmentationType.DETECTRON
    elif HRNET_AVAILABLE:
        logger.info("Detectron2 not available, using HRNet as default AI segmentation method.")
        return AISegmentationType.HRNET
    else:
        logger.warning("Neither Detectron2 nor HRNet available. AI segmentation will not work.")
        # Return DETECTRON as fallback, but it will fail at runtime
        return AISegmentationType.DETECTRON


class Config:
    """Central configuration class for SpheroidPy.
    
    This class holds global configuration settings that can be modified
    at runtime. Changes affect all instances of classes that use these settings.
    
    Attributes:
        ai_segmentation: Which AI segmentation method to use when 'ai' is
                         specified in segmentation methods. 
                         Default: Automatically set based on availability
                         (Detectron2 if available, otherwise HRNet).
    
    Example:
        >>> from SpheroidPy.utils.config import Config, AISegmentationType
        >>> # Check current default
        >>> print(Config.ai_segmentation)
        >>> 
        >>> # Set HRNet as default AI segmentation method (if available)
        >>> Config.ai_segmentation = AISegmentationType.HRNET
        >>> 
        >>> # Now segmentation with 'ai' will use HRNet
        >>> img.segmentation(('ai', 'brightfield'))
    """
    
    _ai_segmentation: AISegmentationType = _get_default_ai_segmentation()
    
    @classmethod
    def _get_available_methods(cls) -> list[str]:
        """Get list of available AI segmentation methods."""
        available = []
        if DETECTRON2_AVAILABLE:
            available.append("DETECTRON")
        if HRNET_AVAILABLE:
            available.append("HRNET")
        return available
    
    @classmethod
    def _validate_ai_segmentation(cls, value: AISegmentationType):
        """Validate that the requested AI segmentation method is available."""
        if not isinstance(value, AISegmentationType):
            raise TypeError(f"ai_segmentation must be an AISegmentationType, got {type(value)}")
        
        # Validate availability
        if value == AISegmentationType.DETECTRON and not DETECTRON2_AVAILABLE:
            available = cls._get_available_methods()
            raise ValueError(
                f"Detectron2 is not available. Please install Detectron2 to use this method. "
                f"Available methods: {available}"
            )
        elif value == AISegmentationType.HRNET and not HRNET_AVAILABLE:
            available = cls._get_available_methods()
            raise ValueError(
                f"HRNet is not available. Please install PyTorch to use this method. "
                f"Available methods: {available}"
            )


# Create a descriptor to handle property-like access on the class
class _AISegmentationDescriptor:
    """Descriptor for ai_segmentation property on Config class."""
    
    def __get__(self, obj, objtype=None):
        return Config._ai_segmentation
    
    def __set__(self, obj, value):
        Config._validate_ai_segmentation(value)
        Config._ai_segmentation = value
        logger.info(f"AI segmentation method set to: {value.value}")


# Attach descriptor to Config class
Config.ai_segmentation = _AISegmentationDescriptor()

