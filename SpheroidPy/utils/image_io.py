"""
Utility functions for reading image files, including support for .zvi (Zeiss Vision Image) format.
"""
import struct
import numpy as np
from pathlib import Path
from typing import Optional
from collections import namedtuple

try:
    import OleFileIO_PL
    OLEFILEIO_AVAILABLE = True
except ImportError:
    OLEFILEIO_AVAILABLE = False

import cv2
import logging

logger = logging.getLogger("SpheroidPy.utils.image_io")

# ZVI format structures
ZviImageTuple = namedtuple('ZviImageTuple',
                       'Version FileName Width Height Depth PixelFormat Count '
                       'ValidBitsPerPixel m_PluginCLSID Others Layers Scaling')

ZviItemTuple = namedtuple('ZviItemTuple',
                       'Version FileName Width Height Depth PixelFormat Count '
                       'ValidBitsPerPixel Others Layers Scaling Image')

ImageTuple = namedtuple('ImageTuple',
                       'Version Width Height Depth PixelWidth PixelFormat '
                       'ValidBitsPerPixel Array')

PixelFormat = {1: (3, 'ByteBGR'),
               2: (4, 'ByteBGRA'),
               3: (1, 'Byte'),
               4: (2, 'Word'),
               5: (4, 'Long'),
               6: (4, 'Float'),
               7: (8, 'Double'),
               8: (6, 'WordBGR'),
               9: (4, 'LongBGR')}


def i32(data):
    """Return int32 from len4 string."""
    low, high = struct.unpack('<hh', data[:4])
    return (high << 16) + low


def read_struct(data, t):
    """Read a t type from data(str)."""
    next_data = data[2:]  # skip vartype I16

    if t == '?' or t == 'EMPTY' or t == 'NULL':
        return [None, next_data]
    if t == 'I2':
        low = struct.unpack('<h', next_data[:2])
        return [low[0], next_data[2:]]
    if t == 'I4':
        r = i32(next_data[:4])
        return [r, next_data[4:]]
    if t == 'BLOB':
        size = i32(next_data[:4])
        r = next_data[4:4+size]
        return [r, next_data[4+size:]]
    if t == 'BSTR':
        low, high = struct.unpack('<hh', next_data[:4])
        size = (high << 16) + low
        if size > 0:
            s = struct.unpack(f'{size}s', next_data[4:4+size])[0]
            next_data = next_data[4+4+size:]
        else:
            s = ''
            next_data = next_data[4+4:]
        return [s, next_data]
    raise ValueError(f'unknown type: {t}')


def read_image_container_content(stream):
    """Returns a ZviImageTuple from a stream."""
    data = stream.read()
    next_data = data
    [version, next_data] = read_struct(next_data, 'I4')
    [filename, next_data] = read_struct(next_data, 'BSTR')
    [width, next_data] = read_struct(next_data, 'I4')
    [height, next_data] = read_struct(next_data, 'I4')
    [depth, next_data] = read_struct(next_data, 'I4')
    [pixel_format, next_data] = read_struct(next_data, 'I4')
    [count, next_data] = read_struct(next_data, 'I4')
    [valid_bits_per_pixel, next_data] = read_struct(next_data, 'I4')
    [m_PluginCLSID, next_data] = read_struct(next_data, 'I4')
    [others, next_data] = read_struct(next_data, 'I4')
    [layers, next_data] = read_struct(next_data, 'I4')
    [scaling, next_data] = read_struct(next_data, 'I2')

    zvi_image = ZviImageTuple(version, filename, width, height, depth, pixel_format,
                    count, valid_bits_per_pixel, m_PluginCLSID, others, layers, scaling)
    return zvi_image


def parse_image(data):
    """Returns ImageTuple from raw image data(header+image)."""
    version = i32(data[:4])
    width = i32(data[4:8])
    height = i32(data[8:12])
    depth = i32(data[12:16])
    pixel_width = i32(data[16:20])
    pixel_format = i32(data[20:24])
    valid_bits_per_pixel = i32(data[24:28])
    
    # Read image data
    pixel_format_info = PixelFormat.get(pixel_format, (1, 'Byte'))
    bytes_per_pixel = pixel_format_info[0]
    total_bytes = width * height * bytes_per_pixel
    
    if pixel_format == 4:  # Word (uint16)
        raw = np.frombuffer(data[28:28+total_bytes], dtype=np.uint16)
        array = np.reshape(raw, (height, width))
        # Normalize to 0-255
        if array.max() > 0:
            array = (array / array.max() * 255).astype(np.uint8)
        else:
            array = array.astype(np.uint8)
    elif pixel_format == 8:  # WordBGR (16-bit per channel)
        raw = np.frombuffer(data[28:28+total_bytes], dtype=np.uint16)
        array = np.reshape(raw, (height, width, 3))
    elif pixel_format == 1:  # ByteBGR
        raw = np.frombuffer(data[28:28+total_bytes], dtype=np.uint8)
        array = np.reshape(raw, (height, width, 3))
    elif pixel_format == 3:  # Byte (grayscale)
        raw = np.frombuffer(data[28:28+total_bytes], dtype=np.uint8)
        array = np.reshape(raw, (height, width))
    else:
        # Fallback: try to read as uint8
        raw = np.frombuffer(data[28:28+total_bytes], dtype=np.uint8)
        if bytes_per_pixel == 1:
            array = np.reshape(raw, (height, width))
        else:
            array = np.reshape(raw, (height, width, bytes_per_pixel))

    image = ImageTuple(version, width, height, depth, pixel_width, pixel_format,
                        valid_bits_per_pixel, array)
    return image


def read_item_storage_content(stream):
    """Returns ZviItemTuple from the stream."""
    data = stream.read()
    next_data = data
    [version, next_data] = read_struct(next_data, 'I4')
    [filename, next_data] = read_struct(next_data, 'BSTR')
    [width, next_data] = read_struct(next_data, 'I4')
    [height, next_data] = read_struct(next_data, 'I4')
    [depth, next_data] = read_struct(next_data, 'I4')
    [pixel_format, next_data] = read_struct(next_data, 'I4')
    [count, next_data] = read_struct(next_data, 'I4')
    [valid_bits_per_pixel, next_data] = read_struct(next_data, 'I4')
    [others, next_data] = read_struct(next_data, 'BLOB')
    [layers, next_data] = read_struct(next_data, 'BLOB')
    [scaling, next_data] = read_struct(next_data, 'BLOB')
    
    # Offset is image size + header size(28)
    offset = width * height * PixelFormat[pixel_format][0] + 28
    # Parse the actual image data
    image = parse_image(data[-offset:])
    
    # Group results into one single structure (namedtuple)
    zvi_item = ZviItemTuple(version, filename, width, height, depth, pixel_format,
                    count, valid_bits_per_pixel, others, layers, scaling, image)

    return zvi_item


def read_zvi_image(filepath: str | Path, plane: int = 0) -> np.ndarray:
    """
    Read a .zvi (Zeiss Vision Image) file and return image as numpy array.
    
    Args:
        filepath: Path to the .zvi file
        plane: Image plane index (default: 0 for first plane)
        
    Returns:
        numpy array of the image (BGR format for color images, grayscale for single channel)
        
    Raises:
        ImportError: If OleFileIO_PL is not available
        IOError: If file cannot be read
    """
    if not OLEFILEIO_AVAILABLE:
        raise ImportError("OleFileIO_PL is required to read .zvi files. Install it with: pip install olefile")
    
    filepath = Path(filepath)
    if not filepath.exists():
        raise IOError(f"ZVI file not found: {filepath}")
    
    try:
        ole = OleFileIO_PL.OleFileIO(str(filepath))
        s = ['Image', f'Item({plane})', 'Contents']
        stream = ole.openstream(s)
        zvi_item = read_item_storage_content(stream)
        ole.close()
        
        # Extract image array
        img_array = zvi_item.Image.Array
        
        # Convert to OpenCV-compatible format (BGR or grayscale uint8)
        if len(img_array.shape) == 3:
            channels = img_array.shape[2]
            if img_array.dtype == np.uint16:
                # Convert 16-bit data to 8-bit by scaling
                max_val = np.max(img_array)
                if max_val > 0:
                    img_array = (img_array / (max_val / 255.0)).astype(np.uint8)
                else:
                    img_array = img_array.astype(np.uint8)
            if channels == 3:
                # Assume it's RGB, convert to BGR for OpenCV compatibility
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            elif channels == 4:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
            elif channels > 3:
                # Keep only first three channels, convert to BGR
                img_array = img_array[:, :, :3]
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        elif len(img_array.shape) == 2:
            # Grayscale image, ensure uint8
            if img_array.dtype != np.uint8:
                max_val = np.max(img_array)
                if max_val > 0:
                    img_array = (img_array / (max_val / 255.0)).astype(np.uint8)
                else:
                    img_array = img_array.astype(np.uint8)
        
        # Force grayscale output for consistency with brightfield processing
        #if len(img_array.shape) == 3:
        #    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        return img_array
        
    except Exception as e:
        raise IOError(f"Failed to read ZVI file {filepath}: {e}")


def imread(filepath: str | Path, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    """
    Read an image file, preferring the fast OpenCV imread() path first and 
    automatically falling back to .zvi reading if necessary.
    
    This function behaves like a drop-in replacement for cv2.imread(), but with 
    additional support for .zvi (Zeiss Vision Image) files. For all formats, the 
    function first attempts loading via OpenCV for maximum performance. Only if 
    that fails and the file extension is .zvi, the slower .zvi loader is used.
    
    Args:
        filepath: Path to the image file.
        flags: OpenCV imread flags (only relevant for non-.zvi files).
        
    Returns:
        numpy array of the image, or None if the file cannot be read.
    """
    filepath = Path(filepath)

    # Check file existence early
    if not filepath.exists():
        logger.warning(f"Image file not found: {filepath}")
        return None

    # --- 1) Try fast OpenCV path first ---
    img = cv2.imread(str(filepath), flags)
    if img is not None:
        return img  # Fast path succeeded!

    # --- 2) Fallback: Check for .zvi ---
    if filepath.suffix.lower() == ".zvi":
        try:
            return read_zvi_image(filepath, plane=0)
        except ImportError as e:
            logger.error(f"Cannot read .zvi file: {e}")
            return None
        except Exception as e:
            logger.error(f"Error reading .zvi file {filepath}: {e}")
            return None

    # --- 3) All methods failed ---
    logger.warning(f"cv2.imread failed and file is not .zvi: {filepath}")
    return None
