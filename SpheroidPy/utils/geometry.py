from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelextrema
import numpy as np
import logging
from typing import Optional, Tuple
from skimage.measure import EllipseModel
logger = logging.getLogger("SpheroidPy.utils.geometry")

def find_inflection_point(y_array, x_array=None, sigma=None):
    """
    Find the (global) maximum, minimum and inflection point of a curve given x and y arrays.

    Parameters:
    - x_array (numpy array): The x-values (e.g., normalized distance).
    - y_array (numpy array): The y-values (e.g., normalized intensity).

    Returns:
    - ...
    """

    if sigma is not None:
        # Smooth the data using a Gaussian filter
        y_array = gaussian_filter1d(y_array, sigma=sigma)

    if x_array is None:
        x_array = np.linspace(0, 1, len(y_array))

    # Calculate the first derivative
    dy_dx_array = np.gradient(y_array, x_array)

    #plt.plot(x_array,np.abs(dy_dx_array))
    #plt.show()

    # Global maximum
    max_index = np.argmax(y_array)
    x_global_max = x_array[max_index]
    y_global_max = y_array[max_index]

    # Global maximum
    min_index = np.argmin(y_array)
    x_global_min = x_array[min_index]
    y_global_min = y_array[min_index]

    # Global inflection point (where second derivative is closest to zero)
    inflection_index = np.argmax(np.abs(dy_dx_array))
    x_global_inflection = x_array[inflection_index]
    y_global_inflection = y_array[inflection_index]

    try:
        # Find the global minimum value
        inflection_min_index = argrelextrema(np.abs(dy_dx_array), np.less)[0][-1]
        x_inflection_min = x_array[inflection_min_index]
        y_inflection_min = y_array[inflection_min_index]
    except:
        inflection_min_index,x_inflection_min,y_inflection_min = None, None, None

    return {'maximum': (x_global_max, y_global_max, max_index), 'minimum': (x_global_min, y_global_min, min_index), 'inflection': (x_global_inflection, y_global_inflection, inflection_index), 'inflection_min': (x_inflection_min, y_inflection_min, inflection_min_index)}


def touches_border(contour: np.ndarray, x_max, y_max, x_min=0, y_min=0) -> bool:

    # Überprüfen, ob der Array den Rand berührt
    x_touch = np.any((contour[:, 0] <= x_min) | (contour[:, 0] >= x_max))
    y_touch = np.any((contour[:, 1] <= y_min) | (contour[:, 1] >= y_max))

    return x_touch or y_touch

def remove_border_points(contour: np.ndarray, x_max, y_max, x_min=0, y_min=0) -> np.ndarray:
    # Überprüfen, welche Punkte den Rand berühren
    border_touching = (contour[:, 0] <= x_min) | (contour[:, 0] >= x_max) | (contour[:, 1] <= y_min) | (contour[:, 1] >= y_max)

    # Punkte filtern, die den Rand nicht berühren
    filtered_contour = contour[border_touching==False]
    #print(contour[border_touching==True][::70])

    return filtered_contour


def fit_ellipse(contour_points: np.ndarray) -> Optional[Tuple[Tuple[float, float], float, float, np.ndarray]]:
    """Fit an ellipse to a set of contour points.

    Args:
        contour_points: Array of shape (N, 2) containing x,y coordinates.

    Returns:
        tuple: (center, area, radius, contour) if successful, None if fitting fails.
            center: (x,y) coordinates of ellipse center.
            area: Area of fitted ellipse.
            radius: Effective radius (geometric mean of semi-axes).
            contour: Array of points along fitted ellipse.
    """
    # 1. Input Validation and early exit
    if not isinstance(contour_points, np.ndarray):
        logger.error("Input must be a numpy array.")
        return None

    valid_points = contour_points[~np.isnan(contour_points).any(axis=1)]
    if len(valid_points) < 5:  # Need at least 5 points for ellipse fitting
        logger.warning("Not enough valid points for ellipse fitting. Need at least 5.")
        return None

    try:
        # 2. Fit the ellipse model
        ellipse_model = EllipseModel()
        success = ellipse_model.estimate(valid_points)

        if not success:
            logger.warning("Ellipse fitting failed to converge.")
            return None

        # 3. Extract and validate parameters
        xc, yc, a, b, theta = ellipse_model.params

        if a <= 0 or b <= 0:
            logger.warning(f"Invalid ellipse axes (a={a}, b={b}).")
            return None

        if not all(np.isfinite([xc, yc, a, b, theta])):
            logger.error("Invalid ellipse parameters detected (NaN/inf).")
            return None

        # 4. Calculate derived values
        center = (xc, yc)
        ellipse_area = np.pi * a * b
        effective_radius = np.sqrt(a * b)

        # 5. Generate a dense, fitted ellipse contour
        # Use a fixed number of points for consistency, e.g., 100
        t = np.linspace(0, 2 * np.pi, 100)
        x = xc + a * np.cos(t) * np.cos(theta) - b * np.sin(t) * np.sin(theta)
        y = yc + a * np.cos(t) * np.sin(theta) + b * np.sin(t) * np.cos(theta)
        fitted_contour = np.array(list(zip(x, y)), dtype=np.float32)

        logger.info("Ellipse fitting successful.")

        return center, ellipse_area, effective_radius, fitted_contour

    except Exception as e:
        logger.error(f"Unexpected error during ellipse fitting: {str(e)}")
        return None


def add_fitted_contour(contour_original: np.ndarray, contour_fitted: np.ndarray | None,
                       x_max: int, y_max: int, x_min: int = 0, y_min: int = 0) -> np.ndarray:
    """Combine original and fitted contours, using fitted points for border regions.

    Args:
        contour_original: Original contour points.
        contour_fitted: Fitted ellipse contour points (or None if fitting failed).
        x_max, y_max: Maximum x,y coordinates.
        x_min, y_min: Minimum x,y coordinates (default 0).

    Returns:
        Combined contour array.
    """
    if contour_fitted is None:
        return contour_original

    # 1. Punkte der Originalkontur filtern, die innerhalb der Grenzen liegen
    # Diese werden verwendet, wo die originale Kontur zuverlässig ist
    mask_original = (
            (contour_original[:, 0] > x_min) & (contour_original[:, 0] < x_max) &
            (contour_original[:, 1] > y_min) & (contour_original[:, 1] < y_max)
    )
    inner_contour = contour_original[mask_original]

    # 2. Bestimme welche Bereiche der gefitteten Kontur verwendet werden sollen
    # Verwende gefittete Punkte für:
    # - Bereiche außerhalb der Grenzen (Border-Bereiche)
    # - Bereiche nahe den Grenzen, wo die originale Kontur unzuverlässig ist
    
    # Erweitere die Grenzen leicht für einen sanfteren Übergang
    transition_margin = 2  # pixels
    
    # Punkte der gefitteten Kontur, die die Border-Bereiche abdecken
    mask_fitted_outer = (
            (contour_fitted[:, 0] <= x_min + transition_margin) | 
            (contour_fitted[:, 0] >= x_max - transition_margin) |
            (contour_fitted[:, 1] <= y_min + transition_margin) | 
            (contour_fitted[:, 1] >= y_max - transition_margin)
    )
    fitted_contour_points = contour_fitted[mask_fitted_outer]

    # 3. Kombiniere innere originale Kontur mit gefitteten Border-Bereichen
    if len(inner_contour) > 0:
        combined_points = np.vstack([inner_contour, fitted_contour_points])
    else:
        # Fallback: Wenn keine inneren Punkte, verwende nur gefittete Kontur
        combined_points = contour_fitted

    # 4. Sortiere die kombinierten Punkte, um eine geschlossene Kontur zu bilden
    # Sortierung nach Winkel vom Mittelpunkt aus
    center = np.mean(combined_points, axis=0)
    y, x = combined_points[:, 1] - center[1], combined_points[:, 0] - center[0]
    angles = np.arctan2(y, x)
    sorted_indices = np.argsort(angles)

    return combined_points[sorted_indices].astype(np.float32)