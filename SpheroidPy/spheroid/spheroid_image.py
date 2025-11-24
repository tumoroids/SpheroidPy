from typing import Union
import logging

# Detectron2 utilities
from SpheroidPy.utils.detectron_utils import (
    DETECTRON2_AVAILABLE,
    get_detectron_predictor
)

# Other imports that don't depend on detectron2
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import tkinter as tk

import pkg_resources

from scipy.ndimage import binary_dilation, binary_fill_holes, binary_closing
from scipy.ndimage import label, center_of_mass
from scipy.ndimage import gaussian_filter1d
from skimage.morphology import remove_small_objects
from scipy.sparse.linalg import spsolve
from matplotlib.colors import Normalize
from skimage.filters import threshold_otsu, threshold_yen
from skimage.measure import find_contours
from skimage.segmentation import clear_border


from SpheroidPy.utils.visualisation import plot_mesh
from SpheroidPy.utils.segmentation import ManualSegmentation
from SpheroidPy.utils.pde_utils import compute_laplace_matrix, apply_dirichlet_boundary_conditions, get_boundary_nodes
from SpheroidPy.utils.geometry import touches_border, fit_ellipse, remove_border_points, add_fitted_contour, find_inflection_point
from SpheroidPy.utils.fluorescence_analysis import (
    calculate_profile_statistics, find_inflection_points, calculate_thresholds,
    identify_functional_zones, plot_fluorescence_analysis, analyze_fluorescence_profile,
    fit_sigmoid_auto
)
from SpheroidPy.utils.image_io import imread as read_image

import matplotlib

logger = logging.getLogger("SpheroidPy.spheroid_image")


class SpheroidImage:
    """
    A class for analyzing and processing images of spheroids.

    Attributes
    ----------
    image_path_dict : dict
        Mapping of channel name to image path.
    image_size : tuple
        Height and width of the brightfield image in µm.
    hdf5_path : str or Path, optional
        Path to associated HDF5 file (if provided).
    hdf5_key : str, optional
        Key for accessing data in HDF5 file (if provided).
    analysis_results : dict
        Stores computed analysis results.
    contour : ndarray or None
        Contour of the spheroid (if detected).
    contour_touches_border : bool or None
        Whether the spheroid contour touches the image border.

    Examples
    --------
    >>> img = SpheroidImage(
    ...     brightfield="path/to/brightfield.tif", #required!
    ...     fluorescence_green="path/to/green.tif",
    ...     fluorescence_red="path/to/red.tif",
    ...     fluorescence_blue="path/to/blue.tif",
    ...     image_size=(1080, 1920),
    ...     hdf5_path="results.h5",
    ...     hdf5_key="spheroid_01"
    ... )
    """

    def __init__(self, brightfield: Path | str, **kwargs):
        """
        Initialize a SpheroidImage.

        Parameters
        ----------
        brightfield : Path or str
            Path to the brightfield image (required).
        fluorescence_green : Path or str, optional
            Path to green fluorescence image.
        fluorescence_red : Path or str, optional
            Path to red fluorescence image.
        fluorescence_blue : Path or str, optional
            Path to blue fluorescence image.
        image_size : tuple of int, optional
            Image size as (height, width). If not provided, inferred from brightfield.
        hdf5_path : str or Path, optional
            Path to associated HDF5 file.
        hdf5_key : str, optional
            Key for accessing data in HDF5 file.
        """
        self.image_path_dict = {}
        self._initialize_channels(brightfield, **kwargs)
        self._initialize_attributes(**kwargs)
        # Lazy-loaded predictors and caches
        self.predictor = None

    def _initialize_channels(self, brightfield: Path | str, **kwargs):
        """
        Initialize the image channels for the spheroid.

        Parameters
        ----------
        brightfield : Path or str
            Path to the brightfield image file (required!).
        kwargs : dict, optional
            Additional imaging channels, where the keys can include:
            - fluorescence_green : Path or str
                Path to green fluorescence image.
            - fluorescence_red : Path or str
                Path to red fluorescence image.
            - fluorescence_blue : Path or str
                Path to blue fluorescence image.

        Notes
        -----
        - Loads each available image using OpenCV (`cv2.imread`).
        - Stores the valid channel paths in `self.image_path_dict`.
        - Sets `self.shape` to the shape of the last successfully loaded image.
        - If a channel cannot be loaded, an error message is printed but the process continues.
        """
        channels = {
            'brightfield': brightfield,
            'fluorescence_green': kwargs.get('fluorescence_green'),
            'fluorescence_red': kwargs.get('fluorescence_red'),
            'fluorescence_blue': kwargs.get('fluorescence_blue')
        }

        for channel, path in channels.items():
            if path and os.path.exists(str(path)):
                self.image_path_dict[channel] = Path(path)
            elif path:
                logger.warning(f'Image path for channel "{channel}" does not exist: {path}')


    def _initialize_attributes(self, **kwargs):
        """
        Initialize other attributes such as image size and HDF5 metadata.

        Parameters
        ----------
        kwargs : dict
            May include:
            - image_size : tuple
            - hdf5_path : str or Path
            - hdf5_key : str
        """
        # Load image once to get both image_size and pixel dimensions
        provided_size = kwargs.get('image_size')
        pixel_dims = None
        if 'brightfield' in self.image_path_dict:
            pixel_img = read_image(str(self.image_path_dict['brightfield']))
            if pixel_img is not None:
                pixel_dims = pixel_img.shape[:2]  # (height_px, width_px)
        self._pixel_dimensions = pixel_dims

        if provided_size is not None:
            width_um, height_um = provided_size
            self.image_size = (float(width_um), float(height_um))
        else:
            if pixel_dims is None:
                raise ValueError("Cannot load brightfield image to determine image size")
            # assume 1 px = 1 µm if not provided
            height_px, width_px = pixel_dims
            self.image_size = (float(width_px), float(height_px))
            
        self.hdf5_path = kwargs.get('hdf5_path')
        self.hdf5_key = kwargs.get('hdf5_key')
        self.analysis_results = {}
        self.contour = None
        self.contour_touches_border = None

    def segmentation(self,
                     methods: Union[str, tuple, list] = 'fluorescence_green',
                     reconstruct_border: bool = True,
                     border_margin: int = 10,
                     **kwargs) -> None:
        """Segment the spheroid using specified method(s) and channel(s).

        Supports single or multiple segmentation attempts with different methods.
        Will try methods in order until successful.

        Args:
            methods: Segmentation specification in one of these formats:
                - str: Channel to use with default thresholding (e.g. 'fluorescence_green')
                - tuple: (method, channel) e.g. ('ai', 'brightfield')
                - list: List of (method, channel) tuples to try in order
            reconstruct_border: Whether to reconstruct spheroid border when it
                extends beyond image bounds
            border_margin: Margin in pixels to use when detecting border contact
            **kwargs: Method-specific parameters:
                For thresholding:
                    threshold: Intensity threshold multiplier (default: 1.35)
                    use_yen: Whether to use Yen's method (default: True)
                For AI:
                    confidence: Detection confidence threshold (default: 0.65)
        """
        if isinstance(methods, str):
            methods = [('thresholding', methods)]
        elif isinstance(methods, tuple):
            methods = [methods]

        if not isinstance(methods, list):
            raise TypeError("Methods must be str, tuple, or list of tuples.")

        success = False
        self.contour = None
        self.contour_touches_border = None

        for seg_method, channel in methods:
            logger.info(f"Attempting segmentation with '{seg_method}' on channel '{channel}'.")
            try:
                if seg_method == 'thresholding':
                    self.contour = self.segmentation_thresholding(
                        channel,
                        thres=kwargs.get('threshold', 1.35),
                        thres_yen=kwargs.get('use_yen', True),
                        border_margin=border_margin
                    )
                elif seg_method == 'ai':
                    self.contour = self.segmentation_detectron(
                        channel,
                        thres=kwargs.get('confidence', 0.65),
                        border_margin=border_margin
                    )
                else:
                    logger.error(f"Unknown segmentation method '{seg_method}'. Skipping.")
                    continue

                if self.contour is not None and len(self.contour) > 5:
                    success = True
                    break
                else:
                    logger.warning(f"Segmentation with '{seg_method}' on channel '{channel}' failed to find a valid contour.")
            except Exception as e:
                logger.error(f"Segmentation failed for '{seg_method}' on channel '{channel}' with error: {e}")
                continue

        if not success:
            raise Exception("All specified segmentation methods failed.")

        # 1. Randeinzug und Rekonstruktion
        if self.contour_touches_border and reconstruct_border:
            try:
                # Get image dimensions in pixels for border reconstruction
                if not hasattr(self, '_pixel_dimensions') or self._pixel_dimensions is None:
                    # Fallback: load image if dimensions not cached
                    sample_img = read_image(str(self.image_path_dict['brightfield']))
                    if sample_img is None:
                        logger.warning("Cannot load brightfield image for border reconstruction.")
                        return
                    height_px, width_px = sample_img.shape[:2]
                    self._pixel_dimensions = (height_px, width_px)
                else:
                    height_px, width_px = self._pixel_dimensions
                
                # Remove border points from original contour before fitting
                # This gives us a clean contour for ellipse fitting
                from SpheroidPy.utils.geometry import remove_border_points
                cleaned_contour = remove_border_points(
                    self.contour,
                    width_px - border_margin,
                    height_px - border_margin,
                    border_margin,
                    border_margin
                )
                
                # Fit ellipse to cleaned contour (should give better estimate of full shape)
                ellipse_result = fit_ellipse(cleaned_contour)
                if ellipse_result is None:
                    logger.warning("Ellipse fitting failed. Cannot reconstruct border.")
                    return
                
                fitted_contour = ellipse_result[-1]  # Get the fitted contour
                
                self.contour = add_fitted_contour(
                    self.contour,
                    fitted_contour,
                    width_px - border_margin,   # x_max in pixels
                    height_px - border_margin,  # y_max in pixels
                    border_margin,              # x_min in pixels
                    border_margin               # y_min in pixels
                )
                logger.info("Spheroid border successfully reconstructed.")
            except Exception as e:
                logger.warning(f"Ellipse could not be reconstructed. Using original contour. Error: {e}")

    def brightfield(self) -> np.ndarray:
        return read_image(str(self.image_path_dict['brightfield']))

    def fluorescence(self, color: str) -> np.ndarray:
        color_dict = {'green': 'fluorescence_green', 'red': 'fluorescence_red', 'blue': 'fluorescence_blue'}
        if color not in color_dict or color_dict[color] not in self.image_path_dict:
            raise Exception(f'Color {color} not supported!')
        else:
            return read_image(str(self.image_path_dict[color_dict[color]]))

    def radial_profile(self, channels: list = ['green', 'red'], plot: bool = True, 
                      angle_step: int = 2, normalize: bool = True, 
                      return_absolute: bool = False, smoothing: float = 0,
                      savepath: str | None = None) -> dict:
        """Calculate the radial intensity profile of the spheroid.
        
        Measures fluorescence intensity along radial lines from the center 
        to the boundary at regular angular intervals.
        
        Args:
            channels: List of fluorescence channels to analyze. Options: 'green', 'red', 'blue'
            plot: Whether to display a plot of the profiles
            angle_step: Angular step size in degrees between measurements
            normalize: Whether to normalize intensity profiles to [0,1]. If False, absolute 
                      intensities are displayed. For multiple channels with normalize=False, 
                      separate y-axes are used (left and right).
            return_absolute: Whether to return absolute distances in addition to relative
            smoothing: Gaussian smoothing parameter (sigma). If 0, no smoothing is applied.
                      Applied before normalization.
            savepath: Optional path to save the plot. If None, plot is only displayed.
            
        Returns:
            Dictionary containing:
                mean_radius: Average radius of the spheroid in pixels
                intensity_profiles: Dictionary of intensity profiles for each channel
                absolute_distances: Absolute distances in μm (if return_absolute=True)
        """
        from shapely.geometry import Polygon, LineString
        import skimage.draw as draw
        from scipy.interpolate import interp1d

        # Scaling px → μm using image_size
        bf_img = read_image(str(self.image_path_dict['brightfield']))
        if bf_img is None:
            raise ValueError("Cannot load brightfield image for scaling")
        h_px, w_px = bf_img.shape[:2]
        if hasattr(self, 'image_size') and self.image_size is not None:
            height_um = float(self.image_size[1])
            width_um = float(self.image_size[0])
            sx = width_um / max(1, w_px)
            sy = height_um / max(1, h_px)
        else:
            sx = sy = 1.0

        # Center in px and μm
        M = cv2.moments(self.contour)
        x_center_px = int(M["m10"] / M["m00"])
        y_center_px = int(M["m01"] / M["m00"])
        x_center_um = x_center_px * sx
        y_center_um = y_center_px * sy

        # Approximate contour and convert to Shapely Polygon in μm
        epsilon = 0.1 * min(sx, sy)
        approx = cv2.approxPolyDP(self.contour, epsilon, True)
        contour_um = approx.reshape(-1, 2).astype(float)
        contour_um[:, 0] *= sx
        contour_um[:, 1] *= sy
        polygon = Polygon(contour_um)

        # Validate channels and check availability
        valid_channels = ['green', 'red', 'blue']
        for channel in channels:
            if channel not in valid_channels:
                raise ValueError(f"Channel '{channel}' not supported. Options: {valid_channels}")
            if f'fluorescence_{channel}' not in self.image_path_dict:
                raise ValueError(f"Fluorescence channel '{channel}' not available in image data")
        
        # Initialize storage for radii (μm) and intensities
        radius_array_um = []
        intensity_interpolation = {channel: [] for channel in channels}

        # Iterate over angles
        for angle in range(0, 360, angle_step):
            try:
                # Ray end-point in μm
                x_line_um = x_center_um + 1000.0 * np.cos(np.radians(angle))
                y_line_um = y_center_um + 1000.0 * np.sin(np.radians(angle))

                # Create a line and find intersections in μm
                line = LineString([(x_center_um, y_center_um), (x_line_um, y_line_um)])
                intersections = line.intersection(polygon)

                # Skip if no intersection
                if intersections.is_empty:
                    continue

                # Closest intersection point in μm
                intersection_um = list(intersections.coords)[1]
                # Convert to px for sampling
                ix_px = int(intersection_um[0] / sx)
                iy_px = int(intersection_um[1] / sy)

                for channel in channels:
                    channel_image = self.fluorescence(channel)
                    # Sample along the line in pixels
                    line_coords_px = np.array(draw.line(x_center_px, y_center_px, ix_px, iy_px)).T
                    intensity_values = cv2.cvtColor(channel_image, cv2.COLOR_BGR2GRAY)[line_coords_px[:, 1], line_coords_px[:, 0]]

                    # Radius along this ray in μm from center to boundary
                    if channel == channels[0]:
                        r_um = np.sqrt((intersection_um[0] - x_center_um) ** 2 + (intersection_um[1] - y_center_um) ** 2)
                        radius_array_um.append(r_um)

                    # Build normalized distance for interpolation (0..1)
                    n = len(intensity_values)
                    x_rel = np.linspace(0, 1, num=n)
                    interp_func = interp1d(x_rel, intensity_values, kind='cubic', bounds_error=False, fill_value=0)
                    intensity_interpolation[channel].append(interp_func)
            except Exception:
                pass

        # Compute mean radius in μm
        mean_radius_um = float(np.mean(radius_array_um)) if radius_array_um else 0.0


        # Compute mean intensity profiles
        x_all = np.linspace(0, 1, num=100)
        intensity_smooth = {}
        absolute_distances = None
        relative_distances = x_all.copy()

        for color in intensity_interpolation:
            if not intensity_interpolation[color]:
                continue
            interpolations = np.array([f(x_all) for f in intensity_interpolation[color]])
            mean_intensity = np.mean(interpolations, axis=0)
            std_intensity = np.std(interpolations, axis=0)

            # Apply Gaussian smoothing if requested (before normalization)
            if smoothing > 0:
                mean_intensity = gaussian_filter1d(mean_intensity, sigma=smoothing)
                std_intensity = gaussian_filter1d(std_intensity, sigma=smoothing)

            # Store absolute intensities before normalization (for plotting)
            mean_intensity_abs = mean_intensity.copy()
            std_intensity_abs = std_intensity.copy()

            # Normalize if requested
            if normalize:
                max_intensity = mean_intensity.max()
                if max_intensity > 0:
                    mean_intensity = mean_intensity / max_intensity
                    std_intensity = std_intensity / max_intensity

            intensity_smooth[color] = {
                'mean': mean_intensity,
                'std': std_intensity,
                'mean_absolute': mean_intensity_abs,  # Store absolute values for plotting
                'std_absolute': std_intensity_abs,
            }
        
        # Absolute distances in μm = relative * mean outer radius
        if return_absolute:
            absolute_distances = relative_distances * mean_radius_um

        # Optional plotting
        if plot:
            fig, ax = plt.subplots()
            if return_absolute and absolute_distances is not None:
                x_plot = absolute_distances  # µm
                xlabel = 'Distance (µm)'
            else:
                x_plot = relative_distances  # 0–1
                xlabel = 'Normalized Distance'

            # Default labels for common channels
            label_map = {'green': 'green', 'red': 'red', 'blue': 'blue'}
            
            # Use dual-axis plotting if multiple channels and not normalized
            use_dual_axis = len(channels) > 1 and not normalize
            
            if use_dual_axis:
                # Create second axis for multiple channels
                ax2 = ax.twinx()
                axes_list = [ax, ax2]
                # Track which channels are on which axis for labeling
                channels_on_axis = [[], []]  # [left_channels, right_channels]
            else:
                axes_list = [ax]
            
            for idx, channel in enumerate(channels):
                if channel not in intensity_smooth:
                    continue
                
                # Select axis: first channel on left, second on right (if dual axis)
                # For more than 2 channels, alternate: left, right, left, right, ...
                if use_dual_axis:
                    axis_idx = idx % 2  # 0 for left, 1 for right
                    current_ax = axes_list[axis_idx]
                    channels_on_axis[axis_idx].append(label_map.get(channel, channel.upper()))
                else:
                    current_ax = ax
                
                # Choose data based on normalize parameter
                if normalize:
                    mean = intensity_smooth[channel]['mean']
                    std = intensity_smooth[channel]['std']
                    ylabel = 'Normalized Intensity'
                else:
                    mean = intensity_smooth[channel]['mean_absolute']
                    std = intensity_smooth[channel]['std_absolute']
                    ylabel = 'Intensity (a.u.)'
                
                label = label_map.get(channel, channel.upper())
                line = current_ax.plot(x_plot, mean, label=label, color=channel)
                current_ax.fill_between(x_plot, mean - std, mean + std, alpha=0.2, color=channel)
            
            # Set y-axis labels after all channels are plotted
            if use_dual_axis:
                # Left axis (first channel and odd-indexed channels)
                if channels_on_axis[0]:
                    left_label = ', '.join(channels_on_axis[0])
                    # Find first channel on left axis
                    first_left_channel = next((ch for i, ch in enumerate(channels) if i % 2 == 0), None)
                    left_color = first_left_channel if first_left_channel else 'black'
                    ax.set_ylabel(f'{ylabel} ({left_label})', color=left_color)
                    ax.tick_params(axis='y', labelcolor=left_color)
                # Right axis (second channel and even-indexed channels)
                if channels_on_axis[1]:
                    right_label = ', '.join(channels_on_axis[1])
                    # Find first channel on right axis
                    first_right_channel = next((ch for i, ch in enumerate(channels) if i % 2 == 1), None)
                    right_color = first_right_channel if first_right_channel else 'black'
                    ax2.set_ylabel(f'{ylabel} ({right_label})', color=right_color)
                    ax2.tick_params(axis='y', labelcolor=right_color)
            else:
                ax.set_ylabel(ylabel)
            
            ax.set_xlabel(xlabel)
            if normalize:
                ax.set_title('Normalized Radial Profile', fontweight='bold')
            else:
                ax.set_title('Radial Profile (Absolute Intensities)', fontweight='bold')
            
            # Combine legends if using dual axis
            if use_dual_axis:
                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(lines1 + lines2, labels1 + labels2, loc='best')
            else:
                ax.legend(loc='best')
            
            plt.tight_layout()
            if savepath is not None:
                plt.savefig(savepath, transparent=True, dpi=300, bbox_inches='tight')
            plt.show()

        result = {
            'mean_radius': mean_radius_um,               # μm
            'intensity_profiles': intensity_smooth,
            'relative_distances': relative_distances,    # 0..1
        }
        if return_absolute and absolute_distances is not None:
            result['absolute_distances'] = absolute_distances
        return result

    def thr_radius(self, channel: str, thr: float, plot: bool = False,
                   normalize: bool = True, smoothing: float = 2,
                   angle_step: int = 2, background_subtraction: bool = True,
                   savepath: str | None = None) -> float | None:
        """
        Find the radius where the radial profile first drops below a threshold value.
        
        Searches from the outer boundary (r=1 or max radius) inward to find the first
        point where the intensity profile falls below the specified threshold.
        
        Args:
            channel: Fluorescence channel to analyze ('green', 'red', or 'blue')
            thr: Threshold value. If normalize=True, should be between 0 and 1.
                 If normalize=False, should be an absolute intensity value.
            plot: Whether to display a plot showing the profile and threshold crossing
            normalize: Whether to use normalized intensity profile (0-1) or absolute intensities
            smoothing: Gaussian smoothing parameter (sigma) for radial profile
            angle_step: Angular step size in degrees for radial profile calculation
            background_subtraction: If True, subtracts background based on outermost values
                                   (last 10% of profile points). Applied after normalization if normalize=True.
            savepath: Optional path to save the plot. If None, plot is only displayed.
            
        Returns:
            Radius value where threshold is first crossed:
                - If normalize=True: relative radius (0-1)
                - If normalize=False: absolute radius in μm
            Returns None if threshold is never crossed or if channel is not available
        """
        # Validate channel
        valid_channels = ['green', 'red', 'blue']
        if channel not in valid_channels:
            raise ValueError(f"Channel '{channel}' not supported. Options: {valid_channels}")
        if f'fluorescence_{channel}' not in self.image_path_dict:
            raise ValueError(f"Fluorescence channel '{channel}' not available in image data")
        
        # Get radial profile data
        profile_data = self.radial_profile(
            channels=[channel],
            plot=False,
            angle_step=angle_step,
            normalize=normalize,
            return_absolute=not normalize,  # Return absolute distances if not normalized
            smoothing=smoothing
        )
        
        # Extract profile for the channel
        if channel not in profile_data['intensity_profiles']:
            logger.warning(f"No intensity profile available for channel '{channel}'")
            return None
        
        intensity_profile = profile_data['intensity_profiles'][channel]
        
        # Choose appropriate intensity values based on normalize parameter
        if normalize:
            intensity = intensity_profile['mean'].copy()
        else:
            intensity = intensity_profile['mean_absolute'].copy()
        
        # Apply background subtraction if requested (after normalization)
        if background_subtraction:
            # Use outermost 10% of points to estimate background
            n_points = len(intensity)
            n_background = max(1, int(0.1 * n_points))  # At least 1 point
            background_region = intensity[-n_background:]  # Last n_background points
            background_value = np.mean(background_region)
            
            # Subtract background from entire profile
            intensity = intensity - background_value
            
            # Clip negative values to zero
            intensity = np.maximum(intensity, 0.0)
            
            logger.info(f"Background subtraction applied: background = {background_value:.4f} (from {n_background} outermost points)")
        
        # Get distance arrays
        relative_distances = profile_data['relative_distances']
        if not normalize and 'absolute_distances' in profile_data:
            distances = profile_data['absolute_distances']
        else:
            # If normalized, use relative distances; convert to absolute if needed later
            mean_radius = profile_data['mean_radius']
            distances = relative_distances * mean_radius
        
        # Search from inside (r=0) to outside (r=1) for the first transition
        # from above threshold to below threshold
        # This means searching the normal array from left to right (index 0 to -1)
        
        # Find points below and above threshold
        below_threshold = intensity < thr
        above_threshold = intensity >= thr
        
        # Check if threshold is ever crossed
        if not np.any(below_threshold):
            logger.warning(f"Threshold {thr} never crossed for channel '{channel}'. Profile minimum: {intensity.min():.4f}")
            return None
        
        if not np.any(above_threshold):
            # Profile is entirely below threshold - return outermost point
            logger.warning(f"Profile is entirely below threshold {thr} for channel '{channel}'. Returning outermost radius.")
            if normalize:
                return 1.0
            else:
                return float(distances[-1])
        
        # Find the first transition from above to below threshold (from inside to outside)
        # We look for a point that is below threshold AND has a previous point (more inside) that is above threshold
        # Since we're searching from inside (index 0) to outside (index -1), we check:
        # - Current point is below threshold
        # - Previous point (i-1, more inside) is above threshold
        transition_found = False
        first_below_idx = None
        
        for i in range(1, len(intensity)):
            # Current point is below threshold
            # Previous point (i-1, more inside) is above threshold
            if below_threshold[i] and above_threshold[i-1]:
                first_below_idx = i
                transition_found = True
                break
        
        # If no transition found, check edge cases
        if not transition_found:
            # Check if profile starts below threshold at the inside
            if below_threshold[0]:
                # Profile starts below threshold, so threshold radius is at the boundary
                first_below_idx = 0
            else:
                # This shouldn't happen, but handle it
                logger.warning(f"Could not find threshold crossing for channel '{channel}'. Using innermost point.")
                first_below_idx = 0
        
        # Interpolate to find exact crossing point if we found a transition
        if transition_found and first_below_idx > 0:
            # Linear interpolation between the point above threshold (i-1) and below threshold (i)
            i_above = first_below_idx - 1
            i_below = first_below_idx
            
            intensity_above = intensity[i_above]
            intensity_below = intensity[i_below]
            
            # Interpolate distance
            if normalize:
                dist_above = relative_distances[i_above]
                dist_below = relative_distances[i_below]
            else:
                dist_above = distances[i_above]
                dist_below = distances[i_below]
            
            # Linear interpolation: thr = intensity_above + alpha * (intensity_below - intensity_above)
            # where alpha is the interpolation factor
            if abs(intensity_below - intensity_above) > 1e-10:
                alpha = (thr - intensity_above) / (intensity_below - intensity_above)
                thr_radius_value = dist_above + alpha * (dist_below - dist_above)
            else:
                # If intensities are too close, use the point below threshold
                thr_radius_value = dist_below if normalize else distances[first_below_idx]
        else:
            # No transition found or at boundary - use discrete point
            if normalize:
                thr_radius_value = relative_distances[first_below_idx]
            else:
                thr_radius_value = distances[first_below_idx]
        
        # Optional plotting
        if plot:
            fig, ax = plt.subplots()
            
            # Plot profile
            if normalize:
                x_plot = relative_distances
                xlabel = 'Normalized Distance'
                ylabel = 'Normalized Intensity'
            else:
                x_plot = distances
                xlabel = 'Distance (µm)'
                ylabel = 'Intensity (a.u.)'
            
            ax.plot(x_plot, intensity, label=f'{channel} profile', color=channel, linewidth=2)
            ax.fill_between(x_plot, 
                          intensity - intensity_profile.get('std', 0),
                          intensity + intensity_profile.get('std', 0),
                          alpha=0.2, color=channel)
            
            # Plot threshold line
            ax.axhline(y=thr, color='red', linestyle='--', linewidth=2, label=f'Threshold = {thr:.3f}')
            
            # Mark threshold crossing point (use interpolated value)
            thr_x = thr_radius_value
            
            # Interpolate intensity at the crossing point for plotting
            if transition_found and first_below_idx > 0:
                # Use linear interpolation to get intensity at exact crossing point
                i_above = first_below_idx - 1
                i_below = first_below_idx
                intensity_above = intensity[i_above]
                intensity_below = intensity[i_below]
                
                if normalize:
                    dist_above = relative_distances[i_above]
                    dist_below = relative_distances[i_below]
                else:
                    dist_above = distances[i_above]
                    dist_below = distances[i_below]
                
                if abs(dist_below - dist_above) > 1e-10:
                    alpha = (thr_x - dist_above) / (dist_below - dist_above)
                    thr_intensity = intensity_above + alpha * (intensity_below - intensity_above)
                else:
                    thr_intensity = thr
            else:
                # Use threshold value if at boundary or no transition
                thr_intensity = thr
            
            ax.axvline(x=thr_x, color='red', linestyle=':', linewidth=1.5, 
                      label=f'Threshold radius = {thr_x:.3f}' + (' (rel)' if normalize else ' (µm)'))
            ax.plot(thr_x, thr_intensity, 'ro', markersize=10, label='Crossing point')
            
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(f'Threshold Radius Detection ({channel} channel)', fontweight='bold')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            if savepath is not None:
                plt.savefig(savepath, transparent=True, dpi=300, bbox_inches='tight')
            plt.show()
        
        return float(thr_radius_value)

    def fluorescence_profile(self, channels: list[str], 
                                   analysis_type: str = 'comprehensive',
                                   return_absolute: bool = True,
                                   plot: bool = False,
                                   smoothing: float = 0,
                                   savepath: str | None = None) -> dict:
        """
        Comprehensive analysis of fluorescence intensity profiles.
        
        Analyzes radial fluorescence profiles to extract meaningful biological
        parameters like inflection points, thresholds, and functional zones.
        
        Args:
            channels: List of fluorescence channels to analyze
            analysis_type: Type of analysis ('comprehensive', 'inflection', 'threshold', 'zones')
            return_absolute: Whether to return absolute distances in μm
            plot: Whether to display analysis plots
            smoothing: Gaussian smoothing parameter (sigma) for radial profile. If 0, no smoothing is applied.
            savepath: Optional path to save the plot. If None, plot is only displayed.
            
        Returns:
            Dictionary containing analysis results for each channel:
                - inflection_points: Critical points in the profile
                - thresholds: Intensity thresholds at different positions
                - functional_zones: Defined biological zones
                - statistics: Statistical measures (max, min, mean, std)
                - absolute_distances: Physical distances in μm (if requested)
        """
        # Get radial profile data
        profile_data = self.radial_profile(
            channels=channels, 
            plot=False, 
            angle_step=5,
            normalize=True,
            return_absolute=return_absolute,
            smoothing=smoothing
        )
        
        results = {}
        
        for channel in channels:
            if channel not in profile_data['intensity_profiles']:
                continue
                
            intensity = profile_data['intensity_profiles'][channel]['mean']
            std = profile_data['intensity_profiles'][channel]['std']
            
            # Relative distances (0 to 1)
            relative_dist = np.linspace(0, 1, len(intensity))
            
            # Absolute distances if available
            if return_absolute and 'absolute_distances' in profile_data:
                absolute_dist = profile_data['absolute_distances']
            else:
                absolute_dist = None
            
            # Use utility functions for analysis
            channel_results = analyze_fluorescence_profile(
                intensity, std, relative_dist, absolute_dist, analysis_type
            )
            
            results[channel] = channel_results
        
        # Store in analysis results
        self.analysis_results['fluorescence_analysis'] = results
        
        if plot:
            plot_fluorescence_analysis(results, analysis_type, savepath=savepath)
        
        return results

    def functional_radius(
        self,
        channels: list[str] = ['green', 'red'],
        uncertainty_thr: float = 0.3,
        plateau_thr: float | None = 0.5,
        increasing: bool = True,
        smoothing: float = 0,
        ):
        """
        Determine functional radii from fluorescence profiles by fitting sigmoids
        (A fixed to 1). Fit is only accepted if std at r0 < uncertainty_thr
        and optional plateau criterion is met.
        
        Args:
            channels: List of fluorescence channels to analyze
            uncertainty_thr: Uncertainty threshold for sigmoid fitting
            plateau_thr: Optional plateau threshold for sigmoid fitting
            increasing: If True, fits increasing sigmoid (0->1), if False fits decreasing sigmoid (1->0)
            smoothing: Gaussian smoothing parameter (sigma) for radial profile. If 0, no smoothing is applied.
        """
        analysis_results = self.radial_profile(
            channels=channels,
            return_absolute=True,
            plot=False,
            smoothing=smoothing,
        )

        functional_radii = {}
        # Outer radius in μm from radial_profile; will be used to convert relative → absolute
        outer_radius_um = analysis_results.get('mean_radius', np.nan)
        # Relative distance axis (0..1) for fitting, length matches intensity arrays
        # Use one channel's length to define the grid if available
        sample_len = None
        for ch in channels:
            prof = analysis_results['intensity_profiles'].get(ch)
            if prof is not None and 'mean' in prof:
                sample_len = len(prof['mean'])
                break
        if sample_len is None:
            raise ValueError("No intensity profiles available for sigmoid fit.")
        r_rel = np.linspace(0.0, 1.0, sample_len)

        # Store outer radius (mean_radius from radial_profile)
        outer_radius = analysis_results.get('mean_radius', np.nan)
        functional_radii['outer'] = outer_radius

        for ch in channels:
            profile = analysis_results['intensity_profiles'].get(ch)
            if profile is None or 'mean' not in profile:
                continue

            I_mean = profile['mean']
            I_std = profile.get('std')

            # --- Fit Sigmoid on relative distances ---
            fit_rel = fit_sigmoid_auto(
                r_rel,
                I_mean,
                I_std=I_std,
                uncertainty_thr=uncertainty_thr,
                plateau_thr=plateau_thr,
                increasing=increasing,
            )

            ch_radii = {"fit": fit_rel}  # fit['r0'] and ['s'] are relative (0..1)

            if fit_rel["fit_success"]:
                r0_rel, s_rel = fit_rel["r0"], fit_rel["s"]
                # Relative transitions
                transition_outer_rel = r0_rel + 1.3 * s_rel
                transition_inner_rel = r0_rel - 1.3 * s_rel
                # Convert to absolute (μm) using outer radius
                if np.isfinite(outer_radius_um):
                    r0_abs = r0_rel * outer_radius_um
                    s_abs = s_rel * outer_radius_um
                    transition_outer_abs = transition_outer_rel * outer_radius_um
                    transition_inner_abs = transition_inner_rel * outer_radius_um
                else:
                    r0_abs = np.nan
                    s_abs = np.nan
                    transition_outer_abs = np.nan
                    transition_inner_abs = np.nan

                # Store both relative and absolute (backward-compatible keys include absolute)
                ch_radii.update({
                    "fit_success": True,
                    "fit_relative": {"r0": r0_rel, "s": s_rel, "r2": fit_rel.get("r2", np.nan)},
                    "transition_outer_relative": transition_outer_rel,
                    "transition_inner_relative": transition_inner_rel,
                    # Absolute values
                    "fit_absolute": {"r0": r0_abs, "s": s_abs, "r2": fit_rel.get("r2", np.nan)},
                    "transition_outer": transition_outer_abs,  # backward-compatible absolute key
                    "transition_inner": transition_inner_abs,
                    "fit_quality": fit_rel.get("r2", np.nan),
                })
            else:
                ch_radii.update({
                    "fit_success": False,
                    "fit_relative": {"r0": np.nan, "s": np.nan, "r2": np.nan},
                    "transition_outer_relative": np.nan,
                    "transition_inner_relative": np.nan,
                    "fit_absolute": {"r0": np.nan, "s": np.nan, "r2": np.nan},
                    "transition_outer": np.nan,
                    "transition_inner": np.nan,
                    "fit_quality": np.nan,
                })

            functional_radii[ch] = ch_radii

        self.analysis_results['functional_radii'] = functional_radii
        return functional_radii

    @property
    def scaled_contour(self) -> np.ndarray:
        """Get the contour scaled to physical coordinates.
        
        Converts contour from pixel coordinates to physical coordinates (μm)
        based on the image size.
        
        Returns:
            Array of shape (N,2) containing scaled (x,y) contour coordinates in μm
        """
        if self.contour is None:
            return None
        if not hasattr(self, '_pixel_dimensions') or self._pixel_dimensions is None:
            raise ValueError("Pixel dimensions not available for scaling contour.")

        height_px, width_px = self._pixel_dimensions
        width_um, height_um = self.image_size

        scaled_contour = self.contour.astype(float).copy()
        scaled_contour[:, 0] = (scaled_contour[:, 0] / max(1, width_px - 1)) * width_um
        # flip y-axis so origin matches matplotlib extent
        scaled_contour[:, 1] = height_um - (scaled_contour[:, 1] / max(1, height_px - 1)) * height_um
        return scaled_contour

    @property
    def area(self) -> float | None:
        """Calculate the area enclosed by the spheroid contour.
        
        Uses the shoelace formula to compute the area enclosed by the contour,
        scaled to physical units (μm²).
        
        Returns:
            Area in μm² if contour exists, None otherwise
        """
        if self.contour is None or len(self.contour) < 3:
            return None

        # Polygon area in pixel units (shoelace formula)
        x = self.contour[:, 0].astype(float)
        y = self.contour[:, 1].astype(float)
        area_px = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

        # Map pixels -> micrometers using image_size and cached pixel dimensions
        if hasattr(self, '_pixel_dimensions') and self._pixel_dimensions is not None:
            h_px, w_px = self._pixel_dimensions
        else:
            # Fallback: load image if dimensions not cached (shouldn't happen normally)
            sample_img = read_image(str(self.image_path_dict['brightfield']))
            if sample_img is not None:
                h_px, w_px = sample_img.shape[:2]
                # Cache for future use
                self._pixel_dimensions = (h_px, w_px)
            else:
                # Ultimate fallback: treat 1 px = 1 μm
                sx = 1.0
                sy = 1.0
                area_um2 = area_px * sx * sy
                return area_um2
        
        if hasattr(self, 'image_size') and self.image_size is not None:
            # Stay consistent with plotting convention used elsewhere in this class
            height_um = float(self.image_size[1])
            width_um = float(self.image_size[0])
            sx = width_um / max(1, w_px)
            sy = height_um / max(1, h_px)
        else:
            # Fallback: treat 1 px = 1 μm
            sx = 1.0
            sy = 1.0

        area_um2 = area_px * sx * sy
        return area_um2

    @property
    def radius(self) -> float | None:
        """Calculate the effective radius of the spheroid.
        
        Computes radius as r_eff = sqrt(Area/π), giving the radius of a circle
        with equivalent area to the spheroid.
        
        Returns:
            Effective radius in μm if valid contour exists, None otherwise
        """
        if self.contour is not None:
            radius = np.sqrt(self.area/np.pi)
            if radius > 0:
                return radius
            else:
                return None
        else:
            return None

    def metric_fluorescence(self, 
                           fluorescence_color: str = 'green', 
                           ignore_border: bool = False) -> tuple[float | None, float | None]:
        """
        Calculate fluorescence intensity metrics within the spheroid contour.
        
        Args:
            fluorescence_color: Color channel to analyze ('green', 'red', 'blue')
            ignore_border: Whether to proceed if spheroid touches image border
            
        Returns:
            Tuple of (cumulative intensity, mean intensity) or (None, None) if invalid
        """
        # Early return if touching border and not ignored
        if self.contour_touches_border and not ignore_border:
            return None, None

        # Find matching channel
        channel_key = next(
            (channel for channel in self.image_path_dict 
             if fluorescence_color in channel), None)
        
        if not channel_key:
            raise ValueError(f'No fluorescence channel found for color {fluorescence_color}')

        # Get and process fluorescence image (grayscale)
        fluo_img_color = read_image(str(self.image_path_dict[channel_key]), cv2.IMREAD_COLOR)
        if fluo_img_color is None:
            return None, None
        fluo_img = cv2.cvtColor(fluo_img_color, cv2.COLOR_BGR2GRAY)

        # Create mask from contour
        mask = np.zeros_like(fluo_img, dtype=np.uint8)
        contour_clipped = np.clip(self.contour, 
                                 [0, 0], 
                                 [fluo_img.shape[1] - 1, fluo_img.shape[0] - 1])
        contour_int = np.round(contour_clipped).astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [contour_int], 255)

        # Calculate intensity metrics
        intensity_values = fluo_img[mask == 255]
        return np.sum(intensity_values), np.mean(intensity_values)


    def segmentation_detectron(self, channel: str, thres: float = .65, border_margin: int = 5) -> np.ndarray | None:
        """Segment spheroid using Detectron2 instance segmentation model.

        Args:
            channel: Image channel to use for segmentation.
            thres: Detection confidence threshold.
            border_margin: Margin in pixels for border detection.

        Returns:
            np.ndarray of contour
        """

        # 1. Input validation and image loading
        if channel not in self.image_path_dict:
            raise ValueError(f"Channel '{channel}' not found.")

        image = read_image(str(self.image_path_dict[channel]))
        if image is None:
            raise IOError(f"Failed to load image from path: {self.image_path_dict[channel]}")

        # 2. Get and configure predictor
        predictor = self._get_predictor()
        cfg = predictor.cfg
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = thres
        predictor.cfg = cfg

        # 3. Get predictions and filter
        outputs = predictor(image)
        instances = outputs["instances"].to("cpu")

        if len(instances) == 0:
            logger.warning("No spheroid detected. Check your threshold or image quality.")
            self.contour = None
            self.contour_touches_border = None
            return None

        # Select the best prediction (highest score), robust across versions
        fields = instances.get_fields()
        if 'scores' in fields and len(instances) > 0:
            try:
                idx = int(fields['scores'].argmax().item())
            except Exception:
                idx = int(np.argmax(fields['scores']))
        else:
            idx = 0

        # Use list/slice indexing to preserve tensor dimensions and avoid Boxes dim errors
        best_group = instances[idx:idx+1]
        best_score = None
        if 'scores' in fields and len(instances) > 0:
            try:
                best_score = float(fields['scores'][idx].item())
            except Exception:
                try:
                    best_score = float(np.asarray(fields['scores'])[idx])
                except Exception:
                    best_score = None

        # Ensure pred_masks exist
        if not hasattr(best_group, 'pred_masks') or best_group.pred_masks is None or len(best_group.pred_masks) == 0:
            logger.warning("Best instance has no pred_masks. Skipping.")
            self.contour = None
            self.contour_touches_border = None
            return None

        pred_mask = best_group.pred_masks[0].numpy().squeeze()

        # 4. Find the main contour from the mask
        contours_found, _ = cv2.findContours(pred_mask.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours_found:
            logger.warning("No contour found for the detected spheroid mask.")
            self.contour = None
            self.contour_touches_border = None
            return None

        # Select the largest contour if multiple were found
        main_contour = max(contours_found, key=cv2.contourArea).squeeze()

        cleared_mask = clear_border(pred_mask, buffer_size=border_margin)
        touches_border = not np.array_equal(pred_mask, cleared_mask)
        self.contour_touches_border = touches_border
        self.contour = main_contour.astype(np.float32)

        # 6. Store and return results
        self.analysis_results['detectron_threshold'] = thres
        self.analysis_results['detectron_score'] = best_score

        if best_score is not None:
            logger.info(f"Spheroid detected with score: {best_score:.2f}. Touches border: {touches_border}.")
        else:
            logger.info(f"Spheroid detected. Touches border: {touches_border}.")

        return self.contour

    def segmentation_detectron_alt(self, channel: str, thres: float = .65, border_margin: int = 5):
        """Segment spheroid using Detectron2 instance segmentation model.
        
        Uses a pre-trained Mask R-CNN model to detect and segment the spheroid.
        
        Args:
            channel: Image channel to use for segmentation
            thres: Detection confidence threshold
            
        Returns:
            tuple: (contour, touches_border) where contour is array of boundary points
            and touches_border indicates if spheroid extends beyond image bounds
        """
        # Option 1: Using pkg_resources
        weight = pkg_resources.resource_filename('SpheroidPy', 'weights/detectron_model_final.pth')

        # Option 2: Using __file__ and Path
        #weight = str(Path(__file__).parent.parent / 'weights' / 'detectron_model_final.pth')

        cfg = get_cfg()
        cfg.MODEL.DEVICE = 'cpu'

        cfg.DATALOADER.NUM_WORKERS = 2
        cfg.SOLVER.IMS_PER_BATCH = 4
        cfg.SOLVER.MAX_ITER = 500
        cfg.SOLVER.STEPS = []
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = os.path.join(weight)  # path to the model we just trained
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = thres  # set a custom testing threshold

        predictor = DefaultPredictor(cfg)

        image = read_image(str(self.image_path_dict[channel]))
        height, width = image.shape[0], image.shape[1]
        self._height, self._width = height, width

        #######
        outputs = predictor(image)
        #######

        contours = []
        for pred_mask in outputs["instances"].to("cpu").pred_masks:
            # pred_mask is of type torch.Tensor, and the values are boolean (True, False)
            # Convert it to a 8-bit numpy array, which can then be used to find contours
            mask = pred_mask.numpy().astype('uint8')
            # Speichern der Maske als TIFF-Datei
            #tiff.imwrite(f'/Users/cedric/Desktop/Training_Data/Hep3B/mask/Hep3B_{self.well}_mask.tif', mask)
            contour, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

            contours.append(contour[0])  # contour is a tuple (OpenCV 4.5.2), so take the first element which is the array of contour points

        contour = contours[0].reshape(-1, 2).astype(np.float32)
        border_bool = touches_border(contour, width - 10, height - 10)
        #print(border_bool, width, height)
        if border_bool:
            contour = remove_border_points(contour, width - 10, height - 10)
        return contour, border_bool

    def segmentation_thresholding(self, channel: str, thres: float = 1.35,
                                  thres_yen: bool = True, border_margin: int = 5) -> np.ndarray | None:
        """Segment spheroid using thresholding methods.

        Args:
            channel: Image channel to use for segmentation
            thres: Threshold multiplier to adjust segmentation sensitivity
            thres_yen: Whether to use Yen's method (True) or Otsu's method (False)
            border_margin: Margin in pixels for border detection

        Returns:
            contour array
        """
        if channel not in self.image_path_dict:
            raise ValueError(f"Channel '{channel}' not found.")

        img_path = str(self.image_path_dict[channel])
        img = read_image(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise IOError(f"Failed to load image from path: {img_path}")

        # Normalize fluorescence images to [0, 255] for consistent thresholding
        if 'fluorescence' in channel:
            img_min, img_max = img.min(), img.max()
            # Safety check: avoid division by zero or very small ranges
            if img_max - img_min < 10:  # Very low contrast image
                logger.warning(f"Very low contrast image for {channel} (range: {img_max - img_min}). Using original image.")
            else:
                # Min-max normalization to [0, 255]
                img = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)

        # 1. Automatisierte Helligkeitsbestimmung
        # Bestimme, ob das Spheroid dunkler oder heller ist
        # Annahme, die auf dem Kanalnamen basiert.
        if channel == 'brightfield':
            is_dark = True
        elif 'fluorescence' in channel:
            is_dark = False
        else:
            # Fallback-Logik, wenn der Kanalname unbekannt ist.
            # Hier könnte eine Heuristik wie die Histogramm-Analyse angewendet werden.
            # Allerdings ist das Risiko, dass sie fehlschlägt, hoch.
            # Eine bessere Alternative ist eine explizite Fehlermeldung oder die Verwendung eines
            # Konfigurationsparameters.
            is_dark = np.mean(img) < 128
            logger.warning(f"Unknown channel type '{channel}'. Using heuristic to determine brightness.")

        # 2. Bilde den Schwellenwert
        if thres_yen:
            thres_val = threshold_yen(img)
        else:
            thres_val = threshold_otsu(img)

        final_thres = thres_val * thres

        bool_mask = img < final_thres if is_dark else img > final_thres

        # 3. Bereinige die Maske mit morphologischen Operationen
        bool_mask = binary_dilation(bool_mask, iterations=3)
        bool_mask = binary_fill_holes(bool_mask)
        bool_mask = remove_small_objects(bool_mask, min_size=1000)
        bool_mask = binary_closing(bool_mask, iterations=3)
        bool_mask = binary_fill_holes(bool_mask)

        # 4. Finde das größte Objekt in der bereinigten Maske
        labeled_mask, num_labels = label(bool_mask)
        if num_labels == 0:
            logger.warning("No objects found after thresholding and cleaning. Try adjusting parameters.")
            self.contour = None
            self.contour_touches_border = None
            logger.warning("No objects found after thresholding.")
            return None

        sizes = np.bincount(labeled_mask.ravel())
        main_label = sizes[1:].argmax() + 1
        main_mask = labeled_mask == main_label

        # 5. Finde die Kontur des Hauptobjekts
        contours = find_contours(main_mask, 0.5)

        if not contours:
            logger.warning("No contours found for the main object.")
            self.contour = None
            self.contour_touches_border = None
            return self.contour

        main_contour = max(contours, key=len)
        # skimage returns (row, col) = (y, x). Convert to (x, y) to match plotting & detectron.
        main_contour = np.array(main_contour, dtype=np.float32)
        self.contour = np.column_stack((main_contour[:, 1], main_contour[:, 0])).astype(np.float32)

        # 6. Prüfe, ob die Kontur den Rand berührt
        cleared_mask = clear_border(main_mask, buffer_size=border_margin)
        touches_border = not np.array_equal(main_mask, cleared_mask)
        self.contour_touches_border = touches_border

        logger.info(f"Spheroid segmentation complete for channel '{channel}'. Contour touches border: {touches_border}")

        return self.contour
        """
           Image segmentation function to create a custom polygon mask, and evalute radius and position of the masked object.
           Need to use %matplotlib qt in jupyter notebook
           Args:
               img(array): Grayscale image as a Numpy array
           Returns:
               dict: Dictionary with keys: mask, radius, centroid (x/y)
        """
        from roipoly import RoiPoly
        matplotlib.use("TkAgg")

        img = read_image(str(self.image_path_dict[channel]))
        img_greyscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height = img.shape[0]
        width = img.shape[1]
        # click polygon mask interactive
        plt.ion()
        plt.imshow(img, extent=[0, width, height, 0])
        plt.text(0.5, 1.05, 'Click Polygon Mask with left click, finish with right click', fontsize=12,
                 horizontalalignment='center',
                 verticalalignment='center', c='darkred', transform=plt.gca().transAxes)  # transform = ax.transAxes)
        # click the mask here
        my_roi = RoiPoly(color='r')
        # Extract mask and segementation details
        mask = np.flipud(my_roi.get_mask(img_greyscale))  # flip mask due to imshow
        # print(dict_['mask'])
        mask_uint8 = mask.astype(np.uint8) * 255
        contour, _ = cv2.findContours(mask_uint8, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        contour = contour[0].reshape(-1, 2).astype(np.float32)
        # Kontur an der y-Achse spiegeln, so dass sie zu bild koordinaten passt
        contour[:, 1] = mask_uint8.shape[0] - contour[:, 1]
        print('contour:',contour)
        # hide pop-up windows
        #plt.close()
        plt.ioff()
        plt.close('all')
        # return dictionary containing spheroid information
        print('inline :)')
        matplotlib.use('module://matplotlib_inline.backend_inline')  # Setzt das Backend auf 'inline' für Jupyter
        return contour, False

    def segmentation_manual(self, channel: str):
        """
        Launch interactive manual segmentation and set self.contour.
        channel: one of 'brightfield', 'fluorescence_green', 'fluorescence_red', 'fluorescence_blue'
        """
        # Prepare images dict in expected keys: 'brightfield','green','red','blue'
        images = {}
        # Load brightfield if available
        if 'brightfield' in self.image_path_dict:
            bf = read_image(str(self.image_path_dict['brightfield']))
            images['brightfield'] = bf
        # Map fluorescence channels
        for key, short in [('fluorescence_green','green'), ('fluorescence_red','red'), ('fluorescence_blue','blue')]:
            if key in self.image_path_dict:
                images[short] = read_image(str(self.image_path_dict[key]))

        # Derive initial channel for UI
        init_map = {
            'brightfield': 'brightfield',
            'fluorescence_green': 'green',
            'fluorescence_red': 'red',
            'fluorescence_blue': 'blue'
        }
        initial = init_map.get(channel, 'brightfield')

        # Launch tool
        root = tk.Tk()
        app = ManualSegmentation(root, images=images, initial_channel=initial)
        # Match the example behavior: let user close the window manually
        root.mainloop()

        # Collect results
        if app.contour is not None:
            self.contour = app.contour.astype(np.float32)
            self.contour_touches_border = bool(app.contour_touches_border)
        #return self.contour, self.contour_touches_border

    def diffusion_stationary(self, boundary_value: float = 10, 
                            diffusion_rate: float = 1000, 
                            reaction_rate: float = .1,
                            plot: bool = True,
                            accuracy: int = 10,
                            threshold_concentration: float | None = None,
                            savepath: str | None = None):
        """Solve the stationary diffusion-reaction equation for the spheroid.
        
        Solves ∇⋅(D∇c)-kc=0 using finite element method, where:
        - c is concentration
        - D is diffusion coefficient
        - k is reaction rate
        
        Args:
            boundary_value: Concentration at spheroid boundary
            diffusion_rate: Diffusion coefficient in μm²/s
            reaction_rate: Reaction rate in mol/μm²/s
            plot: Whether to display solution plot
            accuracy: Mesh resolution parameter
            threshold_concentration: plotting a line at the threshold concentration
            savepath: Optional path to save the plot. If None, plot is only displayed.
            
        Returns:
            tuple: (points, triangles, solution) where:
                points: Mesh vertex coordinates
                triangles: Mesh connectivity
                solution: Concentration values at mesh points
        """
        # Scale diffusion parameters from physical units (μm²/s) to mesh units
        # Get pixel dimensions for scaling
        sample_img = read_image(str(self.image_path_dict['brightfield']))
        if sample_img is None:
            raise ValueError("Cannot load brightfield image for scaling")
        
        height_px, width_px = sample_img.shape[:2]
        
        # Calculate scaling factors: μm to mesh units
        if hasattr(self, 'image_size') and self.image_size is not None:
            height_um = float(self.image_size[1])  # height in μm
            width_um = float(self.image_size[0])   # width in μm
            scale_x = width_px / width_um   # pixels per μm
            scale_y = height_px / height_um # pixels per μm
            # Use average scaling for isotropic diffusion
            scale_avg = (scale_x + scale_y) / 2
        else:
            # Fallback: assume 1 pixel = 1 μm
            scale_avg = 1.0
        
        # Scale diffusion coefficient: D_mesh = D_physical * scale²
        diffusion_rate_scaled = diffusion_rate * (scale_avg ** 2)
        # Scale reaction rate: k_mesh = k_physical * scale²  
        reaction_rate_scaled = reaction_rate * (scale_avg ** 2)

        points, triangles = self.mesh(300, False, accuracy)
        A = compute_laplace_matrix(points, triangles, reaction_rate_scaled, diffusion_rate_scaled)
        b = np.zeros(len(points))  # Kein Quellterm im Inneren (steady-state)
        apply_dirichlet_boundary_conditions(A, b, get_boundary_nodes(self.contour[::accuracy], points), boundary_value)  # Dirichlet-Randbedingungen
        solution = spsolve(A.tocsr(), b)
        if plot:
            # Background image and physical scaling using image_size (H, W in µm)
            bg_img = read_image(str(self.image_path_dict['brightfield']))
            h_px, w_px = bg_img.shape[:2]
            if hasattr(self, 'image_size') and self.image_size is not None:
                # Follow the same convention used in plot(): height_um from image_size[1], width_um from image_size[0]
                height_um = float(self.image_size[1])
                width_um = float(self.image_size[0])
            else:
                height_um = float(h_px)
                width_um = float(w_px)
            sx = width_um / max(1, w_px)
            sy = height_um / max(1, h_px)

            # Scale FEM points to µm for plotting
            pts_um = points.copy()
            pts_um[:, 0] = pts_um[:, 0] * sx
            pts_um[:, 1] = pts_um[:, 1] * sy

            plt.imshow(cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB), extent=[0, width_um, height_um, 0])
            fill = plt.tricontourf(pts_um[:, 0], pts_um[:, 1], triangles, solution, levels=100, cmap='hot', alpha=.5, norm=Normalize(vmin=0, vmax=boundary_value, clip=False))
            cbar = plt.colorbar(fill, label='Nutrient Concentration (a.u.)', norm=Normalize(vmin=0, vmax=boundary_value, clip=False))
            cbar.ax.collections[0].set_edgecolor("face")
            # Add a specific dotted contour line for a certain value
            if threshold_concentration is not None:
                specific_value = threshold_concentration
                dotted_contour = plt.tricontour(pts_um[:, 0], pts_um[:, 1], triangles, solution, levels=[specific_value], colors='blue', linestyle='dotted', linewidths=1)
            plt.plot([], [], color='blue', linestyle=':', linewidth=1, label='critical $c_{~Nutrient}$')
            plt.legend()

            plt.xlim(0, width_um)
            plt.ylim(height_um, 0)

            plt.xlabel('x-position [µm]')
            plt.ylabel('y-position [µm]')
            plt.title('Steady-State Diffusion with Reaction', fontweight='bold')
            if savepath is not None:
                plt.savefig(savepath, transparent=True, dpi=300, bbox_inches='tight')
            plt.show()
        return points, triangles, solution

    def show(self, channels: list | None = None, show_contour: bool = True, figsize: tuple | None = None,
             savepath: str | None = None):
        """
        Plot selected imaging channels side-by-side and optionally overlay the spheroid contour.

        Args:
            channels: List of channel identifiers to plot in the given order. Supported names:
                'brightfield', 'fluorescence_green', 'fluorescence_red', 'fluorescence_blue'.
                Shorthands 'green', 'red', 'blue' are also accepted.
                If None, plots all available channels in default order.
            show_contour: If True and a contour exists, overlay it on each subplot.
            figsize: Figure size (width, height). If None, auto-scales with number of channels.
            savepath: Optional path to save the plot. If None, plot is only displayed.
        """
        # Determine which channels to plot and in what order
        default_order = ['brightfield', 'fluorescence_green', 'fluorescence_red', 'fluorescence_blue']
        shorthand_map = {
            'green': 'fluorescence_green',
            'red': 'fluorescence_red',
            'blue': 'fluorescence_blue',
        }

        if channels is None:
            channels_to_plot = [c for c in default_order if c in self.image_path_dict]
        else:
            resolved = []
            for ch in channels:
                key = shorthand_map.get(ch, ch)
                if key in self.image_path_dict:
                    resolved.append(key)
            channels_to_plot = resolved

        if not channels_to_plot:
            raise ValueError("No valid channels to plot. Check provided channel names and availability.")

        n = len(channels_to_plot)
        if figsize is None:
            figsize = (4*n, 4)

        fig, axes = plt.subplots(1, n, figsize=figsize)
        if n == 1:
            axes = [axes]

        # Load and show each channel
        for i, (ax, ch) in enumerate(zip(axes, channels_to_plot)):
            img = read_image(str(self.image_path_dict[ch]), cv2.IMREAD_UNCHANGED)
            if img is None:
                ax.set_title(f"{ch} (not found)")
                ax.axis('off')
                continue

            # Convert for display
            if img.ndim == 2:
                disp = img
                cmap = 'gray'
            else:
                # BGR -> RGB for display
                disp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                cmap = None

            height_px, width_px = img.shape[:2]
            # Determine scaling from pixels to micrometers using image_size (H, W in µm)
            if hasattr(self, 'image_size') and self.image_size is not None:
                height_um = float(self.image_size[1])
                width_um = float(self.image_size[0])
                sx = width_um / max(1, width_px)
                sy = height_um / max(1, height_px)
            else:
                # Fallback: treat pixels as micrometers 1:1
                height_um = float(height_px)
                width_um = float(width_px)
                sx = 1.0
                sy = 1.0

            # Display image with physical extent in µm, keeping top-left origin
            ax.imshow(disp, extent=[0, width_um, height_um, 0], cmap=cmap)
            ax.set_title(ch, fontweight='bold')
            #ax.set_xlim(0, width_um)
            #ax.set_ylim(height_um, 0)
            ax.set_xlabel('x [µm]')
            ax.set_ylabel('y [µm]')

            # Overlay contour if requested and available – visually close by linking last to first
            if show_contour and getattr(self, 'contour', None) is not None and len(self.contour) > 1:
                try:
                    contour = self.contour
                    cx = contour[:, 0].astype(float) * sx
                    cy = contour[:, 1].astype(float) * sy
                    if cx[0] != cx[-1] or cy[0] != cy[-1]:
                        cx = np.r_[cx, cx[:1]]
                        cy = np.r_[cy, cy[:1]]
                    ax.plot(cx, cy, '-', color='yellow', linewidth=1.5)
                except Exception:
                    pass

        plt.tight_layout()
        if savepath is not None:
            plt.savefig(savepath, transparent=True, dpi=300, bbox_inches='tight')
        plt.show()
    def mesh(self, max_area: float = 500, plot: bool = True, accuracy: int = 1,
             savepath: str | None = None):
        """Generate a triangular mesh of the spheroid contour.
        
        Uses MeshPy to create a triangulation of the region enclosed by the contour.
        
        Args:
            max_area: Maximum allowed area for triangles in the mesh
            plot: Whether to display a plot of the generated mesh
            accuracy: Sample every nth point from contour for mesh generation
            savepath: Optional path to save the plot. If None, plot is only displayed.
            
        Returns:
            tuple: (points, triangles) where points is an array of vertex coordinates
            and triangles is an array of triangle vertex indices
        """
        try:
            from meshpy.triangle import MeshInfo, build
        except ImportError:
            print("MeshPy is not installed. This might lead to errors in the mesh generation if used for PDE simulations.")
            return None, None
        # Setup für MeshPy
        mesh_info = MeshInfo()
        mesh_info.set_points(self.contour[::accuracy].tolist())
        # Definiere Kanten der Kontur
        segments = [(i, (i + 1) % len(self.contour[::accuracy])) for i in range(len(self.contour[::accuracy]))]
        mesh_info.set_facets(segments)
        # Generiere das Mesh
        mesh = build(mesh_info, max_volume=max_area)
        points = np.array(mesh.points)
        triangles = np.array(mesh.elements)
        # Plot Mesh (scaled to µm if image_size available)
        if plot:
            bf = read_image(str(self.image_path_dict['brightfield']))
            h_px, w_px = bf.shape[:2]
            if hasattr(self, 'image_size') and self.image_size is not None:
                height_um = float(self.image_size[1])
                width_um = float(self.image_size[0])
            else:
                height_um = float(h_px)
                width_um = float(w_px)
            sx = width_um / max(1, w_px)
            sy = height_um / max(1, h_px)
            pts_um = points.copy()
            pts_um[:, 0] = pts_um[:, 0] * sx
            pts_um[:, 1] = pts_um[:, 1] * sy
            plot_mesh(pts_um, triangles, savepath=savepath)
        return points, triangles

    def _get_predictor(self):
        """Lazy-load the Detectron2 predictor."""
        if self.predictor is None:
            logger.info("Initializing Detectron2 predictor. This may take a moment.")
            self.predictor = get_detectron_predictor()
        return self.predictor


