from typing import Union
    import logging

# Detectron2 utilities
from SpheroidPy.utils.detectron_utils import (
    DETECTRON2_AVAILABLE,
    get_detectron_predictor
)

# HRNet utilities
try:
    from SpheroidPy.utils.hrnet_utils import (
        TORCH_AVAILABLE as HRNET_AVAILABLE,
        load_hrnet_model,
        preprocess_image as hrnet_preprocess_image
    )
    if HRNET_AVAILABLE:
        import torch
except ImportError:
    HRNET_AVAILABLE = False
    load_hrnet_model = None
    hrnet_preprocess_image = None
    torch = None

# Other imports that don't depend on detectron2
import os
import cv2
import h5py
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import tkinter as tk

import pkg_resources

from scipy.ndimage import label, center_of_mass
from scipy.sparse.linalg import spsolve
from matplotlib.colors import Normalize
from skimage.measure import find_contours
from skimage.segmentation import clear_border
from skimage.filters import threshold_yen, threshold_otsu, threshold_local
from skimage.morphology import (
    binary_opening,
    binary_closing,
    remove_small_objects,
    binary_dilation,
)
from scipy.ndimage import binary_fill_holes

from scipy.ndimage import gaussian_filter1d



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
from SpheroidPy.utils.config import Config, AISegmentationType

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
        Height and width of the image in µm (from any available channel).
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
    >>> # With brightfield (traditional usage)
    >>> img = SpheroidImage(
    ...     brightfield="path/to/brightfield.tif",
    ...     fluorescence_green="path/to/green.tif",
    ...     image_size=(1080, 1920)
    ... )
    >>> 
    >>> # Without brightfield (at least one channel required)
    >>> img = SpheroidImage(
    ...     fluorescence_green="path/to/green.tif",
    ...     fluorescence_red="path/to/red.tif",
    ...     image_size=(1080, 1920)
    ... )
    """

    def __init__(self, brightfield: Path | str = None, **kwargs):
        """
        Initialize a SpheroidImage.

        Parameters
        ----------
        brightfield : Path or str, optional
            Path to the brightfield image. If not provided, at least one other channel
            must be provided (fluorescence_green, fluorescence_red, or fluorescence_blue).
        fluorescence_green : Path or str, optional
            Path to green fluorescence image.
        fluorescence_red : Path or str, optional
            Path to red fluorescence image.
        fluorescence_blue : Path or str, optional
            Path to blue fluorescence image.
        image_size : tuple of int, optional
            Image size as (height, width). If not provided, inferred from available images.
        hdf5_path : str or Path, optional
            Path to associated HDF5 file.
        hdf5_key : str, optional
            Key for accessing data in HDF5 file.
        """
        self.image_path_dict = {}
        self._initialize_channels(brightfield, **kwargs)
        self._initialize_attributes(**kwargs)
        # Lazy-loaded predictors and caches
        self.predictor = None  # Detectron2 predictor
        self.hrnet_model = None  # HRNet model
        # Storage for multiple-spheroid contours
        self._contours: list[np.ndarray] = []
        self._contours_scaled: list[np.ndarray] = []

    def _initialize_channels(self, brightfield: Path | str = None, **kwargs):
        """
        Initialize the image channels for the spheroid.

        Parameters
        ----------
        brightfield : Path or str, optional
            Path to the brightfield image file. If not provided, at least one other
            channel must be provided.
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
        - At least one channel must be successfully loaded.
        
        Raises
        ------
        ValueError: If no channels are provided or all provided channels are invalid.
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
        
        # Ensure at least one channel was loaded
        if not self.image_path_dict:
            raise ValueError(
                "No valid image channels found. Please provide at least one of: "
                "brightfield, fluorescence_green, fluorescence_red, or fluorescence_blue"
            )


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
        # Try brightfield first, then any available channel
        provided_size = kwargs.get('image_size')
        pixel_dims = None
        
        # Try brightfield first
        if 'brightfield' in self.image_path_dict:
            pixel_img = read_image(str(self.image_path_dict['brightfield']))
            if pixel_img is not None:
                pixel_dims = pixel_img.shape[:2]  # (height_px, width_px)
        
        # Fallback to any available channel if brightfield not available
        if pixel_dims is None:
            for channel_name in ['fluorescence_green', 'fluorescence_red', 'fluorescence_blue']:
                if channel_name in self.image_path_dict:
                    pixel_img = read_image(str(self.image_path_dict[channel_name]))
                    if pixel_img is not None:
                        pixel_dims = pixel_img.shape[:2]  # (height_px, width_px)
                        break
        
        self._pixel_dimensions = pixel_dims

        if provided_size is not None:
            width_um, height_um = provided_size
            self.image_size = (float(width_um), float(height_um))
        else:
            if pixel_dims is None:
                raise ValueError("Cannot determine image size. No valid images could be loaded.")
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
                    Example: [('thresholding', 'fluorescence_green'), ('ai', 'brightfield')]
                
                Supported methods:
                - 'thresholding': Uses threshold-based segmentation
                - 'ai': Uses AI segmentation (Detectron2 or HRNet, determined by Config.ai_segmentation)
                
                Note: When method is 'ai', the actual AI model used is determined by
                the global configuration Config.ai_segmentation. The default is automatically
                set based on availability: Detectron2 if available, otherwise HRNet.
                To change this, use:
                    from SpheroidPy.utils.config import Config, AISegmentationType
                    Config.ai_segmentation = AISegmentationType.HRNET  # or AISegmentationType.DETECTRON
            reconstruct_border: Whether to reconstruct spheroid border when it
                extends beyond image bounds
            border_margin: Margin in pixels to use when detecting border contact
            **kwargs: Method-specific parameters:
                For thresholding:
                    threshold: Intensity threshold multiplier (default: 1.2)
                    use_yen: Whether to use Yen's method (default: True)
                For AI (Detectron2 or HRNet, depending on Config.ai_segmentation):
                    confidence: Detection confidence threshold (default: 0.65)
        
        Examples:
            # Simple thresholding
            img.segmentation('fluorescence_green')
            
            # AI segmentation (uses Config.ai_segmentation to determine method)
            # Default: Detectron2 if available, otherwise HRNet
            img.segmentation(('ai', 'brightfield'), confidence=0.7)
            
            # Change default AI method globally:
            from SpheroidPy.utils.config import Config, AISegmentationType
            Config.ai_segmentation = AISegmentationType.HRNET
            
            # Try multiple methods in order
            img.segmentation([
                ('thresholding', 'fluorescence_green'),
                ('ai', 'brightfield')
            ])
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
                        thres=kwargs.get('threshold', 1.2),
                        thres_yen=kwargs.get('use_yen', True),
                        border_margin=border_margin
                    )
                elif seg_method == 'ai':
                    # Use centralized configuration to determine which AI method to use
                    if Config.ai_segmentation == AISegmentationType.HRNET:
                        logger.info(f"Using HRNet for AI segmentation (configured via Config.ai_segmentation)")
                        self.contour = self.segmentation_hrnet(
                            channel,
                            thres=kwargs.get('confidence', 0.65),
                            border_margin=border_margin
                        )
                    else:  # Default to Detectron2
                        logger.info(f"Using Detectron2 for AI segmentation (configured via Config.ai_segmentation)")
                    self.contour = self.segmentation_detectron(
                        channel,
                        thres=kwargs.get('confidence', 0.65),
                        border_margin=border_margin
                    )
                else:
                    logger.error(f"Unknown segmentation method '{seg_method}'. Supported methods: 'thresholding', 'ai'. Skipping.")
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
                    # Try brightfield first, then any available channel
                    sample_img = None
                    if 'brightfield' in self.image_path_dict:
                        sample_img = read_image(str(self.image_path_dict['brightfield']))
                    if sample_img is None:
                        # Try any available channel
                        for ch in ['fluorescence_green', 'fluorescence_red', 'fluorescence_blue']:
                            if ch in self.image_path_dict:
                                sample_img = read_image(str(self.image_path_dict[ch]))
                                if sample_img is not None:
                                    break
                    if sample_img is None:
                        logger.warning("Cannot load any image for border reconstruction.")
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

            # Automatically save contour to HDF5 if hdf5_path and hdf5_key are available
            if self.contour is not None and hasattr(self, 'hdf5_path'):
                self._save_contour_to_hdf5()

    def _save_contour_to_hdf5(self) -> None:
        """Private helper method to save contour and touches_border to HDF5.
        
        Saves the contour and touches_border attribute to the HDF5 file if both
        hdf5_path and hdf5_key are available. If contour is None, removes the contour
        from HDF5. This method is called automatically after segmentation, but can also
        be called manually if needed.
        
        Returns:
            None (silently fails if HDF5 is not available or key doesn't exist)
        """
        if not hasattr(self, 'hdf5_path') or not self.hdf5_path:
            return
        if not hasattr(self, 'hdf5_key') or not self.hdf5_key:
            return
        
        try:
            with h5py.File(self.hdf5_path, 'a') as hdf_file:
                # Get or create the group for this spheroid image
                if self.hdf5_key in hdf_file:
                    spheroid_image_group = hdf_file[self.hdf5_key]
                    if self.contour is None:
                        # Remove contour from HDF5 if it exists
                        if 'contour' in spheroid_image_group:
                            del spheroid_image_group['contour']
                        # Set touches_border to 0
                        spheroid_image_group.attrs['touches_border'] = 0
                    else:
                        # Save contour
                        if 'contour' in spheroid_image_group:
                            del spheroid_image_group['contour']
                        spheroid_image_group.create_dataset('contour', data=self.contour)
                        # Store as int for PyTables compatibility
                        touches_border_val = self.contour_touches_border if self.contour_touches_border is not None else False
                        spheroid_image_group.attrs['touches_border'] = 1 if touches_border_val else 0
                else:
                    logger.debug(
                        f"HDF5 key '{self.hdf5_key}' not found in file. Contour not saved to HDF5.")
        except Exception as e:
            logger.warning(f"Could not save contour to HDF5: {e}")

    def brightfield(self) -> np.ndarray | None:
        """Get brightfield image if available.
        
        Returns:
            Brightfield image as numpy array, or None if not available.
        """
        if 'brightfield' not in self.image_path_dict:
            return None
        return read_image(str(self.image_path_dict['brightfield']))

    def fluorescence(self, color: str) -> np.ndarray:
        color_dict = {'green': 'fluorescence_green', 'red': 'fluorescence_red', 'blue': 'fluorescence_blue'}
        if color not in color_dict or color_dict[color] not in self.image_path_dict:
            raise Exception(f'Color {color} not supported!')
        else:
            return read_image(str(self.image_path_dict[color_dict[color]]))

    def radial_profile(self, channels: list = ['green', 'red'], plot: bool = True, 
                        normalize: bool = True, 
                      return_absolute: bool = False, smoothing: float = 2,
                      savepath: str | None = None, nbins: int = 200) -> dict:
        """Calculate the radial intensity profile via 2D shell averaging with Distance-to-Boundary (DTB).
        
        Uses normalized radial bins (ρ = d_in / R_max, where ρ=0 at center, ρ=1 at boundary).
        Each pixel inside the spheroid mask is assigned to a radial bin based on its
        normalized distance to the boundary using Euclidean Distance Transform (EDT).
        This method is mathematically equivalent to Local Boundary Normalization (LBN) but
        much faster (O(N) vs O(N²)) and more robust for noisy masks.
        
        Key features:
        - Uses Distance-to-Boundary (DTB) normalization: ρ = dist_in / max(dist_in)
        - Normalized coordinate ρ = 0 at center, ρ = 1 at boundary (guaranteed for all shapes)
        - Form-invariant: works correctly for ellipses, irregular shapes, and noisy masks
        - Extremely fast: EDT is C-optimized, no raycasting needed
        - Bins are defined in normalized space [0,1] first, then converted to absolute if needed
        
        Args:
            channels: List of fluorescence channels to analyze. Options: 'green', 'red', 'blue'
            plot: Whether to display a plot of the profiles
            normalize: Whether to normalize intensity profiles to [0,1]. If False, absolute 
                      intensities are displayed. For multiple channels with normalize=False, 
                      separate y-axes are used (left and right).
            return_absolute: Whether to return absolute distances in addition to relative
            smoothing: Gaussian smoothing parameter (sigma). If 0, no smoothing is applied.
                      Applied before normalization.
            savepath: Optional path to save the plot. If None, plot is only displayed.
            nbins: Number of radial bins (default: 200)
            
        Returns:
            Dictionary containing:
                mean_radius: Outer radius estimate in μm (max mask radius)
                intensity_profiles: Dictionary of intensity profiles for each channel
                relative_distances: Bin centers normalized to [0,1] (ρ values)
                absolute_distances: Absolute distances in μm (if return_absolute=True)
        """
        # Require a valid contour/mask
        if self.contour is None or len(self.contour) < 3:
            raise ValueError("No valid contour available for radial profile computation.")

        # Import distance transform (used for both reference and per-channel calculations)
        from scipy.ndimage import distance_transform_edt

        # Validate channels
        valid_channels = ['green', 'red', 'blue']
        for channel in channels:
            if channel not in valid_channels:
                raise ValueError(f"Channel '{channel}' not supported. Options: {valid_channels}")
            if f'fluorescence_{channel}' not in self.image_path_dict:
                raise ValueError(f"Fluorescence channel '{channel}' not available in image data")

        # Load reference image to get dimensions for scaling
        # Use brightfield if available, otherwise use first available channel
        ref_channel = None
        if 'brightfield' in self.image_path_dict:
            ref_channel = 'brightfield'
        else:
            # Find first available channel
            for ch in ['fluorescence_green', 'fluorescence_red', 'fluorescence_blue']:
                if ch in self.image_path_dict:
                    ref_channel = ch
                    break
        
        if ref_channel is None:
            raise ValueError("No image channels available for scaling.")
        
        bf_img = read_image(str(self.image_path_dict[ref_channel]))
        if bf_img is None:
            raise ValueError(f"Cannot load {ref_channel} image for scaling.")
        bf_h_px, bf_w_px = bf_img.shape[:2]
        
        # Scaling px → μm (based on brightfield as reference)
        if hasattr(self, 'image_size') and self.image_size is not None:
            height_um = float(self.image_size[1])
            width_um = float(self.image_size[0])
            sx = width_um / max(1, bf_w_px)
            sy = height_um / max(1, bf_h_px)
        else:
            height_um = float(bf_h_px)
            width_um = float(bf_w_px)
            sx = sy = 1.0
        scale_avg = (sx + sy) / 2.0

        # Get contour in brightfield coordinates (reference frame)
        contour_bf = self.contour.copy()
        
        # Create reference mask from brightfield for computing mean_radius_um
        mask_bf = np.zeros((bf_h_px, bf_w_px), dtype=np.uint8)
        contour_int_bf = np.round(contour_bf).astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask_bf, [contour_int_bf], 1)
        mask_bf_bool = mask_bf.astype(bool)
        
        # Compute reference R_max from brightfield using DTB for mean_radius calculation
        if mask_bf_bool.sum() > 0:
            # Distance to boundary (inside the mask)
            dist_in_bf = distance_transform_edt(mask_bf_bool)
            # Maximum interior distance = morphological radius
            R_max_bf = dist_in_bf.max()
            mean_radius_um = R_max_bf * scale_avg if R_max_bf > 0 else 0.0
        else:
            mean_radius_um = 0.0
        
        # Bin centers are the same for all channels (normalized [0, 1])
        bins = np.linspace(0, 1, nbins + 1)
        rho_centers = 0.5 * (bins[:-1] + bins[1:])  # Bin centers in normalized space

        intensity_smooth = {}
        
        # Process each channel - create mask for each channel's image size
        for channel in channels:
            channel_image = self.fluorescence(channel)
            if channel_image is None:
                    continue

            # Convert to grayscale if needed
            if channel_image.ndim == 3:
                gray = cv2.cvtColor(channel_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = channel_image
            
            # Get channel image dimensions
            ch_h_px, ch_w_px = gray.shape[:2]
            
            # Scale contour to match channel image size if dimensions differ
            if ch_h_px != bf_h_px or ch_w_px != bf_w_px:
                # Scale contour from brightfield coordinates to channel coordinates
                scale_x = ch_w_px / max(1, bf_w_px)
                scale_y = ch_h_px / max(1, bf_h_px)
                contour_scaled = contour_bf.copy()
                contour_scaled[:, 0] = contour_scaled[:, 0] * scale_x
                contour_scaled[:, 1] = contour_scaled[:, 1] * scale_y
            else:
                contour_scaled = contour_bf.copy()
            
            # Build binary mask for this channel's image size
            mask = np.zeros((ch_h_px, ch_w_px), dtype=np.uint8)
            contour_int = np.round(contour_scaled).astype(np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [contour_int], 1)
            mask_bool = mask.astype(bool)

            if mask_bool.sum() == 0:
                logger.warning(f"Mask is empty for channel '{channel}'. Skipping.")
                continue

            # Distance-to-Boundary (DTB) normalization - fast and mathematically equivalent to LBN
            # 1. Distance to boundary (inside the mask)
            dist_in = distance_transform_edt(mask_bool)
            
            # 2. Maximum interior distance = morphological radius (distance from center to boundary)
            R_max = dist_in.max()
            if R_max <= 0:
                logger.warning(f"Maximum radius is zero for channel '{channel}'. Skipping.")
                continue
            
            # 3. Normalized radial depth: ρ = dist_in / R_max
            # ρ = 0 at center, ρ = 1 at boundary (guaranteed for all shapes)
            rho = 1 - dist_in / (R_max + 1e-12)
            rho = np.clip(rho, 0, 1)

            
            # Extract values for masked pixels only
            rho_vals = rho[mask_bool]
            
            # Ensure ρ ∈ [0,1]
            rho_vals = np.clip(rho_vals, 0, 1)

            # Bin the normalized radii [0, 1] (bins already defined outside loop)
            bin_idx = np.digitize(rho_vals, bins) - 1  # -1 to align with 0-indexed bins
            
            # Extract intensity values for masked pixels
            intensities = gray[mask_bool]

            # Compute mean and std per bin
            profile_mean = np.zeros(nbins, dtype=float)
            profile_std = np.zeros(nbins, dtype=float)
            
            for i in range(nbins):
                vals = intensities[bin_idx == i]
                if len(vals) > 0:
                    profile_mean[i] = vals.mean()
                    profile_std[i] = vals.std()
                else:
                    profile_mean[i] = 0.0
                    profile_std[i] = 0.0

            # Optional smoothing (before normalization)
            if smoothing > 0:
                profile_mean = gaussian_filter1d(profile_mean, sigma=smoothing)
                profile_std = gaussian_filter1d(profile_std, sigma=smoothing)

            # Store absolute values before normalization
            mean_abs = profile_mean.copy()
            std_abs = profile_std.copy()

            # Normalize if requested
            if normalize:
                max_val = profile_mean.max()
                if max_val > 0:
                    profile_mean = profile_mean / max_val
                    profile_std = profile_std / max_val

            intensity_smooth[channel] = {
                'mean': profile_mean,
                'std': profile_std,
                'mean_absolute': mean_abs,
                'std_absolute': std_abs,
            }
        
        # Relative distances are the normalized bin centers (ρ values)
        relative_distances = rho_centers.copy()
        
        # Absolute distances in μm (convert normalized ρ back to physical distance)
        # Use mean_radius_um (already in μm) for conversion: absolute_dist = ρ * mean_radius_um
        absolute_distances = None
        if return_absolute and mean_radius_um > 0:
            absolute_distances = rho_centers * mean_radius_um

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
                   background_subtraction: bool = True,
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
            sample_img = None
            # Try brightfield first, then any available channel
            if 'brightfield' in self.image_path_dict:
                sample_img = read_image(str(self.image_path_dict['brightfield']))
            if sample_img is None:
                for ch in ['fluorescence_green', 'fluorescence_red', 'fluorescence_blue']:
                    if ch in self.image_path_dict:
                        sample_img = read_image(str(self.image_path_dict[ch]))
                        if sample_img is not None:
                            break
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
        if self.contour is not None and len(self.contour) > 2:
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

    def segmentation_hrnet(self, channel: str, thres: float = 0.65, border_margin: int = 5) -> np.ndarray | None:
        """Segment spheroid using HRNet segmentation model.

        Args:
            channel: Image channel to use for segmentation.
            thres: Probability threshold for segmentation mask (default: 0.65).
            border_margin: Margin in pixels for border detection.

        Returns:
            np.ndarray of contour, or None if segmentation fails.
        """
        if not HRNET_AVAILABLE or torch is None:
            raise ImportError(
                "PyTorch is not installed. Please install PyTorch to use HRNet segmentation."
            )

        # 1. Input validation and image loading
        if channel not in self.image_path_dict:
            raise ValueError(f"Channel '{channel}' not found.")

        image = read_image(str(self.image_path_dict[channel]), cv2.IMREAD_ANYDEPTH)
        if image is None:
            raise IOError(f"Failed to load image from path: {self.image_path_dict[channel]}")

        # Log image statistics for debugging
        logger.debug(f"HRNet input image - shape: {image.shape}, dtype: {image.dtype}, "
                    f"min: {image.min()}, max: {image.max()}, mean: {image.mean():.2f}")

        # 2. Optimized torch-native preprocessing (all operations in torch, no numpy conversions)
        # Check if device is available (GPU support)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Convert to torch tensor immediately (single conversion)
        image_t = torch.from_numpy(image).to(device=device, dtype=torch.float32)
        
        # For fluorescence images, apply intensity rescaling using torch-native operations
        # This matches the original Deep-Tumour-Spheroid preprocessing:
        # 1. Intensity rescaling (exposure.rescale_intensity equivalent) → [0, 255]
        # 2. Grayscale → RGB (color.grey2rgb equivalent)
        # 3. Normalize to [0, 1] by dividing by 255 (ToTensor equivalent)
        # 4. ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        is_fluorescence = 'fluorescence' in channel.lower() or 'green' in channel.lower() or 'red' in channel.lower() or 'blue' in channel.lower()
        if is_fluorescence:
            # Apply intensity rescaling for fluorescence images using torch operations
            # This is equivalent to skimage.exposure.rescale_intensity
            from SpheroidPy.utils.hrnet_utils import rescale_intensity_torch
            image_t = rescale_intensity_torch(image_t, out_range=(0.0, 255.0))
            logger.debug(f"Applied torch-native intensity rescaling for fluorescence image")
        
        # Use optimized torch-native preprocessing (works directly with torch tensors)
        # For fluorescence: already_rescaled=True → divides by 255 (like ToTensor)
        # For brightfield: already_rescaled=False → uses min/max normalization
        # Note: ImageNet normalization might cause issues with fluorescence images
        # Try without ImageNet normalization if segmentation fails
        from SpheroidPy.utils.hrnet_utils import preprocess_image_torch
        x_tensor = preprocess_image_torch(image_t, device=device, already_rescaled=is_fluorescence, use_imagenet_norm=True)

        # 3. Get model and run inference
        model = self._get_hrnet_model()
        # Move model to device if it's not already there (for GPU support)
        if device != "cpu" and next(model.parameters(), None) is not None:
            model = model.to(device)
        
        # Suppress the align_corners warning from TorchScript model
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*align_corners.*")
            with torch.no_grad():
                # Model returns logits - handle both TorchScript and regular model formats
                output = model(x_tensor)
                
                # Handle different output formats
                if isinstance(output, tuple):
                    logits = output[0]  # If model returns tuple
                else:
                    logits = output  # Direct logits
                
                # Handle different logit shapes:
                # - [batch, classes, H, W] -> take first batch item
                # - [classes, H, W] -> use directly
                if logits.dim() == 4:
                    logits = logits[0]  # Remove batch dimension: [batch, classes, H, W] -> [classes, H, W]
                
                # Apply softmax to get probabilities for both classes
                probs = torch.softmax(logits, dim=0)  # [2, H, W] - [background, foreground]
                probs_foreground = probs[1]  # Foreground probability (class 1)
                probs_background = probs[0]  # Background probability (class 0)
                
                # For fluorescence images, check if we need to invert (bright spheroid vs dark background)
                # If background probability is higher in bright areas, we might need to use background class
                # But typically class 1 should be foreground (spheroid)
                probs_tensor = probs_foreground  # Keep on device for thresholding
                
                # Debug: Log probability statistics (compute on device, then convert)
                prob_min, prob_max, prob_mean = float(probs_tensor.min()), float(probs_tensor.max()), float(probs_tensor.mean())
                bg_prob_mean = float(probs_background.mean())
                fg_prob_mean = float(probs_foreground.mean())
                logger.debug(f"HRNet probability stats - FG min: {prob_min:.4f}, max: {prob_max:.4f}, "
                            f"mean: {fg_prob_mean:.4f}, BG mean: {bg_prob_mean:.4f}, threshold: {thres}")
                
                # For fluorescence: if foreground probability is very low overall, try inverting
                # (maybe the model was trained with inverted labels)
                if is_fluorescence and fg_prob_mean < 0.3 and bg_prob_mean > 0.7:
                    logger.warning(f"Fluorescence image: FG prob very low ({fg_prob_mean:.4f}), "
                                 f"BG prob very high ({bg_prob_mean:.4f}). "
                                 f"Trying inverted interpretation (using background class as foreground).")
                    probs_tensor = probs_background  # Use background class as foreground for fluorescence

        # 4. Create binary mask from probabilities (threshold on device, then convert to numpy)
        pred_mask_tensor = (probs_tensor > thres).to(torch.uint8)
        pred_mask = pred_mask_tensor.cpu().numpy()
        
        # Debug: Log mask statistics
        mask_pixels = pred_mask.sum()
        total_pixels = pred_mask.size
        mask_percentage = 100 * mask_pixels / total_pixels if total_pixels > 0 else 0
        logger.debug(f"HRNet mask - pixels above threshold: {mask_pixels}, "
                    f"total pixels: {total_pixels}, percentage: {mask_percentage:.2f}%")

        if pred_mask.sum() == 0:
            # Try with a lower threshold if no pixels detected (threshold on device)
            lower_thres = max(0.1, prob_mean)  # Use mean probability or 0.1, whichever is higher
            logger.warning(f"No spheroid detected with threshold {thres}. "
                          f"Probability range: [{prob_min:.4f}, {prob_max:.4f}], mean: {prob_mean:.4f}. "
                          f"Trying lower threshold: {lower_thres:.4f}")
            pred_mask_tensor = (probs_tensor > lower_thres).to(torch.uint8)
            pred_mask = pred_mask_tensor.cpu().numpy()
            
            if pred_mask.sum() == 0:
                logger.error(f"No spheroid detected even with lower threshold {lower_thres:.4f}. "
                            f"Maximum probability: {prob_max:.4f}. "
                            f"Consider checking image quality or using a different segmentation method.")
                self.contour = None
                self.contour_touches_border = None
                return None
            else:
                logger.info(f"Spheroid detected with adjusted threshold {lower_thres:.4f} "
                           f"(original threshold {thres} was too high)")

        # 5. Find contours from the mask
        contours_found, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours_found:
            logger.warning("No contour found for the detected spheroid mask.")
            self.contour = None
            self.contour_touches_border = None
            return None

        # Select the largest contour if multiple were found
        main_contour = max(contours_found, key=cv2.contourArea).squeeze()

        # 6. Check if contour touches border
        cleared_mask = clear_border(pred_mask, buffer_size=border_margin)
        touches_border = not np.array_equal(pred_mask, cleared_mask)
        self.contour_touches_border = touches_border
        self.contour = main_contour.astype(np.float32)

        # 7. Store and return results
        # HRNet has no instance score, so use average pixel confidence (compute on device)
        mask_bool_tensor = pred_mask_tensor.bool()
        if mask_bool_tensor.any():
            avg_score = float(probs_tensor[mask_bool_tensor].mean())
        else:
            avg_score = 0.0
        self.analysis_results['hrnet_threshold'] = thres
        self.analysis_results['hrnet_score'] = avg_score

        logger.info(f"Spheroid detected with HRNet (avg confidence: {avg_score:.2f}). Touches border: {touches_border}.")

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
            contour array, or None if segmentation fails
        """
        if channel not in self.image_path_dict:
            raise ValueError(f"Channel '{channel}' not found.")

        img_path = str(self.image_path_dict[channel])
        img = read_image(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise IOError(f"Failed to load image from path: {img_path}")

        height, width = img.shape[0], img.shape[1]
        self._height, self._width = height, width

        # Bildvorverarbeitung nur für Fluoreszenzkanäle (ähnlich der alten _image() Methode)
        # Aber ohne PIL - verwende nur numpy/cv2 Operationen
        if 'fluorescence' in channel:
            # Ähnlich der alten mono_incr Vorverarbeitung, aber ohne PIL
            # 1. Normalisiere zu [0, 255] falls nötig
            img_min, img_max = img.min(), img.max()
            if img_max - img_min >= 10:
                img = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
            
            # 2. Leichte Kontrastverstärkung ähnlich mono_incr=1.5
            # Inversion mit Kontrastverstärkung: 255 - clip(incr * (255 - img), 0, 255)
            mono_incr = 1
            #print("Using mono_incr = 	", mono_incr)
            img = 255 - np.clip(mono_incr * (255 - img.astype(float)), 0, 255).astype(np.uint8)

        # 1. Automatisierte Helligkeitsbestimmung
        if channel == 'brightfield':
            is_dark = True
        elif 'fluorescence' in channel:
            is_dark = False
        else:
            is_dark = np.mean(img) < 128
            logger.warning(f"Unknown channel type '{channel}'. Using heuristic to determine brightness.")

        # 2. Bilde den Schwellenwert
        if thres_yen:
            #print("Using Yen's method")
            thres_val = threshold_yen(img)
        else:
            #print("Using Otsu's method")
            thres_val = threshold_otsu(img)

        final_thres = thres_val * thres

        bool_mask = img < final_thres if is_dark else img > final_thres

        # 3. Bereinige die Maske mit morphologischen Operationen
        for _ in range(3):
            bool_mask = binary_dilation(bool_mask)
        bool_mask = binary_fill_holes(bool_mask)
        bool_mask = remove_small_objects(bool_mask, min_size=1000)
        for _ in range(3):
            bool_mask = binary_closing(bool_mask)
        bool_mask = binary_fill_holes(bool_mask)

        # 4. Identifiziere Spheroid als das zentrierteste Objekt (wie in alter Version)
        labeled_mask, num_labels = label(bool_mask)
        if num_labels == 0:
            logger.warning("No objects found after thresholding and cleaning. Try adjusting parameters.")
            self.contour = None
            self.contour_touches_border = None
            return None

        # Berechne Zentrum der Masse für jedes Objekt
        try:
            centers = []
            for label_id in range(1, num_labels + 1):
                component_mask = (labeled_mask == label_id)
                if component_mask.sum() > 0:
                    com = center_of_mass(component_mask)
                    if not (np.isnan(com[0]) or np.isnan(com[1])):
                        centers.append((label_id, com[0], com[1]))
            
            if not centers:
                # Fallback: verwende größtes Objekt
        sizes = np.bincount(labeled_mask.ravel())
        main_label = sizes[1:].argmax() + 1
        main_mask = labeled_mask == main_label
                logger.warning("Could not calculate centers. Using largest object instead.")
            else:
                # Berechne Distanz zum Bildzentrum für jedes Objekt
                image_center = np.array([height / 2, width / 2])
                distances = []
                for label_id, cy, cx in centers:
                    dist = np.sqrt((cy - image_center[0])**2 + (cx - image_center[1])**2)
                    distances.append((label_id, dist))
                
                # Wähle Objekt, das am nächsten zum Zentrum ist
                main_label = min(distances, key=lambda x: x[1])[0]
                main_mask = (labeled_mask == main_label)
                logger.info(f"Selected object {main_label} (closest to center) from {num_labels} objects.")
        except Exception as e:
            # Fallback: verwende größtes Objekt
            logger.warning(f"Center-based selection failed: {e}. Using largest object instead.")
            sizes = np.bincount(labeled_mask.ravel())
            main_label = sizes[1:].argmax() + 1
            main_mask = labeled_mask == main_label

        # 5. Finde die Kontur des Hauptobjekts (verwende cv2.findContours wie in alter Version)
        binary_mask = np.uint8(main_mask) * 255
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        
        if not contours or len(contours) == 0:
            logger.warning("No contours found for the selected object.")
            self.contour = None
            self.contour_touches_border = None
            return None

        # Verwende die größte Kontur (falls mehrere gefunden wurden)
        main_contour = max(contours, key=cv2.contourArea)
        contour = main_contour.reshape(-1, 2).astype(np.float32)

        # 6. Prüfe, ob die Kontur den Rand berührt
        border_bool = touches_border(contour, width - border_margin, height - border_margin, 
                                    border_margin, border_margin)
        
        # Entferne Randpunkte wenn Rand berührt wird
        if border_bool:
            contour = remove_border_points(contour, width - border_margin, height - border_margin,
                                          border_margin, border_margin)
        
        self.contour = contour
        self.contour_touches_border = border_bool

        logger.info(f"Spheroid segmentation complete for channel '{channel}'. "
                   f"Contour touches border: {border_bool}, points: {len(contour)}")

        return self.contour

    def segmentation_thresholding_multiplespheroids(self,channel: str,thres: float = 1.35,thres_yen: bool = True,border_margin: int = 5,
        min_size: int = 500,
        use_convex_hull: bool = True,
        gaussian_sigma: float = 1.2,
        local_block_size: int = 75,
        local_offset: float = 10,
        contour_smoothing_sigma: float = 1.0,
        ) -> list[np.ndarray]:
        """
        Improved threshold-based segmentation for multiple spheroids in 2D images.

        Combines global + local thresholding, improved morphological cleanup,
        and robust contour extraction with optional convex hull refinement.
        """

        if channel not in self.image_path_dict:
            raise ValueError(f"Channel '{channel}' not found.")

        img_path = str(self.image_path_dict[channel])
        img = read_image(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise IOError(f"Failed to load image: {img_path}")

        # Normalize fluorescence images
        if "fluorescence" in channel:
            img_min, img_max = img.min(), img.max()
            if img_max - img_min >= 10:
                img = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)

        # Determine dark/bright objects
        if channel == "brightfield":
            is_dark = True
        elif "fluorescence" in channel:
            is_dark = False
        else:
            is_dark = np.mean(img) < 128

        # --- PREPROCESSING ---
        img_smooth = cv2.GaussianBlur(img, (0, 0), gaussian_sigma)

        # --- GLOBAL THRESHOLD ---
        gth = threshold_yen(img_smooth) if thres_yen else threshold_otsu(img_smooth)
        gth = gth * thres
        global_mask = img_smooth < gth if is_dark else img_smooth > gth

        # --- LOCAL THRESHOLD Fallback ---
        local_th = threshold_local(img_smooth, block_size=local_block_size, offset=local_offset)
        local_mask = img_smooth < local_th if is_dark else img_smooth > local_th

        # Combine masks (robust in heterogeneous lighting)
        bool_mask = global_mask | local_mask

        # --- MORPHOLOGY ---
        # binary_opening and binary_closing from skimage don't support iterations parameter
        # So we call them multiple times
        bool_mask = binary_opening(bool_mask)  # iterations=1, so just one call
        for _ in range(2):
            bool_mask = binary_closing(bool_mask)
        bool_mask = binary_fill_holes(bool_mask)
        bool_mask = remove_small_objects(bool_mask, min_size=min_size)

        # --- LABELING ---
        labeled, num = label(bool_mask)
        if num == 0:
            self._contours = []
            self._contours_scaled = []
            return []

        height, width = img.shape
        contours_list: list[np.ndarray] = []

        for label_id in range(1, num + 1):
            component = labeled == label_id
            if component.sum() < min_size:
                continue

            comp_contours = find_contours(component.astype(float), 0.5)
            if not comp_contours:
                continue

            # Largest contour = outer boundary
            comp_contours.sort(key=lambda c: len(c), reverse=True)
            contour = np.array(comp_contours[0], dtype=np.float32)
            contour_xy = np.column_stack((contour[:, 1], contour[:, 0]))

            # --- OPTIONAL CONVEX HULL ---
            if use_convex_hull and len(contour_xy) >= 3:
                hull = cv2.convexHull(contour_xy.astype(np.float32))
                hull_area = cv2.contourArea(hull)
                raw_area = cv2.contourArea(contour_xy)
                # Apply convex hull only if it improves shape significantly
                if raw_area > 0 and hull_area / raw_area < 1.25:
                    contour_xy = hull.reshape(-1, 2)

            # --- OPTIONAL SMOOTHING ---
            if contour_smoothing_sigma > 0:
                from scipy.ndimage import gaussian_filter1d
                contour_xy[:, 0] = gaussian_filter1d(contour_xy[:, 0], contour_smoothing_sigma)
                contour_xy[:, 1] = gaussian_filter1d(contour_xy[:, 1], contour_smoothing_sigma)

            # --- BORDER HANDLING ---
            if touches_border(contour_xy, width - border_margin, height - border_margin):
                # Skip objects touching the border instead of deleting points
                continue

            if len(contour_xy) > 2:
                contours_list.append(contour_xy)

        self._contours = contours_list
        self._contours_scaled = self._compute_scaled_contours(contours_list)
        
        return contours_list

    def _compute_scaled_contours(self, contours: list[np.ndarray]) -> list[np.ndarray]:
        """Scale contours from pixel to physical coordinates (μm) if possible."""
        scaled = []
        if not contours:
            return scaled
        if not hasattr(self, "_pixel_dimensions") or self._pixel_dimensions is None:
            return scaled
        height_px, width_px = self._pixel_dimensions
        if not hasattr(self, "image_size") or self.image_size is None:
            return scaled
        width_um, height_um = self.image_size
        for contour in contours:
            c = contour.astype(float).copy()
            c[:, 0] = (c[:, 0] / max(1, width_px - 1)) * width_um
            c[:, 1] = height_um - (c[:, 1] / max(1, height_px - 1)) * height_um
            scaled.append(c)
        return scaled

    def _show_multiple(self, channel: str = "brightfield", savepath: str | None = None) -> None:
        """Plot multiple spheroid contours over the given channel."""
        if channel not in self.image_path_dict:
            raise ValueError(f"Channel '{channel}' not found.")

        img = read_image(str(self.image_path_dict[channel]), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise IOError(f"Failed to load image from path: {self.image_path_dict[channel]}")

        if img.ndim == 2:
            disp = img
            cmap = "gray"
        else:
            disp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cmap = None

        height_px, width_px = img.shape[:2]
        if hasattr(self, "image_size") and self.image_size is not None:
            height_um = float(self.image_size[1])
            width_um = float(self.image_size[0])
            sx = width_um / max(1, width_px)
            sy = height_um / max(1, height_px)
        else:
            height_um = float(height_px)
            width_um = float(width_px)
            sx = sy = 1.0

        plt.figure(figsize=(6, 6 * (height_um / width_um if width_um else 1)))
        # Show with correct orientation (origin at top-left like the image)
        plt.imshow(disp, extent=[0, width_um, 0, height_um], origin="upper", cmap=cmap)

        # Choose scaled contours if available
        contours_to_plot = self._contours_scaled if self._contours_scaled else self._contours
        for contour in contours_to_plot:
            if contour is None or len(contour) < 2:
                continue
            cx = contour[:, 0].astype(float)
            cy = contour[:, 1].astype(float)
            if contours_to_plot is self._contours:
                # convert px to μm if using pixel coords
                cx = cx * sx
                cy = cy * sy
            if cx[0] != cx[-1] or cy[0] != cy[-1]:
                cx = np.r_[cx, cx[:1]]
                cy = np.r_[cy, cy[:1]]
            plt.plot(cx, cy, '-', linewidth=1.5)

        plt.xlabel("x [μm]")
        plt.ylabel("y [μm]")
        plt.title(f"Multiple spheroid contours ({channel})")
        plt.tight_layout()
        if savepath is not None:
            plt.savefig(savepath, transparent=True, dpi=300, bbox_inches="tight")
        plt.show()


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
        # Default to first available channel if channel not found
        if channel in init_map:
            initial = init_map[channel]
        else:
            # Use first available channel as default
            if 'brightfield' in images:
                initial = 'brightfield'
            elif 'green' in images:
                initial = 'green'
            elif 'red' in images:
                initial = 'red'
            elif 'blue' in images:
                initial = 'blue'
            else:
                initial = 'brightfield'  # Fallback, but should not happen

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
        # Try brightfield first, then any available channel
        sample_img = None
        if 'brightfield' in self.image_path_dict:
            sample_img = read_image(str(self.image_path_dict['brightfield']))
        if sample_img is None:
            # Try any available channel
            for ch in ['fluorescence_green', 'fluorescence_red', 'fluorescence_blue']:
                if ch in self.image_path_dict:
                    sample_img = read_image(str(self.image_path_dict[ch]))
                    if sample_img is not None:
                        break
        if sample_img is None:
            raise ValueError("Cannot load any image for scaling")
        
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
            # Try brightfield first, then any available channel
            bg_img = None
            if 'brightfield' in self.image_path_dict:
                bg_img = read_image(str(self.image_path_dict['brightfield']))
            if bg_img is None:
                # Try any available channel
                for ch in ['fluorescence_green', 'fluorescence_red', 'fluorescence_blue']:
                    if ch in self.image_path_dict:
                        bg_img = read_image(str(self.image_path_dict[ch]))
                        if bg_img is not None:
                            break
            if bg_img is None:
                raise ValueError("Cannot load any image for plotting")
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
            # Try brightfield first, then any available channel
            bf = None
            if 'brightfield' in self.image_path_dict:
                bf = read_image(str(self.image_path_dict['brightfield']))
            if bf is None:
                # Try any available channel
                for ch in ['fluorescence_green', 'fluorescence_red', 'fluorescence_blue']:
                    if ch in self.image_path_dict:
                        bf = read_image(str(self.image_path_dict[ch]))
                        if bf is not None:
                            break
            if bf is None:
                raise ValueError("Cannot load any image for plotting")
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
    
    def _get_hrnet_model(self):
        """Lazy-load the HRNet model."""
        if self.hrnet_model is None:
            if not HRNET_AVAILABLE:
                raise ImportError(
                    "PyTorch is not installed. Please install PyTorch to use HRNet segmentation."
                )
            logger.info("Initializing HRNet model. This may take a moment.")
            self.hrnet_model = load_hrnet_model()
        return self.hrnet_model


