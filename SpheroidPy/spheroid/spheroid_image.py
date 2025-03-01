# Detectron2 Imports
try:
    import detectron2
    import torch
    
    from detectron2.utils.logger import setup_logger
    setup_logger()
    
    # import some common detectron2 utilities
    from detectron2 import model_zoo
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    
    # supress unnecessary output from Detectron in JupyterNotebook
    import logging
    logging.getLogger("detectron2").setLevel(logging.ERROR)
    
    DETECTRON2_AVAILABLE = True
except ImportError:
    DETECTRON2_AVAILABLE = False

# Other imports that don't depend on detectron2
import os
import json
import cv2
import random
import sys
import math
import zipfile
import shutil
from zipfile import ZipFile
import numpy as np
import tifffile as tiff
from matplotlib import pyplot as plt
from pathlib import Path

import pkg_resources

import h5py
from scipy.ndimage import binary_erosion, binary_dilation, binary_fill_holes, binary_opening, binary_closing
from scipy.ndimage import label, center_of_mass
from skimage.filters import gaussian, threshold_otsu, threshold_yen
from skimage.morphology import remove_small_objects
from PIL import Image
from scipy.sparse.linalg import spsolve
from matplotlib.colors import Normalize

from SpheroidPy.utils.visualisation import plot_mesh
from SpheroidPy.utils.pde_utils import compute_laplace_matrix, apply_dirichlet_boundary_conditions, get_boundary_nodes
from SpheroidPy.utils.geometry import touches_border, fit_ellipse, remove_border_points, add_fitted_contour, find_inflection_point

import matplotlib


class SpheroidImage:
    """A class for analyzing and processing images of spheroids."""

    def __init__(self, brightfield: Path | str, **kwargs):
        self.image_path_dict = {}
        self._initialize_channels(brightfield, **kwargs)
        self._initialize_attributes(**kwargs)

    def _initialize_channels(self, brightfield: Path | str, **kwargs):
        """Initialize imaging channels."""
        channels = {
            'brightfield': brightfield,
            'fluorescence_green': kwargs.get('fluorescence_green'),
            'fluorescence_red': kwargs.get('fluorescence_red'),
            'fluorescence_blue': kwargs.get('fluorescence_blue')
        }
        
        for channel, path in channels.items():
            if path is not None:
                try:
                    img = cv2.imread(str(path))
                    self.shape = img.shape
                    self.image_path_dict[channel] = path
                except Exception as e:
                    print(f'Error loading {channel} channel: {e}')

    def _initialize_attributes(self, **kwargs):
        """Initialize other attributes."""
        if 'image_size' in kwargs:
            self.image_size = kwargs['image_size']
        else:
            img = cv2.imread(str(self.image_path_dict['brightfield']))
            self.image_size = img.shape[:2]
            
        self.hdf5_path = kwargs.get('hdf5_path')
        self.hdf5_key = kwargs.get('hdf5_key')
        self.analysis_results = {}
        self.contour = None
        self.contour_touches_border = None

    def segmentation(self, 
                    methods: str | tuple | list = 'fluorescence_green',
                    reconstruct_border: bool = True, 
                    border_margin: int = 5,
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
        
        Examples:
            # Simple thresholding on green channel
            spheroid.segmentation('fluorescence_green')
            
            # AI segmentation with custom confidence
            spheroid.segmentation(('ai', 'brightfield'), confidence=0.8)
            
            # Try thresholding first, fall back to AI
            spheroid.segmentation([
                ('thresholding', 'fluorescence_green'),
                ('ai', 'brightfield')
            ], threshold=1.35, confidence=0.7)
        """
        # Convert string input to default thresholding config
        if isinstance(methods, str):
            methods = [('thresholding', methods)]
        # Convert single tuple to list
        elif isinstance(methods, tuple):
            methods = [methods]
        
        # Ensure methods is now a list of tuples
        if not isinstance(methods, list):
            raise ValueError("Methods must be string, tuple, or list of tuples")
        
        success = False
        for seg_method, channel in methods:
            try:
                if seg_method == 'thresholding':
                    self.contour, self.contour_touches_border = self.segmentation_thresholding(
                        channel, 
                        thres=kwargs.get('threshold', 1.35),
                        thres_yen=kwargs.get('use_yen', True),
                        border_margin=border_margin
                    )
                    
                elif seg_method == 'ai':
                    self.contour, self.contour_touches_border = self.segmentation_detectron(
                        channel,
                        thres=kwargs.get('confidence', 0.65)
                    )
                    
                else:
                    raise ValueError(f"Unknown segmentation method: {seg_method}")
                    
                # Check if segmentation was successful
                if self.contour is not None and len(self.contour) > 5:
                    success = True
                    
                    # Handle border reconstruction
                    if self.contour_touches_border and reconstruct_border:
                        try:
                            self.contour = add_fitted_contour(
                                self.contour,
                                fit_ellipse(self.contour)[-1],
                                self._width - border_margin - 1,
                                self._height - border_margin - 1,
                                border_margin + 1,
                                border_margin + 1
                            )
                        except:
                            print('Ellipse could not be reconstructed! Using cut contour instead.')
                            
                    break  # Exit loop on success
                    
            except Exception as e:
                #print(f"Segmentation failed with {seg_method} on {channel}: {str(e)}")
                continue
            
        if not success:
            #print("All segmentation attempts failed")
            self.contour = None
            self.contour_touches_border = False

    def brightfield(self) -> np.ndarray:
        return cv2.imread(str(self.image_path_dict['brightfield']))

    def fluorescence(self, color: str) -> np.ndarray:
        color_dict = {'green': 'fluorescence_green', 'red': 'fluorescence_red', 'blue': 'fluorescence_blue'}
        if color not in color_dict or color_dict[color] not in self.image_path_dict:
            raise Exception(f'Color {color} not supported!')
        else:
            return cv2.imread(str(self.image_path_dict[color_dict[color]]))

    def radial_profile(self, plot: bool = True, angle_step: int = 2) -> dict:
        """Calculate the radial intensity profile of the spheroid.
        
        Measures fluorescence intensity along radial lines from the center 
        to the boundary at regular angular intervals.
        
        Args:
            plot: Whether to display a plot of the profiles
            angle_step: Angular step size in degrees between measurements
            
        Returns:
            Dictionary containing:
                mean_radius: Average radius of the spheroid
                intensity_profiles: Dictionary of normalized intensity profiles
                    for each fluorescence channel
        """
        from shapely.geometry import Polygon, LineString
        import skimage.draw as draw
        from scipy.interpolate import interp1d

        # Calculate the center of the tumoroid
        M = cv2.moments(self.contour)
        x_center = int(M["m10"] / M["m00"])
        y_center = int(M["m01"] / M["m00"])

        # Approximate contour and convert to Shapely Polygon
        epsilon = 0.1
        approx = cv2.approxPolyDP(self.contour, epsilon, True)
        polygon = Polygon(approx.reshape(-1, 2))

        # Initialize storage for radii and intensities
        radius_array = []
        intensity_dict = {'green': [], 'red': []}
        intensity_interpolation = {'green': [], 'red': []}

        # Iterate over angles
        for angle in range(0, 360, angle_step):
            try:
                # Calculate the end point of the line
                x_line = x_center + 1000 * np.cos(np.radians(angle))
                y_line = y_center + 1000 * np.sin(np.radians(angle))

                # Create a line and find intersections
                line = LineString([(x_center, y_center), (x_line, y_line)])
                intersections = line.intersection(polygon)

                # Skip if no intersection
                if intersections.is_empty:
                    continue

                # Get the closest intersection point
                intersection_coords = list(intersections.coords)
                intersection = tuple(map(int, intersection_coords[1]))


                for channel_image, color in zip([self.fluorescence('green'), self.fluorescence('red')], ['green', 'red']):
                    # Get pixel values along the line
                    line_coords = np.array(draw.line(x_center, y_center, intersection[0], intersection[1])).T
                    intensity_values = cv2.cvtColor(channel_image, cv2.COLOR_BGR2GRAY)[line_coords[:, 1], line_coords[:, 0]]

                    #intensity_values = intensity_values[:,1] #todo: change for channel

                    # Store radius and intensity
                    if color == 'green':
                        radius_array.append(len(intensity_values))
                    intensity_dict[color].append(intensity_values)

                    # Interpolate intensity
                    interp_func = interp1d( np.linspace(0, 1, len(intensity_values)), intensity_values, kind='cubic', bounds_error=False, fill_value=0)
                    intensity_interpolation[color].append(interp_func)
            except:
                pass

        # Compute mean radius
        mean_radius = np.mean(radius_array)


        # Compute mean intensity profiles
        x_all = np.linspace(0, 1, num=100)
        intensity_smooth = {}
        for color in intensity_interpolation:

            interpolations = np.array([f(x_all) for f in intensity_interpolation[color]])
            mean_intensity = np.mean(interpolations, axis=0)
            std_intensity = np.std(interpolations, axis=0)

            # Normalize and store
            intensity_smooth[color] = {
                'mean': mean_intensity / mean_intensity.max(),
                'std': std_intensity / mean_intensity.max(),
            }

        # Optional plotting
        if plot:
            plt.figure()
            for color, label in zip(['green', 'red'], ['eGFP', 'PI']):
                mean = intensity_smooth[color]['mean']
                std = intensity_smooth[color]['std']
                plt.plot(x_all, mean, label=label, color=color)
                plt.fill_between(x_all, mean - std, mean + std, alpha=0.2, color=color)
            plt.legend()
            plt.xlabel('Normalized Distance')
            plt.ylabel('Normalized Intensity')
            plt.show()

        return {'mean_radius': mean_radius, 'intensity_profiles': intensity_smooth}

    def functional_radius(self, live_color: str | None = 'green', 
                         death_color: str | None = 'red') -> dict:
        """Calculate functional radii based on fluorescence profiles.
        
        Uses fluorescence intensity profiles to determine:
        - Outer radius (total spheroid size)
        - Inhibited radius (where cell growth is impaired)
        - Necrotic radius (where cells are dead/dying)
        
        Args:
            live_color: Color channel for live cell marker
            death_color: Color channel for dead cell marker
            
        Returns:
            Dictionary containing:
                outer: Total spheroid radius in μm
                inhibited: Radius where growth inhibition begins
                necrotic: Radius where necrosis begins
        """
        radial_profile_dict = self.radial_profile(plot=False, angle_step=5)

        radius = radial_profile_dict['mean_radius']
        intensities_dict = radial_profile_dict['intensity_profiles']

        self.analysis_results['structure'] = {'radial_profiles': intensities_dict}

        x_necrotic, x_inhibited = None, None

        if death_color is not None:
            #death_fluorescence = f'fluorescence_{death_color}'
            min_max_inflection_dict = find_inflection_point( intensities_dict[death_color]['mean'], np.linspace(0, radius, len(intensities_dict[death_color]['mean']) ), sigma=3)
            _, y_min, _ = min_max_inflection_dict['minimum']
            if y_min < 0.7: # threshold for when to count it as necrotic and not noise!
                x_necrotic, _, necrotic_index = min_max_inflection_dict['inflection']

        if live_color is not None:
            #live_fluorescence = f'fluorescence_{live_color}'
            y, x = intensities_dict[live_color]['mean'], np.linspace(0, radius, len(intensities_dict[live_color]['mean']))
            if x_necrotic is not None:
                x, y = x[necrotic_index:], y[necrotic_index:]

            min_max_inflection_dict = find_inflection_point( y, x, sigma=3 )
            x_inhibited,_,_ = min_max_inflection_dict['inflection_min']
        self.analysis_results['functional_radii'] = {'outer': radius*(self.radius/radius), 'inhibited': x_inhibited*(self.radius/radius) if x_inhibited is not None else None, 'necrotic': x_necrotic*(self.radius/radius) if x_necrotic is not None else None}
        return self.analysis_results['functional_radii']

    @property
    def scaled_contour(self) -> np.ndarray:
        """Get the contour scaled to physical coordinates.
        
        Converts contour from pixel coordinates to physical coordinates (μm)
        based on the image size.
        
        Returns:
            Array of shape (N,2) containing scaled (x,y) contour coordinates in μm
        """
        # Umrechnung der Konturen von Pixelkoordinaten in den Extent-Bereich
        # Angenommen, die Konturen sind im Bereich von [0, ncols] und [0, nrows]
        ncols, nrows = 1390, 1040 #self.shape
        x_min, y_min, x_max, y_max = 0, 0, self.image_size[0], self.image_size[1]

        # Skalierung der Konturkoordinaten
        scaled_contour = self.contour.copy()
        scaled_contour[:, 0] = (self.contour[:, 0] / ncols) * (x_max - x_min) + x_min  # Umrechnung der x-Koordinaten
        scaled_contour[:, 1] = y_max-(self.contour[:, 1] / nrows) * (y_max - y_min) + y_min  # Umrechnung der y-Koordinaten

        return scaled_contour

    @property
    def area(self) -> float | None:
        """Calculate the area enclosed by the spheroid contour.
        
        Uses the shoelace formula to compute the area enclosed by the contour,
        scaled to physical units (μm²).
        
        Returns:
            Area in μm² if contour exists, None otherwise
        """
        if self.contour is not None:
            return 0.5 * np.abs(np.dot(self.contour[:, 0], np.roll(self.contour[:, 1], 1)) - np.dot(self.contour[:, 1], np.roll(self.contour[:, 0], 1))) * (max(self.image_size)/max(self.shape))**2
        else:
            return None

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

        # Get and process fluorescence image
        fluo_img = self._image(channel_key, contour=False, mono=True, mono_incr=1)
        fluo_img = np.amax(fluo_img) - fluo_img / 255

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
    
    def segmentation_unet(self, channel:str, thres: float = .8): #todo: to be implemented
        a=0

    def segmentation_detectron(self, channel: str, thres: float = .65):
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

        image = cv2.imread(str(self.image_path_dict[channel]))
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

    def segmentation_thresholding(self, channel: str, thres: float = 1.35, thres_yen: bool = True, border_margin: int = 10):
        """
        Image segmentation function to create spheroid mask, radius, and position of a spheroid in a grayscale image.

        Args:
            img(array): Grayscale image as a Numpy array
            thres(float): To adjust the segmentation

        Returns:
            dict: Dictionary with keys: mask, radius, centroid (x/y)
        """

        # load Channel-Image as Mono-Image
        img = self._image(channel=channel, contour=False, mono=True)
        height, width = img.shape[0], img.shape[1]
        self._height, self._width = height, width

        if thres_yen:
            bool_mask = img < threshold_otsu(img) * thres
        else:
            bool_mask = img < threshold_yen(img) * thres


        # remove other objects
        bool_mask = binary_dilation(bool_mask, iterations=3)
        bool_mask = binary_fill_holes(bool_mask)
        bool_mask = remove_small_objects(bool_mask, min_size=1000)
        bool_mask = binary_closing(bool_mask, iterations=3)
        bool_mask = binary_fill_holes(bool_mask)

        # identify spheroid as the most centered object
        labeled_mask, max_lbl = label(bool_mask)
        center = np.array(center_of_mass(bool_mask, labeled_mask, range(1, max_lbl + 1)))

        try:
            distance_to_center = np.sqrt(np.sum((center - np.array([height / 2, width / 2])) ** 2, axis=1))

            bool_mask = (labeled_mask == distance_to_center.argmin() + 1)

            # determine radius of spheroid
            radius = np.sqrt(np.sum(bool_mask) / np.pi)

            # determine center position of spheroid
            cy, cx = center[distance_to_center.argmin()]

            #############################
        except:
            raise Exception("Error: Could not detect spheroid. You might adjust the segmentation.")

        color = (0, 0, 255)
        binary_mask = np.uint8(bool_mask) * 255
        contour, _ = cv2.findContours(binary_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        contour = contour[0].reshape(-1, 2).astype(np.float32)
        border_bool = touches_border(contour, width-border_margin, height-border_margin, border_margin, border_margin)
        #print(border_bool, width, height)
        if border_bool:
            contour = remove_border_points(contour, width-border_margin, height-border_margin, border_margin, border_margin)
        return contour,border_bool

    def segmentation_manual(self, channel: str):
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

        img = cv2.imread(str(self.image_path_dict[channel]))
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

    def diffusion_stationary(self, boundary_value: float = 10, 
                            diffusion_rate: float = 1000, 
                            reaction_rate: float = .1,
                            plot: bool = True,
                            plot_full: bool = True, 
                            accuracy: int = 10):
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
            plot_full: Whether to show full solution field
            accuracy: Mesh resolution parameter
            
        Returns:
            tuple: (points, triangles, solution) where:
                points: Mesh vertex coordinates
                triangles: Mesh connectivity
                solution: Concentration values at mesh points
        """
        diffusion_rate = self._width / self.image_size[0] * diffusion_rate
        reaction_rate = self._width / self.image_size[0] * reaction_rate

        points, triangles = self.mesh(300, False, accuracy)
        A = compute_laplace_matrix(points, triangles, reaction_rate, diffusion_rate)
        b = np.zeros(len(points))  # Kein Quellterm im Inneren (steady-state)
        apply_dirichlet_boundary_conditions(A, b, get_boundary_nodes(self.contour[::accuracy], points), boundary_value)  # Dirichlet-Randbedingungen
        solution = spsolve(A.tocsr(), b)
        if plot:
            plt.imshow(cv2.imread(str(self.image_path_dict['brightfield'])))
            fill = plt.tricontourf(points[:, 0], points[:, 1], triangles, solution, levels=100, cmap='hot', alpha=.5, norm=Normalize(vmin=0, vmax=boundary_value, clip=False))
            cbar = plt.colorbar(fill, label='Nutrient Concentration (a.u.)', norm=Normalize(vmin=0, vmax=boundary_value, clip=False))
            cbar.ax.collections[0].set_edgecolor("face")
            #plt.tricontour(points[:, 0], points[:, 1], triangles, solution, levels=100, colors='black', alpha=.05,norm=Normalize(vmin=0, vmax=boundary_value, clip=False))
            # Add a specific dotted contour line for a certain value
            specific_value = 2
            dotted_contour = plt.tricontour(points[:, 0], points[:, 1], triangles, solution, levels=[specific_value], colors='blue', linestyle='dotted', linewidths=1)
            plt.plot([], [], color='blue', linestyle=':', linewidth=1, label='critical $c_{~Nutrient}$')
            plt.legend()

            plt.xlim(0, self._width)
            plt.ylim(self._height,0)

            plt.xlabel('x-position [µm]')
            plt.ylabel('y-position [µm]')
            plt.title('Steady-State Diffusion with Reaction (FEM)')
            plt.show()
        return points, triangles, solution
    def mesh(self, max_area: float = 500, plot: bool = True, accuracy: int = 1):
        """Generate a triangular mesh of the spheroid contour.
        
        Uses MeshPy to create a triangulation of the region enclosed by the contour.
        
        Args:
            max_area: Maximum allowed area for triangles in the mesh
            plot: Whether to display a plot of the generated mesh
            accuracy: Sample every nth point from contour for mesh generation
            
        Returns:
            tuple: (points, triangles) where points is an array of vertex coordinates
            and triangles is an array of triangle vertex indices
        """
        from meshpy.triangle import MeshInfo, build
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
        # Plot Mesh
        plot_mesh(points, triangles) if plot else None
        return points, triangles

    def _image(self, 
               channel: str | list[str], 
               contour: bool = False, 
               mono: bool = True, 
               mono_incr: float = 1.5) -> np.ndarray | list[np.ndarray]:
        """
        Get image(s) for specified channel(s) with optional processing.
        
        Args:
            channel: Single channel name or list of channel names
            contour: Whether to draw contour on image
            mono: Whether to convert to monochrome
            mono_incr: Intensity multiplier for monochrome conversion
            
        Returns:
            Single processed image or list of processed images
        """
        def process_single_image(channel: str) -> np.ndarray:
            if channel not in self.image_path_dict:
                raise ValueError(f"Channel '{channel}' not found")
            
            img = cv2.imread(str(self.image_path_dict[channel]))
            
            if contour and self.contour is not None:
                cv2.polylines(img, 
                             [self.contour.astype(np.int32).reshape((-1, 1, 2))], 
                             True, (0, 0, 255), 2)
            
            if mono:
                # Convert to PIL for better color processing
                image_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                image_8bit = image_pil.convert('P', palette=Image.Palette.ADAPTIVE, colors=256)
                img = np.array(image_8bit)
                img = 255 - np.clip(mono_incr * (255 - img), 0, 255)
            
            return img

        if isinstance(channel, str):
            return process_single_image(channel)
        elif isinstance(channel, list):
            return [process_single_image(c) for c in channel]
        else:
            raise TypeError("channel must be string or list of strings")


