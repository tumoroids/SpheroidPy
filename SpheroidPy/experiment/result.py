from __future__ import annotations
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from openpyxl import Workbook, load_workbook
from openpyxl.utils import get_column_letter
from openpyxl.styles import PatternFill, Alignment, Font, Protection, Border, Side

from pathlib import Path
import numpy as np
from tqdm.notebook import tqdm
import random, re, os, h5py
import pandas as pd
from datetime import datetime
import cv2
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from SpheroidPy.experiment.base import Base

if TYPE_CHECKING:
    from SpheroidPy.experiment.experiment import Experiment
    from SpheroidPy import Platemap
from SpheroidPy.spheroid import spheroid_image

from SpheroidPy.spheroid.spheroid_image import SpheroidImage
from SpheroidPy.spheroid.spheroid_series import SpheroidSeries
from SpheroidPy.spheroid.spheroid_collection import SpheroidCollection

from SpheroidPy.utils.file_management import collect_data, print_directory_tree
from SpheroidPy.utils.geometry import touches_border, fit_ellipse, remove_border_points, add_fitted_contour, \
    find_inflection_point

import multiprocessing as mp
from functools import partial
from tqdm import tqdm


def _process_well_from_file_standalone(well_data):
    """Standalone helper function to process a well when loading from file.

    Args:
        well_data: Tuple of (well, timepoint_data) where timepoint_data contains
                  all necessary information extracted from HDF5

    Returns:
        Tuple of (well, spheroid_series, well_results)
    """
    from SpheroidPy.spheroid.spheroid_series import SpheroidSeries
    from SpheroidPy.spheroid.spheroid_image import SpheroidImage

    well, timepoint_data = well_data
    spheroid_series = SpheroidSeries(f'Spheroid-{well}')
    well_results = {}

    for timepoint, data in timepoint_data.items():
        image_paths = {}
        for channel, path in data['image_paths'].items():
            image_paths[channel] = path

        spheroid = SpheroidImage(
            image_paths['brightfield'],
            fluorescence_green=image_paths.get('fluorescence_green'),
            fluorescence_red=image_paths.get('fluorescence_red'),
            image_size=tuple(data['image_size']),
            hdf5_key=data['hdf5_key']
        )

        if data.get('contour') is not None:
            spheroid.contour = data['contour']
            spheroid.contour_touches_border = data.get('touches_border', False)

        well_results[timepoint] = {
            'spheroid': spheroid,
            'image_paths': image_paths
        }

    return well, spheroid_series, well_results


class Result(Base):
    """A class for managing and analyzing experimental results.

    Handles loading, processing and analysis of image data for multiple wells
    and timepoints. Manages associated platemap data and analysis metrics.

    Attributes:
        platemap: Associated Platemap object defining well contents
        spheroid_dict: Dictionary mapping well IDs to SpheroidSeries objects
        files_dict: Nested dict mapping channels -> wells -> timepoints to image paths
        time_points_array: List of absolute timepoints that have been loaded
        relative_time_array: List of relative times in hours from experiment start
        metric_dfs: Dictionary storing computed metrics for the experiment
    """

    platemap: Platemap  # Platemap that has been associated with this result

    # SpheroidResults
    spheroid_dict: dict  # stores all SpheroidSeries

    # File Management
    files_dict: dict  # dictionary of dictionaries with all channels -> wells -> timepoints that have been loaded
    time_points_array: list  # list with all absolute timepoints that have been loaded
    relative_time_array: list  # list with relative times of the above (in [h])

    #
    metric_dfs: dict  # stores the metrics of an experiment (once they have been calculated)

    # Visualisation
    plot_name_dict: dict = {'radius': r'effective Radius ($r_{eff}=\frac{Area}{\pi}$) [µm]'}
    data_has_been_loaded: bool

    def __init__(self, name: str, experiment: Experiment):

        super().__init__(name, experiment, 'Result')

        self.data_has_been_loaded = False
        self.spheroid_dict = {}

        with h5py.File(self.hdf5_path, 'a') as hdf_file:
            instance_group = hdf_file[self.hdf5_key]
            instance_group.attrs['name'] = self.name  # todo: wieder einblenden
            instance_group.attrs['index'] = self.index

            platemap_group = instance_group.create_group('Platemap')

        from SpheroidPy.experiment.platemap import Platemap
        self.platemap = Platemap(self)

        # storage for AnalysesMetrics
        self.metric_dfs = {}

        self.num_cores = mp.cpu_count()

    @classmethod
    def from_file(cls, name: str, index: int, experiment: Experiment, instace_group: h5py.Group, hdf5_path: str):
        result = object.__new__(cls)

        # General Properties
        result.type_name = 'Result'
        result.experiment = experiment

        result.name = name
        result.index = index

        result.hdf5_key = f'Result/{index}-{name}'

        # Platemap
        platemap_group = instace_group['Platemap']
        from SpheroidPy import Platemap
        result.platemap = Platemap.from_file(result, platemap_group)

        # Initialize storage
        result.data_has_been_loaded = True
        result.spheroid_dict = {}
        result.metric_dfs = {}
        result.files_dict = {}
        result.time_points_array = []

        # Setup multiprocessing
        result.num_cores = mp.cpu_count()

        # Extract data from HDF5 before parallel processing
        if 'ImageSeries' in instace_group:
            image_series_group = instace_group['ImageSeries']
            well_data = []

            for well in image_series_group:
                timepoint_data = {}
                well_group = image_series_group[well]

                for timepoint in well_group:
                    timepoint_group = well_group[timepoint]

                    # Extract image paths
                    image_paths = {}
                    for channel, path in timepoint_group['image_paths'].items():
                        if isinstance(path[()], bytes):
                            path_string = path[()].decode("utf-8")
                        else:
                            path_string = str(path[()])
                        image_paths[channel] = path_string

                    # Extract contour data if it exists
                    contour = None
                    touches_border = False
                    if "contour" in timepoint_group:
                        contour_dataset = timepoint_group["contour"]
                        if contour_dataset.shape != ():
                            contour = contour_dataset[:]
                            touches_border = timepoint_group.attrs.get("touches_border", False)

                    timepoint_data[timepoint] = {
                        'image_paths': image_paths,
                        'image_size': tuple(timepoint_group.attrs['image_size_x_y']),
                        'hdf5_key': str(timepoint_group.name),
                        'contour': contour,
                        'touches_border': touches_border
                    }

                well_data.append((well, timepoint_data))

            # Process wells in parallel with extracted data
            with mp.Pool(processes=result.num_cores) as pool:
                for well, series, well_results in tqdm(
                        pool.imap(_process_well_from_file_standalone, well_data),  # Use standalone function
                        total=len(well_data),
                        desc=f'Loading Result {result.index} - {result.name}'
                ):
                    result.spheroid_dict[well] = series
                    for timepoint, data in well_results.items():
                        series.add_spheroid_image(data['spheroid'], timepoint)

                        # Update files_dict
                        for channel, path in data['image_paths'].items():
                            if channel not in result.files_dict:
                                result.files_dict[channel] = {}
                            if well not in result.files_dict[channel]:
                                result.files_dict[channel][well] = {}
                            result.files_dict[channel][well][timepoint] = path
                            result.time_points_array.append(timepoint)

            for spheroid_series in result.spheroid_dict.values():
                spheroid_series._result = result

            # Process timepoints
            result.time_points_array = sorted(set(result.time_points_array))
            time_points_dt = [datetime.strptime(tp, '%Y-%m-%d %H:%M:%S') for tp in result.time_points_array]
            result.relative_time_array = [(tp - time_points_dt[0]).total_seconds() / 3600 for tp in time_points_dt]

        return result

    @staticmethod
    def _process_channel_data(args):
        """Static helper function to process image channels in parallel.

        Args:
            args: Tuple of (img_folder_path, filters, channel_name)

        Returns:
            Tuple of (channel_name, files_dict, time_points)
        """
        from SpheroidPy.utils.file_management import collect_data
        img_folder_path, filters, channel = args
        if filters is None:
            return channel, None, None
        return channel, *collect_data(img_folder_path, filters)

    @staticmethod
    def _process_well_data(args):
        """Static helper function to process a single well.

        Args:
            args: Tuple containing (well, timepoints, files_dict, relative_time_array, image_size, hdf5_key)

        Returns:
            Tuple of (well, results)
        """
        well, timepoints, files_dict, relative_time_array, image_size, hdf5_key = args
        results = []

        for timepoint, rel_timepoint in zip(timepoints, relative_time_array):
            if timepoint not in files_dict['brightfield'][well]:
                continue

            # Collect image paths for this timepoint
            image_paths = {}
            for channel in files_dict:
                if well in files_dict[channel] and timepoint in files_dict[channel][well]:
                    image_paths[channel] = str(Path(files_dict[channel][well][timepoint]).absolute())

            # Create spheroid image
            spheroid = SpheroidImage(
                image_paths['brightfield'],
                fluorescence_green=image_paths.get('fluorescence_green'),
                fluorescence_red=image_paths.get('fluorescence_red'),
                image_size=image_size,
                hdf5_key=f'{hdf5_key}/ImageSeries/{well}/{timepoint}'
            )

            results.append({
                'well': well,
                'timepoint': timepoint,
                'rel_timepoint': rel_timepoint,
                'spheroid': spheroid,
                'image_paths': image_paths
            })

        return well, results

    def load_images(self, img_folder_path: Path,
                    brightfield_filters: str | list = None,
                    fluorescence_green_filters: str | list | None = None,
                    fluorescence_red_filters: str | list | None = None,
                    fluorescence_blue_filters: str | list | None = None,
                    image_size: tuple | None = None) -> pd.DataFrame:
        """Load and process image data from specified folder using multiprocessing.

        Args:
            img_folder_path: Path to folder containing image files
            brightfield_filters: Filter pattern(s) for brightfield images
            fluorescence_green_filters: Optional filter pattern(s) for green fluorescence
            fluorescence_red_filters: Optional filter pattern(s) for red fluorescence
            fluorescence_blue_filters: Optional filter pattern(s) for blue fluorescence
            image_size: Optional tuple (width, height) specifying image size in μm

        Returns:
            DataFrame summarizing loaded image data

        Raises:
            Exception: If data has already been loaded for this Result
        """
        if self.data_has_been_loaded:
            raise Exception('Data has already been loaded')

        # Initialize storage
        self.files_dict = {}
        self.time_points_array = []

        # Prepare channel data
        channel_data = [
            (img_folder_path, brightfield_filters, 'brightfield'),
            (img_folder_path, fluorescence_green_filters, 'fluorescence_green'),
            (img_folder_path, fluorescence_red_filters, 'fluorescence_red'),
            (img_folder_path, fluorescence_blue_filters, 'fluorescence_blue')
        ]

        # Collect file paths for each channel in parallel
        with mp.Pool(processes=self.num_cores) as pool:
            for channel, files_dict, time_points in pool.imap(self._process_channel_data, channel_data):
                if files_dict is not None:
                    self.files_dict[channel] = files_dict
                    if not self.time_points_array:
                        self.time_points_array = time_points

        # Calculate relative times
        time_points_dt = [datetime.strptime(tp, '%Y-%m-%d %H:%M:%S') for tp in self.time_points_array]
        self.relative_time_array = [(tp - time_points_dt[0]).total_seconds() / 3600 for tp in time_points_dt]

        # Get valid wells from platemap
        wells = list(self.files_dict['brightfield'].keys())
        wells = [well for well in wells if any(well in value_list for value_list in self.platemap.replicates.values())]
        wells.sort(key=lambda x: (int(x[1:]), x[0]))

        print('Looking for Wells:', wells)

        # Process wells in parallel
        well_data = [
            (well, self.time_points_array, self.files_dict, self.relative_time_array,
             image_size, self.hdf5_key)
            for well in wells
        ]

        with mp.Pool(processes=self.num_cores) as pool:
            with h5py.File(self.hdf5_path, 'a') as hdf_file:
                for well, results in tqdm(
                        pool.imap(self._process_well_data, well_data),
                        total=len(well_data),
                        desc="Loading wells"
                ):
                    # Create series for this well
                    self.spheroid_dict[well] = SpheroidSeries(f'Spheroid-{well}')

                    # Process results for this well
                    for result in results:
                        # Add to spheroid series
                        self.spheroid_dict[well].add_spheroid_image(
                            result['spheroid'],
                            result['timepoint']
                        )

                        # Save to HDF5
                        timepoint_group = hdf_file.require_group(
                            f'{self.hdf5_key}/ImageSeries/{well}/{result["timepoint"]}'
                        )
                        timepoint_group.attrs['date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        timepoint_group.attrs['rel_timepoint'] = f'{result["rel_timepoint"]}'
                        timepoint_group.attrs['unit'] = 'h'
                        timepoint_group.attrs['image_size_x_y'] = result['spheroid'].image_size

                        # Save image paths
                        image_path_group = timepoint_group.require_group('image_paths')
                        for channel, path in result['image_paths'].items():
                            image_path_group.create_dataset(channel, data=path)

        self.data_has_been_loaded = True
        print('Images have been loaded successfully!')

        for spheroid_series in self.spheroid_dict.values():
            spheroid_series._result = self

    @property
    def nr_images(self) -> pd.DataFrame:

        data = {
            'index': ['A', '', '', '', 'B', '', '', '', 'C', '', '', '', 'D', '', '', '', 'E', '', '', '', 'F', '', '',
                      '', 'G', '', '', '', 'H', '', '', ''],
            1: [np.nan] * 8 * 4,
            2: [np.nan] * 8 * 4,
            3: [np.nan] * 8 * 4,
            4: [np.nan] * 8 * 4,
            5: [np.nan] * 8 * 4,
            6: [np.nan] * 8 * 4,
            7: [np.nan] * 8 * 4,
            8: [np.nan] * 8 * 4,
            9: [np.nan] * 8 * 4,
            10: [np.nan] * 8 * 4,
            11: [np.nan] * 8 * 4,
            12: [np.nan] * 8 * 4,
            'Channel': ['brightfield', 'fluorescence_green', 'fluorescence_red', 'fluorescence_blue', 'brightfield',
                        'fluorescence_green', 'fluorescence_red', 'fluorescence_blue', 'brightfield',
                        'fluorescence_green', 'fluorescence_red', 'fluorescence_blue', 'brightfield',
                        'fluorescence_green', 'fluorescence_red', 'fluorescence_blue', 'brightfield',
                        'fluorescence_green', 'fluorescence_red', 'fluorescence_blue', 'brightfield',
                        'fluorescence_green', 'fluorescence_red', 'fluorescence_blue', 'brightfield',
                        'fluorescence_green', 'fluorescence_red', 'fluorescence_blue', 'brightfield',
                        'fluorescence_green', 'fluorescence_red', 'fluorescence_blue'],
        }

        image_nr_dict = {}
        channel_nr_dict = {'brightfield': 0, 'fluorescence_green': 1, 'fluorescence_red': 2, 'fluorescence_blue': 3}
        for channel in self.files_dict:
            for well in self.files_dict[channel]:
                letter_nr, col = int(ord(well[0].lower()) - ord('a')), int(well[1:])
                row = letter_nr * 4 + channel_nr_dict[channel]

                nr_images = len(self.files_dict[channel][well])
                data[col][row] = str(nr_images)

        # Function to style rows with colors and borders
        def colorize_and_add_borders(row, index, row_group_count=4):
            # Color palette for groups
            colors = ["#C8C8C8", "#D5FFCC", "#FFCCCC", "#CCE5FF"]  # Gray, Green, Red, Blue
            group_index = index % row_group_count  # Grouping index
            color = colors[group_index]  # Assign color

            # Set background color
            styles = [f"background-color: {color}; border-right: 2px solid black;" for _ in row]

            # Add border if it's the last row in the group
            if (index + 1) % row_group_count == 0:
                styles = [style + " border-bottom: 2px solid black;" for style in styles]

            return styles

        # Apply styling function to the DataFrame
        def style_dataframe(df):
            return df.style.apply(
                lambda row: colorize_and_add_borders(row, row.name), axis=1
            )

        df = pd.DataFrame(data)
        # Replace NaNs with empty strings
        df.fillna("", inplace=True)
        pd.options.display.float_format = '{:d}'.format  # Deaktiviert die Anzeige als float
        return style_dataframe(pd.DataFrame(df))

    @staticmethod
    def _process_segmentation_well(args):
        """Process segmentation for a single well.

        Args:
            args: Tuple containing:
                - well: Well identifier
                - spheroid_dict: Dictionary of spheroid images
                - methods: Segmentation methods specification
                - reconstruct_border: Whether to reconstruct border
                - kwargs: Additional segmentation parameters

        Returns:
            Tuple of (well, success_count, total_count, contour_data)
        """
        well, spheroid_dict, methods, reconstruct_border, kwargs = args
        success_count = 0
        total_count = 0
        contour_data = {}

        for timepoint, spheroid_image in spheroid_dict[well].spheroid_image_dict.items():
            try:
                spheroid_image.segmentation(
                    methods=methods,
                    reconstruct_border=reconstruct_border,
                    **kwargs
                )
                if spheroid_image.contour is not None:
                    success_count += 1
                    contour_data[timepoint] = {
                        'contour': spheroid_image.contour,
                        'touches_border': spheroid_image.contour_touches_border,
                        'hdf5_key': spheroid_image.hdf5_key
                    }
                total_count += 1
            except Exception as e:
                print(f"Error processing {well}: {str(e)}")
                continue

        return well, success_count, total_count, contour_data

    def segmentation(self,
                     methods: str | tuple | list = ('thresholding', 'fluorescence_green'),
                     reconstruct_border: bool = True,
                     **kwargs) -> None:
        """Perform spheroid segmentation across all images using multiprocessing.

        Supports single or multiple segmentation attempts with different methods.
        Will try methods in order until successful for each image.

        Args:
            methods: Segmentation specification in one of these formats:
                - str: Channel to use with default thresholding (e.g. 'fluorescence_green')
                - tuple: (method, channel) e.g. ('ai', 'brightfield')
                - list: List of (method, channel) tuples to try in order
            reconstruct_border: Whether to reconstruct spheroid border when it
                extends beyond image bounds
            **kwargs: Method-specific parameters:
                For thresholding:
                    threshold: Intensity threshold multiplier (default: 1.35)
                    use_yen: Whether to use Yen's method (default: True)
                    border_margin: Margin in pixels for border detection (default: 5)
                For AI:
                    confidence: Detection confidence threshold (default: 0.65)

        Examples:
            # Simple thresholding on green channel
            result.segmentation('fluorescence_green')

            # AI segmentation with custom confidence
            result.segmentation(('ai', 'brightfield'), confidence=0.8)

            # Try thresholding first, fall back to AI
            result.segmentation([
                ('thresholding', 'fluorescence_green'),
                ('ai', 'brightfield')
            ], threshold=1.35, confidence=0.7)
        """
        if not self.data_has_been_loaded:
            raise Exception('No data has been loaded!')

        # Prepare arguments for parallel processing
        wells = list(self.spheroid_dict.keys())
        well_args = [
            (well, self.spheroid_dict, methods, reconstruct_border, kwargs)
            for well in wells
        ]

        # Process wells in parallel
        with mp.Pool(processes=self.num_cores) as pool:
            results = list(tqdm(
                pool.imap(self._process_segmentation_well, well_args),
                total=len(well_args),
                desc="Processing segmentation"
            ))

        # Summarize results
        total_success = sum(success for _, success, _, _ in results)
        total_images = sum(total for _, _, total, _ in results)

        print(f"\nSegmentation complete:")
        print(f"Successfully segmented {total_success}/{total_images} images")

        # Save contours to HDF5 and update SpheroidImage instances
        with h5py.File(self.hdf5_path, 'a') as hdf_file:
            for well, _, _, contour_data in results:
                for timepoint, data in contour_data.items():
                    # Save to HDF5
                    spheroid_image_group = hdf_file[data['hdf5_key']]
                    if 'contour' in spheroid_image_group:
                        del spheroid_image_group['contour']
                    spheroid_image_group.create_dataset('contour', data=data['contour'])
                    spheroid_image_group.attrs['touches_border'] = data['touches_border']

                    # Update SpheroidImage instance
                    spheroid_image = self.spheroid_dict[well].spheroid_image_dict[timepoint]
                    spheroid_image.contour = data['contour']
                    spheroid_image.contour_touches_border = data['touches_border']

    @property
    def nr_images(self) -> pd.DataFrame:

        data = {
            'index': ['A', '', '', '', 'B', '', '', '', 'C', '', '', '', 'D', '', '', '', 'E', '', '', '', 'F', '', '',
                      '', 'G', '', '', '', 'H', '', '', ''],
            1: [np.nan] * 8 * 4,
            2: [np.nan] * 8 * 4,
            3: [np.nan] * 8 * 4,
            4: [np.nan] * 8 * 4,
            5: [np.nan] * 8 * 4,
            6: [np.nan] * 8 * 4,
            7: [np.nan] * 8 * 4,
            8: [np.nan] * 8 * 4,
            9: [np.nan] * 8 * 4,
            10: [np.nan] * 8 * 4,
            11: [np.nan] * 8 * 4,
            12: [np.nan] * 8 * 4,
            'Channel': ['brightfield', 'fluorescence_green', 'fluorescence_red', 'fluorescence_blue', 'brightfield',
                        'fluorescence_green', 'fluorescence_red', 'fluorescence_blue', 'brightfield',
                        'fluorescence_green', 'fluorescence_red', 'fluorescence_blue', 'brightfield',
                        'fluorescence_green', 'fluorescence_red', 'fluorescence_blue', 'brightfield',
                        'fluorescence_green', 'fluorescence_red', 'fluorescence_blue', 'brightfield',
                        'fluorescence_green', 'fluorescence_red', 'fluorescence_blue', 'brightfield',
                        'fluorescence_green', 'fluorescence_red', 'fluorescence_blue', 'brightfield',
                        'fluorescence_green', 'fluorescence_red', 'fluorescence_blue'],
        }

        image_nr_dict = {}
        channel_nr_dict = {'brightfield': 0, 'fluorescence_green': 1, 'fluorescence_red': 2, 'fluorescence_blue': 3}
        for channel in self.files_dict:
            for well in self.files_dict[channel]:
                letter_nr, col = int(ord(well[0].lower()) - ord('a')), int(well[1:])
                row = letter_nr * 4 + channel_nr_dict[channel]

                nr_images = len(self.files_dict[channel][well])
                data[col][row] = str(nr_images)

        # Function to style rows with colors and borders
        def colorize_and_add_borders(row, index, row_group_count=4):
            # Color palette for groups
            colors = ["#C8C8C8", "#D5FFCC", "#FFCCCC", "#CCE5FF"]  # Gray, Green, Red, Blue
            group_index = index % row_group_count  # Grouping index
            color = colors[group_index]  # Assign color

            # Set background color
            styles = [f"background-color: {color}; border-right: 2px solid black;" for _ in row]

            # Add border if it's the last row in the group
            if (index + 1) % row_group_count == 0:
                styles = [style + " border-bottom: 2px solid black;" for style in styles]

            return styles

        # Apply styling function to the DataFrame
        def style_dataframe(df):
            return df.style.apply(
                lambda row: colorize_and_add_borders(row, row.name), axis=1
            )

        df = pd.DataFrame(data)
        # Replace NaNs with empty strings
        df.fillna("", inplace=True)
        pd.options.display.float_format = '{:d}'.format  # Deaktiviert die Anzeige als float
        return style_dataframe(pd.DataFrame(df))

    @property
    def timepoints(self) -> tuple[np.ndarray, np.ndarray]:
        """Get experiment timepoints in absolute and relative formats.

        Returns:
            Tuple of (absolute_times, relative_times) where:
                absolute_times: Array of datetime strings
                relative_times: Array of hours from experiment start
        """
        return self.time_points_array, self.relative_time_array

    @property
    def replicates(self) -> dict[str, SpheroidCollection]:
        """Get replicate groups based on platemap conditions.

        Returns:
            Dictionary mapping condition strings to SpheroidCollection objects
            representing spheroids with the same experimental conditions.
        """
        replicates_dict = {}  # {condition: SpheroidCollection}

        # Group spheroids by condition from platemap
        for key in self.platemap.replicates.keys():
            # Create condition string from platemap key
            condition = ', '.join([f"{item[0]}: {item[1]}" for item in key])
            wells_list = self.platemap.replicates[key]

            # Get valid wells that exist in spheroid_dict
            valid_wells = [key for key in wells_list if key in self.spheroid_dict]

            if not valid_wells:
                print(f"Warning: No valid wells found for condition '{condition}' in result '{self.name}'")
                continue

            # Create spheroid list for this condition
            spheroids = [self.spheroid_dict[well] for well in valid_wells]

            # Create or update SpheroidCollection
            if condition not in replicates_dict:
                replicates_dict[condition] = SpheroidCollection(
                    name=condition,
                    spheroid_list=spheroids,
                    result=self,
                    hdf5_path=self.hdf5_path
                )
            else:
                replicates_dict[condition].add_replicate_group(spheroids, self)

        return replicates_dict

    def metric(self, name: str = 'radius', mean: bool = False,
               plot: bool = False, skip_nan: bool = False,
               interpolate: bool = True, ignore_border: bool = True) -> pd.DataFrame | tuple:
        """Calculate specified metric across wells and timepoints.

        Computes metrics like radius or area for each well/timepoint and optionally
        averages across replicates defined in the platemap.

        Args:
            name: Metric to calculate ('radius', 'area', 'fluorescence_*')
            mean: Whether to average across replicate wells
            plot: Whether to display plot of results
            skip_nan: Whether to exclude NaN values from plots
            interpolate: Whether to interpolate missing timepoints
            ignore_border: Whether to exclude spheroids touching image border

        Returns:
            DataFrame containing metric values, optionally with mean/std across replicates

        Raises:
            Exception: If no data has been loaded
        """

        if not self.data_has_been_loaded:
            raise Exception('No Data has been loaded so far!')

        # check if it has been computed previously
        if name in self.metric_dfs:
            result_df = self.metric_dfs[name]
        else:
            print('start')
            wells = list(self.files_dict['brightfield'].keys())
            wells.sort(key=lambda x: (int(x[1:]), x[0]))
            result_df = pd.DataFrame(float(np.nan), index=self.relative_time_array, columns=wells)
            name_split_array = name.split('_')
            if name_split_array[0] == 'fluorescence':
                result_df_cumulative = result_df.copy()
                result_df_mean = result_df.copy()

            # Convert list of wells to string for tqdm description
            wells_list = list(self.spheroid_dict.keys())

            # Use context manager for tqdm
            from contextlib import closing
            with closing(tqdm(range(len(wells_list)),
                              desc=f'Processing wells',
                              leave=True)) as pbar:
                # Process wells
                for i, index in zip(pbar, wells_list):
                    spheroid_series = self.spheroid_dict[index]
                    for timepoint in list(spheroid_series.spheroid_image_dict.keys()):
                        rel_timepoint = (datetime.strptime(timepoint, '%Y-%m-%d %H:%M:%S') - datetime.strptime(
                            self.time_points_array[0], '%Y-%m-%d %H:%M:%S')).total_seconds() / 3600
                        spheroid_image = spheroid_series.spheroid_image_dict[timepoint]

                        if name == 'radius':
                            result_df.loc[rel_timepoint, index] = spheroid_image.radius
                        if name == 'area':
                            result_df.loc[rel_timepoint, index] = spheroid_image.area

                        # other options: assume fluorescence :)
                        if name_split_array[0] == 'fluorescence':
                            color = name_split_array[1]
                            try:
                                metric_fluorescence_tuple = spheroid_image.metric_fluorescence(color, ignore_border)
                                result_df_cumulative.loc[rel_timepoint, index], result_df_mean.loc[
                                    rel_timepoint, index] = metric_fluorescence_tuple
                            except:
                                result_df_cumulative.loc[rel_timepoint, index] = None
                                result_df_mean.loc[rel_timepoint, index] = None

                pbar.close()  # Explicitly close the progress bar

            if name_split_array[0] == 'fluorescence':
                self.metric_dfs[f'fluorescence_{name_split_array[1]}_mean'] = result_df_mean
                self.metric_dfs[f'fluorescence_{name_split_array[1]}_cumulative'] = result_df_cumulative
                if name_split_array[2] == 'mean':
                    result_df = result_df_mean
                if name_split_array[2] == 'cumulative':
                    result_df = result_df_cumulative
            else:
                self.metric_dfs[name] = result_df

        # result_df = self.metric_dfs['fluorescence_red_cumulative']/self.metric_dfs['fluorescence_green_cumulative']

        if interpolate:
            # Interpolieren der NaN-Werte mit linearer Methode
            result_df = result_df.interpolate(method='linear', axis=0, limit_direction='both')

        if mean:
            ''' return mean/std values for every replicate instead of for every Well '''
            for key in list(self.platemap.replicates.keys()):
                condition = ', '.join([f"{item[0]}: {item[1]}" for item in key])
                wells_list = self.platemap.replicates[key]
                valid_wells = [key for key in wells_list if key in result_df.columns]
                try:
                    results_mean_df[(condition, 'mean')] = result_df[valid_wells].mean(axis=1, skipna=True)
                    results_mean_df[(condition, 'std')] = result_df[valid_wells].std(axis=1, skipna=True)
                except:
                    results_mean_df = pd.DataFrame({
                        (condition, 'mean'): result_df[valid_wells].mean(axis=1, skipna=True),
                        (condition, 'std'): result_df[valid_wells].std(axis=1, skipna=True)
                    })
            if plot:  # seperat Plot-Style if mean is desired
                df_reshaped = np.array(results_mean_df.columns).reshape(int(len(list(results_mean_df.columns)) / 2), 2)
                for tupel_ in df_reshaped:
                    for el in tupel_:
                        if el[-1] == 'mean':
                            concentration = el[0]
                            mean_array = results_mean_df[el]
                        elif el[-1] == 'std':
                            std_array = results_mean_df[el]
                    x = np.array(results_mean_df.index.to_list()) / 24
                    if skip_nan:
                        valid_indices = ~np.isnan(mean_array)

                        # Arrays filtern
                        mean_array = mean_array[valid_indices]
                        std_array = std_array[valid_indices]
                        x = x[valid_indices]

                    plt.fill_between(x, mean_array - std_array, mean_array + std_array,
                                     alpha=0.5, color='gray')
                    plt.plot(x, mean_array, label=f"{concentration}")

                    # plt.errorbar(np.array(df.index.to_list())/24, mean_array, yerr=std_array, label=f"Hep3B: {concentration}", fmt='-o', capsize=5)

                plt.xlabel('Time [d]')
                # plt.ylabel(rf'{self.plot_name_dict[name]}')
                plt.ylim(0)
                # plt.xlim(1, 21)
                plt.legend(fontsize=5)
                plt.show()
            result_df = results_mean_df

        elif plot:
            ''' additional Plot of Result '''
            for index in list(result_df.keys()):
                if skip_nan:
                    valid_indices = ~np.isnan(result_df[index])
                    # Arrays filtern
                    x = (np.array(result_df.index.to_list()) / 24)[valid_indices]
                    y = (result_df[index])[valid_indices]
                else:
                    x = (np.array(result_df.index.to_list()) / 24)
                    y = (result_df[index])
                plt.plot(x, y, label=f"{index}")
            plt.xlabel('Time [d]')
            # plt.ylabel(rf'{self.plot_name_dict[name]}')
            plt.ylim(0)
            plt.xlim(1, 21)
            # plt.legend()
            plt.show()
            plt.show()

        return result_df

    def interactive(self, plot_anzahl: int):
        """Launch interactive visualization and analysis interface.

        Creates interactive widgets for:
        - Browsing through wells and timepoints
        - Viewing different imaging channels
        - Performing manual and automated segmentation
        - Visualizing analysis results

        Args:
            plot_anzahl: Number of image panels to display
        """
        wells_array = sorted(list(self.spheroid_dict.keys()), key=lambda x: (x[0], int(x[1:])))
        initial_well = wells_array[0]
        spheroid_series = self.spheroid_dict[initial_well]

        result_metric_array = ['radius', 'area']
        self.result_metric = 'radius'
        self.result_df = self.metric(self.result_metric, mean=False, plot=False)
        self.result_df_mean = self.metric(self.result_metric, mean=False, plot=False)

        # Methoden für Buttons
        def manual_segmentation(change):
            spheroid_series = self.spheroid_dict[well_pulldown.value]
            spheroid_image = spheroid_series.spheroid_image_dict[self.time_points_array[slider.value]]
            try:
                pass
                # spheroid_image.segmentation_manual('brightfield')
            except:
                print('error!')
                # spheroid_image.segmentation_manual('fluorescence_green')

        def thresholding_segmentation(change):
            spheroid_series = self.spheroid_dict[well_pulldown.value]
            spheroid_image = spheroid_series.spheroid_image_dict[self.time_points_array[slider.value]]
            # redo Segmentation
            try:
                spheroid_image.contour, spheroid_image.contour_touches_border = spheroid_image.segmentation_thresholding(
                    'fluorescence_green', thresholding_input.value)
                if spheroid_image.contour_touches_border:
                    try:
                        border_margin = 5
                        spheroid_image.contour = add_fitted_contour(spheroid_image.contour,
                                                                    fit_ellipse(spheroid_image.contour)[-1],
                                                                    spheroid_image._width - border_margin - 1,
                                                                    spheroid_image._height - border_margin - 1,
                                                                    border_margin + 1, border_margin + 1)
                    except:
                        pass
                del self.metric_dfs[self.result_metric]
                self.result_df = self.metric(self.result_metric, mean=False, plot=False)
                self.result_df_mean = self.metric(self.result_metric, mean=False, plot=False)
                # save generated contour
                with h5py.File(self.hdf5_path, 'a') as hdf_file:
                    spheroid_image_group = hdf_file[spheroid_image.hdf5_key]
                    del spheroid_image_group['contour']
                    spheroid_image_group.create_dataset('contour', data=spheroid_image.contour)
            except:
                pass
            update_display()

        def ai_segmentation(change):
            spheroid_series = self.spheroid_dict[well_pulldown.value]
            spheroid_image = spheroid_series.spheroid_image_dict[self.time_points_array[slider.value]]
            # redo Segmentation
            try:
                spheroid_image.contour, spheroid_image.contour_touches_border = spheroid_image.segmentation_detectron(
                    'brightfield', ai_input.value)
                if spheroid_image.contour_touches_border and reconstruct_border:
                    spheroid_image.contour = add_fitted_contour(spheroid_image.contour,
                                                                fit_ellipse(spheroid_image.contour)[-1],
                                                                spheroid_image._width - border_margin - 1,
                                                                spheroid_image._height - border_margin - 1,
                                                                border_margin + 1, border_margin + 1)
                self.result_df = self.metric(self.result_metric, mean=False, plot=False)
                self.result_df_mean = self.metric(self.result_metric, mean=False, plot=False)
                # save generated contour
                with h5py.File(self.hdf5_path, 'a') as hdf_file:
                    spheroid_image_group = hdf_file[spheroid_image.hdf5_key]
                    del spheroid_image_group['contour']
                    spheroid_image_group.create_dataset('contour', data=spheroid_image.contour)
            except:
                pass
            update_display()

        def delete_contour(change):
            spheroid_series = self.spheroid_dict[well_pulldown.value]
            spheroid_image = spheroid_series.spheroid_image_dict[self.time_points_array[slider.value]]
            # delete contour
            spheroid_image.contour = None
            spheroid_image.contour_touches_border = False
            # remove contour from hdf5-file
            with h5py.File(self.hdf5_path, 'a') as hdf_file:
                spheroid_image_group = hdf_file[spheroid_image.hdf5_key]
                del spheroid_image_group['contour']
                spheroid_image_group.create_dataset('contour', data=np.nan)

            # reload relevant metric
            self.result_df = self.metric(self.result_metric, mean=False, plot=False)
            self.result_df_mean = self.metric(self.result_metric, mean=False, plot=False)
            update_display()

        def delete_spheroid_image(change):
            print("Spheroid image deleted.")

        def handle_key(event):
            if event.key == 'right':
                slider.value = min(slider.value + 1, len(relative_times) - 1)
            elif event.key == 'left':
                slider.value = max(slider.value - 1, 0)

        ''' Sort Timepoints '''
        list(spheroid_series.spheroid_image_dict.keys()).sort(key=lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        relative_times = []
        start_time = datetime.strptime(self.time_points_array[0], '%Y-%m-%d %H:%M:%S')
        for time_str in self.time_points_array:
            current_time = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
            time_diff = current_time - start_time
            days = time_diff.days
            hours = time_diff.seconds // 3600
            relative_times.append(f"{days}d {hours}h")

        ''' Layout '''
        slider = widgets.IntSlider(value=0, min=0, max=len(relative_times) - 1,
                                   step=1,
                                   description='Time',
                                   layout=widgets.Layout(width='70%'))
        label = widgets.Label(value=relative_times[slider.value])
        well_pulldown = widgets.Dropdown(
            options=wells_array,
            description='Well',
            layout=widgets.Layout(width='150px')
        )

        outputs = []
        kanal_checkboxes = []
        dropdowns = []

        for i in range(plot_anzahl + 1):
            output = widgets.Output(layout=widgets.Layout(width='95%', height='300px'))
            outputs.append(output)
            if i in range(plot_anzahl):
                dropdown = widgets.Dropdown(
                    options=['brightfield', 'fluorescence_green', 'fluorescence_red', 'fluorescence_blue'],
                    description='Channel',
                    layout=widgets.Layout(width='70%')
                )
            else:
                dropdown = widgets.Dropdown(
                    options=result_metric_array,
                    description='Metric',
                    layout=widgets.Layout(width='70%')
                )
            dropdown.observe(lambda change: update_display(), names='value')
            dropdowns.append(dropdown)

        def update_display():
            spheroid_series = self.spheroid_dict[well_pulldown.value]
            spheroid_image = spheroid_series.spheroid_image_dict[self.time_points_array[slider.value]]
            for i in range(plot_anzahl + 1):
                with outputs[i]:
                    outputs[i].clear_output(wait=True)
                    fig, ax = plt.subplots(figsize=(4, 3))
                    for spine in ax.spines.values():
                        spine.set_edgecolor('black')
                        spine.set_linewidth(.5)

                    if i in range(plot_anzahl):
                        if dropdowns[i].value in spheroid_image.image_path_dict:
                            ax.imshow(cv2.cvtColor(cv2.imread(spheroid_image.image_path_dict[dropdowns[i].value]),
                                                   cv2.COLOR_BGR2RGB),
                                      extent=[0, spheroid_image.image_size[0], 0, spheroid_image.image_size[1]])
                            size_bar = AnchoredSizeBar(ax.transData, 300, '300 µm', 'lower right', pad=0, color='white',
                                                       frameon=False, size_vertical=10, borderpad=0.5)
                            ax.add_artist(size_bar)
                        else:
                            ax.text(800, 600, 'No Image', fontsize=15, ha='center', va='center', color='gray',
                                    alpha=0.7)

                        if show_contour_checkbox.value and spheroid_image.contour is not None:
                            ax.plot(spheroid_image.scaled_contour[:, 0],
                                    spheroid_image.scaled_contour[:, 1],
                                    color='b')
                        label.value = relative_times[slider.value]
                        ax.axis('off')
                        ax.set_aspect('equal', adjustable='box')
                        plt.xlim(0, spheroid_image.image_size[0])
                        plt.ylim(0, spheroid_image.image_size[1])
                        plt.show()
                    else:
                        if prolif_plot_checkbox.value:
                            if dropdowns[i].value is not self.result_metric:
                                self.result_metric = dropdowns[i].value
                                self.result_df = self.metric(self.result_metric, mean=False, plot=False)
                                self.result_df_mean = self.metric(self.result_metric, mean=False, plot=False)

                            # time_points = list(range(len(relative_times)))
                            # proliferation_data = [i * 2 for i in time_points]
                            # ax.plot(time_points, proliferation_data, label="Proliferation")
                            index = well_pulldown.value
                            if self.result_metric == 'radius':
                                y_value = spheroid_image.radius
                            if self.result_metric == 'area':
                                y_value = spheroid_image.area
                            ax.plot(np.array(self.result_df.index.to_list()) / 24, self.result_df[index],
                                    label=f"{index}")
                            if spheroid_image.contour is not None:
                                plt.plot(self.relative_time_array[slider.value] / 24, y_value, 'ro',
                                         markersize=5)  # 'ro' steht für roten Punkt
                            ax.set_xlabel("Time [d]")
                            ax.set_ylabel(self.result_metric)
                            ax.legend()
                            ax.grid(True)
                            plt.show()
                        else:
                            # Alles unsichtbar machen
                            ax.axis('off')  # Achsen ausschalten
                            fig.patch.set_visible(False)  # Hintergrund des Plots unsichtbar machen

                            plt.show()

        slider.observe(lambda change: update_display(), names='value')
        well_pulldown.observe(lambda change: update_display(), names='value')

        for i in range(plot_anzahl):
            dropdowns[i].observe(lambda change, i=i: update_display(), names='value')

        control_boxes = []
        for i in range(plot_anzahl + 1):
            control_box = widgets.VBox([dropdowns[i]],
                                       layout=widgets.Layout(margin='80px 0 0 0'))
            control_boxes.append(widgets.VBox([outputs[i], control_box]))

        plot_box = widgets.HBox(control_boxes,
                                layout=widgets.Layout(justify_content='center', width='100%', margin='0 0 0px 0'))

        slider_hbox = widgets.HBox([well_pulldown, slider, label], layout=widgets.Layout(align_items='center'))

        show_contour_checkbox = widgets.Checkbox(value=False, description="Contour",
                                                 layout=widgets.Layout(width="auto"))
        show_contour_checkbox.observe(lambda change: update_display(), names='value')
        segmentation_options_checkbox = widgets.Checkbox(value=False, description="Display Segmentation Options",
                                                         layout=widgets.Layout(width="auto"))
        prolif_plot_checkbox = widgets.Checkbox(value=True, description="Show Proliferation Plot",
                                                layout=widgets.Layout(width="auto"))
        delete_spheroid_button = widgets.Button(description="Delete SpheroidImage", on_click=delete_spheroid_image,
                                                layout=widgets.Layout(margin='0 50px 0 200px', width='200px'))
        delete_spheroid_button.style.button_color = 'red'

        tickbox_and_delete_button = widgets.HBox(
            [segmentation_options_checkbox, show_contour_checkbox, prolif_plot_checkbox, delete_spheroid_button],
            layout=widgets.Layout(margin='0 0px 0 0'))

        header = widgets.Label(value="Segmentation Options:",
                               layout=widgets.Layout(font_weight="bold", font_size="16px", margin='5px 0'))
        manual_button = widgets.Button(description="Manual", layout=widgets.Layout(margin='0 10px'))
        thresholding_button = widgets.Button(description="Thresholding", layout=widgets.Layout(margin='0 10px'))
        thresholding_input = widgets.FloatText(value=1.0, layout=widgets.Layout(width='80px'))
        thresholding_input.step = 0.1
        ai_button = widgets.Button(description="AI", layout=widgets.Layout(margin='0 10px'))
        ai_input = widgets.FloatText(value=0.7, layout=widgets.Layout(width='80px'))
        ai_input.step = 0.05
        delete_contour_button = widgets.Button(description="Delete Contour", layout=widgets.Layout(margin='0 10px'))
        delete_contour_button.style.button_color = 'coral'

        manual_button.on_click(manual_segmentation)
        thresholding_button.on_click(thresholding_segmentation)
        ai_button.on_click(ai_segmentation)
        delete_contour_button.on_click(delete_contour)

        segmentation_buttons = widgets.HBox([
            header,
            manual_button,
            widgets.HBox([thresholding_button, thresholding_input]),
            widgets.HBox([ai_button, ai_input]),
            delete_contour_button
        ], layout=widgets.Layout(justify_content='center', margin='0px 0'))

        def toggle_segmentation_buttons(change):
            if change.new:
                segmentation_buttons_box.children = [tickbox_and_delete_button, segmentation_buttons]
            else:
                segmentation_buttons_box.children = [tickbox_and_delete_button]

        segmentation_options_checkbox.observe(toggle_segmentation_buttons, names='value')
        segmentation_buttons_box = widgets.VBox([tickbox_and_delete_button], layout=widgets.Layout(margin='5px 0 0 0'))

        prolif_plot_checkbox.observe(lambda change: update_display(), names='value')

        complete_display = widgets.VBox([
            slider_hbox,
            plot_box,
            segmentation_buttons_box
        ], layout=widgets.Layout(margin='0 0 0px 0'))

        display(complete_display)
        update_display()

    def __results_mean(self, dict_key: str) -> pd.DataFrame:
        df_results = self.results_df_dict[dict_key]
        for key in list(self.platemap.replicates.keys()):
            condition = ', '.join([f"{item[0]}: {item[1]}" for item in key])
            wells_list = self.platemap.replicates[key]
            valid_wells = [key for key in wells_list if key in df_results.columns]
            try:
                results_mean_df[(condition, 'mean')] = df_results[valid_wells].mean(axis=1, skipna=True)
                results_mean_df[(condition, 'std')] = df_results[valid_wells].std(axis=1, skipna=True)
            except:
                results_mean_df = pd.DataFrame({
                    (condition, 'mean'): df_results[valid_wells].mean(axis=1, skipna=True),
                    (condition, 'std'): df_results[valid_wells].std(axis=1, skipna=True)
                })
        return results_mean_df

    # todo
    def _worksheet(self, workbook: Workbook) -> Workbook:
        pass

    def save_time_periods_to_hdf5(self):
        """Save time periods to HDF5 file for all collections."""
        for collection in self.replicates.values():
            collection.save_time_periods_to_hdf5()

    def load_time_periods_from_hdf5(self):
        """Load time periods from HDF5 file for all collections."""
        for collection in self.replicates.values():
            collection.load_time_periods_from_hdf5()

    def time_period(self, name: str,
                    start_time: str | datetime | None = None,
                    end_time: str | datetime | None = None,
                    description: str | None = None):
        """Smart method to set or update time period for all collections."""
        for collection in self.replicates.values():
            collection.time_period(
                name=name,
                start_time=start_time,
                end_time=end_time,
                description=description
            )
        self.save_time_periods_to_hdf5()

    def set_time_period(self, name: str,
                        start_time: str | datetime | None = None,
                        end_time: str | datetime | None = None,
                        description: str | None = None):
        """Set time period for all spheroid collections."""
        for collection in self.replicates.values():
            collection.set_time_period(
                name=name,
                start_time=start_time,
                end_time=end_time,
                description=description
            )
        self.save_time_periods_to_hdf5()

    def update_time_period(self, name: str,
                           start_time: str | datetime | None = None,
                           end_time: str | datetime | None = None,
                           description: str | None = None):
        """Update time period for all spheroid collections."""
        for collection in self.replicates.values():
            collection.update_time_period(
                name=name,
                start_time=start_time,
                end_time=end_time,
                description=description
            )
        self.save_time_periods_to_hdf5()

    def remove_time_period(self, name: str):
        """Remove time period from all spheroid collections."""
        for collection in self.replicates.values():
            collection.remove_time_period(name)
        self.save_time_periods_to_hdf5()