from __future__ import annotations
from typing import TYPE_CHECKING
import logging

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

# Suppress noisy numexpr INFO logs in main and worker processes
logging.getLogger("numexpr").setLevel(logging.WARNING)

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
    plot_name_dict: dict = {'radius': r'effective Radius ($r_{eff}=\sqrt{\frac{Area}{\pi}}$) [µm]'}
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
        wells = [well for well in wells if any(well in value_list for value_list in self.platemap.replicates().values())]
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

    @staticmethod
    def _process_segmentation_well(args):
        """Process segmentation for a single well.

        Args:
            args: Tuple containing:
                - well: Well identifier
                - timepoints: List of timepoints for this well
                - spheroid_images: Dictionary mapping timepoints to spheroid images
                - methods: Segmentation methods specification
                - reconstruct_border: Whether to reconstruct border
                - border_margin: Margin in pixels for border detection
                - kwargs: Additional segmentation parameters

        Returns:
            Tuple of (well, success_count, total_count, contour_data)
        """
        well, timepoints, spheroid_images, methods, reconstruct_border, border_margin, kwargs = args
        success_count = 0
        total_count = 0
        contour_data = {}

        for timepoint in timepoints:
            if timepoint not in spheroid_images:
                continue
                
            spheroid_image = spheroid_images[timepoint]
            try:
                spheroid_image.segmentation(
                    methods=methods,
                    reconstruct_border=reconstruct_border,
                    border_margin=border_margin,
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
                print(f"Error processing {well} at {timepoint}: {str(e)}")
                continue

        return well, success_count, total_count, contour_data

    def segmentation(self,
                     methods: str | tuple | list = ('thresholding', 'fluorescence_green'),
                     reconstruct_border: bool = True,
                     border_margin: int = 10,
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
            border_margin: Margin in pixels for border detection (default: 10)
            **kwargs: Method-specific parameters:
                For thresholding:
                    threshold: Intensity threshold multiplier (default: 1.35)
                    use_yen: Whether to use Yen's method (default: True)
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

        # Prepare arguments for parallel processing - only pass necessary data
        wells = list(self.spheroid_dict.keys())
        well_args = []
        
        for well in wells:
            if well not in self.spheroid_dict:
                continue
                
            # Get timepoints and prepare spheroid images dict for this well
            timepoints = list(self.spheroid_dict[well].spheroid_image_dict.keys())
            spheroid_images = self.spheroid_dict[well].spheroid_image_dict
            
            well_args.append((
                well, 
                timepoints,
                spheroid_images,
                methods, 
                reconstruct_border,
                border_margin,
                kwargs
            ))

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

    def replicates(self, element_name: str | None = None) -> dict:
        """Get replicate groups from platemap.
        
        Args:
            element_name: Optional name of cell line or compound to filter by
            
        Returns:
            Dictionary mapping conditions to SpheroidCollection objects
        """
        replicate_wells = self.platemap.replicates(element_name)
        collections = {}
        
        for condition, wells in replicate_wells.items():
            # Get valid spheroid series for these wells
            spheroid_list = [self.spheroid_dict[well] 
                           for well in wells 
                           if well in self.spheroid_dict]
            
            if spheroid_list:
                collections[condition] = SpheroidCollection(
                    name=str(condition),
                    spheroid_list=spheroid_list,
                    result=self,
                    hdf5_path=self.hdf5_path
                )
        
        return collections

    def metric(self, name: str = 'radius', mean: bool = False,
               plot: bool = False, skip_nan: bool = False,
               interpolate: bool = True, ignore_border: bool = True,
               element_name: str | None = None, update: bool = True,
               timepoint: float | str | datetime | None = None) -> pd.DataFrame | tuple:
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
            element_name: Optional name of cell line or compound to filter replicates by
            update: Whether to update cached metrics
            timepoint: Optional specific timepoint to calculate metrics for:
                - Relative timepoint in hours from experiment start
                - Datetime string in format 'YYYY-MM-DD HH:MM:SS'
                - Datetime object
                If None, calculates for all timepoints

        Returns:
            DataFrame containing metric values, optionally with mean/std across replicates
        """
        if not self.data_has_been_loaded:
            raise Exception('No Data has been loaded so far!')

        # Convert timepoint to relative hours if needed
        if timepoint is not None:
            if isinstance(timepoint, str):
                timepoint = datetime.strptime(timepoint, '%Y-%m-%d %H:%M:%S')
            if isinstance(timepoint, datetime):
                # Get first timepoint from any spheroid series
                first_series = next(iter(self.replicates(element_name).values())).spheroid_series[0]
                first_time = datetime.strptime(sorted(first_series.spheroid_image_dict.keys())[0], '%Y-%m-%d %H:%M:%S')
                timepoint = (timepoint - first_time).total_seconds() / 3600

        # Check if metric has been computed previously
        metric_key = f"{name}_{'mean' if mean else 'raw'}"
        if metric_key in self.metric_dfs and element_name is None and not update and timepoint is None:
            result_df = self.metric_dfs[metric_key]
        else:
            if mean:
                # Calculate metrics using SpheroidCollections (replicates)
                collection_dfs = []
                for collection in self.replicates(element_name).values():
                    df = collection.metric(
                        name=name,
                        mean=True,
                        interpolate=interpolate,
                        ignore_border=ignore_border,
                        skip_nan=skip_nan,
                        timepoint=timepoint
                    )
                    collection_dfs.append(df)
                
                # Combine all collection results
                result_df = pd.concat(collection_dfs, axis=1)
                
            else:
                # Calculate metrics for each spheroid series
                all_dfs = []
                for well, spheroid_series in self.spheroid_dict.items():
                    df = spheroid_series.metric(
                        name=name,
                        interpolate=interpolate,
                        ignore_border=ignore_border,
                        timepoint=timepoint,
                        plot=False
                    )
                    # Rename the column to use well ID directly instead of "Spheroid-" prefix
                    df.columns = [well]
                    all_dfs.append(df)

                # Combine all individual DataFrames
                result_df = pd.concat(all_dfs, axis=1)
            
            # Store for future use only if not filtered by element and calculating all timepoints
            if element_name is None and timepoint is None:
                self.metric_dfs[metric_key] = result_df

        # Handle plotting if requested
        if plot:
            self._plot_metric(result_df, mean, skip_nan, name)

        return result_df

    def _plot_metric(self, df: pd.DataFrame, mean: bool, skip_nan: bool, name: str):
        """Plot metric data with error bars if mean=True."""
        plt.figure(figsize=(6, 4))
        
        if mean:
            x = np.array(df.index.to_list()) / 24  # Convert to days
            
            # Handle both old-style ('condition', 'mean') and new-style 'mean' column names
            if isinstance(df.columns, pd.MultiIndex):
                # Old style - multiindex columns
                for condition in df.columns.levels[0]:
                    mean_col = (condition, 'mean')
                    std_col = (condition, 'std')
                    
                    mean_array = df[mean_col]
                    std_array = df[std_col]
                    
                    if skip_nan:
                        valid_mask = ~np.isnan(mean_array)
                        x_plot = x[valid_mask]
                        mean_plot = mean_array[valid_mask]
                        std_plot = std_array[valid_mask]
                    else:
                        x_plot = x
                        mean_plot = mean_array
                        std_plot = std_array
                    
                    plt.fill_between(x_plot, 
                                   mean_plot - std_plot,
                                   mean_plot + std_plot,
                                   alpha=0.3)
                    plt.plot(x_plot, mean_plot, label=condition)
            else:
                # New style - simple mean/std columns
                mean_array = df['mean']
                std_array = df['std']
                
                if skip_nan:
                    valid_mask = ~np.isnan(mean_array)
                    x_plot = x[valid_mask]
                    mean_plot = mean_array[valid_mask]
                    std_plot = std_array[valid_mask]
                else:
                    x_plot = x
                    mean_plot = mean_array
                    std_plot = std_array
                
                plt.fill_between(x_plot, 
                               mean_plot - std_plot,
                               mean_plot + std_plot,
                               alpha=0.3)
                plt.plot(x_plot, mean_plot)
        else:
            # Plot individual series
            for col in df.columns:
                y = df[col]
                x = np.array(df.index.to_list()) / 24
                
                if skip_nan:
                    valid_mask = ~np.isnan(y)
                    plt.plot(x[valid_mask], y[valid_mask], label=col, alpha=0.7)
                else:
                    plt.plot(x, y, label=col, alpha=0.7)
        
        plt.grid(True, alpha=0.3)
        plt.xlabel('Time [d]')
        
        # Use plot_name_dict if available
        if name in self.plot_name_dict:
            plt.ylabel(rf'{self.plot_name_dict[name]}')
        else:
            plt.ylabel(name)
            
        if not mean or len(df.columns) < 10:  # Only show legend if not too many series
            plt.legend()
            
        plt.title(f'{name} over time' + (' (mean ± std)' if mean else ''))
        plt.tight_layout()

    def save_time_periods_to_hdf5(self):
        """Save time periods to HDF5 file for all collections."""
        for collection in self.replicates().values():
            collection.save_time_periods_to_hdf5()

    def load_time_periods_from_hdf5(self):
        """Load time periods from HDF5 file for all collections."""
        for collection in self.replicates().values():
            collection.load_time_periods_from_hdf5()

    def time_period(self, name: str,
                    start_time: str | datetime | None = None,
                    end_time: str | datetime | None = None,
                    description: str | None = None):
        """Smart method to set or update time period for all collections."""
        for collection in self.replicates().values():
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
        for collection in self.replicates().values():
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
        for collection in self.replicates().values():
            collection.update_time_period(
                name=name,
                start_time=start_time,
                end_time=end_time,
                description=description
            )
        self.save_time_periods_to_hdf5()

    def remove_time_period(self, name: str):
        """Remove time period from all spheroid collections."""
        for collection in self.replicates().values():
            collection.remove_time_period(name)
        self.save_time_periods_to_hdf5()

    def condition_slice(self, element_name: str, timepoint: float | str | datetime, 
                        metric: str = 'radius', plot: bool = True) -> pd.DataFrame:
        """Analyze metric values across conditions at a specific timepoint.
        
        Args:
            element_name: Name of cell line or compound to analyze
            timepoint: Either:
                - Relative timepoint in hours from experiment start
                - Datetime string in format 'YYYY-MM-DD HH:MM:SS'
                - Datetime object
            metric: Metric to analyze ('radius', 'area', 'fluorescence_*')
            plot: Whether to display a line plot of results
            
        Returns:
            DataFrame with columns:
                - condition: Cell line or compound name
                - condition_value: Numerical value of the condition (if applicable)
                - mean: Mean value of the metric
                - std: Standard deviation of the metric
        """
        # Convert timepoint to relative hours if needed
        if isinstance(timepoint, str):
            timepoint = datetime.strptime(timepoint, '%Y-%m-%d %H:%M:%S')
        if isinstance(timepoint, datetime):
            # Get first timepoint from any spheroid series
            first_series = next(iter(self.replicates(element_name).values())).spheroid_series[0]
            first_time = datetime.strptime(sorted(first_series.spheroid_image_dict.keys())[0], '%Y-%m-%d %H:%M:%S')
            timepoint = (timepoint - first_time).total_seconds() / 3600
        
        # Get metrics for the element's replicates at the specific timepoint
        # Always use interpolation to handle missing values
        df = self.metric(
            name=metric,
            mean=True,
            element_name=element_name,
            plot=False,
            timepoint=timepoint,
            interpolate=True  # Force interpolation
        )
        
        # Extract values at that timepoint
        slice_df = df.iloc[0]  # Since we only have one timepoint now
        
        # Reshape into more readable format
        conditions = []
        condition_values = []
        means = []
        stds = []
        
        for col in df.columns:
            if col[1] == 'mean':  # Only process mean columns
                condition = col[0]  # Get condition name from MultiIndex
                
                # Handle string representation of tuple
                if isinstance(condition, str) and condition.startswith("((") and condition.endswith(",)"):
                    # Remove outer parentheses and split by comma
                    inner_part = condition.strip("()").strip(",")  # Remove outer () and trailing comma
                    # Remove the next layer of parentheses
                    content = inner_part.strip("()").split(",")
                    
                    # Extract cell line name (remove quotes)
                    condition_name = content[0].strip().strip("'")
                    
                    # Extract numeric value (should be the second element)
                    try:
                        condition_value = float(content[1].strip())
                    except (ValueError, IndexError):
                        condition_value = None
                        
                # Handle tuple format from platemap's find_replicates
                elif isinstance(condition, tuple):
                    if len(condition) == 1 and isinstance(condition[0], tuple):
                        inner_tuple = condition[0]
                        condition_name = inner_tuple[0]
                        try:
                            condition_value = float(inner_tuple[1])
                        except (ValueError, TypeError):
                            condition_value = inner_tuple[1]
                    else:
                        condition_name = element_name
                        try:
                            condition_value = float(condition[0])
                        except (ValueError, TypeError):
                            condition_value = condition[0]
                else:
                    condition_name = element_name
                    try:
                        condition_value = float(condition)
                    except (ValueError, TypeError):
                        condition_value = condition
                
                conditions.append(condition_name)
                condition_values.append(condition_value)
                means.append(slice_df[col])
                stds.append(slice_df[(condition, 'std')])
        
        result_df = pd.DataFrame({
            'condition': conditions,
            'condition_value': condition_values,
            'mean': means,
            'std': stds
        })
        
        # Sort by condition values if they're numeric
        if all(isinstance(x, (int, float)) for x in condition_values):
            result_df = result_df.sort_values('condition_value')

        if plot:
            plt.figure(figsize=(8, 6))  # Größer für Poster

            # Posterfreundlicher Plotstil
            #plt.style.use("seaborn-white")  # Clean look

            # Fehlerbalken-Liniendiagramm
            plt.errorbar(
                result_df['condition_value'],
                result_df['mean'],
                yerr=result_df['std'],
                fmt='o-',  # Punkte mit Linien
                color="#005c99",  # kontrastreicher Blauton
                ecolor="#005c99",  # neutrale Fehlerbalkenfarbe
                capsize=6,
                elinewidth=2,
                linewidth=2.5,
                markersize=8,
                alpha=0.9
            )

            # Achsentitel
            #plt.xlabel(f"{element_name} concentration", fontsize=20, labelpad=10)
            plt.xlabel(f"Seeding Cell Number", fontsize=20, labelpad=10)
            ylabel = "Spheroid Area $[µm^2]$"#self.plot_name_dict.get(metric, metric)
            plt.ylabel(ylabel, fontsize=20, labelpad=10)

            # Titel
            plt.title(f"{metric} at {timepoint:.1f} h", fontsize=24, fontweight="bold", pad=15)
            plt.title(f"Huh7", fontsize=20, fontweight="bold", pad=15)

            # Achsenticks vergrößern
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)

            # Optionale logarithmische Skala für die X-Achse
            if all(isinstance(x, (int, float)) for x in condition_values):
                max_val = max(condition_values)
                min_val = min(condition_values)
                if min_val > 0 and max_val / min_val > 100:
                    plt.xscale('log')
                    #plt.xlabel(f"log-scaled {element_name} concentration", fontsize=20, labelpad=10)

            # Gitter subtil halten
            plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.3)

            # Ränder optimieren
            plt.tight_layout()

            # Plot anzeigen
            plt.show()

        return result_df