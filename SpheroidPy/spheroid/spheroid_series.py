from datetime import datetime
import logging
from pathlib import Path
import os

import cv2
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
import numpy as np
from tqdm import tqdm
from matplotlib.colors import Normalize, TwoSlopeNorm, LinearSegmentedColormap
import h5py
import pandas as pd
import base64
from typing import Optional, Callable
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D

from SpheroidPy.spheroid.spheroid_image import SpheroidImage
from SpheroidPy.utils.time_period import TimePeriod, DatetimeOrRange
from SpheroidPy.spheroid.models.ward_and_king import WardAndKing
from SpheroidPy.spheroid.models.greenspan import GreenspanModel
from SpheroidPy.utils.file_management import collect_data
from SpheroidPy.utils.necrotic_models import (
    get_fit_model,
    get_default_initial_guess,
    get_default_bounds
)
import multiprocessing as mp

logger = logging.getLogger("SpheroidPy.spheroid_series")

def _process_necrotic_radius_single(args):
    """
    Helper function for multiprocessing to calculate necrotic radius for a single spheroid image.
    
    Args:
        args: Tuple containing:
            - timepoint: Timepoint identifier
            - image_path_dict: Dictionary of image paths
            - contour: Contour array or None
            - image_size: Image size tuple or None
            - channel: Fluorescence channel
            - thr: Threshold value
            - smoothing: Smoothing parameter
            - time_days: Time in days
    
    Returns:
        Tuple: (timepoint, outer_radius, necrotic_radius, time_days)
    """
    timepoint, image_path_dict, contour, image_size, channel, thr, smoothing, time_days = args
    
    try:
        # Recreate SpheroidImage from paths
        # Brightfield is optional now, but at least one channel must be present
        brightfield_path = image_path_dict.get('brightfield')
        
        kwargs = {}
        if 'fluorescence_green' in image_path_dict:
            kwargs['fluorescence_green'] = image_path_dict['fluorescence_green']
        if 'fluorescence_red' in image_path_dict:
            kwargs['fluorescence_red'] = image_path_dict['fluorescence_red']
        if 'fluorescence_blue' in image_path_dict:
            kwargs['fluorescence_blue'] = image_path_dict['fluorescence_blue']
        if image_size is not None:
            kwargs['image_size'] = image_size
        
        # Create SpheroidImage (brightfield can be None if other channels are available)
        try:
            spheroid_image = SpheroidImage(brightfield=brightfield_path, **kwargs)
        except ValueError as e:
            # If no channels available, return NaN
            return (timepoint, np.nan, np.nan, time_days)
        
        # Set contour if available
        if contour is not None:
            spheroid_image.contour = contour
        
        # Check if spheroid has a contour before attempting analysis
        if spheroid_image.contour is None:
            return (timepoint, np.nan, np.nan, time_days)
        
        # Get true radius (outer radius)
        true_radius = spheroid_image.radius
        if true_radius is None:
            return (timepoint, np.nan, np.nan, time_days)
        
        # Calculate threshold radius using thr_radius method
        threshold_radius = spheroid_image.thr_radius(
            channel=channel,
            thr=thr,
            plot=False,
            normalize=True,
            smoothing=smoothing,
            background_subtraction=True
        )
        
        # Calculate necrotic radius: threshold_radius * true_radius
        # If threshold_radius is None (no threshold found) or >= 1.0 (entire profile below threshold), set to 0
        if threshold_radius is None or threshold_radius >= 1.0:
            necrotic_radius = 0.0
        else:
            # threshold_radius is relative (0-1), multiply by true_radius to get absolute
            necrotic_radius = threshold_radius * true_radius
        
        return (timepoint, true_radius, necrotic_radius, time_days)
        
    except Exception as e:
        logger.warning(f"Failed to process timepoint {timepoint} in multiprocessing: {e}")
        return (timepoint, np.nan, np.nan, time_days)

class SpheroidSeries:
    """
    A series of spheroid images over time.

    Represents one biological replicate or sample tracked over multiple timepoints.

    Attributes
    ----------
    name : str
        Name of this spheroid series.
    spheroid_image_dict : dict
        Dictionary mapping timepoints to SpheroidImage instances.
    time_periods : dict of str -> TimePeriod
        Dictionary mapping period names to TimePeriod instances.
    _result : object or None
        Reference to the result of the last calculation.

    Parameters
    ----------
    name : str
        Name of the spheroid series.
    """
    name: str
    spheroid_image_dict: dict   # {timepoint: SpheroidImage}

    def __init__(self, name: str):
        """
        Initialize a SpheroidSeries.

        Parameters
        ----------
        name : str
            Name of the spheroid series.

        Notes
        -----
        - Initializes `spheroid_image_dict` as an empty dictionary.
        - Initializes `time_periods` as an empty dictionary.
        - Sets `_result` to None.
        """
        self.name: str = name
        self.spheroid_image_dict: dict[datetime.datetime, SpheroidImage] = {}
        self.time_periods: dict[str, TimePeriod] = {}
        self._result = None
        self._period_cache: dict[str, dict] = {}  # Cache for get_images_in_period results
        self.analysis_metrics: dict[str, dict[str, dict[str, float]]] = {}
        self._DEFAULT_METRIC_PERIOD = "__full_series__"

    @property
    def hdf5_path(self) -> str:
        """Get HDF5 file path from first spheroid image."""
        if not self.spheroid_image_dict:
            raise ValueError("No spheroid images available")
        return next(iter(self.spheroid_image_dict.values())).hdf5_path

    @property
    def hdf5_key(self) -> str:
        """Get HDF5 key from first spheroid image."""
        if not self.spheroid_image_dict:
            raise ValueError("No spheroid images available")
        # Get the base key (remove the timepoint part)
        full_key = next(iter(self.spheroid_image_dict.values())).hdf5_key
        return '/'.join(full_key.split('/')[:-1])

    def add_spheroid_image(self, spheroid: SpheroidImage, timepoint: datetime | str):
        """Add a spheroid image at a specific timepoint.
        
        Args:
            spheroid: SpheroidImage instance to add
            timepoint: When the image was taken (datetime or string in format 'YYYY-MM-DD HH:MM:SS')
        """
        self.spheroid_image_dict[timepoint] = spheroid
        # Sort by converting strings to datetime for comparison, but keep original format
        if all(isinstance(k, str) for k in self.spheroid_image_dict.keys()):
            # All keys are strings - sort as strings (they should be in sortable format)
            self.spheroid_image_dict = dict(sorted(self.spheroid_image_dict.items()))
        elif all(isinstance(k, datetime) for k in self.spheroid_image_dict.keys()):
            # All keys are datetime - sort as datetime
            self.spheroid_image_dict = dict(sorted(self.spheroid_image_dict.items()))
        else:
            # Mixed types - convert strings to datetime for sorting, then convert back
            # This shouldn't happen in normal operation, but handle it gracefully
            sorted_items = sorted(
                self.spheroid_image_dict.items(),
                key=lambda x: datetime.strptime(x[0], '%Y-%m-%d %H:%M:%S') if isinstance(x[0], str) else x[0]
            )
            self.spheroid_image_dict = dict(sorted_items)
        # Invalidate all period caches since the image collection changed
        self._period_cache.clear()
    
    @staticmethod
    def _process_channel_data(args):
        """Static helper function to process image channels in parallel.

        Args:
            args: Tuple of (img_folder_path, filters, channel_name, global_filter)

        Returns:
            Tuple of (channel_name, files_dict, time_points)
        """
        img_folder_path, filters, channel, global_filter = args
        if filters is None:
            return channel, None, None
        
        # Combine global_filter with channel-specific filters
        combined_filters = filters
        if global_filter:
            if isinstance(global_filter, str):
                combined_filters = [global_filter, filters] if isinstance(filters, str) else [global_filter] + (filters if isinstance(filters, list) else [filters])
            elif isinstance(global_filter, list):
                combined_filters = global_filter + (filters if isinstance(filters, list) else [filters])
        
        # Use collect_data but merge all indices (wells) into a single timepoint-based structure
        data_dict, time_points = collect_data(img_folder_path, combined_filters)
        
        # Merge all indices into a single timepoint-based dict
        # Structure: {timepoint: path} instead of {index: {timepoint: path}}
        merged_dict = {}
        for index_dict in data_dict.values():
            for timepoint, path in index_dict.items():
                # If multiple files for same timepoint, take the first one
                # (or could be extended to handle multiple files)
                if timepoint not in merged_dict:
                    if isinstance(path, list):
                        merged_dict[timepoint] = path[0]  # Take first if list
                    else:
                        merged_dict[timepoint] = path
        
        return channel, merged_dict, time_points
    
    @staticmethod
    def _process_timepoint_data(args):
        """Static helper function to process a single timepoint.

        Args:
            args: Tuple containing (timepoint, files_dict, image_size, hdf5_key)

        Returns:
            Tuple of (timepoint, spheroid_image, image_paths)
        """
        timepoint, files_dict, image_size, hdf5_key = args
        
        # Collect image paths for this timepoint
        image_paths = {}
        for channel in files_dict:
            if timepoint in files_dict[channel]:
                path = files_dict[channel][timepoint]
                if isinstance(path, list):
                    path = path[0]  # Take first if multiple
                image_paths[channel] = str(Path(path).absolute())
        
        # At least one channel must be present (brightfield is optional now)
        if not image_paths:
            return None  # Skip if no channels available
        
        # Create spheroid image (brightfield can be None if other channels are available)
        spheroid = SpheroidImage(
            brightfield=image_paths.get('brightfield'),  # Can be None
            fluorescence_green=image_paths.get('fluorescence_green'),
            fluorescence_red=image_paths.get('fluorescence_red'),
            fluorescence_blue=image_paths.get('fluorescence_blue'),
            image_size=image_size,
            hdf5_key=f'{hdf5_key}/ImageSeries/{timepoint}' if hdf5_key else None
        )
        
        return {
            'timepoint': timepoint,
            'spheroid': spheroid,
            'image_paths': image_paths
        }
    
    def load_images(self, 
                    img_folder_path: Path | str,
                    global_filter: str | list[str] | None = None,
                    brightfield_filter: str | list[str] | None = None,
                    green_filter: str | list[str] | None = None,
                    red_filter: str | list[str] | None = None,
                    blue_filter: str | list[str] | None = None,
                    image_size: tuple | None = None) -> None:
        """Load and process image data from specified folder using multiprocessing.
        
        This method loads images based on timepoints (not wells). It collects all images
        matching the filters and groups them by timepoint.
        
        Args:
            img_folder_path: Path to folder containing image files
            global_filter: Global filter pattern(s) applied to all channels (str or list[str])
            brightfield_filter: Filter pattern(s) for brightfield images (str or list[str])
            green_filter: Optional filter pattern(s) for green fluorescence (str or list[str])
            red_filter: Optional filter pattern(s) for red fluorescence (str or list[str])
            blue_filter: Optional filter pattern(s) for blue fluorescence (str or list[str])
            image_size: Optional tuple (width, height) specifying image size in μm
            
        Raises:
            Exception: If images have already been loaded for this series
        """
        if self.spheroid_image_dict:
            raise Exception('Images have already been loaded for this series. Clear spheroid_image_dict first if you want to reload.')
        
        img_folder_path = Path(img_folder_path)
        num_cores = mp.cpu_count()
        
        # Prepare channel data (with global_filter)
        channel_data = [
            (img_folder_path, brightfield_filter, 'brightfield', global_filter),
            (img_folder_path, green_filter, 'fluorescence_green', global_filter),
            (img_folder_path, red_filter, 'fluorescence_red', global_filter),
            (img_folder_path, blue_filter, 'fluorescence_blue', global_filter)
        ]
        
        # Collect file paths for each channel in parallel
        files_dict = {}
        time_points_array = []
        with mp.Pool(processes=num_cores) as pool:
            for channel, channel_files_dict, time_points in pool.imap(self._process_channel_data, channel_data):
                if channel_files_dict is not None:
                    files_dict[channel] = channel_files_dict
                    if not time_points_array:
                        time_points_array = time_points
        
        if not time_points_array:
            raise ValueError(f"No timepoints found in {img_folder_path} with the given filters")
        
        # Calculate relative times
        time_points_dt = [datetime.strptime(tp, '%Y-%m-%d %H:%M:%S') for tp in time_points_array]
        relative_time_array = [(tp - time_points_dt[0]).total_seconds() / 3600 for tp in time_points_dt]
        
        # Get HDF5 key if available
        hdf5_key = None
        if hasattr(self, 'hdf5_key'):
            try:
                hdf5_key = self.hdf5_key
            except:
                pass
        
        # Process timepoints in parallel
        timepoint_data = [
            (timepoint, files_dict, image_size, hdf5_key)
            for timepoint in time_points_array
        ]
        
        with mp.Pool(processes=num_cores) as pool:
            results = list(tqdm(
                pool.imap(self._process_timepoint_data, timepoint_data),
                total=len(timepoint_data),
                desc=f"Loading images for series '{self.name}'"
            ))
        
        # Add valid results to series
        hdf5_path = None
        if hasattr(self, 'hdf5_path'):
            try:
                hdf5_path = self.hdf5_path
            except:
                pass
        
        # Process results and optionally save to HDF5
        if hdf5_path and hdf5_key:
            with h5py.File(hdf5_path, 'a') as hdf_file:
                for result in results:
                    if result is None:
                        continue
                    
                    # Convert timepoint string to datetime
                    timepoint_dt = datetime.strptime(result['timepoint'], '%Y-%m-%d %H:%M:%S')
                    
                    # Add to spheroid series
                    self.add_spheroid_image(result['spheroid'], timepoint_dt)
                    
                    # Save to HDF5
                    timepoint_group = hdf_file.require_group(
                        f'{hdf5_key}/ImageSeries/{result["timepoint"]}'
                    )
                    timepoint_group.attrs['date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    timepoint_group.attrs['rel_timepoint'] = f'{relative_time_array[time_points_array.index(result["timepoint"])]}'
                    timepoint_group.attrs['unit'] = 'h'
                    if result['spheroid'].image_size:
                        timepoint_group.attrs['image_size_x_y'] = result['spheroid'].image_size
                    
                    # Save image paths
                    if 'image_paths' in timepoint_group:
                        del timepoint_group['image_paths']
                    image_path_group = timepoint_group.require_group('image_paths')
                    for channel, path in result['image_paths'].items():
                        image_path_group.create_dataset(channel, data=str(Path(path).absolute()).encode("utf-8"))
        else:
            # No HDF5, just add to series
            for result in results:
                if result is None:
                    continue
                
                # Convert timepoint string to datetime
                timepoint_dt = datetime.strptime(result['timepoint'], '%Y-%m-%d %H:%M:%S')
                
                # Add to spheroid series
                self.add_spheroid_image(result['spheroid'], timepoint_dt)
        
        print(f'Images have been loaded successfully for series "{self.name}"!')
        print(f'Loaded {len(self.spheroid_image_dict)} timepoints')

    def _normalize_metric_period(self, time_period: str | None) -> str:
        return time_period or self._DEFAULT_METRIC_PERIOD

    def _denormalize_metric_period(self, period_key: str | None) -> str | None:
        if period_key == self._DEFAULT_METRIC_PERIOD:
            return None
        return period_key

    def store_analysis_metrics(self, metric_name: str, metrics: dict[str, float], time_period: str | None = None) -> None:
        """
        Store derived analysis metric values (optionally scoped to a time period) for later aggregation.

        Parameters
        ----------
        metric_name : str
            The name under which the metric values are stored (e.g., 'necrotic_radius').
        metrics : dict[str, float]
            Dictionary where the key is the metric/feature name (e.g., 'growth_rate') and the value is a float.
        time_period : str | None, optional
            Name of the time period the metrics belong to. ``None`` stores the metrics for the full series.

        Returns
        -------
        None
            No return value. Stores the metrics in the attribute 'self.analysis_metrics'.

        Notes
        -----
        Entries with value None are filtered and not stored.
        """
        # Remove key-value pairs with None values; store the rest under the given metric name
        cleaned = {k: v for k, v in metrics.items() if v is not None}
        if cleaned:
            period_key = self._normalize_metric_period(time_period)
            self.analysis_metrics.setdefault(metric_name, {})
            self.analysis_metrics[metric_name][period_key] = cleaned

    def get_analysis_metrics(self, metric_name: str, time_period: str | None = None) -> dict | None:
        """
        Retrieve previously stored metrics for a given metric name and optional time period.

        Parameters
        ----------
        metric_name : str
            The key under which metrics were stored previously.
        time_period : str | None, optional
            When provided, only metrics for the specified period are returned.
            Otherwise a mapping of all periods to their metric dictionaries is returned.

        Returns
        -------
        dict or None
            Dictionary with metric values if available, otherwise None.
        """
        data = self.analysis_metrics.get(metric_name)
        if not data:
            return None
        if time_period is None:
            return {
                self._denormalize_metric_period(period_key): metrics
                for period_key, metrics in data.items()
            }
        period_key = self._normalize_metric_period(time_period)
        return data.get(period_key)

    def available_analysis_metrics(self) -> list[str]:
        """
        Return a list of all metric names that have stored values for this series.

        Returns
        -------
        list[str]
            List of strings, each representing a metric name with stored analysis values.
        """
        return list(self.analysis_metrics.keys())

    def available_metric_periods(self, metric_name: str) -> list[str | None]:
        """
        List the stored time periods for a given metric.

        Returns
        -------
        list[str | None]
            Available period labels (``None`` represents the full series).
        """
        data = self.analysis_metrics.get(metric_name)
        if not data:
            return []
        return [self._denormalize_metric_period(key) for key in data.keys()]

    def save_time_periods_to_hdf5(self):
        """Save time periods to HDF5 file."""
        with h5py.File(self.hdf5_path, 'a') as hdf_file:
            series_group = hdf_file[self.hdf5_key]
            
            # Remove existing time periods if any
            if 'time_periods' in series_group:
                del series_group['time_periods']
                
            # Create time periods group
            periods_group = series_group.create_group('time_periods')
            
            # Save each time period
            for name, period in self.time_periods.items():
                period_group = periods_group.create_group(name)
                period_group.attrs['start_time'] = period.start_time.isoformat()
                period_group.attrs['end_time'] = period.end_time.isoformat()
                
                # Save created_at and modified_at if they exist
                if hasattr(period, 'created_at') and period.created_at:
                    period_group.attrs['created_at'] = period.created_at.isoformat()
                if hasattr(period, 'modified_at') and period.modified_at:
                    period_group.attrs['modified_at'] = period.modified_at.isoformat()
                
                if period.description:
                    period_group.attrs['description'] = period.description
                
                # Save exclude list if present
                if period.exclude:
                    exclude_group = period_group.create_group('exclude')
                    for idx, ex in enumerate(period.exclude):
                        if isinstance(ex, datetime):
                            # Single timestamp
                            exclude_group.create_dataset(f'{idx}', data=ex.isoformat().encode('utf-8'))
                            exclude_group[f'{idx}'].attrs['type'] = 'datetime'
                        elif isinstance(ex, tuple):
                            # Time range tuple
                            start, end = ex
                            exclude_item_group = exclude_group.create_group(f'{idx}')
                            exclude_item_group.attrs['type'] = 'range'
                            if start:
                                exclude_item_group.attrs['start'] = start.isoformat()
                            if end:
                                exclude_item_group.attrs['end'] = end.isoformat()

    def load_time_periods_from_hdf5(self):
        """Load time periods from HDF5 file."""
        with h5py.File(self.hdf5_path, 'r') as hdf_file:
            series_group = hdf_file[self.hdf5_key]
            self.time_periods = {}
            
            if 'time_periods' in series_group:
                periods_group = series_group['time_periods']
                
                for name in periods_group:
                    period_group = periods_group[name]
                    
                    # Load time period attributes
                    start_time = datetime.fromisoformat(period_group.attrs['start_time'])
                    end_time = datetime.fromisoformat(period_group.attrs['end_time'])
                    description = period_group.attrs.get('description', None)
                    
                    # Load exclude list if present
                    exclude_list = []
                    if 'exclude' in period_group:
                        exclude_group = period_group['exclude']
                        # Sort keys to maintain order
                        exclude_keys = sorted(exclude_group.keys(), key=lambda x: int(x) if x.isdigit() else 0)
                        for key in exclude_keys:
                            exclude_item = exclude_group[key]
                            if exclude_item.attrs.get('type') == 'datetime':
                                # Single timestamp
                                exclude_str = exclude_item[()].decode('utf-8') if isinstance(exclude_item[()], bytes) else exclude_item[()]
                                exclude_list.append(datetime.fromisoformat(exclude_str))
                            elif exclude_item.attrs.get('type') == 'range':
                                # Time range tuple
                                start = exclude_item.attrs.get('start')
                                end = exclude_item.attrs.get('end')
                                start_dt = datetime.fromisoformat(start) if start else None
                                end_dt = datetime.fromisoformat(end) if end else None
                                exclude_list.append((start_dt, end_dt))
                    
                    # Create TimePeriod instance
                    period = TimePeriod(
                        name=name,
                        start_time=start_time,
                        end_time=end_time,
                        description=description,
                        exclude=exclude_list
                    )
                    
                    # Load created_at and modified_at if they exist
                    if 'created_at' in period_group.attrs:
                        period.created_at = datetime.fromisoformat(period_group.attrs['created_at'])
                    if 'modified_at' in period_group.attrs:
                        period.modified_at = datetime.fromisoformat(period_group.attrs['modified_at'])
                    
                    self.time_periods[name] = period

    def time_period(self, name: str,
                   start_time: str | datetime | None = None,
                   end_time: str | datetime | None = None,
                   exclude: list[DatetimeOrRange] = [],
                   description: str | None = None,
                   remove: bool = False) -> Optional[TimePeriod]:
        """
        Add, update, or remove a named `TimePeriod`.

        Parameters
        ----------
        name : str
            Unique identifier for the time period.

        start_time : datetime | str | None
            Start timestamp of the period. If None, defaults to the first
            available timestamp in the dataset. String format: 'YYYY-MM-DD HH:MM:SS'.

        end_time : datetime | str | None
            End timestamp of the period. If None, defaults to the last
            available timestamp in the dataset. String format: 'YYYY-MM-DD HH:MM:SS'.

        exclude : list[DatetimeOrRange]
            List of timestamps or time ranges to exclude from this period.

        description : str | None
            Optional textual description for context.

        remove : bool, default=False
            If True, removes the specified time period instead of creating
            or updating it.

        Returns
        -------
        TimePeriod | None
            The created or updated `TimePeriod`, or None if removed.

        Raises
        ------
        ValueError
            If there are no spheroid images to infer default start/end times.
        """
        if remove:
            self.time_periods.pop(name, None)
            # Invalidate cache for this period
            self._period_cache.pop(name, None)
            return None

        # Convert string times to datetime if needed
        if isinstance(start_time, str):
            start_time = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
        if isinstance(end_time, str):
            end_time = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')
            
        # Infer start/end defaults from sorted keys
        if not self.spheroid_image_dict:
            raise ValueError("No spheroid images available to infer default time bounds")
        
        sorted_times = sorted(self.spheroid_image_dict.keys())
        default_start = sorted_times[0]
        default_end = sorted_times[-1]

        if name in self.time_periods:
            # Update existing period
            tp = self.time_periods[name]
            if start_time is not None:
                tp.start_time = start_time
            if end_time is not None:
                tp.end_time = end_time
            if description is not None:
                tp.description = description
            # Update exclusions if provided
            if exclude:
                tp.exclude = exclude
            # Invalidate cache for this period since it was modified
            self._period_cache.pop(name, None)
        else:
            # Create a new time period
            tp = TimePeriod(
            name=name,
                start_time=start_time or default_start,
                end_time=end_time or default_end,
                description=description,
                exclude=exclude
            )
            self.time_periods[name] = tp

        return tp
           
    def get_images_in_period(self, name: str, use_cache: bool = True) -> dict:
        """
        Retrieve all spheroid images that fall within a specified time period,
        automatically excluding timestamps or intervals defined in the period.

        Parameters
        ----------
        name : str
            The name of the time period.
        use_cache : bool, default=True
            Whether to use cached results if available. Set to False to force
            recalculation (useful when time period or images have changed).

        Returns
        -------
        dict
            Dictionary mapping timestamps to SpheroidImage objects within the 
            specified time range that are not excluded.

        Raises
        ------
        KeyError
            If the specified time period does not exist.

        ValueError
            If the time period is missing start or end times.

        Notes
        -----
        - Results are cached for performance. Cache is automatically invalidated
          when time periods are modified or images are added/removed.
        - Filtering uses the `TimePeriod.contains()` method to respect all
          exclusions.
        - Use `use_cache=False` to force recalculation when you know the data
          has changed but the cache hasn't been invalidated.
        """
        if name not in self.time_periods:
            raise KeyError(f"No time period named '{name}' defined.")

        tp = self.time_periods[name]
        if tp.start_time is None or tp.end_time is None:
            raise ValueError(f"Time period '{name}' must define start and end times.")

        # Check cache first
        if use_cache and name in self._period_cache:
            return self._period_cache[name]

        # Helper function to convert timestamp to datetime if needed
        def _ensure_dt(val):
            return val if isinstance(val, datetime) else datetime.strptime(val, '%Y-%m-%d %H:%M:%S')

        # Filter by period and exclusions, returning dict format
        result = {
            timestamp: image 
            for timestamp, image in self.spheroid_image_dict.items()
            if tp.contains(_ensure_dt(timestamp))
        }
        
        # Cache the result
        self._period_cache[name] = result
        return result

    def clear_period_cache(self, period_name: str | None = None) -> None:
        """
        Clear the period cache for better memory management or when data changes.
        
        Parameters
        ----------
        period_name : str | None
            Specific period name to clear cache for, or None to clear all caches.
        """
        if period_name is None:
            self._period_cache.clear()
        else:
            self._period_cache.pop(period_name, None)

    def export_video(
        self,
        channel: str = 'brightfield',
        overlay_channels: list[str] = [],
        increases: list[str] = [],
        video_name: str = 'test.mp4',
        scalebar: bool = True,
        fps: int = 5,
        contour: bool = True,
        time: bool = True,
        exclude_out_of_contour: bool = True,
        diffusion_dict: dict = {},
        time_period: str | None = None,
        include_metric: str | None = None,
        metric_dict: dict | None = None,
        discard_non_conotour_images: bool = True,
        ax_metric=None,
    ):
        """
        Exports the spheroid series images to a video.

        :param parent_folder:
        :param channel: Channel to export (default: brightfield)
        :param overlay: other channels/contours/fields, that get overlayed (default: none)
        :param format: format of the video (default: ???)
        :param time_period: Optional name of a time period (created via time_period()).
                            If provided, only images within that time period are exported.
        :param include_metric: Optional metric name (e.g. 'radius', 'area', 'fluorescence_green_mean').
                               If provided, a metric plot panel is rendered to the right of the image
                               (grey curve + moving orange point), similar to the default plot shown in show().
        :param metric_dict: Optional dict to further specify metric rendering & calculation.
                            Supported keys (all optional):
                              - 'metric_name': override include_metric (string)
                              - 'ignore_border': bool (passed to SpheroidImage.metric)
                              - 'interpolate': bool (default False)
                              - 'plot_width_ratio': float (default 1.0, width of plot panel relative to image width)
                              - 'line_color': matplotlib color (default 'grey')
                              - 'point_color': matplotlib color (default '#e29266')
                              - 'dpi': int (default 100)
        :param ax_metric: Optional matplotlib Axes to use for the metric panel.
                          If provided, export_video will draw/update the moving orange point on this Axes
                          (you can pre-style it, pre-plot the curve, set limits, etc.), then render the Axes'
                          Figure canvas into the video panel each frame.
        :param discard_non_conotour_images: If True, skip frames where no contour is available.
                                            (Name kept for backward compatibility with existing notebooks.)
        :return:
        """
        # Initialisierung
        video = None
        time_0 = None

        # Optional: restrict exported frames to a named time period
        if time_period is not None:
            images_dict = self.get_images_in_period(time_period)
        else:
            images_dict = self.spheroid_image_dict

        if not images_dict:
            print("No images to export.")
            return

        if discard_non_conotour_images:
            images_dict = {
                tp: img for tp, img in images_dict.items()
                if getattr(img, "contour", None) is not None
            }
            if not images_dict:
                print("No images with contour available to export.")
                return

        # Normalize metric options
        metric_dict = metric_dict or {}
        metric_name = metric_dict.get("metric_name", include_metric)
        ignore_border = bool(metric_dict.get("ignore_border", False))
        interpolate_metric = bool(metric_dict.get("interpolate", False))
        plot_width_ratio = float(metric_dict.get("plot_width_ratio", 1.0))
        line_color = metric_dict.get("line_color", "grey")
        point_color = metric_dict.get("point_color", "#e29266")
        plot_dpi = int(metric_dict.get("dpi", 100))
        # Optional: style controls for the metric panel rendering
        # - style: matplotlib style name or list of style names
        # - rcParams: dict of matplotlib rcParams overrides (applied only for panel render)
        # - curve_kwargs: kwargs passed to ax.plot() for the grey curve
        # - point_kwargs: kwargs passed to ax.plot() for the moving point
        # - grid_kwargs: kwargs passed to ax.grid()
        # - xlabel/ylabel: override axis labels
        mpl_style = metric_dict.get("style", None)
        mpl_rcparams = metric_dict.get("rcParams", None)
        curve_kwargs = dict(metric_dict.get("curve_kwargs", {}) or {})
        point_kwargs = dict(metric_dict.get("point_kwargs", {}) or {})
        grid_kwargs = dict(metric_dict.get("grid_kwargs", {}) or {})
        xlabel_override = metric_dict.get("xlabel", None)
        ylabel_override = metric_dict.get("ylabel", None)
        # Global readability scaling for metric panels (applied to both generic metrics and necrotic_core)
        panel_scale = float(metric_dict.get("scale", 1.6))
        legend_fontsize = float(metric_dict.get("legend_fontsize", 11)) * panel_scale
        panel_gap_px = int(metric_dict.get("panel_gap_px", 25))  # padding between image and plot panel

        # Pre-compute metric series once (for the plot panel), if requested
        metric_times_days = None
        metric_values = None
        metric_df = None
        # Special metric: necrotic core (outer radius + necrotic radius)
        necrotic_series = None  # dict with keys: time(days), outer_radius, necrotic_radius
        if metric_name is not None:
            try:
                # Special-case: necrotic core plot panel uses necrotic_radius() data,
                # not the generic metric() dataframe.
                if str(metric_name).lower() in ("necrotic_core", "necrotic_radius", "necroticcore"):
                    # Allow configuring necrotic computation via metric_dict
                    nec_kwargs = dict(metric_dict.get("necrotic_kwargs", {}) or {})
                    # Defaults mirror necrotic_radius() signature; plot is always False here
                    nec_channel = nec_kwargs.pop("channel", metric_dict.get("channel", "red"))
                    nec_thr = float(nec_kwargs.pop("thr", metric_dict.get("thr", 0.5)))
                    nec_smoothing = float(nec_kwargs.pop("smoothing", metric_dict.get("smoothing", 2)))
                    nec_fit = bool(nec_kwargs.pop("fit", False))  # we don't need fits for the panel
                    nec_use_mp = bool(nec_kwargs.pop("use_multiprocessing", True))
                    nec_result = self.necrotic_radius(
                        channel=nec_channel,
                        plot=False,
                        time_period=time_period,
                        smoothing=nec_smoothing,
                        thr=nec_thr,
                        use_multiprocessing=nec_use_mp,
                        fit=nec_fit,
                        **nec_kwargs,
                    )
                    necrotic_series = nec_result
                    # We'll use this as our "metric panel"; keep metric_name as 'necrotic_core'
                    metric_name = "necrotic_core"
                else:
                    # Use the existing metric() machinery so we stay consistent with SpheroidImage.metric()
                    # NOTE: metric() index is relative hours; convert to days for plotting.
                    metric_df = self.metric(
                        name=str(metric_name),
                        time_period=time_period,
                        interpolate=interpolate_metric,
                        ignore_border=ignore_border,
                        plot=False,
                    )
                    metric_times_days = np.asarray(metric_df.index, dtype=float) / 24.0
                    # For MultiIndex fluorescence without kind, plot the 'mean' column if present.
                    if isinstance(metric_df.columns, pd.MultiIndex):
                        try:
                            metric_values = metric_df.xs('mean', level=1, axis=1).iloc[:, 0].to_numpy(dtype=float)
                        except Exception:
                            metric_values = metric_df.iloc[:, 0].to_numpy(dtype=float)
                    else:
                        metric_values = metric_df.iloc[:, 0].to_numpy(dtype=float)
            except Exception as e:
                print(f"Warning: could not compute metric '{metric_name}' for export_video(): {e}")
                metric_name = None

        # Central gating: only widen video AND render panel when we actually have panel data
        has_metric_panel = (
            metric_name is not None
            and (
                (metric_times_days is not None and metric_values is not None)
                or (str(metric_name).lower() == "necrotic_core" and necrotic_series is not None)
            )
        )

        # If a custom Axes is provided for the metric panel, prepare artists once and only update per frame.
        metric_ax = None
        metric_fig = None
        metric_point_artist = None
        metric_unit = ""
        if metric_name is not None:
            if str(metric_name).lower() == "area":
                metric_unit = "µm²"
            elif str(metric_name).lower() == "radius":
                metric_unit = "µm"

        if metric_name is not None and ax_metric is not None:
            try:
                import matplotlib
                from matplotlib.axes import Axes as _MplAxes
                from matplotlib.figure import Figure as _MplFigure
                from matplotlib.backends.backend_agg import FigureCanvasAgg

                if isinstance(ax_metric, _MplAxes):
                    metric_ax = ax_metric
                    metric_fig = metric_ax.figure
                elif isinstance(ax_metric, _MplFigure):
                    metric_fig = ax_metric
                    # take first axes if exists, else create one
                    metric_ax = metric_fig.axes[0] if metric_fig.axes else metric_fig.add_subplot(1, 1, 1)
                else:
                    metric_ax = None
                    metric_fig = None

                if metric_fig is not None:
                    # Ensure Agg canvas so we can read RGB buffer reliably
                    FigureCanvasAgg(metric_fig)

                # If the user didn't pre-plot anything, we plot the grey curve like show()
                if metric_ax is not None and (not metric_ax.lines) and metric_times_days is not None and metric_values is not None:
                    metric_ax.plot(metric_times_days, metric_values, color=line_color, zorder=5)

                # Create a point artist we will update per frame
                if metric_ax is not None:
                    (metric_point_artist,) = metric_ax.plot(
                        [np.nan],
                        [np.nan],
                        marker=".",
                        linestyle="None",
                        markersize=10,
                        color=point_color,
                        zorder=6,
                        label="",
                    )
            except Exception as e:
                print(f"Warning: ax_metric could not be initialized and will be ignored: {e}")
                metric_ax = None
                metric_fig = None
                metric_point_artist = None

        def _render_metric_panel(current_time_days: float, current_value: float | None, panel_h: int, panel_w: int) -> np.ndarray:
            """
            Render a metric plot panel (RGB) with grey curve and moving orange point.
            Output shape: (panel_h, panel_w, 3) uint8, RGB.
            """
            import matplotlib
            import matplotlib.pyplot as _plt

            def _canvas_to_rgb(canvas) -> np.ndarray:
                """Return canvas buffer as RGB across Matplotlib versions."""
                w, h = canvas.get_width_height()
                if hasattr(canvas, "tostring_rgb"):
                    rgb_buf = canvas.tostring_rgb()
                    return np.frombuffer(rgb_buf, dtype=np.uint8).reshape((h, w, 3))
                # Matplotlib versions that only expose RGBA buffer
                rgba = np.asarray(canvas.buffer_rgba(), dtype=np.uint8).reshape((h, w, 4))
                return rgba[:, :, :3]

            # If a user-provided ax_metric exists, update it and render its canvas
            if metric_ax is not None and metric_fig is not None and metric_point_artist is not None:
                try:
                    # Update moving point + legend label like show()
                    if current_value is not None and not np.isnan(current_value):
                        metric_point_artist.set_data([current_time_days], [current_value])
                        unit_part = f" {metric_unit}" if metric_unit else ""
                        metric_point_artist.set_label(f"{float(current_value):.1f}{unit_part} @ {float(current_time_days):.2f} d")
                        metric_ax.legend()
                    else:
                        metric_point_artist.set_data([np.nan], [np.nan])

                    metric_fig.tight_layout()
                    metric_fig.canvas.draw()
                    rgb = _canvas_to_rgb(metric_fig.canvas)
                    # Resize to desired panel size if needed
                    if rgb.shape[0] != panel_h or rgb.shape[1] != panel_w:
                        rgb = cv2.resize(rgb, (panel_w, panel_h), interpolation=cv2.INTER_AREA)
                    return rgb
                except Exception:
                    # Fall through to default render if something goes wrong
                    pass

            fig_w = max(1, panel_w) / max(1, plot_dpi)
            fig_h = max(1, panel_h) / max(1, plot_dpi)

            # Render inside a style/rcParams context (optional)
            style_ctx = _plt.style.context(mpl_style) if mpl_style is not None else _plt.style.context([])
            rc_ctx = matplotlib.rc_context(mpl_rcparams) if isinstance(mpl_rcparams, dict) else matplotlib.rc_context()
            with style_ctx, rc_ctx:
                fig, ax = _plt.subplots(figsize=(fig_w, fig_h), dpi=plot_dpi)

                # --- Special metric panel: necrotic_core (outer + necrotic radius) ---
                if metric_name is not None and str(metric_name).lower() == "necrotic_core" and necrotic_series is not None:
                    from matplotlib.colors import LinearSegmentedColormap

                    # Match necrotic_radius() plot "feel" (axes/legend/grid),
                    # but keep the soft gradient-line aesthetic from the user's snippet.
                    # Global scale-up for readability in video panels (fonts, ticks, markers, linewidths)
                    scale = panel_scale

                    title = metric_dict.get("title", "Necrotic Radius")
                    grid_alpha = float(metric_dict.get("grid_alpha", 0.3))
                    label_fs = float(metric_dict.get("label_fontsize", 14)) * scale
                    title_fs = float(metric_dict.get("title_fontsize", 16)) * scale
                    tick_fs = float(metric_dict.get("tick_fontsize", 11)) * scale

                    # Colors (dark -> light)
                    outer_dark = metric_dict.get("outer_dark", "#4b4b4b")
                    outer_light = metric_dict.get("outer_light", "#cfcfcf")
                    nec_dark = metric_dict.get("necrotic_dark", "#B71C1C")
                    nec_light = metric_dict.get("necrotic_light", "#EF9A9A")

                    # Point styling
                    ms = float(metric_dict.get("marker_size", 8)) * scale
                    mew = float(metric_dict.get("marker_edgewidth", 2.0)) * scale
                    points_alpha = float(metric_dict.get("points_alpha", 0.55))
                    current_alpha = float(metric_dict.get("current_alpha", 0.95))

                    # Line styling (gradient segments)
                    grad_lw = float(metric_dict.get("grad_lw", 6.0)) * scale
                    grad_alpha = float(metric_dict.get("grad_alpha", 0.07))

                    def _plot_gradient(ax_, x, y, c1, c2, lw=6, alpha=0.07, zorder=1):
                        cmap = LinearSegmentedColormap.from_list("grad", [c1, c2])
                        norm = _plt.Normalize(float(np.nanmin(x)), float(np.nanmax(x)))
                        for ii in range(len(x) - 1):
                            ax_.plot(
                                x[ii:ii+2],
                                y[ii:ii+2],
                                color=cmap(norm(x[ii])),
                                linewidth=lw,
                                solid_capstyle="round",
                                alpha=alpha,
                                zorder=zorder,
                            )

                    # Extract time + radii
                    t_days = np.asarray(necrotic_series.get("time", []), dtype=float)
                    outer = np.asarray(necrotic_series.get("outer_radius", []), dtype=float)
                    nec = np.asarray(necrotic_series.get("necrotic_radius", []), dtype=float)
                    fit_params = necrotic_series.get("fit_parameters") if isinstance(necrotic_series, dict) else None
                    has_any_fit = isinstance(fit_params, dict) and (
                        fit_params.get("outer_radius") is not None
                        or fit_params.get("necrotic_radius") is not None
                    )
                    # Default behavior: if fit is present, don't connect datapoints with lines.
                    connect_points_with_fit = bool(metric_dict.get("connect_points_with_fit", False))
                    show_data_connections = (not has_any_fit) or connect_points_with_fit
                    fit_grad_lw = float(metric_dict.get("fit_grad_lw", grad_lw * 1.35))
                    fit_grad_alpha = float(metric_dict.get("fit_grad_alpha", max(0.12, grad_alpha * 2.2)))
                    fit_line_lw = float(metric_dict.get("fit_line_lw", max(2.5, 2.8 * scale)))
                    fit_line_alpha = float(metric_dict.get("fit_line_alpha", 0.92))
                    draw_fit_foreground_line = bool(metric_dict.get("fit_foreground_line", False))
                    # Optional thin dark-red overlay line for outer + nec fits (without connecting datapoints)
                    plotstyle = metric_dict.get("plotstyle", None)
                    fit_overlay_style = metric_dict.get("fit_overlay_style", None)
                    _plotstyle_str = str(plotstyle or fit_overlay_style or "").lower()
                    draw_fit_thin_dark_red_overlay = _plotstyle_str in (
                        "thin_dark_red",
                        "darkred_thin",
                        "dark_red_thin",
                        "fit_thin_dark_red",
                    )
                    # Match style from _plot_necrotic_radius_series(): dashed fit lines
                    # - outer fit: '#555555' (grey)
                    # - nec fit:   '#aa0000' (dark red)
                    thin_overlay_lw = float(metric_dict.get("fit_thin_overlay_lw", max(1.1, 1.8 * scale)))
                    thin_overlay_alpha = float(metric_dict.get("fit_thin_overlay_alpha", 0.99))
                    thin_overlay_linestyle = metric_dict.get("fit_thin_overlay_linestyle", "--")
                    thin_overlay_color_outer = metric_dict.get("fit_thin_overlay_color_outer", "#555555")
                    thin_overlay_color_nec = metric_dict.get("fit_thin_overlay_color_nec", "#aa0000")

                    # Keep finite values only for plotting points/lines
                    valid_outer = np.isfinite(t_days) & np.isfinite(outer)
                    valid_nec = np.isfinite(t_days) & np.isfinite(nec)

                    # Outer radius (gradient line + transparent points)
                    scat_outer = None
                    if valid_outer.sum() >= 2:
                        x = t_days[valid_outer]
                        y = outer[valid_outer]
                        if show_data_connections:
                            _plot_gradient(ax, x, y, outer_dark, outer_light, lw=grad_lw, alpha=grad_alpha, zorder=1)
                        outer_line_handle, = ax.plot([], [], color=outer_dark, linewidth=2, label="Outer radius")  # legend handle
                        scat_outer, = ax.plot(
                            x,
                            y,
                            "o",
                            markersize=ms,
                            markerfacecolor=outer_light,
                            markeredgecolor=outer_dark,
                            markeredgewidth=mew,
                            alpha=points_alpha,
                            zorder=5,
                        )

                    # Necrotic core (gradient line + transparent points)
                    scat_nec = None
                    if valid_nec.sum() >= 2:
                        x = t_days[valid_nec]
                        y = nec[valid_nec]
                        if show_data_connections:
                            _plot_gradient(ax, x, y, nec_dark, nec_light, lw=grad_lw, alpha=grad_alpha, zorder=1)
                        nec_line_handle, = ax.plot([], [], color=nec_dark, linewidth=2, label="Necrotic radius")  # legend handle
                        scat_nec, = ax.plot(
                            x,
                            y,
                            "o",
                            markersize=ms,
                            markerfacecolor=nec_light,
                            markeredgecolor=nec_dark,
                            markeredgewidth=mew,
                            alpha=points_alpha,
                            zorder=5,
                        )

                    # Optional fit overlays: thick, soft background line + crisp foreground line.
                    if isinstance(fit_params, dict) and t_days.size > 1:
                        try:
                            t_fit = np.linspace(float(np.nanmin(t_days)), float(np.nanmax(t_days)), 200)
                        except Exception:
                            t_fit = None
                        if t_fit is not None:
                            # Outer linear fit
                            outer_fit = fit_params.get("outer_radius")
                            if isinstance(outer_fit, dict) and outer_fit.get("slope") is not None and outer_fit.get("intercept") is not None:
                                r_fit_outer = float(outer_fit["slope"]) * t_fit + float(outer_fit["intercept"])
                                _plot_gradient(
                                    ax, t_fit, r_fit_outer,
                                    outer_dark, outer_light,
                                    lw=fit_grad_lw, alpha=fit_grad_alpha, zorder=2
                                )
                                if draw_fit_thin_dark_red_overlay:
                                    ax.plot(
                                        t_fit,
                                        r_fit_outer,
                                        thin_overlay_linestyle,
                                        color=thin_overlay_color_outer,
                                        linewidth=thin_overlay_lw,
                                        alpha=thin_overlay_alpha,
                                        zorder=4,
                                    )
                                if draw_fit_foreground_line:
                                    ax.plot(
                                        t_fit,
                                        r_fit_outer,
                                        "-",
                                        color=outer_dark,
                                        linewidth=fit_line_lw,
                                        alpha=fit_line_alpha,
                                        zorder=3,
                                    )

                            # Necrotic fit
                            nec_fit = fit_params.get("necrotic_radius")
                            if isinstance(nec_fit, dict):
                                model_name = nec_fit.get("model", None)
                                if model_name:
                                    try:
                                        fit_func = get_fit_model(model_name)
                                        if model_name == "generic":
                                            param_order = ["t_nec", "slope", "plateau", "a"]
                                        elif model_name == "const_uptake":
                                            param_order = ["R0", "v", "r_l"]
                                        else:
                                            param_order = []
                                        if param_order and all(nec_fit.get(p) is not None for p in param_order):
                                            params_list = [float(nec_fit[p]) for p in param_order]
                                            r_fit_nec = fit_func(t_fit, *params_list)
                                            _plot_gradient(
                                                ax, t_fit, r_fit_nec,
                                                nec_dark, nec_light,
                                                lw=fit_grad_lw, alpha=fit_grad_alpha, zorder=2
                                            )
                                            if draw_fit_thin_dark_red_overlay:
                                                ax.plot(
                                                    t_fit,
                                                    r_fit_nec,
                                                        thin_overlay_linestyle,
                                                    color=thin_overlay_color_nec,
                                                    linewidth=thin_overlay_lw,
                                                    alpha=thin_overlay_alpha,
                                                    zorder=4,
                                                )
                                            if draw_fit_foreground_line:
                                                ax.plot(
                                                    t_fit,
                                                    r_fit_nec,
                                                    "-",
                                                    color=nec_dark,
                                                    linewidth=fit_line_lw,
                                                    alpha=fit_line_alpha,
                                                    zorder=3,
                                                )
                                    except Exception:
                                        pass

                    # Highlight current datapoints (same style, just less transparent + slightly larger)
                    if t_days.size > 0 and np.isfinite(current_time_days):
                        idx = int(np.nanargmin(np.abs(t_days - float(current_time_days))))
                        cur_ms = float(metric_dict.get("current_marker_size", ms * 1.25))
                        if idx < outer.size and np.isfinite(outer[idx]):
                            ax.plot(
                                t_days[idx],
                                outer[idx],
                                "o",
                                markersize=cur_ms,
                                markerfacecolor=outer_light,
                                markeredgecolor=outer_dark,
                                markeredgewidth=mew,
                                alpha=current_alpha,
                                zorder=10,
                            )
                        if idx < nec.size and np.isfinite(nec[idx]):
                            ax.plot(
                                t_days[idx],
                                nec[idx],
                                "o",
                                markersize=cur_ms,
                                markerfacecolor=nec_light,
                                markeredgecolor=nec_dark,
                                markeredgewidth=mew,
                                alpha=current_alpha,
                                zorder=10,
                            )

                    # Axes/legend like necrotic_radius()
                    ax.set_xlabel(xlabel_override or "Time [d]", fontsize=label_fs)
                    ax.set_ylabel(ylabel_override or "Radius [µm]", fontsize=label_fs)
                    ax.set_title(title, fontsize=title_fs)
                    ax.grid(True, alpha=grid_alpha)
                    ax.tick_params(axis="both", labelsize=tick_fs)
                    # Thicker spines for readability
                    spine_lw = float(metric_dict.get("spine_linewidth", 2.0)) * scale
                    for sp in ("left", "bottom"):
                        try:
                            ax.spines[sp].set_linewidth(spine_lw)
                        except Exception:
                            pass
                    # Legend structure like necrotic_radius(): add a combined datapoints entry
                    try:
                        from matplotlib.legend_handler import HandlerTuple
                        from matplotlib.lines import Line2D

                        handles, labels = ax.get_legend_handles_labels()

                        # Build the combined datapoints handle (outer / nec)
                        if scat_outer is not None and scat_nec is not None:
                            slash = Line2D([0], [0], marker=r'$\!/\!$', color='#777777',
                                           linestyle='None', markersize=max(6.0, 7.0 * scale))
                            handles.append((scat_outer, slash, scat_nec))
                            labels.append("Datapoints")

                            ax.legend(
                                handles,
                                labels,
                                loc="best",
                                fontsize=legend_fontsize,
                                handler_map={tuple: HandlerTuple(ndivide=None)},
                            )
                        else:
                            ax.legend(loc="best", fontsize=legend_fontsize)
                    except Exception:
                        ax.legend(loc="best", fontsize=legend_fontsize)

                    # Limits: keep user overrides if provided
                    if "xlim" in metric_dict:
                        ax.set_xlim(*metric_dict["xlim"])
                    else:
                        if np.isfinite(t_days).any():
                            ax.set_xlim(float(np.nanmin(t_days)), float(np.nanmax(t_days)))
                    if "ylim" in metric_dict:
                        ax.set_ylim(*metric_dict["ylim"])
                    else:
                        # auto y-range from both curves
                        y_all = np.r_[outer[np.isfinite(outer)], nec[np.isfinite(nec)]]
                        if y_all.size:
                            y0 = float(np.nanmin(y_all))
                            y1 = float(np.nanmax(y_all))
                            pad = 0.05 * max(1e-6, (y1 - y0))
                            ax.set_ylim(y0 - pad, y1 + pad)

                    # Make plot area tight so the content/labels appear larger within the fixed panel pixels
                    fig.tight_layout(pad=float(metric_dict.get("tight_layout_pad", 0.2)))
                    fig.canvas.draw()
                    rgb = _canvas_to_rgb(fig.canvas)
                    _plt.close(fig)
                    return rgb

                # Plot the full curve (like show() default: grey line)
                if metric_times_days is not None and metric_values is not None and len(metric_times_days) == len(metric_values):
                    # Apply scaling for generic metric panels too
                    scale = panel_scale
                    label_fs = float(metric_dict.get("label_fontsize", 14)) * scale
                    title_fs = float(metric_dict.get("title_fontsize", 14)) * scale
                    tick_fs = float(metric_dict.get("tick_fontsize", 11)) * scale
                    spine_lw = float(metric_dict.get("spine_linewidth", 2.0)) * scale

                    # Curve style defaults (show() uses grey line)
                    _curve_kwargs = {
                        "color": line_color,
                        "zorder": 5,
                        "linewidth": float(metric_dict.get("curve_linewidth", 2.0)) * scale,
                    }
                    _curve_kwargs.update(curve_kwargs)
                    ax.plot(metric_times_days, metric_values, **_curve_kwargs)

                    # Orange moving point with legend label like show()
                    if current_value is not None and not np.isnan(current_value):
                        # Unit hint (match show() label behaviour)
                        if str(metric_name).lower() == "area":
                            unit = "µm²"
                        elif str(metric_name).lower() == "radius":
                            unit = "µm"
                        else:
                            unit = ""
                        unit_part = f" {unit}" if unit else ""
                        point_label = f"{current_value:.1f}{unit_part} @ {current_time_days:.2f} d"

                        _point_kwargs = {
                            "marker": ".",
                            "linestyle": "None",
                            "markersize": float(metric_dict.get("current_marker_size", 10)) * scale,
                            "color": point_color,
                            "zorder": 6,
                            "label": point_label,
                        }
                        _point_kwargs.update(point_kwargs)
                        ax.plot(current_time_days, current_value, **_point_kwargs)
                        # In show(), legend is only shown if current_val exists
                        ax.legend(fontsize=legend_fontsize)

                    # Set xlim with small margins (match show() feel)
                    try:
                        ax.set_xlim(float(np.nanmin(metric_times_days)) - 0.3, float(np.nanmax(metric_times_days)) + 0.2)
                    except Exception:
                        pass
                else:
                    ax.text(0.5, 0.5, "No metric data", ha="center", va="center")

                # Grid (match show() default alpha=0.3)
                _grid_kwargs = {"alpha": 0.3}
                _grid_kwargs.update(grid_kwargs)
                ax.grid(True, **_grid_kwargs)

                # Labels/ticks/spines (scaled)
                ax.set_xlabel(xlabel_override or "Time [d]", fontsize=float(metric_dict.get("label_fontsize", 14)) * panel_scale)
                if ylabel_override is not None:
                    ax.set_ylabel(str(ylabel_override), fontsize=float(metric_dict.get("label_fontsize", 14)) * panel_scale)
                else:
                    # Simple ylabel heuristic
                    if str(metric_name).lower() == "area":
                        ax.set_ylabel("Area [µm²]", fontsize=float(metric_dict.get("label_fontsize", 14)) * panel_scale)
                    elif str(metric_name).lower() == "radius":
                        ax.set_ylabel("Radius [µm]", fontsize=float(metric_dict.get("label_fontsize", 14)) * panel_scale)
                    else:
                        ax.set_ylabel(str(metric_name), fontsize=float(metric_dict.get("label_fontsize", 14)) * panel_scale)

                ax.tick_params(axis="both", labelsize=float(metric_dict.get("tick_fontsize", 11)) * panel_scale)
                for sp in ("left", "bottom"):
                    try:
                        ax.spines[sp].set_linewidth(float(metric_dict.get("spine_linewidth", 2.0)) * panel_scale)
                    except Exception:
                        pass

                fig.tight_layout()
                fig.canvas.draw()
                rgb = _canvas_to_rgb(fig.canvas)
                _plt.close(fig)
                return rgb

        # Iteriere durch die Zeitstempel und Bilder
        for i, (time_key, spheroid) in enumerate(tqdm(images_dict.items(), desc="Processing Images")):
            # Lade das Bild basierend auf dem Kanal
            spheroid = images_dict[time_key]
            base_image_array = cv2.imread(str(spheroid.image_path_dict[channel]))[1:-1]

            if base_image_array is None:
                print(f"Warnung: Bild konnte nicht geladen werden: {spheroid.image_path_dict[channel]}")
                continue

            # Konvertiere den Zeitstempel in ein datetime-Objekt (Keys können str oder datetime sein)
            def _ensure_dt(val):
                return val if isinstance(val, datetime) else datetime.strptime(str(val), "%Y-%m-%d %H:%M:%S")

            time_key_dt = _ensure_dt(time_key)

            # Initialisiere das Video beim ersten Durchlauf
            if video is None:
                height, width = base_image_array.shape[:2]
                # If including metric panel, video width increases
                if has_metric_panel:
                    panel_w = int(width * max(plot_width_ratio, 0.2))
                    out_w = width + panel_gap_px + panel_w
                    out_h = height
                else:
                    out_w, out_h = width, height
                video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'XVID'), fps, (out_w, out_h))
                time_0 = time_key_dt

            # Erstelle eine Kopie des Basisbilds für Overlays
            combined_image = base_image_array.copy() #todo:ändern!!!

            if time:
                # Berechne die Zeitdifferenz
                time_diff = time_key_dt - time_0
                days = time_diff.days
                hours = time_diff.seconds // 3600
                time_difference_in_hours_and_days = f"{days}d {hours}h"

                # Füge den Zeitstempel als Text ins Bild ein
                cv2.putText(combined_image, time_difference_in_hours_and_days, (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 3.5, (255, 255, 255), 2)

            if contour and spheroid.contour is not None:
                # Mit cv2.drawContours zeichnen (grün, Linienstärke 2)
                cv2.drawContours(combined_image, [spheroid.contour.reshape((-1, 1, 2)).astype(np.int32)], -1, (255, 255, 255), 2)

            ''' Overlay '''
            color = {'red':(0,0,255), 'green':(0,255,0), 'blue':(255,0,0)}
            # Iteriere durch die Overlay-Kanäle
            for overlay_channel, increase in zip(overlay_channels, increases):
                # in case of fluorscence
                if overlay_channel.split('_')[0]=='fluorescence':
                    # Lade das Overlay-Bild (Fluoreszenzkanal)
                    try:
                        overlay_image = cv2.imread(str(spheroid.image_path_dict[overlay_channel]), cv2.IMREAD_GRAYSCALE)[:len(combined_image)]
                    except:
                        print(f"Warnung: Overlay-Bild konnte nicht geladen werden: {overlay_channel}")
                        continue

                    # Erstelle ein farbiges Overlay basierend auf der angegebenen Farbe
                    overlay_colored = np.stack( [np.zeros_like(overlay_image, dtype=np.uint8)] * 3, axis=-1)
                    for c, value in enumerate(color[overlay_channel.split('_')[1]]):
                        overlay_colored[:, :, c] = (overlay_image * value).astype(np.uint8)
                        overlay_colored[:, :, c] = np.clip(overlay_colored[:, :, c], 0, 255)

                        # fluoreszenz außerhalb von contour ausblenden
                        if exclude_out_of_contour==True and spheroid.contour is not None:
                            # Maske erstellen (gleiche Größe wie das Bild)
                            mask = np.zeros(overlay_colored[:, :, c].shape, dtype=np.uint8)
                            # Kontur auf der Maske zeichnen (gefüllt)
                            cv2.drawContours(mask, [spheroid.contour.reshape((-1, 1, 2)).astype(np.int32)], -1, 255, thickness=cv2.FILLED)

                            # Maske auf das Bild anwenden
                            overlay_colored[:, :, c] = cv2.bitwise_and(overlay_colored[:, :, c], overlay_colored[:, :, c], mask=mask)


                    # Kombiniere das Overlay-Bild mit dem Basisbild
                    alpha = overlay_image / 255.0 * increase  # Transparenz basierend auf Helligkeit
                    alpha = np.clip(alpha, 0, 1)

                    for c in range(3):
                        #combined_image[:, :, c] = (1 - alpha) * combined_image[:, :, c] #+ alpha * overlay_colored[:, :, c]
                        combined_image[:, :, c] = (1 - alpha) * combined_image[:, :,c] + alpha * overlay_colored[:, :, c]

                if overlay_channel == 'diffusion' and spheroid.contour is not None:
                    # solve Diffusion PDE
                    points, triangles, solution = spheroid.diffusion_stationary(diffusion_rate=diffusion_dict['diffusion_rate'], reaction_rate=diffusion_dict['reaction_rate'], boundary_value=diffusion_dict['boundary_value'], plot_full=False, plot=False, accuracy=30)

                    # Create a figure without axes
                    fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
                    ax.axis('off')  # Turn off axes
                    fig.tight_layout(pad=0)  # Keine Ränder

                    ax.imshow(cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB))  # Konvertiere ARGB zu BGRA
                    if 'cmap' in diffusion_dict:
                        ax.tricontourf(points[:, 0], points[:, 1], triangles, solution, levels=100, cmap=diffusion_dict['cmap'], alpha=.1*increase,
                                       norm=Normalize(vmin=0, vmax=diffusion_dict['boundary_value'], clip=False))
                    if 'contour_line_value' in diffusion_dict:
                        ax.tricontour(points[:, 0], points[:, 1], triangles, solution,
                                                        levels=[diffusion_dict['contour_line_value']], colors='blue', linestyle='dotted',
                                                        linewidths=1)
                        if 'contour_line_name' in diffusion_dict:
                            ax.plot([], [], color='blue', linestyle='solid', linewidth=1, label=diffusion_dict['contour_line_name'])
                            ax.legend(prop={'size': 20})

                    plt.xlim(0, width)
                    plt.ylim(height, 0)

                    # Matplotlib-Plot in ein OpenCV-kompatibles Bild umwandeln
                    fig.canvas.draw()
                    if hasattr(fig.canvas, "tostring_rgb"):
                        plot_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                        plot_image = plot_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                    else:
                        w_c, h_c = fig.canvas.get_width_height()
                        rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape((h_c, w_c, 4))
                        plot_image = rgba[:, :, :3]
                    combined_image = cv2.cvtColor(plot_image, cv2.COLOR_RGB2BGR)

            if scalebar:
                # Länge und Breite der Skalierungsleiste definieren
                scale_bar_length = int(width/spheroid.image_size[0]*300)  # Länge in Pixeln (entspricht z.B. 300µm)
                scale_bar_thickness = 15  # Dicke der Skalierungsleiste

                # Position der Skalierungsleiste festlegen
                x_start = 50  # Abstand vom linken Rand
                y_start = height - 50  # Abstand vom unteren Rand

                # Rechteck zeichnen (Skalierungsleiste)
                x_end = x_start + scale_bar_length
                y_end = y_start - scale_bar_thickness
                cv2.rectangle(combined_image, (x_start, y_start), (x_end, y_end), (255, 255, 255), -1)  # Weiße Leiste

                # Beschriftung hinzufügen
                # -*- coding: utf-8 -*-
                scale_text = "300 um"  # Beispieltext
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.5
                font_thickness = 2
                text_size = cv2.getTextSize(scale_text, font, font_scale, font_thickness)[0]
                text_x = x_start + (scale_bar_length - text_size[0]) // 2
                text_y = y_start - 30
                cv2.putText(combined_image, scale_text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)

            # Optional: append metric panel to the right (like show() default)
            if has_metric_panel:
                try:
                    # Determine current time/value for orange point.
                    # Prefer the closest metric time to this frame's time (in days).
                    current_days = (time_key_dt - time_0).total_seconds() / 86400.0
                    # necrotic_core: use necrotic_series to pick current value (outer radius for label)
                    if str(metric_name).lower() == "necrotic_core" and necrotic_series is not None:
                        t_days = np.asarray(necrotic_series.get("time", []), dtype=float)
                        outer = np.asarray(necrotic_series.get("outer_radius", []), dtype=float)
                        if t_days.size > 0:
                            idx_closest = int(np.nanargmin(np.abs(t_days - current_days)))
                            current_time_days = float(t_days[idx_closest])
                            current_value = float(outer[idx_closest]) if idx_closest < outer.size and np.isfinite(outer[idx_closest]) else None
                        else:
                            current_time_days = float(current_days)
                            current_value = None
                    else:
                        if metric_times_days is not None and len(metric_times_days) > 0:
                            idx_closest = int(np.nanargmin(np.abs(metric_times_days - current_days)))
                            current_value = float(metric_values[idx_closest]) if metric_values is not None else None
                            current_time_days = float(metric_times_days[idx_closest])
                        else:
                            current_time_days = float(current_days)
                            current_value = None

                    panel_w = int(width * max(plot_width_ratio, 0.2))
                    panel_h = height
                    panel_rgb = _render_metric_panel(current_time_days, current_value, panel_h, panel_w)
                    panel_bgr = cv2.cvtColor(panel_rgb, cv2.COLOR_RGB2BGR)
                    # Ensure panel height matches
                    if panel_bgr.shape[0] != combined_image.shape[0]:
                        panel_bgr = cv2.resize(panel_bgr, (panel_w, combined_image.shape[0]), interpolation=cv2.INTER_AREA)
                    gap = np.zeros((combined_image.shape[0], panel_gap_px, 3), dtype=np.uint8)
                    frame_out = np.concatenate([combined_image, gap, panel_bgr], axis=1)
                except Exception as e:
                    print(f"Warning: failed to render metric panel for frame {time_key}: {e}")
                    # Keep dimensions consistent with VideoWriter: append blank panel
                    panel_w = int(width * max(plot_width_ratio, 0.2))
                    blank_panel = np.zeros((combined_image.shape[0], panel_w, 3), dtype=np.uint8)
                    gap = np.zeros((combined_image.shape[0], panel_gap_px, 3), dtype=np.uint8)
                    frame_out = np.concatenate([combined_image, gap, blank_panel], axis=1)
            else:
                frame_out = combined_image

            # Schreibe das Bild ins Video
            video.write(frame_out)

        # Beende das Video und gebe Ressourcen frei
        if video is not None:
            print("Video abgeschlossen. Datei wurde gespeichert unter:", video_name)
            video.release()
        else:
            print("Kein Video erstellt. Möglicherweise wurden keine Bilder geladen.")

    def segmentation(self,
                     methods: list | tuple = [('thresholding', 'fluorescence_green'), ('ai', 'brightfield')],
                     reconstruct_border: bool = True,
                     border_margin: int = 5,
                     **kwargs) -> None:
        """
        Segment all SpheroidImage items.

        Args:
            methods: List/Tuple of segmentation specs, e.g.
                [('thresholding','fluorescence_green'), ('ai','brightfield')]
            reconstruct_border: Whether to reconstruct border if contour touches image border
            border_margin: Margin (px) for border detection
        """
        if not self.spheroid_image_dict:
            print("No images in series to segment.")
            return

        total = len(self.spheroid_image_dict)
        success = 0
        for tp, sph_img in self.spheroid_image_dict.items():
            try:
                sph_img.segmentation(
                    methods=methods,
                    reconstruct_border=reconstruct_border,
                    border_margin=border_margin,
                    **kwargs
                )
                if sph_img.contour is not None:
                    success += 1
            except Exception as e:
                print(f"Segmentation failed at {tp}: {e}")
                continue

        print(f"Segmentation finished: {success}/{total} images segmented successfully.")

    def necrotic_radius(self, channel: str = 'red', plot: bool = True,
                       time_period: str = None, smoothing: float = 2,
                       thr: float = 0.5, use_multiprocessing: bool = True,
                       fit: bool = True, fit_model: str | None = 'generic',
                       **kwargs) -> dict:
        """
        Calculate necrotic radius over time for a series of spheroid images.
        
        The necrotic radius is calculated as the threshold radius (from thr_radius method)
        multiplied by the spheroid radius. If no threshold is found, the necrotic radius is 0.
        
        Args:
            channel: Fluorescence channel to analyze ('green', 'red', or 'blue')
            plot: Whether to display a plot of the results
            time_period: Optional time period name to limit analysis to specific time range
            smoothing: Gaussian smoothing parameter (sigma) for radial profile
            thr: Threshold value for detecting necrotic region (0-1 if normalized)
            use_multiprocessing: Whether to use multiprocessing for parallel computation
            fit: Whether to fit functions to the data. If True, fits:
                - Linear function to outer_radius
                - Selected model to necrotic_radius (see fit_model parameter)
            fit_model: Model to use for fitting necrotic radius. Options:
                - 'generic': Generic model with t_nec, slope, plateau, a (default)
                - 'const_uptake': Constant oxygen uptake model with R0, v, r_l
                - None: Skip necrotic radius fitting
            **kwargs: Additional arguments for fitting:
                - params: Dictionary of fixed parameter values (not fitted).
                    For 'generic': {'t_nec': float, 'slope': float, 'plateau': float, 'a': float}
                    For 'const_uptake': {'R0': float, 'v': float, 'r_l': float}
                    Parameters in this dict will be kept fixed during fitting.
                - fit_params_initial: Dictionary of initial parameter guesses for fitting.
                    For 'generic': {'t_nec': float, 'slope': float, 'plateau': float, 'a': float}
                    For 'const_uptake': {'R0': float, 'v': float, 'r_l': float}
                    Only parameters not in 'params' will be fitted.
                - fit_params_bounds: Dictionary of parameter bounds as tuples (lower, upper).
                    Keys match parameter names. If bounds are provided, initial guesses will be set
                    to the midpoint of bounds (unless one bound is inf) for parameters not in 'params'.
            
        Returns:
            Dictionary containing:
                - time: List of timepoints in days
                - outer_radius: List of true (outer) radii in μm
                - necrotic_radius: List of necrotic radii in μm
                - fit_parameters: Nested dictionary with fit parameters (if fit=True):
                    - outer_radius: Dictionary with slope, intercept, r2, covariance
                    - necrotic_radius: Dictionary with model-specific parameters, r2, covariance
                - growth_rate: Growth rate in μm/day (slope from linear fit, if fit=True and both fits successful)
                - critical_radius: Critical radius in μm (outer_radius at t_nec, if fit=True and both fits successful)

        Notes:
            The derived scalar metrics (growth rate, critical radius, fit
            parameters) are cached via :meth:`store_analysis_metrics` so that
            collection/result/experiment level helpers can compare replicates.
        """
        # Get images to analyze (either all or from specific time period)
        if time_period is not None:
            if time_period not in self.time_periods:
                raise KeyError(f"Time period '{time_period}' not found. Available: {list(self.time_periods.keys())}")
            images_dict = self.get_images_in_period(time_period)
        else:
            images_dict = self.spheroid_image_dict
        
        # Prepare timepoints for time calculation
        timepoints_list = sorted(images_dict.keys())
        def _ensure_dt(val):
            return val if isinstance(val, datetime) else datetime.strptime(val, '%Y-%m-%d %H:%M:%S')
        if timepoints_list:
            t0 = _ensure_dt(timepoints_list[0])
            time_hours_dict = {
                tp: (_ensure_dt(tp) - t0).total_seconds() / 3600 
                for tp in timepoints_list
            }
        else:
            time_hours_dict = {}
        
        if use_multiprocessing and len(images_dict) > 1:
            # Use multiprocessing for parallel computation
            import multiprocessing as mp
            
            # Prepare arguments for parallel processing
            process_args = []
            for tp, spheroid_image in images_dict.items():
                # Convert Path objects to strings for pickling
                image_path_dict_str = {
                    k: str(v) if v is not None else None 
                    for k, v in spheroid_image.image_path_dict.items()
                }
                process_args.append((
                    tp,
                    image_path_dict_str,  # Pass paths as strings
                    spheroid_image.contour.copy() if spheroid_image.contour is not None else None,
                    spheroid_image.image_size if hasattr(spheroid_image, 'image_size') else None,
                    channel,
                    thr,
                    smoothing,
                    time_hours_dict.get(tp, 0) / 24  # Time in days
                ))
            
            # Process in parallel
            num_cores = mp.cpu_count()
            with mp.Pool(processes=num_cores) as pool:
                results = list(tqdm(
                    pool.imap(_process_necrotic_radius_single, process_args),
                    total=len(process_args),
                    desc="Calculating necrotic radius"
                ))
            
            # Sort results by timepoint to maintain order
            results.sort(key=lambda x: x[0])
            
            # Extract data
            time_data = [r[3] for r in results]  # time in days
            true_radius_data = [r[1] for r in results]  # outer radius
            necrotic_radius_data = [r[2] for r in results]  # necrotic radius
        else:
            # Sequential processing (original code)
            time_data = []
            true_radius_data = []
            necrotic_radius_data = []
            
            # Process each timepoint
            for tp, spheroid_image in images_dict.items():
                try:
                    # Check if spheroid has a contour before attempting analysis
                    if spheroid_image.contour is None:
                        logger.warning(f"No contour available for timepoint {tp}. Skipping necrotic radius analysis.")
                        # Add NaN values for this timepoint
                        true_radius_data.append(np.nan)
                        necrotic_radius_data.append(np.nan)
                        time_data.append(time_hours_dict.get(tp, 0) / 24)
                        continue
                    
                    # Get true radius (outer radius)
                    true_radius = spheroid_image.radius
                    if true_radius is None:
                        logger.warning(f"Could not calculate radius for timepoint {tp}. Skipping.")
                        true_radius_data.append(np.nan)
                        necrotic_radius_data.append(np.nan)
                        time_data.append(time_hours_dict.get(tp, 0) / 24)
                        continue
                    
                    # Calculate threshold radius using thr_radius method
                    threshold_radius = spheroid_image.thr_radius(
                        channel=channel,
                        thr=thr,
                        plot=False,
                        normalize=True,
                        smoothing=smoothing,
                        background_subtraction=True
                    )
                    
                    # Calculate necrotic radius: threshold_radius * true_radius
                    # If threshold_radius is None (no threshold found) or 1.0 (entire profile below threshold, no transition found), set to 0
                    if threshold_radius is None or threshold_radius >= 1.0:
                        necrotic_radius = 0.0
                    else:
                        # threshold_radius is relative (0-1), multiply by true_radius to get absolute
                        necrotic_radius = threshold_radius * true_radius
                    
                    true_radius_data.append(true_radius)
                    necrotic_radius_data.append(necrotic_radius)
                    time_data.append(time_hours_dict.get(tp, 0) / 24)
                    
                except Exception as e:
                    logger.warning(f"Failed to process timepoint {tp}: {e}")
                    # Add NaN values for failed timepoints
                    true_radius_data.append(np.nan)
                    necrotic_radius_data.append(np.nan)
                    time_data.append(time_hours_dict.get(tp, 0) / 24)
        
        # Create result dictionary
        result = {
            'time': time_data,
            'outer_radius': true_radius_data,
            'necrotic_radius': necrotic_radius_data
        }
        
        # Perform fitting if requested
        if fit:
            from scipy.optimize import curve_fit
            
            # Filter out NaN values for fitting
            time_array = np.array(time_data)
            outer_radius_array = np.array(true_radius_data)
            necrotic_radius_array = np.array(necrotic_radius_data)
            
            valid_mask_outer = ~np.isnan(outer_radius_array) & ~np.isnan(time_array)
            valid_mask_necrotic = ~np.isnan(necrotic_radius_array) & ~np.isnan(time_array) & (necrotic_radius_array > 0)
            
            # Fit linear function to outer radius
            if np.sum(valid_mask_outer) >= 2:
                try:
                    valid_time_outer = time_array[valid_mask_outer]
                    valid_outer = outer_radius_array[valid_mask_outer]
                    
                    # Linear fit: r(t) = a * t + b
                    def linear_func(t, a, b):
                        return a * t + b
                    
                    popt_outer, pcov_outer = curve_fit(linear_func, valid_time_outer, valid_outer)
                    
                    # Calculate R²
                    y_pred = linear_func(valid_time_outer, *popt_outer)
                    ss_res = np.sum((valid_outer - y_pred) ** 2)
                    ss_tot = np.sum((valid_outer - np.mean(valid_outer)) ** 2)
                    r2_outer = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
                    
                    fit_outer_radius = {
                        'slope': float(popt_outer[0]),
                        'intercept': float(popt_outer[1]),
                        'r2': float(r2_outer),
                        'covariance': pcov_outer.tolist()
                    }
                except Exception as e:
                    logger.warning(f"Failed to fit outer radius: {e}")
                    fit_outer_radius = None
            else:
                logger.warning("Not enough valid data points for outer radius fitting")
                fit_outer_radius = None
            
            # Fit selected model to necrotic radius
            if fit_model is not None and np.sum(valid_mask_necrotic) >= 2:
                try:
                    valid_time_necrotic = time_array[valid_mask_necrotic]
                    valid_necrotic = necrotic_radius_array[valid_mask_necrotic]
                    
                    # Extract kwargs for fitting
                    params = kwargs.get('params', {})  # Fixed parameters (not fitted)
                    fit_params_initial = kwargs.get('fit_params_initial', {})  # Initial guesses
                    fit_params_bounds = kwargs.get('fit_params_bounds', {})  # Parameter bounds
                    
                    # Get fit function for selected model
                    fit_func = get_fit_model(fit_model)
                    
                    # Determine parameter order for the model
                    if fit_model == 'generic':
                        param_order = ['t_nec', 'slope', 'plateau', 'a']
                    elif fit_model == 'const_uptake':
                        param_order = ['R0', 'v', 'r_l']
                    else:
                        param_order = []
                    
                    # Separate fixed and free parameters
                    fixed_params = {k: v for k, v in params.items() if k in param_order}
                    free_params = [p for p in param_order if p not in fixed_params]
                    
                    if not free_params:
                        logger.warning(f"All parameters are fixed. Cannot perform fitting.")
                        fit_necrotic_radius = None
                    else:
                        # Create wrapper function that only accepts free parameters
                        def fit_wrapper(t, *free_values):
                            """Wrapper function that combines fixed and free parameters."""
                            # Build full parameter list
                            full_params = {}
                            free_idx = 0
                            for param_name in param_order:
                                if param_name in fixed_params:
                                    full_params[param_name] = fixed_params[param_name]
                                else:
                                    full_params[param_name] = free_values[free_idx]
                                    free_idx += 1
                            # Call original function with full parameter list
                            return fit_func(t, *[full_params[p] for p in param_order])
                        
                        # Get default bounds for all parameters
                        default_lower_all, default_upper_all = get_default_bounds(
                            fit_model, valid_time_necrotic, valid_necrotic
                        )
                        
                        # Build bounds and initial guesses only for free parameters
                        bounds_lower = []
                        bounds_upper = []
                        p0 = []
                        bounds_dict_free = {}
                        
                        for param_name in free_params:
                            param_idx = param_order.index(param_name)
                            
                            # Get bounds (user-provided or default)
                            if param_name in fit_params_bounds:
                                lower_val, upper_val = fit_params_bounds[param_name]
                            else:
                                lower_val = default_lower_all[param_idx]
                                upper_val = default_upper_all[param_idx]
                            
                            bounds_lower.append(lower_val)
                            bounds_upper.append(upper_val)
                            bounds_dict_free[param_name] = (lower_val, upper_val)
                            
                            # Get initial guess (user-provided, midpoint of bounds, or default)
                            if param_name in fit_params_initial:
                                p0.append(fit_params_initial[param_name])
                            else:
                                # Use default initial guess
                                default_p0_all = get_default_initial_guess(
                                    fit_model, valid_time_necrotic, valid_necrotic, None
                                )
                                p0_val = default_p0_all[param_idx]
                                
                                # Adjust based on bounds if both are finite
                                if not (np.isinf(lower_val) or np.isinf(upper_val)):
                                    p0_val = (lower_val + upper_val) / 2.0
                                elif not np.isinf(lower_val):
                                    p0_val = lower_val + 0.1 * abs(lower_val) if lower_val != 0 else 0.1
                                elif not np.isinf(upper_val):
                                    p0_val = upper_val - 0.1 * abs(upper_val) if upper_val != 0 else 0.1
                                
                                p0.append(p0_val)
                        
                        bounds = (bounds_lower, bounds_upper)
                        
                        # Perform fit with wrapper function (only free parameters)
                        popt_necrotic_free, pcov_necrotic_free = curve_fit(
                            fit_wrapper,
                            valid_time_necrotic,
                            valid_necrotic,
                            p0=p0,
                            bounds=bounds,
                            maxfev=5000
                        )
                        
                        # Reconstruct full parameter list (fixed + fitted)
                        popt_necrotic_full = {}
                        free_idx = 0
                        for param_name in param_order:
                            if param_name in fixed_params:
                                popt_necrotic_full[param_name] = fixed_params[param_name]
                            else:
                                popt_necrotic_full[param_name] = popt_necrotic_free[free_idx]
                                free_idx += 1
                        
                        # Calculate R² using full parameter list
                        y_pred = fit_func(valid_time_necrotic, *[popt_necrotic_full[p] for p in param_order])
                        ss_res = np.sum((valid_necrotic - y_pred) ** 2)
                        ss_tot = np.sum((valid_necrotic - np.mean(valid_necrotic)) ** 2)
                        r2_necrotic = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
                        
                        # Build covariance matrix (only for free parameters, fixed params have 0 variance)
                        # For fixed parameters, we set covariance to 0
                        n_total = len(param_order)
                        n_free = len(free_params)
                        cov_full = np.zeros((n_total, n_total))
                        free_indices = [param_order.index(p) for p in free_params]
                        for i, idx_i in enumerate(free_indices):
                            for j, idx_j in enumerate(free_indices):
                                cov_full[idx_i, idx_j] = pcov_necrotic_free[i, j]
                        
                        # Store fit parameters in a dictionary with appropriate keys
                        fit_necrotic_radius = {
                            **{k: float(v) for k, v in popt_necrotic_full.items()},
                            'r2': float(r2_necrotic),
                            'covariance': cov_full.tolist(),
                            'model': fit_model,
                            'fixed_params': list(fixed_params.keys()) if fixed_params else []
                        }
                except Exception as e:
                    logger.warning(f"Failed to fit necrotic radius with model '{fit_model}': {e}")
                    fit_necrotic_radius = None
            else:
                if fit_model is None:
                    fit_necrotic_radius = None
                else:
                    logger.warning("Not enough valid data points for necrotic radius fitting")
                    fit_necrotic_radius = None
            
            # Calculate growth rate and critical radius if both fits are successful
            growth_rate = None
            critical_radius = None
            
            if fit_outer_radius is not None and fit_necrotic_radius is not None:
                # Growth rate = slope from linear fit (μm/day)
                growth_rate = fit_outer_radius['slope']
                
                # Critical radius calculation depends on model
                if fit_model == 'generic' and 't_nec' in fit_necrotic_radius:
                    # Critical radius = outer_radius at t_nec (from linear fit)
                    t_nec = fit_necrotic_radius['t_nec']
                    critical_radius = fit_outer_radius['slope'] * t_nec + fit_outer_radius['intercept']
                elif fit_model == 'const_uptake' and 'r_l' in fit_necrotic_radius:
                    # For const_uptake, critical radius is r_l (oxygen diffusion length)
                    critical_radius = fit_necrotic_radius['r_l']
            
            # Store fit results in nested dictionary structure
            result['fit_parameters'] = {
                'outer_radius': fit_outer_radius,
                'necrotic_radius': fit_necrotic_radius
            }
            
            # Store calculated values
            result['growth_rate'] = growth_rate  # μm/day
            result['critical_radius'] = critical_radius  # μm
        
        # Optional plotting
        if plot:
            self._plot_necrotic_radius_series(result, channel, time_period, fit=fit, fit_model=fit_model)

        fit_params = result.get('fit_parameters') or {}
        outer_fit = fit_params.get('outer_radius') or {}
        nec_fit = fit_params.get('necrotic_radius') or {}
        
        # Build metrics dictionary based on model type
        metrics_dict = {
            'growth_rate': result.get('growth_rate'),
            'critical_radius': result.get('critical_radius'),
            'outer_slope': outer_fit.get('slope'),
            'outer_intercept': outer_fit.get('intercept'),
            'outer_r2': outer_fit.get('r2'),
            'necrotic_r2': nec_fit.get('r2'),
            'necrotic_model': nec_fit.get('model', fit_model)
        }
        
        # Add model-specific parameters
        if nec_fit.get('model') == 'generic' or fit_model == 'generic':
            metrics_dict.update({
                'necrotic_t_nec': nec_fit.get('t_nec'),
                'necrotic_slope': nec_fit.get('slope'),
                'necrotic_plateau': nec_fit.get('plateau'),
                'necrotic_a': nec_fit.get('a'),
            })
        elif nec_fit.get('model') == 'const_uptake' or fit_model == 'const_uptake':
            metrics_dict.update({
                'necrotic_R0': nec_fit.get('R0'),
                'necrotic_v': nec_fit.get('v'),
                'necrotic_r_l': nec_fit.get('r_l'),
            })
        
        self.store_analysis_metrics(
            'necrotic_radius',
            metrics_dict,
            time_period=time_period,
        )
        
        return result
    
    def _plot_necrotic_radius_series(self, data: dict, channel: str, time_period: str | None = None, 
                                     fit: bool = False, fit_model: str | None = 'generic'):
        """Helper method to plot necrotic radius series."""
        fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
        
        time = data['time']
        true_radius = data['outer_radius']
        necrotic_radius = data['necrotic_radius']
        
        # Filter out NaN values for plotting
        valid_mask = ~np.isnan(true_radius) & ~np.isnan(necrotic_radius)
        if np.any(valid_mask):
            valid_time = np.array(time)[valid_mask]
            valid_true = np.array(true_radius)[valid_mask]
            valid_necrotic = np.array(necrotic_radius)[valid_mask]

            # Plot true radius (points only, no lines)
            scat_true, = ax.plot(valid_time, valid_true, 'o', color='darkgrey', alpha=0.6,
                   markersize=5, markeredgewidth=1.3, label='True Radius (Outer)' if not fit else None)
            
            # Plot necrotic radius (points only, no lines)
            scat_nec, = ax.plot(valid_time, valid_necrotic, 'o', color="#D98C8C", alpha=0.6,
                   markersize=5, markeredgewidth=1.3, label='Necrotic Radius' if not fit else None)
            
            # Plot fits if available
            if fit and 'fit_parameters' in data:
                fit_params = data['fit_parameters']
                
                # Plot linear fit for outer radius
                if fit_params.get('outer_radius') is not None:
                    outer_fit = fit_params['outer_radius']
                    t_fit = np.linspace(valid_time.min(), valid_time.max(), 100)
                    r_fit_outer = outer_fit['slope'] * t_fit + outer_fit['intercept']
                    r2_outer = outer_fit.get('r2', 0.0)
                    ax.plot(t_fit, r_fit_outer, '--', color='#555555', 
                           linewidth=2, alpha=0.99, 
                           label=f"Outer radius (R²={r2_outer:.2f})")
                
                # Plot necrotic radius fit using selected model
                if fit_params.get('necrotic_radius') is not None:
                    necrotic_fit = fit_params['necrotic_radius']
                    t_fit = np.linspace(valid_time.min(), valid_time.max(), 100)
                    
                    # Get fit function for the model used
                    model_name = necrotic_fit.get('model', fit_model or 'generic')
                    fit_func = get_fit_model(model_name)
                    
                    # Extract parameters based on model (works with both fixed and fitted params)
                    if model_name == 'generic':
                        # Get parameters in correct order
                        param_order = ['t_nec', 'slope', 'plateau', 'a']
                        params_list = [necrotic_fit.get(p) for p in param_order]
                        if all(p is not None for p in params_list):
                            r_fit_necrotic = fit_func(t_fit, *params_list)
                        else:
                            logger.warning(f"Missing parameters for generic model fit plot")
                            r_fit_necrotic = None
                    elif model_name == 'const_uptake':
                        # Get parameters in correct order
                        param_order = ['R0', 'v', 'r_l']
                        params_list = [necrotic_fit.get(p) for p in param_order]
                        if all(p is not None for p in params_list):
                            r_fit_necrotic = fit_func(t_fit, *params_list)
                        else:
                            logger.warning(f"Missing parameters for const_uptake model fit plot")
                            r_fit_necrotic = None
                    else:
                        # Fallback: try to use parameters list if available
                        if 'parameters' in necrotic_fit:
                            r_fit_necrotic = fit_func(t_fit, *necrotic_fit['parameters'])
                        else:
                            logger.warning(f"Cannot plot fit for unknown model: {model_name}")
                            r_fit_necrotic = None
                    
                    if r_fit_necrotic is not None:
                        r2_necrotic = necrotic_fit.get('r2', 0.0)
                        model_label = model_name.replace('_', ' ').title()
                        ax.plot(t_fit, r_fit_necrotic, '--', color='#aa0000', 
                               linewidth=2, alpha=0.99, 
                               label=f"Necrotic radius ({model_label}, R²={r2_necrotic:.2f})")

                    #ax.legend(handles, labels, handler_map={tuple: HandlerTuple(ndivide=None)})

                    plt.xlim(valid_time.min()-.3, valid_time.max()+.2)
        
        ax.set_xlabel('Time [d]', fontsize=12)
        ax.set_ylabel('Radius [μm]', fontsize=12)
        ax.set_title(f'Necrotic Radius', fontsize=16)
        ax.grid(True, alpha=0.3)

        handles, labels = ax.get_legend_handles_labels()

        # Beide Scatter zu einem Eintrag bündeln
        slash = Line2D([0], [0], marker=r'$\!/\!$', color='darkgrey',
               linestyle='None', markersize=7)
        handles.append((scat_true, slash, scat_nec))
        labels.append("Datapoints")

        if fit:
            # Only show legend if fits are available
            if 'fit_parameters' in data and (data['fit_parameters'].get('outer_radius') is not None or 
                                            data['fit_parameters'].get('necrotic_radius') is not None):
                ax.legend(handles, labels, handler_map={tuple: HandlerTuple(ndivide=None)})
            else:
                ax.legend(handles, labels, handler_map={tuple: HandlerTuple(ndivide=None)})
        
        plt.tight_layout()
        plt.savefig(f'{self.name}_necrotic_radius_series.png', dpi=300)
        plt.show()

    def save_mask(self, time_period: str | None = None, include_touches_border: bool = False, timespacing: float = 0.0) -> None:
        """
        Save masks as TIF files for all spheroid images in the series.
        
        Creates binary masks where pixels inside the contour are white (255) and 
        pixels outside are black (0). Masks are saved in the parent directory 
        in a 'mask' subfolder.
        
        Args:
            time_period: Optional time period name to limit which images to process.
                        If None, processes all images in the series.
            include_touches_border: If False, skips images where the contour touches 
                                   the image border. If True, includes all images.
            timespacing: Minimum time spacing in hours between saved masks. If set to > 0,
                        only masks that are at least this many hours apart will be saved.
                        Default is 0.0 (no spacing requirement).
        
        Returns:
            None
        
        Notes:
            - Mask filenames follow the format: {well_name}_{year}y{month}m{day}d_{hour}h{minute}m_mask.tif
            - Masks are saved in the parent directory of the first image's path, in a 'mask' subfolder
            - The 'mask' folder is created if it doesn't exist
            - Only images with valid contours are processed
            - When timespacing > 0, masks are filtered to ensure minimum time spacing between saved files
        """
        # Get images to process (either all or from specific time period)
        if time_period is not None:
            if time_period not in self.time_periods:
                raise KeyError(f"Time period '{time_period}' not found. Available: {list(self.time_periods.keys())}")
            images_dict = self.get_images_in_period(time_period)
        else:
            images_dict = self.spheroid_image_dict
        
        if not images_dict:
            logger.warning("No images to process for mask saving.")
            return
        
        # Get parent directory from first image path
        first_image = next(iter(images_dict.values()))
        if not first_image.image_path_dict:
            raise ValueError("No image paths available to determine parent directory.")
        
        # Use the first available image path to determine parent directory
        first_path = None
        for path in first_image.image_path_dict.values():
            if path is not None:
                first_path = Path(path)
                break
        
        if first_path is None:
            raise ValueError("No valid image path found to determine parent directory.")
        
        # Get parent directory and create mask subfolder
        parent_dir = first_path.parent.parent  # Go up one level from image directory
        mask_dir = parent_dir / "mask"
        mask_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract well name from series name (e.g., "Spheroid-B7" -> "B7")
        well_name = self.name
        if "Spheroid-" in well_name:
            well_name = well_name.replace("Spheroid-", "")
        elif "-" in well_name:
            # Try to extract well name (assume format like "B7" or similar)
            parts = well_name.split("-")
            well_name = parts[0] if parts else well_name
        
        # Process each image (sort by timepoint to ensure chronological order for timespacing)
        sorted_timepoints = sorted(images_dict.keys())
        
        # Helper function to convert timepoint to datetime
        def _ensure_dt(val):
            if isinstance(val, datetime):
                return val
            try:
                return datetime.strptime(str(val), '%Y-%m-%d %H:%M:%S')
            except:
                return None
        
        # Track last saved timepoint for timespacing
        last_saved_timepoint = None
        
        saved_count = 0
        skipped_count = 0
        
        for timepoint in tqdm(sorted_timepoints, desc="Saving masks"):
            spheroid_image = images_dict[timepoint]
            
            # Check timespacing requirement
            if timespacing > 0.0 and last_saved_timepoint is not None:
                current_dt = _ensure_dt(timepoint)
                last_dt = _ensure_dt(last_saved_timepoint)
                
                if current_dt is not None and last_dt is not None:
                    time_diff_hours = (current_dt - last_dt).total_seconds() / 3600.0
                    if time_diff_hours < timespacing:
                        logger.debug(f"Timepoint {timepoint} skipped due to timespacing requirement "
                                    f"({time_diff_hours:.2f}h < {timespacing}h).")
                        skipped_count += 1
                        continue
            # Skip if no contour
            if spheroid_image.contour is None:
                logger.debug(f"No contour available for timepoint {timepoint}. Skipping.")
                skipped_count += 1
                continue
            
            # Skip if touches border and include_touches_border is False
            if not include_touches_border and spheroid_image.contour_touches_border:
                logger.debug(f"Contour touches border for timepoint {timepoint}. Skipping.")
                skipped_count += 1
                continue
            
            # Get image dimensions from first available image
            image_path = None
            for path in spheroid_image.image_path_dict.values():
                if path is not None:
                    image_path = Path(path)
                    break
            
            if image_path is None:
                logger.warning(f"No image path available for timepoint {timepoint}. Skipping.")
                skipped_count += 1
                continue
            
            # Load image to get dimensions
            try:
                img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    logger.warning(f"Could not load image from {image_path}. Skipping.")
                    skipped_count += 1
                    continue
                height, width = img.shape
            except Exception as e:
                logger.warning(f"Error loading image from {image_path}: {e}. Skipping.")
                skipped_count += 1
                continue
            
            # Create binary mask
            mask = np.zeros((height, width), dtype=np.uint8)
            
            # Get contour (already in pixel coordinates)
            contour = spheroid_image.contour
            if contour is not None:
                # Convert to integer coordinates for OpenCV
                contour_pixels = contour.astype(np.int32)
                
                # Ensure contour is in the correct shape for cv2.fillPoly
                # cv2.fillPoly expects a list of contours, each as (N, 1, 2) or (N, 2)
                if contour_pixels.ndim == 2 and contour_pixels.shape[1] == 2:
                    # Reshape to (N, 1, 2) if needed
                    contour_pixels = contour_pixels.reshape(-1, 1, 2)
                
                # Fill the contour area with white (255)
                cv2.fillPoly(mask, [contour_pixels], 255)
            
            # Format timepoint for filename
            if isinstance(timepoint, datetime):
                dt = timepoint
            else:
                # Try to parse as string
                try:
                    dt = datetime.strptime(str(timepoint), '%Y-%m-%d %H:%M:%S')
                except:
                    # Fallback: use current time or timepoint string
                    logger.warning(f"Could not parse timepoint {timepoint}. Using current time.")
                    dt = datetime.now()
            
            # Create filename: {well_name}_{year}y{month}m{day}d_{hour}h{minute}m_mask.tif
            filename = f"{well_name}_{dt.year:04d}y{dt.month:02d}m{dt.day:02d}d_{dt.hour:02d}h{dt.minute:02d}m.tif"
            mask_path = mask_dir / filename
            
            # Save mask as TIF
            try:
                cv2.imwrite(str(mask_path), mask)
                saved_count += 1
                last_saved_timepoint = timepoint  # Update last saved timepoint
                logger.debug(f"Saved mask: {mask_path}")
            except Exception as e:
                logger.error(f"Error saving mask to {mask_path}: {e}")
                skipped_count += 1
        
        logger.info(f"Mask saving completed: {saved_count} masks saved, {skipped_count} skipped.")
        if saved_count > 0:
            logger.info(f"Masks saved to: {mask_dir}")

    @property
    def timepoints(self) -> list:
        """Get timepoints as list of datetime objects."""
        return sorted(self.spheroid_image_dict.keys())

    @property
    def relative_timepoints(self) -> list:
        """Get timepoints relative to the first timepoint."""
        timepoints = sorted(self.spheroid_image_dict.keys())
        def _ensure_dt(val):
            return val if isinstance(val, datetime) else datetime.strptime(val, '%Y-%m-%d %H:%M:%S')
        t0 = _ensure_dt(timepoints[0])
        return [(_ensure_dt(tp) - t0).total_seconds() / 3600 for tp in timepoints]

    def interpolate_metric(self, metric_df: pd.DataFrame, timepoint: float | None = None) -> pd.DataFrame:
        """Interpolate missing values in metric DataFrame.
        
        Args:
            metric_df: DataFrame with timepoints as index and metric values as data
            timepoint: Optional specific timepoint to interpolate for
            
        Returns:
            DataFrame with interpolated values
        """
        # Ensure numeric dtype for interpolation; coerce invalid to NaN
        metric_df = metric_df.apply(pd.to_numeric, errors="coerce")
        # Sort index to ensure timepoints are in chronological order
        metric_df = metric_df.sort_index()
        
        if timepoint is not None:
            # Get closest timepoint
            closest_time = metric_df.index[np.abs(np.array(metric_df.index) - timepoint).argmin()]
            
            # If value is NaN at closest timepoint, try to interpolate
            if pd.isna(metric_df.iloc[metric_df.index.get_loc(closest_time)]).any():
                # Find valid values before and after target timepoint
                valid_times = metric_df.dropna()
                
                if len(valid_times) >= 2:  # Need at least 2 points to interpolate
                    # Get points before and after
                    before_df = valid_times[valid_times.index <= timepoint]
                    after_df = valid_times[valid_times.index > timepoint]
                    
                    if not before_df.empty and not after_df.empty:
                        before_time = before_df.index[-1]
                        after_time = after_df.index[0]
                        before_values = before_df.iloc[-1]
                        after_values = after_df.iloc[0]
                        
                        # Linear interpolation
                        slope = (after_values - before_values) / (after_time - before_time)
                        interpolated = before_values + slope * (timepoint - before_time)
                        
                        # Create new DataFrame with interpolated values
                        return pd.DataFrame(interpolated).T
            
            # If interpolation not possible or not needed, return closest value
            return metric_df.iloc[[metric_df.index.get_loc(closest_time)]]
        else:
            # Interpolate only between existing points; keep NaN vor dem ersten
            # und nach dem letzten gültigen Wert bestehen.
            return metric_df.interpolate(
                method='linear',
                axis=0,
                limit_direction='both',
                limit_area='inside',
            )

    def metric(self, name: str = 'radius',
           timepoint: float | str | datetime | None = None,
           time_period: str | None = None,
           interpolate: bool = False,
           ignore_border: bool = False,
           plot: bool = False,
           skip_nan: bool = False,
           ax: Optional[plt.Axes] = None,
           plot_kwargs: Optional[dict] = None,
           savepath: Optional[str] = None) -> pd.DataFrame | tuple[pd.DataFrame, plt.Axes]:
        """Calculate a metric for this series and optionally plot it.
        
        Parameters
        ----------
        name : str, default='radius'
            Metric to calculate. Available options:
            - Basic metrics: 'radius', 'area'
            - Fluorescence metrics: 'fluorescence_green', 'fluorescence_red', 'fluorescence_blue'
            - Fluorescence with specific type: 'fluorescence_green_mean', 'fluorescence_green_cumulative', etc.
            
            Use 'radius' for effective radius (sqrt(Area/π)) in μm, 'area' for area in μm².
            For fluorescence, use base name (e.g., 'fluorescence_green') to get both cumulative and mean,
            or add '_mean'/'_cumulative' suffix to get only one type.
        
        timepoint : float | str | datetime | None, default=None
            Optional specific timepoint to evaluate. If None, returns values for all timepoints.
            Accepted formats: float (relative time in hours), str (datetime string 'YYYY-MM-DD HH:MM:SS'),
            or datetime object.
        
        time_period : str | None, default=None
            Optional time period name to limit analysis to specific time range.
            Must be a time period defined via ``time_period()`` method. If None, analyzes all available timepoints.
        
        interpolate : bool, default=True
            Whether to interpolate missing values. If True, missing values are linearly
            interpolated (or a single-row value is returned for a specific timepoint).
            If False, data is left as-is with NaN values.
        
        ignore_border : bool, default=True
            Whether to exclude spheroids touching image border when calculating metrics.
            If True, spheroids that touch the image border are excluded from all metric calculations
            (returns None/NaN). If False, metrics are calculated regardless of border contact.
            Recommended to keep True to avoid edge artifacts.
        
        plot : bool, default=False
            Whether to display a plot of the metric over time.
        
        skip_nan : bool, default=False
            Only affects plotting. If True, NaN points are not drawn (creates gaps in the curve).
            Data in the returned DataFrame remains unchanged (NaN values are still present).
        
        ax : Optional[plt.Axes], default=None
            Optional matplotlib Axes object to plot on. If provided and ``plot=True``,
            the plot will be drawn on this axes. If None and ``plot=True``, a new
            figure with default size (6, 4) is created.
        
        plot_kwargs : Optional[dict], default=None
            Optional dictionary of keyword arguments passed to ``ax.plot()`` for customizing
            line appearance. Merged with orange default style (color='#D97706',
            linestyle='-', linewidth=2). Keine Marker-Punkte, nur Linie.
        
        savepath : Optional[str], default=None
            Optional path to save the figure. Only used if ``plot=True``.
            Works with both internally created figures and externally provided axes.
            Can specify any file format supported by matplotlib (PNG, PDF, SVG, etc.).
        
        Returns
        -------
        pd.DataFrame
            DataFrame with metric values:
            - Index: Timepoints in relative hours (from first timepoint)
            - Columns: One column for this series (or MultiIndex with ('cumulative', 'mean') for
              fluorescence metrics without specific type)
            - Values: Metric values at each timepoint
            
            Special cases:
            - If ``timepoint`` is specified: Returns single-row DataFrame for that timepoint
            - If ``name='fluorescence_*'`` (without _mean/_cumulative): Returns MultiIndex columns
              with ('cumulative', 'mean') sub-columns
        
        Examples
        --------
        >>> # Calculate radius over time
        >>> df = series.metric('radius')
        
        >>> # Get radius at specific timepoint
        >>> df = series.metric('radius', timepoint=24.0)
        
        >>> # Calculate fluorescence with plot
        >>> df = series.metric('fluorescence_green_mean', plot=True)
        
        >>> # Use time period
        >>> df = series.metric('radius', time_period='growth_phase')
        """
        # 1) Zeitachsen (ggf. auf Perioden eingeschränkt)
        if time_period is not None:
            images_dict = self.get_images_in_period(time_period)
        else:
            images_dict = self.spheroid_image_dict

        timepoints = sorted(images_dict.keys())
        # Compute relative times based on the first timestamp in this subset
        def _ensure_dt(val):
            return val if isinstance(val, datetime) else datetime.strptime(val, '%Y-%m-%d %H:%M:%S')
        if timepoints:
            t0 = _ensure_dt(timepoints[0])
            rel_times = [(_ensure_dt(tp) - t0).total_seconds() / 3600 for tp in timepoints]
        else:
            rel_times = []
        series_id = self.name.replace('Spheroid-', '')

        # 2) Parse metric name to determine if it's fluorescence (for DataFrame structure)
        parts = name.split('_')
        is_fluo = (parts[0] == 'fluorescence')
        fluo_kind = parts[2] if is_fluo and len(parts) > 2 else None  # 'mean' | 'cumulative' | None

        # 3) Werte effizient sammeln - use metric() method from SpheroidImage
        # metric() already handles all the logic (mean/cumulative/both) and returns appropriate values
        if not is_fluo or fluo_kind is not None:
            # Simple metrics (radius, area) or fluorescence with specific kind (mean/cumulative)
            # metric() returns a single value
            values = []
            for t in timepoints:
                sph = images_dict[t]
                try:
                    val = sph.metric(name, ignore_border=ignore_border)
                    values.append(val)
                except ValueError:
                    # Re-raise ValueError (invalid metric name) - don't catch it
                    raise
                except Exception as e:
                    logger.warning(f"Failed to calculate metric '{name}' for timepoint {t}: {e}")
                    values.append(None)
            df = pd.DataFrame({series_id: values}, index=rel_times)
        else:
            # Fluorescence metrics without kind specified - metric() returns tuple (cumulative, mean)
            cum_vals, mean_vals = [], []
            for t in timepoints:
                sph = images_dict[t]
                try:
                    result = sph.metric(name, ignore_border=ignore_border)
                    # metric() returns tuple (cumulative, mean) for fluorescence without kind
                    if isinstance(result, tuple):
                        cum_v, mean_v = result
                    else:
                        # Fallback (shouldn't happen, but handle gracefully)
                        cum_v, mean_v = None, None
                except ValueError:
                    # Re-raise ValueError (invalid metric name) - don't catch it
                    raise
                except Exception as e:
                    logger.warning(f"Failed to calculate metric '{name}' for timepoint {t}: {e}")
                    cum_v, mean_v = None, None
                cum_vals.append(cum_v)
                mean_vals.append(mean_v)

            # Return both as MultiIndex DataFrame
            df = pd.DataFrame(
                { (series_id, 'cumulative'): cum_vals,
                  (series_id, 'mean'): mean_vals },
                index=rel_times
            )
            df.columns = pd.MultiIndex.from_tuples(df.columns)

        # 4) timepoint (optional) → ggf. in relative Stunden konvertieren
        if timepoint is not None:
            if isinstance(timepoint, str):
                timepoint = datetime.strptime(timepoint, '%Y-%m-%d %H:%M:%S')
            if isinstance(timepoint, datetime):
                base_t0 = timepoints[0] if isinstance(timepoints[0], datetime) else datetime.strptime(timepoints[0], '%Y-%m-%d %H:%M:%S')
                timepoint = (timepoint - base_t0).total_seconds() / 3600

        # Interpolation nur wenn nötig
        if interpolate:
            df = self.interpolate_metric(df, timepoint)
        elif timepoint is not None:
            # ohne Interpolation: wähle nächsten Punkt
            idx = np.abs(np.array(df.index) - float(timepoint)).argmin()
            df = df.iloc[[idx]]

        # 5) Plot (optional)
        if plot:
            # Axes-Handling analog zu radial_profile:
            fig = None
            if ax is None:
                fig, ax = plt.subplots(figsize=(6, 4))
            else:
                fig = ax.figure
            
            # Prepare plot_kwargs
            if plot_kwargs is None:
                plot_kwargs = {}
            
            # Default plot style: orange Linie, keine Marker
            default_kwargs = {
                'color': '#D97706',  # gleiche Orange wie im Histogram-Viewer
                'linestyle': '-',
                'linewidth': 2,
            }
            # Nutzer-Overrides zulassen
            default_kwargs.update(plot_kwargs)
            
            if isinstance(df.columns, pd.MultiIndex):
                # beide Fluoreszenz-Varianten (cumulative / mean)
                x = np.array(df.index) / 24
                for sub in df.columns.levels[1]:
                    y = df[(series_id, sub)].to_numpy(dtype=float)
                    if skip_nan:
                        m = ~np.isnan(y)
                        ax.plot(x[m], y[m], label=sub, **default_kwargs)
                    else:
                        ax.plot(x, y, label=sub, **default_kwargs)
                ax.legend()
            else:
                x = np.array(df.index) / 24
                y = df.iloc[:, 0].to_numpy(dtype=float)
                if skip_nan:
                    m = ~np.isnan(y)
                    ax.plot(x[m], y[m], **default_kwargs)
                else:
                    ax.plot(x, y, **default_kwargs)
            
            # Default formatting
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Time [d]')
            if self._result and name in getattr(self._result, 'plot_name_dict', {}):
                ax.set_ylabel(rf'{self._result.plot_name_dict[name]}')
            else:
                ax.set_ylabel(name)
            ax.set_title(f'{name} - {self.name}')
            
            # Save if requested
            if savepath is not None:
                if fig is None:
                    fig = ax.figure
                fig.savefig(savepath, bbox_inches='tight')
            
            # Show only if we created the figure ourselves (wie radial_profile)
            if fig is not None:
                plt.tight_layout()
                plt.show()

            return df, ax

        return df

    def growth_rate(self, metric_name: str = 'radius', time_period: str | None = None, model: str = 'linear') -> tuple[float, float]:
        """
        Calculate the growth rate (slope) of a metric over time using linear or logarithmic regression.
        
        Parameters
        ----------
        metric_name : str, default='radius'
            Name of the metric to analyze ('radius', 'area', or fluorescence metrics).
        time_period : str | None, default=None
            Optional time period name to limit analysis to specific time range.
            If None, uses all available timepoints.
        model : str, default='linear'
            Model to use for fitting. Options:
            - 'linear': Linear growth (y = a*t + b), returns slope in original units
            - 'log': Logarithmic/exponential growth (log(y) = a*t + b), returns growth rate in log space
        
        Returns
        -------
        tuple[float, float]
            Tuple containing (growth_rate, std_error) where:
            - growth_rate: The slope of the fit:
              * For 'linear': slope in original units (e.g., µm/day for radius)
              * For 'log': coefficient in log space (change in log(metric) per day)
            - std_error: Standard error of the slope estimate
        
        Raises
        ------
        ValueError
            If insufficient data points are available (< 2 points), if the metric is not supported,
            or if model is not 'linear' or 'log'.
        KeyError
            If the specified time_period does not exist.
        
        Notes
        -----
        - For radius with 'linear': growth_rate is in µm/day
        - For area with 'linear': growth_rate is in µm²/day
        - For 'log': growth_rate represents the rate of change in log space (dimensionless per day)
        - NaN values are automatically excluded from the fit
        - Uses scipy.optimize.curve_fit for regression
        - For 'log' model, values must be positive (values <= 0 are excluded)
        """
        if model not in ['linear', 'log']:
            raise ValueError(f"Model '{model}' not supported. Options: 'linear', 'log'.")
        
        # Get metric data over time
        metric_df = self.metric(name=metric_name, time_period=time_period, interpolate=False)
        
        if metric_df.empty:
            raise ValueError(f"No data available for metric '{metric_name}' in the specified time period.")
        
        # Extract time (in days) and values
        # metric_df index is in hours, convert to days
        times_days = np.array(metric_df.index) / 24.0
        
        # Handle MultiIndex columns (for fluorescence without kind specified)
        if isinstance(metric_df.columns, pd.MultiIndex):
            # Use 'mean' if available, otherwise 'cumulative'
            if ('mean',) in metric_df.columns.levels[1]:
                values = metric_df.xs('mean', level=1, axis=1).iloc[:, 0].values
            else:
                values = metric_df.xs('cumulative', level=1, axis=1).iloc[:, 0].values
        else:
            values = metric_df.iloc[:, 0].values
        
        # Filter out NaN values
        valid_mask = ~np.isnan(values) & ~np.isnan(times_days)
        
        # For log model, also filter out non-positive values
        if model == 'log':
            valid_mask = valid_mask & (values > 0)
        
        valid_times = times_days[valid_mask]
        valid_values = values[valid_mask]
        
        if len(valid_times) < 2:
            raise ValueError(f"Insufficient data points for fitting: {len(valid_times)} valid points (need at least 2).")
        
        from scipy.optimize import curve_fit
        
        if model == 'log':
            # Logarithmic growth: fit log(values) vs time
            # Returns growth rate as coefficient in log space
            def log_func(t, a, b):
                return a * t + b
            
            # Fit log(values) = a*t + b
            log_values = np.log(valid_values)
            try:
                popt, pcov = curve_fit(log_func, valid_times, log_values)
                growth_rate = popt[0]  # coefficient in log space
                std_error = np.sqrt(pcov[0, 0])
            except Exception as e:
                raise ValueError(f"Failed to fit logarithmic model: {e}")
        else:  # model == 'linear'
            # Linear fit: y = a * t + b
            def linear_func(t, a, b):
                return a * t + b
            
            try:
                popt, pcov = curve_fit(linear_func, valid_times, valid_values)
                growth_rate = popt[0]  # slope
                std_error = np.sqrt(pcov[0, 0])  # standard error of slope
            except Exception as e:
                raise ValueError(f"Failed to fit linear model: {e}")
        
        return (float(growth_rate), float(std_error))

    def intensity(self,
                  channel: str = 'brightfield',
                  mean: float | None = None,
                  value_range: tuple[float, float] | None = None,
                  reset: bool = False,
                  timepoint: int | str | datetime | None = None,
                  show: bool = True) -> np.ndarray:
        """
        Configure an intensity window for a given channel across the series.

        The interactive viewer is shown for a single reference timepoint
        (by default the last available), and the resulting window is then
        applied to all SpheroidImage items in this series for consistent
        visualization (e.g. in :meth:`show` or :class:`SpheroidImage.show`).

        Parameters
        ----------
        channel : str, default='brightfield'
            Channel to adjust ('brightfield', 'green', 'red', 'blue',
            or full keys like 'fluorescence_green').
        mean : float | None
            Optional initial mean intensity in [0,255].
        value_range : (float, float) | None
            Optional initial [low, high] window in [0,255].
        reset : bool, default=False
            If True, ignore any previously stored windows and recompute
            defaults from the reference image (unless mean/range are given).
        timepoint : int | str | datetime | None, default=None
            Reference timepoint for interactive adjustment. If None, uses
            the last available timepoint. If int (hours) or str, the closest
            real timepoint is used.
        show : bool, default=True
            If True, open the interactive histogram viewer for the reference
            image. If False, only set/update the window programmatically.

        Returns
        -------
        np.ndarray
            Display-ready image of the reference SpheroidImage after
            applying the configured window.
        """
        if not self.spheroid_image_dict:
            raise ValueError("No spheroid images available in series.")

        # Helper for timepoint resolution
        def _ensure_dt(val):
            if isinstance(val, datetime):
                return val
            if isinstance(val, str):
                return datetime.strptime(val, '%Y-%m-%d %H:%M:%S')
            if isinstance(val, (int, float)):
                sorted_times = sorted(self.spheroid_image_dict.keys())
                if not sorted_times:
                    raise ValueError("No timepoints available in series")
                t0 = sorted_times[0] if isinstance(sorted_times[0], datetime) else datetime.strptime(sorted_times[0], '%Y-%m-%d %H:%M:%S')
                return t0 + pd.Timedelta(hours=float(val))
            return val

        # Choose reference timepoint
        sorted_times = sorted(self.spheroid_image_dict.keys())
        if timepoint is None:
            ref_tp = sorted_times[-1]
        else:
            target_dt = _ensure_dt(timepoint)
            ref_tp = min(sorted_times,
                         key=lambda x: abs((_ensure_dt(x) - target_dt).total_seconds()))

        ref_img: SpheroidImage = self.spheroid_image_dict[ref_tp]

        # Run intensity configuration on reference image
        disp_img = ref_img.intensity(
            channel=channel,
            mean=mean,
            value_range=value_range,
            reset=reset,
            show=show,
        )

        # Propagate resulting window to all images in the series
        key = ref_img._get_channel_key(channel)
        cfg = ref_img.intensity_windows.get(key)
        if cfg is not None:
            for sph in self.spheroid_image_dict.values():
                sph.intensity_windows[key] = dict(cfg)  # shallow copy

        return disp_img

    def radial_profile(self, timepoints: list[int | str | datetime] | None = None, 
                       channels: str | list[str] = ['green'], 
                       return_absolute: bool = False,
                       normalize: bool = True,
                       smoothing: float = 2,
                       plot: bool = True,
                       savepath: str | None = None,
                       visualise_change: bool = False,
                       ax: Optional[plt.Axes] = None) -> dict | tuple[dict, plt.Axes]:
        """
        Calculate and plot radial profiles for multiple timepoints.
        
        Parameters
        ----------
        timepoints : list[int | str | datetime] | None, default=None
            List of timepoints to analyze. If None, uses all available timepoints.
            Can be:
            - int: Relative time in hours
            - str: Datetime string in format 'YYYY-MM-DD HH:MM:SS'
            - datetime: Datetime object
        channels : str | list[str], default=['green']
            Fluorescence channel(s) to analyze. Can be a single channel string ('green', 'red', 'blue')
            or a list of channels.
        return_absolute : bool, default=False
            If True, uses absolute distances in μm. If False, uses normalized distances (0-1).
            Note: For absolute distances, each timepoint may have different x-axis ranges.
        normalize : bool, default=True
            Whether to normalize intensity profiles to [0,1]
        smoothing : float, default=2
            Gaussian smoothing parameter (sigma) for radial profile
        plot : bool, default=True
            Whether to display a plot.
        savepath : str | None, default=None
            Optional path to save the plot (only used if ``plot=True``).
        visualise_change : bool, default=False
            If True, compute the relative change in intensity per day for each radial
            position (basierend auf paarweisen Differenzen zwischen aufeinanderfolgenden
            Zeitpunkten für den ersten Kanal in ``channels``) und zeige diese als
            RdYlGn-Colormap-Hintergrund an, mit einer Colorbar "Change in Intensity [%/d]".
        ax : matplotlib.axes.Axes | None, default=None
            Optional matplotlib Axes object to plot on. If provided and ``plot=True``,
            the plot is drawn onto this axes and *no* ``plt.show()`` is called
            (analog zum Verhalten von :meth:`metric`). Wird kein ``ax`` übergeben,
            wird eine neue Figure/Axes erzeugt und nach dem Plot angezeigt.
        visualise_change : bool, default=False
            If True, compute the relative change in intensity per day for each radial
            position (based on linear regression over time for the first channel in
            ``channels``) and display it as a RdYlGn colormap background behind the
            profiles, with a colorbar labelled "Change in Intensity [%/d]".
        
        Returns
        -------
        dict
            Dictionary containing:
            - timepoints: List of processed timepoint identifiers
            - profiles: Dictionary mapping timepoint -> channel -> profile data
            - distances: Dictionary mapping timepoint -> distance array (relative or absolute)
        """
        from SpheroidPy.utils.color_palettes import CARTO_SEQUENTIAL
        
        # Normalize channels to list
        if isinstance(channels, str):
            channels = [channels]
        
        # Helper function to convert timepoint to datetime
        def _ensure_dt(val):
            if isinstance(val, datetime):
                return val
            elif isinstance(val, str):
                return datetime.strptime(val, '%Y-%m-%d %H:%M:%S')
            elif isinstance(val, (int, float)):
                # Convert relative hours to datetime
                sorted_times = sorted(self.spheroid_image_dict.keys())
                if not sorted_times:
                    raise ValueError("No timepoints available in series")
                t0 = sorted_times[0] if isinstance(sorted_times[0], datetime) else datetime.strptime(sorted_times[0], '%Y-%m-%d %H:%M:%S')
                return t0 + pd.Timedelta(hours=float(val))
            else:
                raise ValueError(f"Cannot convert {type(val)} to datetime")
        
        # Process timepoints and find matching images
        processed_timepoints = []
        timepoint_images = {}
        
        # If timepoints is None, use all available timepoints
        if timepoints is None:
            timepoints = sorted(self.spheroid_image_dict.keys())
            # Convert to list if needed
            if not isinstance(timepoints, list):
                timepoints = list(timepoints)
        
        for tp in timepoints:
            tp_dt = _ensure_dt(tp)
            # Find closest timepoint in series
            sorted_times = sorted(self.spheroid_image_dict.keys())
            closest_tp = min(sorted_times, 
                           key=lambda x: abs((_ensure_dt(x) - tp_dt).total_seconds()))
            
            # Check if close enough (within 1 hour)
            time_diff = abs((_ensure_dt(closest_tp) - tp_dt).total_seconds() / 3600)
            if time_diff > 1.0:
                logger.warning(f"Timepoint {tp} is more than 1 hour away from closest match {closest_tp}. Skipping.")
                continue
            
            if closest_tp not in timepoint_images:
                processed_timepoints.append(closest_tp)
                timepoint_images[closest_tp] = self.spheroid_image_dict[closest_tp]
        
        if not processed_timepoints:
            raise ValueError("No valid timepoints found")
        
        # Sort timepoints chronologically
        processed_timepoints.sort(key=lambda x: _ensure_dt(x))
        
        # Calculate profiles for each timepoint
        profiles_data = {}
        distances_data = {}
        
        for tp in processed_timepoints:
            img = timepoint_images[tp]
            try:
                result = img.radial_profile(
                    channels=channels,
                    plot=False,
                    normalize=normalize,
                    return_absolute=return_absolute,
                    smoothing=smoothing
                )
                profiles_data[tp] = result['intensity_profiles']
                if return_absolute and 'absolute_distances' in result:
                    distances_data[tp] = result['absolute_distances']
                else:
                    distances_data[tp] = result['relative_distances']
            except Exception as e:
                logger.warning(f"Failed to calculate radial profile for timepoint {tp}: {e}")
                continue
        
        # Plotting
        if plot:
            # Axes-Handling ähnlich wie in metric():
            # - Wenn ax=None: neue Figure/Axes erstellen und am Ende plt.show() aufrufen.
            # - Wenn ax vorhanden: darauf zeichnen, keine neue Figure anzeigen.
            fig: Optional[plt.Figure] = None
            if ax is None:
                fig, ax = plt.subplots(figsize=(8, 6))
            else:
                fig = ax.figure
            
            # Determine global x-range from distances (used for background and limits)
            all_distances = []
            for dist in distances_data.values():
                if dist is not None and len(dist) > 0:
                    all_distances.extend(dist)
            if all_distances:
                all_distances = np.array(all_distances, dtype=float)
                xmin, xmax = float(all_distances.min()), float(all_distances.max())
            else:
                xmin = xmax = None
            
            # Color palette mapping for channels (using channel-similar palettes)
            # Use Green for green, Red/Orange for red, Teal for blue
            channel_palettes = {
                'green': CARTO_SEQUENTIAL.get('Green', ['#00441b', '#006d2c', '#238b45', '#41ab5d', '#74c476', '#a1d99b', '#c7e9c0']),
                'red': CARTO_SEQUENTIAL.get('Red', ['#7f0000', '#b31b1b', '#d94801', '#f16913', '#fd8d3c', '#fdae6b', '#fee6ce']),
                'blue': CARTO_SEQUENTIAL.get('Teal', ['#004c4c', '#006d6d', '#238b8b', '#41a9a9', '#74c8c8', '#a1e3e3', '#c8f5f5']),
            }
            
            # Generate colors for timepoints (darker for later timepoints)
            n_timepoints = len(processed_timepoints)
            
            # Optional Hintergrund: paarweiser Änderungsrate (%/d) zwischen aufeinanderfolgenden
            # Profilen des ersten Kanals; die Color-Map wird nur zwischen den jeweiligen
            # Kurvensegmenten gezeichnet.
            if visualise_change and n_timepoints >= 2 and len(channels) > 0 and xmin is not None:
                base_channel = channels[0]
                # Sammle alle Änderungsraten, um symmetrische Normierung um 0 zu bekommen
                all_rates = []
                interval_data = []
                t_days = [(_ensure_dt(tp) - _ensure_dt(processed_timepoints[0])).total_seconds() / 86400.0
                          for tp in processed_timepoints]
                
                for k in range(n_timepoints - 1):
                    tp0 = processed_timepoints[k]
                    tp1 = processed_timepoints[k + 1]
                    ch_profiles0 = profiles_data.get(tp0, {})
                    ch_profiles1 = profiles_data.get(tp1, {})
                    if base_channel not in ch_profiles0 or base_channel not in ch_profiles1:
                        continue
                    prof0 = ch_profiles0[base_channel]
                    prof1 = ch_profiles1[base_channel]
                    y0 = np.asarray(prof0.get('mean'), dtype=float)
                    y1 = np.asarray(prof1.get('mean'), dtype=float)
                    # Distanzachsen der früheren und späteren Kurve für dieses Intervall
                    dist0 = np.asarray(distances_data.get(tp0), dtype=float)
                    dist1 = np.asarray(distances_data.get(tp1), dtype=float)
                    if dist0.size == 0 or dist1.size == 0:
                        continue
                    n_r = dist0.size
                    # Falls Längen nicht übereinstimmen, überspringen
                    if y0.size != n_r or y1.size != dist1.size:
                        continue
                    # Interpolation der späteren Kurve auf die Radialpositionen der früheren Kurve
                    # für die Darstellung (Berechnung der Rate bleibt bin-basiert auf y0/y1).
                    try:
                        y1_plot = np.interp(dist0, dist1, y1)
                    except Exception:
                        y1_plot = y1.copy()
                    # x-Kanten für kleine Rechtecke zwischen den Kurven (intervallspezifisch)
                    x_edges = np.empty(n_r + 1, dtype=float)
                    x_edges[1:-1] = 0.5 * (dist0[:-1] + dist0[1:])
                    x_edges[0] = dist0[0] - (x_edges[1] - dist0[0])
                    x_edges[-1] = dist0[-1] + (dist0[-1] - x_edges[-2])
                    
                    dt_days = t_days[k + 1] - t_days[k]
                    if dt_days == 0:
                        continue
                    # Prozentuale Änderungsrate pro Tag relativ zur Baseline (frühere Kurve)
                    rate = np.full(n_r, np.nan, dtype=float)
                    baseline = y0.copy()
                    valid = np.isfinite(baseline) & np.isfinite(y1) & (baseline > 0)
                    rate[valid] = ((y1[valid] - baseline[valid]) / baseline[valid]) / dt_days * 100.0
                    all_rates.append(rate[valid])
                    interval_data.append({'y0': y0, 'y1_plot': y1_plot, 'rate': rate, 'x_edges': x_edges})
                
                if interval_data and all_rates:
                    all_rates_flat = np.concatenate(all_rates)
                    v_abs = float(np.nanmax(np.abs(all_rates_flat)))
                    if v_abs == 0.0:
                        v_abs = 1.0
                    # Behalte die volle Normierung (-v_abs ... +v_abs), aber verwende nur
                    # den mittleren Teil der Colormap (keine extremen Endfarben).
                    norm = TwoSlopeNorm(vmin=-v_abs, vcenter=0.0, vmax=v_abs)
                    base_cmap = plt.get_cmap('RdYlGn')
                    # z.B. nur die mittleren 50 % der Colormap (0.25 .. 0.75) verwenden
                    cmap = LinearSegmentedColormap.from_list(
                        'RdYlGn_mid',
                        base_cmap(np.linspace(0.15, 0.85, 256))
                    )
                    
                    # y-Bereich aus allen Profilen für sinnvolle Achsenskala
                    all_means = []
                    all_stds = []
                    for tp in processed_timepoints:
                        ch_profiles = profiles_data.get(tp, {})
                        if base_channel not in ch_profiles:
                            continue
                        prof = ch_profiles[base_channel]
                        m = np.asarray(prof.get('mean'), dtype=float)
                        s = np.asarray(prof.get('std', np.zeros_like(m)), dtype=float)
                        all_means.append(m)
                        all_stds.append(s)
                    if all_means:
                        all_means_arr = np.concatenate(all_means)
                        all_stds_arr = np.concatenate(all_stds)
                        ymin = float(np.nanmin(all_means_arr - all_stds_arr))
                        ymax = float(np.nanmax(all_means_arr + all_stds_arr))
                        if not np.isfinite(ymin) or not np.isfinite(ymax) or ymin == ymax:
                            ymin, ymax = 0.0, 1.0
                        pad = 0.05 * (ymax - ymin)
                        ymin -= pad
                        ymax += pad
                    else:
                        ymin, ymax = 0.0, 1.0
                    
                    for data in interval_data:
                        y0 = data['y0']
                        y1_plot = data['y1_plot']
                        rate = data['rate']
                        x_edges = data['x_edges']
                        n_r_local = len(rate)
                        for j in range(n_r_local):
                            r_val = rate[j]
                            if not np.isfinite(r_val):
                                continue
                            y_low = float(min(y0[j], y1_plot[j]))
                            y_high = float(max(y0[j], y1_plot[j]))
                            if not np.isfinite(y_low) or not np.isfinite(y_high) or y_low == y_high:
                                continue
                            color = cmap(norm(r_val))
                            ax.fill_between(
                                [x_edges[j], x_edges[j + 1]],
                                [y_low, y_low],
                                [y_high, y_high],
                                color=color,
                                alpha=0.15,
                                linewidth=0,
                                zorder=0,
                            )
                    
                    # Colorbar rechts neben dem Plot
                    from matplotlib.cm import ScalarMappable
                    sm = ScalarMappable(norm=norm, cmap=cmap)
                    sm.set_array([])
                    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
                    cbar.set_label("Change in Intensity [%/d]")
                    
                    ax.set_ylim(ymin, ymax)
            
            # Plot radial profiles for each channel/timepoint
            for ch_idx, channel in enumerate(channels):
                if channel not in channel_palettes:
                    # Fallback to gray palette
                    palette = ['#333333', '#555555', '#777777', '#999999', '#bbbbbb', '#dddddd']
                else:
                    palette = channel_palettes[channel]
                
                # Use palette centered around middle, darker for later timepoints
                n_colors = len(palette)
                mid_idx = n_colors // 2
                
                for tp_idx, tp in enumerate(processed_timepoints):
                    if tp not in profiles_data or channel not in profiles_data[tp]:
                        continue
                    
                    profile = profiles_data[tp][channel]
                    distances = distances_data[tp]
                    
                    # Select color: later timepoints get darker colors
                    # Map tp_idx to palette index (0 = earliest, darker for later)
                    if n_timepoints == 1:
                        color_idx = mid_idx
                    else:
                        # Map [0, n_timepoints-1] to palette indices
                        # Earlier timepoints use lighter colors (higher indices)
                        # Later timepoints use darker colors (lower indices)
                        normalized_pos = tp_idx / (n_timepoints - 1) if n_timepoints > 1 else 0.5
                        # Reverse: 0 -> high index (light), 1 -> low index (dark)
                        color_idx = int((1 - normalized_pos) * (n_colors - 1))
                        color_idx = max(0, min(n_colors - 1, color_idx))
                    
                    color = palette[color_idx]
                    
                    # Format timepoint label
                    tp_dt = _ensure_dt(tp)
                    t0_dt = _ensure_dt(processed_timepoints[0])
                    rel_hours = (tp_dt - t0_dt).total_seconds() / 3600
                    label = f"{channel} @ {rel_hours:.1f}h"
                    
                    mean = profile['mean']
                    std = profile.get('std', np.zeros_like(mean))
                    
                    # Es gibt nur eine y-Achse: immer auf ax plotten
                    ax.plot(distances, mean, color=color, label=label, linewidth=2)
                    ax.fill_between(distances, mean - std, mean + std, 
                                    color=color, alpha=0.2)
            
            # Set labels
            if return_absolute:
                ax.set_xlabel('Radial distance [µm]')
            else:
                ax.set_xlabel('Normalized distance (ρ)')
            
            if normalize:
                ax.set_ylabel('Normalized intensity')
            else:
                ax.set_ylabel('Intensity (a.u.)')
            
            ax.set_title(f'Radial Profiles Over Time - {self.name}', fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            ax.legend(loc='best', fontsize=9)
            
            # Set xlims: crop to min/max of all timepoints
            if xmin is not None and xmax is not None:
                ax.set_xlim(xmin, xmax)
            
            if fig is not None:
                fig.tight_layout()
                if savepath is not None:
                    fig.savefig(savepath, transparent=True, dpi=300, bbox_inches='tight')
                # Nur anzeigen, wenn wir die Figure selbst erzeugt haben
                if ax is not None and ax.figure is fig:
                    plt.show()
            else:
                # ax wurde extern übergeben – dennoch speichern, falls gewünscht
                if savepath is not None and ax is not None:
                    ax.figure.savefig(savepath, transparent=True, dpi=300, bbox_inches='tight')
        
        result = {
            'timepoints': processed_timepoints,
            'profiles': profiles_data,
            'distances': distances_data,
        }
        # Optionales zweites Rückgabe-Objekt: Plot-Handle (ax),
        # aber nur, wenn tatsächlich geplottet wurde (plot=True).
        if plot:
            return result, ax
        # plot=False → reine Datenrückgabe wie früher
        return result

    def __repr__(self):
        return self.name + (f'\n> Number of Timepoints: {len(self.spheroid_image_dict.keys())}'
                            f'\n> Channels: {list(self.spheroid_image_dict.values())[4].image_path_dict.keys()}\n\n')

    def show(self):
        """Interaktive Visualisierung aller Bilder der Serie mit modernem Layout und Boxen um die Container."""
        import os
        if not self.spheroid_image_dict:
            print("No images in series.")
            return

        # Zeitpunkte und Kanäle
        timepoints = sorted(self.spheroid_image_dict.keys())
        channels = ['brightfield', 'fluorescence_green', 'fluorescence_red', 'fluorescence_blue']
        channel_dropdown = widgets.Dropdown(options=channels, value='brightfield', description='Channel:')
        show_contour = widgets.Checkbox(value=True, description='Show Contour')
        time_slider = widgets.IntSlider(min=0, max=len(timepoints)-1, value=0, description='Time:', continuous_update=True)
        time_label = widgets.Label()

        def get_base64_image_html(image_path, height=60):
            try:
                with open(image_path, 'rb') as f:
                    data = base64.b64encode(f.read()).decode('utf-8')
                return f"<img src='data:image/png;base64,{data}' style='height:{height}px; float:right;'>"
            except Exception as e:
                print(f"Logo konnte nicht geladen werden: {e}")
                return ""

        # --- Kopfzeile mit Name, Untertitel, Slider (zentriert) und Logo ---
        logo_path = '/Users/cedric/Documents/Labor/SpheroidPy/SpheroidPy/static/images/logo_full_horizontal.png'
        if os.path.exists(logo_path):
            logo_html = get_base64_image_html(logo_path)
        else:
            logo_html = ""
        name_html = widgets.HTML(f"<h2 style='margin:0'>{self.name}</h2>")
        subtitle_html = widgets.HTML("<div style='color:#888; font-weight:400; font-size:1.1em; margin-bottom:2px;'>SpheroidSeries Visualisation</div>")
        name_block = widgets.VBox([name_html, subtitle_html], layout=widgets.Layout(margin='0px', padding='0px'))
        slider_box = widgets.HBox([time_slider, time_label], layout=widgets.Layout(justify_content='center', align_items='center'))
        header_row = widgets.HBox([
            name_block,
            slider_box,
            widgets.HTML(logo_html)
        ], layout=widgets.Layout(justify_content='space-between', align_items='center', width='100%'))
        header_container = widgets.VBox([header_row], layout=widgets.Layout(
            border='none', border_radius='10px', padding='0px', margin='0px', width='100%'))

        # Titles above image and plot (outside the figure) – bold, minimal gap to content
        image_title = widgets.HTML(
            value="",
            layout=widgets.Layout(padding="0 8px 2px", margin="0px 0px -12px 0px", width="100%", text_align="center")
        )
        plot_title = widgets.HTML(
            value="",
            layout=widgets.Layout(padding="0 8px 2px", margin="0px 0px -12px 0px", flex="1 1 auto", min_width="0", overflow="hidden")
        )
        metric_dropdown = widgets.Dropdown(options=['radius', 'area', 'profile'], value='radius', description='Metric:', layout=widgets.Layout(flex='0 0 auto', width='150px', height='18px'), style={'description_width':'60px','font_size':'5px'})

        plot_header_row = widgets.HBox(
            [plot_title, metric_dropdown],
            layout=widgets.Layout(width='100%', min_width='0', align_items='center', overflow='hidden', justify_content='space-between', padding='0 30px 0px 8px')
        )
        
        # --- Bildanzeige ---
        image_output = widgets.Output()
        # Image controls (channel + contour) centered, tight spacing, no horizontal scroll
        image_controls = widgets.HBox(
            [channel_dropdown, show_contour],
            layout=widgets.Layout(
                justify_content="center",
                align_items="center",
                width="100%",
                gap="0px",
                padding="6px 0px",
                overflow="hidden",
                min_width="0",
            ),
        )
        image_output.layout.width = '100%'
        def update_image(*args):
            idx = time_slider.value
            channel = channel_dropdown.value
            update_time_label(idx)
            if idx >= len(timepoints):
                image_title.value = ""
                return
            timepoint = timepoints[idx]
            image_title.value = f"<div style='text-align: center;'><span style='font-size: 12px; font-weight: 600; color: #333;'>{timepoint} | {channel}</span></div>"
            sph_img = self.spheroid_image_dict[timepoint]
            try:
                # Verwende die intensity-Fensterung, falls für diesen Kanal gesetzt
                if channel == 'brightfield':
                    img = sph_img.display_channel('brightfield')
                elif channel.startswith('fluorescence_'):
                    color = channel.split('_')[1]
                    img = sph_img.display_channel(f'fluorescence_{color}')
                    if img is not None and img.ndim == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                else:
                    img = sph_img.display_channel('brightfield')
            except Exception as e:
                image_output.clear_output(wait=True)
                with image_output:
                    print(f"Error loading image: {e}")
                return
            image_output.clear_output(wait=True)
            with image_output:
                # Dynamische Breite/Höhe für das Bild
                container_width = 600  # px, kann ggf. dynamisch bestimmt werden
                aspect = sph_img.image_size[1] / sph_img.image_size[0] if sph_img.image_size[0] else 1
                fig, ax = plt.subplots(figsize=(8, 8 * aspect))
                if img.ndim == 2:
                    ax.imshow(img, extent=[0, sph_img.image_size[0], 0, sph_img.image_size[1]], cmap='gray')
                else:
                    ax.imshow(img, extent=[0, sph_img.image_size[0], 0, sph_img.image_size[1]])
                if show_contour.value and sph_img.contour is not None:
                    try:
                        contour = sph_img.scaled_contour
                        # visually close by linking last to first
                        xs = contour[:,0]
                        ys = contour[:,1]
                        if xs[0] != xs[-1] or ys[0] != ys[-1]:
                            xs = np.r_[xs, xs[:1]]
                            ys = np.r_[ys, ys[:1]]
                        ax.plot(xs, ys, 'w-', linewidth=2)
                    except Exception as e:
                        print(f"Error drawing contour: {e}")
                
                # 250 µm scalebar (white line + label below, bottom-right corner with spacing)
                w, h = sph_img.image_size[0], sph_img.image_size[1]
                bar_len = 250
                margin_x, margin_y = 0.05 * w, 0.05 * h
                x_left = w - margin_x - bar_len
                x_right = w - margin_x
                y_bar = margin_y
                ax.plot([x_left, x_right], [y_bar, y_bar], "w-", linewidth=2.5, solid_capstyle="butt")
                ax.text((x_left + x_right) / 2, y_bar - 0.015 * h, "250 µm", color="white", fontsize=10, ha="center", va="top", family="sans-serif")
                
                ax.axis('off')
                plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
                plt.show()
        image_panel = widgets.VBox([image_title, image_output, image_controls], layout=widgets.Layout(
            border='none', border_radius='10px', padding='0px', margin='0px', flex='1 1 0%', width='100%', overflow='hidden'))

        def _channel_to_fluorescence(ch_desc):
            """Map channel dropdown value to fluorescence channel name for radial_profile."""
            if ch_desc == 'fluorescence_green':
                return 'green'
            if ch_desc == 'fluorescence_red':
                return 'red'
            if ch_desc == 'fluorescence_blue':
                return 'blue'
            return 'green'  # fallback for brightfield (profile needs fluorescence)

        # --- Beispielplot (Radius über Zeit) ---
        plot_output = widgets.Output()
        def update_plot(*args):
            metric = metric_dropdown.value
            if metric == "radius":
                metric_label = "Radius"
            elif metric == "area":
                metric_label = "Area"
            else:
                metric_label = "Radial profile"
            over_time = " over time" if metric in ("radius", "area") else ""
            plot_title.value = f"<div style='text-align: center;'><span style='font-size: 14px; font-weight: 600; color: #333;'>{metric_label}{over_time} – {self.name}</span></div>"
            
            plot_output.clear_output(wait=True)
            with plot_output:
                try:
                    if metric == "profile":
                        # Normalized radial profile for current timepoint
                        idx = time_slider.value
                        if idx >= len(timepoints):
                            return
                        timepoint = timepoints[idx]
                        image = self.spheroid_image_dict.get(timepoint)
                        if image is None or image.contour is None or len(image.contour) < 3:
                            plt.figure(figsize=(6, 4.1))
                            plt.text(0.5, 0.5, 'No contour for radial profile.\nSegment the image first.', ha='center', va='center')
                            plt.axis('off')
                            plt.show()
                        else:
                            # Determine all available fluorescence channels for this image
                            available_channels = []
                            try:
                                img_channels = getattr(image, "image_path_dict", {}) or {}
                                for ch in ["green", "red", "blue"]:
                                    if f"fluorescence_{ch}" in img_channels:
                                        available_channels.append(ch)
                            except Exception:
                                available_channels = []

                            if not available_channels:
                                plt.figure(figsize=(6, 4.1))
                                plt.text(
                                    0.5,
                                    0.5,
                                    "No fluorescence channels available\nfor radial profile.",
                                    ha="center",
                                    va="center",
                                )
                                plt.axis("off")
                                plt.show()
                            else:
                                # Compute normalized intensity profiles for all available channels
                                result = image.radial_profile(
                                    channels=available_channels,
                                    plot=False,
                                    normalize=True,
                                    return_absolute=True,
                                )

                                # Absolute radial distance in µm for x-axis
                                r_um = result.get("absolute_distances", None)
                                if r_um is None:
                                    # Fallback to normalized distances if absolute not available
                                    r_um = result.get("relative_distances", None)
                                    xlabel = "Normalized distance (ρ)"
                                else:
                                    xlabel = "Radial distance [µm]"

                                intensity_profiles = result.get("intensity_profiles", {})

                                if not intensity_profiles or r_um is None:
                                    plt.figure(figsize=(6, 4.1))
                                    plt.text(
                                        0.5,
                                        0.5,
                                        "No radial profile data available.",
                                        ha="center",
                                        va="center",
                                    )
                                    plt.axis("off")
                                    plt.show()
                                else:
                                    fig, ax = plt.subplots(figsize=(6, 4.1))

                                    color_map = {
                                        "green": "green",
                                        "red": "red",
                                        "blue": "blue",
                                    }

                                    for ch in available_channels:
                                        prof = intensity_profiles.get(ch)
                                        if not prof:
                                            continue
                                        y_mean = prof.get("mean")
                                        y_std = prof.get("std")
                                        if y_mean is None or y_std is None:
                                            continue

                                        col = color_map.get(ch, "grey")
                                        ax.plot(r_um, y_mean, color=col, label=ch)
                                        ax.fill_between(
                                            r_um,
                                            y_mean - y_std,
                                            y_mean + y_std,
                                            color=col,
                                            alpha=0.2,
                                        )

                                    ax.set_xlabel(xlabel)
                                    ax.set_ylabel("Normalized intensity")
                                    # Set xlims: start at min value (not 0) to max value
                                    if r_um is not None and len(r_um) > 0:
                                        ax.set_xlim(r_um.min(), r_um.max())
                                    ax.grid(True, alpha=0.3)
                                    ax.legend()
                                    plt.tight_layout()
                                    plt.show()
                    else:
                        def _ensure_dt(val):
                            return val if isinstance(val, datetime) else datetime.strptime(val, '%Y-%m-%d %H:%M:%S')
                        base_t0 = _ensure_dt(timepoints[0])
                        times = [ (_ensure_dt(tp) - base_t0).total_seconds()/24/3600 for tp in timepoints ]
                        
                        if metric == "area":
                            values = [self.spheroid_image_dict[tp].area for tp in timepoints]
                            ylabel = "Area [µm²]"
                        else:
                            values = [self.spheroid_image_dict[tp].radius for tp in timepoints]
                            ylabel = "Radius [µm]"
                        values = [v if v is not None else np.nan for v in values]
                        idx = time_slider.value
                        current_time = times[idx] if idx < len(times) else 0
                        current_val = values[idx] if idx < len(values) else None
                        
                        fig, ax = plt.subplots(figsize=(6, 4.1))
                        
                        # Plot data first
                        ax.plot(times, values, color='grey', zorder=5)
                        if current_val is not None and not np.isnan(current_val):
                            ax.plot(current_time, current_val, '.', markersize=10,
                                    label=f"{current_val:.1f} {'µm' if metric == 'radius' else 'µm²'} @ {current_time:.2f} d", color="#e29266", zorder=6)
                        ax.set_xlabel("Time [d]")
                        ax.set_ylabel(ylabel)
                        ax.grid(True, alpha=0.3)
                        if current_val is not None and not np.isnan(current_val):
                            ax.legend()
                        
                        # Add time periods as gray transparent background regions (after plot to get correct y-limits)
                        if hasattr(self, 'time_periods') and self.time_periods:
                            # Get actual y-axis limits after plotting
                            y_min, y_max = ax.get_ylim()
                            y_range = y_max - y_min
                            
                            for period_name, period in self.time_periods.items():
                                if period.start_time is None or period.end_time is None:
                                    continue
                                
                                # Convert period times to days relative to base_t0
                                period_start_dt = period.start_time if isinstance(period.start_time, datetime) else datetime.strptime(str(period.start_time), '%Y-%m-%d %H:%M:%S')
                                period_end_dt = period.end_time if isinstance(period.end_time, datetime) else datetime.strptime(str(period.end_time), '%Y-%m-%d %H:%M:%S')
                                
                                period_start_days = (period_start_dt - base_t0).total_seconds() / 24 / 3600
                                period_end_days = (period_end_dt - base_t0).total_seconds() / 24 / 3600
                                
                                # Only show if period overlaps with data range
                                if period_end_days >= times[0] and period_start_days <= times[-1]:
                                    # Draw gray transparent background (behind plot)
                                    ax.axvspan(period_start_days, period_end_days, 
                                             alpha=0.15, color='gray', zorder=0)
                                    
                                    # Add period name centered at top of the region
                                    period_center = (period_start_days + period_end_days) / 2
                                    # Position text at top of plot (95% of y-range from bottom)
                                    text_y = y_min + 0.95 * y_range
                                    ax.text(period_center, text_y, period_name, 
                                           ha='center', va='bottom', 
                                           fontsize=9, color='black', 
                                           zorder=10)
                        
                        plt.tight_layout()
                        plt.show()
                except Exception as e:
                    plt.figure(figsize=(6, 4.1))
                    plt.text(0.5, 0.5, f'No {metric} data available\n{str(e)}', ha='center', va='center')
                    plt.axis('off')
                    plt.show()
        plot_panel = widgets.VBox([plot_header_row, plot_output], layout=widgets.Layout(
            border='none', border_radius='10px', padding='0px', margin='0px', flex='1 1 0%', width='100%', align_items='flex-start', justify_content='flex-start', overflow='hidden'))

        # --- Segmentierungsoptionen (wie Visualisation._create_segmentation_controls, aber ohne HDF5) ---
        thresholding_input = widgets.FloatText(value=1.0, step=0.1, layout=widgets.Layout(width='60px'))
        ai_input = widgets.FloatText(value=0.7, step=0.05, layout=widgets.Layout(width='60px'))
        manual_btn = widgets.Button(description='Manual', layout=widgets.Layout(width='100px'))
        thresholding_btn = widgets.Button(description='Thresholding', layout=widgets.Layout(width='100px'))
        ai_btn = widgets.Button(description='AI', layout=widgets.Layout(width='100px'))
        delete_btn = widgets.Button(description='Delete Contour', button_style='danger', layout=widgets.Layout(width='120px'))

        def manual_segmentation(b):
            idx = time_slider.value
            sph_img = self.spheroid_image_dict[timepoints[idx]]
            # Map dropdown value to SpheroidImage channel key
            dropdown_val = channel_dropdown.value
            if dropdown_val == 'brightfield':
                ch = 'brightfield'
            elif dropdown_val.startswith('fluorescence_'):
                ch = dropdown_val
            else:
                ch = 'brightfield'
            try:
                sph_img.segmentation_manual(ch)
                # Save contour to HDF5
                if hasattr(sph_img, '_save_contour_to_hdf5'):
                    sph_img._save_contour_to_hdf5()
                update_image()
                update_plot()
            except Exception as e:
                print(f"Manual segmentation failed: {e}")
        def thresholding_segmentation(b):
            idx = time_slider.value
            sph_img = self.spheroid_image_dict[timepoints[idx]]
            try:
                dropdown_val = channel_dropdown.value
                ch = dropdown_val if dropdown_val == 'brightfield' or dropdown_val.startswith('fluorescence_') else 'brightfield'
                res = sph_img.segmentation_thresholding(ch, thresholding_input.value)
                # Support both legacy (tuple) and current (single contour) returns
                if isinstance(res, tuple) and len(res) == 2:
                    sph_img.contour, sph_img.contour_touches_border = res
                else:
                    sph_img.contour = res
                    # contour_touches_border was set inside segmentation_thresholding
                # Save contour to HDF5
                if hasattr(sph_img, '_save_contour_to_hdf5'):
                    sph_img._save_contour_to_hdf5()
                update_image()
                update_plot()
            except Exception as e:
                print(f"Thresholding segmentation failed: {e}")
        def ai_segmentation(b):
            idx = time_slider.value
            sph_img = self.spheroid_image_dict[timepoints[idx]]
            try:
                dropdown_val = channel_dropdown.value
                ch = dropdown_val if dropdown_val in ['brightfield', 'fluorescence_green', 'fluorescence_red', 'fluorescence_blue'] else 'brightfield'
                # Use the shared AI dispatcher so Config.ai_segmentation is respected.
                sph_img.segmentation(
                    methods=[('ai', ch)],
                    border_margin=5,
                    confidence=ai_input.value,
                )
                # Save contour to HDF5
                if hasattr(sph_img, '_save_contour_to_hdf5'):
                    sph_img._save_contour_to_hdf5()
                update_image()
                update_plot()
            except Exception as e:
                print(f"AI segmentation failed: {e}")
        def delete_contour(b):
            idx = time_slider.value
            sph_img = self.spheroid_image_dict[timepoints[idx]]
            sph_img.contour = None
            sph_img.contour_touches_border = False
            # Save deletion to HDF5 (remove contour from file)
            if hasattr(sph_img, '_save_contour_to_hdf5'):
                sph_img._save_contour_to_hdf5()  # This will handle None contour correctly
            update_image()
            update_plot()

        manual_btn.on_click(manual_segmentation)
        thresholding_btn.on_click(thresholding_segmentation)
        ai_btn.on_click(ai_segmentation)
        delete_btn.on_click(delete_contour)

        seg_title = widgets.HTML("<h4 style='color: #2c3e50; margin:0;'>Segmentation Options</h4>")
        seg_buttons = widgets.HBox([
            manual_btn,
            widgets.HBox([thresholding_btn, thresholding_input]),
            widgets.HBox([ai_btn, ai_input]),
            delete_btn
        ], layout=widgets.Layout(justify_content='flex-start', align_items='center', width='100%', gap='40px'))
        seg_panel = widgets.VBox([
            seg_title,
            seg_buttons
        ], layout=widgets.Layout(margin='0px', padding='10px', border='none', border_radius='5px', width='100%'))

        # --- Layout-Callbacks ---
        def on_any_change(*args):
            update_image()
            update_plot()
        time_slider.observe(lambda change: on_any_change(), names='value')
        channel_dropdown.observe(lambda change: update_image(), names='value')
        show_contour.observe(lambda change: update_image(), names='value')
        metric_dropdown.observe(lambda change: update_plot(), names='value')

        # --- Zeitlabel-Update ---
        def update_time_label(idx):
            def _ensure_dt(val):
                return val if isinstance(val, datetime) else datetime.strptime(val, '%Y-%m-%d %H:%M:%S')
            t0 = _ensure_dt(timepoints[0])
            t = _ensure_dt(timepoints[idx])
            delta = t - t0
            days = delta.days
            hours = delta.seconds // 3600
            time_label.value = f"{days}d {hours}h"

        # --- Rechte Spalte: Plot mit Segmentation Options darunter ---
        right_column = widgets.VBox([
            plot_panel,
            seg_panel
        ], layout=widgets.Layout(
            border='0px', border_radius='0px', padding='0px', margin='0px', flex='1 1 0%', width='100%'))

        # --- Hauptreihe: Bild links, Plot+Seg rechts ---
        main_row = widgets.HBox([image_panel, right_column], layout=widgets.Layout(
            gap='0px', width='100%', align_items='stretch'))

        # --- Segmentierungsoptionen wie gehabt ---
        # seg_panel = widgets.VBox([
        #     widgets.HTML("<h4 style='color: #2c3e50;'>Segmentation Options</h4>"),
        #     seg_controls
        # ], layout=widgets.Layout(margin='0px', padding='0px', border='1px solid #ccc', border_radius='5px', width='100%'))

        # --- Gesamtlayout: Header + Hauptreihe ---
        vbox = widgets.VBox([
            header_container,
            main_row
        ], layout=widgets.Layout(width='95%', max_width='1400px', align_items='stretch'))

        # Wrapper für linksbündige Anzeige im Notebook
        outer_box = widgets.HBox([vbox], layout=widgets.Layout(width='100%', overflow_x='hidden'))

        # Display first, then render initial content.
        # In some notebook frontends the first Output draw can be dropped if done too early.
        display(outer_box)

        def _initial_render():
            update_time_label(time_slider.value)
            update_image()
            update_plot()

        try:
            from IPython import get_ipython
            ip = get_ipython()
            if ip is not None and hasattr(ip, "kernel") and hasattr(ip.kernel, "io_loop"):
                ip.kernel.io_loop.add_callback(_initial_render)
            else:
                _initial_render()
        except Exception:
            _initial_render()