from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Callable
import logging

from pathlib import Path
import numpy as np
from tqdm import tqdm
import h5py
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import re

from SpheroidPy.spheroid.spheroid_image import SpheroidImage
from SpheroidPy.spheroid.spheroid_series import SpheroidSeries
from SpheroidPy.spheroid.spheroid_collection import SpheroidCollection

from SpheroidPy.utils.file_management import collect_data
from SpheroidPy.utils.color_palettes import CARTO_SEQUENTIAL

import multiprocessing as mp

if TYPE_CHECKING:
    from SpheroidPy.experiment.result import Result
    from SpheroidPy.experiment.platemap import Platemap

logger = logging.getLogger("SpheroidPy.experiment.livecell_replicate")


def _process_well_from_file_standalone(well_data):
    """Standalone helper function to process a well when loading from file.

    Args:
        well_data: Tuple of (well, timepoint_data) or (well, timepoint_data, hdf5_path).
                  timepoint_data contains all necessary information extracted from HDF5.
                  If hdf5_path (str or Path) is provided as third element, contours
                  can be saved back to HDF5 after segmentation.

    Returns:
        Tuple of (well, spheroid_series, well_results)
    """
    if len(well_data) >= 3:
        well, timepoint_data, hdf5_path = well_data[0], well_data[1], well_data[2]
    else:
        well, timepoint_data = well_data[0], well_data[1]
        hdf5_path = None
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
            hdf5_key=data['hdf5_key'],
            hdf5_path=hdf5_path,
        )

        if data.get('contour') is not None:
            spheroid.contour = data['contour']
            # Convert int to bool for compatibility (stored as 0/1 for PyTables)
            touches_border_val = data.get('touches_border', 0)
            if isinstance(touches_border_val, (int, np.integer)):
                spheroid.contour_touches_border = bool(touches_border_val)
            else:
                spheroid.contour_touches_border = bool(touches_border_val) if touches_border_val else False

        well_results[timepoint] = {
            'spheroid': spheroid,
            'image_paths': image_paths
        }

    return well, spheroid_series, well_results


class LiveCellReplicate:
    """A class representing a biological replicate with associated platemap and image data.
    
    A LiveCellReplicate acts as an intermediate layer between Result and SpheroidCollection
    for live-cell imaging experiments. Each replicate has its own Platemap and can load images,
    creating SpheroidCollection objects based on the platemap mapping (technical replicates).
    
    This is an optional feature for Results that use plate-based well layouts.
    
    Attributes:
        result: The Result this replicate belongs to
        name: Display name of the replicate
        layout: Number of wells in the plate (currently only 96 is supported)
        platemap: Associated Platemap object defining well contents
        spheroid_dict: Dictionary mapping well IDs to SpheroidSeries objects
        files_dict: Nested dict mapping channels -> wells -> timepoints to image paths
        time_points_array: List of absolute timepoints that have been loaded
        relative_time_array: List of relative times in hours from experiment start
        data_has_been_loaded: Whether images have been loaded
    """
    
    def __init__(self, name: str, result: "Result", layout: int = 96):
        """Initialize a new replicate with result context.
        
        Args:
            name: Display name of the replicate
            result: The Result this replicate belongs to (required)
            layout: Number of wells in the plate (currently only 96 is supported)
        """
        # Validate layout
        if layout != 96:
            raise ValueError(f"Currently only 96-well plates are supported, got layout={layout}")
        
        self.name = name
        self.result = result
        self.layout = layout
        self.data_has_been_loaded = False
        self.spheroid_dict: dict[str, SpheroidSeries] = {}
        self.files_dict: dict = {}
        self.time_points_array: list = []
        self.relative_time_array: list = []
        self.num_cores = mp.cpu_count()
        self._created_at = datetime.now()
        
        # Create platemap for this replicate
        from SpheroidPy.experiment.platemap import Platemap
        self.platemap = Platemap(self)
        
        # Initialize HDF5 structure
        self._initialize_hdf5_structure()
    
    @property
    def hdf5_path(self) -> Path:
        """Get HDF5 file path from result."""
        # First, try stored hdf5_path (set during from_file)
        if hasattr(self, '_hdf5_path') and self._hdf5_path:
            return Path(self._hdf5_path)
        # Try result's hdf5_path property (which gets it from experiment)
        if hasattr(self.result, 'hdf5_path') and self.result.hdf5_path:
            return Path(self.result.hdf5_path)
        # Fallback: try to get directly from experiment
        if hasattr(self.result, 'experiment') and self.result.experiment is not None:
            if hasattr(self.result.experiment, 'hdf5_path') and self.result.experiment.hdf5_path:
                return Path(self.result.experiment.hdf5_path)
        raise ValueError(
            f"Cannot determine HDF5 path for replicate '{self.name}'. "
            f"Result '{self.result.name}' must be added to an Experiment with an HDF5 path."
        )
    
    def _get_hdf5_key(self) -> str:
        """Get HDF5 key for this replicate."""
        if not hasattr(self.result, 'replicates_list') or self not in self.result.replicates_list:
            # If not in list yet, use length as index
            replicate_index = len(getattr(self.result, 'replicates_list', []))
        else:
            replicate_index = self.result.replicates_list.index(self)
        
        # Get result hdf5_key
        if hasattr(self.result, 'hdf5_key'):
            result_key = self.result.hdf5_key
        else:
            # Fallback: construct from result name and _result_index
            result_index = getattr(self.result, '_result_index', 0)
            result_key = f"Result/{result_index}-{self.result.name}"
        
        return f'{result_key}/ReplicateInfo/Replicates/{replicate_index}-{self.name}'
    
    @property
    def hdf5_key(self) -> str:
        """Get HDF5 key for this replicate."""
        return self._get_hdf5_key()
    
    def _initialize_hdf5_structure(self):
        """Initialize HDF5 storage structure."""
        try:
            with h5py.File(self.hdf5_path, 'a') as hdf_file:
                # Create or get the replicate group
                replicate_group = hdf_file.require_group(self.hdf5_key)
                replicate_group.attrs['name'] = self.name
                replicate_group.attrs['layout'] = self.layout
                replicate_group.attrs['created_at'] = self._created_at.isoformat()
                replicate_group.attrs['modified_at'] = datetime.now().isoformat()
                
                # Create platemap group
                platemap_group = replicate_group.require_group('Platemap')
        except Exception as e:
            logger.warning(f"Could not initialize HDF5 structure for replicate: {e}")
    
    @staticmethod
    def _process_channel_data(args):
        """Static helper function to process image channels in parallel.

        Args:
            args: Tuple of (img_folder_path, filters, channel_name)

        Returns:
            Tuple of (channel_name, files_dict, time_points)
        """
        img_folder_path, filters, channel = args
        if filters is None:
            return channel, None, None
        return channel, *collect_data(img_folder_path, filters)

    @staticmethod
    def _process_well_data(args):
        """Static helper function to process a single well.

        Args:
            args: Tuple containing (well, timepoints, files_dict, relative_time_array, image_size, hdf5_key, hdf5_path)

        Returns:
            Tuple of (well, results)
        """
        well, timepoints, files_dict, relative_time_array, image_size, hdf5_key, hdf5_path = args
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
                hdf5_key=f'{hdf5_key}/ImageSeries/{well}/{timepoint}',
                hdf5_path=hdf5_path
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
                    image_size: tuple | None = None) -> None:
        """Load and process image data from specified folder using multiprocessing.

        Args:
            img_folder_path: Path to folder containing image files
            brightfield_filters: Filter pattern(s) for brightfield images
            fluorescence_green_filters: Optional filter pattern(s) for green fluorescence
            fluorescence_red_filters: Optional filter pattern(s) for red fluorescence
            fluorescence_blue_filters: Optional filter pattern(s) for blue fluorescence
            image_size: Optional tuple (width, height) specifying image size in μm

        Raises:
            Exception: If data has already been loaded for this Replicate
        """
        if self.data_has_been_loaded:
            raise Exception('Data has already been loaded for this replicate')

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

        print(f'Looking for Wells in replicate "{self.name}":', wells)

        # Process wells in parallel
        well_data = [
            (well, self.time_points_array, self.files_dict, self.relative_time_array,
             image_size, self.hdf5_key, str(self.hdf5_path) if hasattr(self, 'hdf5_path') else None)
            for well in wells
        ]

        with mp.Pool(processes=self.num_cores) as pool:
            for well, results in tqdm(
                    pool.imap(self._process_well_data, well_data),
                    total=len(well_data),
                    desc=f"Loading wells for replicate '{self.name}'"
            ):
                # Create series for this well
                self.spheroid_dict[well] = SpheroidSeries(f'Spheroid-{well}')

                # Process results for this well
                for result in results:
                    # Add to spheroid series (timepoint stays as string)
                    self.spheroid_dict[well].add_spheroid_image(
                        result['spheroid'],
                        result['timepoint']
                    )
                
                # Note: ImageSeries are NOT saved here - they will be saved in Collections/
                # when Result.save_to_hdf5() is called after _update_collections_from_replicates()

        self.data_has_been_loaded = True
        print(f'Images have been loaded successfully for replicate "{self.name}"!')

        for spheroid_series in self.spheroid_dict.values():
            spheroid_series._result = self.result
        
        # Update collections in result
        if hasattr(self.result, '_update_collections_from_replicates'):
            self.result._update_collections_from_replicates()
        
        # Automatically save to HDF5 if result has an hdf5_path
        # This ensures Collections are saved immediately after loading images
        if hasattr(self.result, 'hdf5_path') and self.result.hdf5_path:
            try:
                # Check if result has a valid index (it should if it's part of an experiment)
                if hasattr(self.result, '_result_index') and self.result._result_index is not None:
                    # Use the stored index - it's the most reliable
                    self.result.save_to_hdf5(self.result.hdf5_path)
                elif hasattr(self.result, 'experiment') and self.result.experiment:
                    # Fallback: try to find index from experiment (update _result_index if found)
                    result_index = None
                    for idx, (name, res) in enumerate(self.result.experiment.results_dict.items()):
                        if res == self.result or name == self.result.name:
                            result_index = idx
                            self.result._result_index = idx  # Store it for future use
                            break
                    
                    if result_index is not None:
                        self.result.save_to_hdf5(self.result.hdf5_path)
                    else:
                        # Result not found in experiment - don't auto-save
                        logger.debug(f"Result '{self.result.name}' not in experiment.results_dict, skipping auto-save")
                else:
                    # No experiment or index - skip auto-save to avoid incorrect indexing
                    logger.debug(f"Result '{self.result.name}' has no experiment or index, skipping auto-save")
            except Exception as e:
                import logging
                logger = logging.getLogger("SpheroidPy.experiment.livecell_replicate")
                logger.warning(f"Could not auto-save collections after loading images: {e}")
            except Exception as e:
                import logging
                logger = logging.getLogger("SpheroidPy.experiment.livecell_replicate")
                logger.warning(f"Could not auto-save collections after loading images: {e}")
    
    def get_collections(self) -> dict[tuple, SpheroidCollection]:
        """Get SpheroidCollection objects based on platemap replicates.
        
        Creates SpheroidCollection objects for each unique condition in the platemap,
        grouping wells with the same condition values as technical replicates.
        These collections reference the same SpheroidSeries objects (no duplication).
        
        If a collection with the same name already exists in result.collections, it will
        be reused and updated rather than creating a new one. This prevents duplicate
        SpheroidSeries when _update_collections_from_replicates() is called multiple times.
        
        Returns:
            Dictionary mapping condition tuples to SpheroidCollection objects
        """
        replicate_wells = self.platemap.replicates()
        collections = {}
        
        for condition, wells in replicate_wells.items():
            # Get valid spheroid series for these wells
            spheroid_list = [self.spheroid_dict[well] 
                           for well in wells 
                           if well in self.spheroid_dict]
            
            if spheroid_list:
                # Create collection name from condition, including replicate name for uniqueness
                # This ensures that different biological replicates with the same condition
                # get separate collections (they are stored as separate entries in Result.collections)
                if isinstance(condition, tuple) and len(condition) == 1:
                    cond = condition[0]
                    if isinstance(cond, tuple):
                        collection_name = f"{self.name}_{cond[0]}={cond[1]}"
                    else:
                        collection_name = f"{self.name}_{str(cond)}"
                else:
                    collection_name = f"{self.name}_{str(condition)}"
                
                # Check if a collection with this name already exists in result.collections
                # Convert condition to result condition format for lookup
                existing_collection = None
                if hasattr(self.result, 'collections'):
                    result_cond_tuple = self.result._convert_platemap_condition_to_result_condition(condition)
                    if result_cond_tuple in self.result.collections:
                        for coll in self.result.collections[result_cond_tuple]:
                            if coll.name == collection_name:
                                existing_collection = coll
                                break
                
                if existing_collection:
                    # Reuse existing collection - update it to match current state
                    # Remove duplicates first
                    seen_names = set()
                    unique_series = []
                    for s in existing_collection.spheroid_series:
                        if s.name not in seen_names:
                            seen_names.add(s.name)
                            unique_series.append(s)
                    existing_collection.spheroid_series = unique_series
                    
                    # Update series list to match current spheroid_list (by name)
                    existing_series_names = {s.name for s in existing_collection.spheroid_series}
                    new_series_names = {s.name for s in spheroid_list}
                    
                    # Remove series that are no longer in platemap
                    series_to_remove = [s for s in existing_collection.spheroid_series
                                      if s.name not in new_series_names]
                    for s in series_to_remove:
                        existing_collection.spheroid_series.remove(s)
                    
                    # Add new series that aren't already in collection
                    series_to_add = [s for s in spheroid_list 
                                   if s.name not in existing_series_names]
                    if series_to_add:
                        existing_collection.spheroid_series.extend(series_to_add)
                    
                    # Update replicate source
                    if not hasattr(existing_collection, '_replicate_source'):
                        existing_collection._replicate_source = []
                    if self not in existing_collection._replicate_source:
                        existing_collection._replicate_source.append(self)
                    
                    collection = existing_collection
                else:
                    # Create new collection
                    # Get hdf5_path for collection (use property, not attribute check)
                    try:
                        collection_hdf5_path = str(self.hdf5_path) if self.hdf5_path else None
                    except (ValueError, AttributeError):
                        collection_hdf5_path = None
                    
                    collection = SpheroidCollection(
                        name=collection_name,
                        spheroid_list=spheroid_list,
                        result=self.result,
                        hdf5_path=collection_hdf5_path
                    )
                    
                    # Mark this collection as belonging to this replicate
                    if not hasattr(collection, '_replicate_source'):
                        collection._replicate_source = []
                    if self not in collection._replicate_source:
                        collection._replicate_source.append(self)
                
                collections[condition] = collection
        
        return collections
    
    @property
    def time_periods(self) -> dict:
        """
        Get time periods that are common to all SpheroidSeries in this replicate.
        
        Returns a dictionary of time periods that exist in all series with the same
        start_time, end_time, and description. If series have different time periods,
        only the common ones are returned.
        
        Returns
        -------
        dict[str, TimePeriod]
            Dictionary of time periods common to all series.
        """
        if not self.spheroid_dict:
            return {}
        
        # Get time_periods from first series
        first_series = next(iter(self.spheroid_dict.values()))
        if not hasattr(first_series, 'time_periods') or not first_series.time_periods:
            return {}
        
        common_periods = {}
        first_periods = first_series.time_periods
        
        # Check each period in first series
        for period_name, first_period in first_periods.items():
            is_common = True
            # Check if all other series have the same period
            for series in self.spheroid_dict.values():
                if not hasattr(series, 'time_periods') or period_name not in series.time_periods:
                    is_common = False
                    break
                other_period = series.time_periods[period_name]
                # Compare key attributes
                if (other_period.start_time != first_period.start_time or
                    other_period.end_time != first_period.end_time or
                    other_period.description != first_period.description):
                    is_common = False
                    break
            
            if is_common:
                common_periods[period_name] = first_period
        
        return common_periods
    
    def time_period(self, name: str,
                   start_time: str | datetime | None = None,
                   end_time: str | datetime | None = None,
                   exclude: list = [],
                   description: str | None = None,
                   remove: bool = False) -> Optional:
        """
        Add, update, or remove a named `TimePeriod` for all SpheroidSeries in this replicate.
        
        This method applies the time period to all SpheroidSeries in the replicate.
        Time periods are stored at the Series level in HDF5, not at the Replicate level.

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
            If True, removes the specified time period from all series instead of creating
            or updating it.

        Returns
        -------
        TimePeriod | None
            The created or updated `TimePeriod`, or None if removed.

        Raises
        ------
        ValueError
            If there are no timepoints available to infer default start/end times.
        """
        if not self.spheroid_dict:
            raise ValueError("No SpheroidSeries available in this replicate. Load images first.")
        
        if remove:
            # Remove from all series
            for series in self.spheroid_dict.values():
                if hasattr(series, 'time_periods') and name in series.time_periods:
                    series.time_periods.pop(name, None)
                    if hasattr(series, '_period_cache'):
                        series._period_cache.pop(name, None)
                    # Save to HDF5 if series has save method
                    if hasattr(series, 'save_time_periods_to_hdf5'):
                        try:
                            series.save_time_periods_to_hdf5()
                        except Exception:
                            pass  # Best effort
            return None

        # Convert string times to datetime if needed
        if isinstance(start_time, str):
            start_time = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
        if isinstance(end_time, str):
            end_time = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')
            
        # Infer start/end defaults from time_points_array
        if not self.time_points_array:
            raise ValueError("No timepoints available to infer default time bounds")
        
        def _ensure_dt(val):
            return val if isinstance(val, datetime) else datetime.strptime(val, '%Y-%m-%d %H:%M:%S')
        
        sorted_times = sorted([_ensure_dt(tp) for tp in self.time_points_array])
        default_start = sorted_times[0]
        default_end = sorted_times[-1]

        # Use defaults from time_points_array (global for replicate)
        final_start_time = start_time if start_time is not None else default_start
        final_end_time = end_time if end_time is not None else default_end
        
        from SpheroidPy.utils.time_period import TimePeriod
        
        # Create or get TimePeriod object
        first_series = next(iter(self.spheroid_dict.values()))
        if not hasattr(first_series, 'time_periods'):
            from SpheroidPy.utils.time_period import TimePeriod
            first_series.time_periods = {}
        
        if name in first_series.time_periods:
            # Update existing period
            tp = first_series.time_periods[name]
            tp.start_time = final_start_time
            tp.end_time = final_end_time
            if description is not None:
                tp.description = description
            if exclude:
                tp.exclude = exclude
            # Invalidate cache
            if hasattr(first_series, '_period_cache'):
                first_series._period_cache.pop(name, None)
        else:
            # Create new TimePeriod
            tp = TimePeriod(
                name=name,
                start_time=final_start_time,
                end_time=final_end_time,
                description=description,
                exclude=exclude
            )
            first_series.time_periods[name] = tp
        
        # Apply to all other series
        for series in self.spheroid_dict.values():
            if series != first_series:
                if not hasattr(series, 'time_periods'):
                    series.time_periods = {}
                # Update or add period
                series.time_periods[name] = tp
                # Invalidate cache
                if hasattr(series, '_period_cache'):
                    series._period_cache.pop(name, None)
        
        # Save to HDF5 for all series (best effort)
        for series in self.spheroid_dict.values():
            if hasattr(series, 'save_time_periods_to_hdf5'):
                try:
                    series.save_time_periods_to_hdf5()
                except Exception:
                    pass  # Best effort, don't fail if HDF5 save fails

        return tp
    
    def get_series_in_period(self, name: str, use_cache: bool = True) -> dict[str, SpheroidSeries]:
        """
        Retrieve all SpheroidSeries filtered by time period.
        
        Uses the time_period from the first available series. All series should have
        the same time_periods (set via time_period() method).

        Parameters
        ----------
        name : str
            The name of the time period.
        use_cache : bool, default=True
            Whether to use cached results if available (uses series-level cache).

        Returns
        -------
        dict[str, SpheroidSeries]
            Dictionary mapping well IDs to filtered SpheroidSeries objects.

        Raises
        ------
        KeyError
            If the specified time period does not exist in any series.
        """
        if not self.spheroid_dict:
            raise KeyError(f"No SpheroidSeries available in this replicate.")
        
        # Get time period from first series
        first_series = next(iter(self.spheroid_dict.values()))
        if not hasattr(first_series, 'time_periods') or name not in first_series.time_periods:
            raise KeyError(f"No time period named '{name}' defined in series.")
        
        tp = first_series.time_periods[name]
        if tp.start_time is None or tp.end_time is None:
            raise ValueError(f"Time period '{name}' must define start and end times.")
        
        # Helper function to convert timestamp to datetime if needed
        def _ensure_dt(val):
            return val if isinstance(val, datetime) else datetime.strptime(val, '%Y-%m-%d %H:%M:%S')
        
        # Filter series by period using series.get_images_in_period if available
        filtered_series = {}
        for well, series in self.spheroid_dict.items():
            if hasattr(series, 'get_images_in_period'):
                # Use series method if available (uses cache)
                try:
                    filtered_images = series.get_images_in_period(name, use_cache=use_cache)
                    if filtered_images:
                        # Create a copy of the series with filtered images
                        filtered_series_obj = SpheroidSeries(series.name)
                        filtered_series_obj.spheroid_image_dict = filtered_images
                        filtered_series_obj._result = series._result
                        filtered_series[well] = filtered_series_obj
                except KeyError:
                    # Period doesn't exist in this series, skip it
                    continue
            else:
                # Fallback: manual filtering
                filtered_images = {
                    timestamp: image 
                    for timestamp, image in series.spheroid_image_dict.items()
                    if tp.contains(_ensure_dt(timestamp))
                }
                
                if filtered_images:
                    # Create a copy of the series with filtered images
                    filtered_series_obj = SpheroidSeries(series.name)
                    filtered_series_obj.spheroid_image_dict = filtered_images
                    filtered_series_obj._result = series._result
                    filtered_series[well] = filtered_series_obj
        
        return filtered_series
    
    
    def save_to_hdf5(self) -> None:
        """Save replicate data to HDF5."""
        try:
            with h5py.File(self.hdf5_path, 'a') as hdf_file:
                replicate_group = hdf_file.require_group(self.hdf5_key)
                replicate_group.attrs['name'] = self.name
                replicate_group.attrs['layout'] = self.layout
                replicate_group.attrs['modified_at'] = datetime.now().isoformat()
                
                # Save platemap
                self.platemap._save_all_to_hdf5()
        except Exception as e:
            logger.warning(f"Could not save replicate to HDF5: {e}")
    
    def segmentation(
        self,
        methods: list | tuple = [('thresholding', 'fluorescence_green'), ('ai', 'brightfield')],
        reconstruct_border: bool = True,
        border_margin: int = 10,
        show_progress: bool = True,
        **kwargs,
    ) -> None:
        """
        Segment all spheroid series in this replicate.

        Args:
            methods: List/Tuple of segmentation specs, e.g.
                [('thresholding','fluorescence_green'), ('ai','brightfield')]
            reconstruct_border: Whether to reconstruct border if contour touches image border.
            border_margin: Margin (px) for border detection.
            show_progress: Whether to show progress bar (default: True). Set to False when called from higher-level methods.
            **kwargs: Forwarded to ``SpheroidImage.segmentation`` (e.g. threshold, use_yen, confidence).
        """
        import multiprocessing as mp
        from tqdm import tqdm
        from SpheroidPy.spheroid.spheroid_collection import SpheroidCollection
        
        if not self.data_has_been_loaded:
            print("No data loaded for this replicate. Call load_images() first.")
            return
        
        if not self.spheroid_dict:
            print("No spheroid series in replicate to segment.")
            return

        # Collect all series for unified progress tracking
        all_series_args = []
        series_to_well = {}  # Map series_name -> well
        
        for well, series in self.spheroid_dict.items():
            timepoints = list(series.spheroid_image_dict.keys())
            spheroid_images = series.spheroid_image_dict
            
            series_args = (
                series.name,
                timepoints,
                spheroid_images,
                methods,
                reconstruct_border,
                border_margin,
                kwargs
            )
            all_series_args.append(series_args)
            series_to_well[series.name] = well

        if not all_series_args:
            print("No active spheroid series in replicate to segment.")
            return

        # Process all series in parallel with single progress bar
        num_cores = mp.cpu_count()
        with mp.Pool(processes=num_cores) as pool:
            if show_progress:
                results = list(tqdm(
                    pool.imap(SpheroidCollection._process_segmentation_series, all_series_args),
                    total=len(all_series_args),
                    desc=f"Replicate '{self.name}'"
                ))
            else:
                results = list(pool.imap(SpheroidCollection._process_segmentation_series, all_series_args))

        # Update SpheroidImage instances with segmentation results
        # Note: Contours are automatically saved to HDF5 by SpheroidImage._save_contour_to_hdf5()
        # after segmentation, so we don't need to save them manually here
        total_success = 0
        total_images = 0
        for series_name, success, total, contour_data in results:
            total_success += success
            total_images += total
            
            well = series_to_well[series_name]
            series = self.spheroid_dict[well]
            for timepoint, data in contour_data.items():
                spheroid_image = series.spheroid_image_dict[timepoint]
                spheroid_image.contour = data['contour']
                spheroid_image.contour_touches_border = data['touches_border']
                spheroid_image._save_contour_to_hdf5()

        if show_progress:
            print(f"\nSegmentation complete for replicate '{self.name}':")
            print(f"Successfully segmented {total_success}/{total_images} images across {len(all_series_args)} series")

    def compare(self, condition: str, metric: str = 'radius', plot: bool = True,
                timepoint: int | float | str | list[int | float | str] | None = None,
                fit: bool = False, **kwargs) -> pd.DataFrame:
        """
        Compare collections across a specific condition, grouping by other conditions.

        For the specified condition, finds all collections that differ only in that condition
        but are identical in all other conditions. Each such group becomes a column in the
        resulting DataFrame, with rows representing the values of the specified condition.

        Args:
            condition: Name of the condition to vary (e.g., 'concentration', 'temperature')
            metric: Metric to calculate (default: 'radius')
            plot: Whether to plot the results (default: True)
            timepoint: Optional specific timepoint(s) to calculate metrics for.
                - If int/float: Timepoint in hours (relative time)
                - If str: Timepoint as datetime string (e.g., '2025-03-27 20:00:00')
                - If list: Multiple timepoints; each plotted with legend entry
                - If None: Average across all timepoints (default)
            fit: Whether to fit models to the data (default: False)
                - If condition is a cell line (cell number): fits effective cell volume
                - If condition is a compound: fits Hill curve (IC50, Hill slope)
            **kwargs: Additional parameters for fitting:
                - For cell line fits: 'aspect_ratio' (default: 1.0)
                - For compound fits: passed to Hill curve fitting

        Returns:
            DataFrame with:
                - Index: Values of the specified condition
                - Columns: One per group of collections that differ only in other conditions
                - Values: Metric values (mean ± std across technical replicates within each collection)
                - If timepoint is specified, returns mean and std for that timepoint
                - If timepoint is a list, returns MultiIndex columns (group, timepoint)
                - If timepoint is None, returns mean across all timepoints
                - If fit=True, also returns fit parameters as attributes on the DataFrame
                - attrs['available_timepoints']: Sorted list of timepoints with actual data
                - attrs['extrapolation_warnings']: List of warnings when target was outside data range

        Note:
            Extrapolation: When the requested timepoint is outside the data range, the closest
            available point is used. Values at missing timepoints are filled via pandas
            interpolate(method='linear', limit_direction='both'), i.e. linear extrapolation.
        """
        import matplotlib.pyplot as plt
        
        # Handle backward compatibility: if metric is an int, it's probably timepoint
        if isinstance(metric, (int, float)) and timepoint is None:
            timepoint = metric
            metric = 'radius'
            logger.warning(
                f"Interpreting second argument as timepoint. Use compare('{condition}', timepoint={timepoint}) "
                f"for clarity. Using metric='radius'."
            )
        
        # Ensure metric is a string
        if not isinstance(metric, str):
            raise TypeError(f"metric must be a string, got {type(metric)}")

        if not self.data_has_been_loaded:
            print("No data loaded for this replicate. Call load_images() first.")
            return pd.DataFrame()

        # Get all collections
        collections = self.get_collections()

        if not collections:
            print("No collections found. Ensure platemap is configured and images are loaded.")
            return pd.DataFrame()

        # Parse condition tuples and extract condition values
        # Condition tuples have format: ((name, value),) or ((name1, value1), (name2, value2), ...)
        condition_data = {}  # {condition_tuple: (condition_dict, collection)}

        for cond_tuple, collection in collections.items():
            # Convert condition tuple to dictionary
            cond_dict = {}
            if isinstance(cond_tuple, tuple):
                for item in cond_tuple:
                    if isinstance(item, tuple) and len(item) == 2:
                        name, value = item
                        cond_dict[name] = value

            condition_data[cond_tuple] = (cond_dict, collection)

        # Check if the specified condition exists
        all_condition_names = set()
        for cond_dict, _ in condition_data.values():
            all_condition_names.update(cond_dict.keys())

        if condition not in all_condition_names:
            print(f"Condition '{condition}' not found in platemap. Available conditions: {sorted(all_condition_names)}")
            return pd.DataFrame()

        # Group collections by their "other conditions" (all except the specified one)
        # Collections in the same group differ only in the specified condition
        groups = {}  # {other_conditions_tuple: {condition_value: collection}}

        for cond_tuple, (cond_dict, collection) in condition_data.items():
            # Extract the value of the specified condition
            if condition not in cond_dict:
                continue

            condition_value = cond_dict[condition]

            # Create a tuple of other conditions (excluding the specified one)
            other_conditions = tuple(
                sorted((name, value) for name, value in cond_dict.items() if name != condition)
            )

            if other_conditions not in groups:
                groups[other_conditions] = {}

            groups[other_conditions][condition_value] = collection

        if not groups:
            print(f"No collections found with condition '{condition}'.")
            return pd.DataFrame()

        # Calculate metrics for each collection
        # Get all unique condition values (for consistent index)
        all_condition_values = set()
        for group_dict in groups.values():
            all_condition_values.update(group_dict.keys())
        all_condition_values = sorted(all_condition_values)

        # Build DataFrame
        result_data = {}
        group_labels = []

        for other_conditions, group_dict in groups.items():
            # Create label for this group (other conditions)
            if other_conditions:
                label_parts = [f"{name}={value}" for name, value in other_conditions]
                group_label = ", ".join(label_parts)
            else:
                group_label = "default"

            group_labels.append(group_label)

            # Calculate metric for each collection in this group
            metric_values_mean = []
            metric_values_std = []
            for cond_value in all_condition_values:
                if cond_value in group_dict:
                    collection = group_dict[cond_value]
                    # Calculate metric with mean=True to average technical replicates
                    metric_df = collection.metric(name=metric, mean=True, plot=False)

                    # Extract mean and std values (metric_df has MultiIndex columns with (collection_name, 'mean') and (collection_name, 'std'))
                    if not metric_df.empty:
                        # Get the mean and std columns
                        if isinstance(metric_df.columns, pd.MultiIndex):
                            # Find mean and std columns
                            mean_col = None
                            std_col = None
                            for col in metric_df.columns:
                                if col[1] == 'mean':
                                    mean_col = col
                                elif col[1] == 'std':
                                    std_col = col

                            if mean_col:
                                mean_values = metric_df[mean_col].values
                            else:
                                mean_values = metric_df.mean(axis=1).values

                            if std_col:
                                std_values = metric_df[std_col].values
                            else:
                                std_values = np.zeros_like(mean_values)
                        else:
                            # Single column - take mean across columns if multiple
                            mean_values = metric_df.mean(axis=1).values
                            std_values = np.zeros_like(mean_values)

                        # Store as Series with timepoints as index
                        mean_series = pd.Series(mean_values, index=metric_df.index)
                        std_series = pd.Series(std_values, index=metric_df.index)
                        metric_values_mean.append(mean_series)
                        metric_values_std.append(std_series)
                    else:
                        metric_values_mean.append(pd.Series(dtype=float))
                        metric_values_std.append(pd.Series(dtype=float))
                else:
                    metric_values_mean.append(pd.Series(dtype=float))
                    metric_values_std.append(pd.Series(dtype=float))

            # Combine all series for this group
            # Use the union of all timepoints as index
            all_timepoints = set()
            for series in metric_values_mean:
                if not series.empty:
                    all_timepoints.update(series.index)

            if all_timepoints:
                combined_index = sorted(all_timepoints)
                combined_data_mean = {}
                combined_data_std = {}
                for idx, cond_value in enumerate(all_condition_values):
                    mean_series = metric_values_mean[idx]
                    std_series = metric_values_std[idx]
                    if not mean_series.empty:
                        # Reindex to include all timepoints, interpolate missing values
                        reindexed_mean = mean_series.reindex(combined_index).interpolate(method='linear',
                                                                                         limit_direction='both')
                        reindexed_std = std_series.reindex(combined_index).interpolate(method='linear',
                                                                                       limit_direction='both')
                        combined_data_mean[cond_value] = reindexed_mean
                        combined_data_std[cond_value] = reindexed_std
                    else:
                        combined_data_mean[cond_value] = pd.Series(index=combined_index, dtype=float)
                        combined_data_std[cond_value] = pd.Series(index=combined_index, dtype=float)

                # Create DataFrames for this group: rows=timepoints, columns=condition values
                group_df_mean = pd.DataFrame(combined_data_mean, index=combined_index)
                group_df_std = pd.DataFrame(combined_data_std, index=combined_index)
                result_data[group_label] = {'mean': group_df_mean, 'std': group_df_std}
            else:
                # No data for this group
                result_data[group_label] = {'mean': pd.DataFrame(), 'std': pd.DataFrame()}

        # Combine all groups into a single DataFrame
        # Structure: MultiIndex columns (group_label, condition_value), rows=timepoints
        if not result_data:
            print("No metric data could be calculated.")
            return pd.DataFrame()

        # Normalize timepoint to list (single value -> list of one)
        target_timepoints = None
        if timepoint is not None:
            if isinstance(timepoint, (list, tuple)):
                target_timepoints = list(timepoint)
            else:
                target_timepoints = [timepoint]
            # Parse each timepoint
            parsed = []
            for tp in target_timepoints:
                if isinstance(tp, (int, float)):
                    parsed.append(float(tp))
                elif isinstance(tp, str):
                    try:
                        dt = datetime.strptime(tp, '%Y-%m-%d %H:%M:%S')
                    except ValueError:
                        try:
                            dt = datetime.fromisoformat(tp)
                        except ValueError:
                            print(
                                f"Could not parse timepoint '{tp}'. Expected format: 'YYYY-MM-DD HH:MM:SS' or ISO format.")
                            return pd.DataFrame()
                    parsed.append(dt)
                else:
                    print(f"Unsupported timepoint type: {type(tp)}")
                    return pd.DataFrame()
            target_timepoints = parsed

        def _to_hours_for_compare(idx_val, first_time=None) -> float | None:
            """Convert index value to hours for comparison, or None if not comparable."""
            if isinstance(idx_val, (int, float)):
                return float(idx_val)
            if isinstance(idx_val, datetime):
                if first_time:
                    return (idx_val - first_time).total_seconds() / 3600
                return idx_val.timestamp() / 3600
            if isinstance(idx_val, str):
                try:
                    dt = datetime.strptime(idx_val, '%Y-%m-%d %H:%M:%S')
                except ValueError:
                    try:
                        dt = datetime.fromisoformat(idx_val)
                    except ValueError:
                        return None
                if first_time:
                    return (dt - first_time).total_seconds() / 3600
                return dt.timestamp() / 3600
            return None

        first_time = None
        if self.relative_time_array and self.time_points_array:
            try:
                first_time = datetime.strptime(self.time_points_array[0], '%Y-%m-%d %H:%M:%S')
            except Exception:
                pass

        def _find_closest_and_check_extrapolation(group_df_mean, target_tp):
            """Find closest index and return (closest_idx, is_extrapolated, available_hours)."""
            index_vals = list(group_df_mean.index)
            if not index_vals:
                return None, False, []
            target_hours = target_tp if isinstance(target_tp, (int, float)) else None
            if target_hours is None and isinstance(target_tp, datetime) and first_time:
                target_hours = (target_tp - first_time).total_seconds() / 3600
            elif target_hours is None and isinstance(target_tp, datetime):
                target_hours = target_tp.timestamp() / 3600
            available_hours = []
            for iv in index_vals:
                h = _to_hours_for_compare(iv, first_time)
                if h is not None:
                    available_hours.append((iv, h))
            if not available_hours:
                return index_vals[0], False, []
            available_hours.sort(key=lambda x: x[1])
            hrs = [x[1] for x in available_hours]
            min_h, max_h = min(hrs), max(hrs)
            closest = min(available_hours, key=lambda x: abs(x[1] - target_hours))
            closest_idx = closest[0]
            is_extrapolated = target_hours < min_h or target_hours > max_h
            return closest_idx, is_extrapolated, [x[1] for x in available_hours]

        # Collect available timepoints (union across groups)
        all_available_timepoints = set()
        extrapolation_warnings = []

        # For plotting and simpler output, create a DataFrame with condition values as index
        # and groups as columns, where each cell contains the metric value at a specific timepoint
        if target_timepoints is None:
            # Mean across all timepoints
            summary_data_mean = {}
            summary_data_std = {}
            for group_label, group_data in result_data.items():
                group_df_mean = group_data['mean']
                group_df_std = group_data['std']
                if not group_df_mean.empty:
                    all_available_timepoints.update(
                        _to_hours_for_compare(i, first_time) for i in group_df_mean.index
                        if _to_hours_for_compare(i, first_time) is not None
                    )
                    summary_data_mean[group_label] = group_df_mean.mean(axis=0)
                    summary_data_std[group_label] = group_df_std.mean(axis=0)
            result_df_mean = pd.DataFrame(summary_data_mean)
            result_df_std = pd.DataFrame(summary_data_std)
            timepoint_str = " (mean across timepoints)"
        else:
            # One or more specific timepoints
            if len(target_timepoints) == 1:
                target_tp = target_timepoints[0]
                summary_data_mean = {}
                summary_data_std = {}
                for group_label, group_data in result_data.items():
                    group_df_mean = group_data['mean']
                    group_df_std = group_data['std']
                    if not group_df_mean.empty:
                        closest_idx, is_extrap, avail = _find_closest_and_check_extrapolation(
                            group_df_mean, target_tp
                        )
                        all_available_timepoints.update(avail)
                        if closest_idx is not None:
                            summary_data_mean[group_label] = group_df_mean.loc[closest_idx]
                            summary_data_std[group_label] = group_df_std.loc[closest_idx]
                            if is_extrap:
                                tp_str = f"{target_tp}h" if isinstance(target_tp, (int, float)) else str(target_tp)
                                avail_sorted = sorted(set(avail))
                                extrapolation_warnings.append(
                                    f"Timepoint {tp_str}: No data at this time. "
                                    f"Using closest available point. "
                                    f"Extrapolation: linear (pandas interpolate, limit_direction='both'). "
                                    f"Available timepoints [h]: {avail_sorted}"
                                )
                                logger.warning(extrapolation_warnings[-1])
                                print(f"WARNING: {extrapolation_warnings[-1]}")
                result_df_mean = pd.DataFrame(summary_data_mean)
                result_df_std = pd.DataFrame(summary_data_std)
                timepoint_str = f" at {target_tp}"
            else:
                # Multiple timepoints: build (group, timepoint) columns
                summary_mean_dict = {}  # (group_label, tp_label) -> Series
                summary_std_dict = {}
                for target_tp in target_timepoints:
                    tp_label = f"{target_tp}h" if isinstance(target_tp, (int, float)) else str(target_tp)
                    for group_label, group_data in result_data.items():
                        group_df_mean = group_data['mean']
                        group_df_std = group_data['std']
                        if not group_df_mean.empty:
                            closest_idx, is_extrap, avail = _find_closest_and_check_extrapolation(
                                group_df_mean, target_tp
                            )
                            all_available_timepoints.update(avail)
                            if closest_idx is not None:
                                key = (group_label, tp_label)
                                summary_mean_dict[key] = group_df_mean.loc[closest_idx]
                                summary_std_dict[key] = group_df_std.loc[closest_idx]
                                if is_extrap:
                                    extrapolation_warnings.append(
                                        f"Timepoint {tp_label}: No data at this time. "
                                        f"Using closest available. Available [h]: {sorted(set(avail))}"
                                    )
                                    logger.warning(extrapolation_warnings[-1])
                                    print(f"WARNING: {extrapolation_warnings[-1]}")
                # Build MultiIndex columns: (group, timepoint)
                if summary_mean_dict:
                    result_df_mean = pd.DataFrame(summary_mean_dict)
                    result_df_std = pd.DataFrame(summary_std_dict)
                else:
                    result_df_mean = pd.DataFrame()
                    result_df_std = pd.DataFrame()
                timepoint_str = f" at {target_timepoints}"

        if result_df_mean.empty:
            print("No summary data could be calculated.")
            return pd.DataFrame()

        result_df_mean.index.name = condition
        result_df_std.index.name = condition

        # Store available timepoints, extrapolation warnings, and multi-timepoint flag
        result_df_mean.attrs['available_timepoints'] = sorted(all_available_timepoints)
        result_df_mean.attrs['extrapolation_warnings'] = extrapolation_warnings
        result_df_mean.attrs['multiple_timepoints'] = (
            target_timepoints is not None and len(target_timepoints) > 1
        )

        # Perform fitting if requested
        fit_results = {}
        if fit and not result_df_mean.empty:
            # Determine if condition is a cell line (cell number) or compound
            is_cell_line = condition in self.platemap.cell_lines
            is_compound = condition in self.platemap.compounds
            
            if is_cell_line:
                # Fit effective cell volume: V_spheroid = 4/3 * pi * r^3 * aspect_ratio = V_cell * n_cells
                fit_results = self._fit_cell_volume(
                    result_df_mean, result_df_std, condition, metric, **kwargs
                )
            elif is_compound:
                # Fit Hill curve: IC50 and Hill slope
                fit_results = self._fit_hill_curve(
                    result_df_mean, result_df_std, condition, metric, **kwargs
                )
            else:
                logger.warning(
                    f"Condition '{condition}' is neither a cell line nor a compound. "
                    f"Skipping fit. Available cell lines: {list(self.platemap.cell_lines.keys())}, "
                    f"Available compounds: {list(self.platemap.compounds.keys())}"
                )
        
        # Plot if requested
        if plot and not result_df_mean.empty:
            plt.figure(figsize=(10, 6))
            
            # Determine if condition values are logarithmic or linear based on spacing
            # Log-like: multiplicative spacing (ratios similar) - large gaps at large values
            # Linear: additive spacing (differences similar) - similar gap sizes
            condition_values = result_df_mean.index.values
            use_log_scale = False
            
            try:
                numeric_values = []
                for v in condition_values:
                    try:
                        if isinstance(v, (int, float)):
                            if not (np.isnan(v) or np.isinf(v)) and v > 0:
                                numeric_values.append(float(v))
                        elif isinstance(v, str):
                            val = float(v)
                            if val > 0:
                                numeric_values.append(val)
                    except (ValueError, TypeError):
                        continue
                
                if numeric_values and len(numeric_values) >= 3:
                    numeric_values = sorted(set(numeric_values))
                    # Consecutive ratios (multiplicative spacing)
                    ratios = [numeric_values[i + 1] / numeric_values[i]
                              for i in range(len(numeric_values) - 1)
                              if numeric_values[i] > 0]
                    # Consecutive differences (additive spacing)
                    diffs = [numeric_values[i + 1] - numeric_values[i]
                             for i in range(len(numeric_values) - 1)]
                    
                    if ratios and diffs:
                        mean_ratio = np.mean(ratios)
                        cv_ratios = np.std(ratios) / mean_ratio if mean_ratio > 0 else float('inf')
                        mean_diff = np.mean(np.abs(diffs))
                        cv_diffs = np.std(diffs) / mean_diff if mean_diff > 0 else float('inf')
                        # Log-like: ratios more uniform than differences, and ratios not all ~1
                        use_log_scale = (cv_ratios < cv_diffs and
                                         abs(mean_ratio - 1.0) > 0.1 and
                                         mean_ratio > 0)
            except Exception:
                use_log_scale = False

            # Check if we have multiple timepoints (tuple columns: (group, timepoint))
            has_multi_timepoint = result_df_mean.attrs.get('multiple_timepoints', False)
            markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
            colors = plt.cm.tab10(np.linspace(0, 1, 10))

            for col_idx, col in enumerate(result_df_mean.columns):
                mean_values = result_df_mean[col]
                std_values = result_df_std[col]
                
                # Legend label for multiple timepoints
                if has_multi_timepoint and len(col) == 2:
                    group_label, tp_label = col
                    legend_label = f"{group_label} @ {tp_label}"
                else:
                    legend_label = str(col) if has_multi_timepoint else None
                
                marker = markers[col_idx % len(markers)] if has_multi_timepoint else 'o'
                color = colors[col_idx % len(colors)] if has_multi_timepoint else None
                
                eb = plt.errorbar(
                    result_df_mean.index,
                    mean_values,
                    yerr=std_values,
                    fmt=marker,
                    markersize=8,
                    capsize=5,
                    capthick=2,
                    alpha=0.7,
                    label=legend_label if has_multi_timepoint else None,
                    color=color
                )
                
                if fit and col in fit_results:
                    fit_data = fit_results[col]
                    if 'x_fit' in fit_data and 'y_fit' in fit_data:
                        x_fit = fit_data['x_fit']
                        y_fit = fit_data['y_fit']
                        plt.plot(x_fit, y_fit, '--', alpha=0.7, linewidth=2,
                                color=color if color is not None else 'gray')

            timepoint_str = f" at {timepoint}" if timepoint is not None else " (mean across timepoints)"
            plt.xlabel(condition)
            plt.ylabel(metric)
            title = f'{metric} vs {condition}{timepoint_str} (mean ± std)'
            if fit and fit_results:
                # Add fit parameters to title
                fit_params_str = []
                for col, fit_data in fit_results.items():
                    if 'parameters' in fit_data:
                        params = fit_data['parameters']
                        if 'V_cell' in params:
                            fit_params_str.append(f"V_cell={params['V_cell']:.2e} μm³")
                        elif 'IC50' in params:
                            fit_params_str.append(f"IC50={params['IC50']:.2e}, Hill={params['Hill_slope']:.2f}")
                if fit_params_str:
                    title += f" | {', '.join(fit_params_str)}"
            plt.title(title)
            plt.grid(True, alpha=0.3)
            if has_multi_timepoint:
                plt.legend()
            if use_log_scale:
                plt.xscale('log')
            plt.tight_layout()
            plt.show()

        # Store fit results as DataFrame attribute
        if fit_results:
            result_df_mean.attrs['fit_results'] = fit_results

        # Return DataFrame with mean values (std can be accessed via result_df_std if needed)
        # For convenience, we could return a MultiIndex DataFrame, but for now return mean
        return result_df_mean
    
    def normalised_drug_response(
        self,
        metric_name: str = "radius",
        condition_name: str = None,
        pos_ctrl_name: str = None,
        time_period: str | None = None,
        plot: bool = True,
        fit: bool = False,
        ax: Optional[plt.Axes] = None,
    ) -> pd.DataFrame | tuple[pd.DataFrame, plt.Axes]:
        """
        Calculate normalised drug response (NDR) following the method from:
        Gupta et al. (2020) "A normalized drug response metric improves accuracy and 
        consistency of anticancer drug sensitivity quantification in cell-based screening"
        https://www.nature.com/articles/s42003-020-0765-z
        
        Implements the NDR formula from the paper (Methods section):
            NDR = max(-1, (1 - 2^(log2(FC_drug) / log2(FC_posCtrl))) / 
                          (1 - 2^(log2(FC_negCtrl) / log2(FC_posCtrl))))
        
        where:
            - FC_drug = fold change for drug-treated condition (endpoint/startpoint)
            - FC_negCtrl = fold change for negative control (concentration = 0)
            - FC_posCtrl = fold change for positive control (highest concentration)
        
        NDR values are bounded to [-1, 1] where:
            - NDR = -1: Complete cell death (stronger than positive control)
            - NDR = 0: No effect (same as negative control)
            - NDR = 1: Maximum effect (same as positive control)
        
        The negative control is automatically identified as the condition with value 0
        for the specified condition_name. The positive control is either:
        - The highest concentration of pos_ctrl_name (if specified)
        - The highest concentration of condition_name (if pos_ctrl_name is None)
        
        Note: NDR always uses fold changes (endpoint/startpoint). Requires at least 2 timepoints.
        
        Args:
            metric_name: Metric to use for calculation (default: 'radius')
                Can be any metric supported by SpheroidCollection.metric(), including
                fluorescence metrics like 'fluorescence_green_cumulative', 'fluorescence_green_mean', etc.
            condition_name: Name of the compound condition to analyze (required)
            pos_ctrl_name: str | None, default=None
                If specified, uses highest concentration of this compound as positive control.
                Otherwise uses highest concentration of condition_name.
            time_period: Optional time period name to limit analysis to specific time range
            plot: Whether to plot the results (default: True)
            fit: bool, default=False
                If True, calculates NDR for each timepoint and performs sigmoid fits.
                Returns MultiIndex DataFrame with (timepoint, concentration) and fit parameters.
            ax: Optional[plt.Axes], default=None
                Optional matplotlib Axes object to plot on. If provided and plot=True,
                the plot is drawn on this axes. If None and plot=True, a new figure is created.
        
        Returns:
            DataFrame with:
                - If fit=False:
                  * Index: Concentration values of the condition
                  * Columns: 'normalised_response', 'fold_change', 'metric_value', 'metric_value_std'
                - If fit=True:
                  * Index: MultiIndex (timepoint, concentration)
                  * Columns: 'normalised_response', 'fold_change', 'metric_value', 'metric_value_std', 
                    'fit_params' (dict with NDR_inf, EC50, h, NDR50)
                  * Also returns ax if plot=True: tuple[DataFrame, plt.Axes]
        """
        import matplotlib.pyplot as plt
        
        if not self.data_has_been_loaded:
            print("No data loaded for this replicate. Call load_images() first.")
            return pd.DataFrame()
        
        if condition_name is None:
            raise ValueError("condition_name must be specified")
        
        # Get all collections
        collections = self.get_collections()
        
        if not collections:
            print("No collections found. Ensure platemap is configured and images are loaded.")
            return pd.DataFrame()
        
        # Parse condition tuples to extract condition values
        condition_data = {}  # {condition_tuple: (condition_dict, collection)}
        
        for cond_tuple, collection in collections.items():
            cond_dict = {}
            if isinstance(cond_tuple, tuple):
                for item in cond_tuple:
                    if isinstance(item, tuple) and len(item) == 2:
                        name, value = item
                        cond_dict[name] = value
            condition_data[cond_tuple] = (cond_dict, collection)
        
        # Check if condition exists
        all_condition_names = set()
        for cond_dict, _ in condition_data.values():
            all_condition_names.update(cond_dict.keys())
        
        if condition_name not in all_condition_names:
            print(f"Condition '{condition_name}' not found in platemap. Available: {sorted(all_condition_names)}")
            return pd.DataFrame()
        
        # Find collections for the specified condition
        condition_collections = {}  # {concentration: collection}
        
        for cond_tuple, (cond_dict, collection) in condition_data.items():
            # Only include collections that have the condition we're analyzing
            if condition_name in cond_dict:
                concentration = cond_dict[condition_name]
                # Convert to float for consistent comparison
                try:
                    conc_float = float(concentration) if not isinstance(concentration, (int, float)) else float(concentration)
                    condition_collections[conc_float] = collection
                except (ValueError, TypeError):
                    # Skip non-numeric concentrations
                    logger.warning(f"Skipping non-numeric concentration: {concentration}")
                    continue
        
        if not condition_collections:
            print(f"No collections found for condition '{condition_name}'")
            return pd.DataFrame()
        
        # Find negative control (concentration = 0)
        neg_ctrl_concentration = None
        neg_ctrl_collection = None
        
        # Check for exact 0.0
        if 0.0 in condition_collections:
            neg_ctrl_concentration = 0.0
            neg_ctrl_collection = condition_collections[0.0]
        else:
            # Check for values very close to 0 (within tolerance)
            for conc in condition_collections.keys():
                if abs(conc) < 1e-6:
                    neg_ctrl_concentration = conc
                    neg_ctrl_collection = condition_collections[conc]
                    break
        
        if neg_ctrl_collection is None:
            print(f"Negative control (concentration = 0) not found for condition '{condition_name}'. "
                  f"Available concentrations: {sorted(condition_collections.keys())}")
            return pd.DataFrame()
        
        # Find positive control
        concentrations = sorted(condition_collections.keys())
        if not concentrations:
            print("No valid concentrations found")
            return pd.DataFrame()
        
        if pos_ctrl_name is not None:
            # Find collections with pos_ctrl_name
            pos_ctrl_collections = {}
            for cond_tuple, (cond_dict, collection) in condition_data.items():
                if pos_ctrl_name in cond_dict:
                    concentration = cond_dict[pos_ctrl_name]
                    try:
                        conc_float = float(concentration) if not isinstance(concentration, (int, float)) else float(concentration)
                        pos_ctrl_collections[conc_float] = collection
                    except (ValueError, TypeError):
                        logger.warning(f"Skipping non-numeric concentration for pos_ctrl: {concentration}")
                        continue
            
            if pos_ctrl_collections:
                pos_ctrl_concentration = max(pos_ctrl_collections.keys())
                pos_ctrl_collection = pos_ctrl_collections[pos_ctrl_concentration]
            else:
                # Fallback to highest concentration of condition_name
                logger.warning(f"Positive control '{pos_ctrl_name}' not found. Using highest concentration of '{condition_name}' as fallback.")
                pos_ctrl_concentration = concentrations[-1]
                pos_ctrl_collection = condition_collections[pos_ctrl_concentration]
        else:
            # Default: highest concentration of condition_name
            pos_ctrl_concentration = concentrations[-1]
            pos_ctrl_collection = condition_collections[pos_ctrl_concentration]
        
        # Helper function to extract metric values from DataFrame
        def extract_metric_values(metric_df):
            """Extract metric values from DataFrame, handling MultiIndex columns."""
            if metric_df.empty:
                return None, None
            
            # Handle MultiIndex columns
            if isinstance(metric_df.columns, pd.MultiIndex):
                # MultiIndex structure depends on metric_name and mean parameter:
                # - If mean=True: (collection_name, 'mean') and (collection_name, 'std')
                # - If metric_name='fluorescence_*' (without _mean/_cumulative): (collection_name, 'cumulative', 'mean') or similar
                # - If metric_name='fluorescence_*_cumulative' with mean=True: (collection_name, 'mean') and (collection_name, 'std')
                
                # Get level 1 values safely
                try:
                    if hasattr(metric_df.columns, 'levels') and len(metric_df.columns.levels) > 1:
                        level1_values = list(metric_df.columns.levels[1])
                    else:
                        level1_values = list(metric_df.columns.get_level_values(1).unique())
                except Exception:
                    level1_values = []
                
                # Try to extract 'mean' first (for aggregation when mean=True)
                if 'mean' in level1_values:
                    try:
                        values = metric_df.xs('mean', level=1, axis=1).iloc[:, 0]
                    except (KeyError, IndexError):
                        # Fallback: use first column
                        values = metric_df.iloc[:, 0]
                # If 'cumulative' is in level 1 (for fluorescence without _mean/_cumulative suffix)
                elif 'cumulative' in level1_values:
                    try:
                        values = metric_df.xs('cumulative', level=1, axis=1).iloc[:, 0]
                    except (KeyError, IndexError):
                        values = metric_df.iloc[:, 0]
                else:
                    # Fallback: use first column
                    values = metric_df.iloc[:, 0]
            else:
                # Simple column structure
                values = metric_df.iloc[:, 0]
            
            valid_values = values.dropna()
            if valid_values.empty:
                return None, None
            
            # Get std if available
            std_value = None
            if isinstance(metric_df.columns, pd.MultiIndex):
                try:
                    if hasattr(metric_df.columns, 'levels') and len(metric_df.columns.levels) > 1:
                        level1_values = list(metric_df.columns.levels[1])
                    else:
                        level1_values = list(metric_df.columns.get_level_values(1).unique())
                    
                    if 'std' in level1_values:
                        std_values = metric_df.xs('std', level=1, axis=1).iloc[:, 0]
                        std_valid = std_values.dropna()
                        if not std_valid.empty:
                            std_value = std_valid.iloc[-1]
                except (KeyError, IndexError, Exception):
                    pass
            
            return valid_values, std_value
        
        # Helper function to get fold change per timepoint
        def get_fold_change_per_timepoint(collection, metric_name, time_period):
            """Get fold changes for all timepoints relative to baseline (first timepoint).
            
            Returns dict mapping timepoint -> (FC, std_FC)
            """
            try:
                metric_df = collection.metric(
                    name=metric_name,
                    mean=True,
                    time_period=time_period,
                    plot=False
                )
                
                values, _ = extract_metric_values(metric_df)
                if values is None or len(values) < 2:
                    return {}
                
                # Get std values if available
                std_values = None
                if isinstance(metric_df.columns, pd.MultiIndex):
                    try:
                        if hasattr(metric_df.columns, 'levels') and len(metric_df.columns.levels) > 1:
                            level1_values = list(metric_df.columns.levels[1])
                        else:
                            level1_values = list(metric_df.columns.get_level_values(1).unique())
                        
                        if 'std' in level1_values:
                            std_series = metric_df.xs('std', level=1, axis=1).iloc[:, 0]
                            std_valid = std_series.dropna()
                            if not std_valid.empty:
                                std_values = std_valid
                    except (KeyError, IndexError, Exception):
                        pass
                
                baseline = values.iloc[0]
                if baseline <= 0:
                    return {}
                
                timepoints = values.index.values
                fc_dict = {}
                
                for i, (t, val) in enumerate(values.items()):
                    if val <= 0:
                        continue
                    
                    fc = val / baseline
                    
                    # Calculate uncertainty
                    std_fc = None
                    if std_values is not None and i < len(std_values):
                        std_val = std_values.iloc[i] if hasattr(std_values, 'iloc') else std_values[i] if i < len(std_values) else None
                        std_baseline = std_values.iloc[0] if hasattr(std_values, 'iloc') else std_values[0] if len(std_values) > 0 else None
                        
                        if std_val is not None and std_baseline is not None:
                            # Error propagation: std_FC ≈ FC * sqrt((std_val/val)^2 + (std_baseline/baseline)^2)
                            rel_error_val = (std_val / val) ** 2 if val > 0 else 0
                            rel_error_baseline = (std_baseline / baseline) ** 2 if baseline > 0 else 0
                            std_fc = fc * np.sqrt(rel_error_val + rel_error_baseline)
                    
                    fc_dict[t] = (fc, std_fc)
                
                return fc_dict
            except Exception as e:
                logger.warning(f"Could not calculate fold changes per timepoint: {e}")
                return {}
        
        # Helper function to get fold change (endpoint/startpoint) - for backward compatibility
        def get_fold_change(collection, metric_name, time_period):
            """Get fold change (endpoint/startpoint) for a collection.
            
            Returns fold change and its uncertainty using error propagation.
            """
            try:
                metric_df = collection.metric(
                    name=metric_name,
                    mean=True,
                    time_period=time_period,
                    plot=False
                )
                
                values, std_endpoint = extract_metric_values(metric_df)
                if values is None or len(values) < 2:
                    return None, None, None
                
                startpoint = values.iloc[0]
                endpoint = values.iloc[-1]
                
                if startpoint <= 0:
                    return None, None, None
                
                fold_change = endpoint / startpoint
                
                # Calculate uncertainty in fold change using error propagation
                # FC = endpoint / startpoint
                # std_FC = FC * sqrt((std_end/end)^2 + (std_start/start)^2)
                # For now, we only have std_endpoint (std of endpoint), so we approximate:
                # If we don't have std_start, we can use std_endpoint as approximation
                # or calculate it from the first timepoint if available
                
                # Try to get std for startpoint
                std_startpoint = None
                if isinstance(metric_df.columns, pd.MultiIndex):
                    try:
                        if hasattr(metric_df.columns, 'levels') and len(metric_df.columns.levels) > 1:
                            level1_values = list(metric_df.columns.levels[1])
                        else:
                            level1_values = list(metric_df.columns.get_level_values(1).unique())
                        
                        if 'std' in level1_values:
                            std_series = metric_df.xs('std', level=1, axis=1).iloc[:, 0]
                            std_valid = std_series.dropna()
                            if not std_valid.empty and len(std_valid) > 0:
                                std_startpoint = std_valid.iloc[0]  # std of first timepoint
                    except (KeyError, IndexError, Exception):
                        pass
                
                # Calculate fold change uncertainty
                if std_endpoint is not None:
                    # Error propagation for FC = endpoint / startpoint
                    # std_FC ≈ FC * sqrt((std_end/end)^2 + (std_start/start)^2)
                    if std_startpoint is not None:
                        # Both uncertainties available
                        rel_error_end = (std_endpoint / endpoint) ** 2 if endpoint > 0 else 0
                        rel_error_start = (std_startpoint / startpoint) ** 2 if startpoint > 0 else 0
                        std_fold_change = fold_change * np.sqrt(rel_error_end + rel_error_start)
                    else:
                        # Only endpoint uncertainty available - approximate
                        # Use relative error of endpoint as approximation
                        rel_error = std_endpoint / endpoint if endpoint > 0 else 0
                        std_fold_change = fold_change * rel_error
                else:
                    std_fold_change = None
                
                return fold_change, startpoint, std_fold_change
            except Exception as e:
                logger.warning(f"Could not calculate fold change: {e}")
                return None, None, None
        
        # Helper function to calculate NDR from fold changes
        def calculate_ndr(fc_drug, fc_neg, fc_pos):
            """Calculate NDR using the formula from Gupta et al. (2020)."""
            # Avoid division by zero and invalid logarithms
            if fc_drug <= 0 or fc_neg <= 0 or fc_pos <= 0:
                return np.nan
            
            # Calculate log2 values
            log2_fc_drug = np.log2(fc_drug)
            log2_fc_neg = np.log2(fc_neg)
            log2_fc_pos = np.log2(fc_pos)
            
            # Avoid division by zero in log2 ratios
            if abs(log2_fc_pos) < 1e-10:
                return np.nan
            
            # Calculate numerator: 1 - 2^(log2(FC_drug) / log2(FC_pos))
            numerator = 1.0 - (2.0 ** (log2_fc_drug / log2_fc_pos))
            
            # Calculate denominator: 1 - 2^(log2(FC_neg) / log2(FC_pos))
            denominator = 1.0 - (2.0 ** (log2_fc_neg / log2_fc_pos))
            
            # Avoid division by zero
            if abs(denominator) < 1e-10:
                return np.nan
            
            # Calculate NDR and apply max(-1, ...) constraint
            ndr_value = numerator / denominator
            return max(-1.0, ndr_value)
        
        # If fit=True, calculate NDR for each timepoint and perform sigmoid fits
        if fit:
            return self._normalised_drug_response_with_fit(
                metric_name, condition_name, pos_ctrl_name, time_period,
                neg_ctrl_collection, pos_ctrl_collection, condition_collections,
                concentrations, pos_ctrl_concentration,
                calculate_ndr, get_fold_change_per_timepoint, plot, ax
            )
        
        # Original implementation: endpoint only
        # NDR always uses fold changes (endpoint/startpoint)
        neg_fc, neg_start, neg_ctrl_std = get_fold_change(neg_ctrl_collection, metric_name, time_period)
        pos_fc, pos_start, pos_ctrl_std = get_fold_change(pos_ctrl_collection, metric_name, time_period)
        
        if neg_fc is None or pos_fc is None:
            print("Could not calculate fold changes for controls. Ensure at least 2 timepoints are available.")
            return pd.DataFrame()
        
        # Check for valid fold changes (must be positive)
        if neg_fc <= 0 or pos_fc <= 0:
            print(f"Invalid fold changes: neg_ctrl={neg_fc}, pos_ctrl={pos_fc}. Fold changes must be positive.")
            return pd.DataFrame()
        
        # Calculate NDR for each concentration using the correct formula from the paper
        # NDR = max(-1, (1 - 2^(log2(FC_drug) / log2(FC_posCtrl))) / 
        #              (1 - 2^(log2(FC_negCtrl) / log2(FC_posCtrl))))
        result_data = []
        
        # Process all concentrations of condition_name
        for concentration in concentrations:
            collection = condition_collections[concentration]
            try:
                # Get fold change for this concentration
                fc, start, metric_std = get_fold_change(collection, metric_name, time_period)
                
                if fc is None or fc <= 0:
                    logger.warning(f"Could not calculate valid fold change for concentration {concentration}")
                    continue
                
                # Calculate NDR using the correct formula
                ndr_value = calculate_ndr(fc, neg_fc, pos_fc)
                
                result_data.append({
                    'concentration': concentration,
                    'normalised_response': ndr_value,
                    'fold_change': fc,
                    'metric_value': fc,  # Store fold change as metric_value for consistency
                    'metric_value_std': metric_std if metric_std is not None else np.nan
                })
            except Exception as e:
                logger.warning(f"Could not calculate NDR for concentration {concentration}: {e}")
                continue
        
        # Also add positive control if it's a different compound (so user can see its fold change)
        if pos_ctrl_name is not None and pos_ctrl_name != condition_name:
            try:
                # Calculate NDR for positive control (should be 1.0 by definition)
                pos_ndr = calculate_ndr(pos_fc, neg_fc, pos_fc)
                result_data.append({
                    'concentration': f"{pos_ctrl_name}_{pos_ctrl_concentration}",
                    'normalised_response': pos_ndr,  # Should be 1.0 by definition
                    'fold_change': pos_fc,
                    'metric_value': pos_fc,
                    'metric_value_std': pos_ctrl_std if pos_ctrl_std is not None else np.nan
                })
            except Exception as e:
                logger.warning(f"Could not add positive control to results: {e}")
        
        if not result_data:
            print("No metric data could be calculated")
            return pd.DataFrame()
        
        result_df = pd.DataFrame(result_data)
        result_df = result_df.set_index('concentration')
        
        # Plot if requested
        if plot and not result_df.empty:
            plt.figure(figsize=(10, 6))
            
            # Convert concentration index to numeric for plotting
            concentrations_plot = []
            for conc in result_df.index:
                try:
                    concentrations_plot.append(float(conc))
                except (ValueError, TypeError):
                    concentrations_plot.append(0.0)
            
            # Determine if concentrations are log-spaced
            use_log_scale = False
            if len(concentrations_plot) >= 3:
                concentrations_plot_sorted = sorted(set([c for c in concentrations_plot if c > 0]))
                if len(concentrations_plot_sorted) >= 3:
                    ratios = [concentrations_plot_sorted[i+1] / concentrations_plot_sorted[i]
                             for i in range(len(concentrations_plot_sorted)-1)
                             if concentrations_plot_sorted[i] > 0]
                    diffs = [concentrations_plot_sorted[i+1] - concentrations_plot_sorted[i]
                            for i in range(len(concentrations_plot_sorted)-1)]
                    if ratios and diffs:
                        cv_ratios = np.std(ratios) / np.mean(ratios) if np.mean(ratios) > 0 else float('inf')
                        cv_diffs = np.std(diffs) / np.mean(np.abs(diffs)) if np.mean(np.abs(diffs)) > 0 else float('inf')
                        use_log_scale = (cv_ratios < cv_diffs and abs(np.mean(ratios) - 1.0) > 0.1)
            
            # Plot normalised response with error bars if available
            valid_mask = ~np.isnan(result_df['normalised_response'])
            if np.any(valid_mask):
                valid_conc = [concentrations_plot[i] for i in range(len(concentrations_plot)) if valid_mask.iloc[i]]
                valid_resp = result_df['normalised_response'][valid_mask].values
                
                # Check if we have std values for error bars
                has_std = 'metric_value_std' in result_df.columns
                if has_std:
                    valid_std = result_df['metric_value_std'][valid_mask].values
                    # Propagate error through NDR formula
                    # NDR = max(-1, (1 - 2^(log2(FC_drug)/log2(FC_pos))) / (1 - 2^(log2(FC_neg)/log2(FC_pos))))
                    # For error propagation, we approximate using relative errors
                    if 'fold_change' in result_df.columns:
                        fold_changes = result_df['fold_change'][valid_mask].values
                        # Avoid division by zero
                        fold_changes = np.maximum(fold_changes, 1e-10)
                        # Approximate error propagation: std_NDR ≈ (std_FC / FC) * sensitivity
                        # The sensitivity depends on the NDR formula, but for small errors we can approximate
                        # using the relative error of fold changes
                        relative_errors = valid_std / fold_changes
                        # Scale by a factor to account for the NDR formula complexity
                        # This is a simplified approximation
                        error_bars = relative_errors * 0.5  # Approximate scaling factor
                    else:
                        error_bars = None
                else:
                    error_bars = None
                
                if error_bars is not None and not np.all(np.isnan(error_bars)):
                    plt.errorbar(valid_conc, valid_resp, yerr=error_bars, 
                               fmt='o-', markersize=8, linewidth=2, capsize=5,
                               label='Normalised response', alpha=0.8)
                else:
                    plt.plot(valid_conc, valid_resp, 'o-', markersize=8, linewidth=2, 
                           label='Normalised response', alpha=0.8)
                
                plt.axhline(y=-1, color='gray', linestyle='--', alpha=0.5, label='Complete death (-1)')
                plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Negative control (0)')
                plt.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='Positive control (1)')
            
            plt.xlabel(f'{condition_name} concentration', fontsize=12)
            plt.ylabel('Normalised Drug Response (NDR)', fontsize=12)
            
            # Title with correct NDR formula from paper
            title = f'Normalised Drug Response: {condition_name}\n'
            title += r'(NDR = max(-1, (1 - 2^(log₂(FC_drug)/log₂(FC_pos))) / (1 - 2^(log₂(FC_neg)/log₂(FC_pos)))))'
            
            plt.title(title, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.legend()
            if use_log_scale:
                plt.xscale('log')
            plt.tight_layout()
            plt.show()
        
        return result_df
    
    def _normalised_drug_response_with_fit(
        self,
        metric_name: str,
        condition_name: str,
        pos_ctrl_name: str | None,
        time_period: str | None,
        neg_ctrl_collection,
        pos_ctrl_collection,
        condition_collections: dict,
        concentrations: list,
        pos_ctrl_concentration: float,
        calculate_ndr,
        get_fold_change_per_timepoint,
        plot: bool,
        ax: Optional[plt.Axes],
    ) -> pd.DataFrame | tuple[pd.DataFrame, plt.Axes]:
        """Calculate NDR for each timepoint and perform sigmoid fits.
        
        This is the extended version that calculates NDR for all timepoints
        and fits sigmoid curves per timepoint.
        """
        import seaborn as sns
        from scipy.optimize import curve_fit
        
        # Get fold changes per timepoint for controls
        neg_fc_dict = get_fold_change_per_timepoint(neg_ctrl_collection, metric_name, time_period)
        pos_fc_dict = get_fold_change_per_timepoint(pos_ctrl_collection, metric_name, time_period)
        
        if not neg_fc_dict or not pos_fc_dict:
            print("Could not calculate fold changes per timepoint for controls. Ensure at least 2 timepoints are available.")
            return pd.DataFrame()
        
        # Get common timepoints (exclude baseline - first timepoint has FC=1 by definition)
        all_timepoints = sorted(neg_fc_dict.keys())
        if len(all_timepoints) < 2:
            print("Need at least 2 timepoints (baseline + at least one measurement).")
            return pd.DataFrame()
        
        timepoints = all_timepoints[1:]  # Exclude first timepoint (baseline)
        if not timepoints:
            print("No valid timepoints found (excluding baseline).")
            return pd.DataFrame()
        
        # Cache fold changes per concentration (performance optimization)
        # Call get_fold_change_per_timepoint once per concentration, not per timepoint
        fc_dict_cache = {}  # {concentration: {timepoint: (fc, std_fc)}}
        for concentration in concentrations:
            collection = condition_collections[concentration]
            fc_dict = get_fold_change_per_timepoint(collection, metric_name, time_period)
            fc_dict_cache[concentration] = fc_dict
        
        # Build NDR matrix: NDR[timepoint, concentration]
        ndr_matrix = {}  # {timepoint: {concentration: ndr_value}}
        fold_change_matrix = {}  # {timepoint: {concentration: (fc, std_fc)}}
        
        for t in timepoints:
            neg_fc, neg_std = neg_fc_dict[t]
            pos_fc, pos_std = pos_fc_dict[t]
            
            if neg_fc <= 0 or pos_fc <= 0:
                continue
            
            ndr_matrix[t] = {}
            fold_change_matrix[t] = {}
            
            # Calculate NDR for each concentration at this timepoint
            for concentration in concentrations:
                fc_dict = fc_dict_cache[concentration]
                
                if t in fc_dict:
                    fc, fc_std = fc_dict[t]
                    if fc > 0:
                        ndr_value = calculate_ndr(fc, neg_fc, pos_fc)
                        ndr_matrix[t][concentration] = ndr_value
                        fold_change_matrix[t][concentration] = (fc, fc_std)
        
        # Perform sigmoid fits per timepoint
        # NDR(c) = NDR_inf + (1 - NDR_inf) / (1 + (c / EC50)^h)
        def sigmoid_ndr(conc, ndr_inf, ec50, h):
            """3-parameter Hill sigmoid for NDR."""
            return ndr_inf + (1.0 - ndr_inf) / (1.0 + (conc / ec50) ** h)
        
        fit_params = {}  # {timepoint: {'NDR_inf': ..., 'EC50': ..., 'h': ..., 'NDR50': ...}}
        
        for t in timepoints:
            if t not in ndr_matrix or not ndr_matrix[t]:
                continue
            
            # Get data for this timepoint
            concs = []
            ndr_vals = []
            
            for c in sorted(concentrations):
                if c in ndr_matrix[t] and not np.isnan(ndr_matrix[t][c]) and c > 0:
                    concs.append(c)
                    ndr_vals.append(ndr_matrix[t][c])
            
            if len(concs) < 3:  # Need at least 3 points for 3-parameter fit
                continue
            
            concs = np.array(concs)
            ndr_vals = np.array(ndr_vals)
            
            # Initial guesses
            ndr_inf_guess = np.mean(ndr_vals[-3:])  # Average of highest concentrations
            ec50_guess = np.median(concs)
            h_guess = 1.0
            
            # Bounds: NDR_inf ∈ [-1, 1], EC50 > 0, h ∈ [0.1, 5]
            bounds = ([-1.0, 1e-6, 0.1], [1.0, np.inf, 5.0])
            
            try:
                popt, pcov = curve_fit(
                    sigmoid_ndr,
                    concs,
                    ndr_vals,
                    p0=[ndr_inf_guess, ec50_guess, h_guess],
                    bounds=bounds,
                    maxfev=5000
                )
                
                ndr_inf, ec50, h = popt
                
                # Calculate NDR50 (concentration where NDR = 0.5)
                # NDR(c) = 0.5 = NDR_inf + (1 - NDR_inf) / (1 + (c / EC50)^h)
                # Solve for c: 0.5 - NDR_inf = (1 - NDR_inf) / (1 + (c / EC50)^h)
                # (1 + (c / EC50)^h) = (1 - NDR_inf) / (0.5 - NDR_inf)
                # (c / EC50)^h = (1 - NDR_inf) / (0.5 - NDR_inf) - 1
                # c = EC50 * (((1 - NDR_inf) / (0.5 - NDR_inf) - 1)^(1/h))
                if abs(0.5 - ndr_inf) > 1e-10:
                    ratio = (1.0 - ndr_inf) / (0.5 - ndr_inf) - 1.0
                    if ratio > 0 and h > 0:
                        ndr50 = ec50 * (ratio ** (1.0 / h))
                    else:
                        ndr50 = np.nan
                else:
                    ndr50 = np.nan
                
                fit_params[t] = {
                    'NDR_inf': float(ndr_inf),
                    'EC50': float(ec50),
                    'h': float(h),
                    'NDR50': float(ndr50) if not np.isnan(ndr50) else np.nan
                }
            except Exception as e:
                logger.warning(f"Could not fit sigmoid for timepoint {t}: {e}")
                continue
        
        # Build result DataFrame with MultiIndex
        result_data = []
        for t in timepoints:
            for c in concentrations:
                if t in ndr_matrix and c in ndr_matrix[t]:
                    fc, fc_std = fold_change_matrix[t].get(c, (np.nan, np.nan))
                    fit_dict = fit_params.get(t, {})
                    
                    result_data.append({
                        'timepoint': t,
                        'concentration': c,
                        'normalised_response': ndr_matrix[t][c],
                        'fold_change': fc,
                        'metric_value': fc,
                        'metric_value_std': fc_std if (fc_std is not None and not np.isnan(fc_std)) else np.nan,
                        'fit_params': fit_dict if fit_dict else None
                    })
        
        if not result_data:
            print("No NDR data could be calculated for any timepoint.")
            return pd.DataFrame()
        
        result_df = pd.DataFrame(result_data)
        result_df = result_df.set_index(['timepoint', 'concentration'])
        
        # Plot if requested
        if plot:
            ax = self._plot_ndr_with_fit(
                ndr_matrix, fit_params, concentrations, timepoints, condition_name, ax
            )
            return result_df, ax
        
        return result_df
    
    def _plot_ndr_with_fit(
        self,
        ndr_matrix: dict,
        fit_params: dict,
        concentrations: list,
        timepoints: list,
        condition_name: str,
        ax: Optional[plt.Axes] = None
    ) -> plt.Axes:
        """Plot NDR data with sigmoid fits, following the specified style."""
        import seaborn as sns
        
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        else:
            fig = ax.figure
        
        # Color palette: one color per timepoint
        palette = sns.color_palette("rocket", len(timepoints))
        
        # X-axis: log-scaled, extended xlim
        conc_nonzero = [c for c in concentrations if c > 0]
        if not conc_nonzero:
            return ax
        
        x_lo = np.log10(min(conc_nonzero)) - 1.5
        x_hi = np.log10(max(conc_nonzero)) + 1.5
        conc_fine = np.logspace(x_lo, x_hi, 200)
        
        # Y-axis limits
        ymin, ymax = -1.15, 1.15
        ax.set_ylim(ymin, ymax)
        ax.set_xlim(10**x_lo, 10**x_hi)
        ax.set_xscale('log')
        
        # 1. Background gradient: green above y=0, red below
        # Use imshow for better performance (single object instead of 512 axhspan calls)
        import matplotlib.colors as mcolors
        
        n = 256
        
        # Green gradient above y=0
        grad_green = np.linspace(0.005, 0.05, n).reshape(n, 1)
        rgba_green = np.zeros((n, 1, 4))
        rgba_green[..., 1] = 0.6  # Green channel
        rgba_green[..., 3] = grad_green  # Alpha channel
        ax.imshow(
            rgba_green,
            extent=[10**x_lo, 10**x_hi, 0, ymax],
            origin='lower',
            aspect='auto',
            zorder=0,
            transform=ax.transData  # Important for log scale
        )
        
        # Red gradient below y=0
        grad_red = np.linspace(0.08, 0.01, n).reshape(n, 1)
        rgba_red = np.zeros((n, 1, 4))
        rgba_red[..., 0] = 0.6  # Red channel
        rgba_red[..., 3] = grad_red  # Alpha channel
        ax.imshow(
            rgba_red,
            extent=[10**x_lo, 10**x_hi, ymin, 0],
            origin='lower',
            aspect='auto',
            zorder=0,
            transform=ax.transData  # Important for log scale
        )
        
        # 2. Reference lines
        ax.axhline(0, color='gray', linestyle='--', linewidth=2.5, zorder=1)
        ax.axhline(1, color='gray', linestyle=':', linewidth=1.0, zorder=1)
        
        # 3. Data points and fit curves per timepoint
        def sigmoid_ndr(conc, ndr_inf, ec50, h):
            return ndr_inf + (1.0 - ndr_inf) / (1.0 + (conc / ec50) ** h)
        
        for i, t in enumerate(timepoints):
            if t not in ndr_matrix:
                continue
            
            # Data points
            concs_t = [c for c in conc_nonzero if c in ndr_matrix[t] and not np.isnan(ndr_matrix[t][c])]
            ndr_vals = [ndr_matrix[t][c] for c in concs_t]
            
            if concs_t and ndr_vals:
                ax.scatter(
                    concs_t, ndr_vals,
                    color=palette[i],
                    s=50,
                    alpha=0.7,
                    zorder=3,
                    label=f'{t:.0f} h'
                )
            
            # Fit curve
            if t in fit_params:
                params = fit_params[t]
                ndr_inf = params['NDR_inf']
                ec50 = params['EC50']
                h = params['h']
                
                ndr_curve = sigmoid_ndr(conc_fine, ndr_inf, ec50, h)
                ax.plot(
                    conc_fine, ndr_curve,
                    '-',
                    color=palette[i],
                    linewidth=2,
                    zorder=4
                )
        
        # 4. Labels and legend
        ax.legend(fontsize=10, title='Datapoints')
        ax.set_xlabel('Concentration [µM]', fontsize=13)
        ax.set_ylabel('NDR', fontsize=13)
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def _fit_cell_volume(self, result_df_mean: pd.DataFrame, result_df_std: pd.DataFrame,
                        condition: str, metric: str, **kwargs) -> dict:
        """Fit effective cell volume from spheroid volume and cell number.
        
        Fits: V_spheroid = 4/3 * pi * r^3 * aspect_ratio = V_cell * n_cells
        Solves for V_cell (effective cell volume).
        
        Args:
            result_df_mean: DataFrame with mean metric values (condition values as index)
            result_df_std: DataFrame with std metric values
            condition: Name of the condition (cell line)
            metric: Metric name (should be 'radius' or 'area')
            **kwargs: Additional parameters:
                - aspect_ratio: Aspect ratio for spheroid (default: 1.0)
        
        Returns:
            Dictionary mapping column names to fit results with keys:
                - 'parameters': Dict with 'V_cell' and other fit parameters
                - 'x_fit': Array of x values for fit curve
                - 'y_fit': Array of y values for fit curve
                - 'r2': R-squared value
        """
        from scipy.optimize import curve_fit
        
        aspect_ratio = kwargs.get('aspect_ratio', 1.0)
        fit_results = {}
        
        for col in result_df_mean.columns:
            # Get condition values (cell numbers) and metric values
            x_data = result_df_mean.index.values  # Cell numbers
            y_data = result_df_mean[col].values   # Metric values (radius or area)
            
            # Filter out NaN values
            valid_mask = ~np.isnan(x_data) & ~np.isnan(y_data)
            x_valid = x_data[valid_mask]
            y_valid = y_data[valid_mask]
            
            if len(x_valid) < 2:
                logger.warning(f"Not enough valid data points for fitting column '{col}'")
                continue
            
            # Convert metric to volume based on metric type
            if metric == 'radius':
                # V_spheroid = 4/3 * pi * r^3 * aspect_ratio
                volumes = (4/3) * np.pi * (y_valid ** 3) * aspect_ratio
            elif metric == 'area':
                # For area, we need to estimate radius first: A = pi * r^2, so r = sqrt(A/pi)
                # Then V = 4/3 * pi * r^3 * aspect_ratio
                radii = np.sqrt(y_valid / np.pi)
                volumes = (4/3) * np.pi * (radii ** 3) * aspect_ratio
            else:
                logger.warning(f"Metric '{metric}' not supported for cell volume fitting. Use 'radius' or 'area'.")
                continue
            
            # Fit: V_spheroid = V_cell * n_cells
            # Linear fit through origin: V = V_cell * n
            def linear_through_origin(n, V_cell):
                return V_cell * n
            
            try:
                # Fit with initial guess
                V_cell_initial = volumes[-1] / x_valid[-1] if x_valid[-1] > 0 else 1.0
                popt, pcov = curve_fit(
                    linear_through_origin,
                    x_valid,
                    volumes,
                    p0=[V_cell_initial],
                    bounds=(0, np.inf)
                )
                
                V_cell = popt[0]
                
                # Calculate R²
                y_pred = linear_through_origin(x_valid, V_cell)
                ss_res = np.sum((volumes - y_pred) ** 2)
                ss_tot = np.sum((volumes - np.mean(volumes)) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
                
                # Generate fit curve for plotting
                x_fit = np.linspace(x_valid.min(), x_valid.max(), 100)
                if metric == 'radius':
                    # Solve for radius: r = (V_cell * n / (4/3 * pi * aspect_ratio))^(1/3)
                    volumes_fit = linear_through_origin(x_fit, V_cell)
                    y_fit = (volumes_fit / ((4/3) * np.pi * aspect_ratio)) ** (1/3)
                else:  # area
                    # Solve for area: A = pi * r^2, where r = (V_cell * n / (4/3 * pi * aspect_ratio))^(1/3)
                    volumes_fit = linear_through_origin(x_fit, V_cell)
                    radii_fit = (volumes_fit / ((4/3) * np.pi * aspect_ratio)) ** (1/3)
                    y_fit = np.pi * (radii_fit ** 2)
                
                fit_results[col] = {
                    'parameters': {
                        'V_cell': float(V_cell),
                        'aspect_ratio': float(aspect_ratio),
                        'covariance': pcov.tolist()
                    },
                    'x_fit': x_fit,
                    'y_fit': y_fit,
                    'r2': float(r2)
                }
                
                logger.info(f"Cell volume fit for '{col}': V_cell = {V_cell:.2e} μm³, R² = {r2:.3f}")
                
            except Exception as e:
                logger.warning(f"Failed to fit cell volume for column '{col}': {e}")
                continue
        
        return fit_results
    
    def _fit_hill_curve(self, result_df_mean: pd.DataFrame, result_df_std: pd.DataFrame,
                       condition: str, metric: str, **kwargs) -> dict:
        """Fit Hill curve for compound dose-response data.
        
        Fits: response = bottom + (top - bottom) / (1 + (IC50 / concentration)^Hill_slope)
        
        Args:
            result_df_mean: DataFrame with mean metric values (concentrations as index)
            result_df_std: DataFrame with std metric values
            condition: Name of the condition (compound)
            metric: Metric name
            **kwargs: Additional parameters for fitting:
                - bottom: Minimum response (default: estimated from data)
                - top: Maximum response (default: estimated from data)
                - IC50_initial: Initial guess for IC50 (default: median concentration)
                - Hill_initial: Initial guess for Hill slope (default: 1.0)
        
        Returns:
            Dictionary mapping column names to fit results with keys:
                - 'parameters': Dict with 'IC50', 'Hill_slope', 'bottom', 'top', 'r2'
                - 'x_fit': Array of x values for fit curve
                - 'y_fit': Array of y values for fit curve
        """
        from scipy.optimize import curve_fit
        
        fit_results = {}
        
        for col in result_df_mean.columns:
            # Get concentration values and metric values
            x_data = result_df_mean.index.values  # Concentrations
            y_data = result_df_mean[col].values    # Metric values
            
            # Filter out NaN values and ensure positive concentrations
            valid_mask = ~np.isnan(x_data) & ~np.isnan(y_data) & (x_data > 0)
            x_valid = x_data[valid_mask]
            y_valid = y_data[valid_mask]
            
            if len(x_valid) < 3:
                logger.warning(f"Not enough valid data points for Hill curve fitting column '{col}'")
                continue
            
            # Hill curve function
            def hill_curve(conc, IC50, Hill_slope, bottom, top):
                """Hill curve: response = bottom + (top - bottom) / (1 + (IC50/conc)^Hill)"""
                return bottom + (top - bottom) / (1 + (IC50 / conc) ** Hill_slope)
            
            # Get initial guesses and bounds
            bottom_initial = kwargs.get('bottom', np.min(y_valid))
            top_initial = kwargs.get('top', np.max(y_valid))
            IC50_initial = kwargs.get('IC50_initial', np.median(x_valid))
            Hill_initial = kwargs.get('Hill_initial', 1.0)
            
            # Ensure reasonable bounds
            p0 = [IC50_initial, Hill_initial, bottom_initial, top_initial]
            bounds = (
                [x_valid.min() * 0.01, 0.1, -np.inf, -np.inf],  # Lower bounds
                [x_valid.max() * 100, 10.0, np.inf, np.inf]      # Upper bounds
            )
            
            try:
                popt, pcov = curve_fit(
                    hill_curve,
                    x_valid,
                    y_valid,
                    p0=p0,
                    bounds=bounds,
                    maxfev=5000
                )
                
                IC50, Hill_slope, bottom, top = popt
                
                # Ensure values are scalars
                IC50 = float(IC50)
                Hill_slope = float(Hill_slope)
                bottom = float(bottom)
                top = float(top)
                
                # Calculate R²
                y_pred = hill_curve(x_valid, IC50, Hill_slope, bottom, top)
                ss_res = np.sum((y_valid - y_pred) ** 2)
                ss_tot = np.sum((y_valid - np.mean(y_valid)) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
                
                # Generate fit curve for plotting
                x_fit = np.logspace(
                    np.log10(x_valid.min() * 0.1),
                    np.log10(x_valid.max() * 10),
                    100
                )
                y_fit = hill_curve(x_fit, IC50, Hill_slope, bottom, top)
                
                # Handle covariance matrix - ensure it's 2D
                if pcov.ndim == 1:
                    # If 1D, create a diagonal matrix
                    pcov_2d = np.diag(pcov)
                else:
                    pcov_2d = pcov
                
                fit_results[col] = {
                    'parameters': {
                        'IC50': IC50,
                        'Hill_slope': Hill_slope,
                        'bottom': bottom,
                        'top': top,
                        'r2': float(r2),
                        'covariance': pcov_2d.tolist()
                    },
                    'x_fit': x_fit,
                    'y_fit': y_fit
                }
                
                logger.info(
                    f"Hill curve fit for '{col}': IC50 = {IC50:.2e}, "
                    f"Hill slope = {Hill_slope:.2f}, R² = {r2:.3f}"
                )
                
            except Exception as e:
                logger.warning(f"Failed to fit Hill curve for column '{col}': {e}")
                continue
        
        return fit_results

    def metric(
        self,
        name: str = "radius",
        mean: bool = False,
        average: bool = False,
        ignore_border: bool = True,
        plot: bool = False,
        ax: Optional[plt.Axes] = None,
        plot_kwargs: Optional[dict] = None,
        ax_callback: Optional[Callable[[plt.Axes], None]] = None,
        savepath: Optional[str] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Calculate a metric across all collections in this replicate.
        
        This method calculates metrics for all collections that belong to this replicate.
        Each collection represents one condition (same platemap values); all spheroids
        with the same conditions are averaged together within that collection.
        
        Parameters
        ----------
        name : str, default="radius"
            Metric name to calculate. Available options:
            - Basic metrics: 'radius', 'area'
            - Fluorescence metrics: 'fluorescence_green', 'fluorescence_red', 'fluorescence_blue'
            - Fluorescence with specific type: 'fluorescence_green_mean', 'fluorescence_green_cumulative', etc.
            
            Use 'radius' for effective radius (sqrt(Area/π)) in μm, 'area' for area in μm².
            For fluorescence, use base name (e.g., 'fluorescence_green') to get both cumulative and mean,
            or add '_mean'/'_cumulative' suffix to get only one type.
        
        mean : bool, default=False
            If True, calculate mean ± std *per condition* (each collection = one condition).
            This aggregates across biological replicates (collections with identical condition values).
            When True, returns MultiIndex columns: (condition_label, 'mean') and (condition_label, 'std').
            When False, returns individual series or collections (depending on ``average``).
            Overrides ``average`` parameter when True.
        
        average : bool, default=False
            If True and ``mean=False``, average technical replicates within each collection.
            This only affects within-collection aggregation, not across biological replicates.
            When False, returns one column per spheroid series. When True, returns one column per collection.
            Ignored when ``mean=True``.
        
        ignore_border : bool, default=True
            Whether to exclude spheroids touching image border when calculating metrics.
            If True, spheroids that touch the image border are excluded from all metric calculations
            (returns None/NaN). If False, metrics are calculated regardless of border contact.
            Recommended to keep True to avoid edge artifacts.
            Recommended to keep True for fluorescence to avoid edge artifacts.
        
        plot : bool, default=False
            Whether to display a unified plot. If True, creates a single plot showing all
            collections/conditions instead of calling individual collection plots.
            
            The plot shows:
            - X-axis: Time in days (from relative_time_array)
            - Y-axis: Metric values
            - Lines: One per collection/condition
            - If ``mean=True``: Shaded regions show ± std
        
        ax : Optional[plt.Axes], default=None
            Optional matplotlib Axes object to plot on. If provided, the plot will be drawn
            on this axes instead of creating a new figure. If None and plot=True, creates new figure with default size (10, 6).
            Useful for subplots or custom figure layouts.
        
        plot_kwargs : Optional[dict], default=None
            Optional dictionary of keyword arguments passed to ``ax.plot()`` for customizing
            line appearance. Merged with default style (marker='o', linestyle='-').
            Can include any matplotlib plot parameters like 'color', 'linewidth', 'marker', 'alpha', etc.
        
        ax_callback : Optional[Callable[[plt.Axes], None]], default=None
            Optional callable function that receives the axes for additional formatting.
            Called after default formatting (grid, labels, title, legend) but before saving/showing.
            Use for custom formatting like setting axis limits, scales, or adding text/annotations.
        
        savepath : Optional[str], default=None
            Optional path to save the figure. Only used if ``plot=True``.
            Works with both internally created figures and externally provided axes.
            Can specify any file format supported by matplotlib (PNG, PDF, SVG, etc.).
        
        **kwargs : dict, optional
            Additional keyword arguments passed to underlying ``SpheroidCollection.metric`` calls.
            Excludes 'plot', 'mean', and 'average' which are handled at this level.
            Common options include: 'interpolate' (bool), 'timepoint' (float/str/datetime), 
            'time_period' (str), 'skip_nan' (bool), and other parameters supported by SpheroidCollection.metric().
        
        Returns
        -------
        pd.DataFrame
            DataFrame with metric values over time:
            
            **If ``mean=False``:**
            - Columns: One per collection (or per series if ``average=False``)
            - Column names: Condition labels (e.g., 'concentration=0.1; cell_line=SW620')
            - Values: Metric values at each timepoint
            
            **If ``mean=True``:**
            - Columns: MultiIndex with (condition_label, 'mean') and (condition_label, 'std')
            - Values: Mean and standard deviation across technical replicates per condition
            
            **Index:**
            - Timepoints in hours (from relative_time_array) or timepoint strings
            
            **Special cases:**
            - For ``name='fluorescence_*'`` (without _mean/_cumulative): Returns MultiIndex columns
              with ('cumulative', 'mean') sub-columns
            - If ``timepoint`` is specified in kwargs: Returns single-row DataFrame for that timepoint
        
        Examples
        --------
        >>> # Calculate radius over time for all collections
        >>> df = replicate.metric('radius')
        
        >>> # Calculate mean ± std per condition
        >>> df = replicate.metric('radius', mean=True)
        
        >>> # Calculate fluorescence with custom plot
        >>> df = replicate.metric('fluorescence_green_mean', plot=True, 
        ...                       plot_kwargs={'color': 'green', 'linewidth': 2})
        
        >>> # Plot on existing axes
        >>> fig, ax = plt.subplots()
        >>> df = replicate.metric('radius', plot=True, ax=ax)
        
        >>> # Save plot to file
        >>> df = replicate.metric('radius', plot=True, savepath='radius_plot.png')
        
        >>> # Calculate at specific timepoint
        >>> df = replicate.metric('radius', timepoint=48.0)  # 48 hours
        
        >>> # Use time period
        >>> df = replicate.metric('radius', time_period='growth_phase')
        """
        if not self.data_has_been_loaded:
            print("No data loaded for this replicate. Call load_images() first.")
            return pd.DataFrame()
        
        # Get all collections for this replicate
        collections = self.get_collections()
        
        if not collections:
            print("No collections found. Ensure platemap is configured and images are loaded.")
            return pd.DataFrame()

        def _format_condition_label(cond_tuple):
            """Format condition tuple as 'name1=val1; name2=val2' for column names.

            Numeric values are formatted with at most 2 decimal places.
            """
            if not cond_tuple:
                return "default"
            parts = []
            for item in cond_tuple:
                if isinstance(item, tuple) and len(item) == 2:
                    name, value = item
                    if isinstance(value, (int, float)):
                        val_str = f"{float(value):.2f}".rstrip("0").rstrip(".")
                    else:
                        val_str = str(value)
                    parts.append(f"{name}={val_str}")
                else:
                    parts.append(str(item))
            return "; ".join(parts)

        # Build palette assignment for each condition/collection label based on the
        # primary condition (first cell line / compound defined in platemap) and
        # descending numeric value of that condition.
        palette_map: dict[str, list[str]] = {}
        if collections and CARTO_SEQUENTIAL:
            # Parse condition tuples to dictionaries
            condition_data: dict[tuple, dict] = {}
            for cond_tuple in collections.keys():
                cond_dict: dict = {}
                if isinstance(cond_tuple, tuple):
                    for item in cond_tuple:
                        if isinstance(item, tuple) and len(item) == 2:
                            cond_name, cond_value = item
                            cond_dict[cond_name] = cond_value
                condition_data[cond_tuple] = cond_dict

            # Determine primary condition name from platemap order (cell_lines then compounds)
            primary_candidates: list[str] = []
            if hasattr(self, "platemap"):
                try:
                    primary_candidates.extend(getattr(self.platemap, "cell_lines", {}).keys())
                    primary_candidates.extend(getattr(self.platemap, "compounds", {}).keys())
                except Exception:
                    primary_candidates = []

            primary_name: str | None = None
            for cand in primary_candidates:
                if any(cand in cond for cond in condition_data.values()):
                    primary_name = cand
                    break

            # Fallback: use first available condition name if platemap info is not sufficient
            if primary_name is None and condition_data:
                first_cond_dict = next(iter(condition_data.values()))
                if first_cond_dict:
                    primary_name = next(iter(first_cond_dict.keys()))

            if primary_name is not None:
                items: list[dict] = []
                for cond_tuple, cond_dict in condition_data.items():
                    raw_val = cond_dict.get(primary_name)
                    num_val: float | None = None
                    if isinstance(raw_val, (int, float)):
                        num_val = float(raw_val)
                    else:
                        try:
                            num_val = float(str(raw_val))
                        except (TypeError, ValueError):
                            num_val = None
                    label = _format_condition_label(cond_tuple)
                    items.append(
                        {
                            "tuple": cond_tuple,
                            "label": label,
                            "value": num_val,
                        }
                    )

                numeric_items = [it for it in items if it["value"] is not None]
                non_numeric_items = [it for it in items if it["value"] is None]
                # Sort numeric items by ascending primary value so that colors
                # follow increasing concentration (or similar) in the legend.
                numeric_items.sort(key=lambda it: it["value"])
                ordered_items = numeric_items + non_numeric_items

                # Palette order: use the definition order from CARTO_SEQUENTIAL
                # (assumed to be already in the desired sequence, e.g.
                # Gold → Red → Burgundy → ...).
                palette_names = list(CARTO_SEQUENTIAL.keys())
                num_palettes = len(palette_names)
                for idx, it in enumerate(ordered_items):
                    palette_name = palette_names[idx % num_palettes]
                    palette_map[it["label"]] = CARTO_SEQUENTIAL[palette_name]

        if mean:
            # Per-condition averaging: each collection = one condition (same platemap values).
            # Call collection.metric(mean=True) to average technical replicates within each condition.
            per_condition_frames: list[pd.DataFrame] = []
            
            for condition_tuple, collection in collections.items():
                # Use mean=True: average all spheroids with same conditions within this collection
                coll_df = collection.metric(
                    name=name,
                    mean=True,
                    interpolate=True,
                    ignore_border=ignore_border,
                    plot=False,
                    **{k: v for k, v in kwargs.items() if k not in {"plot", "mean", "average"}},
                )
                if coll_df.empty:
                    continue
                
                # Name columns after conditions (separated by ";") instead of collection.name
                condition_label = _format_condition_label(condition_tuple)
                new_cols = [(condition_label, c[1]) for c in coll_df.columns]
                coll_df = coll_df.set_axis(pd.MultiIndex.from_tuples(new_cols), axis=1)
                per_condition_frames.append(coll_df)
            
            if not per_condition_frames:
                print("No metric data could be calculated for any collection.")
                return pd.DataFrame()
            
            # Concatenate: one (condition, mean) and (condition, std) per condition
            result_df = pd.concat(per_condition_frames, axis=1, sort=False)
            result_df = result_df.sort_index(axis=1)
            
        else:
            # No aggregation across collections; optionally average technical replicates
            frames: list[pd.DataFrame] = []
            
            for condition_tuple, collection in collections.items():
                df = collection.metric(
                    name=name,
                    mean=average,
                    ignore_border=ignore_border,
                    plot=False,  # Never plot individual collections
                    **{k: v for k, v in kwargs.items() if k not in {"plot", "mean", "average"}},
                )
                if df.empty:
                    continue
                
                # Name columns after full condition label; each underlying series
                # (and optional sub-metric) becomes one unique column under this
                # label. For simple columns we use (condition_label, series_id),
                # for MultiIndex columns we prepend the condition_label to the
                # existing tuple to preserve series identity.
                condition_label = _format_condition_label(condition_tuple)
                if isinstance(df.columns, pd.MultiIndex):
                    new_columns = [(condition_label, *tuple(col)) for col in df.columns]
                else:
                    suffix = df.columns.tolist() if hasattr(df.columns, "tolist") else list(df.columns)
                    new_columns = [(condition_label, str(s)) for s in suffix]
                df.columns = pd.MultiIndex.from_tuples(new_columns)
                frames.append(df)
            
            if not frames:
                print("No metric data could be calculated for any collection.")
                return pd.DataFrame()
            
            # Combine all frames - use outer join to handle different timepoints
            result_df = pd.concat(frames, axis=1, sort=False, join='outer')
            result_df = result_df.sort_index(axis=1)
        
        # Create unified plot if requested
        if plot:
            self._plot_metric(
                result_df,
                name,
                mean,
                ignore_border,
                ax=ax,
                plot_kwargs=plot_kwargs,
                ax_callback=ax_callback,
                savepath=savepath,
                palette_map=palette_map,
                **kwargs,
            )
        
        return result_df
    
    def growth_rate(
        self,
        metric_name: str = "radius",
        *,
        time_period: str | None = None,
        model: str = "linear",
        mean: bool = False,
    ) -> pd.DataFrame:
        """
        Calculate growth rates across all collections in this replicate.
        
        This method calculates growth rates for all collections that belong to this replicate.
        Each collection represents one condition (same platemap values).
        
        Args:
            metric_name: Metric name forwarded to ``SpheroidCollection.growth_rate`` / ``SpheroidSeries.growth_rate``.
            time_period: Optional time period name to limit analysis to specific time range.
            model: Model to use for fitting (currently only 'linear' is supported).
            mean: If True, calculate mean ± std error per condition. If False, returns individual growth rates.
        
        Returns:
            DataFrame with growth rates:
                - If ``mean=False``: One row per collection, columns: 'growth_rate', 'std_error'
                - If ``mean=True``: One row per condition, columns: 'growth_rate_mean', 'growth_rate_sem'
                - Index: Condition labels (formatted as 'name1=val1; name2=val2')
        """
        if not self.data_has_been_loaded:
            print("No data loaded for this replicate. Call load_images() first.")
            return pd.DataFrame()
        
        # Get all collections for this replicate
        collections = self.get_collections()
        
        if not collections:
            print("No collections found. Ensure platemap is configured and images are loaded.")
            return pd.DataFrame()
        
        def _format_condition_label(cond_tuple):
            """Format condition tuple as 'name1=val1; name2=val2' for row names."""
            if not cond_tuple:
                return "default"
            parts = []
            for item in cond_tuple:
                if isinstance(item, tuple) and len(item) == 2:
                    parts.append(f"{item[0]}={item[1]}")
                else:
                    parts.append(str(item))
            return "; ".join(parts)
        
        if mean:
            # Per-condition averaging: each collection = one condition
            result_rows = []
            
            for condition_tuple, collection in collections.items():
                try:
                    mean_rate, sem = collection.growth_rate(
                        metric_name=metric_name,
                        time_period=time_period,
                        model=model,
                        mean=True
                    )
                    condition_label = _format_condition_label(condition_tuple)
                    result_rows.append({
                        'growth_rate_mean': mean_rate,
                        'growth_rate_sem': sem
                    })
                except (ValueError, KeyError) as e:
                    logger.warning(f"Could not calculate growth rate for condition {condition_tuple}: {e}")
                    continue
            
            if not result_rows:
                print("No growth rate data could be calculated for any collection.")
                return pd.DataFrame()
            
            # Create DataFrame with condition labels as index
            condition_labels = [_format_condition_label(cond) for cond in collections.keys()]
            result_df = pd.DataFrame(result_rows, index=condition_labels[:len(result_rows)])
            
        else:
            # Individual growth rates per collection
            result_rows = []
            condition_labels = []
            
            for condition_tuple, collection in collections.items():
                try:
                    growth_df = collection.growth_rate(
                        metric_name=metric_name,
                        time_period=time_period,
                        model=model,
                        mean=False
                    )
                    if not growth_df.empty:
                        condition_label = _format_condition_label(condition_tuple)
                        # Add condition label to each row
                        for idx, row in growth_df.iterrows():
                            result_rows.append({
                                'condition': condition_label,
                                'series': idx,
                                'growth_rate': row['growth_rate'],
                                'std_error': row['std_error']
                            })
                except (ValueError, KeyError) as e:
                    logger.warning(f"Could not calculate growth rate for condition {condition_tuple}: {e}")
                    continue
            
            if not result_rows:
                print("No growth rate data could be calculated for any collection.")
                return pd.DataFrame()
            
            result_df = pd.DataFrame(result_rows)
            result_df = result_df.set_index(['condition', 'series'])
        
        return result_df
    
    def radial_profile(self, timepoint: int | str | datetime,
                      condition: str | list[str],
                      channels: str | list[str] = ['green'],
                      return_absolute: bool = False,
                      normalize: bool = True,
                      smoothing: float = 2,
                      plot: bool = True,
                      savepath: str | None = None,
                      dual_axis: bool = False) -> dict:
        """
        Compare radial profiles across different conditions for a specific timepoint.
        
        Conditions must be disjoint (e.g., different compound concentrations or cell lines).
        If conditions are not disjoint, raises an error.
        
        Parameters
        ----------
        timepoint : int | str | datetime
            Timepoint to analyze. Can be:
            - int: Relative time in hours
            - str: Datetime string in format 'YYYY-MM-DD HH:MM:SS'
            - datetime: Datetime object
        condition : str | list[str]
            Condition name(s) to compare. Must be a compound or cell_line from platemap.
            If list, compares multiple conditions (they must be disjoint).
        channels : str | list[str], default=['green']
            Fluorescence channel(s) to analyze. Can be a single channel string ('green', 'red', 'blue')
            or a list of channels.
        return_absolute : bool, default=False
            If True, uses absolute distances in μm. If False, uses normalized distances (0-1).
        normalize : bool, default=True
            Whether to normalize intensity profiles to [0,1]
        smoothing : float, default=2
            Gaussian smoothing parameter (sigma) for radial profile
        plot : bool, default=True
            Whether to display a plot
        savepath : str | None, default=None
            Optional path to save the plot
        
        Returns
        -------
        dict
            Dictionary containing:
            - timepoint: Processed timepoint identifier
            - profiles: Dictionary mapping condition_value -> channel -> profile data
            - distances: Distance array (relative or absolute) - may differ between conditions if return_absolute=True
        """
        import matplotlib.pyplot as plt
        from SpheroidPy.utils.color_palettes import CARTO_SEQUENTIAL
        
        # Normalize channels to list
        if isinstance(channels, str):
            channels = [channels]
        
        if not self.data_has_been_loaded:
            print("No data loaded for this replicate. Call load_images() first.")
            return {}
        
        # Get all collections
        collections = self.get_collections()
        
        if not collections:
            print("No collections found. Ensure platemap is configured and images are loaded.")
            return {}
        
        # Normalize condition to list
        if isinstance(condition, str):
            conditions_to_compare = [condition]
        else:
            conditions_to_compare = list(condition)
        
        # Parse condition tuples to extract condition values
        condition_data = {}  # {condition_tuple: (condition_dict, collection)}
        
        for cond_tuple, collection in collections.items():
            cond_dict = {}
            if isinstance(cond_tuple, tuple):
                for item in cond_tuple:
                    if isinstance(item, tuple) and len(item) == 2:
                        name, value = item
                        cond_dict[name] = value
            condition_data[cond_tuple] = (cond_dict, collection)
        
        # Check if all specified conditions exist
        all_condition_names = set()
        for cond_dict, _ in condition_data.values():
            all_condition_names.update(cond_dict.keys())
        
        for cond_name in conditions_to_compare:
            if cond_name not in all_condition_names:
                raise ValueError(f"Condition '{cond_name}' not found in platemap. Available: {sorted(all_condition_names)}")
        
        # Group collections by their values for the specified conditions
        # Use OR logic: collection must have at least ONE of the specified conditions
        condition_groups = {}  # {condition_value_tuple: list of collections}
        
        for cond_tuple, (cond_dict, collection) in condition_data.items():
            # Check if collection has at least one of the specified conditions
            has_any_condition = False
            condition_values_list = []
            
            for cond_name in conditions_to_compare:
                if cond_name in cond_dict:
                    has_any_condition = True
                    condition_values_list.append((cond_name, cond_dict[cond_name]))
            
            if not has_any_condition:
                # This collection doesn't have any of the specified conditions - skip it
                continue
            
            # Create a key from the condition values found
            # Sort by condition name for consistency
            condition_values_list.sort(key=lambda x: x[0])
            condition_key = tuple((name, val) for name, val in condition_values_list)
            
            # Group collections by condition values (allow multiple collections per group)
            if condition_key not in condition_groups:
                condition_groups[condition_key] = []
            condition_groups[condition_key].append(collection)
        
        if not condition_groups:
            raise ValueError(f"No collections found with any of the specified conditions: {conditions_to_compare}")
        
        # Helper function to convert timepoint to datetime
        def _ensure_dt(val):
            if isinstance(val, datetime):
                return val
            elif isinstance(val, str):
                return datetime.strptime(val, '%Y-%m-%d %H:%M:%S')
            elif isinstance(val, (int, float)):
                # Convert relative hours to datetime
                if not self.time_points_array:
                    raise ValueError("No timepoints available")
                t0 = datetime.strptime(self.time_points_array[0], '%Y-%m-%d %H:%M:%S')
                return t0 + pd.Timedelta(hours=float(val))
            else:
                raise ValueError(f"Cannot convert {type(val)} to datetime")
        
        # Process timepoint
        tp_dt = _ensure_dt(timepoint)
        
        # Calculate profiles for each condition group
        profiles_data = {}
        distances_data = {}
        condition_values_map = {}  # Map condition_label -> condition_values tuple
        
        for condition_key, collection_list in condition_groups.items():
            # Format condition label from condition_key (which is tuple of (name, value) pairs)
            label_parts = [f"{name}={value}" for name, value in condition_key]
            condition_label = "; ".join(label_parts)
            
            # Store mapping for color scheme lookup
            condition_values_map[condition_label] = condition_key
            
            # Average across multiple collections if present
            all_profiles = []
            all_distances = []
            
            for collection in collection_list:
                try:
                    # Use collection's radial_profile method with mean=True
                    result = collection.radial_profile(
                        timepoint=tp_dt,
                        channels=channels,
                        return_absolute=return_absolute,
                        normalize=normalize,
                        smoothing=smoothing,
                        mean=True,  # Average across technical replicates
                        plot=False
                    )
                    
                    all_profiles.append(result['profiles'])
                    all_distances.append(result['distances'])
                    
                except Exception as e:
                    logger.warning(f"Failed to calculate radial profile for collection {collection.name} in condition {condition_label}: {e}")
                    continue
            
            if not all_profiles:
                logger.warning(f"No valid profiles for condition {condition_label}")
                continue
            
            # Average across collections if multiple exist
            if len(all_profiles) > 1:
                averaged_profiles = {}
                for ch in channels:
                    mean_list = []
                    std_list = []
                    for prof_dict in all_profiles:
                        if ch in prof_dict:
                            mean_list.append(prof_dict[ch]['mean'])
                            std_list.append(prof_dict[ch]['std'])
                    
                    if mean_list:
                        mean_array = np.stack(mean_list, axis=0)
                        std_array = np.stack(std_list, axis=0)
                        
                        avg_mean = np.mean(mean_array, axis=0)
                        avg_std = np.sqrt(np.mean(std_array**2, axis=0) + np.var(mean_array, axis=0))
                        
                        averaged_profiles[ch] = {
                            'mean': avg_mean,
                            'std': avg_std
                        }
                profiles_data[condition_label] = averaged_profiles
            else:
                profiles_data[condition_label] = all_profiles[0]
            
            # Use first distance array
            if all_distances:
                distances_data[condition_label] = all_distances[0]
        
        if not profiles_data:
            raise ValueError("No valid radial profiles could be calculated for any condition")
        
        # Plotting
        if plot:
            # Use dual-axis only if explicitly requested and exactly 2 channels
            use_dual_axis = dual_axis and len(channels) == 2
            
            if use_dual_axis:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax2 = ax.twinx()
            else:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax2 = None
            
            # Color palette mapping for channels (using channel-similar palettes)
            # Use Green for green, Red/Orange for red, Teal for blue
            channel_palettes = {
                'green': CARTO_SEQUENTIAL.get('Green', ['#00441b', '#006d2c', '#238b45', '#41ab5d', '#74c476', '#a1d99b', '#c7e9c0']),
                'red': CARTO_SEQUENTIAL.get('Red', ['#7f0000', '#b31b1b', '#d94801', '#f16913', '#fd8d3c', '#fdae6b', '#fee6ce']),
                'blue': CARTO_SEQUENTIAL.get('Teal', ['#004c4c', '#006d6d', '#238b8b', '#41a9a9', '#74c8c8', '#a1e3e3', '#c8f5f5']),
            }
            
            # Generate color schemes for each condition based on condition values
            # For each condition, extract values and create color mapping
            condition_color_schemes = {}
            
            for cond_name in conditions_to_compare:
                # Extract all unique values for this condition from condition_key tuples
                condition_values_list = []
                for cond_key in condition_groups.keys():
                    # cond_key is tuple of (name, value) pairs
                    for name, value in cond_key:
                        if name == cond_name:
                            condition_values_list.append(value)
                            break
                
                if not condition_values_list:
                    continue
                
                # Remove duplicates and sort
                unique_values = sorted(set(condition_values_list))
                
                # Try to convert to numeric for sorting
                try:
                    numeric_values = [float(v) if isinstance(v, (int, float, str)) and str(v).replace('.', '').replace('-', '').isdigit() else v for v in unique_values]
                    if all(isinstance(v, (int, float)) for v in numeric_values):
                        unique_values = sorted(set(numeric_values))
                except (ValueError, TypeError):
                    pass
                
                # Create color palette for this condition (centered around middle)
                # Use a different palette for each condition
                palette_names = list(CARTO_SEQUENTIAL.keys())
                cond_palette_idx = conditions_to_compare.index(cond_name) % len(palette_names)
                palette_name = palette_names[cond_palette_idx]
                palette = CARTO_SEQUENTIAL.get(palette_name, ['#333333', '#555555', '#777777', '#999999', '#bbbbbb', '#dddddd'])
                
                # Map condition values to colors (equidistant, centered around middle)
                n_colors = len(palette)
                mid_idx = n_colors // 2
                value_to_color = {}
                
                if len(unique_values) == 1:
                    value_to_color[unique_values[0]] = palette[mid_idx]
                else:
                    for val_idx, val in enumerate(unique_values):
                        # Map to palette indices (equidistant, centered)
                        normalized_pos = val_idx / (len(unique_values) - 1) if len(unique_values) > 1 else 0.5
                        # Center around middle: 0 -> mid, 1 -> edges
                        color_idx = int(normalized_pos * (n_colors - 1))
                        color_idx = max(0, min(n_colors - 1, color_idx))
                        value_to_color[val] = palette[color_idx]
                
                condition_color_schemes[cond_name] = value_to_color
            
            # Plot profiles
            for condition_label, condition_profiles in profiles_data.items():
                distances = distances_data.get(condition_label)
                if distances is None:
                    continue
                
                # Get condition key from map (which is tuple of (name, value) pairs)
                condition_key = condition_values_map.get(condition_label)
                if condition_key is None:
                    # Fallback: try to extract from label
                    condition_vals = {}
                    if ';' in condition_label:
                        parts = condition_label.split(';')
                        for part in parts:
                            if '=' in part:
                                cond_name, cond_val = part.split('=', 1)
                                cond_name = cond_name.strip()
                                try:
                                    cond_val = float(cond_val.strip())
                                except (ValueError, TypeError):
                                    cond_val = cond_val.strip()
                                condition_vals[cond_name] = cond_val
                    else:
                        if '=' in condition_label:
                            cond_name, cond_val = condition_label.split('=', 1)
                            cond_name = cond_name.strip()
                            try:
                                cond_val = float(cond_val.strip())
                            except (ValueError, TypeError):
                                cond_val = cond_val.strip()
                            condition_vals[cond_name] = cond_val
                    
                    # Convert to tuple format
                    condition_key = tuple((name, condition_vals[name]) for name in sorted(condition_vals.keys()) if name in conditions_to_compare)
                
                # Determine color based on first condition found in key (or use default)
                color = None
                if condition_key and conditions_to_compare:
                    # Find first condition from conditions_to_compare that appears in condition_key
                    for cond_name in conditions_to_compare:
                        for name, value in condition_key:
                            if name == cond_name:
                                if cond_name in condition_color_schemes:
                                    color = condition_color_schemes[cond_name].get(value, '#333333')
                                break
                        if color is not None:
                            break
                
                if color is None:
                    # Fallback to tab10 colors
                    cond_idx = list(profiles_data.keys()).index(condition_label)
                    color = plt.cm.tab10(cond_idx % 10)
                
                for ch_idx, channel in enumerate(channels):
                    if channel not in condition_profiles:
                        continue
                    
                    profile = condition_profiles[channel]
                    mean = profile['mean']
                    std = profile.get('std', np.zeros_like(mean))
                    
                    # Use different line styles for different channels
                    linestyle = ['-', '--', '-.'][ch_idx % 3]
                    
                    label = f"{condition_label} - {channel}"
                    
                    # Select axis for dual-axis plotting
                    if use_dual_axis:
                        current_ax = ax if ch_idx == 0 else ax2
                    else:
                        current_ax = ax
                    
                    current_ax.plot(distances, mean, color=color, linestyle=linestyle, 
                                   label=label, linewidth=2)
                    current_ax.fill_between(distances, mean - std, mean + std, 
                                           color=color, alpha=0.15)
            
            # Set labels
            if return_absolute:
                ax.set_xlabel('Radial distance [µm]')
            else:
                ax.set_xlabel('Normalized distance (ρ)')
            
            if use_dual_axis:
                # Dual y-axis labels (black text)
                ax.set_ylabel(f'Intensity (a.u.) - {channels[0]}', color='black')
                ax.tick_params(axis='y', labelcolor='black')
                if len(channels) > 1:
                    ax2.set_ylabel(f'Intensity (a.u.) - {channels[1]}', color='black')
                    ax2.tick_params(axis='y', labelcolor='black')
            elif normalize:
                ax.set_ylabel('Normalized intensity')
            else:
                ax.set_ylabel('Intensity (a.u.)')
            
            title = f'Radial Profiles Comparison - {self.name}'
            if len(conditions_to_compare) == 1:
                title += f'\nCondition: {conditions_to_compare[0]}'
            else:
                title += f'\nConditions: {", ".join(conditions_to_compare)}'
            ax.set_title(title, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Combine legends if using dual axis
            if use_dual_axis:
                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=9)
            else:
                ax.legend(loc='best', fontsize=9)
            
            # Set xlims: use union of all distance ranges
            all_distances = []
            for dist in distances_data.values():
                if dist is not None and len(dist) > 0:
                    all_distances.extend(dist)
            if all_distances:
                all_distances = np.array(all_distances)
                ax.set_xlim(all_distances.min(), all_distances.max())
            
            plt.tight_layout()
            if savepath is not None:
                plt.savefig(savepath, transparent=True, dpi=300, bbox_inches='tight')
            plt.show()
        
        return {
            'timepoint': tp_dt,
            'profiles': profiles_data,
            'distances': distances_data
        }
    
    def _plot_metric(
        self,
        df: pd.DataFrame,
        name: str,
        mean: bool,
        ignore_border: bool,
        skip_nan: bool = True,
        ax: Optional[plt.Axes] = None,
        plot_kwargs: Optional[dict] = None,
        ax_callback: Optional[Callable[[plt.Axes], None]] = None,
        savepath: Optional[str] = None,
        palette_map: Optional[dict[str, list[str]]] = None,
        **kwargs,
    ) -> None:
        """Create a unified plot for all collections in this replicate."""
        import matplotlib.pyplot as plt
        
        # Track whether we created the figure (only then call plt.show())
        created_figure = ax is None
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4.8))
        else:
            fig = ax.figure

        # Transparent background and modern axes style
        if fig is not None:
            fig.patch.set_alpha(0.0)
        ax.set_facecolor("none")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Prepare plot_kwargs
        if plot_kwargs is None:
            plot_kwargs = {}
        
        # Use relative_time_array if available, otherwise use index
        # Create x as pandas Series aligned with df.index to ensure proper indexing
        if self.relative_time_array and len(self.relative_time_array) == len(df.index):
            x = pd.Series(np.array(self.relative_time_array) / 24, index=df.index)  # Convert to days
        else:
            # Fallback: use index as time (convert index to numeric if possible)
            if isinstance(df.index, pd.DatetimeIndex):
                # If index is datetime, convert to days
                x = pd.Series((df.index - df.index[0]).total_seconds() / 86400, index=df.index)
            else:
                # Use numeric index
                x = pd.Series(range(len(df.index)), index=df.index)
        
        if mean:
            # Plot mean ± std
            if isinstance(df.columns, pd.MultiIndex):
                default_orange = "#D97706"
                for label in df.columns.get_level_values(0).unique():
                    mean_col = (label, 'mean')
                    std_col = (label, 'std')
                    
                    if mean_col in df.columns and std_col in df.columns:
                        mean_array = df[mean_col]
                        std_array = df[std_col]
                        
                        if skip_nan:
                            valid_mask = ~np.isnan(mean_array)
                            x_plot = x[valid_mask].values
                            mean_plot = mean_array[valid_mask].values
                            std_plot = std_array[valid_mask].values
                        else:
                            x_plot = x
                            mean_plot = mean_array
                            std_plot = std_array

                        # Choose mid color from assigned palette (or fallback)
                        palette = palette_map.get(label) if palette_map else None
                        if palette:
                            line_color = palette[len(palette) // 2]
                        else:
                            line_color = default_orange

                        ax.fill_between(
                            x_plot,
                            mean_plot - std_plot,
                            mean_plot + std_plot,
                            alpha=0.2,
                            color=line_color,
                        )

                        # Default line style without markers; user can override via plot_kwargs
                        default_kwargs = {
                            "color": line_color,
                            "linestyle": "-",
                            "linewidth": 2.5,
                        }
                        default_kwargs.update(plot_kwargs)
                        ax.plot(x_plot, mean_plot, label=f"{label}", **default_kwargs)
            else:
                # Fallback for non-MultiIndex columns
                default_kwargs = {
                    "color": "#D97706",
                    "linestyle": "-",
                    "linewidth": 2.5,
                }
                default_kwargs.update(plot_kwargs)
                for col in df.columns:
                    y = df[col]
                    if skip_nan:
                        valid_mask = ~np.isnan(y)
                        ax.plot(x[valid_mask].values, y[valid_mask].values, label=col, **default_kwargs)
                    else:
                        ax.plot(x.values, y.values, label=col, **default_kwargs)
        else:
            # Plot individual series from each collection
            default_kwargs = {
                "linestyle": "-",
                "linewidth": 2.0,
                "alpha": 0.9,
            }
            default_kwargs.update(plot_kwargs)
            
            if isinstance(df.columns, pd.MultiIndex):
                # MultiIndex columns: (collection_name, series_name)
                for label in df.columns.get_level_values(0).unique():
                    # Get all columns for this collection/condition label
                    collection_cols = [col for col in df.columns if col[0] == label]

                    # Determine palette for this label. Within a palette, assign
                    # colors sequentially per series. The palette entries are
                    # defined from dark → light; we reverse this so that the
                    # first series uses the lightest shade, the next slightly
                    # darker, etc.
                    palette = palette_map.get(label) if palette_map else None
                    num_colors = len(palette) if palette is not None else 0
                    if palette:
                        ordered_indices = list(range(num_colors - 1, -1, -1))
                    else:
                        ordered_indices = []
                    
                    # Group columns by their "series id" (second level) so that
                    # each well/series is plotted only once, even if multiple
                    # sub-metrics exist on deeper levels. We keep the first
                    # column for each series id.
                    series_map: dict[str, tuple] = {}
                    for col in collection_cols:
                        if isinstance(col, tuple) and len(col) >= 2:
                            series_id = str(col[1])
                        else:
                            series_id = str(col)
                        if series_id not in series_map:
                            series_map[series_id] = col

                    for idx, (series_id, col) in enumerate(series_map.items()):
                        y = df[col]
                        # Ensure we always work with a 1D Series (some pandas
                        # combinations may return a DataFrame for MultiIndex
                        # selection). In that rare case, use the first column.
                        if isinstance(y, pd.DataFrame):
                            y = y.iloc[:, 0]
                        # Build a readable legend label. For (condition, series)
                        # we show "condition: series"; for deeper MultiIndex
                        # (e.g. condition, series, kind) we append extra levels
                        # in parentheses so each curve remains distinguishable.
                        if isinstance(col, tuple):
                            if len(col) == 2:
                                col_label = f"{col[0]}: {col[1]}"
                            else:
                                extra = ", ".join(str(c) for c in col[2:])
                                col_label = f"{col[0]}: {col[1]} ({extra})"
                        else:
                            col_label = str(col)

                        # Choose color within palette for this series
                        if palette:
                            if ordered_indices:
                                series_color = palette[ordered_indices[idx % len(ordered_indices)]]
                            else:
                                series_color = palette[idx % num_colors]
                            kwargs_local = dict(default_kwargs)
                            kwargs_local["color"] = series_color
                        else:
                            kwargs_local = dict(default_kwargs)
                        
                        if skip_nan:
                            valid_mask = ~np.isnan(y)
                            ax.plot(
                                x[valid_mask].values,
                                y[valid_mask].values,
                                label=col_label,
                                **kwargs_local,
                            )
                        else:
                            ax.plot(
                                x.values,
                                y.values,
                                label=col_label,
                                **kwargs_local,
                            )
            else:
                # Simple column names
                for idx, col in enumerate(df.columns):
                    y = df[col]
                    if skip_nan:
                        valid_mask = ~np.isnan(y)
                        ax.plot(x[valid_mask].values, y[valid_mask].values, 
                               label=col,
                               **default_kwargs)
                    else:
                        ax.plot(x.values, y.values,
                               label=col,
                               **default_kwargs)

        # x-limits only from min to max of valid (non-NaN) data
        if not df.empty:
            arr = df.to_numpy(dtype=float)
            any_valid = ~np.all(np.isnan(arr), axis=1)
            if any_valid.any():
                x_valid = x[any_valid].values
                ax.set_xlim(float(np.nanmin(x_valid)), float(np.nanmax(x_valid)))

        # Default formatting (can be overridden by ax_callback)
        ax.grid(True, linestyle="--", alpha=0.4, color="gray")
        ax.set_axisbelow(True)
        ax.set_xlabel("Time [d]", fontsize=14, labelpad=10)

        # y-label with heuristic units, consistent with collection.metric
        unit = ""
        pretty_name: str
        if hasattr(self, "result") and hasattr(self.result, "plot_name_dict") and name in getattr(
            self.result, "plot_name_dict", {}
        ):
            pretty_name = self.result.plot_name_dict[name]
        else:
            pretty_name = name.replace("_", " ").title()

        if name.startswith("radius") or name.endswith("radius"):
            unit = " [µm]"
        elif name.startswith("area") or name.endswith("area"):
            unit = " [µm²]"
        elif name.startswith("fluorescence"):
            unit = " (a.u.)"

        ax.set_ylabel(f"{pretty_name}{unit}", fontsize=14, labelpad=10)

        # Legend styling: smaller font for many individual lines when mean=False
        if mean:
            ax.legend(fontsize=11, framealpha=0.5, edgecolor="gray", loc="best")
        else:
            ax.legend(fontsize=9, framealpha=0.5, edgecolor="gray", loc="best")

        ax.set_title(
            f"{pretty_name} over time - {self.name}" + (" (mean ± std)" if mean else ""),
            fontsize=16,
            fontweight="bold",
            pad=10,
        )
        
        # Apply custom callback if provided
        if ax_callback is not None:
            ax_callback(ax)
        
        # Save if requested
        if savepath is not None:
            if fig is None:
                fig = ax.figure
            fig.savefig(savepath, bbox_inches='tight')
        
        # Show only if we created the figure ourselves
        if created_figure and fig is not None:
            plt.tight_layout()
            plt.show()

    @classmethod
    def from_file(cls, name: str, replicate_index: int, result: "Result", 
                  replicate_group: h5py.Group, hdf5_path: str) -> "LiveCellReplicate":
        """Load a replicate from HDF5 file.
        
        Args:
            name: Name of the replicate
            replicate_index: Index of the replicate
            result: The Result this replicate belongs to
            replicate_group: HDF5 group for this replicate
            hdf5_path: Path to HDF5 file
            
        Returns:
            LiveCellReplicate instance loaded from HDF5
        """
        replicate = object.__new__(cls)
        replicate.name = name
        replicate.result = result
        # Store hdf5_path for later use (important when experiment is not yet set)
        replicate._hdf5_path = str(hdf5_path)
        # Load layout (default to 96 for backward compatibility)
        replicate.layout = replicate_group.attrs.get('layout', 96)
        # Load data_has_been_loaded (stored as int 0/1 for PyTables compatibility)
        data_loaded = replicate_group.attrs.get('data_has_been_loaded', 1)
        if isinstance(data_loaded, (int, np.integer)):
            replicate.data_has_been_loaded = bool(data_loaded)
        else:
            replicate.data_has_been_loaded = bool(data_loaded) if data_loaded else False
        replicate.spheroid_dict = {}
        replicate.files_dict = {}
        replicate.time_points_array = []
        replicate.relative_time_array = []
        replicate.num_cores = mp.cpu_count()
        
        # Load created_at
        if 'created_at' in replicate_group.attrs:
            replicate._created_at = datetime.fromisoformat(replicate_group.attrs['created_at'])
        else:
            replicate._created_at = datetime.now()
        
        # Load platemap
        from SpheroidPy.experiment.platemap import Platemap
        if 'Platemap' in replicate_group:
            platemap_group = replicate_group['Platemap']
            replicate.platemap = Platemap.from_file(replicate, platemap_group, hdf5_path)
        else:
            replicate.platemap = Platemap(replicate)
        
        # ImageSeries are NOT loaded from ReplicateInfo - they are loaded from Collections/
        # The spheroid_dict will be populated by Result.from_file() after collections are loaded
        # This method only loads metadata and platemap
        # Note: time_periods are stored at Series level, not Replicate level

        return replicate
    
    def show(self, plot_count: int = 2) -> None:
        """Interactive viewer for this replicate, similar to Results tab in visualization.
        
        Args:
            plot_count: Number of image panels to display (default: 2)
        """
        if not self.data_has_been_loaded:
            print("No data loaded for this replicate. Call load_images() first.")
            return
        
        if not self.spheroid_dict:
            print("No spheroid data available for this replicate.")
            return
        
        # Import required modules
        import ipywidgets as widgets
        from IPython.display import display
        import matplotlib.pyplot as plt
        import cv2
        from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
        
        # Get plate dimensions from platemap
        rows = self.platemap.row
        cols = self.platemap.col
        
        # State variables
        current_well = None
        current_time_index = 0
        widget_states = {
            'show_contour': True,
            'thresholding_value': 1.0,
            'ai_value': 0.7,
            'channel_1': 'brightfield',
            'channel_2': 'fluorescence_green'
        }
        
        # Select initial well
        if self.spheroid_dict:
            current_well = sorted(self.spheroid_dict.keys())[0]
        
        # Time control (moved to replicate info box)
        max_time = len(self.time_points_array) - 1 if self.time_points_array else 0
        time_slider = widgets.IntSlider(
            description='Time:',
            min=0,
            max=max_time,
            value=min(current_time_index, max_time),
            continuous_update=True,
            layout=widgets.Layout(flex='1 1 auto', margin='5px 0'),
            style={'description_width': 'initial'}
        )
        
        # Wrap slider in a Box with gray background to ensure background is visible
        time_slider_wrapper = widgets.Box(
            [time_slider],
            layout=widgets.Layout(
                flex='1 1 auto',
                background_color='#f0f0f0',
                padding='0',
                margin='5px 0'
            )
        )
        
        time_label = widgets.Label(
            layout=widgets.Layout(width='auto', margin='5px 10px 5px 0', min_width='60px')
        )
        
        # Initialize time label (will be updated after update_time_label is defined)
        initial_hours = int(self.relative_time_array[current_time_index]) if self.relative_time_array else 0
        initial_days = initial_hours // 24
        initial_remaining_hours = initial_hours % 24
        
        def _get_condition_for_well(well: str) -> str:
            """Get cell lines and compounds values for a well. Returns HTML string."""
            if not well or not self.platemap:
                return ""
            row = well[0] if len(well) >= 1 else ""
            col_str = well[1:] if len(well) > 1 else ""
            if not row or not col_str:
                return ""

            def _get_col(df):
                """Resolve column: platemap may use int (1,2,3) or str ('1','2','3')."""
                if col_str in df.columns:
                    return col_str
                if col_str.isdigit():
                    col_int = int(col_str)
                    if col_int in df.columns:
                        return col_int
                return None

            cell_line_parts = []
            compound_parts = []
            for name, df in self.platemap.cell_lines.items():
                try:
                    col = _get_col(df)
                    if col is not None and row in df.index:
                        val = df.loc[row, col]
                        if pd.notna(val) and val != "":
                            cell_line_parts.append(f"<i>{name}</i>: {val}")
                except Exception:
                    pass
            for name, df in self.platemap.compounds.items():
                try:
                    col = _get_col(df)
                    if col is not None and row in df.index:
                        val = df.loc[row, col]
                        if pd.notna(val) and val != "":
                            # Compound-Werte auf 2 Nachkommastellen begrenzen
                            if isinstance(val, (int, float)):
                                val_str = f"{float(val):.2f}"
                            else:
                                val_str = str(val)
                            compound_parts.append(f"<i>{name}</i>: {val_str}")
                except Exception:
                    pass
            lines = []
            if cell_line_parts:
                lines.append(f"- Cell lines: {'; '.join(cell_line_parts)};")
            if compound_parts:
                lines.append(f"- Compounds: {'; '.join(compound_parts)};")
            if not lines:
                return "<p><b>Condition:</b></p><div style='margin-left: 20px;'><p>- (no platemap data)</p></div>"
            return f"<p><b>Condition:</b></p><div style='margin-left: 20px;'><p>{'</p><p>'.join(lines)}</p></div>"

        def _get_replicate_info_html(well: str) -> str:
            """Full replicate info HTML including Condition (inside gray box)."""
            condition_part = _get_condition_for_well(well or "")
            return f"""
            <div style='padding: 10px 10px 0 10px; background-color: #f0f0f0; border-radius: 5px 0 0 0; margin-right: 0; height: 100%;'>
                <h2 style='color: #2c3e50; margin: 0;'>Replicate: {self.name}</h2>
                <div style='margin-left: 20px; margin-top: 10px;'>
                    <p><b>Layout:</b> {self.layout} wells ({rows}×{cols});         <b>Wells with data:</b> {len(self.spheroid_dict)}</p>
                    <p><b>Time Range:</b> {self.time_points_array[0] if self.time_points_array else 'N/A'} to {self.time_points_array[-1] if self.time_points_array else 'N/A'}</p>
                    {condition_part}
                </div>
            </div>
            """.strip()

        # Create header with replicate info (height will match well selection)
        # HTML part with gray background - Condition included inside gray box; flex to fill, slider fixed below
        replicate_info_html = widgets.HTML(
            value=_get_replicate_info_html(current_well or ""),
            layout=widgets.Layout(background_color='#f0f0f0', width='100%', flex='1 1 auto', min_height='0', overflow='hidden')
        )

        def update_condition_display():
            replicate_info_html.value = _get_replicate_info_html(current_well or "")

        # Time controls container - slider and label side by side
        # Wrap in HBox with gray background to ensure it's visible behind the slider
        time_controls_hbox = widgets.HBox(
            [time_slider_wrapper, time_label],
            layout=widgets.Layout(
                width='100%',
                align_items='center',
                justify_content='flex-start',
                margin='0',
                background_color='#f0f0f0'
            )
        )
        
        time_controls_container = widgets.VBox(
            [time_controls_hbox],
            layout=widgets.Layout(
                padding='0 10px 10px 10px',
                background_color='#f0f0f0',
                border_radius='0 0 0 5px',
                width='100%',
                flex='0 0 auto'
            )
        )
        
        # Combine HTML and time controls: Info füllt Höhe, Time-Slider unten fixiert; Gesamthöhe = Platemap
        replicate_info = widgets.VBox(
            [replicate_info_html, time_controls_container],
            layout=widgets.Layout(
                background_color='#f0f0f0',
                border_radius='5px 0 0 5px',
                margin_right='0',
                width='100%',
                height='100%',
                min_height='0',
                justify_content='space-between',
                overflow='hidden'
            )
        )
        
        # Create plate layout
        def create_plate_layout():
            row_letters = 'ABCDEFGHIJKLMNOP'[:rows]
            well_buttons = []
            
            for row in row_letters:
                row_buttons = []
                for col in range(1, cols + 1):
                    well = f"{row}{col}"
                    has_data = well in self.spheroid_dict
                    
                    btn = widgets.Button(
                        description=well,
                        layout=widgets.Layout(
                            width='30px',
                            height='25px',
                            padding='0px'
                        ),
                        style=widgets.ButtonStyle(
                            button_color=('#E67E22' if well == current_well else ('#FFE0CC' if has_data else '#F0F0F0')),
                            font_size='9px'  # smaller font size for well labels - must be in style, not layout
                        ),
                        disabled=not has_data
                    )
                    
                    if has_data:
                        def on_well_click(b, w=well):
                            nonlocal current_well
                            current_well = w
                            # Recreate plate layout to update colors
                            plate_layout.clear_output()
                            with plate_layout:
                                display(create_plate_layout())
                            update_condition_display()
                            update_plot()
                            update_image()
                        btn.on_click(on_well_click)
                    
                    row_buttons.append(btn)
                
                well_buttons.append(widgets.HBox(
                    row_buttons,
                    layout=widgets.Layout(margin='0px')
                ))
            
            return widgets.VBox(
                well_buttons,
                layout=widgets.Layout(margin='5px', overflow='hidden', width='300px', max_width='300px')
            )
        
        plate_layout = widgets.Output(layout=widgets.Layout(width='300px', max_width='300px', flex='0 0 auto', overflow='hidden'))
        
        # Create image output and plot output
        image_output = widgets.Output()
        plot_output = widgets.Output()
        
        # Channel selector and contour toggle
        channel_options = ['brightfield', 'fluorescence_green', 
                          'fluorescence_red', 'fluorescence_blue']
        channel_dropdown = widgets.Dropdown(
            options=channel_options,
            value=widget_states['channel_1'],
            description='Channel:',
            layout=widgets.Layout(flex='0 0 auto', width='190px', margin='0'),
            style={'description_width': '65px'}
        )
        
        def on_channel_change(change):
            widget_states['channel_1'] = change.new
        channel_dropdown.observe(on_channel_change, names='value')
        
        show_contour = widgets.Checkbox(
            value=widget_states['show_contour'],
            description='Show Contour',
            layout=widgets.Layout(flex='0 0 auto', width='auto', margin='0')
        )
        
        def on_contour_change(change):
            widget_states['show_contour'] = change.new
        show_contour.observe(on_contour_change, names='value')
        
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
        
        def update_image():
            if not current_well or current_well not in self.spheroid_dict:
                image_title.value = ""
            else:
                timepoint = self.time_points_array[time_slider.value]
                channel = channel_dropdown.value
                image_title.value = f"<div style='text-align: center;'><span style='font-size: 12px; font-weight: 600; color: #333;'>{timepoint} | {channel} | {current_well}</span></div>"
            image_output.clear_output(wait=True)
            with image_output:
                if not current_well or current_well not in self.spheroid_dict:
                    return
                
                try:
                    series = self.spheroid_dict[current_well]
                    timepoint = self.time_points_array[time_slider.value]
                    image = series.spheroid_image_dict[timepoint]
                    channel = channel_dropdown.value
                    
                    if channel == 'brightfield':
                        img = image.brightfield()
                    elif channel.startswith('fluorescence_'):
                        color = channel.split('_')[1]
                        img = image.fluorescence(color)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    else:
                        img = image.brightfield()
                    
                    if img is not None:
                        aspect = image.image_size[1] / image.image_size[0] if image.image_size[0] else 1
                        fig, ax = plt.subplots(figsize=(8, 8 * aspect))
                        ax.imshow(img, extent=[0, image.image_size[0], 0, image.image_size[1]])
                        
                        if show_contour.value and image.contour is not None:
                            contour = image.scaled_contour
                            xs = contour[:, 0]
                            ys = contour[:, 1]
                            if xs[0] != xs[-1] or ys[0] != ys[-1]:
                                xs = np.r_[xs, xs[:1]]
                                ys = np.r_[ys, ys[:1]]
                            ax.plot(xs, ys, "w-", linewidth=2)
                        
                        # 250 µm scalebar (white line + label below, bottom-right corner with spacing)
                        w, h = image.image_size[0], image.image_size[1]
                        bar_len = 250
                        margin_x, margin_y = 0.05 * w, 0.05 * h
                        x_left = w - margin_x - bar_len
                        x_right = w - margin_x
                        y_bar = margin_y
                        ax.plot([x_left, x_right], [y_bar, y_bar], "w-", linewidth=2.5, solid_capstyle="butt")
                        ax.text((x_left + x_right) / 2, y_bar - 0.015 * h, "250 µm", color="white", fontsize=10, ha="center", va="top", family="sans-serif")
                        
                        ax.axis("off")
                        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
                        plt.show()
                except Exception as e:
                    print(f"Error loading image: {e}")
        
        def _channel_to_fluorescence(ch_desc):
            """Map channel dropdown value to fluorescence channel name for radial_profile."""
            if ch_desc == 'fluorescence_green':
                return 'green'
            if ch_desc == 'fluorescence_red':
                return 'red'
            if ch_desc == 'fluorescence_blue':
                return 'blue'
            return 'green'  # fallback for brightfield (profile needs fluorescence)

        def update_plot():
            metric = metric_dropdown.value
            if metric == "radius":
                metric_label = "Radius"
            elif metric == "area":
                metric_label = "Area"
            else:
                metric_label = "Radial profile"
            if current_well and current_well in self.spheroid_dict:
                over_time = " over time" if metric in ("radius", "area") else ""
                plot_title.value = f"<div style='text-align: center;'><span style='font-size: 14px; font-weight: 600; color: #333;'>{metric_label}{over_time} – {current_well}</span></div>"
            else:
                plot_title.value = ""
            plot_output.clear_output(wait=True)
            with plot_output:
                if current_well and current_well in self.spheroid_dict:
                    try:
                        series = self.spheroid_dict[current_well]
                        timepoints = sorted(series.spheroid_image_dict.keys())
                        if metric == "profile":
                            # Normalized radial profile for current timepoint (x = normalized distance, not µm)
                            timepoint = self.time_points_array[time_slider.value]
                            image = series.spheroid_image_dict.get(timepoint)
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
                            if metric == "area":
                                values = [series.spheroid_image_dict[tp].area for tp in timepoints]
                                ylabel = "Area [µm²]"
                            else:
                                values = [series.spheroid_image_dict[tp].radius for tp in timepoints]
                                ylabel = "Radius [µm]"
                            values = [v if v is not None else np.nan for v in values]
                            if self.relative_time_array and len(self.relative_time_array) == len(timepoints):
                                times = [t / 24 for t in self.relative_time_array]
                            else:
                                times = list(range(len(timepoints)))
                            current_time = times[current_time_index] if current_time_index < len(times) else 0
                            current_val = values[current_time_index] if current_time_index < len(values) else None
                            fig, ax = plt.subplots(figsize=(6, 4.1))
                            
                            # Plot data first
                            ax.plot(times, values, color="grey", zorder=5)
                            if current_val is not None and not np.isnan(current_val):
                                ax.plot(current_time, current_val, ".", markersize=10,
                                        label=f"{current_val:.1f} {'µm' if metric == 'radius' else 'µm²'} @ {current_time:.2f} d", color="#e29266", zorder=6)
                            ax.set_xlabel("Time [d]")
                            ax.set_ylabel(ylabel)
                            ax.grid(True, alpha=0.3)
                            if current_val is not None and not np.isnan(current_val):
                                ax.legend()
                            
                            # Add time periods as gray transparent background regions (after plot to get correct y-limits)
                            if hasattr(series, 'time_periods') and series.time_periods:
                                # Get actual y-axis limits after plotting
                                y_min, y_max = ax.get_ylim()
                                y_range = y_max - y_min
                                
                                # Get base timepoint for conversion
                                def _ensure_dt(val):
                                    return val if isinstance(val, datetime) else datetime.strptime(val, '%Y-%m-%d %H:%M:%S')
                                base_t0 = _ensure_dt(timepoints[0])
                                
                                for period_name, period in series.time_periods.items():
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
        
        # Define update_time_label after update_image and update_plot are defined
        def update_time_label(change):
            nonlocal current_time_index
            current_time_index = change.new
            if self.relative_time_array:
                hours = int(self.relative_time_array[change.new])
                days = hours // 24
                remaining_hours = hours % 24
                time_label.value = f"{int(days)}d {int(remaining_hours)}h"
            update_plot()
            update_image()
        
        # Set up time slider observer and initialize label
        time_slider.observe(update_time_label, names='value')
        time_label.value = f"{int(initial_days)}d {int(initial_remaining_hours)}h"
        
        # Segmentation controls
        thresholding_input = widgets.FloatText(value=1.0, step=0.1, layout=widgets.Layout(width="60px"))
        ai_input = widgets.FloatText(value=0.7, step=0.05, layout=widgets.Layout(width="60px"))
        manual_btn = widgets.Button(description="Manual", layout=widgets.Layout(width="100px"))
        thresholding_btn = widgets.Button(description="Thresholding", layout=widgets.Layout(width="110px"))
        ai_btn = widgets.Button(description="AI", layout=widgets.Layout(width="100px"))
        delete_btn = widgets.Button(description="Delete Contour", button_style="danger", layout=widgets.Layout(width="120px"))
        
        def manual_segmentation(_):
            if not current_well or current_well not in self.spheroid_dict:
                return
            series = self.spheroid_dict[current_well]
            timepoint = self.time_points_array[time_slider.value]
            image = series.spheroid_image_dict[timepoint]
            ch = channel_dropdown.value
            try:
                image.segmentation_manual(ch)
                # Save contour to HDF5
                if hasattr(image, '_save_contour_to_hdf5'):
                    image._save_contour_to_hdf5()
                update_image()
                update_plot()
            except Exception as e:
                print(f"Manual segmentation failed: {e}")
        
        def thresholding_segmentation(_):
            if not current_well or current_well not in self.spheroid_dict:
                return
            series = self.spheroid_dict[current_well]
            timepoint = self.time_points_array[time_slider.value]
            image = series.spheroid_image_dict[timepoint]
            ch = channel_dropdown.value
            try:
                res = image.segmentation_thresholding(ch, thresholding_input.value)
                if isinstance(res, tuple) and len(res) == 2:
                    image.contour, image.contour_touches_border = res
                else:
                    image.contour = res
                # Save contour to HDF5
                if hasattr(image, '_save_contour_to_hdf5'):
                    image._save_contour_to_hdf5()
                update_image()
                update_plot()
            except Exception as e:
                print(f"Thresholding segmentation failed: {e}")
        
        def ai_segmentation(_):
            if not current_well or current_well not in self.spheroid_dict:
                return
            series = self.spheroid_dict[current_well]
            timepoint = self.time_points_array[time_slider.value]
            image = series.spheroid_image_dict[timepoint]
            ch = channel_dropdown.value
            try:
                image.segmentation(
                    methods=[('ai', ch)],
                    border_margin=5,
                    confidence=ai_input.value,
                )
                # Save contour to HDF5
                if hasattr(image, '_save_contour_to_hdf5'):
                    image._save_contour_to_hdf5()
                update_image()
                update_plot()
            except Exception as e:
                print(f"AI segmentation failed: {e}")
        
        def delete_contour(_):
            if not current_well or current_well not in self.spheroid_dict:
                return
            series = self.spheroid_dict[current_well]
            timepoint = self.time_points_array[time_slider.value]
            image = series.spheroid_image_dict[timepoint]
            image.contour = None
            image.contour_touches_border = False
            # Save deletion to HDF5 (remove contour from file)
            if hasattr(image, '_save_contour_to_hdf5'):
                image._save_contour_to_hdf5()  # This will handle None contour correctly
            update_image()
            update_plot()
        
        manual_btn.on_click(manual_segmentation)
        thresholding_btn.on_click(thresholding_segmentation)
        ai_btn.on_click(ai_segmentation)
        delete_btn.on_click(delete_contour)
        
        seg_title = widgets.HTML("<h4 style='color: #2c3e50; margin:0;'>Segmentation Options</h4>")
        seg_buttons = widgets.HBox(
            [
                manual_btn,
                widgets.HBox([thresholding_btn, thresholding_input]),
                widgets.HBox([ai_btn, ai_input]),
                delete_btn,
            ],
            layout=widgets.Layout(justify_content="flex-start", align_items="center", width="100%", gap="40px"),
        )
        seg_panel = widgets.VBox(
            [seg_title, seg_buttons],
            layout=widgets.Layout(
                margin="0px",
                padding="10px",
                border="none",
                width="100%",
            ),
        )
        
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
        image_output.layout.width = "100%"
        image_panel = widgets.VBox(
            [image_title, image_output, image_controls],
            layout=widgets.Layout(border="none", padding="0px", margin="0px 40px 0 0px", flex="1 1 0%", width="100%", min_width="0", overflow="hidden"),
        )
        
        # Right column: plot heading row (title + metric dropdown) + plot + segmentation controls
        plot_panel = widgets.VBox(
            [plot_header_row, plot_output],
            layout=widgets.Layout(
                border="none",
                padding="0px",
                margin="0px",
                flex="1 1 0%",
                width="100%",
                align_items="flex-start",
                justify_content="flex-start",
            ),
        )
        right_column = widgets.VBox(
            [plot_panel, seg_panel],
            layout=widgets.Layout(
                border="0px",
                border_radius="0px",
                padding="0px",
                margin="10px 0px 0px 0px",
                flex="1 1 0%",
                width="100%",
            ),
        )
        
        def update_view():
            # Update plate layout
            plate_layout.clear_output()
            with plate_layout:
                display(create_plate_layout())
            update_plot()
            update_image()
        
        # Set up observers
        channel_dropdown.observe(lambda change: update_image(), names='value')
        show_contour.observe(lambda change: update_image(), names='value')
        metric_dropdown.observe(lambda change: update_plot(), names='value')
        
        # Combine all elements
        # Header: replicate info + well selection side by side (well selection aligned right, no gap)
        # Wrap replicate_info in a container to make it flexible
        replicate_info_container = widgets.Box(
            [replicate_info],
            layout=widgets.Layout(flex='1 1 auto',margin='0 30px 0 0', overflow='hidden', min_width='0')
        )
        # Make replicate_info height match plate_layout height; no horizontal scroll
        header_section = widgets.HBox(
            [replicate_info_container, plate_layout],
            layout=widgets.Layout(margin='10px 0px ', gap='124px', align_items='stretch', justify_content='space-between', width='100%', overflow='hidden')
        )
        
        # Main row: image left, plot + segmentation right
        main_row = widgets.HBox(
            [image_panel, right_column],
            layout=widgets.Layout(gap="124px", width="100%", align_items="stretch"),
        )
        
        # Display everything
        root_widget = widgets.VBox([
            header_section,
            main_row
        ], layout=widgets.Layout(padding='10px', width='95%', max_width='1400px'))
        display(root_widget)

        def _initial_render():
            update_view()

        try:
            from IPython import get_ipython
            ip = get_ipython()
            if ip is not None and hasattr(ip, "kernel") and hasattr(ip.kernel, "io_loop"):
                ip.kernel.io_loop.add_callback(_initial_render)
            else:
                _initial_render()
        except Exception:
            _initial_render()

