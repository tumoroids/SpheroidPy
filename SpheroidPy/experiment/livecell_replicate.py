from __future__ import annotations
from typing import TYPE_CHECKING
import logging

from pathlib import Path
import numpy as np
from tqdm import tqdm
import h5py
import pandas as pd
from datetime import datetime
import re

from SpheroidPy.spheroid.spheroid_image import SpheroidImage
from SpheroidPy.spheroid.spheroid_series import SpheroidSeries
from SpheroidPy.spheroid.spheroid_collection import SpheroidCollection

from SpheroidPy.utils.file_management import collect_data

import multiprocessing as mp

if TYPE_CHECKING:
    from SpheroidPy.experiment.result import Result
    from SpheroidPy.experiment.platemap import Platemap

logger = logging.getLogger("SpheroidPy.experiment.livecell_replicate")


def _process_well_from_file_standalone(well_data):
    """Standalone helper function to process a well when loading from file.

    Args:
        well_data: Tuple of (well, timepoint_data) where timepoint_data contains
                  all necessary information extracted from HDF5

    Returns:
        Tuple of (well, spheroid_series, well_results)
    """
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
                timepoint: int | str | None = None) -> pd.DataFrame:
        """
        Compare collections across a specific condition, grouping by other conditions.

        For the specified condition, finds all collections that differ only in that condition
        but are identical in all other conditions. Each such group becomes a column in the
        resulting DataFrame, with rows representing the values of the specified condition.

        Args:
            condition: Name of the condition to vary (e.g., 'concentration', 'temperature')
            metric: Metric to calculate (default: 'radius')
            plot: Whether to plot the results (default: True)
            timepoint: Optional specific timepoint to calculate metrics for.
                - If int: Timepoint in hours (relative time)
                - If str: Timepoint as datetime string (e.g., '2025-03-27 20:00:00')
                - If None: Average across all timepoints (default)

        Returns:
            DataFrame with:
                - Index: Values of the specified condition
                - Columns: One per group of collections that differ only in other conditions
                - Values: Metric values (mean ± std across technical replicates within each collection)
                - If timepoint is specified, returns mean and std for that timepoint
                - If timepoint is None, returns mean across all timepoints
        """
        import matplotlib.pyplot as plt

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

        # Handle timepoint selection
        target_timepoint = None
        if timepoint is not None:
            # Convert timepoint to appropriate format
            if isinstance(timepoint, int):
                # Timepoint in hours (relative time)
                target_timepoint = timepoint
            elif isinstance(timepoint, str):
                # Timepoint as datetime string
                try:
                    target_timepoint = datetime.strptime(timepoint, '%Y-%m-%d %H:%M:%S')
                except ValueError:
                    try:
                        target_timepoint = datetime.fromisoformat(timepoint)
                    except ValueError:
                        print(
                            f"Could not parse timepoint '{timepoint}'. Expected format: 'YYYY-MM-DD HH:MM:SS' or ISO format.")
                        return pd.DataFrame()

        # For plotting and simpler output, create a DataFrame with condition values as index
        # and groups as columns, where each cell contains the metric value at a specific timepoint
        summary_data_mean = {}
        summary_data_std = {}

        for group_label, group_data in result_data.items():
            group_df_mean = group_data['mean']
            group_df_std = group_data['std']

            if not group_df_mean.empty:
                if target_timepoint is not None:
                    # Find the closest timepoint
                    if isinstance(target_timepoint, int):
                        # Convert hours to timepoint index (assuming index is in hours)
                        # Find closest timepoint in hours
                        closest_idx = None
                        min_diff = float('inf')
                        for idx in group_df_mean.index:
                            if isinstance(idx, (int, float)):
                                diff = abs(idx - target_timepoint)
                            else:
                                # Try to convert to hours
                                try:
                                    if isinstance(idx, str):
                                        # Try to parse as datetime and convert to hours
                                        dt = datetime.strptime(idx, '%Y-%m-%d %H:%M:%S')
                                        # Calculate hours from first timepoint
                                        if self.relative_time_array and self.time_points_array:
                                            first_time = datetime.strptime(self.time_points_array[0],
                                                                           '%Y-%m-%d %H:%M:%S')
                                            hours = (dt - first_time).total_seconds() / 3600
                                            diff = abs(hours - target_timepoint)
                                        else:
                                            continue
                                    else:
                                        continue
                                except:
                                    continue

                            if diff < min_diff:
                                min_diff = diff
                                closest_idx = idx

                        if closest_idx is not None:
                            summary_data_mean[group_label] = group_df_mean.loc[closest_idx]
                            summary_data_std[group_label] = group_df_std.loc[closest_idx]
                    else:
                        # datetime object - find closest timepoint
                        closest_idx = None
                        min_diff = float('inf')
                        for idx in group_df_mean.index:
                            try:
                                if isinstance(idx, str):
                                    dt = datetime.strptime(idx, '%Y-%m-%d %H:%M:%S')
                                elif isinstance(idx, datetime):
                                    dt = idx
                                else:
                                    continue

                                diff = abs((dt - target_timepoint).total_seconds())
                                if diff < min_diff:
                                    min_diff = diff
                                    closest_idx = idx
                            except:
                                continue

                        if closest_idx is not None:
                            summary_data_mean[group_label] = group_df_mean.loc[closest_idx]
                            summary_data_std[group_label] = group_df_std.loc[closest_idx]
                else:
                    # Calculate mean across all timepoints for each condition value
                    summary_data_mean[group_label] = group_df_mean.mean(axis=0)
                    # For std across timepoints, we calculate the std of means
                    summary_data_std[group_label] = group_df_std.mean(axis=0)

        if not summary_data_mean:
            print("No summary data could be calculated.")
            return pd.DataFrame()

        result_df_mean = pd.DataFrame(summary_data_mean)
        result_df_std = pd.DataFrame(summary_data_std)
        result_df_mean.index.name = condition
        result_df_std.index.name = condition

        # Plot if requested
        if plot and not result_df_mean.empty:
            plt.figure(figsize=(10, 6))

            for col in result_df_mean.columns:
                mean_values = result_df_mean[col]
                std_values = result_df_std[col]

                # Plot with error bars
                plt.errorbar(
                    result_df_mean.index,
                    mean_values,
                    yerr=std_values,
                    fmt='o-',
                    label=col,
                    linewidth=2,
                    markersize=8,
                    capsize=5,
                    capthick=2
                )

            timepoint_str = f" at {timepoint}" if timepoint is not None else " (mean across timepoints)"
            plt.xlabel(condition)
            plt.ylabel(metric)
            plt.title(f'{metric} vs {condition}{timepoint_str} (mean ± std)')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.show()

        # Return DataFrame with mean values (std can be accessed via result_df_std if needed)
        # For convenience, we could return a MultiIndex DataFrame, but for now return mean
        return result_df_mean

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
            'show_contour': False,
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
        
        # Create header with replicate info (height will match well selection)
        # HTML part with gray background that extends to cover the full area
        replicate_info_html = widgets.HTML(
            value=f"""
            <div style='padding: 10px 10px 0 10px; background-color: #f0f0f0; border-radius: 5px 0 0 0; margin-right: 0; height: 100%;'>
                <h2 style='color: #2c3e50; margin: 0;'>Replicate: {self.name}</h2>
                <div style='margin-left: 20px; margin-top: 10px;'>
                    <p><b>Layout:</b> {self.layout} wells ({rows}×{cols})</p>
                    <p><b>Wells with data:</b> {len(self.spheroid_dict)}</p>
                    <p><b>Time Range:</b> {self.time_points_array[0] if self.time_points_array else 'N/A'} to {self.time_points_array[-1] if self.time_points_array else 'N/A'}</p>
                </div>
            </div>
            """,
            layout=widgets.Layout(background_color='#f0f0f0', width='100%')
        )
        
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
                width='100%'
            )
        )
        
        # Combine HTML and time controls
        # height='100%' ensures it matches the well selection height via align_items='stretch'
        replicate_info = widgets.VBox(
            [replicate_info_html, time_controls_container],
            layout=widgets.Layout(
                background_color='#f0f0f0',
                border_radius='5px 0 0 5px',
                margin_right='0',
                width='100%',
                height='100%'
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
                            font_size='6px',
                            padding='0px'
                        ),
                        style=widgets.ButtonStyle(
                            button_color='#FFD700' if well == current_well else ('#ADD8E6' if has_data else '#F0F0F0')
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
                layout=widgets.Layout(margin='5px', overflow='hidden', width='300px')
            )
        
        plate_layout = widgets.Output(layout=widgets.Layout(width='300px', flex='0 0 auto'))
        
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
            layout=widgets.Layout(flex='0 0 auto', min_width='180px')
        )
        
        def on_channel_change(change):
            widget_states['channel_1'] = change.new
        channel_dropdown.observe(on_channel_change, names='value')
        
        show_contour = widgets.Checkbox(
            value=widget_states['show_contour'],
            description='Show Contour',
            layout=widgets.Layout(flex='0 0 auto', width='auto')
        )
        
        def on_contour_change(change):
            widget_states['show_contour'] = change.new
        show_contour.observe(on_contour_change, names='value')
        
        def update_image():
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
                        
                        ax.set_title(f"{timepoint} | {channel} | {current_well}")
                        ax.axis("off")
                        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
                        plt.show()
                except Exception as e:
                    print(f"Error loading image: {e}")
        
        def update_plot():
            plot_output.clear_output(wait=True)
            with plot_output:
                if current_well and current_well in self.spheroid_dict:
                    try:
                        series = self.spheroid_dict[current_well]
                        timepoints = sorted(series.spheroid_image_dict.keys())
                        radii = [series.spheroid_image_dict[tp].radius for tp in timepoints]
                        
                        # Convert timepoints to relative times
                        if self.relative_time_array and len(self.relative_time_array) == len(timepoints):
                            times = [t / 24 for t in self.relative_time_array]
                        else:
                            # Fallback: use index as time
                            times = list(range(len(timepoints)))
                        
                        current_time = times[current_time_index] if current_time_index < len(times) else 0
                        current_radius = radii[current_time_index] if current_time_index < len(radii) else None
                        
                        fig, ax = plt.subplots(figsize=(6, 4.1))
                        ax.plot(times, radii, color="grey")
                        
                        if current_radius is not None and not np.isnan(current_radius):
                            ax.plot(
                                current_time,
                                current_radius,
                                ".",
                                markersize=10,
                                label=f"{current_radius:.1f} µm @ {current_time:.2f} d",
                                color="#e29266",
                            )
                        ax.set_xlabel("Time [d]")
                        ax.set_ylabel("Radius [µm]")
                        ax.set_title(f"Radius over time - {current_well}")
                        ax.grid(True, alpha=0.3)
                        if current_radius is not None and not np.isnan(current_radius):
                            ax.legend()
                        plt.tight_layout()
                        plt.show()
                    except Exception as e:
                        plt.figure(figsize=(6, 4.1))
                        plt.text(0.5, 0.5, f'No radius data available\n{str(e)}', 
                                ha='center', va='center')
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
                res = image.segmentation_detectron(ch, ai_input.value)
                if isinstance(res, tuple) and len(res) == 2:
                    image.contour, image.contour_touches_border = res
                else:
                    image.contour = res
                # Save contour to HDF5
                if hasattr(image, '_save_contour_to_hdf5'):
                    image._save_contour_to_hdf5()
                update_image()
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
                border="1px solid #ccc",
                border_radius="5px",
                width="100%",
            ),
        )
        
        # Image controls (only channel and contour, time slider moved to replicate info)
        # Optimize spacing: no gap, full width utilization
        image_controls = widgets.HBox(
            [channel_dropdown, show_contour],
            layout=widgets.Layout(
                justify_content="flex-start", 
                align_items="center", 
                width="100%", 
                gap="0px",
                padding="5px 10px"
            ),
        )
        image_output.layout.width = "100%"
        image_panel = widgets.VBox(
            [image_output, image_controls],
            layout=widgets.Layout(border="1px solid #bbb", border_radius="10px", padding="0px", margin="0px", flex="1 1 0%", width="100%"),
        )
        
        # Right column: plot + segmentation controls
        plot_panel = widgets.VBox(
            [plot_output],
            layout=widgets.Layout(
                border="1px solid #bbb",
                border_radius="10px",
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
                margin="0px",
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
        
        # Initial update
        update_view()
        
        # Set up observers
        channel_dropdown.observe(lambda change: update_image(), names='value')
        show_contour.observe(lambda change: update_image(), names='value')
        
        # Combine all elements
        # Header: replicate info + well selection side by side (well selection aligned right, no gap)
        # Wrap replicate_info in a container to make it flexible
        replicate_info_container = widgets.Box(
            [replicate_info],
            layout=widgets.Layout(flex='1 1 auto', overflow='hidden')
        )
        # Make replicate_info height match plate_layout height
        # align_items='stretch' ensures both boxes have the same height
        header_section = widgets.HBox(
            [replicate_info_container, plate_layout],
            layout=widgets.Layout(margin='10px 0', gap='0px', align_items='stretch', justify_content='space-between', width='100%')
        )
        
        # Main row: image left, plot + segmentation right
        main_row = widgets.HBox(
            [image_panel, right_column],
            layout=widgets.Layout(gap="0px", width="100%", align_items="stretch"),
        )
        
        # Display everything
        display(widgets.VBox([
            header_section,
            main_row
        ], layout=widgets.Layout(padding='10px', width='95%', max_width='1400px')))

