from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable

import h5py
import pandas as pd

from .result import Result


class Experiment:
    """Lightweight experiment container for comparing multiple results.

    A :class:`Experiment` groups multiple :class:`Result` objects (biological
    replicates) that represent the same experimental setup. The class provides
    utilities to aggregate and compare metrics across results, particularly for
    comparing means of different results for the same conditions (collections).

    The class supports optional HDF5 persistence when a ``path`` is provided.
    For live-cell imaging workflows that require experiment tracking and specialized
    HDF5 persistence, use :class:`LiveCellExperiment` instead.
    """

    def __init__(
        self,
        name: str,
        results: Iterable[Result] | None = None,
        path: Path | str | None = None,
        description: str | None = None,
    ) -> None:
        """Create a new experiment container.

        Args:
            name: Display name of the experiment.
            results: Optional iterable of ``Result`` objects that will be attached
                immediately. Each result's ``name`` attribute is used as the key.
            path: Optional directory for the experiment HDF5 file. Can be a Path object
                or a string. If provided, the experiment will be saved to HDF5. The current
                working directory is used when omitted.
            description: Free-form metadata saved alongside the experiment.
        """
        self.name = name
        self.description = description
        self.created_at = datetime.now()
        self.modified_at = datetime.now()
        
        # Convert iterable to dict using result.name as key
        if results is None:
            self.results_dict: dict[str, Result] = {}
        elif isinstance(results, Iterable):
            self.results_dict = {result.name: result for result in results}
        else:
            raise TypeError(f"results must be Iterable[Result] or None, got {type(results)}")
        self._metric_cache: dict[str, pd.DataFrame] = {}
        
        # Node registration for Base class compatibility
        self._node_counters: dict[str, int] = {}
        
        # Prepare HDF5 storage if path is provided
        self.hdf5_path: Path | None = None
        if path is not None:
            self.hdf5_path = self._prepare_hdf5(path)
            self.save_metadata()
            # Save all results if they exist
            if self.results_dict:
                self.save_all()


    # ------------------------------------------------------------------ #
    # Result management
    # ------------------------------------------------------------------ #
    def add_result(self, result: Result) -> None:
        """Attach a new ``Result`` to this experiment."""
        if not isinstance(result, Result):
            raise TypeError("result must be an instance of Result")
        # Set experiment reference on result
        result.experiment = self
        # Set result index (position in results_dict)
        result._result_index = len(self.results_dict)
        self.results_dict[result.name] = result
        self._metric_cache.clear()
        self.modified_at = datetime.now()
        if self.hdf5_path:
            self.save_metadata()
            # Save the complete result hierarchy (uses result._result_index)
            result.save_to_hdf5(self.hdf5_path)

    def result(self, name: str) -> Result:
        """Get or create a result by name.
        
        If a result with the given name already exists, it is returned.
        Otherwise, a new result is created, added to the experiment, and returned.
        
        Args:
            name: Name of the result to get or create.
            
        Returns:
            The existing or newly created Result instance.
        """
        if name in self.results_dict:
            return self.results_dict[name]
        
        # Create new result
        new_result = Result(name=name)
        self.add_result(new_result)
        return new_result

    def remove_result(self, name: str) -> None:
        """Remove a result from the experiment."""
        if name in self.results_dict:
            del self.results_dict[name]
            self._metric_cache.clear()
        self.modified_at = datetime.now()
        if self.hdf5_path:
            self.save_metadata()
            # Re-save all to update indices
            self.save_all()

    # ------------------------------------------------------------------ #
    # Metric utilities
    # ------------------------------------------------------------------ #
    def metric(
        self,
        name: str = "radius",
        *,
        results: Iterable[str] | None = None,
        mean: bool = True,
        aggregate_axis: str | None = None,
        plot: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        """Calculate a metric across all attached results.

        Aggregates metrics from multiple results to compare the same conditions
        (collections) across biological replicates.

        Args:
            name: Metric forwarded to ``Result.metric``.
            results: Optional iterable to limit aggregation to a subset of
                result names.
            mean: Whether to average technical replicates inside each collection
                (forwarded to ``Result.metric``).
            aggregate_axis: Optional axis name for post-processing the combined
                DataFrame (``"result"`` or ``"time"``).
            plot: Whether to display a unified plot across all results.
            **kwargs: Additional keyword arguments passed to the underlying
                ``Result.metric`` call.
        """
        selected = (
            {key: self.results_dict[key] for key in results}
            if results is not None
            else self.results_dict
        )

        if not selected:
            raise ValueError("Experiment has no results.")

        cache_key = f"{name}|mean={mean}|agg={aggregate_axis}|{tuple(sorted(kwargs.items()))}|{tuple(sorted(selected.keys()))}"
        if cache_key in self._metric_cache:
            combined = self._metric_cache[cache_key]
        else:
            frames = []
            for result_name, result in selected.items():
                # Always call result.metric with plot=False to avoid individual plots
                df = result.metric(
                    name=name,
                    mean=mean,
                    plot=False,  # Never plot individual results
                    **{k: v for k, v in kwargs.items() if k != 'plot'},
                )
                if isinstance(df.columns, pd.MultiIndex):
                    suffix = df.columns.get_level_values(-1)
                else:
                    suffix = df.columns
                df.columns = pd.MultiIndex.from_product([[result_name], suffix])
                frames.append(df)

            combined = pd.concat(frames, axis=1).sort_index(axis=1)

            if aggregate_axis == "result":
                combined = combined.groupby(level=1, axis=1).mean()
            elif aggregate_axis == "time":
                combined = combined.groupby(combined.index).mean()

            self._metric_cache[cache_key] = combined

        # Create unified plot if requested
        if plot:
            self._plot_metric(combined, name, mean, **kwargs)

        return combined

    def summary(
        self,
        name: str = "radius",
        *,
        statistic: str = "last",
        **kwargs,
    ) -> pd.DataFrame:
        """Return one-row-per-result summaries for a metric.

        Args:
            name: Metric name forwarded to ``Result.summary``.
            statistic: Statistic to compute (``"last"``, ``"max"``, ``"min"``,
                ``"mean"``).
            **kwargs: Additional arguments forwarded to ``Result.summary``.

        Returns:
            DataFrame with one row per result and columns for each collection/condition.
        """
        frames = []
        for result_name, result in self.results_dict.items():
            frames.append(result.summary(name=name, statistic=statistic, **kwargs).rename(result_name))
        return pd.DataFrame(frames)

    def _plot_metric(
        self,
        df: pd.DataFrame,
        name: str,
        mean: bool,
        skip_nan: bool = True,
        **kwargs,
    ) -> None:
        """Create a unified plot for all results in this experiment."""
        import matplotlib.pyplot as plt
        import numpy as np

        plt.figure(figsize=(12, 6))
        x = np.array(df.index) / 24  # Convert to days

        if mean:
            # Plot mean ± std for each result/collection combination
            for result_name in df.columns.get_level_values(0).unique():
                result = self.results_dict[result_name]

                # Get all collection labels for this result
                collection_labels = df.columns.get_level_values(1).unique()
                if 'mean' in collection_labels and 'std' in collection_labels:
                    # Standard mean/std structure
                    mean_col = (result_name, 'mean')
                    std_col = (result_name, 'std')
                    if mean_col in df.columns and std_col in df.columns:
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
                        plt.plot(x_plot, mean_plot, 'o-', label=result_name)
                else:
                    # MultiIndex with (result, collection, stat) structure
                    for collection_label in collection_labels:
                        mean_col = (result_name, collection_label, 'mean')
                        std_col = (result_name, collection_label, 'std')
                        if mean_col in df.columns and std_col in df.columns:
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
                            plt.plot(x_plot, mean_plot, 'o-',
                                   label=f"{result_name}: {collection_label}")
        else:
            # Plot individual series from each result
            for result_name in df.columns.get_level_values(0).unique():
                result = self.results_dict[result_name]
                result_cols = [col for col in df.columns if col[0] == result_name]

                for col in result_cols:
                    y = df[col]
                    col_label = f"{result_name}: {col[1]}" if isinstance(col, tuple) and len(col) > 1 else str(col)

                    if skip_nan:
                        valid_mask = ~np.isnan(y)
                        plt.plot(x[valid_mask], y[valid_mask], 'o-',
                               label=col_label, alpha=0.7)
                    else:
                        plt.plot(x, y, 'o-', label=col_label, alpha=0.7)

        plt.grid(True, alpha=0.3)
        plt.xlabel('Time [d]')
        plt.ylabel(name)
        plt.title(f'{name} over time - {self.name}' + (' (mean ± std)' if mean else ''))

        # Only show legend if not too many series
        if not mean or len(df.columns.get_level_values(0).unique()) < 10:
            plt.legend()

        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------ #
    # HDF5 Persistence
    # ------------------------------------------------------------------ #
    def _prepare_hdf5(self, path: Path | str) -> Path:
        """Create the experiment HDF5 container, overwriting if it already exists."""
        base_path = Path(path)
        base_path.mkdir(parents=True, exist_ok=True)
        file_path = base_path / f"Experiment_{self.name}.h5"
        
        # Remove existing file if it exists to start fresh
        if file_path.exists():
            file_path.unlink()
        
        # Create new HDF5 file
        with h5py.File(file_path, "w") as hdf_file:
            hdf_file.attrs["name"] = self.name
            hdf_file.attrs["description"] = self.description or ""
            hdf_file.attrs["created_at"] = self.created_at.isoformat()
            hdf_file.attrs["modified_at"] = self.modified_at.isoformat()
            # Create Result group (compatible with LiveCellExperiment structure)
            hdf_file.create_group("Result", track_order=True)
        
        return file_path

    def save_metadata(self) -> None:
        """Save experiment metadata to HDF5."""
        if not self.hdf5_path:
            return
        
        with h5py.File(self.hdf5_path, "a") as hdf_file:
            # Update attributes
            hdf_file.attrs["name"] = self.name
            hdf_file.attrs["description"] = self.description or ""
            hdf_file.attrs["created_at"] = self.created_at.isoformat()
            hdf_file.attrs["modified_at"] = self.modified_at.isoformat()
    
    def save_all(self) -> None:
        """Save experiment and all results with complete hierarchy to HDF5."""
        if not self.hdf5_path:
            raise RuntimeError("Cannot save: experiment has no HDF5 path. Provide 'path' in __init__.")
        
        # Save experiment metadata
        self.save_metadata()
        
        # Track result indices (compatible with LiveCellExperiment structure)
        with h5py.File(self.hdf5_path, "a") as hdf_file:
            # Ensure Result group exists
            if "Result" not in hdf_file:
                hdf_file.create_group("Result", track_order=True)
            
            result_group = hdf_file["Result"]
            
            # Build mapping of result names to their correct indices
            # {name: index} for current results
            name_to_index = {name: idx for idx, name in enumerate(self.results_dict.keys())}
            
            # Track which result names are currently in results_dict
            current_result_names = set(self.results_dict.keys())
            
            # Build set of correct keys: {index-name}
            current_results = {f"{idx}-{name}" for name, idx in name_to_index.items()}
            
            # First pass: collect all existing keys that need to be removed
            keys_to_remove = []
            existing_results = list(result_group.keys())  # Use list to avoid modification during iteration
            
            for result_key in existing_results:
                # Try to parse key: "index-name"
                try:
                    index_str, name = result_key.split("-", 1)
                    existing_index = int(index_str)
                except (ValueError, IndexError):
                    # Can't parse - treat as invalid if not in current_results
                    if result_key not in current_results:
                        keys_to_remove.append(result_key)
                    continue
                
                # Check if name is in current results
                if name in current_result_names:
                    # Name exists - check if index is correct
                    correct_index = name_to_index[name]
                    correct_key = f"{correct_index}-{name}"
                    
                    # Remove if index is wrong (duplicate)
                    if result_key != correct_key:
                        keys_to_remove.append(result_key)
                else:
                    # Name not in current results - remove it
                    keys_to_remove.append(result_key)
            
            # Remove all invalid/duplicate keys
            for key in keys_to_remove:
                if key in result_group:
                    del result_group[key]
        
        # Update indices and save each result with its complete hierarchy
        for index, (result_name, result) in enumerate(self.results_dict.items()):
            result._result_index = index  # Update index in case it changed
            result.save_to_hdf5(self.hdf5_path)

    @classmethod
    def from_file(cls, filepath: Path | str, load_results: bool = True) -> "Experiment":
        """Load an experiment from an HDF5 file.
        
        Args:
            filepath: Path to the HDF5 file containing the experiment.
            load_results: If True, load complete Result objects with full hierarchy.
                If False, only load experiment metadata.
            
        Returns:
            Experiment instance loaded from the file.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Experiment file not found: {filepath}")
        
        experiment = object.__new__(cls)
        
        with h5py.File(filepath, "r") as hdf_file:
            # Load basic attributes
            experiment.name = hdf_file.attrs["name"]
            experiment.description = hdf_file.attrs.get("description", None)
            experiment.created_at = datetime.fromisoformat(hdf_file.attrs["created_at"])
            experiment.modified_at = datetime.fromisoformat(hdf_file.attrs["modified_at"])
            experiment.hdf5_path = filepath
            
            # Initialize storage
            experiment.results_dict = {}
            experiment._metric_cache = {}
            
            # Load results if requested
            if load_results and "Result" in hdf_file:
                result_group = hdf_file["Result"]
                # Get all keys and filter out any that are not actual result groups
                # (e.g., if "Result" itself appears as a key, skip it)
                all_keys = list(result_group.keys())
                result_keys = []
                for key in all_keys:
                    # Skip if this is the Result group itself (shouldn't happen, but check anyway)
                    if key == "Result":
                        continue
                    # Check if it's a valid result key (should be "index-name" format)
                    try:
                        result_item = result_group[key]
                        if isinstance(result_item, h5py.Group):
                            result_keys.append(key)
                    except (KeyError, AttributeError):
                        continue
                
                if not result_keys:
                    print("Warning: No valid result keys found in HDF5 file under 'Result' group")
                else:
                    print(f"Found {len(result_keys)} result key(s) in HDF5 file: {result_keys}")
                
                # Track which results we've already loaded (by name) to avoid duplicates
                loaded_result_names = set()
                
                # Add progress bar for loading results
                from tqdm import tqdm
                for result_key in tqdm(result_keys, desc="Loading results", unit="result"):
                    # Parse result key: "index-name"
                    try:
                        index_str, result_name = result_key.split("-", 1)
                        result_index = int(index_str)
                    except (ValueError, IndexError):
                        # Fallback: try to extract index from key
                        result_index = 0
                        result_name = result_key
                    
                    # Skip if we've already loaded a result with this name
                    if result_name in loaded_result_names:
                        continue
                    
                    # Check if result group has required attributes before attempting to load
                    result_item = None
                    try:
                        result_item = result_group[result_key]
                        if not isinstance(result_item, h5py.Group):
                            print(f"Warning: Result key '{result_key}' is not a group, skipping")
                            continue
                    except (KeyError, AttributeError) as e:
                        # Skip if we can't access the result group
                        print(f"Warning: Cannot access result group '{result_key}': {e}")
                        continue
                    
                    # Get name from attributes or use parsed name
                    actual_name = result_name
                    try:
                        if result_item and "name" in result_item.attrs:
                            # Try to decode if it's bytes
                            name_attr = result_item.attrs["name"]
                            if isinstance(name_attr, bytes):
                                actual_name = name_attr.decode("utf-8")
                            else:
                                actual_name = str(name_attr)
                    except Exception as e:
                        # If we can't read the name attribute, use parsed name
                        print(f"Warning: Could not read 'name' attribute for '{result_key}', using parsed name '{result_name}': {e}")
                        actual_name = result_name
                    
                    if not actual_name:
                        print(f"Warning: Result key '{result_key}' has no valid name, skipping")
                        continue
                    
                    try:
                        result = Result.from_file(
                            name=actual_name,
                            result_index=result_index,
                            hdf5_path=filepath,
                        )
                        # Set experiment reference on result (important for hdf5_path resolution)
                        result.experiment = experiment
                        experiment.results_dict[actual_name] = result
                        loaded_result_names.add(actual_name)
                    except Exception as e:
                        import traceback
                        print(f"Warning: Failed to load result '{actual_name}' (key: '{result_key}'): {e}")
                        # Only print full traceback if it's not a simple KeyError
                        if not isinstance(e, KeyError):
                            print(f"Traceback: {traceback.format_exc()}")
        
        return experiment

    # ------------------------------------------------------------------ #
    # Segmentation
    # ------------------------------------------------------------------ #
    def segmentation(
        self,
        methods: list | tuple = [('thresholding', 'fluorescence_green'), ('ai', 'brightfield')],
        reconstruct_border: bool = True,
        border_margin: int = 10,
        **kwargs,
    ) -> None:
        """
        Segment all collections across all results in this experiment.

        Args:
            methods: List/Tuple of segmentation specs, e.g.
                [('thresholding','fluorescence_green'), ('ai','brightfield')]
            reconstruct_border: Whether to reconstruct border if contour touches image border.
            border_margin: Margin (px) for border detection.
            **kwargs: Forwarded to ``SpheroidCollection.segmentation`` (e.g. threshold, use_yen, confidence).
        """
        import multiprocessing as mp
        from tqdm import tqdm
        
        if not self.results_dict:
            print("No results in experiment to segment.")
            return

        # Collect all series across all results for unified progress tracking
        all_series_args = []
        series_to_result = {}  # Map series_name -> (result, collection, cond_tuple, idx)
        
        for result_name, result in self.results_dict.items():
            for cond_tuple, idx, collection in result._iter_collections():
                active_series = collection.get_spheroids()
                if not active_series:
                    continue
                
                for series in active_series:
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
                    series_to_result[series.name] = (result, collection, cond_tuple, idx)

        if not all_series_args:
            print("No active spheroid series in experiment to segment.")
            return

        # Process all series in parallel with single progress bar
        from SpheroidPy.spheroid.spheroid_collection import SpheroidCollection
        num_cores = mp.cpu_count()
        with mp.Pool(processes=num_cores) as pool:
            results = list(tqdm(
                pool.imap(SpheroidCollection._process_segmentation_series, all_series_args),
                total=len(all_series_args),
                desc=f"Experiment '{self.name}'"
            ))

        # Update SpheroidImage instances with segmentation results
        total_success = 0
        total_images = 0
        for series_name, success, total, contour_data in results:
            total_success += success
            total_images += total
            
            result, collection, cond_tuple, idx = series_to_result[series_name]
            series = next(s for s in collection.get_spheroids() if s.name == series_name)
            for timepoint, data in contour_data.items():
                spheroid_image = series.spheroid_image_dict[timepoint]
                spheroid_image.contour = data['contour']
                spheroid_image.contour_touches_border = data['touches_border']

        # Save contours to HDF5 if collections have hdf5_path
        hdf5_paths = {coll.hdf5_path for result in self.results_dict.values() 
                     for _, _, coll in result._iter_collections() if coll.hdf5_path}
        for hdf5_path in hdf5_paths:
            import h5py
            with h5py.File(hdf5_path, 'a') as hdf_file:
                for series_name, _, _, contour_data in results:
                    for timepoint, data in contour_data.items():
                        try:
                            spheroid_image_group = hdf_file[data['hdf5_key']]
                            if 'contour' in spheroid_image_group:
                                del spheroid_image_group['contour']
                            spheroid_image_group.create_dataset('contour', data=data['contour'])
                            # Store as int for PyTables compatibility
                            touches_border_val = data['touches_border'] if data.get('touches_border') is not None else False
                            spheroid_image_group.attrs['touches_border'] = 1 if touches_border_val else 0
                        except KeyError:
                            pass  # Skip if HDF5 key doesn't exist

        print(f"\nSegmentation complete for experiment '{self.name}':")
        print(f"Successfully segmented {total_success}/{total_images} images across {len(all_series_args)} series")

    # ------------------------------------------------------------------ #
    # Interactive visualization
    # ------------------------------------------------------------------ #
    def show(self) -> None:
        """Interactive viewer for all results, collections and series in this experiment."""
        if not self.results_dict:
            print("No results in experiment.")
            return
        
        # Import required modules
        import os
        import base64
        import cv2
        import numpy as np
        import matplotlib.pyplot as plt
        import ipywidgets as widgets
        from IPython.display import display
        from SpheroidPy.spheroid.spheroid_collection import SpheroidCollection
        from SpheroidPy.spheroid.spheroid_series import SpheroidSeries
        
        # Helper to get active series of a collection (respects ignored list)
        def _active_series(coll: SpheroidCollection) -> list[SpheroidSeries]:
            return coll.get_spheroids()
        
        # Result dropdown
        result_names = list(self.results_dict.keys())
        result_dropdown = widgets.Dropdown(
            options=result_names,
            value=result_names[0],
            description="Result:",
            layout=widgets.Layout(min_width="170px", max_width="220px", flex="1 1 0%"),
            style={"description_width": "60px"},
        )
        
        def get_current_result():
            return self.results_dict[result_dropdown.value]
        
        # Build condition -> replicate mapping for current result
        def _build_cond_options(result):
            cond_options = []
            for cond_tuple, coll_list in result.collections.items():
                label = result._condition_label(cond_tuple)
                cond_options.append((label, cond_tuple, coll_list))
            return cond_options
        
        def _update_cond_mappings(result):
            """Update global cond mappings for current result."""
            nonlocal cond_options, cond_label_map, cond_to_reps
            cond_options = _build_cond_options(result)
            cond_label_map = {label: cond_tuple for label, cond_tuple, _ in cond_options}
            cond_to_reps = {cond_tuple: coll_list for _, cond_tuple, coll_list in cond_options}
        
        current_result = get_current_result()
        cond_options = []
        cond_label_map = {}
        cond_to_reps = {}
        _update_cond_mappings(current_result)
        
        if not cond_options:
            print(f"No collections in result '{current_result.name}'.")
            return
        
        condition_dropdown = widgets.Dropdown(
            options=[label for label, _, _ in cond_options],
            value=cond_options[0][0],
            description="Condition:",
            layout=widgets.Layout(min_width="140px", max_width="220px", flex="1 1 0%"),
            style={"description_width": "70px"},
        )
        
        def _rep_options(cond_tuple: tuple) -> list[str]:
            return [str(i + 1) for i in range(len(cond_to_reps[cond_tuple]))]
        
        replicate_dropdown = widgets.Dropdown(
            options=_rep_options(cond_options[0][1]),
            value="1",
            description="Replicate:",
            layout=widgets.Layout(min_width="120px", max_width="140px", flex="1 1 0%"),
            style={"description_width": "70px"},
        )
        
        def get_current_collection() -> SpheroidCollection:
            result = get_current_result()
            # Rebuild cond_options for current result
            cond_opts = _build_cond_options(result)
            cond_label_map_local = {label: cond_tuple for label, cond_tuple, _ in cond_opts}
            cond_to_reps_local = {cond_tuple: coll_list for _, cond_tuple, coll_list in cond_opts}
            cond_tuple = cond_label_map_local[condition_dropdown.value]
            rep_idx = int(replicate_dropdown.value) - 1
            return cond_to_reps_local[cond_tuple][rep_idx]
        
        def _series_options(coll: SpheroidCollection) -> list[str]:
            return [s.name for s in _active_series(coll)]
        
        current_collection = get_current_collection()
        current_series_list = _active_series(current_collection)
        if not current_series_list:
            print(f"No active spheroid series in collection '{current_collection.name}'.")
            return
        
        series_dropdown = widgets.Dropdown(
            options=_series_options(current_collection),
            value=_series_options(current_collection)[0],
            description="Series:",
            layout=widgets.Layout(min_width="170px", max_width="220px", flex="1 1 0%"),
            style={"description_width": "60px"},
        )
        
        def get_current_series() -> SpheroidSeries:
            coll = get_current_collection()
            series_name = series_dropdown.value
            return next(s for s in _active_series(coll) if s.name == series_name)
        
        # Channel selector, contour toggle, time slider
        channels = ["brightfield", "fluorescence_green", "fluorescence_red", "fluorescence_blue"]
        channel_dropdown = widgets.Dropdown(
            options=channels,
            value="brightfield",
            description="Channel:",
            layout=widgets.Layout(width="250px"),
        )
        show_contour = widgets.Checkbox(value=True, description="Show Contour", layout=widgets.Layout(width="220px"))
        
        # Placeholder sliders and labels (updated when series changes)
        time_slider = widgets.IntSlider(min=0, max=0, value=0, description="Time:", continuous_update=True, layout=widgets.Layout(min_width="170px", flex="2 1 0%"), style={"description_width": "50px"}, readout=False)
        time_label = widgets.Label(layout=widgets.Layout(min_width="60px", margin="0 0 0 -2px"))
        
        def _ensure_dt(val):
            return val if isinstance(val, datetime) else datetime.strptime(val, "%Y-%m-%d %H:%M:%S")
        
        def _update_time_controls(series: SpheroidSeries) -> None:
            timepoints = sorted(series.spheroid_image_dict.keys())
            time_slider.max = max(len(timepoints) - 1, 0)
            time_slider.value = 0
            if timepoints:
                t0 = _ensure_dt(timepoints[0])
                delta = _ensure_dt(timepoints[0]) - t0
                days = delta.days
                hours = delta.seconds // 3600
                time_label.value = f"{days}d {hours}h"
            else:
                time_label.value = ""
        
        # Containers for image and plot outputs
        image_output = widgets.Output()
        plot_output = widgets.Output()
        
        def get_base64_image_html(image_path, height=60):
            try:
                with open(image_path, "rb") as f:
                    data = base64.b64encode(f.read()).decode("utf-8")
                return f"<img src='data:image/png;base64,{data}' style='height:{height}px; float:right;'>"
            except Exception as e:  # pragma: no cover - best-effort UI helper
                print(f"Logo konnte nicht geladen werden: {e}")
                return ""
        
        logo_path = "/Users/cedric/Documents/Labor/SpheroidPy/SpheroidPy/static/images/logo_full_horizontal.png"
        logo_html = get_base64_image_html(logo_path) if os.path.exists(logo_path) else ""
        
        # Intelligent name truncation
        def truncate_name(name: str, max_chars: int = 10) -> str:
            """Truncate name intelligently:
            - If first word > 10 chars: show first 10 chars + "..."
            - If multiple words: show all words that fit in 10 chars, then "..."
            """
            words = name.split()
            if not words:
                return name
            
            # If first word is longer than max_chars, truncate it
            if len(words[0]) > max_chars:
                return name[:max_chars] + "..."
            
            # Otherwise, collect words that fit
            result = []
            current_length = 0
            for word in words:
                # Add 1 for space if not first word
                needed = len(word) + (1 if result else 0)
                if current_length + needed <= max_chars:
                    result.append(word)
                    current_length += needed
                else:
                    break
            
            if result:
                truncated = " ".join(result)
                # If we didn't show all words, add ellipsis
                if len(result) < len(words) or len(truncated) < len(name):
                    return truncated + "..."
                return truncated
            else:
                return name[:max_chars] + "..."
        
        display_name = truncate_name(self.name, max_chars=10)
        name_html = widgets.HTML(f"<div style='margin:0; width:100%; overflow:hidden;'><h2 style='margin:0; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; width:100%;'>{display_name}</h2></div>")
        subtitle_html = widgets.HTML("<div style='color:#888; font-weight:400; font-size:1.1em; margin-bottom:2px;'>Experiment</div>")
        name_block = widgets.VBox(
                                    [name_html, subtitle_html],
                                    layout=widgets.Layout(
                                        margin="0px",
                                        padding="0px",
                                        min_width="120px",
                                        max_width="220px",
                                        flex="0 0 130px",     # reservierter Titel-Slot
                                        overflow="hidden"
                                    )
                                )

        
        # Build header row with dropdowns (result, condition, replicate, series, time)
        series_controls = widgets.HBox(
            [result_dropdown, condition_dropdown, replicate_dropdown, series_dropdown, time_slider, time_label],
            layout=widgets.Layout(justify_content="center", align_items="center", gap="12px", width="100%"),
        )
        header_row = widgets.HBox(
            [name_block, series_controls, widgets.HTML(logo_html)],
            layout=widgets.Layout(justify_content="space-between", align_items="center", width="100%"),
        )
        header_container = widgets.VBox(
            [header_row],
            layout=widgets.Layout(border="1px solid #bbb", border_radius="14px", padding="8px 12px", margin="0px", width="100%"),
        )
        
        # Image controls
        image_controls = widgets.HBox(
            [channel_dropdown, show_contour],
            layout=widgets.Layout(justify_content="center", align_items="center", width="100%", gap="0px"),
        )
        image_output.layout.width = "100%"
        image_panel = widgets.VBox(
            [image_output, image_controls],
            layout=widgets.Layout(border="1px solid #bbb", border_radius="10px", padding="0px", margin="0px", flex="1 1 0%", width="100%"),
        )
        
        def update_time_label(idx: int, series: SpheroidSeries) -> None:
            timepoints = sorted(series.spheroid_image_dict.keys())
            if not timepoints or idx >= len(timepoints):
                time_label.value = ""
                return
            t0 = _ensure_dt(timepoints[0])
            t = _ensure_dt(timepoints[idx])
            delta = t - t0
            days = delta.days
            hours = delta.seconds // 3600
            time_label.value = f"{days}d {hours}h"
        
        def update_image(*args) -> None:
            series = get_current_series()
            timepoints = sorted(series.spheroid_image_dict.keys())
            idx = time_slider.value
            if idx >= len(timepoints):
                return
            
            sph_img = series.spheroid_image_dict[timepoints[idx]]
            channel = channel_dropdown.value
            try:
                if channel == "brightfield":
                    img = sph_img.brightfield()
                elif channel.startswith("fluorescence_"):
                    color = channel.split("_")[1]
                    img = sph_img.fluorescence(color)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                else:
                    img = sph_img.brightfield()
            except Exception as e:  # pragma: no cover - UI best effort
                image_output.clear_output()
                with image_output:
                    print(f"Error loading image: {e}")
                return
            
            image_output.clear_output()
            with image_output:
                container_width = 600
                aspect = sph_img.image_size[1] / sph_img.image_size[0] if sph_img.image_size[0] else 1
                fig, ax = plt.subplots(figsize=(8, 8 * aspect))
                ax.imshow(img, extent=[0, sph_img.image_size[0], 0, sph_img.image_size[1]])
                if show_contour.value and sph_img.contour is not None:
                    try:
                        contour = sph_img.scaled_contour
                        xs = contour[:, 0]
                        ys = contour[:, 1]
                        if xs[0] != xs[-1] or ys[0] != ys[-1]:
                            xs = np.r_[xs, xs[:1]]
                            ys = np.r_[ys, ys[:1]]
                        ax.plot(xs, ys, "w-", linewidth=2)
                    except Exception as e:  # pragma: no cover - UI best effort
                        print(f"Error drawing contour: {e}")
                ax.set_title(f"{timepoints[idx]} | {channel} | {series.name}")
                ax.axis("off")
                plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
                plt.show()
        
        def update_plot(*args) -> None:
            plot_output.clear_output()
            with plot_output:
                series = get_current_series()
                timepoints = sorted(series.spheroid_image_dict.keys())
                if not timepoints:
                    print("No timepoints available.")
                    return
                def _ensure_dt_local(val):
                    return val if isinstance(val, datetime) else datetime.strptime(val, "%Y-%m-%d %H:%M:%S")
                base_t0 = _ensure_dt_local(timepoints[0])
                times = [(_ensure_dt_local(tp) - base_t0).total_seconds() / 24 / 3600 for tp in timepoints]
                radii = [series.spheroid_image_dict[tp].radius for tp in timepoints]
                idx = time_slider.value
                current_time = times[idx] if idx < len(times) else 0
                current_radius = radii[idx] if idx < len(radii) else None
                
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
                ax.set_title(f"Radius over time - {series.name}")
                ax.grid(True, alpha=0.3)
                if current_radius is not None and not np.isnan(current_radius):
                    ax.legend()
                plt.tight_layout()
                plt.show()
        
        # Segmentation controls (manual / thresholding / AI)
        thresholding_input = widgets.FloatText(value=1.0, step=0.1, layout=widgets.Layout(width="60px"))
        ai_input = widgets.FloatText(value=0.7, step=0.05, layout=widgets.Layout(width="60px"))
        manual_btn = widgets.Button(description="Manual", layout=widgets.Layout(width="100px"))
        thresholding_btn = widgets.Button(description="Thresholding", layout=widgets.Layout(width="110px"))
        ai_btn = widgets.Button(description="AI", layout=widgets.Layout(width="100px"))
        delete_btn = widgets.Button(description="Delete Contour", button_style="danger", layout=widgets.Layout(width="120px"))
        
        def manual_segmentation(_):
            series = get_current_series()
            timepoints = sorted(series.spheroid_image_dict.keys())
            idx = time_slider.value
            if idx >= len(timepoints):
                return
            sph_img = series.spheroid_image_dict[timepoints[idx]]
            dropdown_val = channel_dropdown.value
            ch = dropdown_val if dropdown_val == "brightfield" or dropdown_val.startswith("fluorescence_") else "brightfield"
            try:
                sph_img.segmentation_manual(ch)
                update_image()
            except Exception as e:  # pragma: no cover - UI best effort
                print(f"Manual segmentation failed: {e}")
        
        def thresholding_segmentation(_):
            series = get_current_series()
            timepoints = sorted(series.spheroid_image_dict.keys())
            idx = time_slider.value
            if idx >= len(timepoints):
                return
            sph_img = series.spheroid_image_dict[timepoints[idx]]
            dropdown_val = channel_dropdown.value
            ch = dropdown_val if dropdown_val == "brightfield" or dropdown_val.startswith("fluorescence_") else "brightfield"
            try:
                res = sph_img.segmentation_thresholding(ch, thresholding_input.value)
                if isinstance(res, tuple) and len(res) == 2:
                    sph_img.contour, sph_img.contour_touches_border = res
                else:
                    sph_img.contour = res
                update_image()
            except Exception as e:  # pragma: no cover - UI best effort
                print(f"Thresholding segmentation failed: {e}")
        
        def ai_segmentation(_):
            series = get_current_series()
            timepoints = sorted(series.spheroid_image_dict.keys())
            idx = time_slider.value
            if idx >= len(timepoints):
                return
            sph_img = series.spheroid_image_dict[timepoints[idx]]
            dropdown_val = channel_dropdown.value
            ch = dropdown_val if dropdown_val in ["brightfield", "fluorescence_green", "fluorescence_red", "fluorescence_blue"] else "brightfield"
            try:
                res = sph_img.segmentation_detectron(ch, ai_input.value)
                if isinstance(res, tuple) and len(res) == 2:
                    sph_img.contour, sph_img.contour_touches_border = res
                else:
                    sph_img.contour = res
                update_image()
            except Exception as e:  # pragma: no cover - UI best effort
                print(f"AI segmentation failed: {e}")
        
        def delete_contour(_):
            series = get_current_series()
            timepoints = sorted(series.spheroid_image_dict.keys())
            idx = time_slider.value
            if idx >= len(timepoints):
                return
            sph_img = series.spheroid_image_dict[timepoints[idx]]
            sph_img.contour = None
            sph_img.contour_touches_border = False
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
        
        def _refresh_all_controls():
            """Refresh all dropdowns and controls when result changes."""
            result = get_current_result()
            _update_cond_mappings(result)
            
            if not cond_options:
                print(f"No collections in result '{result.name}'.")
                return
            
            # Update condition dropdown
            cond_labels = [label for label, _, _ in cond_options]
            condition_dropdown.options = cond_labels
            condition_dropdown.value = cond_labels[0] if cond_labels else None
            
            # Update replicate dropdown
            if cond_options:
                cond_tuple = cond_options[0][1]
                rep_opts = _rep_options(cond_tuple)
                replicate_dropdown.options = rep_opts
                replicate_dropdown.value = rep_opts[0] if rep_opts else None
            
            # Update series dropdown
            coll = get_current_collection()
            series_opts = _series_options(coll)
            series_dropdown.options = series_opts
            series_dropdown.value = series_opts[0] if series_opts else None
            
            _update_time_controls(get_current_series())
            update_image()
            update_plot()
        
        def _refresh_series_for_current_collection():
            coll = get_current_collection()
            options = _series_options(coll)
            series_dropdown.options = options
            series_dropdown.value = options[0] if options else None
            _update_time_controls(get_current_series())
            update_image()
            update_plot()
        
        def on_result_change(*args) -> None:
            _refresh_all_controls()
        
        def on_condition_change(*args) -> None:
            cond_tuple = cond_label_map[condition_dropdown.value]
            replicate_dropdown.options = _rep_options(cond_tuple)
            replicate_dropdown.value = replicate_dropdown.options[0] if replicate_dropdown.options else None
            _refresh_series_for_current_collection()
        
        def on_replicate_change(*args) -> None:
            _refresh_series_for_current_collection()
        
        def on_series_change(*args) -> None:
            _update_time_controls(get_current_series())
            update_image()
            update_plot()
        
        # Observe changes
        result_dropdown.observe(on_result_change, names="value")
        condition_dropdown.observe(on_condition_change, names="value")
        replicate_dropdown.observe(on_replicate_change, names="value")
        series_dropdown.observe(on_series_change, names="value")
        time_slider.observe(lambda change: (update_time_label(time_slider.value, get_current_series()), update_image(), update_plot()), names="value")
        channel_dropdown.observe(lambda change: update_image(), names="value")
        show_contour.observe(lambda change: update_image(), names="value")
        
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
        
        main_row = widgets.HBox(
            [image_panel, right_column],
            layout=widgets.Layout(gap="0px", width="100%", align_items="stretch"),
        )
        
        vbox = widgets.VBox(
            [header_container, main_row],
            layout=widgets.Layout(width="95%", max_width="1400px", align_items="stretch"),
        )
        outer_box = widgets.HBox([vbox], layout=widgets.Layout(width="100%", overflow_x="hidden"))
        
        # Initialize controls and display
        _update_time_controls(get_current_series())
        update_time_label(0, get_current_series())
        update_image()
        update_plot()
        display(outer_box)

    # ------------------------------------------------------------------ #
    # Utility
    # ------------------------------------------------------------------ #
    def hdf5_tree(self) -> None:
        """Print the complete HDF5 file structure with all elements.
        
        Displays a tree-like representation of the HDF5 file structure,
        showing all groups, datasets, and their attributes.
        """
        if not self.hdf5_path or not self.hdf5_path.exists():
            print(f"No HDF5 file available at: {self.hdf5_path}")
            return
        
        print(f'HDF5 Structure for: {self.hdf5_path}\n')
        print('=' * 80)
        
        def print_tree(name, obj, indent=0):
            """Recursively print HDF5 structure."""
            prefix = "  " * indent
            
            if isinstance(obj, h5py.Group):
                # Print group name
                print(f"{prefix}📁 {name}/")
                
                # Print group attributes if any
                if obj.attrs:
                    for attr_name in sorted(obj.attrs.keys()):
                        attr_value = obj.attrs[attr_name]
                        # Truncate long values
                        if isinstance(attr_value, (bytes, str)):
                            value_str = str(attr_value)
                            if len(value_str) > 50:
                                value_str = value_str[:47] + "..."
                        else:
                            value_str = str(attr_value)
                            if len(value_str) > 50:
                                value_str = value_str[:47] + "..."
                        print(f"{prefix}    └─ {attr_name}: {value_str}")
                
                # Recursively print children
                for key in sorted(obj.keys()):
                    print_tree(key, obj[key], indent + 1)
                    
            elif isinstance(obj, h5py.Dataset):
                # Print dataset name and info
                shape_str = f"shape={obj.shape}" if obj.shape else "scalar"
                dtype_str = f"dtype={obj.dtype}"
                print(f"{prefix}📄 {name} [{shape_str}, {dtype_str}]")
                
                # Print dataset attributes if any
                if obj.attrs:
                    for attr_name in sorted(obj.attrs.keys()):
                        attr_value = obj.attrs[attr_name]
                        # Truncate long values
                        if isinstance(attr_value, (bytes, str)):
                            value_str = str(attr_value)
                            if len(value_str) > 50:
                                value_str = value_str[:47] + "..."
                        else:
                            value_str = str(attr_value)
                            if len(value_str) > 50:
                                value_str = value_str[:47] + "..."
                        print(f"{prefix}    └─ {attr_name}: {value_str}")
        
        with h5py.File(self.hdf5_path, "r") as hdf_file:
            # Print root attributes
            if hdf_file.attrs:
                print("Root Attributes:")
                for attr_name in sorted(hdf_file.attrs.keys()):
                    attr_value = hdf_file.attrs[attr_name]
                    if isinstance(attr_value, (bytes, str)):
                        value_str = str(attr_value)
                    else:
                        value_str = str(attr_value)
                    if len(value_str) > 70:
                        value_str = value_str[:67] + "..."
                    print(f"  {attr_name}: {value_str}")
                print()
            
            # Print tree structure
            for key in sorted(hdf_file.keys()):
                print_tree(key, hdf_file[key])
        
        print('=' * 80)
    
    def __repr__(self) -> str:  # pragma: no cover - debug helper
        results = ", ".join(self.results_dict.keys()) or "<empty>"
        hdf5_info = f", hdf5={self.hdf5_path}" if self.hdf5_path else ""
        return f"Experiment(name={self.name}, results=[{results}]{hdf5_info})"


