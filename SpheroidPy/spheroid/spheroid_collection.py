from __future__ import annotations
from typing import TYPE_CHECKING
from dataclasses import dataclass
import h5py
import numpy as np
import pandas as pd
from datetime import datetime
import copy
import matplotlib.pyplot as plt
import logging

import os
import base64
import ipywidgets as widgets
from IPython.display import display
import cv2

from SpheroidPy.utils.color_palettes import CARTO_SEQUENTIAL

logger = logging.getLogger("SpheroidPy.spheroid_collection")

if TYPE_CHECKING:
    from SpheroidPy.experiment.result import Result
    from SpheroidPy.spheroid.spheroid_series import SpheroidSeries

class SpheroidCollection:
    """
    Collection of spheroid series, i.e. representing biological replicates.

    Attributes
    ----------
    name : str
        Name of the condition this collection represents.
    spheroid_series : list of SpheroidSeries
        List of spheroid series in this collection.
    ignored_spheroids : set of str
        Names of spheroid series to exclude from analysis.
    _result : Result or None
        Reference to the Result object this collection belongs to.
    hdf5_path : str or None
        Path to HDF5 file for storage.
    """

    def __init__(self, name: str, spheroid_list: list[SpheroidSeries] | None = None, 
                 result: Result | None = None, hdf5_path: str | None = None,
                 color: str | None = None):
        """
        Initialize a SpheroidCollection.

        Parameters
        ----------
        name : str
            Name of the experimental condition this collection represents.
        spheroid_list : list of SpheroidSeries, optional
            Initial list of SpheroidSeries objects to add to the collection.
        result : Result, optional
            Reference to the Result object this collection belongs to.
        hdf5_path : str, optional
            Path to an HDF5 file for storing collection data.
        color : str, optional
            Color palette name from CARTO_SEQUENTIAL. If None, will be assigned
            automatically when added to a Result.

        Notes
        -----
        - If `spheroid_list` is provided, the spheroids are added using `add_spheroids`.
        - `ignored_spheroids` is initialized as an empty set.
        - `color` can be a palette name (e.g., "Burgundy", "Green") or None.
        """
        self.name = name
        self.hdf5_path = hdf5_path
        self._result = result
        self.color = color  # Palette name or None
        
        self.spheroid_series = []
        self.ignored_spheroids = set()
        
        if spheroid_list is not None:
            self.add_spheroids(spheroid_list)

    def add_spheroids(self, spheroid_list: list[SpheroidSeries]):
        """Add spheroid series to this collection.
        
        Args:
            spheroid_list: List of SpheroidSeries to add
        """
        if not spheroid_list:
            print(f"Warning: Empty spheroid list for condition '{self.name}'")
            return
        
        # Check for duplicates by name to avoid adding the same series multiple times
        existing_names = {s.name for s in self.spheroid_series}
        new_series = [s for s in spheroid_list if s.name not in existing_names]
        
        if new_series:
            self.spheroid_series.extend(new_series)
        elif len([s for s in spheroid_list if s.name in existing_names]) > 0:
            # Some series were duplicates, but we silently skip them
            pass
    
    def add_spheroid(self, spheroid_series: SpheroidSeries):
        """Add spheroid series to this collection.
        
        Args:
            spheroid_series: SpheroidSeries to add
        """
        # Check for duplicates by name to avoid adding the same series multiple times
        existing_names = {s.name for s in self.spheroid_series}
        if spheroid_series.name not in existing_names:
            self.spheroid_series.append(spheroid_series)

    def get_spheroids(self, time_period: str | None = None) -> list[SpheroidSeries]:
        """Get list of active spheroids, optionally filtered by time period.
        
        Args:
            time_period: Optional time period name to filter by
            
        Returns:
            List of SpheroidSeries objects
        """
        # Filter out ignored spheroids
        active_spheroids = [s for s in self.spheroid_series 
                          if s.name not in self.ignored_spheroids]
        
        if time_period:
            # Create filtered copies with only timepoints in period
            filtered_spheroids = []
            for spheroid in active_spheroids:
                filtered = copy.copy(spheroid)
                filtered.spheroid_image_dict = spheroid.get_images_in_period(time_period)
                if filtered.spheroid_image_dict:
                    filtered_spheroids.append(filtered)
            return filtered_spheroids
        
        return active_spheroids

    def ignore_spheroid(self, spheroid_series_name: str):
        """Mark a spheroid series as ignored for analysis."""
        self.ignored_spheroids.add(spheroid_series_name)

    def unignore_spheroid(self, spheroid_series_name: str):
        """Remove a spheroid series from the ignored list."""
        self.ignored_spheroids.discard(spheroid_series_name)

    def get_replicate_stats(self) -> pd.DataFrame:
        """Get statistics about replicates in collection.
        
        Returns:
            DataFrame with columns:
            - total_spheroids: Total number of spheroid series
            - active_spheroids: Number not ignored
        """
        active = len(self.get_spheroids())
        total = len(self.spheroid_series)
        
        return pd.DataFrame([{
            'total_spheroids': total,
            'active_spheroids': active,
            'ignored_spheroids': total - active
        }])

    @property
    def result(self):
        """Get the Result associated with this collection."""
        if not self.spheroid_series:
            return None
        return self.spheroid_series[0]._result

    def __repr__(self) -> str:
        stats = self.get_replicate_stats().iloc[0]
        return (f"SpheroidCollection: {self.name}\n"
                f"Total spheroids: {stats['total_spheroids']}\n"
                f"Active spheroids: {stats['active_spheroids']}\n"
                f"Ignored spheroids: {stats['ignored_spheroids']}")

    def merge_collection(self, other_collection: SpheroidCollection):
        """Merge another SpheroidCollection into this one.
        
        Args:
            other_collection: SpheroidCollection to merge into this one
        
        Raises:
            ValueError: If collections have different conditions
        """
        if self.name != other_collection.name:
            raise ValueError(f"Cannot merge collections with different conditions: "
                           f"{self.name} vs {other_collection.name}")
        
        self.spheroid_series.extend(other_collection.spheroid_series)
        self.ignored_spheroids.update(other_collection.ignored_spheroids)

    def save_time_periods_to_hdf5(self):
        """Save time periods to HDF5 file for all spheroid series."""
        for spheroid in self.get_spheroids():
            spheroid.save_time_periods_to_hdf5()

    def load_time_periods_from_hdf5(self):
        """Load time periods from HDF5 file for all spheroid series."""
        for spheroid in self.get_spheroids():
            spheroid.load_time_periods_from_hdf5()

    def set_time_period(self, name: str,
                       start_time: str | datetime | None = None,
                       end_time: str | datetime | None = None,
                       description: str | None = None):
        """Set time period for all spheroid series."""
        for spheroid in self.get_spheroids():
            try:
                spheroid.add_time_period(
                    name=name,
                    start_time=start_time,
                    end_time=end_time,
                    description=description
                )
            except Exception as e:
                print(f"Failed to set time period for {spheroid.name}: {str(e)}")
        self.save_time_periods_to_hdf5()
                        
    def update_time_period(self, name: str,
                          start_time: str | datetime | None = None,
                          end_time: str | datetime | None = None,
                          description: str | None = None):
        """Update time period for all spheroid series."""
        for spheroid in self.get_spheroids():
            try:
                spheroid.update_time_period(
                    name=name,
                    start_time=start_time,
                    end_time=end_time,
                    description=description
                )
            except Exception as e:
                print(f"Failed to update time period for {spheroid.name}: {str(e)}")
        self.save_time_periods_to_hdf5()
                        
    def remove_time_period(self, name: str):
        """Remove time period from all spheroid series."""
        for spheroid in self.get_spheroids():
            spheroid.remove_time_period(name)
        self.save_time_periods_to_hdf5()

    def metric(self, name: str = 'radius', mean: bool = False,
                      interpolate: bool = True, ignore_border: bool = True,
                      skip_nan: bool = True, plot: bool = False,
                      timepoint: float | None = None,
                      time_period: str | None = None) -> pd.DataFrame:
        """Calculate metrics for all spheroids in this collection.
        
        Args:
            name: Metric to calculate ('radius', 'area', 'fluorescence_*')
            mean: Whether to average across spheroids
            interpolate: Whether to interpolate missing timepoints
            ignore_border: Whether to exclude spheroids touching image border
            skip_nan: Whether to exclude NaN values from plots
            plot: Whether to display plot of results
            timepoint: Optional specific timepoint to calculate metrics for
            
        Returns:
            DataFrame with timepoints as index and metrics as columns
        """
        # Get metrics for each spheroid series
        all_dfs = []
        for spheroid_series in self.get_spheroids():
            df = spheroid_series.metric(
                name=name,
                interpolate=interpolate,
                ignore_border=ignore_border,
                plot=False,
                timepoint=timepoint,
                time_period=time_period
            )
            all_dfs.append(df)
            
        if not all_dfs:
            return pd.DataFrame()
            
        # Combine all series
        result_df = pd.concat(all_dfs, axis=1)
        
        if mean:
            # Calculate mean and std
            mean_df = result_df.mean(axis=1)
            std_df = result_df.std(axis=1)
            
            # Create MultiIndex columns
            condition = self.name
            result_df = pd.concat([
                mean_df.rename((condition, 'mean')),
                std_df.rename((condition, 'std'))
            ], axis=1)
            
            # Additional interpolation for mean/std if requested
            if interpolate and timepoint is None:
                result_df = pd.DataFrame({
                    (condition, 'mean'): result_df[(condition, 'mean')].interpolate(
                        method='linear', axis=0, limit_direction='both'),
                    (condition, 'std'): result_df[(condition, 'std')].interpolate(
                        method='linear', axis=0, limit_direction='both')
                })
        
        # Plot if requested
        if plot:
            plt.figure(figsize=(10, 6))
            x = np.array(result_df.index) / 24  # Convert to days
            
            # Get color for this collection
            plot_color = None
            if self.color and self.color in CARTO_SEQUENTIAL:
                palette = CARTO_SEQUENTIAL[self.color]
                # Use darkest color (first) when mean=True, otherwise use lighter colors for individual series
                if mean:
                    plot_color = palette[0]  # Darkest color
                else:
                    # Use a mid-range color for individual series (index 3-4)
                    plot_color = palette[min(4, len(palette) - 1)]
            
            if mean:
                condition = self.name
                mean_array = result_df[(condition, 'mean')]
                std_array = result_df[(condition, 'std')]
                
                if skip_nan:
                    valid_mask = ~np.isnan(mean_array)
                    x_plot = x[valid_mask]
                    mean_plot = mean_array[valid_mask]
                    std_plot = std_array[valid_mask]
                else:
                    x_plot = x
                    mean_plot = mean_array
                    std_plot = std_array
                
                fill_color = plot_color if plot_color else 'gray'
                line_color = plot_color if plot_color else 'black'
                
                plt.fill_between(x_plot, 
                               mean_plot - std_plot,
                               mean_plot + std_plot,
                               alpha=0.3,
                               color=fill_color)
                plt.plot(x_plot, mean_plot, 'o-', label=condition, color=line_color)
            else:
                # Plot individual series with colors from the palette
                if self.color and self.color in CARTO_SEQUENTIAL:
                    palette = CARTO_SEQUENTIAL[self.color]
                    num_colors = len(palette)
                else:
                    palette = None
                    num_colors = 0
                
                for idx, col in enumerate(result_df.columns):
                    y = result_df[col]
                    # Use color from palette if available, cycling through
                    if palette:
                        series_color = palette[idx % num_colors]
                    else:
                        series_color = None
                    
                    if skip_nan:
                        valid_mask = ~np.isnan(y)
                        if series_color:
                            plt.plot(x[valid_mask], y[valid_mask], 'o-', label=col, alpha=0.7, color=series_color)
                        else:
                            plt.plot(x[valid_mask], y[valid_mask], 'o-', label=col, alpha=0.7)
                    else:
                        if series_color:
                            plt.plot(x, y, 'o-', label=col, alpha=0.7, color=series_color)
                        else:
                            plt.plot(x, y, 'o-', label=col, alpha=0.7)
            
            plt.grid(True, alpha=0.3)
            plt.xlabel('Time [d]')
            
            # Use plot_name_dict if available
            if self.result and name in self.result.plot_name_dict:
                plt.ylabel(rf'{self.result.plot_name_dict[name]}')
            else:
                plt.ylabel(name)
                
            if not mean or len(result_df.columns) < 10:  # Only show legend if not too many series
                plt.legend()
                
            plt.title(f'{name} over time' + (' (mean ± std)' if mean else ''))
            plt.tight_layout()
            plt.show()
            
        return result_df

    def series_metric_table(
        self,
        metric_name: str,
        value_keys: list[str] | None = None,
        time_period: str | None = None,
    ) -> pd.DataFrame:
        """Collect stored analysis metrics for each spheroid series.

        Args:
            metric_name: Name passed to ``SpheroidSeries.store_analysis_metrics``.
            value_keys: Optional subset of metric keys to include.
            time_period: Optional time-period label to filter metrics by. When
                omitted the returned DataFrame contains a multi-index of
                ``(series, time_period)`` rows so different windows can be
                compared side-by-side.

        Returns:
            DataFrame indexed by series (and optionally time period) with one
            column per requested metric value.
        """
        rows: list[dict] = []
        selected_keys = list(value_keys) if value_keys is not None else None

        def _display_period_label(label: str | None) -> str:
            return label if label is not None else "full_series"

        for spheroid in self.get_spheroids():
            if time_period is not None:
                metrics = spheroid.get_analysis_metrics(metric_name, time_period=time_period)
                period_entries = [(time_period, metrics)] if metrics else []
            else:
                metrics_map = spheroid.get_analysis_metrics(metric_name)
                period_entries = metrics_map.items() if metrics_map else []

            for period_label, metrics in period_entries:
                if not metrics:
                    continue
                if selected_keys is None:
                    selected_keys = list(metrics.keys())
                row = {'series': spheroid.name}
                if time_period is None:
                    row['time_period'] = _display_period_label(period_label)
                for key in selected_keys or []:
                    row[key] = metrics.get(key)
                rows.append(row)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        index_cols = ['series'] if time_period is not None else ['series', 'time_period']
        df = df.set_index(index_cols)
        metric_cols = selected_keys or []
        return df[metric_cols]

    def analysis_metric(
        self,
        metric_name: str,
        value_keys: list[str] | None = None,
        time_period: str | None = None,
    ) -> pd.DataFrame:
        """
        Calculate per-series values as well as mean and standard error for a stored analysis metric.

        Parameters
        ----------
        metric_name : str
            Name that was passed to ``SpheroidSeries.store_analysis_metrics`` (e.g. ``"necrotic_radius"``).
        value_keys : list[str], optional
            Optional subset of metric keys to include. When omitted all available keys are used.
        time_period : str, optional
            Restrict the aggregation to metrics that were recorded for a specific time period. If ``None``,
            values across all stored time periods are used.

        Returns
        -------
        pandas.DataFrame
            Table containing one row per spheroid series (and time period, if multiple exist) followed by
            two summary rows labelled ``summary_mean`` and ``summary_sem``. The latter rows hold the
            column-wise mean and standard error of the mean, respectively.
        """
        table = self.series_metric_table(
            metric_name,
            value_keys=value_keys,
            time_period=time_period,
        )
        if table.empty:
            return table

        means = table.mean(axis=0, skipna=True)
        sem = table.sem(axis=0, skipna=True)

        if isinstance(table.index, pd.MultiIndex):
            level_names = list(table.index.names)
            summary_tuples = [
                tuple(["mean"] + [""] * (len(level_names) - 1)),
                tuple(["standard error of mean"] + [""] * (len(level_names) - 1)),
            ]
            summary_index = pd.MultiIndex.from_tuples(summary_tuples, names=level_names)
        else:
            summary_index = pd.Index(
                ["mean", "standard error of mean"],
                name=table.index.name,
            )

        summary_df = pd.DataFrame([means, sem], index=summary_index)
        combined = pd.concat([table, summary_df])
        return combined

    def segmentation(self,
                     methods: list | tuple = [('thresholding', 'fluorescence_green'), ('ai', 'brightfield')],
                     reconstruct_border: bool = True,
                     border_margin: int = 10,
                     show_progress: bool = True,
                     **kwargs) -> tuple[int, int]:
        """
        Segment all SpheroidSeries in this collection using multiprocessing.

        Args:
            methods: List/Tuple of segmentation specs, e.g.
                [('thresholding','fluorescence_green'), ('ai','brightfield')]
            reconstruct_border: Whether to reconstruct border if contour touches image border
            border_margin: Margin (px) for border detection
            show_progress: Whether to show progress bar (default: True). Set to False when called from higher-level methods.
            **kwargs: Method-specific parameters:
                For thresholding:
                    threshold: Intensity threshold multiplier (default: 1.35)
                    use_yen: Whether to use Yen's method (default: True)
                For AI:
                    confidence: Detection confidence threshold (default: 0.65)
        
        Returns:
            Tuple of (total_success, total_images) for aggregation at higher levels.
        """
        import multiprocessing as mp
        from tqdm import tqdm
        
        if not self.spheroid_series:
            if show_progress:
                print("No spheroid series in collection to segment.")
            return (0, 0)

        # Get active spheroid series
        active_series = self.get_spheroids()
        if not active_series:
            if show_progress:
                print("No active spheroid series in collection to segment.")
            return (0, 0)

        # Prepare arguments for parallel processing
        series_args = []
        for series in active_series:
            # Get timepoints and prepare spheroid images dict for this series
            timepoints = list(series.spheroid_image_dict.keys())
            spheroid_images = series.spheroid_image_dict
            
            series_args.append((
                series.name, 
                timepoints,
                spheroid_images,
                methods, 
                reconstruct_border,
                border_margin,
                kwargs
            ))

        # Process series in parallel
        num_cores = mp.cpu_count()
        with mp.Pool(processes=num_cores) as pool:
            if show_progress:
                results = list(tqdm(
                    pool.imap(self._process_segmentation_series, series_args),
                    total=len(series_args),
                    desc=f"Collection '{self.name}'"
                ))
            else:
                results = list(pool.imap(self._process_segmentation_series, series_args))

        # Summarize results
        total_success = sum(success for _, success, _, _ in results)
        total_images = sum(total for _, _, total, _ in results)

        if show_progress:
            print(f"\nSegmentation complete for collection '{self.name}':")
            print(f"Successfully segmented {total_success}/{total_images} images across {len(active_series)} series")

        # Update SpheroidImage instances with segmentation results
        for series_name, _, _, contour_data in results:
            series = next(s for s in active_series if s.name == series_name)
            for timepoint, data in contour_data.items():
                spheroid_image = series.spheroid_image_dict[timepoint]
                spheroid_image.contour = data['contour']
                spheroid_image.contour_touches_border = data['touches_border']

        # Save contours to HDF5 if collection has hdf5_path
        if self.hdf5_path:
            import h5py
            with h5py.File(self.hdf5_path, 'a') as hdf_file:
                for series_name, _, _, contour_data in results:
                    for timepoint, data in contour_data.items():
                        # Save to HDF5
                        spheroid_image_group = hdf_file[data['hdf5_key']]
                        if 'contour' in spheroid_image_group:
                            del spheroid_image_group['contour']
                        spheroid_image_group.create_dataset('contour', data=data['contour'])
                        # Store as int for PyTables compatibility
                        touches_border_val = data['touches_border'] if data.get('touches_border') is not None else False
                        spheroid_image_group.attrs['touches_border'] = 1 if touches_border_val else 0
        
        return (total_success, total_images)

    @staticmethod
    def _process_segmentation_series(args):
        """Process segmentation for a single spheroid series.

        Args:
            args: Tuple containing:
                - series_name: Series identifier
                - timepoints: List of timepoints for this series
                - spheroid_images: Dictionary mapping timepoints to spheroid images
                - methods: Segmentation methods specification
                - reconstruct_border: Whether to reconstruct border
                - border_margin: Margin in pixels for border detection
                - kwargs: Additional segmentation parameters

        Returns:
            Tuple of (series_name, success_count, total_count, contour_data)
        """
        series_name, timepoints, spheroid_images, methods, reconstruct_border, border_margin, kwargs = args
        success_count = 0
        total_count = 0
        contour_data = {}

        for timepoint in timepoints:
            if timepoint not in spheroid_images:
                continue
                
            spheroid_image = spheroid_images[timepoint]
            total_count += 1  # Count all images we attempt to segment
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
            except Exception as e:
                print(f"Error processing {series_name} at {timepoint}: {str(e)}")
                # total_count already incremented, success_count not incremented
                continue

        return series_name, success_count, total_count, contour_data

    def show(self):
        """Interactive visualization of all spheroid series in the collection with series selector."""
        
        if not self.spheroid_series:
            print("No spheroid series in collection.")
            return

        # Get active spheroid series
        active_series = self.get_spheroids()
        if not active_series:
            print("No active spheroid series in collection.")
            return

        # Series selector dropdown
        series_names = [s.name for s in active_series]
        series_dropdown = widgets.Dropdown(
            options=series_names, 
            value=series_names[0], 
            description='Series:'
        )
        
        # Get current series
        def get_current_series():
            selected_name = series_dropdown.value
            return next(s for s in active_series if s.name == selected_name)

        # Initialize with first series
        current_series = get_current_series()
        
        # Timepoints and channels
        timepoints = sorted(current_series.spheroid_image_dict.keys())
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

        # Header with collection name, series selector, time slider, and logo
        logo_path = '/Users/cedric/Documents/Labor/SpheroidPy/SpheroidPy/static/images/logo_full_horizontal.png'
        if os.path.exists(logo_path):
            logo_html = get_base64_image_html(logo_path)
        else:
            logo_html = ""
        
        name_html = widgets.HTML(f"<h2 style='margin:0'>{self.name}</h2>")
        subtitle_html = widgets.HTML("<div style='color:#888; font-weight:400; font-size:1.1em; margin-bottom:2px;'>SpheroidCollection Visualisation</div>")
        name_block = widgets.VBox([name_html, subtitle_html], layout=widgets.Layout(margin='0px', padding='0px'))
        
        # Series selector and time controls
        series_controls = widgets.HBox([series_dropdown, time_slider, time_label], 
                                     layout=widgets.Layout(justify_content='center', align_items='center'))
        
        header_row = widgets.HBox([
            name_block,
            series_controls,
            widgets.HTML(logo_html)
        ], layout=widgets.Layout(justify_content='space-between', align_items='center', width='100%'))
        
        header_container = widgets.VBox([header_row], layout=widgets.Layout(
            border='1px solid #bbb', border_radius='10px', padding='0px', margin='0px', width='100%'))

        # Image display
        image_output = widgets.Output()
        image_controls = widgets.HBox([channel_dropdown, show_contour], 
                                    layout=widgets.Layout(justify_content='center', align_items='center', width='100%', gap='0px'))
        image_output.layout.width = '100%'
        
        def update_image(*args):
            # Get current series and update timepoints if series changed
            nonlocal current_series, timepoints
            new_series = get_current_series()
            if new_series != current_series:
                current_series = new_series
                timepoints = sorted(current_series.spheroid_image_dict.keys())
                time_slider.max = len(timepoints) - 1
                time_slider.value = 0
            
            idx = time_slider.value
            channel = channel_dropdown.value
            update_time_label(idx)
            
            if idx >= len(timepoints):
                return
                
            sph_img = current_series.spheroid_image_dict[timepoints[idx]]
            try:
                if channel == 'brightfield':
                    img = sph_img.brightfield()
                elif channel.startswith('fluorescence_'):
                    color = channel.split('_')[1]
                    img = sph_img.fluorescence(color)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                else:
                    img = sph_img.brightfield()
            except Exception as e:
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
                        # visually close by linking last to first
                        xs = contour[:,0]
                        ys = contour[:,1]
                        if xs[0] != xs[-1] or ys[0] != ys[-1]:
                            xs = np.r_[xs, xs[:1]]
                            ys = np.r_[ys, ys[:1]]
                        ax.plot(xs, ys, 'w-', linewidth=2)
                    except Exception as e:
                        print(f"Error drawing contour: {e}")
                ax.set_title(f"{timepoints[idx]} | {channel} | {current_series.name}")
                ax.axis('off')
                plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
                plt.show()
                
        image_panel = widgets.VBox([image_output, image_controls], layout=widgets.Layout(
            border='1px solid #bbb', border_radius='10px', padding='0px', margin='0px', flex='1 1 0%', width='100%'))

        # Radius plot
        plot_output = widgets.Output()
        def update_plot(*args):
            plot_output.clear_output()
            with plot_output:
                current_series = get_current_series()
                timepoints = sorted(current_series.spheroid_image_dict.keys())
                
                def _ensure_dt(val):
                    return val if isinstance(val, datetime) else datetime.strptime(val, '%Y-%m-%d %H:%M:%S')
                base_t0 = _ensure_dt(timepoints[0])
                times = [(_ensure_dt(tp) - base_t0).total_seconds()/24/3600 for tp in timepoints]
                radii = [current_series.spheroid_image_dict[tp].radius for tp in timepoints]
                # convert None to np.nan for plotting
                radii = [(np.nan if r is None else r) for r in radii]
                idx = time_slider.value
                current_time = times[idx] if idx < len(times) else 0
                current_radius = radii[idx] if idx < len(radii) else None
                
                fig, ax = plt.subplots(figsize=(6, 4.1))
                ax.plot(times, radii, color='grey')
                # plot current marker only if value is finite
                if current_radius is not None and not np.isnan(current_radius):
                    ax.plot(current_time, current_radius, '.', markersize=10, 
                           label=f'{current_radius:.1f} µm @ {current_time:.2f} d', color='#e29266')
                ax.set_xlabel('Time [d]')
                ax.set_ylabel('Radius [µm]')
                ax.set_title(f'Radius over time - {current_series.name}')
                ax.grid(True, alpha=0.3)
                # show legend only if we added the current marker with label
                if current_radius is not None and not np.isnan(current_radius):
                    ax.legend()
                plt.tight_layout()
                plt.show()
                
        plot_panel = widgets.VBox([plot_output], layout=widgets.Layout(
            border='1px solid #bbb', border_radius='10px', padding='0px', margin='0px', flex='1 1 0%', width='100%', align_items='flex-start', justify_content='flex-start'))

        # Segmentation options
        thresholding_input = widgets.FloatText(value=1.0, step=0.1, layout=widgets.Layout(width='60px'))
        ai_input = widgets.FloatText(value=0.7, step=0.05, layout=widgets.Layout(width='60px'))
        manual_btn = widgets.Button(description='Manual', layout=widgets.Layout(width='100px'))
        thresholding_btn = widgets.Button(description='Thresholding', layout=widgets.Layout(width='100px'))
        ai_btn = widgets.Button(description='AI', layout=widgets.Layout(width='100px'))
        delete_btn = widgets.Button(description='Delete Contour', button_style='danger', layout=widgets.Layout(width='120px'))

        def manual_segmentation(b):
            current_series = get_current_series()
            idx = time_slider.value
            timepoints = sorted(current_series.spheroid_image_dict.keys())
            if idx >= len(timepoints):
                return
            sph_img = current_series.spheroid_image_dict[timepoints[idx]]
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
            except Exception as e:
                print(f"Manual segmentation failed: {e}")
                
        def thresholding_segmentation(b):
            current_series = get_current_series()
            idx = time_slider.value
            timepoints = sorted(current_series.spheroid_image_dict.keys())
            if idx >= len(timepoints):
                return
            sph_img = current_series.spheroid_image_dict[timepoints[idx]]
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
            except Exception as e:
                print(f"Thresholding segmentation failed: {e}")
                
        def ai_segmentation(b):
            current_series = get_current_series()
            idx = time_slider.value
            timepoints = sorted(current_series.spheroid_image_dict.keys())
            if idx >= len(timepoints):
                return
            sph_img = current_series.spheroid_image_dict[timepoints[idx]]
            try:
                dropdown_val = channel_dropdown.value
                ch = dropdown_val if dropdown_val in ['brightfield', 'fluorescence_green', 'fluorescence_red', 'fluorescence_blue'] else 'brightfield'
                res = sph_img.segmentation_detectron(ch, ai_input.value)
                if isinstance(res, tuple) and len(res) == 2:
                    sph_img.contour, sph_img.contour_touches_border = res
                else:
                    sph_img.contour = res
                # Save contour to HDF5
                if hasattr(sph_img, '_save_contour_to_hdf5'):
                    sph_img._save_contour_to_hdf5()
                update_image()
            except Exception as e:
                print(f"AI segmentation failed: {e}")
                
        def delete_contour(b):
            current_series = get_current_series()
            idx = time_slider.value
            timepoints = sorted(current_series.spheroid_image_dict.keys())
            if idx >= len(timepoints):
                return
            sph_img = current_series.spheroid_image_dict[timepoints[idx]]
            sph_img.contour = None
            sph_img.contour_touches_border = False
            # Save deletion to HDF5 (remove contour from file)
            if hasattr(sph_img, '_save_contour_to_hdf5'):
                sph_img._save_contour_to_hdf5()  # This will handle None contour correctly
            update_image()

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
        ], layout=widgets.Layout(margin='0px', padding='10px', border='1px solid #ccc', border_radius='5px', width='100%'))

        # Layout callbacks
        def on_any_change(*args):
            update_image()
            update_plot()
            
        time_slider.observe(lambda change: on_any_change(), names='value')
        channel_dropdown.observe(lambda change: update_image(), names='value')
        show_contour.observe(lambda change: update_image(), names='value')
        series_dropdown.observe(lambda change: on_any_change(), names='value')

        # Time label update
        def update_time_label(idx):
            current_series = get_current_series()
            timepoints = sorted(current_series.spheroid_image_dict.keys())
            if idx >= len(timepoints):
                return
            def _ensure_dt(val):
                return val if isinstance(val, datetime) else datetime.strptime(val, '%Y-%m-%d %H:%M:%S')
            t0 = _ensure_dt(timepoints[0])
            t = _ensure_dt(timepoints[idx])
            delta = t - t0
            days = delta.days
            hours = delta.seconds // 3600
            time_label.value = f"{days}d {hours}h"

        # Right column: Plot with Segmentation Options below
        right_column = widgets.VBox([
            plot_panel,
            seg_panel
        ], layout=widgets.Layout(
            border='0px', border_radius='0px', padding='0px', margin='0px', flex='1 1 0%', width='100%'))

        # Main row: Image left, Plot+Seg right
        main_row = widgets.HBox([image_panel, right_column], layout=widgets.Layout(
            gap='0px', width='100%', align_items='stretch'))

        # Overall layout: Header + Main row
        vbox = widgets.VBox([
            header_container,
            main_row
        ], layout=widgets.Layout(width='95%', max_width='1400px', align_items='stretch'))

        # Wrapper for left-aligned display in notebook
        outer_box = widgets.HBox([vbox], layout=widgets.Layout(width='100%', overflow_x='hidden'))

        # Initial update
        update_time_label(0)
        update_image()
        update_plot()
        display(outer_box)

