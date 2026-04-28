from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Callable
from dataclasses import dataclass
import h5py
import numpy as np
import pandas as pd
from datetime import datetime
import copy
import matplotlib.pyplot as plt
import matplotlib
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

    def metric(self, name: str = 'radius',
                      mean: bool = False,
                      timepoint: float | None = None,
                      time_period: str | None = None,
                      interpolate: bool = False,
                      ignore_border: bool = False,
                      plot: bool = False,
                      skip_nan: bool = True,
                      ax: Optional[plt.Axes] = None,
                      plot_kwargs: Optional[dict] = None,
                      savepath: Optional[str] = None) -> pd.DataFrame | tuple[pd.DataFrame, plt.Axes]:
        """Calculate metrics for all spheroids in this collection.
        
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
        
        mean : bool, default=False
            Whether to average across spheroids (technical replicates) within this collection.
            When True, calculates mean ± std across all spheroid series and returns MultiIndex columns
            (collection_name, 'mean') and (collection_name, 'std'). When False, returns one column per spheroid series.
        
        timepoint : float | None, default=None
            Optional specific timepoint to calculate metrics for. If None, returns values for all timepoints.
            Accepted format: float (relative time in hours, e.g., 24.0 for 24 hours after first timepoint).
        
        time_period : str | None, default=None
            Optional time period name to limit analysis to specific time range.
            Must be a time period defined via ``time_period()`` method. If None, analyzes all available timepoints.
        
        interpolate : bool, default=True
            Whether to interpolate missing timepoints. If True, missing values are linearly
            interpolated. If False, data is left as-is with NaN values.
        
        ignore_border : bool, default=True
            Whether to exclude spheroids touching image border when calculating metrics.
            If True, spheroids that touch the image border are excluded from all metric calculations
            (returns None/NaN). If False, metrics are calculated regardless of border contact.
            Recommended to keep True to avoid edge artifacts.
        
        plot : bool, default=False
            Whether to display a plot of the results.
        
        skip_nan : bool, default=True
            Only affects plotting. If True, NaN points are not drawn (creates gaps in the curve).
            Data in the returned DataFrame remains unchanged (NaN values are still present).
        
        ax : Optional[plt.Axes], default=None
            Optional matplotlib Axes object to plot on. If provided and ``plot=True``,
            the plot will be drawn on this axes. If None and ``plot=True``, a new
            figure with default size (10, 6) is created.
        
        plot_kwargs : Optional[dict], default=None
            Optional dictionary of keyword arguments passed to ``ax.plot()`` for
            customizing line appearance. 
        
        savepath : Optional[str], default=None
            Optional path to save the figure. Only used if ``plot=True``.
            Works with both internally created figures and externally provided axes.
            Can specify any file format supported by matplotlib (PNG, PDF, SVG, etc.).
        
        Returns
        -------
        pd.DataFrame
            DataFrame with metric values:
            - Index: Timepoints in relative hours (from first timepoint)
            - Columns: 
              * If ``mean=False``: One column per spheroid series
              * If ``mean=True``: MultiIndex columns (collection_name, 'mean') and (collection_name, 'std')
            - Values: Metric values at each timepoint
            
            Special cases:
            - If ``timepoint`` is specified: Returns single-row DataFrame for that timepoint
            - If ``name='fluorescence_*'`` (without _mean/_cumulative): Returns MultiIndex columns
              with ('cumulative', 'mean') sub-columns
        
        Examples
        --------
        >>> # Calculate radius for all series
        >>> df = collection.metric('radius')
        
        >>> # Calculate mean ± std across series
        >>> df = collection.metric('radius', mean=True)
        
        >>> # Calculate at specific timepoint
        >>> df = collection.metric('radius', timepoint=24.0)
        
        >>> # Calculate with plot
        >>> df = collection.metric('radius', plot=True, mean=True)
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
                # Nur innerhalb des vorhandenen Wertebereichs interpolieren;
                # NaN vor dem ersten und nach dem letzten gültigen Punkt bleiben erhalten.
                result_df = pd.DataFrame({
                    (condition, 'mean'): result_df[(condition, 'mean')].interpolate(
                        method='linear',
                        axis=0,
                        limit_direction='both',
                        limit_area='inside',
                    ),
                    (condition, 'std'): result_df[(condition, 'std')].interpolate(
                        method='linear',
                        axis=0,
                        limit_direction='both',
                        limit_area='inside',
                    ),
                })
        
        # Plot if requested
        if plot:
            # Axes handling similar to SpheroidSeries.metric
            fig = None
            if ax is None:
                # Slightly more compact plot than before
                fig, ax = plt.subplots(figsize=(8, 4.8))
            else:
                fig = ax.figure
            
            # Transparent background; reduce top/right spines for a modern look
            if fig is not None:
                fig.patch.set_alpha(0.0)
            ax.set_facecolor('none')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Prepare plot_kwargs
            if plot_kwargs is None:
                plot_kwargs = {}
            
            x = np.array(result_df.index) / 24  # Convert to days
            
            # Get base color for this collection (if a palette is set). When no
            # palette is specified, fall back to the first palette in
            # CARTO_SEQUENTIAL so all lines still share a consistent family.
            base_palette = None
            base_color = None
            if self.color and self.color in CARTO_SEQUENTIAL:
                base_palette = CARTO_SEQUENTIAL[self.color]
            elif CARTO_SEQUENTIAL:
                # Use first available palette as default
                first_name = next(iter(CARTO_SEQUENTIAL.keys()))
                base_palette = CARTO_SEQUENTIAL[first_name]
            if base_palette:
                # Middle color of the palette
                base_color = base_palette[len(base_palette) // 2]
            # Fallback orange similar to the histogram viewer
            default_orange = '#D97706'
            
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
                
                line_color = base_color if base_color else default_orange
                fill_color = line_color
                
                ax.fill_between(
                    x_plot,
                    mean_plot - std_plot,
                    mean_plot + std_plot,
                    alpha=0.2,
                    color=fill_color,
                )
                
                # Orange/line style without markers; user can override via plot_kwargs
                default_kwargs = {
                    'color': line_color,
                    'linestyle': '-',
                    'linewidth': 2.0,
                }
                default_kwargs.update(plot_kwargs)
                ax.plot(x_plot, mean_plot, label=condition, **default_kwargs)
            else:
                # Plot individual series with palette colors, but without markers
                if base_palette is not None:
                    palette = base_palette
                    num_colors = len(base_palette)
                else:
                    palette = None
                    num_colors = 0
                
                default_kwargs = {
                    'linestyle': '-',
                    'linewidth': 2.0,
                    'alpha': 0.9,
                }
                default_kwargs.update(plot_kwargs)
                
                # Build a color index preference order that picks maximally
                # separated entries from the palette (extremes + middle).
                # The *set* of colors is determined by this order, but the
                # plotted sequence will follow the ascending palette index
                # (darkest → lightest) within that selected subset.
                if palette:
                    n = num_colors
                    pref_order: list[int] = []
                    if n > 0:
                        pref_order.append(0)
                    if n > 1:
                        pref_order.append(n - 1)
                    if n > 2:
                        pref_order.append(n // 2)
                    remaining = [i for i in range(n) if i not in pref_order]
                    center = (n - 1) / 2.0
                    remaining.sort(key=lambda i: abs(i - center), reverse=True)
                    pref_order.extend(remaining)
                    # Determine how many distinct colors we actually need
                    n_series = len(result_df.columns)
                    if n_series >= n:
                        selected = list(range(n))  # use full palette in order
                    else:
                        # Take first n_series preferred indices, then sort them
                        selected = sorted(set(pref_order[:n_series]))
                else:
                    selected = []
                
                for idx, col in enumerate(result_df.columns):
                    y = result_df[col]
                    # Use color from palette if available, cycling through
                    if palette:
                        if not selected:
                            series_color = palette[idx % num_colors]
                        else:
                            series_color = palette[selected[idx % len(selected)]]
                        kwargs_local = dict(default_kwargs)
                        kwargs_local['color'] = series_color
                    else:
                        kwargs_local = dict(default_kwargs)
                    
                    if skip_nan:
                        valid_mask = ~np.isnan(y)
                        ax.plot(x[valid_mask], y[valid_mask], label=col, **kwargs_local)
                    else:
                        ax.plot(x, y, label=col, **kwargs_local)
            
            # x-limits only from min to max of valid (non-NaN) data
            if not result_df.empty:
                any_valid = ~np.all(np.isnan(result_df.values), axis=1)
                if any_valid.any():
                    x_valid = x[any_valid]
                    ax.set_xlim(float(x_valid.min()), float(x_valid.max()))
            
            # Default formatting – loosely inspired by the example style
            ax.grid(True, linestyle='--', alpha=0.4, color='gray')
            ax.set_axisbelow(True)
            ax.set_xlabel('Time [d]', fontsize=14, labelpad=10)
            
            # y-label with units
            unit = ''
            pretty_name = None
            if self.result and hasattr(self.result, 'plot_name_dict') and name in getattr(self.result, 'plot_name_dict', {}):
                pretty_name = self.result.plot_name_dict[name]
            else:
                pretty_name = name.replace('_', ' ').title()
            # Simple heuristic for physical units
            if name.startswith('radius') or name.endswith('radius'):
                unit = ' [µm]'
            elif name.startswith('area') or name.endswith('area'):
                unit = ' [µm²]'
            elif name.startswith('fluorescence'):
                unit = ' (a.u.)'
            ax.set_ylabel(f'{pretty_name}{unit}', fontsize=14, labelpad=10)
            
            if not mean or len(result_df.columns) < 10:  # Only show legend if not too many series
                ax.legend(fontsize=11, framealpha=0.5, edgecolor='gray', loc='best')
            
            ax.set_title(
                f'{pretty_name} over time' + (' (mean ± std)' if mean else ''),
                fontsize=16,
                fontweight='bold',
                pad=10,
            )
            
            # Save if requested
            if savepath is not None:
                if fig is None:
                    fig = ax.figure
                fig.savefig(savepath, bbox_inches='tight')
            
            # Show only if we created the figure ourselves
            if fig is not None:
                plt.tight_layout()
                plt.show()
            
            return result_df, ax
        
        return result_df

    def radial_profile(self, timepoint: int | str | datetime, 
                      channels: str | list[str] = ['green'], 
                      absolute_radius: bool = False,
                      normalize: bool = True,
                      smoothing: float = 2,
                      mean: bool = False,
                      plot: bool = True,
                      savepath: str | None = None,
                      dual_axis: bool = False) -> dict:
        """
        Calculate and plot radial profiles for all spheroid series in this collection.
        
        Parameters
        ----------
        timepoint : int | str | datetime
            Timepoint to analyze. Can be:
            - int: Relative time in hours
            - str: Datetime string in format 'YYYY-MM-DD HH:MM:SS'
            - datetime: Datetime object
        channels : str | list[str], default=['green']
            Fluorescence channel(s) to analyze. Can be a single channel string ('green', 'red', 'blue')
            or a list of channels.
        absolute_radius : bool, default=False
            If True, uses absolute distances in μm. If False, uses normalized distances (0-1).
        normalize : bool, default=True
            Whether to normalize intensity profiles to [0,1]
        smoothing : float, default=2
            Gaussian smoothing parameter (sigma) for radial profile
        mean : bool, default=False
            If True, averages profiles across all series. If False, shows individual traces.
        plot : bool, default=True
            Whether to display a plot
        savepath : str | None, default=None
            Optional path to save the plot
        
        Returns
        -------
        dict
            Dictionary containing:
            - timepoint: Processed timepoint identifier
            - profiles: Dictionary mapping series_name -> channel -> profile data (if mean=False)
                       or channel -> averaged profile data (if mean=True)
            - distances: Distance array (relative or absolute)
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
                # Get first series to find base timepoint
                active_series = self.get_spheroids()
                if not active_series:
                    raise ValueError("No active spheroid series in collection")
                sorted_times = sorted(active_series[0].spheroid_image_dict.keys())
                if not sorted_times:
                    raise ValueError("No timepoints available in series")
                t0 = sorted_times[0] if isinstance(sorted_times[0], datetime) else datetime.strptime(sorted_times[0], '%Y-%m-%d %H:%M:%S')
                return t0 + pd.Timedelta(hours=float(val))
            else:
                raise ValueError(f"Cannot convert {type(val)} to datetime")
        
        # Get active spheroid series
        active_series = self.get_spheroids()
        if not active_series:
            raise ValueError("No active spheroid series in collection")
        
        # Process timepoint and find matching images
        tp_dt = _ensure_dt(timepoint)
        profiles_data = {}
        distances_data = None
        
        for series in active_series:
            # Find closest timepoint in series
            sorted_times = sorted(series.spheroid_image_dict.keys())
            if not sorted_times:
                continue
            
            def _ensure_dt_local(val):
                return val if isinstance(val, datetime) else datetime.strptime(val, '%Y-%m-%d %H:%M:%S')
            
            closest_tp = min(sorted_times, 
                           key=lambda x: abs((_ensure_dt_local(x) - tp_dt).total_seconds()))
            
            # Check if close enough (within 1 hour)
            time_diff = abs((_ensure_dt_local(closest_tp) - tp_dt).total_seconds() / 3600)
            if time_diff > 1.0:
                logger.warning(f"Timepoint {timepoint} is more than 1 hour away from closest match {closest_tp} in series {series.name}. Skipping.")
                continue
            
            img = series.spheroid_image_dict.get(closest_tp)
            if img is None or img.contour is None:
                continue
            
            try:
                result = img.radial_profile(
                    channels=channels,
                    plot=False,
                    normalize=normalize,
                    absolute_radius=absolute_radius,
                    smoothing=smoothing,
                )
                
                # Store distances (should be same for all series)
                if distances_data is None:
                    if absolute_radius and 'absolute_distances' in result:
                        distances_data = result['absolute_distances']
                    else:
                        distances_data = result['relative_distances']
                
                if mean:
                    # Store for averaging later
                    if 'mean_profiles' not in profiles_data:
                        profiles_data['mean_profiles'] = {}
                    for ch in channels:
                        if ch not in result['intensity_profiles']:
                            continue
                        if ch not in profiles_data['mean_profiles']:
                            profiles_data['mean_profiles'][ch] = {
                                'mean_list': [],
                                'std_list': []
                            }
                        prof = result['intensity_profiles'][ch]
                        profiles_data['mean_profiles'][ch]['mean_list'].append(prof['mean'])
                        profiles_data['mean_profiles'][ch]['std_list'].append(prof['std'])
                else:
                    # Store individual profiles
                    profiles_data[series.name] = result['intensity_profiles']
                    
            except Exception as e:
                logger.warning(f"Failed to calculate radial profile for series {series.name}: {e}")
                continue
        
        if not profiles_data:
            raise ValueError("No valid radial profiles could be calculated")
        
        # Average profiles if mean=True
        if mean and 'mean_profiles' in profiles_data:
            averaged_profiles = {}
            for ch, data in profiles_data['mean_profiles'].items():
                mean_list = data['mean_list']
                std_list = data['std_list']
                
                if not mean_list:
                    continue
                
                # Stack arrays and compute mean/std across series
                mean_array = np.stack(mean_list, axis=0)  # (n_series, n_bins)
                std_array = np.stack(std_list, axis=0)
                
                # Average mean and std
                avg_mean = np.mean(mean_array, axis=0)
                # For std: combine within-series std with between-series variation
                # std_total = sqrt(mean(std^2) + var(mean))
                avg_std = np.sqrt(np.mean(std_array**2, axis=0) + np.var(mean_array, axis=0))
                
                averaged_profiles[ch] = {
                    'mean': avg_mean,
                    'std': avg_std
                }
            profiles_data = averaged_profiles
        
        # Plotting
        if plot:
            # Use dual-axis only if explicitly requested and exactly 2 channels
            use_dual_axis = dual_axis and len(channels) == 2
            
            if use_dual_axis:
                fig, ax = plt.subplots(figsize=(8, 6))
                ax2 = ax.twinx()
            else:
                fig, ax = plt.subplots(figsize=(8, 6))
                ax2 = None
            
            # Color palette mapping for channels (using channel-similar palettes)
            # Use Green for green, Red/Orange for red, Teal for blue
            channel_palettes = {
                'green': CARTO_SEQUENTIAL.get('Green', ['#00441b', '#006d2c', '#238b45', '#41ab5d', '#74c476', '#a1d99b', '#c7e9c0']),
                'red': CARTO_SEQUENTIAL.get('Red', ['#7f0000', '#b31b1b', '#d94801', '#f16913', '#fd8d3c', '#fdae6b', '#fee6ce']),
                'blue': CARTO_SEQUENTIAL.get('Teal', ['#004c4c', '#006d6d', '#238b8b', '#41a9a9', '#74c8c8', '#a1e3e3', '#c8f5f5']),
            }
            
            if mean:
                # Plot averaged profiles
                for ch_idx, channel in enumerate(channels):
                    if channel not in profiles_data:
                        continue
                    
                    palette = channel_palettes.get(channel, ['#333333', '#555555', '#777777'])
                    color = palette[len(palette) // 2]  # Use middle color for mean
                    
                    profile = profiles_data[channel]
                    mean = profile['mean']
                    std = profile.get('std', np.zeros_like(mean))
                    
                    label = f"{channel} (mean ± std)"
                    
                    # Select axis for dual-axis plotting
                    if use_dual_axis:
                        current_ax = ax if ch_idx == 0 else ax2
                    else:
                        current_ax = ax
                    
                    current_ax.plot(distances_data, mean, color=color, label=label, linewidth=2)
                    current_ax.fill_between(distances_data, mean - std, mean + std, 
                                           color=color, alpha=0.2)
            else:
                # Plot individual series
                n_series = len([k for k in profiles_data.keys() if k != 'mean_profiles'])
                for ch_idx, channel in enumerate(channels):
                    if channel not in channel_palettes:
                        palette = ['#333333', '#555555', '#777777', '#999999', '#bbbbbb', '#dddddd']
                    else:
                        palette = channel_palettes[channel]
                    
                    n_colors = len(palette)
                    series_idx = 0
                    
                    for series_name, series_profiles in profiles_data.items():
                        if series_name == 'mean_profiles':
                            continue
                        if channel not in series_profiles:
                            continue
                        
                        profile = series_profiles[channel]
                        mean = profile['mean']
                        std = profile.get('std', np.zeros_like(mean))
                        
                        # Use colors from palette, cycling through
                        color = palette[series_idx % n_colors]
                        label = f"{channel} - {series_name}"
                        
                        # Select axis for dual-axis plotting
                        if use_dual_axis:
                            current_ax = ax if ch_idx == 0 else ax2
                        else:
                            current_ax = ax
                        
                        current_ax.plot(distances_data, mean, color=color, label=label, linewidth=1.5, alpha=0.7)
                        current_ax.fill_between(distances_data, mean - std, mean + std, 
                                               color=color, alpha=0.15)
                        series_idx += 1
            
            # Set labels
            if absolute_radius:
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
            
            title = f'Radial Profiles - {self.name}'
            if mean:
                title += ' (mean ± std)'
            ax.set_title(title, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Combine legends if using dual axis
            if use_dual_axis:
                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=9)
            else:
                ax.legend(loc='best', fontsize=9)
            
            # Set xlims: start at min value (not 0) to max value
            if distances_data is not None and len(distances_data) > 0:
                ax.set_xlim(distances_data.min(), distances_data.max())
            
            plt.tight_layout()
            if savepath is not None:
                plt.savefig(savepath, transparent=True, dpi=300, bbox_inches='tight')
            plt.show()
        
        return {
            'timepoint': tp_dt,
            'profiles': profiles_data,
            'distances': distances_data
        }


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
        
        # Image display
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
                image_title.value = ""
                return
                
            timepoint = timepoints[idx]
            image_title.value = f"<div style='text-align: center;'><span style='font-size: 12px; font-weight: 600; color: #333;'>{timepoint} | {channel} | {current_series.name}</span></div>"
            
            sph_img = current_series.spheroid_image_dict[timepoint]
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
                image_output.clear_output(wait=True)
                with image_output:
                    print(f"Error loading image: {e}")
                return
                
            image_output.clear_output(wait=True)
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

        # Radius plot
        plot_output = widgets.Output()
        def update_plot(*args):
            metric = metric_dropdown.value
            if metric == "radius":
                metric_label = "Radius"
            elif metric == "area":
                metric_label = "Area"
            else:
                metric_label = "Radial profile"
            current_series = get_current_series()
            over_time = " over time" if metric in ("radius", "area") else ""
            plot_title.value = f"<div style='text-align: center;'><span style='font-size: 14px; font-weight: 600; color: #333;'>{metric_label}{over_time} – {current_series.name}</span></div>"
            
            plot_output.clear_output(wait=True)
            with plot_output:
                current_series = get_current_series()
                timepoints = sorted(current_series.spheroid_image_dict.keys())
                
                try:
                    if metric == "profile":
                        # Normalized radial profile for current timepoint
                        idx = time_slider.value
                        if idx >= len(timepoints):
                            return
                        timepoint = timepoints[idx]
                        image = current_series.spheroid_image_dict.get(timepoint)
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
                                    absolute_radius=True,
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
                        times = [(_ensure_dt(tp) - base_t0).total_seconds()/24/3600 for tp in timepoints]
                        
                        if metric == "area":
                            values = [current_series.spheroid_image_dict[tp].area for tp in timepoints]
                            ylabel = "Area [µm²]"
                        else:
                            values = [current_series.spheroid_image_dict[tp].radius for tp in timepoints]
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
                        if hasattr(current_series, 'time_periods') and current_series.time_periods:
                            # Get actual y-axis limits after plotting
                            y_min, y_max = ax.get_ylim()
                            y_range = y_max - y_min
                            
                            for period_name, period in current_series.time_periods.items():
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
                update_plot()
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
                update_plot()
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

        # Layout callbacks
        def on_any_change(*args):
            update_image()
            update_plot()
            
        time_slider.observe(lambda change: on_any_change(), names='value')
        channel_dropdown.observe(lambda change: update_image(), names='value')
        show_contour.observe(lambda change: update_image(), names='value')
        series_dropdown.observe(lambda change: on_any_change(), names='value')
        metric_dropdown.observe(lambda change: update_plot(), names='value')

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

        # Display first, then render initial content.
        display(outer_box)

        def _initial_render():
            update_time_label(0)
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

