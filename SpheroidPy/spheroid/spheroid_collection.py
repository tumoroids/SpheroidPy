from __future__ import annotations
from typing import TYPE_CHECKING
from dataclasses import dataclass
import h5py
import numpy as np
import pandas as pd
from datetime import datetime
import copy
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from SpheroidPy.experiment.result import Result
    from SpheroidPy.spheroid.spheroid_series import SpheroidSeries

class SpheroidCollection:
    """Collection of spheroid series representing biological replicates.
    
    Groups spheroids from the same experimental condition and provides methods for analysis.
    
    Attributes:
        name: Name of the condition this collection represents
        spheroid_series: List of SpheroidSeries in this collection
        ignored_spheroids: Set of spheroid series names to exclude
        _result: Reference to the Result this collection belongs to
        hdf5_path: Path to HDF5 file for storage
    """

    def __init__(self, name: str, spheroid_list: list[SpheroidSeries] | None = None, 
                 result: Result | None = None, hdf5_path: str | None = None):
        self.name = name
        self.hdf5_path = hdf5_path
        self._result = result
        
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
            
        self.spheroid_series.extend(spheroid_list)

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

    def calculate_metric(self, name: str = 'radius', mean: bool = False,
                      interpolate: bool = True, ignore_border: bool = True,
                      skip_nan: bool = True, plot: bool = False) -> pd.DataFrame:
        """Calculate metrics for all spheroids in this collection.
        
        Args:
            name: Metric to calculate ('radius', 'area', 'fluorescence_*')
            mean: Whether to average across replicate wells
            interpolate: Whether to interpolate missing timepoints
            ignore_border: Whether to exclude spheroids touching image border
            skip_nan: Whether to exclude NaN values from calculations
            plot: Whether to display a plot of the metrics over time
            
        Returns:
            DataFrame containing metric values, with mean/std if requested
        """
        # Get all active spheroids
        spheroids = self.get_spheroids()
        if not spheroids:
            raise ValueError(f"No active spheroids in collection {self.name}")
            
        # Calculate metrics for each spheroid
        all_dfs = []
        for spheroid in spheroids:
            df = spheroid.calculate_metric(
                name=name,
                interpolate=interpolate,
                ignore_border=ignore_border,
                plot=False  # Don't plot individual series
            )
            all_dfs.append(df)
            
        # Combine all individual DataFrames
        result_df = pd.concat(all_dfs, axis=1)
        
        if mean:
            # Calculate mean and std across replicates
            mean_df = result_df.mean(axis=1, skipna=skip_nan)
            std_df = result_df.std(axis=1, skipna=skip_nan)
            
            # Create DataFrame with mean and std columns
            result_df = pd.DataFrame({
                (self.name, 'mean'): mean_df,
                (self.name, 'std'): std_df
            })
            
        # Plot if requested
        if plot:
            plt.figure(figsize=(8, 5))
            x = np.array(result_df.index) / 24  # Convert to days
            
            if mean:
                mean_array = result_df[(self.name, 'mean')]
                std_array = result_df[(self.name, 'std')]
                
                # Skip NaN values in plot
                if skip_nan:
                    valid_mask = ~np.isnan(mean_array)
                    x = x[valid_mask]
                    mean_array = mean_array[valid_mask]
                    std_array = std_array[valid_mask]
                
                plt.fill_between(x, 
                               mean_array - std_array,
                               mean_array + std_array,
                               alpha=0.3, color='gray')
                plt.plot(x, mean_array, label=self.name, color='blue')
                
            else:
                for col in result_df.columns:
                    y = result_df[col]
                    if skip_nan:
                        valid_mask = ~np.isnan(y)
                        plt.plot(x[valid_mask], y[valid_mask], 
                               label=col, alpha=0.7, marker='o', markersize=3)
                    else:
                        plt.plot(x, y, label=col, alpha=0.7, marker='o', markersize=3)
            
            plt.xlabel('Time [d]')
            
            # Use plot_name_dict from Result if available
            if self.result and name in self.result.plot_name_dict:
                plt.ylabel(rf'{self.result.plot_name_dict[name]}')
            else:
                plt.ylabel(name)
                
            plt.title(f'{name} over time' + (' (mean ± std)' if mean else ''))
            plt.grid(True, alpha=0.3)
            if not mean or len(result_df.columns) < 10:  # Only show legend if not too many series
                plt.legend()
            plt.tight_layout()
            plt.show()
            
        return result_df

