from __future__ import annotations
from typing import TYPE_CHECKING
from dataclasses import dataclass
import h5py
import numpy as np
import pandas as pd
from datetime import datetime

if TYPE_CHECKING:
    from SpheroidPy.experiment.result import Result
    from SpheroidPy.spheroid.spheroid_series import SpheroidSeries

class SpheroidCollection:
    """Collection of spheroid series representing biological replicates.
    
    Groups spheroids from the same experimental condition across different
    Results/replicates and provides methods for analysis.
    
    Attributes:
        name: Name of the condition this collection represents
        replicate_dict: Maps dates to lists of spheroid series
        replicate_metadata: Maps spheroid series names to replicate info
        ignored_spheroids: Set of spheroid series names to exclude
    """

    def __init__(self, name: str, spheroid_list: list[SpheroidSeries] | None = None, 
                 result: Result | None = None, hdf5_path: str | None = None):
        self.name = name
        self.hdf5_path = hdf5_path
        
        self.replicate_dict = {}  # {date: [SpheroidSeries]}
        self.replicate_metadata = {}  # {spheroid_name: {"result_name": str, "replicate_number": int, "date_added": datetime}}
        self.ignored_spheroids = set()

        if spheroid_list is not None:
            self.add_replicate_group(spheroid_list, result)

    def add_replicate_group(self, spheroid_list: list[SpheroidSeries], result: Result):
        """Add a group of spheroid series from a Result as replicates.
        
        Args:
            spheroid_list: List of SpheroidSeries to add
            result: Result these spheroids came from
            
        Raises:
            ValueError: If spheroid_list is empty
        """
        if not spheroid_list:
            print(f"Warning: Empty spheroid list for condition '{self.name}' from result '{result.name}'")
            return
        
        # Get date from first spheroid's first timepoint
        init_date = list(spheroid_list[0].spheroid_image_dict.keys())[0]
        
        # Add to replicate dict
        if init_date not in self.replicate_dict:
            self.replicate_dict[init_date] = []
        self.replicate_dict[init_date].extend(spheroid_list)

        # Track replicate metadata
        # Count existing replicates for this result
        replicate_num = len([info for info in self.replicate_metadata.values()
                           if info["result_name"] == result.name])
        
        for spheroid in spheroid_list:
            self.replicate_metadata[spheroid.name] = {
                "result_name": result.name,
                "replicate_number": replicate_num + 1,
                "date_added": datetime.now()
            }

    def get_spheroids(self, result: Result | None = None, 
                     time_period: str | None = None) -> list[SpheroidSeries]:
        """Get list of spheroids, optionally filtered by time period.
        
        Args:
            result: Optional Result to filter by
            time_period: Optional time period name to filter by
            
        Returns:
            List of SpheroidSeries objects
        """
        spheroids = []
        
        for spheroid_list in self.replicate_dict.values():
            for spheroid in spheroid_list:
                if spheroid.name in self.ignored_spheroids:
                    continue
                    
                if result and self.replicate_metadata[spheroid.name]["result_name"] != result.name:
                    continue
                
                if time_period:
                    # Create filtered copy with only timepoints in period
                    filtered_spheroid = spheroid.copy()
                    filtered_spheroid.spheroid_image_dict = spheroid.get_images_in_period(time_period)
                    
                    if filtered_spheroid.spheroid_image_dict:
                        spheroids.append(filtered_spheroid)
                else:
                    spheroids.append(spheroid)
                    
        return spheroids

    def get_replicate_stats(self) -> pd.DataFrame:
        """Get statistics about replicates in collection.
        
        Returns:
            DataFrame with columns:
            - result_name: Name of Result
            - replicate_number: Number within that Result  
            - num_spheroids: Number of spheroid series
            - num_active: Number not ignored
        """
        stats = []
        for result_name in {info["result_name"] for info in self.replicate_metadata.values()}:
            result_spheroids = [name for name, info in self.replicate_metadata.items()
                              if info["result_name"] == result_name]
            
            stats.append({
                'result_name': result_name,
                'num_spheroids': len(result_spheroids),
                'num_active': len([s for s in result_spheroids 
                                 if s not in self.ignored_spheroids])
            })
            
        return pd.DataFrame(stats)

    def ignore_spheroid(self, spheroid_series_name: str):
        """Mark a spheroid series as ignored for analysis."""
        self.ignored_spheroids.add(spheroid_series_name)

    def unignore_spheroid(self, spheroid_series_name: str):
        """Remove a spheroid series from the ignored list."""
        self.ignored_spheroids.discard(spheroid_series_name)

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

        with h5py.File(self.hdf5_path, 'a') as hdf_file:
            for spheroid_series in self.get_spheroids():
                print(f"Processing {spheroid_series.name} from "
                      f"{self.replicate_metadata[spheroid_series.name]['result_name']}")
                
                for spheroid_image in spheroid_series.spheroid_image_dict.values():
                    spheroid_image.segmentation(
                        methods=methods,
                        reconstruct_border=reconstruct_border,
                        **kwargs
                    )
                    
                    # Update HDF5
                    timepoint_group = hdf_file[spheroid_image.hdf5_key]
                    if 'contour' in timepoint_group:
                        del timepoint_group['contour']
                        del timepoint_group.attrs['touches_border']

                    if spheroid_image.contour is not None:
                        timepoint_group.create_dataset('contour', data=spheroid_image.contour)
                        timepoint_group.attrs['touches_border'] = spheroid_image.contour_touches_border

    def __repr__(self) -> str:
        stats = self.get_replicate_stats()
        return (f"SpheroidCollection: {self.name}\n"
                f"Total spheroids: {stats['num_spheroids'].sum()}\n"
                f"Active spheroids: {stats['num_active'].sum()}\n"
                f"Results: {', '.join(stats['result_name'])}")

    def merge_collection(self, other_collection: SpheroidCollection):
        """Merge another SpheroidCollection into this one.
        
        Combines spheroids while maintaining condition grouping and replicate tracking.
        
        Args:
            other_collection: SpheroidCollection to merge into this one
        
        Raises:
            ValueError: If collections have different conditions
        """
        if self.name != other_collection.name:
            raise ValueError(f"Cannot merge collections with different conditions: "
                            f"{self.name} vs {other_collection.name}")
        
        # Merge replicate dictionaries
        for date, spheroid_list in other_collection.replicate_dict.items():
            if date not in self.replicate_dict:
                self.replicate_dict[date] = []
            self.replicate_dict[date].extend(spheroid_list)
        
        # Merge metadata while preserving replicate info
        self.replicate_metadata.update(other_collection.replicate_metadata)
        
        # Merge ignored spheroids
        self.ignored_spheroids.update(other_collection.ignored_spheroids)

    def add_time_period(self, name: str, 
                       start_time: str | datetime | None = None,
                       end_time: str | datetime | None = None,
                       replicate: str | None = None):
        """Add a named time period to spheroids in collection.
        
        Args:
            name: Name of the time period
            start_time: Start datetime, or None to use first timepoint
            end_time: End datetime, or None to use last timepoint
            replicate: Optional result name to apply to specific replicate only.
                      If None, applies to all spheroids that fall within the time range.
        
        Example:
            # Add period to all spheroids within time range
            collection.add_time_period(
                "phase1", 
                "2024-01-01 00:00:00", 
                "2024-01-02 00:00:00"
            )
            
            # Add period only to specific replicate
            collection.add_time_period(
                "early",
                end_time="2024-01-01 12:00:00",
                replicate="Day1_Result"
            )
            
            # Add period using first/last timepoints
            collection.add_time_period("all", replicate="Day2_Result")
        """
        # Convert string times to datetime if needed
        if isinstance(start_time, str):
            start_time = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
        if isinstance(end_time, str):
            end_time = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')
        
        # Get all spheroids to process
        for spheroid_list in self.replicate_dict.values():
            for spheroid in spheroid_list:
                # Skip if not in target replicate
                if replicate and self.replicate_metadata[spheroid.name]["result_name"] != replicate:
                    continue
                    
                # If no specific replicate, check if spheroid has timepoints in range
                if not replicate:
                    timepoints = [datetime.strptime(t, '%Y-%m-%d %H:%M:%S') 
                                for t in spheroid.spheroid_image_dict.keys()]
                    
                    # Skip if no timepoints in range
                    if start_time and end_time and not any(
                        start_time <= t <= end_time for t in timepoints):
                        continue
                        
                # Add time period to spheroid
                spheroid.add_time_period(name, start_time, end_time)

    def get_time_periods(self, replicate: str | None = None) -> dict:
        """Get time periods defined in collection.
        
        Args:
            replicate: Optional result name to get periods for specific replicate
            
        Returns:
            Dictionary mapping period names to sets of (start, end) tuples
        """
        periods = {}
        
        for spheroid_list in self.replicate_dict.values():
            for spheroid in spheroid_list:
                if replicate and self.replicate_metadata[spheroid.name]["result_name"] != replicate:
                    continue
                    
                for name, (start, end) in spheroid.time_periods.items():
                    if name not in periods:
                        periods[name] = set()
                    periods[name].add((start, end))
                    
        return periods

    def time_period_depric(self, name: str, analysis_type: str,
                   start_time: str | datetime | None = None,
                   end_time: str | datetime | None = None,
                   description: str | None = None):
        """Smart method to set or update time period for all spheroid series.
        
        Args:
            name: Unique name for this time period
            analysis_type: Type of analysis this time period is for
            start_time: Start datetime, or None to use first timepoint
            end_time: End datetime, or None to use last timepoint
            description: Optional description
        """
        for spheroid_list in self.replicate_dict.values():
            for spheroid in spheroid_list:
                if spheroid.name not in self.ignored_spheroids:
                    try:
                        spheroid.time_period(
                            name=name,
                            analysis_type=analysis_type,
                            start_time=start_time,
                            end_time=end_time,
                            description=description
                        )
                    except Exception as e:
                        print(f"Failed to set/update time period for {spheroid.name}: {str(e)}")

    def save_time_periods_to_hdf5(self):
        """Save time periods to HDF5 file for all spheroid series."""
        for spheroid_list in self.replicate_dict.values():
            for spheroid in spheroid_list:
                if spheroid.name not in self.ignored_spheroids:
                    spheroid.save_time_periods_to_hdf5()

    def load_time_periods_from_hdf5(self):
        """Load time periods from HDF5 file for all spheroid series."""
        for spheroid_list in self.replicate_dict.values():
            for spheroid in spheroid_list:
                if spheroid.name not in self.ignored_spheroids:
                    spheroid.load_time_periods_from_hdf5()

    def set_time_period(self, name: str,
                       start_time: str | datetime | None = None,
                       end_time: str | datetime | None = None,
                       description: str | None = None):
        """Set time period for all spheroid series in collection."""
        for spheroid_list in self.replicate_dict.values():
            for spheroid in spheroid_list:
                if spheroid.name not in self.ignored_spheroids:
                    try:
                        spheroid.add_time_period(
                            name=name,
                            start_time=start_time,
                            end_time=end_time,
                            description=description
                        )
                    except Exception as e:
                        print(f"Failed to set time period for {spheroid.name}: {str(e)}")
        
        # Save updated time periods to HDF5
        self.save_time_periods_to_hdf5()
                        
    def update_time_period(self, name: str,
                          start_time: str | datetime | None = None,
                          end_time: str | datetime | None = None,
                          description: str | None = None):
        """Update time period for all spheroid series."""
        for spheroid_list in self.replicate_dict.values():
            for spheroid in spheroid_list:
                if spheroid.name not in self.ignored_spheroids:
                    try:
                        spheroid.update_time_period(
                            name=name,
                            start_time=start_time,
                            end_time=end_time,
                            description=description
                        )
                    except Exception as e:
                        print(f"Failed to update time period for {spheroid.name}: {str(e)}")
        
        # Save updated time periods to HDF5
        self.save_time_periods_to_hdf5()
                        
    def remove_time_period(self, name: str):
        """Remove time period from all spheroid series."""
        for spheroid_list in self.replicate_dict.values():
            for spheroid in spheroid_list:
                if spheroid.name not in self.ignored_spheroids:
                    spheroid.remove_time_period(name)
        
        # Save updated time periods to HDF5
        self.save_time_periods_to_hdf5()

    @property
    def result(self):
        """Get the first Result associated with this collection."""
        result = None
        for spheroid_series in self.get_spheroids():
            if result is None:
                result = spheroid_series._result
            elif result != spheroid_series._result:
                result = None
                return result
                #raise ValueError("SpheroidCollection contains spheroids from multiple Results")

        return result

