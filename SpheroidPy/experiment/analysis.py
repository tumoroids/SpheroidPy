from __future__ import annotations
from typing import TYPE_CHECKING
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm
import h5py

from SpheroidPy.experiment.base import Base
from SpheroidPy.spheroid.spheroid_collection import SpheroidCollection
from SpheroidPy.experiment.result import Result

if TYPE_CHECKING:
    from SpheroidPy.experiment.experiment import Experiment

class Analysis(Base):
    """Class for orchestrating analyses across multiple experimental results.
    
    Handles biological replicates and coordinates analysis methods implemented
    in SpheroidImage across multiple Results.
    
    Attributes:
        name: Name of the analysis
        replicates_dict: Maps SpheroidCollection objects to Result objects
    """

    def __init__(self, name: str, experiment: Experiment):
        """Initialize Analysis instance.
        
        Args:
            name: Name of the analysis
            experiment: Parent Experiment instance
        """
        # Initialize base class first
        super().__init__(name, experiment, 'Analysis')
        
        # Initialize instance attributes
        self.replicates_dict: dict[Result, dict[str, SpheroidCollection]] = {}  # List of added Result objects

    def add_replicate(self, spheroids: SpheroidCollection | Result, name: str | None = None):
        """Add a SpheroidCollections (or entire Result) as a biological replicate on which further analyses are performed.
        
        Args:
            spheroids: Result to add as replicate
            name: Optional name for the replicate group (defaults to condition name of SpheroidCollection object)
            
        Saves replicate information to HDF5 file structure: NOT DONE!!!!!!!!!!!!!!!!!!!
        /Analysis/{index}-{name}/replicates/
            ├── results/
            │   └── {result_name}
            └── conditions/
                └── {condition}/
                    ├── name
                    ├── spheroid_list
                    └── metadata
        """

        print(type(spheroids), isinstance(spheroids, SpheroidCollection))
        print(id(spheroids.__class__))  # ID der Klasse des Objekts
        print(id(SpheroidCollection))  # ID der importierten Klasse

        if isinstance(spheroids, Result):
            for condition_name, spheroid_collection in spheroids.replicates.items():
                self.add_replicate(spheroid_collection, condition_name)
        #elif not isinstance(spheroids, SpheroidCollection):
        #    raise Exception("Replicate must be a SpheroidCollection or Result object")

        if name is None:
            name = spheroids.name

        # add spheroids to dict
        print(spheroids.result.name)
        if spheroids.result.name not in self.replicates_dict:
            self.replicates_dict[spheroids.result.name] = {}
        self.replicates_dict[spheroids.result.name][name] = spheroids

        # Save to HDF5
        ''' Commented out to avoid HDF5 file structure
        with h5py.File(self.hdf5_path, 'a') as hdf_file:
            analysis_group = hdf_file[self.hdf5_key]
            
            # Create groups if needed
            if 'replicates' not in analysis_group:
                replicates_group = analysis_group.create_group('replicates')
                results_group = replicates_group.create_group('results')
                conditions_group = replicates_group.create_group('conditions')
            else:
                replicates_group = analysis_group['replicates']
                results_group = replicates_group['results']
                conditions_group = replicates_group['conditions']

            # Save result reference
            result_ref = results_group.create_group(result.name)
            result_ref.attrs['date_added'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            result_ref.attrs['hdf5_key'] = result.hdf5_key
            
            # Save conditions and their spheroid collections
            for condition, collections in self.replicates_dict.items():
                if condition == 'results':
                    continue
                    
                # Create or get condition group
                if condition not in conditions_group:
                    condition_group = conditions_group.create_group(condition)
                else:
                    condition_group = conditions_group[condition]
                
                # Save collection metadata
                for collection in collections:
                    if isinstance(collection, SpheroidCollection):
                        collection_group = condition_group.create_group(
                            f"collection_{len(condition_group)}")
                        
                        # Save basic info
                        collection_group.attrs['name'] = collection.name
                        collection_group.attrs['result'] = result.name
                        collection_group.attrs['date_added'] = datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S")
                        
                        # Save ignored spheroids
                        if collection.ignored_spheroids:
                            ignored_ds = collection_group.create_dataset(
                                'ignored_spheroids',
                                data=[s.encode('utf-8') for s in collection.ignored_spheroids]
                            )
                        
                        # Save spheroid references
                        spheroids_group = collection_group.create_group('spheroids')
                        for spheroid in collection.get_active_spheroids(result):
                            spheroid_ref = spheroids_group.create_group(spheroid.name)
                            for timepoint, image in spheroid.spheroid_image_dict.items():
                                spheroid_ref.attrs[timepoint] = image.hdf5_key
            '''

    def get_spheroids(self, time_period: str | None = None, combine_replicates: bool = False) -> list[SpheroidCollection]:
        """Get all spheroids from replicates.

        Args:
            time_period: Optional time period to filter by
            combine_replicates: Combine replicates of the same condition from separate results

        Returns:
            List of SpheroidCollection objects
        """
        spheroids = []

        for condition, collections in self.replicates_dict.items():
            for collection in collections:
                spheroids.extend(collection.get_spheroids(time_period))

        return spheroids

    def analyze_functional_radii(self, live_channel: str | None = None,
                               death_channel: str | None = None) -> pd.DataFrame:
        """Analyze functional radii across all replicates.
        
        Computes outer, inhibited and necrotic radii for all spheroids
        in each replicate group.
        
        Args:
            live_channel: Override default live cell fluorescence channel
            death_channel: Override default dead cell fluorescence channel
            
        Returns:
            DataFrame containing functional radii measurements for all conditions
        """
        live_ch = live_channel or self.config["live_channel"]
        death_ch = death_channel or self.config["death_channel"]

        results_data = []

        # Iterate through conditions
        for condition, replicate_collections in tqdm(self.replicates_dict.items(), 
                                                   desc="Analyzing conditions"):
            if condition == 'results':
                continue
                
            # Process each replicate collection
            for collection in replicate_collections:
                # Only process active spheroids
                for spheroid_series in collection.get_active_spheroids():
                    for timepoint, spheroid_image in spheroid_series.spheroid_image_dict.items():
                        try:
                            # Calculate functional radii
                            radii = spheroid_image.functional_radius(
                                live_color=live_ch,
                                death_color=death_ch
                            )
                            
                            # Store results
                            results_data.append({
                                'condition': condition,
                                'timepoint': datetime.strptime(timepoint, '%Y-%m-%d %H:%M:%S'),
                                'outer_radius': radii['outer'],
                                'inhibited_radius': radii['inhibited'],
                                'necrotic_radius': radii['necrotic']
                            })
                        except Exception as e:
                            print(f"Error processing {condition} at {timepoint}: {e}")

        # Convert to DataFrame
        df = pd.DataFrame(results_data)
        
        # Store results
        self.analysis_results['functional_radii'] = df
        
        return df

    def get_mean_profiles(self, metric: str = 'functional_radii') -> pd.DataFrame:
        """Calculate mean profiles across replicates.
        
        Args:
            metric: Name of analysis metric to average
            
        Returns:
            DataFrame with mean and std values for each condition/timepoint
        """
        if metric not in self.analysis_results:
            raise ValueError(f"No results found for metric '{metric}'")
            
        df = self.analysis_results[metric]
        
        # Group by condition and timepoint
        grouped = df.groupby(['condition', 'timepoint'])
        
        # Calculate means and standard deviations
        means = grouped.mean()
        stds = grouped.std()
        
        # Combine into single DataFrame
        results = pd.DataFrame()
        for col in df.select_dtypes(include=[np.number]).columns:
            results[f'{col}_mean'] = means[col]
            results[f'{col}_std'] = stds[col]
            
        return results

    def analyze_by_replicate(self, condition: str, analysis_func: callable,
                           time_period: str | None = None, **kwargs) -> pd.DataFrame:
        """Analyze spheroids grouped by replicate.
        
        Args:
            condition: Condition to analyze
            analysis_func: Function that takes a spheroid and returns metrics
            time_period: Optional name of time period to analyze
            **kwargs: Additional arguments for analysis_func
        """
        if condition not in self.replicates_dict:
            raise ValueError(f"Condition {condition} not found")
        
        collection = self.replicates_dict[condition]
        results_data = []
        
        # Get unique Results containing this condition
        result_names = {info["result_name"] for info in collection.replicate_metadata.values()}
        
        for result_name in result_names:
            # Get spheroids from this Result
            result = next(r for r in self.results if r.name == result_name)
            spheroids = collection.get_spheroids(result, time_period)
            
            # Process each spheroid series
            for spheroid_series in spheroids:
                metadata = collection.replicate_metadata[spheroid_series.name]
                
                # Analyze each timepoint
                for timepoint, spheroid_image in spheroid_series.spheroid_image_dict.items():
                    try:
                        metrics = analysis_func(spheroid_image)
                        result_entry = {
                            'timepoint': datetime.strptime(timepoint, '%Y-%m-%d %H:%M:%S'),
                            'result': metadata["result_name"],
                            'replicate_num': metadata["replicate_number"],
                            'spheroid_name': spheroid_series.name
                        }
                        result_entry.update(metrics)
                        results_data.append(result_entry)
                    except Exception as e:
                        print(f"Error analyzing {spheroid_series.name} at {timepoint}: {e}")
        
        return pd.DataFrame(results_data)

    def analyze_functional_structure(self, condition: str, 
                                   live_channel: str | None = None,
                                   death_channel: str | None = None) -> pd.DataFrame:
        """Analyze functional radii and structure across replicates.
        
        Convenience method that uses analyze_by_replicate to compute
        functional radii and related metrics.
        
        Args:
            condition: Condition to analyze
            live_channel: Override default live cell channel
            death_channel: Override default dead cell channel
            
        Returns:
            DataFrame with functional structure metrics by replicate
        """
        live_ch = live_channel or self.config["live_channel"]
        death_ch = death_channel or self.config["death_channel"]
        
        def analyze_structure(spheroid):
            # Get functional radii
            radii = spheroid.functional_radius(
                live_color=live_ch,
                death_color=death_ch
            )
            
            # Calculate derived metrics
            metrics = {
                'outer_radius': radii['outer'],
                'inhibited_radius': radii['inhibited'],
                'necrotic_radius': radii['necrotic'],
            }
            
            # Add relative sizes if available
            if radii['outer']:
                if radii['inhibited']:
                    metrics['inhibited_fraction'] = radii['inhibited'] / radii['outer']
                if radii['necrotic']:
                    metrics['necrotic_fraction'] = radii['necrotic'] / radii['outer']
                
            return metrics
        
        return self.analyze_by_replicate(condition, analyze_structure)

    @classmethod
    def from_file(cls, name: str, index: int, experiment: Experiment, 
                  instance_group: h5py.Group, hdf5_path: str) -> 'Analysis':
        """Load Analysis instance from HDF5 file.
        
        Args:
            name: Name of the analysis
            index: Index in experiment
            experiment: Parent Experiment instance
            instance_group: HDF5 group containing analysis data
            hdf5_path: Path to HDF5 file
            
        Returns:
            Loaded Analysis instance
        """
        analysis = object.__new__(cls)
        
        # Initialize basic attributes
        analysis.name = name
        analysis.index = index
        analysis.experiment = experiment
        analysis.type_name = 'Analysis'
        analysis.hdf5_key = f'Analysis/{index}-{name}'
        analysis.replicates_dict = {}
        analysis.results = []
        analysis.analysis_results = {}
        analysis.config = {}
        
        # Load replicates if they exist
        if 'replicates' in instance_group:
            replicates_group = instance_group['replicates']
            
            # Load results
            if 'results' in replicates_group:
                results_group = replicates_group['results']
                for result_name in results_group:
                    result = experiment.results_dict[result_name]
                    analysis.results.append(result)
                    
                    # Load conditions for this result
                    result_replicates = result.replicates
                    
                    # Add SpheroidCollections to condition groups
                    for condition, spheroid_collection in result_replicates.items():
                        if condition not in analysis.replicates_dict:
                            analysis.replicates_dict[condition] = []
                            
                        # Restore ignored spheroids if they exist
                        if condition in replicates_group['conditions']:
                            condition_group = replicates_group['conditions'][condition]
                            for collection_name in condition_group:
                                collection_group = condition_group[collection_name]
                                if collection_group.attrs['result'] == result.name:
                                    if 'ignored_spheroids' in collection_group:
                                        ignored = [s.decode('utf-8') for s in 
                                                 collection_group['ignored_spheroids'][:]]
                                        spheroid_collection.ignored_spheroids.update(ignored)
                                        
                        analysis.replicates_dict[condition].append(spheroid_collection)
        
        analysis.time_periods = {}
        analysis.active_period = None
        
        # Load time periods if they exist
        if 'time_periods' in instance_group:
            periods_group = instance_group['time_periods']
            for period_name in periods_group:
                period_group = periods_group[period_name]
                
                # Load relative time if exists
                relative = None
                if 'relative' in period_group:
                    relative = tuple(period_group['relative'][:])
                
                # Load absolute times if exist
                absolute = None
                if 'absolute' in period_group:
                    absolute = {}
                    for result_name in period_group['absolute']:
                        dates = period_group['absolute'][result_name]
                        absolute[result_name] = (
                            datetime.fromisoformat(dates.attrs['start']),
                            datetime.fromisoformat(dates.attrs['end'])
                        )
                
                # Create TimeWindow
                analysis.time_periods[period_name] = TimeWindow(
                    name=period_name,
                    relative=relative,
                    absolute=absolute
                )
        
        return analysis

    def x__repr__(self) -> str:
        """String representation showing analysis details."""
        replicate_counts = {k: len(v.get_active_spheroids()) 
                          for k, v in self.replicates_dict.items()}
        
        return (f"ANALYSIS: {self.name}\n\n"
                f"Number of conditions: {len(replicate_counts)}\n"
                f"Conditions and replicates: {replicate_counts}\n"
                f"Added results: {', '.join(r.name for r in self.results)}")

    def set_config(self, **kwargs):
        """Update analysis configuration.
        
        Args:
            **kwargs: Configuration parameters to update
            
        Example:
            analysis.set_config(
                live_channel="green",
                threshold_method="adaptive",
                min_size=200
            )
        """
        self.config.update(kwargs)

    def get_config(self) -> dict:
        """Get current analysis configuration.
        
        Returns:
            Dictionary of current configuration parameters
        """
        return self.config.copy()

    def add_time_period(self, name: str, 
                       start_times: dict[str, str | datetime | None] | None = None,
                       end_times: dict[str, str | datetime | None] | None = None,
                       conditions: list[str] | None = None):
        """Add time periods to spheroids.
        
        Args:
            name: Name for the time period
            start_times: Dict mapping result names to start times (or None for first timepoint)
            end_times: Dict mapping result names to end times (or None for last timepoint)
            conditions: Optional list of conditions to apply to
            
        Example:
            # Specific time ranges for each result
            analysis.add_time_period(
                "phase1",
                start_times={
                    'Day1_Result': '2024-01-01 00:00:00',
                    'Day2_Result': '2024-01-02 00:00:00'
                },
                end_times={
                    'Day1_Result': '2024-01-02 00:00:00',
                    'Day2_Result': '2024-01-03 00:00:00'
                }
            )
            
            # From start of each result to specific times
            analysis.add_time_period(
                "early",
                end_times={
                    'Day1_Result': '2024-01-01 12:00:00',
                    'Day2_Result': '2024-01-02 12:00:00'
                }
            )
            
            # All timepoints for all results
            analysis.add_time_period("all")
        """
        target_conditions = conditions or self.replicates_dict.keys()
        start_times = start_times or {}
        end_times = end_times or {}
        
        for condition in target_conditions:
            if condition in self.replicates_dict:
                collection = self.replicates_dict[condition]
                
                # Add time period to each spheroid in collection
                for spheroid in collection.get_active_spheroids():
                    result_name = collection.replicate_metadata[spheroid.name]["result_name"]
                    spheroid.add_time_period(
                        name,
                        start_times.get(result_name),  # None if result not in dict
                        end_times.get(result_name)     # None if result not in dict
                    )

    def get_time_periods(self) -> dict:
        """Get all defined time periods.
        
        Returns:
            Dictionary mapping period names to TimeWindow objects
        """
        return self.time_periods.copy()

    def get_active_period(self, condition: str | None = None) -> str | None:
        """Get active time period name.
        
        Args:
            condition: Optional condition to check specific collection
            
        Returns:
            Name of active period or None
        """
        if condition:
            if condition not in self.replicates_dict:
                raise ValueError(f"Condition '{condition}' not found")
            return self.replicates_dict[condition].active_period
        return self.active_period

    def save_to_hdf5(self):
        """Save analysis state to HDF5."""
        with h5py.File(self.hdf5_path, 'a') as hdf_file:
            analysis_group = hdf_file[self.hdf5_key]
            
            # Save time periods
            if 'time_periods' in analysis_group:
                del analysis_group['time_periods']
            periods_group = analysis_group.create_group('time_periods')
            
            for name, window in self.time_periods.items():
                period_group = periods_group.create_group(name)
                
                # Save relative time if exists
                if window.relative:
                    period_group.create_dataset('relative', data=window.relative)
                
                # Save absolute times if exist
                if window.absolute:
                    abs_group = period_group.create_group('absolute')
                    for result_name, (start, end) in window.absolute.items():
                        result_group = abs_group.create_group(result_name)
                        result_group.attrs['start'] = start.isoformat()
                        result_group.attrs['end'] = end.isoformat()
