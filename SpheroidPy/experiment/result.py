from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
import os
import base64
import re

import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ipywidgets as widgets
from IPython.display import display
import cv2

from SpheroidPy.spheroid.spheroid_collection import SpheroidCollection
from SpheroidPy.spheroid.spheroid_series import SpheroidSeries
from SpheroidPy.spheroid.spheroid_image import SpheroidImage
from SpheroidPy.utils.color_palettes import CARTO_SEQUENTIAL, pick_maximally_distinct_palettes


@dataclass
class Condition:
    """Represents a numerical condition dimension."""
    name: str
    value: float | int | None = None
    attrs: dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.value is not None and not isinstance(self.value, (int, float)):
            raise ValueError(f"Numerical condition must have numeric value, got {type(self.value)}")


class Result:
    """Container that compares multiple spheroid collections across numerical condition dimensions.
    
    The workflow focuses on numerical conditions only and treats collections with
    identical condition values as biological replicates:
    
    Examples:
        Single numerical condition:
            >>> result = Result(name="drug_screen", condition="concentration")
            >>> result.condition("concentration", {"unit": "µM"})
            >>> result.add_collection(collection, condition=0.1)
        
        Multiple numerical conditions:
            >>> result = Result(name="experiment_1", condition=["dose", "temperature"])
            >>> result.condition("dose", {"unit": "µM"})
            >>> result.condition("temperature", {"unit": "°C"})
            >>> result.add_collection(collection_a, {"dose": 0.5, "temperature": 37})
            >>> result.add_collection(collection_b, {"dose": 0.5, "temperature": 37})  # biological replicate
    """

    def __init__(
        self,
        name: str,
        condition: str | list[str] | None = None,
        attrs: dict[str, Any | dict[str, Any]] | None = None,
        collections: Mapping[tuple | Any, SpheroidCollection | list[SpheroidCollection]] | None = None,
        description: str | None = None,
    ) -> None:
        """Create a new result container with numerical conditions.
        
        Args:
            name: Display name of the result.
            condition: Single condition name, list of names, or None for free-form
                collections (conditions can be added later via ``Result.condition``).
            attrs: Optional mapping of attribute names to values. Values can be:
                - Scalar (applied to all conditions)
                - Dict keyed by condition name
            collections: Optional mapping of condition values -> ``SpheroidCollection``.
                Keys can be:
                - Single numeric value for single-condition results
                - Tuple/list of numeric values ordered like ``condition``
                - Dict mapping condition name -> numeric value
            description: Free-form metadata saved alongside the result.
        """
        self.name = name
        self.description = description
        self.created_at = datetime.now()
        self.modified_at = datetime.now()
        self._metric_cache: dict[str, pd.DataFrame] = {}
        self.experiment = None  # Will be set when added to an Experiment
        self._result_index: int | None = None  # Index within experiment (set when added to Experiment)
        
        # Optional: Live-cell imaging support (replicates with platemaps)
        self.replicates_list: list = []  # List of LiveCellReplicate objects
        
        # Normalize condition names
        if condition is None:
            condition_names: list[str] = []
        elif isinstance(condition, str):
            condition_names = [condition]
        elif isinstance(condition, list):
            condition_names = condition
        else:
            raise TypeError(f"condition must be str, list[str], or None, got {type(condition)}")
        
        attrs = attrs or {}
        self.conditions: list[Condition] = []
        for cond_name in condition_names:
            cond_attrs: dict[str, Any] = {}
            for attr_name, attr_value in attrs.items():
                if isinstance(attr_value, dict):
                    if cond_name in attr_value:
                        cond_attrs[attr_name] = attr_value[cond_name]
                else:
                    cond_attrs[attr_name] = attr_value
            self.conditions.append(Condition(name=cond_name, attrs=cond_attrs))
        
        self.num_conditions = len(self.conditions)
        
        # Collections stored as dict mapping condition tuple -> list of collections
        # Key is a tuple of condition values in the order of self.conditions
        self.collections: dict[tuple, list[SpheroidCollection]] = {}
        
        if collections:
            for cond_key, value in collections.items():
                cond_tuple = self._normalize_condition_key(cond_key)
                
                if isinstance(value, SpheroidCollection):
                    self.collections.setdefault(cond_tuple, []).append(value)
                elif isinstance(value, list):
                    if not all(isinstance(v, SpheroidCollection) for v in value):
                        raise TypeError("All items in collection list must be SpheroidCollection instances")
                    self.collections.setdefault(cond_tuple, []).extend(value)
                else:
                    raise TypeError("collections values must be SpheroidCollection or list of SpheroidCollection")
        
        # Assign colors to collections that don't have one
        if collections:
            self._assign_collection_colors()

    # ------------------------------------------------------------------ #
    # Live-cell imaging support (optional)
    # ------------------------------------------------------------------ #
    def replicate(self, name: str, layout: int = 96) -> "LiveCellReplicate":
        """Get or create a live-cell replicate by name (optional feature).
        
        If a replicate with the given name already exists, it is returned.
        Otherwise, a new LiveCellReplicate is created with its own Platemap.
        This is useful for plate-based experiments where each biological replicate
        has its own plate layout.
        
        Args:
            name: Display name of the replicate
            layout: Number of wells in the plate (currently only 96 is supported, default: 96)
            
        Returns:
            The existing or newly created LiveCellReplicate object
            
        Example:
            >>> result = Result(name="experiment1")
            >>> replicate = result.replicate("replicate1")
            >>> replicate.platemap.cell_line("Huh7", {...})
            >>> replicate.load_images(...)
        """
        # Check if replicate with this name already exists
        for existing_replicate in self.replicates_list:
            if existing_replicate.name == name:
                return existing_replicate
        
        # Create new replicate
        from SpheroidPy.experiment.livecell_replicate import LiveCellReplicate
        replicate = LiveCellReplicate(name=name, result=self, layout=layout)
        self.replicates_list.append(replicate)
        self.modified_at = datetime.now()
        
        # Update collections from all replicates
        self._update_collections_from_replicates()
        
        return replicate
    
    def add_replicate(self, name: str, layout: int = 96) -> "LiveCellReplicate":
        """Add a new live-cell replicate to this result (optional feature).
        
        .. deprecated:: 
            Use :meth:`replicate` instead. This method is kept for backward compatibility.
        
        Creates a LiveCellReplicate with its own Platemap. This is useful for
        plate-based experiments where each biological replicate has its own plate layout.
        
        Args:
            name: Display name of the replicate
            layout: Number of wells in the plate (currently only 96 is supported, default: 96)
            
        Returns:
            The newly created LiveCellReplicate object
            
        Example:
            >>> result = Result(name="experiment1")
            >>> replicate = result.add_replicate("replicate1")
            >>> replicate.platemap.cell_line("Huh7", {...})
            >>> replicate.load_images(...)
        """
        return self.replicate(name, layout=layout)
    
    def _update_collections_from_replicates(self):
        """Update collections from all replicates (internal method).
        
        Recreates collections based on current platemap state. This ensures that
        platemap changes are automatically reflected in the collections.
        Also automatically creates conditions from platemap data.
        """
        if not hasattr(self, 'replicates_list') or not self.replicates_list:
            return
        
        # First, extract conditions from all replicates' platemaps
        # Collect all unique condition names and their values
        condition_data = {}  # {condition_name: set of values}
        
        for replicate in self.replicates_list:
            if not replicate.data_has_been_loaded:
                continue
            try:
                replicate_wells = replicate.platemap.replicates()
                for condition_tuple, wells in replicate_wells.items():
                    # Condition tuple format: ((name, value),) or similar
                    if isinstance(condition_tuple, tuple) and len(condition_tuple) > 0:
                        cond_item = condition_tuple[0]
                        if isinstance(cond_item, tuple) and len(cond_item) == 2:
                            cond_name, cond_value = cond_item
                            if cond_name not in condition_data:
                                condition_data[cond_name] = set()
                            # Only add numeric values
                            if isinstance(cond_value, (int, float)):
                                condition_data[cond_name].add(cond_value)
            except Exception as e:
                import logging
                logger = logging.getLogger("SpheroidPy.experiment.result")
                logger.warning(f"Could not extract conditions from replicate '{replicate.name}': {e}")
        
        # Create/update conditions from platemap data
        # Only create conditions if we don't have any yet, or if we're adding new ones
        if condition_data:
            for cond_name, values in condition_data.items():
                # Check if condition already exists
                existing_cond = next((c for c in self.conditions if c.name == cond_name), None)
                if not existing_cond:
                    # Create new condition
                    self.conditions.append(Condition(name=cond_name, attrs={}))
                    self.num_conditions = len(self.conditions)
        
        # Track which collections came from replicates (to remove old ones that are no longer valid)
        # But preserve collections that are already linked and still valid
        old_replicate_collections = set()
        for cond_tuple, coll_list in self.collections.items():
            for coll in coll_list:
                if hasattr(coll, '_replicate_source') and coll._replicate_source:
                    old_replicate_collections.add((cond_tuple, coll))
        
        # Collect collections from all replicates (recreated based on current platemap)
        new_collections_by_condition = {}  # Track new collections by condition
        for replicate in self.replicates_list:
            if not replicate.data_has_been_loaded:
                continue
            try:
                replicate_collections = replicate.get_collections()
                for condition_tuple, collection in replicate_collections.items():
                    # Convert platemap condition tuple to Result condition tuple
                    # Platemap format: ((name, value),) -> Result format: (value,) for single condition
                    # or ((name1, value1), (name2, value2)) -> (value1, value2) for multiple
                    result_cond_tuple = self._convert_platemap_condition_to_result_condition(condition_tuple)
                    
                    if result_cond_tuple not in new_collections_by_condition:
                        new_collections_by_condition[result_cond_tuple] = []
                    new_collections_by_condition[result_cond_tuple].append(collection)
            except Exception as e:
                import logging
                logger = logging.getLogger("SpheroidPy.experiment.result")
                logger.warning(f"Could not update collections from replicate '{replicate.name}': {e}")
        
        # Remove old replicate collections that are no longer in new collections
        # (i.e., they were removed from platemap or replicate was deleted)
        # Compare by name, not object identity, since get_collections() creates new objects each time
        for cond_tuple, coll in old_replicate_collections:
            # Check if this collection is still in new collections (by name)
            still_exists = False
            if cond_tuple in new_collections_by_condition:
                for new_coll in new_collections_by_condition[cond_tuple]:
                    if new_coll.name == coll.name:
                        still_exists = True
                        break
            
            if not still_exists:
                # Collection no longer exists, remove it
                if cond_tuple in self.collections and coll in self.collections[cond_tuple]:
                    self.collections[cond_tuple].remove(coll)
                    if not self.collections[cond_tuple]:
                        del self.collections[cond_tuple]
        
        # Add new collections (or reuse existing ones)
        for result_cond_tuple, new_coll_list in new_collections_by_condition.items():
            if result_cond_tuple not in self.collections:
                self.collections[result_cond_tuple] = []
            
            for collection in new_coll_list:
                # Check if a collection with the same name already exists
                # Since collection names now include replicate name (e.g., "R1_2e4"),
                # each biological replicate gets its own collection
                existing_collection = None
                existing_index = None
                for idx, existing_coll in enumerate(self.collections[result_cond_tuple]):
                    if existing_coll.name == collection.name:
                        existing_collection = existing_coll
                        existing_index = idx
                        break
                
                if existing_collection:
                    # Collection with this name already exists - update it to match current state
                    # Update replicate source if needed
                    if not hasattr(existing_collection, '_replicate_source'):
                        existing_collection._replicate_source = []
                    # Add replicate sources from new collection to existing
                    if hasattr(collection, '_replicate_source'):
                        for rep in collection._replicate_source:
                            if rep not in existing_collection._replicate_source:
                                existing_collection._replicate_source.append(rep)
                    
                    # Update SpheroidSeries list to match the new collection (in case wells changed)
                    # First, remove any duplicates that might have been added previously
                    # Create a list of unique series by name (keep first occurrence)
                    seen_names = set()
                    unique_series = []
                    for s in existing_collection.spheroid_series:
                        if s.name not in seen_names:
                            seen_names.add(s.name)
                            unique_series.append(s)
                    existing_collection.spheroid_series = unique_series
                    
                    # Now update to match the new collection's series
                    existing_series_names = {s.name for s in existing_collection.spheroid_series}
                    new_series_names = {s.name for s in collection.spheroid_series}
                    
                    # Create a mapping from name to series for the new collection
                    new_series_by_name = {s.name: s for s in collection.spheroid_series}
                    
                    # Rebuild the series list based on new collection, but keep existing objects where possible
                    updated_series = []
                    for s_name in new_series_names:
                        if s_name in existing_series_names:
                            # Keep existing series object
                            existing_series = next(s for s in existing_collection.spheroid_series if s.name == s_name)
                            updated_series.append(existing_series)
                        else:
                            # Add new series
                            updated_series.append(new_series_by_name[s_name])
                    
                    existing_collection.spheroid_series = updated_series
                else:
                    # Add new collection as a separate biological replicate
                    # Even if condition values are the same, different replicates should be separate
                    self.collections[result_cond_tuple].append(collection)
        
        # Clear metric cache when collections change
        self._metric_cache.clear()
    
    def _update_spheroid_image_hdf5_keys(self, result_index: int = 0) -> None:
        """Update hdf5_key attributes for all SpheroidImages to point to Collections structure.
        
        This ensures that SpheroidImages have the correct hdf5_key that matches the
        actual HDF5 file structure in Collections/, not the temporary ReplicateInfo structure.
        
        Args:
            result_index: Index for this result in the experiment (for HDF5 key generation).
        """
        result_key = f"Result/{result_index}-{self.name}"
        
        for cond_tuple, coll_list in self.collections.items():
            # Create condition group name
            cond_parts = []
            for cond, val in zip(self.conditions, cond_tuple):
                if 'unit' in cond.attrs:
                    cond_parts.append(f"{cond.name}={val}{cond.attrs['unit']}")
                else:
                    cond_parts.append(f"{cond.name}={val}")
            cond_group_name = "_".join(cond_parts)
            
            for idx, collection in enumerate(coll_list):
                # Determine collection group name (sanitized, with index if needed)
                # This logic must match the logic in save_to_hdf5()
                safe_collection_name = re.sub(r'[^0-9a-zA-Z_]', '_', collection.name)
                if safe_collection_name and safe_collection_name[0].isdigit():
                    safe_collection_name = f'_{safe_collection_name}'
                
                # Check if we need to add index suffix (for uniqueness)
                # Match the logic in save_to_hdf5: if a previous collection has the same sanitized name,
                # we need to add an index. But we need to simulate the "already exists" check.
                collection_group_name = safe_collection_name
                # Check if any previous collection in this condition tuple has the same sanitized name
                for prev_idx in range(idx):
                    prev_collection = coll_list[prev_idx]
                    prev_safe_name = re.sub(r'[^0-9a-zA-Z_]', '_', prev_collection.name)
                    if prev_safe_name and prev_safe_name[0].isdigit():
                        prev_safe_name = f'_{prev_safe_name}'
                    if prev_safe_name == safe_collection_name:
                        # A previous collection has the same name, so this one needs an index
                        collection_group_name = f"{safe_collection_name}_{idx}"
                        break
                
                # Update hdf5_key for all SpheroidImages in this collection
                for series in collection.get_spheroids():
                    for timepoint, spheroid_image in series.spheroid_image_dict.items():
                        timepoint_str = timepoint.isoformat() if isinstance(timepoint, datetime) else str(timepoint)
                        # Construct the correct hdf5_key pointing to Collections structure
                        correct_hdf5_key = f"{result_key}/Collections/{cond_group_name}/{collection_group_name}/Series/{series.name}/ImageSeries/{timepoint_str}"
                        spheroid_image.hdf5_key = correct_hdf5_key
                        # Also ensure hdf5_path is set correctly
                        if hasattr(self, 'hdf5_path') and self.hdf5_path:
                            spheroid_image.hdf5_path = str(self.hdf5_path)
    
    def _convert_platemap_condition_to_result_condition(self, platemap_condition: tuple) -> tuple:
        """Convert platemap condition tuple to Result condition tuple.
        
        Platemap format: ((name, value),) or ((name1, value1), (name2, value2))
        Result format: (value,) or (value1, value2) ordered by condition names
        
        Args:
            platemap_condition: Condition tuple from platemap
            
        Returns:
            Tuple of condition values in the order of self.conditions
        """
        if not platemap_condition:
            return tuple()
        
        # Extract name-value pairs from platemap condition
        name_value_pairs = {}
        for item in platemap_condition:
            if isinstance(item, tuple) and len(item) == 2:
                name, value = item
                name_value_pairs[name] = value
        
        # If no conditions exist yet, return tuple with just the values (will be handled by _update_collections_from_replicates)
        if not self.conditions:
            # Return values in order they appear in platemap_condition
            return tuple(name_value_pairs.values())
        
        # Build result condition tuple in the order of self.conditions
        result_values = []
        for cond in self.conditions:
            if cond.name in name_value_pairs:
                result_values.append(name_value_pairs[cond.name])
            else:
                # If condition name not found, use None (shouldn't happen in normal flow)
                result_values.append(None)
        
        return tuple(result_values)
    
    # ------------------------------------------------------------------ #
    # Collection handling
    # ------------------------------------------------------------------ #
    def _normalize_condition_key(self, key: Any) -> tuple:
        """Normalize condition key to tuple format."""
        if self.num_conditions == 0:
            raise ValueError("No conditions defined for this result.")
        
        def _ensure_numeric(val: Any) -> float | int:
            if not isinstance(val, (int, float)):
                raise ValueError(f"Condition values must be numeric, got {type(val)}")
            return val
        
        # Dict keyed by condition name
        if isinstance(key, dict):
            missing = [c.name for c in self.conditions if c.name not in key]
            extra = [k for k in key.keys() if k not in {c.name for c in self.conditions}]
            if missing:
                raise ValueError(f"Missing values for conditions: {missing}")
            if extra:
                raise ValueError(f"Unknown condition names provided: {extra}")
            return tuple(_ensure_numeric(key[c.name]) for c in self.conditions)
        
        # Tuple/list positional
        if isinstance(key, (tuple, list)):
            if len(key) != self.num_conditions:
                raise ValueError(f"Condition key must have {self.num_conditions} values, got {len(key)}")
            return tuple(_ensure_numeric(v) for v in key)
        
        # Single numeric value
        if self.num_conditions != 1:
            raise ValueError(f"Single value only valid for 1 condition, but have {self.num_conditions}")
        return (_ensure_numeric(key),)
    
    def _condition_label(self, cond_tuple: tuple) -> str:
        """Generate human-readable label for a condition tuple."""
        parts = []
        for idx, (cond, val) in enumerate(zip(self.conditions, cond_tuple)):
            if 'unit' in cond.attrs:
                parts.append(f"{cond.name}={val}{cond.attrs['unit']}")
            else:
                parts.append(f"{cond.name}={val}")
        return ", ".join(parts)
    
    def _iter_collections(self):
        """Iterate over all collections with their condition keys and indices."""
        for cond_tuple, coll_list in self.collections.items():
            for idx, coll in enumerate(coll_list):
                yield cond_tuple, idx, coll

    def _collection_label(self, cond_tuple: tuple, idx: int) -> str:
        """Human readable collection label used in plotting and UI."""
        cond_label = self._condition_label(cond_tuple)
        return f"{cond_label}; R{idx + 1}"

    def _assign_collection_colors(self) -> None:
        """Assign distinct colors to collections that don't have one."""
        collections_needing_color = [(cond_value, idx, coll) for cond_value, idx, coll in self._iter_collections()
                                     if not coll.color or coll.color not in CARTO_SEQUENTIAL]
        if not collections_needing_color:
            return

        existing_colors = {
            coll.color for _, _, coll in self._iter_collections()
            if coll.color and coll.color in CARTO_SEQUENTIAL
        }

        num_needed = len(collections_needing_color)
        all_palettes = list(CARTO_SEQUENTIAL.keys())
        available_palettes = [p for p in all_palettes if p not in existing_colors]

        selected_palettes = pick_maximally_distinct_palettes(
            min(num_needed, len(available_palettes)),
            candidates=available_palettes
        )

        for (_, _, collection), palette_name in zip(collections_needing_color, selected_palettes):
            collection.color = palette_name
    
    def add_collection(self, collection: SpheroidCollection, condition: float | dict[str, float] | tuple | list) -> None:
        """Attach a new ``SpheroidCollection`` for given condition value(s).
        
        Collections sharing identical condition values are treated as biological
        replicates and stored under the same condition tuple.
        
        Args:
            collection: SpheroidCollection to add
            condition: Condition value(s) for this collection
            
        Raises:
            TypeError: If collection is not a SpheroidCollection
            ValueError: If collection name is reserved (e.g., "ReplicateInfo")
        """
        if not isinstance(collection, SpheroidCollection):
            raise TypeError("collection must be an instance of SpheroidCollection")
        
        # Prevent reserved names that conflict with HDF5 structure
        RESERVED_NAMES = {"ReplicateInfo", "Replicates", "Collections", "Conditions", "Series", "ImageSeries"}
        if collection.name in RESERVED_NAMES:
            raise ValueError(
                f"Collection name '{collection.name}' is reserved for HDF5 structure. "
                f"Please choose a different name. Reserved names: {RESERVED_NAMES}"
            )
        
        cond_tuple = self._normalize_condition_key(condition)

        if not collection.color or collection.color not in CARTO_SEQUENTIAL:
            existing_colors = {
                coll.color for _, _, coll in self._iter_collections()
                if coll.color and coll.color in CARTO_SEQUENTIAL
            }
            available_palettes = [p for p in CARTO_SEQUENTIAL.keys() if p not in existing_colors]
            if available_palettes:
                selected = pick_maximally_distinct_palettes(1, candidates=available_palettes)
                if selected:
                    collection.color = selected[0]

        self.collections.setdefault(cond_tuple, []).append(collection)
        self._metric_cache.clear()
        self.modified_at = datetime.now()

    def condition(self, name: str, attrs: dict[str, Any] | None = None) -> None:
        """Add or update a numerical condition definition."""
        attrs = attrs or {}
        existing = next((c for c in self.conditions if c.name == name), None)
        if existing:
            existing.attrs.update(attrs)
            self.modified_at = datetime.now()
            return
        if self.collections:
            raise ValueError("Cannot add new condition when collections already exist.")
        self.conditions.append(Condition(name=name, attrs=attrs))
        self.num_conditions = len(self.conditions)

    def remove_collection(self, condition_values: tuple | list | Any) -> None:
        """Remove all collections for given condition value(s) from the result.

        Args:
            condition_values: Condition value(s) matching the condition dimensions.
                Can be single value or list/tuple of values.
        """
        cond_tuple = self._normalize_condition_key(condition_values)
        if cond_tuple in self.collections:
            del self.collections[cond_tuple]
            self._metric_cache.clear()
            self.modified_at = datetime.now()

    # ------------------------------------------------------------------ #
    # Metric utilities
    # ------------------------------------------------------------------ #
    def metric(
        self,
        name: str = "radius",
        *,
        average: bool = False,
        mean: bool = False,
        pool: bool = False,
        ignore_border: bool = True,
        plot: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        """Calculate a metric across all attached collections.
        
        Interpretation of replicate levels:
            - Technical replicates = individual spheroid series within a ``SpheroidCollection``
            - Biological replicates = multiple ``SpheroidCollection`` instances sharing the same condition values
        
        The boolean switches control how these levels are handled:
        
        - ``average``: Average technical replicates *within each collection* (per-condition, per-collection curves).
          This is a within-collection operation only and does **not** mix biological replicates.
        
        - ``mean``: Aggregate over *biological replicates* (collections with identical condition values).
          When ``mean=True`` the following holds:
              * ``mean`` overrides ``average`` (``mean`` schlägt ``average``).
              * Interpolation is always enabled so that missing time points (z.B. keine Kontur) are filled
                before aggregation.
        
        - ``pool``:
              * If ``pool=False`` (default) and ``mean=True``:
                    All available replicates (technische + biologische) for a condition are pooled and a
                    single mean ± std is computed per timepoint.
              * If ``pool=True`` and ``mean=True``:
                    Two-stage aggregation (Fehlerfortpflanzung):
                        1. First, technical replicates are averaged per collection
                           (mean ± std over all series in each collection).
                        2. Then, biological mean ± std is computed over these collection means.
        
        When ``mean=False`` no aggregation across biological replicates is performed; the return value then
        contains one curve per collection (and ggf. pro Series, abhängig von ``average``).

        Args:
            name: Metric forwarded to ``SpheroidCollection.metric`` / ``SpheroidSeries.metric``.
            average: If True and ``mean=False``, average technische Replikate innerhalb jeder Collection.
            mean: If True, berechne Mittelwert ± Std über biologische Replikate (siehe oben).
            pool: If True and ``mean=True``, first average biological replicates per collection,
                then compute mean over those (zweistufige Fehlerfortpflanzung).
            ignore_border: Whether to exclude spheroids touching image border when calculating metrics.
            If True, spheroids that touch the image border are excluded from all metric calculations
            (returns None/NaN). If False, metrics are calculated regardless of border contact.
            Recommended to keep True to avoid edge artifacts. Forwarded to collection/series metric calculation.
            plot: Whether to display a unified plot. If True, creates a single plot
                instead of calling individual collection plots.
            **kwargs: Additional keyword arguments passed to the underlying
                ``SpheroidCollection.metric`` call (except 'plot' which is handled here).
        """
        cache_key = f"{name}|average={average}|mean={mean}|pool={pool}|{tuple(sorted(kwargs.items()))}"
        if cache_key in self._metric_cache:
            combined = self._metric_cache[cache_key]
        else:
            # ---------------------- #
            # Aggregation strategy
            # ---------------------- #
            # NOTE: mean overrides average; for mean=True we always work on
            # non-averaged series and enable interpolation.

            if mean:
                # Build per-collection DataFrames with one column per technical replicate (series)
                per_cond_frames: dict[tuple, list[pd.DataFrame]] = {}

                for cond_tuple, idx, collection in self._iter_collections():
                    label = self._collection_label(cond_tuple, idx)
                    # Always use mean=False here to get individual series as replicates
                    coll_df = collection.metric(
                        name=name,
                        mean=False,
                        interpolate=True,  # ensure interpolation when computing biological means
                        ignore_border=ignore_border,
                        plot=False,
                        **{k: v for k, v in kwargs.items() if k not in {"plot", "mean", "average", "pool"}},
                    )
                    if coll_df.empty:
                        continue

                    # Ensure numeric dtype for aggregation
                    coll_df = coll_df.apply(pd.to_numeric, errors="coerce")

                    # Label columns with collection label + series id
                    if isinstance(coll_df.columns, pd.MultiIndex):
                        suffix = coll_df.columns.get_level_values(-1).tolist()
                    else:
                        suffix = coll_df.columns.tolist()
                    new_columns = [(label, s) for s in suffix]
                    coll_df.columns = pd.MultiIndex.from_tuples(new_columns)

                    per_cond_frames.setdefault(cond_tuple, []).append(coll_df)

                if not per_cond_frames:
                    raise ValueError("Result has no collections attached.")

                cond_results: list[pd.DataFrame] = []
                for cond_tuple, df_list in per_cond_frames.items():
                    # Combine all technical replicates for this condition
                    full_df = pd.concat(df_list, axis=1, sort=False)
                    full_df = full_df.sort_index(axis=1)

                    if not pool:
                        # Pool all technical + biological replicates together
                        mean_series = full_df.mean(axis=1, skipna=True)
                        std_series = full_df.std(axis=1, ddof=1, skipna=True)
                    else:
                        # Two-stage: first collapse technical replicates per collection,
                        # then average across collections (biological replicates).
                        per_collection_means: list[pd.Series] = []

                        # Group by first level = collection label
                        for coll_label in full_df.columns.get_level_values(0).unique():
                            sub = full_df.xs(coll_label, axis=1, level=0)
                            coll_mean = sub.mean(axis=1, skipna=True)
                            per_collection_means.append(coll_mean)

                        per_coll_df = pd.concat(per_collection_means, axis=1)
                        mean_series = per_coll_df.mean(axis=1, skipna=True)
                        std_series = per_coll_df.std(axis=1, ddof=1, skipna=True)

                    label = self._condition_label(cond_tuple)
                    cond_df = pd.concat(
                        [
                            mean_series.rename((label, "mean")),
                            std_series.rename((label, "std")),
                        ],
                        axis=1,
                    )
                    cond_results.append(cond_df)

                combined = pd.concat(cond_results, axis=1).sort_index(axis=1)
            else:
                # No aggregation across biological replicates; optionally average technical replicates
                frames: list[pd.DataFrame] = []
                for cond_tuple, idx, collection in self._iter_collections():
                    label = self._collection_label(cond_tuple, idx)
                    df = collection.metric(
                        name=name,
                        mean=average,
                        ignore_border=ignore_border,
                        plot=False,  # Never plot individual collections
                        **{k: v for k, v in kwargs.items() if k not in {"plot", "mean", "average", "pool"}},
                    )
                    if df.empty:
                        continue

                    # Handle MultiIndex columns from collection.metric(mean=True)
                    # When average=True, collection.metric returns columns like (collection.name, 'mean'), (collection.name, 'std')
                    if isinstance(df.columns, pd.MultiIndex):
                        # Extract the suffix (last level: 'mean', 'std', or series names)
                        suffix = df.columns.get_level_values(-1).tolist()
                        # Ensure we have a list of tuples for the new MultiIndex
                        new_columns = [(label, s) for s in suffix]
                    else:
                        # Simple column names - convert to list of tuples
                        suffix = df.columns.tolist() if hasattr(df.columns, 'tolist') else list(df.columns)
                        new_columns = [(label, s) for s in suffix]
                    
                    # Create new MultiIndex with our label as first level
                    df.columns = pd.MultiIndex.from_tuples(new_columns)
                    frames.append(df)

                if not frames:
                    raise ValueError("Result has no collections attached.")

                # Combine all frames, ensuring all collections are included
                combined = pd.concat(frames, axis=1, sort=False)
                # Sort columns by label for consistent ordering
                combined = combined.sort_index(axis=1)

            self._metric_cache[cache_key] = combined
        
        # Create unified plot if requested
        if plot:
            self._plot_metric(combined, name, mean, ignore_border, **kwargs)
        
        return combined

    def compare_metric(
        self,
        metric: str = "radius",
        timepoint: int | float | list[int | float] | None = None,
        *,
        ignore_border: bool = True,
        skip_nan: bool = True,
        figsize: tuple[int, int] = (3.5, 4),
    ) -> None:
        """
        Compare a metric across conditions as bar plots for given timepoints.

        For each requested timepoint a bar plot is shown where each bar is a condition
        (condition_name as group label, condition_value as x tick). Bars show the mean
        across collections (replicates), with std as error bar and replicate points overlaid.

        Args:
            metric: Metric name forwarded to collection.metric.
            timepoint: Single timepoint, list of timepoints, or None (defaults to last available).
            ignore_border: Forwarded to collection.metric.
            skip_nan: Skip NaNs when plotting replicate points.
            figsize: Figure size for each plot.
        """
        # normalize timepoints list
        if timepoint is None:
            tp_list: list[int | float | None] = [None]
        elif isinstance(timepoint, (int, float)):
            tp_list = [timepoint]
        else:
            tp_list = list(timepoint)

        records: list[dict] = []
        color_map: dict[tuple, str] = {}

        for cond_tuple, idx, collection in self._iter_collections():
            label = self._collection_label(cond_tuple, idx)
            df = collection.metric(
                name=metric,
                mean=True,
                ignore_border=ignore_border,
                plot=False,
            )
            # determine columns
            if isinstance(df.columns, pd.MultiIndex):
                mean_col = (collection.name, "mean")
                if mean_col not in df.columns:
                    continue
                series_mean = df[mean_col]
            else:
                series_mean = df.iloc[:, 0]

            # choose color for this condition (middle of palette)
            if collection.color and collection.color in CARTO_SEQUENTIAL:
                pal = CARTO_SEQUENTIAL[collection.color]
                mid_color = pal[len(pal) // 2]
            else:
                mid_color = None
            color_map.setdefault(cond_tuple, mid_color)

            for tp in tp_list:
                if tp is None:
                    # last valid
                    series_tp = series_mean.dropna()
                    if series_tp.empty:
                        continue
                    val = series_tp.iloc[-1]
                    tp_used = series_tp.index[-1]
                else:
                    if tp not in series_mean.index:
                        continue
                    val = series_mean.loc[tp]
                    tp_used = tp
                if skip_nan and (val is None or pd.isna(val)):
                    continue
                
                # Add all condition dimensions to record
                record = {
                    "collection_label": label,
                    "replicate": idx + 1,
                    "timepoint": tp_used,
                    "value": float(val),
                }
                # Add each condition dimension
                for cond, val in zip(self.conditions, cond_tuple):
                    record[cond.name] = val
                
                records.append(record)

        if not records:
            print("No data to plot for compare_metric.")
            return

        df_melt = pd.DataFrame(records)

        for tp in sorted(df_melt["timepoint"].unique()):
            df_tp = df_melt[df_melt["timepoint"] == tp]
            if df_tp.empty:
                continue

            # Create condition label combining all dimensions
            cond_cols = [cond.name for cond in self.conditions]
            if len(cond_cols) == 1:
                # Single condition - use it directly
                x_col = cond_cols[0]
            else:
                # Multiple conditions - combine them
                x_col = "condition_label"
                df_tp = df_tp.assign(
                    condition_label=df_tp.apply(
                        lambda r: ", ".join(f"{col}={r[col]}" for col in cond_cols),
                        axis=1
                    )
                )
            
            # Sort by condition values
            df_tp = df_tp.sort_values(by=cond_cols)

            # x tick labels
            df_tp = df_tp.assign(
                cond_label=df_tp.apply(lambda r: str(r[x_col]), axis=1)
            )

            # palette aligned to cond_label order
            palette = []
            for _, row in df_tp.drop_duplicates(subset="cond_label").iterrows():
                cond_tuple_key = tuple(row[col] for col in cond_cols)
                col = color_map.get(cond_tuple_key)
                palette.append(col if col else None)

            sns.set(style="white")
            plt.figure(figsize=figsize)

            ax = sns.barplot(
                data=df_tp,
                x="cond_label",
                y="value",
                hue="cond_label",
                palette=palette,
                legend=False,
                errorbar="sd",
                capsize=0.2,
                err_kws={"linewidth": 2},
                edgecolor="black",
                linewidth=2.0,
            )

            sns.stripplot(
                data=df_tp,
                x="cond_label",
                y="value",
                hue="cond_label",
                dodge=False,
                palette=palette,
                size=8,
                jitter=True,
                edgecolor="black",
                linewidth=1,
                ax=ax,
                legend=False,
            )

            ax.set_xlabel("Condition" if len(cond_cols) > 1 else cond_cols[0])
            ax.set_ylabel(metric)
            ax.set_title(f"{metric} at timepoint {tp}")
            ax.grid(True, axis="y", alpha=0.3)
            plt.tight_layout()
            plt.show()
    
    def _plot_metric(
        self,
        df: pd.DataFrame,
        name: str,
        mean: bool,
        ignore_border: bool,
        skip_nan: bool = True,
        **kwargs,
    ) -> None:
        """Create a unified plot for all collections in this result."""
        plt.figure(figsize=(10, 6))
        x = np.array(df.index) / 24  # Convert to days

        # Build label -> collection map for replicate-labeled columns
        label_map = {self._collection_label(cond_tuple, idx): coll for cond_tuple, idx, coll in self._iter_collections()}
        
        if mean:
            # Plot mean ± std for each collection
            if isinstance(df.columns, pd.MultiIndex):
                for label in df.columns.get_level_values(0).unique():
                    collection = label_map.get(label)
                    
                    # Get color for this collection
                    plot_color = None
                    if collection and collection.color and collection.color in CARTO_SEQUENTIAL:
                        palette = CARTO_SEQUENTIAL[collection.color]
                        plot_color = palette[len(palette) // 2]  # middle color for mean
                    
                    mean_col = (label, 'mean')
                    std_col = (label, 'std')
                    
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
                                       alpha=0.3,
                                       color='grey')
                        plt.plot(x_plot, mean_plot, 'o-', label=label)
            else:
                # Fallback for non-MultiIndex columns (shouldn't happen with mean=True, but handle it)
                for col in df.columns:
                    y = df[col]
                    if skip_nan:
                        valid_mask = ~np.isnan(y)
                        plt.plot(x[valid_mask], y[valid_mask], 'o-', label=col, alpha=0.7)
                    else:
                        plt.plot(x, y, 'o-', label=col, alpha=0.7)
        else:
            # Plot individual series from each collection
            if isinstance(df.columns, pd.MultiIndex):
                # MultiIndex columns: (collection_label, series_name)
                for label in df.columns.get_level_values(0).unique():
                    collection = label_map.get(label)
                    
                    # Get color palette for this collection
                    palette = None
                    if collection and collection.color and collection.color in CARTO_SEQUENTIAL:
                        palette = CARTO_SEQUENTIAL[collection.color]
                    
                    # Get all columns for this collection
                    collection_cols = [col for col in df.columns if col[0] == label]
                    num_colors = len(palette) if palette else 0
                    
                    for idx, col in enumerate(collection_cols):
                        y = df[col]
                        
                        # Use color from palette if available
                        if palette:
                            series_color = palette[idx % num_colors]
                        else:
                            series_color = None
                        
                        col_label = f"{label}: {col[1]}" if isinstance(col, tuple) and len(col) > 1 else str(col)
                        
                        if skip_nan:
                            valid_mask = ~np.isnan(y)
                            if series_color:
                                plt.plot(x[valid_mask], y[valid_mask], 'o-', 
                                       label=col_label,
                                       alpha=0.7, color=series_color)
                            else:
                                plt.plot(x[valid_mask], y[valid_mask], 'o-',
                                       label=col_label,
                                       alpha=0.7)
                        else:
                            if series_color:
                                plt.plot(x, y, 'o-',
                                       label=col_label,
                                       alpha=0.7, color=series_color)
                            else:
                                plt.plot(x, y, 'o-',
                                       label=col_label,
                                       alpha=0.7)
            else:
                # Simple column names - treat each column as a separate series
                for idx, col in enumerate(df.columns):
                    y = df[col]
                    # Try to determine collection from first available
                    collection = next(iter(label_map.values())) if label_map else None
                    
                    palette = None
                    if collection and collection.color and collection.color in CARTO_SEQUENTIAL:
                        palette = CARTO_SEQUENTIAL[collection.color]
                    
                    series_color = palette[idx % len(palette)] if palette else None
                    
                    if skip_nan:
                        valid_mask = ~np.isnan(y)
                        if series_color:
                            plt.plot(x[valid_mask], y[valid_mask], 'o-', 
                                   label=col,
                                   alpha=0.7, color=series_color)
                        else:
                            plt.plot(x[valid_mask], y[valid_mask], 'o-',
                                   label=col,
                                   alpha=0.7)
                    else:
                        if series_color:
                            plt.plot(x, y, 'o-',
                                   label=col,
                                   alpha=0.7, color=series_color)
                        else:
                            plt.plot(x, y, 'o-',
                                   label=col,
                                   alpha=0.7)
        
        plt.grid(True, alpha=0.3)
        plt.xlabel('Time [d]')
        plt.ylabel(name)
        plt.title(f'{name} over time - {self.name}' + (' (mean ± std)' if mean else ''))
        
        # Only show legend if not too many series
        #if not mean or len(df.columns.get_level_values(0).unique()) < 10:
        plt.legend()
        
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------ #
    # Segmentation
    # ------------------------------------------------------------------ #
    def segmentation(
        self,
        methods: list | tuple = [('thresholding', 'fluorescence_green'), ('ai', 'brightfield')],
        reconstruct_border: bool = True,
        border_margin: int = 10,
        show_progress: bool = True,
        **kwargs,
    ) -> None:
        """
        Segment all collections in this result.

        Args:
            methods: List/Tuple of segmentation specs, e.g.
                [('thresholding','fluorescence_green'), ('ai','brightfield')]
            reconstruct_border: Whether to reconstruct border if contour touches image border.
            border_margin: Margin (px) for border detection.
            show_progress: Whether to show progress bar (default: True). Set to False when called from higher-level methods.
            **kwargs: Forwarded to ``SpheroidCollection.segmentation`` (e.g. threshold, use_yen, confidence).
        """
        import multiprocessing as mp
        from tqdm import tqdm
        
        if not self.collections:
            print("No collections in result to segment.")
            return

        # Collect all series across all collections for unified progress tracking
        all_series_args = []
        series_to_collection = {}  # Map series_name -> (collection, cond_tuple, idx)
        
        for cond_tuple, idx, collection in self._iter_collections():
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
                series_to_collection[series.name] = (collection, cond_tuple, idx)

        if not all_series_args:
            print("No active spheroid series in result to segment.")
            return

        # Process all series in parallel with single progress bar
        num_cores = mp.cpu_count()
        with mp.Pool(processes=num_cores) as pool:
            if show_progress:
                results = list(tqdm(
                    pool.imap(SpheroidCollection._process_segmentation_series, all_series_args),
                    total=len(all_series_args),
                    desc=f"Result '{self.name}'"
                ))
            else:
                results = list(pool.imap(SpheroidCollection._process_segmentation_series, all_series_args))

        # Update SpheroidImage instances with segmentation results
        total_success = 0
        total_images = 0
        for series_name, success, total, contour_data in results:
            total_success += success
            total_images += total
            
            collection, cond_tuple, idx = series_to_collection[series_name]
            series = next(s for s in collection.get_spheroids() if s.name == series_name)
            for timepoint, data in contour_data.items():
                spheroid_image = series.spheroid_image_dict[timepoint]
                spheroid_image.contour = data['contour']
                spheroid_image.contour_touches_border = data['touches_border']

        # Save contours to HDF5 if collections have hdf5_path
        hdf5_paths = {coll.hdf5_path for _, _, coll in self._iter_collections() if coll.hdf5_path}
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

        if show_progress:
            print(f"\nSegmentation complete for result '{self.name}':")
            print(f"Successfully segmented {total_success}/{total_images} images across {len(all_series_args)} series")


    # ------------------------------------------------------------------ #
    # HDF5 Persistence
    # ------------------------------------------------------------------ #
    def radial_profile(self, timepoint: int | str | datetime,
                      condition: str | list[str],
                      channels: str | list[str] = ['green'],
                      absolute_radius: bool = False,
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
            Condition name(s) to compare. Must be a condition defined in this Result.
            If list, compares multiple conditions (they must be disjoint).
        channels : str | list[str], default=['green']
            Fluorescence channel(s) to analyze. Can be a single channel string ('green', 'red', 'blue')
            or a list of channels.
        absolute_radius : bool, default=False
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
            - distances: Distance array (relative or absolute) - may differ between conditions if absolute_radius=True
        """
        import matplotlib.pyplot as plt
        from SpheroidPy.utils.color_palettes import CARTO_SEQUENTIAL
        
        # Normalize channels to list
        if isinstance(channels, str):
            channels = [channels]
        
        if not self.collections:
            print("No collections found in result.")
            return {}
        
        # Normalize condition to list
        if isinstance(condition, str):
            conditions_to_compare = [condition]
        else:
            conditions_to_compare = list(condition)
        
        # Check if all specified conditions exist
        condition_names = [c.name for c in self.conditions]
        for cond_name in conditions_to_compare:
            if cond_name not in condition_names:
                raise ValueError(f"Condition '{cond_name}' not found in result. Available: {sorted(condition_names)}")
        
        # Get condition indices
        condition_indices = [condition_names.index(c) for c in conditions_to_compare]
        
        # Group collections by their values for the specified conditions
        # Check if conditions are disjoint
        condition_groups = {}  # {condition_value_tuple: list of collections}
        
        for cond_tuple, collection_list in self.collections.items():
            # Extract values for the specified conditions
            condition_values = tuple(cond_tuple[i] if i < len(cond_tuple) else None 
                                   for i in condition_indices)
            
            # Check if all specified conditions are present
            if None in condition_values:
                continue
            
            # Check for duplicates (non-disjoint conditions)
            if condition_values in condition_groups:
                raise ValueError(
                    f"Conditions are not disjoint: Multiple collections found with "
                    f"{dict(zip(conditions_to_compare, condition_values))}. "
                    f"Each condition value combination must appear in only one collection group."
                )
            
            condition_groups[condition_values] = collection_list
        
        if not condition_groups:
            raise ValueError(f"No collections found with all specified conditions: {conditions_to_compare}")
        
        # Helper function to convert timepoint to datetime
        def _ensure_dt(val):
            if isinstance(val, datetime):
                return val
            elif isinstance(val, str):
                return datetime.strptime(val, '%Y-%m-%d %H:%M:%S')
            elif isinstance(val, (int, float)):
                # Convert relative hours to datetime
                # Get first collection to find base timepoint
                first_collection_list = next(iter(condition_groups.values()))
                if not first_collection_list:
                    raise ValueError("No collections available")
                first_collection = first_collection_list[0]
                active_series = first_collection.get_spheroids()
                if not active_series:
                    raise ValueError("No active spheroid series in collection")
                sorted_times = sorted(active_series[0].spheroid_image_dict.keys())
                if not sorted_times:
                    raise ValueError("No timepoints available in series")
                t0 = sorted_times[0] if isinstance(sorted_times[0], datetime) else datetime.strptime(sorted_times[0], '%Y-%m-%d %H:%M:%S')
                return t0 + pd.Timedelta(hours=float(val))
            else:
                raise ValueError(f"Cannot convert {type(val)} to datetime")
        
        # Process timepoint
        tp_dt = _ensure_dt(timepoint)
        
        # Calculate profiles for each condition
        profiles_data = {}
        distances_data = {}
        
        for condition_values, collection_list in condition_groups.items():
            # Format condition label
            if len(conditions_to_compare) == 1:
                condition_label = f"{conditions_to_compare[0]}={condition_values[0]}"
            else:
                label_parts = [f"{c}={v}" for c, v in zip(conditions_to_compare, condition_values)]
                condition_label = "; ".join(label_parts)
            
            # Average across all collections in this group (biological replicates)
            all_profiles = []
            all_distances = []
            
            for collection in collection_list:
                try:
                    # Use collection's radial_profile method with mean=True
                    result = collection.radial_profile(
                        timepoint=tp_dt,
                        channels=channels,
                        absolute_radius=absolute_radius,
                        normalize=normalize,
                        smoothing=smoothing,
                        mean=True,  # Average across technical replicates
                        plot=False
                    )
                    
                    all_profiles.append(result['profiles'])
                    all_distances.append(result['distances'])
                    
                except Exception as e:
                    logger.warning(f"Failed to calculate radial profile for collection {collection.name}: {e}")
                    continue
            
            if not all_profiles:
                logger.warning(f"No valid profiles for condition {condition_label}")
                continue
            
            # Average across biological replicates
            averaged_profiles = {}
            for ch in channels:
                mean_list = []
                std_list = []
                for prof_dict in all_profiles:
                    if ch in prof_dict:
                        mean_list.append(prof_dict[ch]['mean'])
                        std_list.append(prof_dict[ch]['std'])
                
                if mean_list:
                    # Stack and average
                    mean_array = np.stack(mean_list, axis=0)
                    std_array = np.stack(std_list, axis=0)
                    
                    avg_mean = np.mean(mean_array, axis=0)
                    avg_std = np.sqrt(np.mean(std_array**2, axis=0) + np.var(mean_array, axis=0))
                    
                    averaged_profiles[ch] = {
                        'mean': avg_mean,
                        'std': avg_std
                    }
            
            profiles_data[condition_label] = averaged_profiles
            
            # Use first distance array (should be similar across replicates)
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
            
            # Generate colors for conditions
            n_conditions = len(profiles_data)
            condition_colors = plt.cm.tab10(np.linspace(0, 1, max(10, n_conditions)))
            
            for cond_idx, (condition_label, condition_profiles) in enumerate(profiles_data.items()):
                distances = distances_data.get(condition_label)
                if distances is None:
                    continue
                
                for ch_idx, channel in enumerate(channels):
                    if channel not in condition_profiles:
                        continue
                    
                    profile = condition_profiles[channel]
                    mean = profile['mean']
                    std = profile.get('std', np.zeros_like(mean))
                    
                    # Use condition color, with different line styles for different channels
                    color = condition_colors[cond_idx % len(condition_colors)]
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

    @property
    def hdf5_path(self) -> Path | None:
        """Get HDF5 file path from experiment, if available."""
        if self.experiment is not None and hasattr(self.experiment, 'hdf5_path'):
            return self.experiment.hdf5_path
        return None
    
    def save_to_hdf5(self, hdf5_path: Path | str, result_index: int | None = None) -> None:
        """Save the complete result hierarchy to HDF5.

        Args:
            hdf5_path: Path to the HDF5 file (typically the experiment's HDF5 file).
            result_index: Optional index override. If None, uses self._result_index.
        """
        # Use provided result_index or fall back to stored _result_index
        if result_index is None:
            result_index = self._result_index if hasattr(self, '_result_index') and self._result_index is not None else 0
        
        # Update collections from replicates before saving to ensure all data is included
        if hasattr(self, 'replicates_list') and self.replicates_list:
            self._update_collections_from_replicates()
        
        # Update hdf5_key attributes for all SpheroidImages to point to Collections structure
        self._update_spheroid_image_hdf5_keys(result_index)
        
        hdf5_path = Path(hdf5_path)
        # Full path for hdf5_key (used for SpheroidImage references)
        full_result_key = f"Result/{result_index}-{self.name}"
        # Just the name part (for accessing within result_parent)
        result_key_name = f"{result_index}-{self.name}"
        
        with h5py.File(hdf5_path, "a") as hdf_file:
            # Ensure Result group exists
            if "Result" not in hdf_file:
                hdf_file.create_group("Result", track_order=True)
            
            result_parent = hdf_file["Result"]
            
            # Remove any existing results with the same name but different index
            # This prevents duplicates when result_index changes
            existing_keys = list(result_parent.keys())
            for key in existing_keys:
                # Check if this key has the same result name but different index
                if key.endswith(f"-{self.name}"):
                    # Extract index from key
                    try:
                        existing_index_str = key.split("-", 1)[0]
                        existing_index = int(existing_index_str)
                        # If index is different, remove the old one
                        if key != result_key_name:
                            del result_parent[key]
                    except (ValueError, IndexError):
                        # If we can't parse the key, but it ends with the name, remove it if different
                        if key != result_key_name:
                            del result_parent[key]
            
            # Use result_key_name (without "Result/" prefix) since result_parent is already the Result group
            if result_key_name in result_parent:
                result_group = result_parent[result_key_name]
            else:
                result_group = result_parent.create_group(result_key_name, track_order=True)
            
            result_group.attrs["name"] = self.name
            result_group.attrs["description"] = self.description or ""
            result_group.attrs["created_at"] = self.created_at.isoformat()
            result_group.attrs["modified_at"] = self.modified_at.isoformat()
            
            # Save condition metadata
            if "Conditions" in result_group:
                del result_group["Conditions"]
            conditions_group = result_group.create_group("Conditions", track_order=True)
            for idx, cond in enumerate(self.conditions):
                cond_group = conditions_group.create_group(f"condition_{idx}", track_order=True)
                cond_group.attrs["name"] = cond.name
                cond_group.attrs["type"] = "numerical"  # All conditions are numerical
                if cond.value is not None:
                    cond_group.attrs["value"] = cond.value
                for attr_name, attr_value in cond.attrs.items():
                    cond_group.attrs[attr_name] = str(attr_value)  # Store as string for compatibility
            
            if "Collections" in result_group:
                del result_group["Collections"]
            collections_group = result_group.create_group("Collections", track_order=True)
            
            for cond_tuple, coll_list in self.collections.items():
                # Create group name from condition tuple
                cond_parts = []
                for cond, val in zip(self.conditions, cond_tuple):
                    if 'unit' in cond.attrs:
                        cond_parts.append(f"{cond.name}={val}{cond.attrs['unit']}")
                    else:
                        cond_parts.append(f"{cond.name}={val}")
                cond_group_name = "_".join(cond_parts)
                cond_group = collections_group.create_group(cond_group_name, track_order=True)
                
                # Store condition values
                for idx, (cond, val) in enumerate(zip(self.conditions, cond_tuple)):
                    cond_group.attrs[f"condition_{idx}_name"] = cond.name
                    cond_group.attrs[f"condition_{idx}_value"] = str(val)  # Store as string for compatibility

                for idx, collection in enumerate(coll_list):
                    # Use collection name as group name (sanitized for HDF5)
                    safe_collection_name = re.sub(r'[^0-9a-zA-Z_]', '_', collection.name)
                    if safe_collection_name and safe_collection_name[0].isdigit():
                        safe_collection_name = f'_{safe_collection_name}'
                    
                    # Check if group already exists (for uniqueness, append index if needed)
                    collection_group_name = safe_collection_name
                    if collection_group_name in cond_group:
                        collection_group_name = f"{safe_collection_name}_{idx}"
                    
                    collection_group = cond_group.create_group(collection_group_name, track_order=True)
                    collection_group.attrs["name"] = collection.name
                    collection_group.attrs["color"] = collection.color or ""

                    if "ignored_spheroids" in collection_group:
                        del collection_group["ignored_spheroids"]
                    if collection.ignored_spheroids:
                        collection_group.create_dataset(
                            "ignored_spheroids",
                            data=[s.encode("utf-8") for s in collection.ignored_spheroids]
                        )

                    if "Series" in collection_group:
                        del collection_group["Series"]
                    series_group = collection_group.create_group("Series", track_order=True)

                    for series in collection.get_spheroids():
                        series_key = f"{series_group.name}/{series.name}"
                        if series.name in series_group:
                            series_data_group = series_group[series.name]
                        else:
                            series_data_group = series_group.create_group(series.name, track_order=True)
                        
                        series_data_group.attrs["name"] = series.name
                        
                        if "ImageSeries" in series_data_group:
                            del series_data_group["ImageSeries"]
                        image_series_group = series_data_group.create_group("ImageSeries", track_order=True)
                        
                        for timepoint, spheroid_image in series.spheroid_image_dict.items():
                            timepoint_str = timepoint.isoformat() if isinstance(timepoint, datetime) else str(timepoint)
                            if timepoint_str in image_series_group:
                                timepoint_group = image_series_group[timepoint_str]
                            else:
                                timepoint_group = image_series_group.create_group(timepoint_str, track_order=True)
                            
                            if "image_paths" in timepoint_group:
                                del timepoint_group["image_paths"]
                            image_paths_group = timepoint_group.create_group("image_paths")
                            for channel, path in spheroid_image.image_path_dict.items():
                                if path:
                                    image_paths_group.create_dataset(
                                        channel,
                                        data=str(path.absolute()).encode("utf-8")
                                    )
                            
                            if spheroid_image.image_size:
                                timepoint_group.attrs["image_size_x_y"] = spheroid_image.image_size
                            
                            # Store as int for PyTables compatibility
                            timepoint_group.attrs["touches_border"] = 1 if (spheroid_image.contour_touches_border or False) else 0
                            
                            if spheroid_image.contour is not None:
                                if "contour" in timepoint_group:
                                    del timepoint_group["contour"]
                                timepoint_group.create_dataset("contour", data=spheroid_image.contour)
                            
                            # Store metrics if available (optional - for caching calculated metrics)
                            # Structure: metrics/metric_name as dataset (float | list)
                            if hasattr(spheroid_image, '_cached_metrics') and spheroid_image._cached_metrics:
                                if "metrics" in timepoint_group:
                                    del timepoint_group["metrics"]
                                metrics_group = timepoint_group.create_group("metrics", track_order=True)
                                for metric_name, metric_value in spheroid_image._cached_metrics.items():
                                    if metric_value is not None:
                                        try:
                                            if isinstance(metric_value, (list, tuple, np.ndarray)):
                                                metrics_group.create_dataset(metric_name, data=np.array(metric_value))
                                            else:
                                                metrics_group.create_dataset(metric_name, data=float(metric_value))
                                        except Exception:
                                            pass  # Skip if metric cannot be serialized
                        
                        if series.time_periods:
                            if "time_periods" in series_data_group:
                                del series_data_group["time_periods"]
                            periods_group = series_data_group.create_group("time_periods")
                            for period_name, period in series.time_periods.items():
                                period_group = periods_group.create_group(period_name)
                                period_group.attrs["start_time"] = period.start_time.isoformat()
                                period_group.attrs["end_time"] = period.end_time.isoformat()
                                period_group.attrs["created_at"] = period.created_at.isoformat()
                                period_group.attrs["modified_at"] = period.modified_at.isoformat()
                                if period.description:
                                    period_group.attrs["description"] = period.description
                        
                        if series.analysis_metrics:
                            if "analysis_metrics" in series_data_group:
                                del series_data_group["analysis_metrics"]
                            metrics_group = series_data_group.create_group("analysis_metrics")
                            for metric_name, periods_dict in series.analysis_metrics.items():
                                metric_group = metrics_group.create_group(metric_name)
                                for period_key, metrics_dict in periods_dict.items():
                                    period_group = metric_group.create_group(period_key)
                                    for key, value in metrics_dict.items():
                                        period_group.attrs[key] = value
            
            # Save ReplicateInfo if replicates exist (optional live-cell imaging support)
            # Note: ImageSeries are NOT duplicated here - they are stored in Collections/
            # This section only stores metadata and platemap, with links to collections
            if hasattr(self, 'replicates_list') and self.replicates_list:
                if "ReplicateInfo" in result_group:
                    del result_group["ReplicateInfo"]
                replicate_info_group = result_group.create_group("ReplicateInfo", track_order=True)
                replicate_info_group.attrs["has_livecell_structure"] = 1  # Store as int for PyTables compatibility
                
                # Save Replicates (metadata and ImageSeries)
                replicates_group = replicate_info_group.create_group("Replicates", track_order=True)
                for replicate_index, replicate in enumerate(self.replicates_list):
                    replicate_key = f"{replicate_index}-{replicate.name}"
                    replicate_group = replicates_group.create_group(replicate_key, track_order=True)
                    replicate_group.attrs["name"] = replicate.name
                    replicate_group.attrs["layout"] = replicate.layout
                    replicate_group.attrs["created_at"] = replicate._created_at.isoformat()
                    replicate_group.attrs["modified_at"] = datetime.now().isoformat()
                    replicate_group.attrs["data_has_been_loaded"] = 1 if replicate.data_has_been_loaded else 0  # Store as int for PyTables compatibility
                    
                    # Save time_points_array and relative_time_array if available
                    if replicate.time_points_array:
                        if "time_points_array" in replicate_group:
                            del replicate_group["time_points_array"]
                        replicate_group.create_dataset(
                            "time_points_array",
                            data=[tp.encode("utf-8") if isinstance(tp, str) else str(tp).encode("utf-8") for tp in replicate.time_points_array]
                        )
                    
                    if replicate.relative_time_array:
                        if "relative_time_array" in replicate_group:
                            del replicate_group["relative_time_array"]
                        replicate_group.create_dataset(
                            "relative_time_array",
                            data=replicate.relative_time_array
                        )
                    
                    # Save platemap
                    replicate.platemap._save_all_to_hdf5()
                    
                    # Store links to collections that belong to this replicate
                    # ImageSeries are stored in Collections/, not duplicated here
                    # This section stores metadata about which Collections belong to this replicate
                    # and how they relate to the platemap conditions
                    if "CollectionLinks" in replicate_group:
                        del replicate_group["CollectionLinks"]
                    if replicate.data_has_been_loaded:
                        collection_links_group = replicate_group.create_group("CollectionLinks", track_order=True)
                        replicate_collections = replicate.get_collections()
                        
                        for platemap_condition_tuple, collection in replicate_collections.items():
                            # Convert platemap condition to result condition
                            result_cond_tuple = self._convert_platemap_condition_to_result_condition(platemap_condition_tuple)
                            
                            # Find this collection in result.collections
                            if result_cond_tuple in self.collections and collection in self.collections[result_cond_tuple]:
                                # Find the index of this collection in the condition's list
                                coll_idx = self.collections[result_cond_tuple].index(collection)
                                
                                # Create condition group name (must match format used in Collections section)
                                cond_parts = []
                                for cond, val in zip(self.conditions, result_cond_tuple):
                                    if 'unit' in cond.attrs:
                                        cond_parts.append(f"{cond.name}={val}{cond.attrs['unit']}")
                                    else:
                                        cond_parts.append(f"{cond.name}={val}")
                                cond_group_name = "_".join(cond_parts)
                                
                                # Create link: use collection name as unique identifier
                                # Format: condition_tuple_collection_name (sanitized for HDF5)
                                safe_collection_name = re.sub(r'[^0-9a-zA-Z_]', '_', collection.name)
                                if safe_collection_name and safe_collection_name[0].isdigit():
                                    safe_collection_name = f'_{safe_collection_name}'
                                
                                # Check if collection group name might have index suffix (for uniqueness)
                                # We need to find the actual group name used
                                link_key_base = f"{cond_group_name}_{safe_collection_name}"
                                link_key = link_key_base
                                link_counter = 0
                                while link_key in collection_links_group:
                                    link_key = f"{link_key_base}_{link_counter}"
                                    link_counter += 1
                                
                                link_group = collection_links_group.create_group(link_key, track_order=True)
                                
                                # Store platemap condition info (for reference)
                                if isinstance(platemap_condition_tuple, tuple) and len(platemap_condition_tuple) > 0:
                                    platemap_cond_item = platemap_condition_tuple[0]
                                    if isinstance(platemap_cond_item, tuple) and len(platemap_cond_item) == 2:
                                        link_group.attrs["platemap_condition_name"] = str(platemap_cond_item[0])
                                        link_group.attrs["platemap_condition_value"] = str(platemap_cond_item[1])
                                
                                # Store result condition info
                                link_group.attrs["result_condition_tuple"] = str(result_cond_tuple)
                                link_group.attrs["collection_index"] = coll_idx
                                link_group.attrs["collection_name"] = collection.name  # Primary identifier for lookup
                                
                                # Build hdf5_path - need to find actual collection group name (might have index suffix)
                                # For now, use the base name; if there are duplicates, the first one will be used
                                # The collection name in attributes is the authoritative source
                                link_group.attrs["hdf5_path"] = f"Collections/{cond_group_name}/{safe_collection_name}"
                                
                                # Store which wells belong to this collection (from platemap)
                                replicate_wells = replicate.platemap.replicates()
                                if platemap_condition_tuple in replicate_wells:
                                    wells_list = replicate_wells[platemap_condition_tuple]
                                    if wells_list:
                                        link_group.create_dataset(
                                            "wells",
                                            data=[w.encode("utf-8") if isinstance(w, str) else str(w).encode("utf-8") for w in wells_list]
                                        )

    @classmethod
    def from_file(
        cls,
        name: str,
        result_index: int,
        hdf5_path: Path | str,
    ) -> "Result":
        """Load a result from HDF5 file.
        
        Args:
            name: Name of the result.
            result_index: Index of the result in the experiment.
            hdf5_path: Path to the HDF5 file.
            
        Returns:
            Result instance loaded from HDF5.
        """
        hdf5_path = Path(hdf5_path)
        result_key = f"Result/{result_index}-{name}"
        
        result = object.__new__(cls)
        
        with h5py.File(hdf5_path, "r") as hdf_file:
            if result_key not in hdf_file:
                raise KeyError(f"Result '{name}' not found in HDF5 file")
            
            result_group = hdf_file[result_key]
            
            # Load metadata
            # Handle name attribute - might be bytes or string
            name_attr = result_group.attrs.get("name")
            if name_attr is None:
                # Fallback: use the name parameter passed to from_file
                result.name = name
            else:
                if isinstance(name_attr, bytes):
                    result.name = name_attr.decode("utf-8")
                else:
                    result.name = str(name_attr)
            
            # Handle description - might be bytes or string
            desc_attr = result_group.attrs.get("description")
            if desc_attr is None:
                result.description = None
            else:
                if isinstance(desc_attr, bytes):
                    result.description = desc_attr.decode("utf-8") if desc_attr else None
                else:
                    result.description = str(desc_attr) if desc_attr else None
            
            # Load timestamps
            created_attr = result_group.attrs.get("created_at")
            if created_attr:
                if isinstance(created_attr, bytes):
                    created_attr = created_attr.decode("utf-8")
                result.created_at = datetime.fromisoformat(created_attr)
            else:
                result.created_at = datetime.now()
            
            modified_attr = result_group.attrs.get("modified_at")
            if modified_attr:
                if isinstance(modified_attr, bytes):
                    modified_attr = modified_attr.decode("utf-8")
                result.modified_at = datetime.fromisoformat(modified_attr)
            else:
                result.modified_at = datetime.now()
            result.collections = {}
            result._metric_cache = {}
            result.replicates_list = []  # Initialize optional replicates list
            result._result_index = result_index  # Store index from loading
            
            # Load conditions FIRST (needed for collection loading)
            result.conditions = []
            if "Conditions" in result_group:
                conditions_group = result_group["Conditions"]
                for cond_idx in sorted(conditions_group.keys(), key=lambda x: int(x.split("_")[1])):
                    cond_group = conditions_group[cond_idx]
                    cond_attrs = {}
                    for attr_name in cond_group.attrs.keys():
                        if attr_name not in ("name", "type", "value"):
                            cond_attrs[attr_name] = cond_group.attrs[attr_name]
                    
                    # Backward compatibility: check for old type attribute
                    cond_type = cond_group.attrs.get("type", "numerical")
                    if cond_type != "numerical":
                        raise ValueError("Categorical conditions are no longer supported.")
                    
                    condition = Condition(
                        name=cond_group.attrs["name"],
                        value=cond_group.attrs.get("value", None),
                        attrs=cond_attrs
                    )
                    result.conditions.append(condition)
            else:
                # Backward compatibility: try to infer from collections
                if "Collections" in result_group:
                    collections_group = result_group["Collections"]
                    first_cond_group_name = next(iter(collections_group.keys()), None)
                    if first_cond_group_name and "=" in first_cond_group_name:
                        # Old format: single condition
                        cond_name = first_cond_group_name.split("=", 1)[0]
                        condition = Condition(name=cond_name)
                        result.conditions.append(condition)
            
            result.num_conditions = len(result.conditions)
            
            # Load collections grouped by condition
            if "Collections" in result_group:
                collections_group = result_group["Collections"]
                for cond_group_name in collections_group.keys():
                    cond_group = collections_group[cond_group_name]
                    
                    # Extract condition values from attributes or group name
                    cond_values = []
                    for idx in range(result.num_conditions):
                        cond_name_attr = f"condition_{idx}_name"
                        cond_value_attr = f"condition_{idx}_value"
                        if cond_name_attr in cond_group.attrs and cond_value_attr in cond_group.attrs:
                            val_str = cond_group.attrs[cond_value_attr]
                            # All conditions are numerical now
                            try:
                                cond_values.append(float(val_str))
                            except (TypeError, ValueError):
                                cond_values.append(val_str)
                        else:
                            # Fallback: try to parse from group name (old format)
                            if "=" in cond_group_name:
                                parts = cond_group_name.split("=", 1)
                                if idx == 0:
                                    try:
                                        cond_values.append(float(parts[1]))
                                    except (TypeError, ValueError):
                                        cond_values.append(parts[1])
                                else:
                                    cond_values.append(None)
                            else:
                                cond_values.append(None)
                    
                    cond_tuple = tuple(cond_values)
                    result.collections[cond_tuple] = []

                    for collection_group_name in cond_group.keys():
                        collection_group = cond_group[collection_group_name]
                        collection_color = collection_group.attrs.get("color", None)
                        if collection_color == "":
                            collection_color = None

                        ignored_spheroids = set()
                        if "ignored_spheroids" in collection_group:
                            ignored_data = collection_group["ignored_spheroids"]
                            if ignored_data.shape != ():
                                ignored_spheroids = {s.decode("utf-8") for s in ignored_data[:]}

                        series_list = []
                        if "Series" in collection_group:
                            series_group = collection_group["Series"]
                            for series_name in series_group.keys():
                                series_data_group = series_group[series_name]
                                
                                series = SpheroidSeries(series_name)
                                
                                if "ImageSeries" in series_data_group:
                                    image_series_group = series_data_group["ImageSeries"]
                                    for timepoint_str in image_series_group.keys():
                                        timepoint_group = image_series_group[timepoint_str]
                                        # Keep timepoint as string (don't convert to datetime)
                                        timepoint = timepoint_str

                                        image_paths = {}
                                        if "image_paths" in timepoint_group:
                                            image_paths_group = timepoint_group["image_paths"]
                                            for channel in image_paths_group.keys():
                                                path_data = image_paths_group[channel]
                                                try:
                                                    # Handle both scalar and array datasets
                                                    if path_data.shape == ():
                                                        path_str = path_data[()].decode("utf-8")
                                                    else:
                                                        # Array dataset - take first element if array, otherwise decode directly
                                                        path_array = path_data[:]
                                                        if len(path_array) > 0:
                                                            path_str = path_array[0].decode("utf-8") if isinstance(path_array[0], bytes) else str(path_array[0])
                                                        else:
                                                            continue
                                                    # Only add if path is not empty
                                                    if path_str and path_str.strip():
                                                        # Resolve relative paths to absolute paths
                                                        path_obj = Path(path_str)
                                                        if not path_obj.is_absolute():
                                                            # Try to resolve relative path
                                                            path_obj = path_obj.resolve()
                                                        image_paths[channel] = path_obj
                                                except (AttributeError, UnicodeDecodeError, IndexError) as e:
                                                    # Skip invalid paths
                                                    import logging
                                                    logger = logging.getLogger("SpheroidPy.experiment.result")
                                                    logger.warning(f"Could not decode path for channel '{channel}' at timepoint '{timepoint_str}': {e}")
                                                    continue

                                        image_size = None
                                        if "image_size_x_y" in timepoint_group.attrs:
                                            image_size = tuple(timepoint_group.attrs["image_size_x_y"])

                                        # Only create SpheroidImage if brightfield path exists and is valid
                                        if "brightfield" not in image_paths or image_paths["brightfield"] is None:
                                            continue
                                        
                                        # Verify brightfield path exists (or at least is a valid Path object)
                                        brightfield_path = image_paths["brightfield"]
                                        if not isinstance(brightfield_path, Path):
                                            continue

                                        # Reconstruct hdf5_key from the HDF5 path (don't store it as attribute)
                                        # timepoint_group.name gives the full path from root, e.g. "/Result/0-Name/Collections/..."
                                        hdf5_key = timepoint_group.name if timepoint_group.name.startswith('/') else f"/{timepoint_group.name}"
                                        
                                        spheroid_image = SpheroidImage(
                                            brightfield=brightfield_path,
                                            fluorescence_green=image_paths.get("fluorescence_green"),
                                            fluorescence_red=image_paths.get("fluorescence_red"),
                                            fluorescence_blue=image_paths.get("fluorescence_blue"),
                                            image_size=image_size,
                                            hdf5_path=str(hdf5_path),
                                            hdf5_key=hdf5_key,
                                        )

                                        if "contour" in timepoint_group:
                                            contour_data = timepoint_group["contour"]
                                            if contour_data.shape != ():
                                                spheroid_image.contour = contour_data[:]
                                                # Convert int to bool for compatibility (stored as 0/1 for PyTables)
                                                touches_border_val = timepoint_group.attrs.get("touches_border", 0)
                                                if isinstance(touches_border_val, (int, np.integer)):
                                                    spheroid_image.contour_touches_border = bool(touches_border_val)
                                                else:
                                                    spheroid_image.contour_touches_border = bool(touches_border_val) if touches_border_val else False
                                        
                                        # Load cached metrics if available (optional)
                                        if "metrics" in timepoint_group:
                                            metrics_group = timepoint_group["metrics"]
                                            if not hasattr(spheroid_image, '_cached_metrics'):
                                                spheroid_image._cached_metrics = {}
                                            for metric_name in metrics_group.keys():
                                                metric_data = metrics_group[metric_name]
                                                if metric_data.shape == ():
                                                    spheroid_image._cached_metrics[metric_name] = float(metric_data[()])
                                                else:
                                                    spheroid_image._cached_metrics[metric_name] = metric_data[:].tolist()

                                        series.add_spheroid_image(spheroid_image, timepoint)

                                if "time_periods" in series_data_group:
                                    periods_group = series_data_group["time_periods"]
                                    from SpheroidPy.utils.time_period import TimePeriod
                                    for period_name in periods_group.keys():
                                        period_group = periods_group[period_name]
                                        period = TimePeriod(
                                            name=period_name,
                                            start_time=datetime.fromisoformat(period_group.attrs["start_time"]),
                                            end_time=datetime.fromisoformat(period_group.attrs["end_time"]),
                                            description=period_group.attrs.get("description", None),
                                        )
                                        period.created_at = datetime.fromisoformat(period_group.attrs["created_at"])
                                        period.modified_at = datetime.fromisoformat(period_group.attrs["modified_at"])
                                        series.time_periods[period_name] = period

                                if "analysis_metrics" in series_data_group:
                                    metrics_group = series_data_group["analysis_metrics"]
                                    for metric_name in metrics_group.keys():
                                        metric_group = metrics_group[metric_name]
                                        series.analysis_metrics[metric_name] = {}
                                        for period_key in metric_group.keys():
                                            period_group = metric_group[period_key]
                                            metrics_dict = {}
                                            for key in period_group.attrs.keys():
                                                metrics_dict[key] = period_group.attrs[key]
                                            series.analysis_metrics[metric_name][period_key] = metrics_dict

                                series_list.append(series)

                        collection = SpheroidCollection(
                            name=collection_group.attrs.get("name", cond_group_name),
                            spheroid_list=series_list,
                            hdf5_path=str(hdf5_path),
                            color=collection_color,
                        )
                        collection.ignored_spheroids = ignored_spheroids
                        result.collections[cond_tuple].append(collection)
            
            # Load ReplicateInfo if it exists (optional live-cell imaging support)
            if "ReplicateInfo" in result_group:
                replicate_info_group = result_group["ReplicateInfo"]
                # Convert int to bool for compatibility (stored as 0/1 for PyTables)
                has_livecell = replicate_info_group.attrs.get("has_livecell_structure", 0)
                if isinstance(has_livecell, (int, np.integer)):
                    has_livecell = bool(has_livecell)
                if has_livecell:
                    if "Replicates" in replicate_info_group:
                        replicates_group = replicate_info_group["Replicates"]
                        for replicate_key in replicates_group.keys():
                            try:
                                # Parse replicate key: "index-name"
                                index_str, replicate_name = replicate_key.split("-", 1)
                                replicate_index = int(index_str)
                            except (ValueError, IndexError):
                                replicate_index = 0
                                replicate_name = replicate_key
                            
                            replicate_group = replicates_group[replicate_key]
                            
                            # Load replicate using from_file
                            from SpheroidPy.experiment.livecell_replicate import LiveCellReplicate
                            replicate = LiveCellReplicate.from_file(
                                name=replicate_name,
                                replicate_index=replicate_index,
                                result=result,
                                replicate_group=replicate_group,
                                hdf5_path=str(hdf5_path)
                            )
                            result.replicates_list.append(replicate)
                            
                            # Load time_points_array and relative_time_array if available
                            if "time_points_array" in replicate_group:
                                time_points_data = replicate_group["time_points_array"]
                                if time_points_data.shape != ():
                                    replicate.time_points_array = [tp.decode("utf-8") for tp in time_points_data[:]]
                                else:
                                    replicate.time_points_array = []
                            
                            if "relative_time_array" in replicate_group:
                                relative_time_data = replicate_group["relative_time_array"]
                                if relative_time_data.shape != ():
                                    replicate.relative_time_array = relative_time_data[:].tolist()
                                else:
                                    replicate.relative_time_array = []
                            
                            # Populate spheroid_dict from Collections (ImageSeries are stored in Collections/, not in ReplicateInfo)
                            # This is the primary way to load replicate data
                            if replicate.data_has_been_loaded:
                                # First, try to link collections to this replicate using CollectionLinks
                                if "CollectionLinks" in replicate_group:
                                    collection_links_group = replicate_group["CollectionLinks"]
                                    for link_key in collection_links_group.keys():
                                        link_group = collection_links_group[link_key]
                                        collection_name = link_group.attrs.get("collection_name")
                                        
                                        # Find the collection by name
                                        for cond_tuple, coll_list in result.collections.items():
                                            for collection in coll_list:
                                                if collection.name == collection_name:
                                                    # Link collection to replicate
                                                    if not hasattr(collection, '_replicate_source'):
                                                        collection._replicate_source = []
                                                    if replicate not in collection._replicate_source:
                                                        collection._replicate_source.append(replicate)
                                                    break  # Found collection, move to next link
                                
                                # Populate spheroid_dict from Collections that belong to this replicate
                                # First try collections with explicit _replicate_source link
                                for cond_tuple, coll_list in result.collections.items():
                                    for collection in coll_list:
                                        # Check if this collection belongs to this replicate
                                        if hasattr(collection, '_replicate_source') and replicate in collection._replicate_source:
                                            for series in collection.get_spheroids():
                                                # Extract well name from series name (format: "Spheroid-{well}")
                                                if series.name.startswith("Spheroid-"):
                                                    well = series.name.replace("Spheroid-", "")
                                                    if well not in replicate.spheroid_dict:
                                                        replicate.spheroid_dict[well] = series
                                
                                # If still no data, try to match collections by platemap conditions
                                if not replicate.spheroid_dict:
                                    # Get platemap conditions for this replicate
                                    replicate_wells = replicate.platemap.replicates()
                                    platemap_conditions = set(replicate_wells.keys())
                                    
                                    # Try to match collections by condition values
                                    for cond_tuple, coll_list in result.collections.items():
                                        for collection in coll_list:
                                            # Check if collection name matches a platemap condition
                                            for platemap_cond in platemap_conditions:
                                                # Convert platemap condition to result condition
                                                result_cond = result._convert_platemap_condition_to_result_condition(platemap_cond)
                                                if result_cond == cond_tuple:
                                                    # Link collection to replicate
                                                    if not hasattr(collection, '_replicate_source'):
                                                        collection._replicate_source = []
                                                    if replicate not in collection._replicate_source:
                                                        collection._replicate_source.append(replicate)
                                                    
                                                    # Populate spheroid_dict
                                                    for series in collection.get_spheroids():
                                                        if series.name.startswith("Spheroid-"):
                                                            well = series.name.replace("Spheroid-", "")
                                                            if well not in replicate.spheroid_dict:
                                                                replicate.spheroid_dict[well] = series
                                
                                # Final fallback: populate from all collections (backward compatibility)
                                if not replicate.spheroid_dict:
                                    # Try to populate from all Collections
                                    for cond_tuple, coll_list in result.collections.items():
                                        for collection in coll_list:
                                            for series in collection.get_spheroids():
                                                # Extract well name from series name (format: "Spheroid-{well}")
                                                if series.name.startswith("Spheroid-"):
                                                    well = series.name.replace("Spheroid-", "")
                                                    if well not in replicate.spheroid_dict:
                                                        replicate.spheroid_dict[well] = series
                                
                                # Reconstruct time_points_array and relative_time_array from spheroid_dict if not loaded
                                if replicate.spheroid_dict and not replicate.time_points_array:
                                    all_timepoints = set()
                                    for series in replicate.spheroid_dict.values():
                                        all_timepoints.update(series.spheroid_image_dict.keys())
                                    
                                    if all_timepoints:
                                        sorted_timepoints = sorted(all_timepoints)
                                        replicate.time_points_array = [tp.isoformat() if isinstance(tp, datetime) else str(tp) for tp in sorted_timepoints]
                                        
                                        # Try to reconstruct relative_time_array from first series
                                        if not replicate.relative_time_array:
                                            first_series = next(iter(replicate.spheroid_dict.values()))
                                            if hasattr(first_series, '_relative_times') and first_series._relative_times:
                                                replicate.relative_time_array = [
                                                    first_series._relative_times.get(tp, 0.0) 
                                                    for tp in sorted_timepoints
                                                ]
                                            else:
                                                # Fallback: calculate from timepoints
                                                if isinstance(sorted_timepoints[0], datetime):
                                                    base_time = sorted_timepoints[0]
                                                    replicate.relative_time_array = [
                                                        (tp - base_time).total_seconds() / 3600 
                                                        if isinstance(tp, datetime) else 0.0
                                                        for tp in sorted_timepoints
                                                    ]
                            
        # After all replicates are loaded, first link collections using CollectionLinks
        # Then update collections and conditions from platemap
        if result.replicates_list:
            # First pass: Link collections to replicates using CollectionLinks
            with h5py.File(hdf5_path, "r") as hdf_file:
                result_group = hdf_file[result_key]
                if "ReplicateInfo" in result_group:
                    replicate_info_group = result_group["ReplicateInfo"]
                    if replicate_info_group.attrs.get("has_livecell_structure", False):
                        if "Replicates" in replicate_info_group:
                            replicates_group = replicate_info_group["Replicates"]
                            for replicate_key in replicates_group.keys():
                                replicate_group = replicates_group[replicate_key]
                                
                                # Find corresponding replicate object
                                replicate = None
                                for r in result.replicates_list:
                                    if r.name == replicate_group.attrs.get("name"):
                                        replicate = r
                                        break
                                
                                if replicate and replicate.data_has_been_loaded:
                                    # Link collections using CollectionLinks
                                    if "CollectionLinks" in replicate_group:
                                        collection_links_group = replicate_group["CollectionLinks"]
                                        for link_key in collection_links_group.keys():
                                            link_group = collection_links_group[link_key]
                                            collection_name = link_group.attrs.get("collection_name")
                                            
                                            # Find the collection by name
                                            for cond_tuple, coll_list in result.collections.items():
                                                for collection in coll_list:
                                                    if collection.name == collection_name:
                                                        # Link collection to replicate
                                                        if not hasattr(collection, '_replicate_source'):
                                                            collection._replicate_source = []
                                                        if replicate not in collection._replicate_source:
                                                            collection._replicate_source.append(replicate)
            
            # Second pass: Create conditions from platemap first (before updating collections)
            # Extract conditions from all replicates' platemaps
            condition_data = {}  # {condition_name: set of values}
            
            for replicate in result.replicates_list:
                if not replicate.data_has_been_loaded:
                    continue
                try:
                    replicate_wells = replicate.platemap.replicates()
                    for condition_tuple, wells in replicate_wells.items():
                        # Condition tuple format: ((name, value),) or similar
                        if isinstance(condition_tuple, tuple) and len(condition_tuple) > 0:
                            cond_item = condition_tuple[0]
                            if isinstance(cond_item, tuple) and len(cond_item) == 2:
                                cond_name, cond_value = cond_item
                                if cond_name not in condition_data:
                                    condition_data[cond_name] = set()
                                # Only add numeric values
                                if isinstance(cond_value, (int, float)):
                                    condition_data[cond_name].add(cond_value)
                except Exception as e:
                    import logging
                    logger = logging.getLogger("SpheroidPy.experiment.result")
                    logger.warning(f"Could not extract conditions from replicate '{replicate.name}': {e}")
            
            # Create/update conditions from platemap data
            if condition_data:
                for cond_name, values in condition_data.items():
                    # Check if condition already exists
                    existing_cond = next((c for c in result.conditions if c.name == cond_name), None)
                    if not existing_cond:
                        # Create new condition
                        result.conditions.append(Condition(name=cond_name, attrs={}))
                        result.num_conditions = len(result.conditions)
            
            # Fourth pass: Populate spheroid_dict for each replicate from Collections
            # (Collections are already loaded, we just need to link them to replicates)
            # Note: We do NOT call _update_collections_from_replicates() here because
            # Collections are already loaded from HDF5 and should be preserved
            for replicate in result.replicates_list:
                if not replicate.data_has_been_loaded:
                    continue
                
                # Populate spheroid_dict from Collections that belong to this replicate
                # First try collections that are explicitly linked via CollectionLinks
                for cond_tuple, coll_list in result.collections.items():
                    for collection in coll_list:
                        # Check if this collection belongs to this replicate
                        if hasattr(collection, '_replicate_source') and replicate in collection._replicate_source:
                            for series in collection.get_spheroids():
                                # Extract well name from series name (format: "Spheroid-{well}")
                                if series.name.startswith("Spheroid-"):
                                    well = series.name.replace("Spheroid-", "")
                                    if well not in replicate.spheroid_dict:
                                        replicate.spheroid_dict[well] = series
                
                # Fallback: If spheroid_dict is still empty, try to match by collection name
                # This handles cases where CollectionLinks might be missing
                # Collection names now have format: "{replicate_name}_{cond_name}={cond_value}"
                if not replicate.spheroid_dict:
                    # Get expected collection names from platemap
                    replicate_wells = replicate.platemap.replicates()
                    for platemap_condition_tuple, wells in replicate_wells.items():
                        # Create expected collection name (with replicate name prefix)
                        if isinstance(platemap_condition_tuple, tuple) and len(platemap_condition_tuple) > 0:
                            cond_item = platemap_condition_tuple[0]
                            if isinstance(cond_item, tuple) and len(cond_item) == 2:
                                expected_collection_name = f"{replicate.name}_{cond_item[0]}={cond_item[1]}"
                                
                                # Find collection with this name
                                for cond_tuple, coll_list in result.collections.items():
                                    for collection in coll_list:
                                        if collection.name == expected_collection_name:
                                            # Link collection to replicate
                                            if not hasattr(collection, '_replicate_source'):
                                                collection._replicate_source = []
                                            if replicate not in collection._replicate_source:
                                                collection._replicate_source.append(replicate)
                                            
                                            # Populate spheroid_dict
                                            for series in collection.get_spheroids():
                                                if series.name.startswith("Spheroid-"):
                                                    well = series.name.replace("Spheroid-", "")
                                                    if well in wells and well not in replicate.spheroid_dict:
                                                        replicate.spheroid_dict[well] = series
                                            break
                
                # Also set _result reference for all series in this replicate
                for series in replicate.spheroid_dict.values():
                    series._result = result
                
                # Reconstruct time_points_array and relative_time_array from spheroid_dict if not loaded
                if replicate.spheroid_dict and not replicate.time_points_array:
                    all_timepoints = set()
                    for series in replicate.spheroid_dict.values():
                        all_timepoints.update(series.spheroid_image_dict.keys())
                    
                    if all_timepoints:
                        sorted_timepoints = sorted(all_timepoints)
                        replicate.time_points_array = [tp.isoformat() if isinstance(tp, datetime) else str(tp) for tp in sorted_timepoints]
                        
                        # Try to reconstruct relative_time_array from first series
                        if not replicate.relative_time_array:
                            first_series = next(iter(replicate.spheroid_dict.values()))
                            if hasattr(first_series, '_relative_times') and first_series._relative_times:
                                replicate.relative_time_array = [
                                    first_series._relative_times.get(tp, 0.0) 
                                    for tp in sorted_timepoints
                                ]
                            else:
                                # Fallback: calculate from timepoints
                                if isinstance(sorted_timepoints[0], datetime):
                                    base_time = sorted_timepoints[0]
                                    replicate.relative_time_array = [
                                        (tp - base_time).total_seconds() / 3600 
                                        if isinstance(tp, datetime) else 0.0
                                        for tp in sorted_timepoints
                                    ]

        result._assign_collection_colors()
        return result

    # ------------------------------------------------------------------ #
    # Interactive visualization
    # ------------------------------------------------------------------ #
    def show(self) -> None:
        """Interactive viewer for all collections and series in this result."""
        if not self.collections:
            print("No collections in result.")
            return

        # Helper to get active series of a collection (respects ignored list)
        def _active_series(coll: SpheroidCollection) -> list[SpheroidSeries]:
            return coll.get_spheroids()

        # Build condition -> replicate mapping
        cond_options = []
        for cond_tuple, coll_list in self.collections.items():
            label = self._condition_label(cond_tuple)
            cond_options.append((label, cond_tuple, coll_list))
        if not cond_options:
            print("No collections in result.")
            return
        
        cond_label_map = {label: cond_tuple for label, cond_tuple, _ in cond_options}
        cond_to_reps = {cond_tuple: coll_list for _, cond_tuple, coll_list in cond_options}

        condition_dropdown = widgets.Dropdown(
            options=[label for label, _, _ in cond_options],
            value=cond_options[0][0],
            description="Condition:",
            layout=widgets.Layout(width="220px"),
        )

        def _rep_options(cond_tuple: tuple) -> list[str]:
            return [str(i + 1) for i in range(len(cond_to_reps[cond_tuple]))]

        replicate_dropdown = widgets.Dropdown(
            options=_rep_options(cond_options[0][1]),
            value="1",
            description="Replicate:",
            layout=widgets.Layout(width="140px"),
        )

        def get_current_collection() -> SpheroidCollection:
            cond_tuple = cond_label_map[condition_dropdown.value]
            rep_idx = int(replicate_dropdown.value) - 1
            return cond_to_reps[cond_tuple][rep_idx]

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
            layout=widgets.Layout(width="210px"),
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
            layout=widgets.Layout(width="170px"),
        )
        show_contour = widgets.Checkbox(value=True, description="Show Contour", layout=widgets.Layout(width="220px"))

        # Placeholder sliders and labels (updated when series changes)
        time_slider = widgets.IntSlider(min=0, max=0, value=0, description="Time:", continuous_update=True, layout=widgets.Layout(width="250px"))
        time_label = widgets.Label()

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

        name_html = widgets.HTML(f"<h2 style='margin:0'>{self.name}</h2>")
        subtitle_html = widgets.HTML("<div style='color:#888; font-weight:400; font-size:1.1em; margin-bottom:2px;'>Result</div>")
        name_block = widgets.VBox([name_html, subtitle_html], layout=widgets.Layout(margin="0px", padding="0px"))

        # Build header row with dropdowns
        series_controls = widgets.HBox(
            [condition_dropdown, replicate_dropdown, series_dropdown, time_slider, time_label],
            layout=widgets.Layout(justify_content="center", align_items="center", gap="6px"),
        )
        header_row = widgets.HBox(
            [name_block, series_controls, widgets.HTML(logo_html)],
            layout=widgets.Layout(justify_content="space-between", align_items="center", width="100%"),
        )
        header_container = widgets.VBox(
            [header_row],
            layout=widgets.Layout(border="none", border_radius="10px", padding="0px", margin="0px", width="100%"),
        )

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
            layout=widgets.Layout(border="none", border_radius="10px", padding="0px", margin="0px", flex="1 1 0%", width="100%", overflow="hidden"),
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
                image_title.value = ""
                return

            timepoint = timepoints[idx]
            channel = channel_dropdown.value
            image_title.value = f"<div style='text-align: center;'><span style='font-size: 12px; font-weight: 600; color: #333;'>{timepoint} | {channel} | {series.name}</span></div>"
            
            sph_img = series.spheroid_image_dict[timepoint]
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
                        xs = contour[:, 0]
                        ys = contour[:, 1]
                        if xs[0] != xs[-1] or ys[0] != ys[-1]:
                            xs = np.r_[xs, xs[:1]]
                            ys = np.r_[ys, ys[:1]]
                        ax.plot(xs, ys, "w-", linewidth=2)
                    except Exception as e:  # pragma: no cover - UI best effort
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
                
                ax.axis("off")
                plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
                plt.show()

        def _channel_to_fluorescence(ch_desc):
            """Map channel dropdown value to fluorescence channel name for radial_profile."""
            if ch_desc == 'fluorescence_green':
                return 'green'
            if ch_desc == 'fluorescence_red':
                return 'red'
            if ch_desc == 'fluorescence_blue':
                return 'blue'
            return 'green'  # fallback for brightfield (profile needs fluorescence)

        def update_plot(*args) -> None:
            metric = metric_dropdown.value
            if metric == "radius":
                metric_label = "Radius"
            elif metric == "area":
                metric_label = "Area"
            else:
                metric_label = "Radial profile"
            series = get_current_series()
            over_time = " over time" if metric in ("radius", "area") else ""
            plot_title.value = f"<div style='text-align: center;'><span style='font-size: 14px; font-weight: 600; color: #333;'>{metric_label}{over_time} – {series.name}</span></div>"
            
            plot_output.clear_output(wait=True)
            with plot_output:
                series = get_current_series()
                timepoints = sorted(series.spheroid_image_dict.keys())
                if not timepoints:
                    print("No timepoints available.")
                    return
                
                try:
                    if metric == "profile":
                        # Normalized radial profile for current timepoint
                        idx = time_slider.value
                        if idx >= len(timepoints):
                            return
                        timepoint = timepoints[idx]
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
                                    # Set xlims to exactly 0 to max value
                                    if r_um is not None and len(r_um) > 0:
                                        ax.set_xlim(0, r_um.max())
                                    ax.grid(True, alpha=0.3)
                                    ax.legend()
                                    plt.tight_layout()
                                    plt.show()
                    else:
                        def _ensure_dt_local(val):
                            return val if isinstance(val, datetime) else datetime.strptime(val, "%Y-%m-%d %H:%M:%S")
                        base_t0 = _ensure_dt_local(timepoints[0])
                        times = [(_ensure_dt_local(tp) - base_t0).total_seconds() / 24 / 3600 for tp in timepoints]
                        
                        if metric == "area":
                            values = [series.spheroid_image_dict[tp].area for tp in timepoints]
                            ylabel = "Area [µm²]"
                        else:
                            values = [series.spheroid_image_dict[tp].radius for tp in timepoints]
                            ylabel = "Radius [µm]"
                        values = [v if v is not None else np.nan for v in values]
                        idx = time_slider.value
                        current_time = times[idx] if idx < len(times) else 0
                        current_val = values[idx] if idx < len(values) else None

                        fig, ax = plt.subplots(figsize=(6, 4.1))
                        ax.plot(times, values, color="grey")
                        if current_val is not None and not np.isnan(current_val):
                            ax.plot(
                                current_time,
                                current_val,
                                ".",
                                markersize=10,
                                label=f"{current_val:.1f} {'µm' if metric == 'radius' else 'µm²'} @ {current_time:.2f} d",
                                color="#e29266",
                            )
                        ax.set_xlabel("Time [d]")
                        ax.set_ylabel(ylabel)
                        ax.grid(True, alpha=0.3)
                        if current_val is not None and not np.isnan(current_val):
                            ax.legend()
                        plt.tight_layout()
                        plt.show()
                except Exception as e:
                    plt.figure(figsize=(6, 4.1))
                    plt.text(0.5, 0.5, f'No {metric} data available\n{str(e)}', ha='center', va='center')
                    plt.axis('off')
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
                # Save contour to HDF5
                if hasattr(sph_img, '_save_contour_to_hdf5'):
                    sph_img._save_contour_to_hdf5()
                update_image()
                update_plot()
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
                # Save contour to HDF5
                if hasattr(sph_img, '_save_contour_to_hdf5'):
                    sph_img._save_contour_to_hdf5()
                update_image()
                update_plot()
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
                border_radius="5px",
                width="100%",
            ),
        )

        def _refresh_series_for_current_collection():
            coll = get_current_collection()
            options = _series_options(coll)
            series_dropdown.options = options
            series_dropdown.value = options[0] if options else None
            _update_time_controls(get_current_series())
            update_image()
            update_plot()

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
        condition_dropdown.observe(on_condition_change, names="value")
        replicate_dropdown.observe(on_replicate_change, names="value")
        series_dropdown.observe(on_series_change, names="value")
        time_slider.observe(lambda change: (update_time_label(time_slider.value, get_current_series()), update_image(), update_plot()), names="value")
        channel_dropdown.observe(lambda change: update_image(), names="value")
        show_contour.observe(lambda change: update_image(), names="value")

        # Right column: plot + segmentation controls
        plot_panel = widgets.VBox(
            [plot_header_row, plot_output],
            layout=widgets.Layout(
                border="none",
                border_radius="10px",
                padding="0px",
                margin="0px",
                flex="1 1 0%",
                width="100%",
                align_items="flex-start",
                justify_content="flex-start",
                overflow="hidden",
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

        # Observe changes
        condition_dropdown.observe(on_condition_change, names="value")
        replicate_dropdown.observe(on_replicate_change, names="value")
        series_dropdown.observe(on_series_change, names="value")
        time_slider.observe(lambda change: (update_time_label(time_slider.value, get_current_series()), update_image(), update_plot()), names="value")
        channel_dropdown.observe(lambda change: update_image(), names="value")
        show_contour.observe(lambda change: update_image(), names="value")
        metric_dropdown.observe(lambda change: update_plot(), names="value")

        # Initialize controls and display
        display(outer_box)

        def _initial_render():
            _update_time_controls(get_current_series())
            update_time_label(0, get_current_series())
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

    # ------------------------------------------------------------------ #
    # Utility
    # ------------------------------------------------------------------ #
    def __repr__(self) -> str:  # pragma: no cover - debug helper
        collections = ", ".join(str(k) for k in self.collections.keys()) or "<empty>"
        cond_names = ", ".join(cond.name for cond in self.conditions) if self.conditions else "no conditions"
        return f"Result(name={self.name}, conditions=[{cond_names}], collections=[{collections}])"


