from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Union, List, Tuple

# Type alias for allowed exclusion types:
# - single timestamps (datetime)
# - time ranges (Tuple[start, end])
DatetimeOrRange = Union[datetime, Tuple[Optional[datetime], Optional[datetime]]]


@dataclass
class TimePeriod:
    """
    Represents a named time interval within a time series, optionally including
    exclusion zones for specific timestamps or sub-intervals that should not be
    considered part of the valid range.

    This class provides a semantic representation of meaningful analysis periods
    (e.g., "growth phase", "treatment phase", "stimulation window") in temporal
    datasets such as image sequences, sensor readings, or experimental data.

    In addition to the main inclusive time window, `TimePeriod` supports the
    definition of exclusion ranges—allowing users to mark specific timestamps
    or entire intervals as invalid or to be ignored during downstream analyses.

    Attributes
    ----------
    name : str
        Unique identifier for this period. Typically used as a key in a
        container dictionary (e.g., `self.time_periods[name]`).

    start_time : datetime | None
        Start of the period. If `None`, it defaults to the first available
        timestamp in the associated dataset.

    end_time : datetime | None
        End of the period. If `None`, it defaults to the last available
        timestamp in the associated dataset.

    description : str | None
        Optional textual description, used for documentation or context.

    exclude : list[DatetimeOrRange]
        A list of exclusion definitions. Each entry can be either a single
        timestamp (`datetime`) or a `(start, end)` tuple defining a time range
        to be excluded. `start` or `end` may be `None` to define open intervals,
        e.g., `(None, datetime(2025, 1, 5))`.

    Examples
    --------
    >>> tp = TimePeriod(
    ...     name="growth_phase",
    ...     start_time=datetime(2025, 1, 1),
    ...     end_time=datetime(2025, 2, 1),
    ...     exclude=[
    ...         datetime(2025, 1, 10),
    ...         (datetime(2025, 1, 15), datetime(2025, 1, 18))
    ...     ]
    ... )
    >>> tp.contains(datetime(2025, 1, 9))
    True
    >>> tp.contains(datetime(2025, 1, 10))
    False
    >>> tp.contains(datetime(2025, 1, 16))
    False
    """

    name: str
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    description: Optional[str] = None
    exclude: List[DatetimeOrRange] = field(default_factory=list)

    # -------------------------------------------------------------------------
    # Core functionality
    # -------------------------------------------------------------------------

    def contains(self, timestamp: datetime) -> bool:
        """
        Check whether a given timestamp lies within the active period and is
        not part of any defined exclusion zones.

        Parameters
        ----------
        timestamp : datetime
            The timestamp to check.

        Returns
        -------
        bool
            `True` if the timestamp lies within the main time window and
            is not excluded; otherwise `False`.

        Notes
        -----
        The check proceeds in two stages:
        1. Verify the timestamp lies between `start_time` and `end_time`.
        2. Check whether the timestamp falls into any exclusion definition.

        Exclusion checking supports:
        - exact matches (`timestamp == ex`)
        - closed intervals (`start <= timestamp <= end`)
        - open-ended intervals using `None` for one boundary
        """
        # --- Check main inclusive range ---
        if self.start_time and timestamp < self.start_time:
            return False
        if self.end_time and timestamp > self.end_time:
            return False

        # --- Check exclusions ---
        for ex in self.exclude:
            # single timestamp exclusion
            if isinstance(ex, datetime):
                if timestamp == ex:
                    return False
            # interval exclusion
            elif isinstance(ex, tuple):
                start, end = ex
                if (start is None or timestamp >= start) and (end is None or timestamp <= end):
                    return False

        return True

    # -------------------------------------------------------------------------
    # Exclusion management
    # -------------------------------------------------------------------------

    def add_exclusion(self, item: DatetimeOrRange) -> None:
        """
        Add an exclusion entry (either a timestamp or a time interval).

        Parameters
        ----------
        item : datetime | tuple[datetime, datetime]
            The exclusion to add. Can be a single timestamp or a (start, end)
            tuple defining a time range.

        Examples
        --------
        >>> tp.add_exclusion(datetime(2025, 1, 10))
        >>> tp.add_exclusion((datetime(2025, 1, 15), datetime(2025, 1, 18)))
        """
        self.exclude.append(item)

    def remove_exclusion(self, item: DatetimeOrRange) -> None:
        """
        Remove a previously defined exclusion entry, if it exists.

        Parameters
        ----------
        item : datetime | tuple[datetime, datetime]
            The exclusion to remove. Matching is performed by exact equality,
            not by overlap.

        Notes
        -----
        Overlapping ranges are not merged or automatically removed — this
        method removes only the exact entry provided.
        """
        self.exclude = [ex for ex in self.exclude if ex != item]

    def clear_exclusions(self) -> None:
        """
        Remove all currently defined exclusions.

        Examples
        --------
        >>> tp.clear_exclusions()
        """
        self.exclude.clear()

    # -------------------------------------------------------------------------
    # Convenience methods
    # -------------------------------------------------------------------------

    def __contains__(self, timestamp: datetime) -> bool:
        """
        Enables the idiomatic Python syntax:
            `if ts in time_period: ...`
        """
        return self.contains(timestamp)

    def __repr__(self) -> str:
        """
        Return a compact, human-readable string representation of this period.
        """
        excl_info = f", {len(self.exclude)} excludes" if self.exclude else ""
        return (
            f"TimePeriod(name='{self.name}', "
            f"start={self.start_time}, end={self.end_time}{excl_info})"
        )
