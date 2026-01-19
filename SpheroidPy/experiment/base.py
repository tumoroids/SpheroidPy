from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import h5py

if TYPE_CHECKING:  # pragma: no cover - typing helper only
    from .experiment import Experiment


class Base:
    """Lightweight base class for experiment graph elements.

    This class purposely mirrors the structure of the original ``experiment``
    module but keeps the implementation domain-agnostic so that specialised
    live-cell imaging classes can inherit from it later on. When no experiment
    context is supplied, the instance behaves as a standalone node that can
    still store analysis data in-memory or to a custom HDF5 file.
    """

    type_name: str = "Node"
    _standalone_counters: dict[str, int] = {}

    def __init__(
        self,
        name: str,
        experiment: Experiment | None = None,
        description: str | None = None,
        hdf5_path: Path | None = None,
    ) -> None:
        self.name = name
        self.experiment = experiment
        self.description = description
        self.created_at = datetime.now()
        self.modified_at = datetime.now()
        self._hdf5_override = Path(hdf5_path) if hdf5_path is not None else None

        if experiment is not None:
            self.index = experiment._register_node(self)
            self.hdf5_key = f"{self.type_name}/{self.index}-{self.name}"
        else:
            self.index = self._next_standalone_index()
            self.hdf5_key = f"{self.type_name}/standalone-{self.index}-{self.name}"

    # ------------------------------------------------------------------ #
    # Convenience properties
    # ------------------------------------------------------------------ #
    @property
    def hdf5_path(self) -> Path:
        if self.experiment is not None:
            return self.experiment.hdf5_path
        if self._hdf5_override is not None:
            return self._hdf5_override
        raise RuntimeError(
            f"{self.__class__.__name__} is not attached to an Experiment and no "
            "custom hdf5_path was provided."
        )

    # ------------------------------------------------------------------ #
    # Persistence helpers
    # ------------------------------------------------------------------ #
    def save_metadata(self) -> None:
        """Persist basic metadata for the node into the HDF5 file."""
        if not self._can_persist():
            return
        with h5py.File(self.hdf5_path, "a") as hdf_file:
            group = hdf_file.require_group(self.hdf5_key)
            group.attrs["name"] = self.name
            group.attrs["created_at"] = self.created_at.isoformat()
            group.attrs["modified_at"] = self.modified_at.isoformat()
            if self.description:
                group.attrs["description"] = self.description

    def load_metadata(self, group: h5py.Group) -> None:
        """Populate metadata fields from an HDF5 group."""
        self.name = group.attrs.get("name", self.name)
        if "created_at" in group.attrs:
            self.created_at = datetime.fromisoformat(group.attrs["created_at"])
        if "modified_at" in group.attrs:
            self.modified_at = datetime.fromisoformat(group.attrs["modified_at"])
        if "description" in group.attrs:
            self.description = group.attrs["description"]

    def touch(self) -> None:
        """Update ``modified_at`` and persist the change."""
        self.modified_at = datetime.now()
        self.save_metadata()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _can_persist(self) -> bool:
        return self.experiment is not None or self._hdf5_override is not None

    @classmethod
    def _next_standalone_index(cls) -> int:
        counter = cls._standalone_counters.setdefault(cls.__name__, 0)
        cls._standalone_counters[cls.__name__] += 1
        return counter


