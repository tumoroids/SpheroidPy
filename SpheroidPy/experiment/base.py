from __future__ import annotations
from typing import TYPE_CHECKING
import h5py
from datetime import datetime


if TYPE_CHECKING:
    from SpheroidPy.experiment.experiment import Experiment


class Base:
    """Base class for Experiment, Result, and Analysis classes.
    
    Attributes:
        name: Name of the instance
        type_name: Type of instance ('Result' or 'Analysis')
        index: Index in parent container
        experiment: Parent Experiment instance
        hdf5_key: HDF5 storage key
        created_at: When this instance was created
        modified_at: When this instance was last modified
        description: Optional description
    """

    name: str
    type_name: str
    index: int
    experiment: Experiment
    hdf5_key: str
    created_at: datetime
    modified_at: datetime
    description: str | None

    def __init__(self, name: str, experiment: Experiment, type_name: str | None = None, description: str | None = None) -> None:
        """Initialize base class.
        
        Args:
            name: Name of the element
            experiment: Parent Experiment instance
            type_name: Type of element ('Result' or 'Analysis')
            description: Optional description
        """
        self.name = name
        self.type_name = type_name
        self.experiment = experiment
        self.description = description
        self.created_at = datetime.now()
        self.modified_at = datetime.now()
        
        # Get index and set hdf5_key before adding to experiment
        self.index = len(getattr(experiment, f"{'results' if type_name=='Result' else 'analyses'}_dict"))
        self.hdf5_key = f'{type_name}/{self.index}-{name}'
        
        # Add element to experiment
        experiment._add_element(self)

    def save_metadata_to_hdf5(self):
        """Save metadata to HDF5 file."""
        with h5py.File(self.hdf5_path, 'a') as hdf_file:
            instance_group = hdf_file[self.hdf5_key]
            instance_group.attrs['created_at'] = self.created_at.isoformat()
            instance_group.attrs['modified_at'] = self.modified_at.isoformat()
            if self.description:
                instance_group.attrs['description'] = self.description

    def load_metadata_from_hdf5(self, instance_group: h5py.Group):
        """Load metadata from HDF5 group."""
        if 'created_at' in instance_group.attrs:
            self.created_at = datetime.fromisoformat(instance_group.attrs['created_at'])
        if 'modified_at' in instance_group.attrs:
            self.modified_at = datetime.fromisoformat(instance_group.attrs['modified_at'])
        if 'description' in instance_group.attrs:
            self.description = instance_group.attrs['description']

    def update_description(self, description: str):
        """Update instance description and modified time."""
        self.description = description
        self.modified_at = datetime.now()
        self.save_metadata_to_hdf5()

    ''' Properties from the associated Experiment Class '''
    @property
    def hdf5_path(self) -> str:
        return self.experiment.hdf5_path
    @property
    def col(self):
        return self.experiment.col
    @property
    def row(self):
        return self.experiment.row