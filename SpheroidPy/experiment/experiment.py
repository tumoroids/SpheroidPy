from openpyxl import Workbook, load_workbook
from openpyxl.utils import get_column_letter
from openpyxl.styles import PatternFill, Alignment, Font, Protection, Border, Side
import subprocess, time, psutil, h5py

from pathlib import Path
import random, re
from datetime import datetime  # Update import

import pandas as pd

# Classes
from SpheroidPy.experiment.platemap import Platemap
from SpheroidPy.experiment.result import Result
from SpheroidPy.experiment.analysis import Analysis#, AnalysisJob
from SpheroidPy.experiment.visualisation import Visualisation
# Methods
from SpheroidPy.utils.file_management import direct_subgroups



class Experiment:
    """A class representing a scientific experiment with plate-based assays.
    
    This class manages experimental data from plate-based assays, including layout,
    results, analyses and visualization. It provides methods for data storage,
    retrieval and analysis.

    Attributes:
        name (str): Name of the experiment
        layout (int): Number of wells in the plate (e.g. 96, 384)
        col (int): Number of columns in the plate layout
        row (int): Number of rows in the plate layout
        results_dict (dict): Dictionary mapping result names to Result objects
        analyses_dict (dict): Dictionary mapping analysis names to Analysis objects
        visualisation (Visualisation): Object for experiment visualization
        hdf5_path (Path): Path to HDF5 file storing experiment data
    """
    
    # Use dataclass for cleaner attribute definition
    from dataclasses import dataclass
    
    # Define constants at class level
    SUPPORTED_LAYOUTS = {
        384: (24, 16),  # (columns, rows)
        96: (12, 8), 
        48: (8, 6),
        24: (6, 4),
        12: (4, 3),
        6: (3, 2)
    }
    
    # Excel style constants
    EXCEL_STYLES = {
        'tab_color': 'E6B8B7',
        'table_colors': ['963634', 'E6B8B7', 'F2DCDB'],
        'border': Border(
            left=Side(style='thin'),
            right=Side(style='thin'), 
            top=Side(style='thin'),
            bottom=Side(style='thin')
        )
    }

    def __init__(self, name: str, layout: int, path: Path | None = None, description: str | None = None) -> None:
        """Initialize experiment with specified layout.
        
        Creates a new experiment instance with the given name and plate layout.
        Sets up data storage directories and initializes dictionaries for results
        and analyses.

        Args:
            name (str): Name of the experiment
            layout (int): Number of wells in the plate (e.g. 96, 384)
            path (Path | None, optional): Custom path for data storage. If None,
                uses current working directory. Defaults to None.
            description: Optional description of the experiment
        """
        # Initialize basic attributes first
        self.name = name
        self.description = description
        self.created_at = datetime.now()
        self.modified_at = datetime.now()
        
        # Validate and set layout
        if layout not in self.SUPPORTED_LAYOUTS:
            raise ValueError(f"Layout {layout} not supported. Must be one of: {list(self.SUPPORTED_LAYOUTS.keys())}")
        self.layout = layout
        self.col, self.row = self.SUPPORTED_LAYOUTS[layout]

        # Initialize storage
        self.results_dict = {}
        self.analyses_dict = {}
        
        # Create visualization instance and set experiment
        self.visualisation = Visualisation()
        self.visualisation.set_experiment(self)

        # Setup data storage path
        self.hdf5_path = self._setup_storage_path(path)
        self._create_hdf5_file()

    def _setup_storage_path(self, path: Path | None = None) -> Path:
        """Setup and create storage directory."""
        storage_path = Path.cwd() if path is None else path
        exp_path = storage_path / self.name
        exp_path.mkdir(parents=True, exist_ok=True)
        return exp_path / f'Experiment_{self.name}_data.h5'

    @classmethod
    def from_file(cls, filepath: Path | str) -> 'Experiment':
        """Create experiment instance from file."""
        experiment = object.__new__(cls)
        
        with h5py.File(filepath, 'r') as hdf_file:
            # Load basic attributes
            experiment.name = hdf_file.attrs['name']
            experiment.layout = hdf_file.attrs['layout']
            experiment.col, experiment.row = experiment.SUPPORTED_LAYOUTS[experiment.layout]
            experiment.hdf5_path = Path(filepath)
            
            # Load metadata
            if 'created_at' in hdf_file.attrs:
                experiment.created_at = datetime.fromisoformat(hdf_file.attrs['created_at'])
            else:
                experiment.created_at = datetime.now()
                
            if 'modified_at' in hdf_file.attrs:
                experiment.modified_at = datetime.fromisoformat(hdf_file.attrs['modified_at'])
            else:
                experiment.modified_at = datetime.now()
                
            if 'description' in hdf_file.attrs:
                experiment.description = hdf_file.attrs['description']
            else:
                experiment.description = None

            # Initialize storage
            experiment.results_dict = {}
            experiment.analyses_dict = {}
            
            # Create visualization instance and set experiment
            experiment.visualisation = Visualisation()
            experiment.visualisation.set_experiment(experiment)

            # Load results and analyses
            for element, class_, dict_ in zip(['Result', 'Analysis'], 
                                            [Result, Analysis], 
                                            [experiment.results_dict, experiment.analyses_dict]):
                if element not in hdf_file:
                    continue
                    
                element_group = hdf_file[element]
                for instance in element_group:
                    if element == 'Analysis':
                        continue  # todo: ToBeImplemented!!!!

                    try:
                        index, name = instance.split("-")[0], "-".join(instance.split("-")[1:])
                        instance_group = element_group[instance]
                        dict_[name] = class_.from_file(name, index, experiment, 
                                                     instance_group, filepath)
                    except Exception as e:
                        print(f"Warning: Failed to load {element} '{instance}': {e}")

        return experiment

    @property
    def results(self) -> Result | dict | None:
        if len(self.results_dict) == 0:
            return None
        elif len(self.results_dict) == 1:
            return list(self.results_dict.values())[0]
        else:
            return self.results_dict

    def result(self, name: str | None = None) -> Result | dict:
        """Get or create a Result object.
        
        Args:
            name: Name of the Result to get/create. If None, returns dictionary of all Results.
            
        Returns:
            If name is None, returns dictionary of all Results.
            If name is provided, returns the specified Result object, creating it if it doesn't exist.
            
        Raises:
            Exception: If name is provided but is not a string.
        """
        if name is None:
            return self.results_dict
        if type(name) == str:
            if name not in self.results_dict:
                Result(name, self)
            return self.results_dict[name]
        else:
            raise Exception(f'{name} is no string')

    @property
    def analyses(self) -> Analysis | dict | None:
        if len(self.analyses_dict) == 0:
            return None
        elif len(self.analyses_dict) == 1:
            return list(self.analyses_dict.values())[0]
        else:
            return self.analyses_dict

    def analysis(self, name: str | None = None) -> Analysis | dict:
            """Get or create an Analysis object.
            
            Args:
                name: Name of the Analysis to get/create. If None, returns dictionary of all Analyses.
                
            Returns:
                If name is None, returns dictionary of all Analyses.
                If name is provided, returns the specified Analysis object, creating it if it doesn't exist.
                
            Raises:
                Exception: If name is provided but is not a string.
            """
            if name is None:
                return self.analyses_dict
            if type(name) == str:
                if name not in self.analyses_dict:
                    Analysis(name, self)
                return self.analyses_dict[name]
            else:
                raise Exception(f'{name} is no string')

    def to_excel(self, platmap_names: str | list | None = None, result_names: str | list | None = None,
                 analysis_names: str | list | None = None) -> None:
        workbook = self._create_excel_workbook_and_sheet()

        # if an element is given as string convert to list
        for el in [platmap_names, result_names, analysis_names]:
            if not isinstance(el, list):
                el = [el]

        for type, _name in zip(self.element_dict.keys(), [platmap_names, result_names, analysis_names]):
            if _name is None:
                for element in self.element_dict[type]:
                    element._worksheet(workbook)
            else:
                for element in self.element_dict[type]:
                    if element.name in _list:
                        element._worksheet(workbook)

    @property
    def hdf5_tree(self) -> None:
        print('Structure of hdf5-file:\n')
        with h5py.File(self.hdf5_path, 'r') as h5file:
            # Funktion zum rekursiven Durchlaufen und Anzeigen nur der Gruppen
            def print_groups(name, obj):
                if isinstance(obj, h5py.Group):
                    print(name)

            # Datei durchsuchen und nur die Gruppen ausgeben
            h5file.visititems(print_groups)

    def _create_hdf5_file(self):
        with h5py.File(self.hdf5_path, 'w') as hdf_file:
            # create groups for all associated types
            for element in ['Result', 'Analysis']:
                hdf_file.create_group(element, track_order=True)

            # save global Properties of Experiment
            hdf_file.attrs['name'] = self.name
            hdf_file.attrs['layout'] = self.layout
            hdf_file.attrs['created_at'] = self.created_at.isoformat()
            hdf_file.attrs['modified_at'] = self.modified_at.isoformat()
            if self.description:
                hdf_file.attrs['description'] = self.description

    def update_description(self, description: str):
        """Update experiment description and modified time."""
        self.description = description
        self.modified_at = datetime.now()
        with h5py.File(self.hdf5_path, 'a') as hdf_file:
            hdf_file.attrs['description'] = description
            hdf_file.attrs['modified_at'] = self.modified_at.isoformat()

    def _add_element(self, element: Result | Analysis) -> int:
        """Add a new element to the experiment.
        
        Args:
            element: Result or Analysis instance to add
            
        Returns:
            Index assigned to the element
            
        Raises:
            ValueError: If element name already exists
            Exception: If invalid element type
        """
        if element.type_name == 'Result':
            element_dict = self.results_dict
        elif element.type_name == 'Analysis':
            element_dict = self.analyses_dict
        else:
            raise Exception(f'Invalid element type: {element.type_name}')

        # Check if name is already taken
        if any(_el.name == element.name for _el in element_dict.values()):
            raise ValueError(f"Element {element.name} already exists! Please choose another name")

        # Add to dictionary
        element_dict[element.name] = element

        # Save in HDF5 file
        with h5py.File(self.hdf5_path, 'a') as hdf_file:
            type_group = hdf_file.require_group(element.type_name) #hdf_file[element.type_name]
            element_group = type_group.create_group(f'{element.index}-{element.name}', 
                                                  track_order=True)
            element_group.attrs['date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return element.index
    
    def _create_excel_workbook_and_sheet(self) -> Workbook:
        '''

        :return:
        '''

        ''' Create Excel-Workbook '''
        workbook = Workbook()
        worksheet = self.workbook.active
        worksheet.title = f'Overview'
        worksheet.sheet_properties.tabColor = self._tab_color
        worksheet.protection.sheet = True  # Blattschutz aktivieren

        ''' Create Worksheet '''
        # Header
        worksheet.merge_cells('B2:H3')
        header_cell = worksheet['B2']
        header_cell.value = "Experiment: Overview"
        header_cell.alignment = Alignment(horizontal="center", vertical="center")
        header_cell.font = Font(bold=True, color="FFFFFF", size=14)
        header_cell.fill = PatternFill(start_color=self._table_color1, end_color=self._table_color1, fill_type="solid")

        # Name and Layout
        worksheet.merge_cells('B4:C4')
        worksheet['B4'] = "Name"
        worksheet['D4'].value = name
        worksheet.merge_cells('B5:B6')
        worksheet['B5'] = "Date"
        worksheet['C5'] = "created"
        worksheet['D5'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        worksheet['C6'] = "modified"
        worksheet['D6'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for row in range(4, 7):
            worksheet.merge_cells(f'D{row}:E{row}')
            worksheet.merge_cells(f'G{row}:H{row}')
            for col in ['C', 'D', 'E', 'B', 'F', 'G', 'H']:
                worksheet[f'{col}{row}'].border = self._border
        worksheet['F4'] = "Layout"
        worksheet['G4'] = f"{layout} Well Format"
        worksheet['F5'] = ".."
        worksheet['F6'] = ".."
        for index in ['B4', 'B5', 'C5', 'C6', 'F4', 'F5', 'F6']:
            worksheet[index].fill = PatternFill(start_color=self._table_color3, end_color=self._table_color3,
                                                fill_type="solid")

        # Tabelle für Platmaps erstellen
        worksheet.merge_cells('B10:B14')
        platemaps_header = worksheet['B10']
        platemaps_header.value = "Platemaps"
        platemaps_header.alignment = Alignment(horizontal="center", vertical="center")
        # platemaps_header.font = default_font
        platemaps_header.fill = PatternFill(start_color=self._table_color2, end_color=self._table_color2,
                                            fill_type="solid")
        for row in range(10, 15):
            worksheet.merge_cells(f'C{row}:D{row}')
            worksheet[f'C{row}'].fill = PatternFill(start_color=self._table_color3, end_color=self._table_color3,
                                                    fill_type="solid")
            for col in ['C', 'B', 'D']:
                self.worksheet[f'{col}{row}'].border = self._border
            # ws[f'C{row}'].font = default_font
        worksheet['C10'] = "Name"
        worksheet['C11'] = "Date"
        worksheet['C12'] = "Cell lines"
        worksheet['C13'] = "Compounds"
        worksheet['C14'] = "index"

        # Tabelle für Results erstellen
        worksheet.merge_cells('B17:B21')
        worksheet['B17'] = "Results"
        worksheet['B17'].alignment = Alignment(horizontal="center", vertical="center")
        # self.worksheet['B17'].font = default_font
        worksheet['B17'].fill = PatternFill(start_color=self._table_color2, end_color=self._table_color2,
                                            fill_type="solid")
        for row in range(17, 22):
            worksheet.merge_cells(f'C{row}:D{row}')
            worksheet[f'C{row}'].fill = PatternFill(start_color=self._table_color3, end_color=self._table_color3,
                                                    fill_type="solid")
            for col in ['C', 'B', 'D']:
                worksheet[f'{col}{row}'].border = self._border
            # ws[f'C{row}'].font = default_font
        worksheet['C17'] = "Name"
        worksheet['C18'] = "Date"
        worksheet['C19'] = "Platemap"
        worksheet['C20'] = "Metric"
        worksheet['C21'] = "index"

        # Tabelle für Analysis erstellen
        worksheet.merge_cells('B24:B28')
        worksheet['B24'] = "Analysis"
        worksheet['B24'].alignment = Alignment(horizontal="center", vertical="center")
        # self.worksheet['B24'].font = default_font
        worksheet['B24'].fill = PatternFill(start_color=self._table_color2, end_color=self._table_color2,
                                            fill_type="solid")
        for row in range(24, 29):
            worksheet.merge_cells(f'C{row}:D{row}')
            worksheet[f'C{row}'].fill = PatternFill(start_color=self._table_color3, end_color=self._table_color3,
                                                    fill_type="solid")
            for col in ['C', 'B', 'D']:
                self.worksheet[f'{col}{row}'].border = self._border
            # ws[f'C{row}'].font = default_font
        worksheet['C24'] = "Name"
        worksheet['C25'] = "Date"
        worksheet['C26'] = "Results"
        worksheet['C27'] = "Parameters"
        worksheet['C28'] = "index"

        # Rahmen und Formatierung anwenden
        for col in ['C', 'D', 'E', 'B']:
            for row in range(4, 7):
                worksheet[f'{col}{row}'].border = self._border

        return workbook

    def __repr__(self) -> str:
        return (f'EXPERIMENT: {self.name}  ({self.layout} Well Plate)\n' +
                f'\n Elements:' +  #({len(self.element_dict["Platemap"])+len(self.element_dict["Result"])+len(self.element_dict["Analysis"])}): '+
                f'\n > Results({len(self.results_dict)}): ' + ', '.join(
                    element.name for element in list(self.results_dict.values())) +
                f'\n > AnalysisJobs({len(self.analyses_dict)}): ' + ', '.join(
                    element.name for element in list(self.analyses_dict.values())))

    @property
    def timepoints(self) -> tuple[list, list]:
        """Get experiment timepoints in absolute and relative format.
        
        Returns:
            tuple: (absolute_times, relative_times) where:
                absolute_times: List of datetime strings
                relative_times: List of hours from experiment start
        """
        if not self.results_dict:
            return [], []
        
        # Get timepoints from first result
        first_result = next(iter(self.results_dict.values()))
        return (first_result.time_points_array, 
                first_result.relative_time_array)

    def delete(self, name: str):
        """Delete a Result or Analysis from the experiment.
        
        Automatically determines element type based on name and removes it from
        both the experiment's dictionaries and the HDF5 file.
        
        Args:
            name: Name of the element to delete
            
        Raises:
            KeyError: If element with given name is not found
            ValueError: If name exists in both Results and Analyses
        """
        # Check where the element exists
        in_results = name in self.results_dict
        in_analyses = name in self.analyses_dict
        
        if not (in_results or in_analyses):
            raise KeyError(f"No Result or Analysis found with name '{name}'")
            
        if in_results and in_analyses:
            raise ValueError(f"Ambiguous name '{name}' - exists in both Results and Analyses. "
                            "Use delete_result() or delete_analysis() instead.")
        
        # Get element and its type
        if in_results:
            element = self.results_dict[name]
            element_dict = self.results_dict
            element_type = 'Result'
        else:
            element = self.analyses_dict[name]
            element_dict = self.analyses_dict
            element_type = 'Analysis'
        
        # Remove from dictionary
        del element_dict[name]
        
        # Remove from HDF5 file
        with h5py.File(self.hdf5_path, 'a') as hdf_file:
            if element.hdf5_key in hdf_file:
                del hdf_file[element.hdf5_key]
                
        print(f"{element_type} '{name}' successfully deleted")

    def _delete_element(self, element_name: str, element_type: str = 'Result'):
        """Delete a Result or Analysis from the experiment.
        
        Removes the element from both the experiment's dictionaries and the HDF5 file.
        
        Args:
            element_name: Name of the Result/Analysis to delete
            element_type: Type of element to delete ('Result' or 'Analysis')
            
        Raises:
            ValueError: If element_type is invalid or element doesn't exist
            KeyError: If element with given name is not found
        """
        if element_type not in ['Result', 'Analysis']:
            raise ValueError("element_type must be 'Result' or 'Analysis'")
        
        # Get appropriate dictionary
        element_dict = (self.results_dict if element_type == 'Result' 
                       else self.analyses_dict)
        
        # Check if element exists
        if element_name not in element_dict:
            raise KeyError(f"{element_type} '{element_name}' not found")
        
        # Get element's index for HDF5 path
        element = element_dict[element_name]
        element_hdf5_key = element.hdf5_key
        
        # Remove from dictionary
        del element_dict[element_name]
        
        # Remove from HDF5 file
        with h5py.File(self.hdf5_path, 'a') as hdf_file:
            if element_hdf5_key in hdf_file:
                del hdf_file[element_hdf5_key]
            
        print(f"{element_type} '{element_name}' successfully deleted")

    def delete_result(self, result_name: str):
        """Delete a Result from the experiment.
        
        Convenience method wrapping delete_element for Results.
        
        Args:
            result_name: Name of the Result to delete
        """
        self._delete_element(result_name, 'Result')
        
    def delete_analysis(self, analysis_name: str):
        """Delete an Analysis from the experiment.
        
        Convenience method wrapping delete_element for Analyses.
        
        Args:
            analysis_name: Name of the Analysis to delete
        """
        self._delete_element(analysis_name, 'Analysis')


if __name__ == '__main__':
    exp = Experiment('Test', 96, Path('TestExp'))


