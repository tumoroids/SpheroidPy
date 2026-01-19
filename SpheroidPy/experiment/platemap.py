from __future__ import annotations
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from openpyxl import Workbook, load_workbook
from openpyxl.utils import get_column_letter
from openpyxl.styles import PatternFill, Alignment, Font, Protection, Border, Side
import subprocess, time, psutil

import pandas as pd
import ipywidgets as widgets
from IPython.display import display, clear_output
import openpyxl, h5py
from datetime import datetime

from pathlib import Path
import random, re
import pandas as pd

if TYPE_CHECKING:
    from SpheroidPy.experiment.livecell_replicate import LiveCellReplicate
    from SpheroidPy.experiment.result import Result
from SpheroidPy.utils.utils import format_thousand_annotation, find_replicates
from SpheroidPy.utils.file_management import direct_subgroups


# Ignorieren von PerformanceWarnings in einem bestimmten Codeblock
import warnings
from pandas.errors import PerformanceWarning
with warnings.catch_warnings():
    warnings.simplefilter("ignore", PerformanceWarning)


class Platemap:
    """A class representing the layout and contents of an experimental plate.
    
    Manages information about cell lines and compounds in each well position.
    Provides functionality for data entry, visualization and analysis of plate layouts.
    
    This is an optional feature for Results that use plate-based well layouts.
    
    Attributes:
        replicate: Associated LiveCellReplicate object (or Result for backward compatibility)
        cell_lines: Dictionary mapping cell line names to their plate positions
        compounds: Dictionary mapping compound names to their plate positions
        hdf5_key: Key for accessing platemap data in HDF5 file
    """

    replicate: "LiveCellReplicate" | "Result"  # Can work with either

    # Dicts for the dataframes
    cell_lines: dict
    compounds: dict

    def __init__(self, replicate: "LiveCellReplicate" | "Result") -> None:
        """Initialize a new Platemap instance.
        
        Creates a new platemap associated with the given Replicate or Result object and initializes
        storage for cell line and compound data.

        Args:
            replicate: The LiveCellReplicate or Result object this platemap belongs to
        """
        self.replicate = replicate
        self.cell_lines = {}
        self.compounds = {}
        self._initialize_hdf5_structure()

    @property
    def hdf5_key(self) -> str:
        """Get HDF5 key for this platemap."""
        if hasattr(self.replicate, 'hdf5_key'):
            return f'{self.replicate.hdf5_key}/Platemap'
        # Fallback for Result without hdf5_key
        # If replicate is a LiveCellReplicate, get index from its result
        # If replicate is a Result, get index directly
        if hasattr(self.replicate, 'result'):
            # replicate is LiveCellReplicate - get index from result
            result = self.replicate.result
            result_index = getattr(result, '_result_index', 0)
            result_key = f"Result/{result_index}-{result.name}"
        elif hasattr(self.replicate, 'name'):
            # replicate is Result - get index directly
            result_index = getattr(self.replicate, '_result_index', 0)
            result_key = f"Result/{result_index}-{self.replicate.name}"
        else:
            raise ValueError("Cannot determine HDF5 key for platemap")
        
        return f'{result_key}/ReplicateInfo/Platemap'

    @property
    def hdf5_path(self) -> Path:
        """Get HDF5 file path."""
        if hasattr(self.replicate, 'hdf5_path') and self.replicate.hdf5_path:
            return self.replicate.hdf5_path
        # Fallback: try to get from result's experiment
        if hasattr(self.replicate, 'result') and hasattr(self.replicate.result, 'hdf5_path'):
            return self.replicate.result.hdf5_path
        if hasattr(self.replicate, 'experiment') and hasattr(self.replicate.experiment, 'hdf5_path'):
            return self.replicate.experiment.hdf5_path
        raise ValueError("Cannot determine HDF5 path for platemap")

    def _initialize_hdf5_structure(self):
        """Initialize HDF5 storage structure."""
        try:
            with h5py.File(self.hdf5_path, 'a') as hdf_file:
                # Use require_group to create if it doesn't exist
                element_group = hdf_file.require_group(self.hdf5_key)
                element_group.attrs['date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                for group_name in ['cell_line', 'compound']:
                    element_group.require_group(group_name)
        except Exception as e:
            import logging
            logger = logging.getLogger("SpheroidPy.experiment.platemap")
            logger.warning(f"Could not initialize HDF5 structure for platemap: {e}")

    def update_plate_data(self, name: str, data_type: str, value_dict: dict | None = None) -> None:
        """Update plate data for cell lines or compounds.
        
        Automatically triggers collection update if replicate is associated with a result.
        """
        if data_type not in ['cell_line', 'compound']:
            raise ValueError("data_type must be 'cell_line' or 'compound'")
            
        data_dict = self.cell_lines if data_type == 'cell_line' else self.compounds
        
        if isinstance(name, str):
            if name in data_dict:
                data_dict[name] = self._update_platemap(
                    name, data_type.title(), 
                    value_dict, 
                    data_dict[name].fillna('')
                )
            else:
                data_dict[name] = self._update_platemap(name, data_type.title(), value_dict)
            self._save_hdf5(data_dict[name], name, data_type)
            
            # Trigger collection update if replicate is associated with a result
            if hasattr(self.replicate, 'result') and hasattr(self.replicate.result, '_update_collections_from_replicates'):
                self.replicate.result._update_collections_from_replicates()

    @classmethod
    def from_file(cls, replicate: "LiveCellReplicate" | "Result", instance_group: h5py.Group, hdf5_path: str) -> "Platemap":
        """Load a platemap from HDF5 file.
        
        Args:
            replicate: The LiveCellReplicate or Result object this platemap belongs to
            instance_group: HDF5 group for this platemap
            hdf5_path: Path to HDF5 file
            
        Returns:
            Platemap instance loaded from HDF5
        """
        platemap = object.__new__(cls)

        # General Properties
        platemap.type_name = 'Platemap'
        platemap.replicate = replicate

        platemap.cell_lines = {}
        platemap.compounds = {}

        # Determine hdf5_key
        if hasattr(replicate, 'hdf5_key'):
            base_key = f'{replicate.hdf5_key}/Platemap'
        else:
            # Fallback for Result without hdf5_key
            # If replicate is a LiveCellReplicate, get index from its result
            # If replicate is a Result, get index directly
            if hasattr(replicate, 'result'):
                # replicate is LiveCellReplicate - get index from result
                result = replicate.result
                result_index = getattr(result, '_result_index', 0)
                base_key = f"Result/{result_index}-{result.name}/ReplicateInfo/Platemap"
            elif hasattr(replicate, 'name'):
                # replicate is Result - get index directly
                result_index = getattr(replicate, '_result_index', 0)
                base_key = f"Result/{result_index}-{replicate.name}/ReplicateInfo/Platemap"
            else:
                base_key = str(instance_group.name)

        for type, dict_ in zip(['cell_line', 'compound'], [platemap.cell_lines, platemap.compounds]):
            if type in instance_group:
                for name in instance_group[type]:
                    path = Path(hdf5_path).absolute()
                    platemap_key = f'{base_key}/{type}/{name}'
                    try:
                        df = pd.read_hdf(str(path), key=f'/{platemap_key}')
                        dict_[name] = df
                    except Exception as e:
                        import logging
                        logger = logging.getLogger("SpheroidPy.experiment.platemap")
                        logger.warning(f"Could not load {type} '{name}': {e}")

        #print(f'Platemap has been loaded!')
        return platemap

    def cell_line(self, name: str | list, value_dict: dict | None = None):
        """Add or update cell line information in the platemap.
        
        Automatically triggers collection update if replicate is associated with a result.
        
        Args:
            name: Cell line name or list of names
            value_dict: Optional dictionary mapping well positions to values
                Format: {'A1': value1, 'B2': value2, ...}
        """
        if isinstance(name, str):
            if name in self.cell_lines:
                self.cell_lines[name] = self._update_platemap(name, 'Cell line', value_dict, self.cell_lines[name].fillna(''))
            else:
                self.cell_lines[name] = self._update_platemap(name, 'Cell line', value_dict)
            self._save_hdf5(self.cell_lines[name], name, 'cell_line')
            self.heatmap(name)
            
            # Trigger collection update
            if hasattr(self.replicate, 'result') and hasattr(self.replicate.result, '_update_collections_from_replicates'):
                self.replicate.result._update_collections_from_replicates()
        elif isinstance(name, list):
            for cell in name:
                self.cell_line(cell)

    def compound(self, name: str | list, value_dict: dict | None = None):
        """Add or update compound information in the platemap.
        
        Automatically triggers collection update if replicate is associated with a result.
        
        Args:
            name: Compound name or list of names
            value_dict: Optional dictionary mapping well positions to values
                Format: {'A1': value1, 'B2': value2, ...}
        """
        if isinstance(name, str):
            if name in self.compounds:
                self.compounds[name] = self._update_platemap(name,'Compound', value_dict, self.compounds[name].fillna(''))
            else:
                self.compounds[name] = self._update_platemap(name, 'Compound', value_dict)
            self._save_hdf5(self.compounds[name], name, 'compound')
            self.heatmap(name)
            
            # Trigger collection update
            if hasattr(self.replicate, 'result') and hasattr(self.replicate.result, '_update_collections_from_replicates'):
                self.replicate.result._update_collections_from_replicates()
        elif isinstance(name, list):
            for compound in name:
                self.compound(compound)

    def replicates(self, element_name: str | None = None) -> dict:
        """Get replicate groups for cell lines and/or compounds.
        
        Args:
            element_name: Optional name of cell line or compound to filter by.
                        If None, returns replicates for all elements.
                        
        Returns:
            Dictionary mapping conditions to lists of well positions
        """
        return self.find_replicates(element_name)

    def heatmap(self, element: str, color_map: str | None = None, ax: plt.Axes | None = None):
        """Generate a heatmap visualization of the platemap data."""
        try:
            # Try to get data from cell lines or compounds
            if element in self.cell_lines:
                data_pivot = self.cell_lines[element]
            else:
                data_pivot = self.compounds[element]
            
            # Convert to numeric, replacing non-numeric values with NaN
            data_pivot = data_pivot.apply(pd.to_numeric, errors='coerce')
            
            # Create annotations, handling NaN values
            annotations = np.array([[format_thousand_annotation(val) if not pd.isna(val) else ''
                                   for val in row] 
                                  for row in data_pivot.values])

            if color_map is None:
                color_map = 'viridis'
            
            # Create heatmap
            heatmap = sns.heatmap(data_pivot, 
                                cmap=color_map, 
                                annot=annotations, 
                                fmt='', 
                                linewidths=1, 
                                linecolor='black', 
                                square=True, 
                                cbar=False, 
                                ax=ax if ax is not None else plt.gca(),
                                annot_kws={"size":6})
            
            # Show if no axes provided
            if ax is None:
                plt.title(f'Platemap: {element}')
                plt.show()
            
            return heatmap
            
        except Exception as e:
            print(f"Error creating heatmap for {element}: {str(e)}")
            if ax is not None:
                plt.sca(ax)
                plt.text(0.5, 0.5, f'Error: {str(e)}', 
                        ha='center', va='center', 
                        transform=ax.transAxes)
            return None

    @property
    def map(self):
        """Display heatmaps for all cell lines and compounds."""
        nrows = max(len(self.cell_lines), len(self.compounds))
        if nrows == 0:
            print("No data to display")
            return
        
        # Create figure with 2 columns
        fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(12, 10))
        
        # Handle single row case
        if nrows == 1:
            axes = np.array([[axes[0]], [axes[1]]])  # Make 2D with correct shape
        
        # Plot cell lines
        for n, cell_line in enumerate(self.cell_lines.keys()):
            self.heatmap(cell_line, ax=axes[n, 0])
            axes[n, 0].set_title(cell_line)
            axes[n, 0].set_xlabel('')
            axes[n, 0].set_ylabel('')
        
        # Plot compounds
        for m, drug in enumerate(self.compounds.keys()):
            self.heatmap(drug, ax=axes[m, 1])
            axes[m, 1].set_title(drug)
            axes[m, 1].set_xlabel('')
            axes[m, 1].set_ylabel('')
        
        plt.tight_layout()
        plt.show()

    def find_replicates(self, element_name: str | None = None) -> dict:
        """Find replicate groups based on element values.
        
        Args:
            element_name: Optional name of cell line or compound to filter by.
                        If None, returns replicates for all elements.
                        
        Returns:
            Dictionary mapping unique conditions to lists of well positions
        """
        if element_name is None:
            # Original behavior - combine all elements
            merged_dict = {k: v for d in (self.cell_lines, self.compounds) for k, v in d.items()}
        else:
            # Filter by specific element
            if element_name in self.cell_lines:
                merged_dict = {element_name: self.cell_lines[element_name]}
            elif element_name in self.compounds:
                merged_dict = {element_name: self.compounds[element_name]}
            else:
                raise ValueError(f"Element '{element_name}' not found in cell lines or compounds")
        
        # Find unique values and their well positions
        replicates = {}
        for name, df in merged_dict.items():
            for value in df.values.flatten():
                if pd.isna(value) or value == '':
                    continue
                    
                # Find wells with this value
                wells = []
                for row in df.index:
                    for col in df.columns:
                        if df.loc[row, col] == value:
                            wells.append(f"{row}{col}")
                            
                if wells:
                    key = ((name, value),)
                    if key not in replicates:
                        replicates[key] = []
                    replicates[key].extend(wells)
        
        return replicates

    @property
    def col(self):
        """Get number of columns from experiment layout."""
        # Try to get from replicate's result's experiment
        if hasattr(self.replicate, 'result') and hasattr(self.replicate.result, 'experiment'):
            if hasattr(self.replicate.result.experiment, 'col'):
                return self.replicate.result.experiment.col
        # Try to get from replicate's experiment (if replicate is Result)
        if hasattr(self.replicate, 'experiment') and hasattr(self.replicate.experiment, 'col'):
            return self.replicate.experiment.col
        # Default fallback
        return 12  # Default for 96-well plate
    
    @property
    def row(self):
        """Get number of rows from experiment layout."""
        # Try to get from replicate's result's experiment
        if hasattr(self.replicate, 'result') and hasattr(self.replicate.result, 'experiment'):
            if hasattr(self.replicate.result.experiment, 'row'):
                return self.replicate.result.experiment.row
        # Try to get from replicate's experiment (if replicate is Result)
        if hasattr(self.replicate, 'experiment') and hasattr(self.replicate.experiment, 'row'):
            return self.replicate.experiment.row
        # Default fallback
        return 8  # Default for 96-well plate

    def _update_platemap(self, name: str, header: str, value_dict: dict | None = None, df: pd.DataFrame | None = None, visualize: bool = True):
        # layout
        rows = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P'][:self.row]
        columns = [str(i) for i in range(1, self.col + 1)]

        if isinstance(df, pd.DataFrame):
            platemap_dataframe = df
        else:
            # empty platemap
            platemap_dataframe = pd.DataFrame('', index=rows, columns=columns)

        if isinstance(value_dict, dict):
            for value, cell_range in value_dict.items():
                start, end = cell_range.split(':')

                # Start- und Endzellen in Zeilen- und Spaltenindizes aufteilen
                start_row, start_col = start[0], int(start[1:])
                end_row, end_col = end[0], int(end[1:])

                # Zeilen- und Spaltenbereiche ermitteln
                row_range = platemap_dataframe.index[
                            platemap_dataframe.index.get_loc(start_row):platemap_dataframe.index.get_loc(end_row) + 1]
                col_range = platemap_dataframe.columns[
                            platemap_dataframe.columns.get_loc(str(start_col)):platemap_dataframe.columns.get_loc(
                                str(end_col)) + 1]

                # Bereich mit dem angegebenen Wert füllen
                for row in row_range:
                    for col in col_range:
                        platemap_dataframe.loc[row, col] = value

        if visualize:
            platemap_dataframe_updated = self._platemap_entry(platemap_dataframe, f'{header}: {name}')
            return platemap_dataframe_updated
        else:
            return platemap_dataframe

    def _platemap_entry(self, df: pd.DataFrame, title: str):
        # Erstellen eines Outputs für das gesamte Widget
        widget_output = widgets.Output()

        # Titel-Widget hinzufügen
        title_widget = widgets.HTML(value=f"<h2 style='text-align:left;'>{title}</h2>")

        # Kopie der DataFrame für Bearbeitungen erstellen
        edited_df = df.copy()

        # Erstelle Widgets für jede Zelle und fülle sie mit Startwerten
        cell_widgets = {row: {col: widgets.Text(value=str(df.loc[row, col]), layout=widgets.Layout(width="50px"))
                              for col in df.columns} for row in df.index}

        # Layout für die Platte mit fettgedruckten, zentrierten Zeilen- und Spaltenbeschriftungen
        header_row = [widgets.HTML(value="", layout=widgets.Layout(width="50px"))] + \
                     [widgets.HTML(value=f"<b>{col}</b>",
                                   layout=widgets.Layout(width="50px", justify_content="center", text_align="center"))
                      for col in df.columns]
        rows_with_labels = [widgets.HBox(header_row)]

        # Layout für jede Zeile mit fettgedruckter Zeilenbeschriftung und Eingabefeldern
        for row in df.index:
            row_widgets = [widgets.HTML(value=f"<b>{row}</b>",
                                        layout=widgets.Layout(width="50px", text_align="center"))] + \
                          [cell_widgets[row][col] for col in df.columns]
            rows_with_labels.append(widgets.HBox(row_widgets))

        # Vbox zum Stapeln der Reihen und Spalten
        table_box = widgets.VBox(rows_with_labels)

        def update_df(button):
            # Speichert Fehlernachrichten für ungültige Eingaben
            error_messages = []

            # Aktualisiere den DataFrame nur mit gültigen numerischen Werten
            for row in df.index:
                for col in df.columns:
                    try:
                        # Konvertiere die Eingabe in Float (damit auch Integer akzeptiert werden)
                        value = float(cell_widgets[row][col].value)
                        edited_df.loc[row, col] = value
                    except ValueError:
                        # Bei ungültiger Eingabe wird der Wert auf NaN gesetzt und eine Fehlermeldung hinzugefügt
                        edited_df.loc[row, col] = float('nan')
                        error_messages.append(f"Ungültige Eingabe in Zelle {row}{col}: Nur Zahlen sind erlaubt.")

            # Entferne das Widget nach dem Speichern
            widget_output.clear_output()
            print("DataFrame has been updated!")

        # Button zum Aktualisieren
        update_button = widgets.Button(description="Update")
        update_button.on_click(update_df)

        # Anzeige der Tabelle und des Buttons im Output-Container
        with widget_output:
            display(title_widget, table_box, update_button)

        display(widget_output)

        return edited_df

    def _save_hdf5(self, df: pd.DataFrame, df_name: str, group_name: str):
        """Save platemap DataFrame to HDF5."""
        if group_name not in ['cell_line', 'compound']:
            raise Exception('Group name must be "cell_line", "compound"')

        # 1) Sanitize the name for a valid HDF5 path (avoid NaturalNameWarning)
        safe_name = re.sub(r'[^0-9a-zA-Z_]', '_', str(df_name))
        # Do not start with a digit
        if safe_name and safe_name[0].isdigit():
            safe_name = f'_{safe_name}'

        # 2) Coerce to numeric so PyTables doesn't pickle object dtypes (avoid PerformanceWarning)
        df_to_store = df.copy()
        df_to_store = df_to_store.apply(pd.to_numeric, errors='coerce')
        # Optional: make column labels numeric when possible
        try:
            df_to_store.columns = [int(c) if isinstance(c, str) and c.isdigit() else c for c in df_to_store.columns]
        except Exception:
            pass

        # 3) Store as table format explicitly
        df_to_store.to_hdf(self.hdf5_path, key=f'{self.hdf5_key}/{group_name}/{safe_name}', mode='a', format='table')
        '''  hdf5 Structure '''
        with h5py.File(self.hdf5_path, 'a') as hdf_file:
            current_group = hdf_file[f'{self.hdf5_key}/{group_name}/{safe_name}']
            current_group.attrs['date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def _save_all_to_hdf5(self):
        """Save all platemap data to HDF5."""
        for name, df in self.cell_lines.items():
            self._save_hdf5(df, name, 'cell_line')
        for name, df in self.compounds.items():
            self._save_hdf5(df, name, 'compound')

    # todo
    def _worksheet(self, workbook: Workbook) -> Workbook:
        pass

