from __future__ import annotations
from typing import TYPE_CHECKING

import ipywidgets as widgets
from IPython.display import display, HTML
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import cv2
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import pandas as pd
import h5py

from SpheroidPy.utils.geometry import fit_ellipse, add_fitted_contour

if TYPE_CHECKING:
    from SpheroidPy.experiment.experiment import Experiment

class Visualisation:
    """Class for interactive visualization of experiment data.
    
    Provides interactive widgets for exploring and analyzing experimental data,
    including image visualization, metric plotting, and data analysis.
    
    Attributes:
        experiment: Associated Experiment object
    """

    def __init__(self, experiment=None):
        """Initialize visualization for an experiment.
        
        Args:
            experiment: Optional Experiment object to visualize
        """
        self.experiment = experiment
        self._current_result = None
        self._current_analysis = None
        self._current_well = None
        self._current_time_index = 0  # Add this to store current time index
        self._viewer = None
        self._result_view = None  # Store reference to result view components
        
        # Add storage for widget states
        self._widget_states = {
            'show_contour': False,
            'thresholding_value': 1.0,
            'ai_value': 0.7,
            'channel_1': 'brightfield',  # Add default channel states
            'channel_2': 'fluorescence_green'
        }

    def set_experiment(self, experiment):
        """Set the experiment to visualize.
        
        Args:
            experiment: Experiment object to visualize
        """
        self.experiment = experiment

    def interactive(self, plot_count: int = 2):
        """Launch interactive visualization interface for the experiment."""
        if not self.experiment:
            print("Error: No experiment set")
            return
            
        if not hasattr(self.experiment, 'name'):
            print("Error: Experiment not properly initialized")
            return

        # Create main tab structure
        self._viewer = widgets.Tab()
        
        # Create tabs
        tabs = {
            'Overview': self._create_overview_tab(),
            'Results': self._create_results_tab(plot_count),
            'Analyses': self._create_analyses_tab()
        }
        
        self._viewer.children = list(tabs.values())
        for i, title in enumerate(tabs.keys()):
            self._viewer.set_title(i, title)
        
        # Add tab change handler to update overview
        def on_tab_change(change):
            if change.new == 0:  # Overview tab index
                # Update overview tab content
                self._viewer.children = (
                    self._create_overview_tab(),  # Recreate overview tab
                    self._viewer.children[1],     # Keep existing Results tab
                    self._viewer.children[2]      # Keep existing Analyses tab
                )
        
        self._viewer.observe(on_tab_change, names='selected_index')
        
        display(self._viewer)

    def _create_overview_tab(self) -> widgets.VBox:
        """Create overview tab showing experiment summary."""
        try:
            # Experiment info with styling
            exp_info = widgets.HTML(
                value=f"""
                <div style='padding: 10px; background-color: #f0f0f0; border-radius: 5px;'>
                    <h2 style='color: #2c3e50;'>Experiment: {self.experiment.name}</h2>
                    <div style='margin-left: 20px;'>
                        <p><b>Layout:</b> {self.experiment.layout} wells ({self.experiment.row}×{self.experiment.col})</p>
                        <p><b>Results:</b> {len(self.experiment.results_dict)}</p>
                        <p><b>Analyses:</b> {len(self.experiment.analyses_dict)}</p>
                    </div>
                </div>
                """
            )
            
            # Create Results section with platemaps
            results_header = widgets.HTML(
                value="<h3 style='color: #2c3e50;'>Results</h3>"
            )
            
            # Calculate layout for platemaps
            num_results = len(self.experiment.results_dict)
            if num_results > 0:
                # Create plate layouts for each result
                plate_layouts = []
                for result_name, result in self.experiment.results_dict.items():
                    # Create header for each result
                    result_header = widgets.HTML(
                        value=f"<h4 style='color: #2c3e50; text-align: center;'>{result_name}</h4>"
                    )
                    
                    # Create grid of well buttons
                    rows = 'ABCDEFGHIJKLMNOP'[:self.experiment.row]
                    cols = range(1, self.experiment.col + 1)
                    
                    well_buttons = []
                    for row in rows:
                        row_buttons = []
                        for col in cols:
                            well = f"{row}{col}"
                            btn = widgets.Button(
                                description=well,
                                layout=widgets.Layout(
                                    width='30px',
                                    height='30px'
                                ),
                                style=widgets.ButtonStyle(
                                    button_color=self._get_well_color(well, result)
                                )
                            )
                            # Add click handler for overview wells
                            btn.on_click(lambda b, well=well, result=result: self._on_overview_well_click(well, result))
                            row_buttons.append(btn)
                        well_buttons.append(widgets.HBox(row_buttons))
                    
                    grid = widgets.VBox(well_buttons)
                    
                    # Combine header and grid
                    plate_layout = widgets.VBox([
                        result_header,
                        grid
                    ], layout=widgets.Layout(
                        margin='0 10px',
                        align_items='center'
                    ))
                    
                    plate_layouts.append(plate_layout)
                
                # Create horizontal layout for all platemaps
                platemaps_box = widgets.HBox(
                    plate_layouts,
                    layout=widgets.Layout(
                        justify_content='center',
                        align_items='flex-start',
                        width='100%'
                    )
                )
            else:
                platemaps_box = widgets.HTML(value="<p>No results available</p>")
            
            # Add metrics summary if available
            metrics_summary = self._create_metrics_summary()
            
            return widgets.VBox([
                exp_info,
                results_header,
                platemaps_box,
                metrics_summary
            ])
            
        except Exception as e:
            return widgets.HTML(value=f"<p style='color: red;'>Error creating overview: {str(e)}</p>")

    def _create_results_tab(self, plot_count: int) -> widgets.VBox:
        """Create results tab with comprehensive visualization layout."""
        
        # Create result selector with matching style and inline dropdown
        result_selector = widgets.HTML(
            value=f"""
            <div style='padding: 10px; background-color: #f0f0f0; border-radius: 5px; margin-bottom: -73px;'>
                <div style='display: flex; align-items: center;'>
                    <h3 style='color: #2c3e50; margin: 0; flex-shrink: 0; min-height: 70px;'>Select Result:&nbsp;</h3>
                    <div id='result-dropdown-placeholder'></div>
                </div>
                <div id='result-info' style='margin-top: 10px; margin-left: 20px; padding-bottom: 5px;'></div>
                <div style='display: flex; margin-top: 20px;'>
                    <div style='display: flex; gap: 20px;'>
                        <div id='plate-layout-placeholder'></div>
                        <div id='dummy-plot-placeholder'></div>
                    </div>
                </div>
            </div>
            """
        )
        
        # Create dummy plot and dropdown
        plot_output = widgets.Output(
            layout=widgets.Layout(
                width='300px',     # Match platemap width
                height='300px'     # Match platemap height
            )
        )

        plot_dropdown = widgets.Dropdown(
            options=['Option 1', 'Option 2', 'Option 3'],
            description='Plot:',
            layout=widgets.Layout(width='200px')
        )

        def update_dummy_plot():
            with plot_output:
                plot_output.clear_output(wait=True)
                if self._current_result and self._current_well:
                    try:
                        # Get radius data for current well
                        radius_data = self._current_result.metric('radius')
                        well_data = radius_data[self._current_well]
                        
                        # Get current time point
                        current_time = radius_data.index[self._current_time_index]/24
                        current_radius = well_data.iloc[self._current_time_index]
                        
                        # Create plot
                        plt.figure(figsize=(4.5, 4))
                        plt.plot(radius_data.index/24, well_data, 'b-')  # Convert time to days
                        
                        # Add red circle at current time point
                        plt.plot(current_time, current_radius, 'ro', markersize=8)
                        
                        plt.xlabel('Time [days]', fontsize=12)
                        plt.ylabel('Radius [µm]', fontsize=12)
                        plt.grid(True, alpha=0.3)
                        plt.tight_layout()
                        plt.show()
                    except Exception as e:
                        plt.figure(figsize=(4.5, 4))
                        plt.text(0.5, 0.5, 'No radius data available', 
                                ha='center', va='center')
                        plt.axis('off')
                        plt.show()

        # Create plot section with header
        plot_section = widgets.VBox([
            widgets.HTML(
                value="<h3 style='color: #2c3e50;'>Metric</h3>"
            ),
            plot_output
        ], layout=widgets.Layout(
            align_items='center'
        ))

        # Create result dropdown
        result_dropdown = widgets.Dropdown(
            options=list(self.experiment.results_dict.keys()),
            description='',
            layout=widgets.Layout(width='200px')
        )
        
        # Create result info display
        result_info = widgets.HTML()
        
        # Create containers for dynamic content
        plate_layout = widgets.Output()
        image_viewer = widgets.Output()
        
        def update_result_info():
            if self._current_result:
                # Get timepoints info
                time_points = self._current_result.time_points_array
                first_time = time_points[0] if time_points else "N/A"
                last_time = time_points[-1] if time_points else "N/A"
                
                # Get condition info from platemap
                condition = "N/A"
                if self._current_well and self._current_result.platemap:
                    try:
                        # Find which replicate contains the current well
                        for rep_info, wells in self._current_result.platemap.replicates.items():
                            if self._current_well in wells:
                                # Take only the first element of the replicate key tuple
                                condition = rep_info[0] if isinstance(rep_info, tuple) else rep_info
                                break
                    except:
                        condition = "No condition information available"
                
                info_html = f"""
                <p><b>Time Range:</b> {first_time} to {last_time}</p>
                <p><b>Condition:</b> {condition}</p>
                """
                result_info.value = info_html
        
        def update_view():
            # Update result info
            update_result_info()
            
            # Update plate layout
            plate_layout.clear_output()
            with plate_layout:
                display(self._create_result_plate_layout())
            
            # Update image viewer
            image_viewer.clear_output()
            with image_viewer:
                display(self._create_image_viewer(plot_count))
        
        def on_result_change(change):
            if change.new:
                previous_well = self._current_well  # Store current well
                self._current_result = self.experiment.results_dict[change.new]
                
                # Only select initial well if we don't have a valid well selection
                if not previous_well or previous_well not in self._current_result.spheroid_dict:
                    self._select_initial_well()
                # Otherwise keep the previously selected well
                
                self._current_time_index = 0  # Reset time index when changing results
                update_view()
        
        result_dropdown.observe(on_result_change, names='value')
        
        # Initialize with first result
        if self.experiment.results_dict:
            first_result = next(iter(self.experiment.results_dict))
            # Set current result and well without triggering the observer
            self._current_result = self.experiment.results_dict[first_result]
            self._select_initial_well()
            # Set dropdown value (this won't trigger the observer)
            result_dropdown.value = first_result
            # Update view manually
            update_view()
        
        # Combine all elements in the styled container
        header_section = widgets.VBox([
            result_selector,
            widgets.HBox([
                widgets.Label(''),
                result_dropdown
            ], layout=widgets.Layout(
                margin='-45px 0 0 120px'
            )),
            widgets.HBox([
                widgets.Label(''),
                result_info
            ], layout=widgets.Layout(
                margin='10px 0 0 20px'
            )),
            widgets.HBox([
                widgets.Label(''),
                widgets.HBox([
                    plate_layout,
                    plot_section
                ], layout=widgets.Layout(
                    margin='0',
                    gap='20px'
                ))
            ], layout=widgets.Layout(
                margin='10px 0 0 20px'
            ))
        ])
        
        # Store references for updates
        self._result_view = {
            'dropdown': result_dropdown,
            'info': result_info,
            'plate_layout': plate_layout,
            'image_viewer': image_viewer,
            'update_info': update_result_info,
            'plot_section': plot_section,
            'update_plot': update_dummy_plot
        }
        
        # Initial update of dummy plot
        update_dummy_plot()
        
        return widgets.VBox([
            header_section,
            image_viewer
        ], layout=widgets.Layout(padding='10px'))

    def _select_initial_well(self):
        """Select the first well that has both data and is in the platemap."""
        if not self._current_result:
            return
        
        # Get wells with data
        data_wells = set(self._current_result.spheroid_dict.keys())
        
        # Get wells in platemap
        platemap_wells = set()
        for mapping in [self._current_result.platemap.cell_lines, 
                       self._current_result.platemap.compounds]:
            for df in mapping.values():
                for well in df.index:
                    if not pd.isna(df.loc[well]).all():
                        platemap_wells.add(well)
        
        # Find wells that have both data and are in platemap
        valid_wells = data_wells.intersection(platemap_wells)
        
        if valid_wells:
            # Sort wells to ensure consistent selection
            self._current_well = sorted(valid_wells)[0]
        else:
            # If no well has both, try just data wells
            self._current_well = sorted(data_wells)[0] if data_wells else None

    def _create_result_plate_layout(self) -> widgets.VBox:
        """Create plate layout specific to current result."""
        header = widgets.HTML(
            value="<h3 style='color: #2c3e50;'>Well Selection</h3>"
        )
        
        # Create grid of well buttons
        rows = 'ABCDEFGHIJKLMNOP'[:self.experiment.row]
        cols = range(1, self.experiment.col + 1)
        
        well_buttons = []
        for row in rows:
            row_buttons = []
            for col in cols:
                well = f"{row}{col}"
                has_data = (self._current_result and 
                           well in self._current_result.spheroid_dict)
                
                btn = widgets.Button(
                    description=well,
                    layout=widgets.Layout(
                        width='30px',
                        height='25px',
                        font_size='6px',
                        padding='0px'
                    ),
                    style=widgets.ButtonStyle(button_color=self._get_well_color(well)),
                    disabled=not has_data
                )
                
                if has_data:
                    btn.on_click(lambda b, well=well: self._on_result_well_click(well))
                row_buttons.append(btn)
            
            well_buttons.append(widgets.HBox(
                row_buttons,
                layout=widgets.Layout(margin='0px')
            ))
        
        return widgets.VBox([
            header,
            widgets.VBox(
                well_buttons,
                layout=widgets.Layout(
                    margin='5px',
                    overflow='hidden'  # Prevent scrolling
                )
            )
        ], layout=widgets.Layout(
            margin='5px',
            width='300px',
            height='auto',    # Auto height
            overflow='hidden' # Prevent scrolling
        ))

    def _create_metric_plot(self) -> widgets.VBox:
        """Create plot for radius metric."""
        header = widgets.HTML(
            value="<h3 style='color: #2c3e50;'>Radius over Time</h3>"
        )
        
        plot_output = widgets.Output(
            layout=widgets.Layout(
                width='600px',
                height='400px'
            )
        )
        
        def update_plot():
            plot_output.clear_output()
            with plot_output:
                if self._current_result and self._current_well:
                    try:
                        # Radius plot
                        radius_data = self._current_result.metric('radius')
                        plt.figure(figsize=(10, 6))
                        plt.plot(radius_data.index/24, radius_data[self._current_well])
                        plt.title('Radius over Time')
                        plt.xlabel('Time [days]')
                        plt.ylabel('Radius [μm]')
                        plt.grid(True, alpha=0.3)
                        plt.tight_layout()
                        plt.show()
                    except Exception as e:
                        plt.text(0.5, 0.5, f'No data available\n{str(e)}', 
                                ha='center', va='center')
        
        return widgets.VBox([header, plot_output])

    def _create_image_viewer(self, plot_count: int) -> widgets.VBox:
        """Create image viewer with time control and channel selection."""
        if not self._current_result:
            return widgets.HTML(value="<p>Select a result first</p>")
        
        # Use stored state for contour checkbox
        show_contour = widgets.Checkbox(
            value=self._widget_states['show_contour'],  # Use stored state
            description='Show Contour',
            indent=False,
            layout=widgets.Layout(margin='0 10px')
        )
        
        # Store checkbox state when changed
        def on_contour_change(change):
            self._widget_states['show_contour'] = change.new
        show_contour.observe(on_contour_change, names='value')
        
        # Time control section using result timepoints
        time_points = self._current_result.time_points_array if self._current_result else []
        rel_times = self._current_result.relative_time_array if self._current_result else []
        max_time = len(time_points) - 1 if time_points else 0
        
        time_slider = widgets.IntSlider(
            description='Time:',
            min=0,
            max=max_time,
            value=min(self._current_time_index, max_time),
            layout=widgets.Layout(width='50%')
        )
        
        def update_time_label(change):
            if rel_times:
                hours = int(rel_times[change.new])  # Convert to int
                days = hours // 24
                remaining_hours = hours % 24
                time_label.value = f"{int(days)}d {int(remaining_hours)}h"  # Convert both to int
                self._current_time_index = change.new
                self._result_view['update_plot']()
        
        def handle_key(event):
            if event.key == 'ArrowRight':
                time_slider.value = min(time_slider.value + 1, max_time)
            elif event.key == 'ArrowLeft':
                time_slider.value = max(time_slider.value - 1, 0)
        
        # Register keyboard event handler - modified to work globally
        display(widgets.HTML("""
            <script>
                // Remove any existing event listener first
                document.removeEventListener('keydown', window._spheroidKeyHandler);
                
                // Create new handler
                window._spheroidKeyHandler = function(event) {
                    if (event.key === 'ArrowRight' || event.key === 'ArrowLeft') {
                        // Prevent default only if not in an input field
                        if (document.activeElement.tagName !== 'INPUT') {
                            event.preventDefault();
                            window.dispatchEvent(new CustomEvent('arrow_key', {detail: event.key}));
                        }
                    }
                };
                
                // Add the new listener
                document.addEventListener('keydown', window._spheroidKeyHandler);
            </script>
        """))
        
        from IPython.display import Javascript
        display(Javascript("""
            window.addEventListener('arrow_key', function(event) {
                var kernel = IPython.notebook.kernel;
                var command = `handle_key(type('KeyEvent', (), {'key': '${event.detail}'})())`;
                kernel.execute(command);
            });
        """))
        
        time_slider.observe(update_time_label, names='value')
        
        # Initialize time label with correct format
        initial_hours = int(rel_times[self._current_time_index]) if rel_times else 0  # Convert to int
        initial_days = initial_hours // 24
        initial_remaining_hours = initial_hours % 24
        time_label = widgets.Label(
            value=f"{int(initial_days)}d {int(initial_remaining_hours)}h"  # Convert both to int
        )
        
        # Channel selection and image display
        channel_options = ['brightfield', 'fluorescence_green', 
                          'fluorescence_red', 'fluorescence_blue']
        
        def create_channel_view(channel_index, default_channel='brightfield'):
            dropdown = widgets.Dropdown(
                options=channel_options,
                value=self._widget_states[f'channel_{channel_index}'],  # Use stored channel
                description='Channel:',
                layout=widgets.Layout(width='200px')
            )
            
            # Store channel selection when changed
            def on_channel_change(change):
                self._widget_states[f'channel_{channel_index}'] = change.new
            dropdown.observe(on_channel_change, names='value')
            
            output = widgets.Output(layout=widgets.Layout(
                width='500px',
                height='500px',
            ))
            return widgets.VBox([dropdown, output])
        
        # Create image views with stored channels
        image_section = widgets.HBox([
            create_channel_view(1),
            create_channel_view(2)
        ], layout=widgets.Layout(justify_content='space-around'))
        
        def update_images(change=None):
            if not self._current_result or not self._current_well:
                return
            
            for channel_view in image_section.children:
                channel = channel_view.children[0].value
                output = channel_view.children[1]
                
                with output:
                    output.clear_output(wait=True)
                    try:
                        spheroid = self._current_result.spheroid_dict[self._current_well]
                        image = spheroid.spheroid_image_dict[time_points[time_slider.value]]
                        
                        # Handle different channel methods
                        if channel == 'brightfield':
                            img = image.brightfield()
                        elif channel.startswith('fluorescence_'):
                            color = channel.split('_')[1]
                            img = image.fluorescence(color)
                            # Convert BGR to RGB for all fluorescence channels
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        else:
                            raise ValueError(f"Unknown channel: {channel}")
                        
                        if img is not None:
                            fig = plt.figure(figsize=(8, 8))
                            plt.imshow(img, extent=[0, image.image_size[0], 0, image.image_size[1]])
                            
                            # Draw contour if checkbox is checked and contour exists
                            if show_contour.value and image.contour is not None:  # Check base contour first
                                contour = image.scaled_contour  # Only access scaled_contour if contour exists
                                plt.plot(contour[:, 0], contour[:, 1], 'w-', linewidth=3)
                            
                            # Add scale bar (300µm)
                            scalebar = AnchoredSizeBar(
                                plt.gca().transData,
                                300,  # 300 µm
                                '300 µm',
                                'lower center',
                                pad=0,
                                color='white',
                                frameon=False,
                                size_vertical=10,
                                borderpad=0.5
                            )
                            plt.gca().add_artist(scalebar)
                            
                            plt.axis('off')
                            plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
                            plt.show()
                        else:
                            print(f"No image data for {channel}")
                            
                    except Exception as e:
                        print(f"Error displaying image: {str(e)}")
        
        # Set up observers
        time_slider.observe(update_images, names='value')
        show_contour.observe(update_images, names='value')
        for channel_view in image_section.children:
            channel_view.children[0].observe(update_images, names='value')
        
        # Create controls section with time slider and contour checkbox
        controls_section = widgets.HBox([
            time_slider,
            time_label,
            show_contour
        ], layout=widgets.Layout(margin='10px 0', align_items='center'))
        
        # Initial image display
        update_images()
        
        # Segmentation controls (pass image_section so handlers can access current channel)
        segmentation_controls = self._create_segmentation_controls(image_section)
        
        return widgets.VBox([
            widgets.HTML(value="<h3 style='color: #2c3e50;'>Image Viewer</h3>"),
            controls_section,
            image_section,
            segmentation_controls
        ], layout=widgets.Layout(margin='20px 0'))

    def _create_segmentation_controls(self, image_section=None) -> widgets.VBox:
        """Create segmentation control panel.
        
        Args:
            image_section: Optional image section widget to access current channel selection
        """
        header = widgets.HTML(
            value="<h4 style='color: #2c3e50;'>Segmentation Options</h4>"
        )
        
        # Create input fields with stored values
        thresholding_input = widgets.FloatText(
            value=self._widget_states['thresholding_value'],
            step=0.1,
            layout=widgets.Layout(width='60px')
        )
        
        ai_input = widgets.FloatText(
            value=self._widget_states['ai_value'],
            step=0.05,
            layout=widgets.Layout(width='60px')
        )
        
        # Store input values when changed
        def on_thresholding_change(change):
            self._widget_states['thresholding_value'] = change.new
        
        def on_ai_change(change):
            self._widget_states['ai_value'] = change.new
            
        thresholding_input.observe(on_thresholding_change, names='value')
        ai_input.observe(on_ai_change, names='value')
        
        # Create buttons
        manual_btn = widgets.Button(
            description='Manual', 
            layout=widgets.Layout(width='100px')
        )
        
        thresholding_btn = widgets.Button(
            description='Thresholding',
            layout=widgets.Layout(width='100px')
        )
        
        ai_btn = widgets.Button(
            description='AI',
            layout=widgets.Layout(width='100px')
        )
        
        delete_btn = widgets.Button(
            description='Delete Contour',
            button_style='danger',
            layout=widgets.Layout(width='120px')
        )

        # Define button handlers
        def manual_segmentation(b):
            if not self._current_result or not self._current_well:
                return
            spheroid = self._current_result.spheroid_dict[self._current_well]
            image = spheroid.spheroid_image_dict[self._current_result.time_points_array[self._current_time_index]]
            
            # Get current channel from first image view dropdown
            if image_section and image_section.children and len(image_section.children) > 0:
                channel_view = image_section.children[0]
                if channel_view.children and len(channel_view.children) > 0:
                    current_channel = channel_view.children[0].value
                else:
                    current_channel = 'brightfield'
            else:
                current_channel = 'brightfield'
            
            try:
                # Use the segmentation_manual method
                image.segmentation_manual(current_channel)
                
                # Save contour to HDF5
                if image.contour is not None:
                    with h5py.File(self._current_result.hdf5_path, 'a') as hdf_file:
                        spheroid_image_group = hdf_file[image.hdf5_key]
                        if 'contour' in spheroid_image_group:
                            del spheroid_image_group['contour']
                        if 'touches_border' in spheroid_image_group.attrs:
                            del spheroid_image_group.attrs['touches_border']
                        spheroid_image_group.create_dataset('contour', data=image.contour)
                        spheroid_image_group.attrs['touches_border'] = image.contour_touches_border
                    
                    # Update metrics for current timepoint
                    rel_time = self._current_result.relative_time_array[self._current_time_index]
                    for metric_name in self._current_result.metric_dfs:
                        if metric_name in ['radius', 'area']:
                            self._current_result.metric_dfs[metric_name].loc[rel_time, self._current_well] = getattr(image, metric_name)
                
                # Update display
                self._update_result_view()
            except Exception as e:
                print(f"Manual segmentation failed: {str(e)}")

        def thresholding_segmentation(b):
            if not self._current_result or not self._current_well:
                return
            spheroid = self._current_result.spheroid_dict[self._current_well]
            image = spheroid.spheroid_image_dict[self._current_result.time_points_array[self._current_time_index]]

            # Get current channel from first image view dropdown
            if image_section and image_section.children and len(image_section.children) > 0:
                channel_view = image_section.children[0]
                if channel_view.children and len(channel_view.children) > 0:
                    current_channel = channel_view.children[0].value
                else:
                    current_channel = 'fluorescence_green'
            else:
                current_channel = 'fluorescence_green'

            try:
                # Use the segmentation() method which handles everything correctly
                image.segmentation(
                    methods=('thresholding', current_channel),
                    reconstruct_border=True,
                    border_margin=10,
                    threshold=thresholding_input.value
                )
                
                # Save contour to HDF5
                if image.contour is not None:
                    with h5py.File(self._current_result.hdf5_path, 'a') as hdf_file:
                        spheroid_image_group = hdf_file[image.hdf5_key]
                        if 'contour' in spheroid_image_group:
                            del spheroid_image_group['contour']
                        if 'touches_border' in spheroid_image_group.attrs:
                            del spheroid_image_group.attrs['touches_border']
                        spheroid_image_group.create_dataset('contour', data=image.contour)
                        spheroid_image_group.attrs['touches_border'] = image.contour_touches_border
                    
                    # Update metrics for current timepoint
                    rel_time = self._current_result.relative_time_array[self._current_time_index]
                    for metric_name in self._current_result.metric_dfs:
                        if metric_name in ['radius', 'area']:
                            self._current_result.metric_dfs[metric_name].loc[rel_time, self._current_well] = getattr(image, metric_name)
                
                # Update display
                self._update_result_view()
            except Exception as e:
                print(f"Thresholding segmentation failed: {str(e)}")

        def ai_segmentation(b):
            if not self._current_result or not self._current_well:
                return
            spheroid = self._current_result.spheroid_dict[self._current_well]
            image = spheroid.spheroid_image_dict[self._current_result.time_points_array[self._current_time_index]]
            
            # Get current channel from first image view dropdown
            if image_section and image_section.children and len(image_section.children) > 0:
                channel_view = image_section.children[0]
                if channel_view.children and len(channel_view.children) > 0:
                    current_channel = channel_view.children[0].value
                else:
                    current_channel = 'brightfield'
            else:
                current_channel = 'brightfield'
            
            try:
                # Use the segmentation() method which handles everything correctly
                image.segmentation(
                    methods=('ai', current_channel),
                    reconstruct_border=True,
                    border_margin=10,
                    confidence=ai_input.value
                )
                
                # Save contour to HDF5
                if image.contour is not None:
                    with h5py.File(self._current_result.hdf5_path, 'a') as hdf_file:
                        spheroid_image_group = hdf_file[image.hdf5_key]
                        if 'contour' in spheroid_image_group:
                            del spheroid_image_group['contour']
                        if 'touches_border' in spheroid_image_group.attrs:
                            del spheroid_image_group.attrs['touches_border']
                        spheroid_image_group.create_dataset('contour', data=image.contour)
                        spheroid_image_group.attrs['touches_border'] = image.contour_touches_border
                               
                    # Update metrics for current timepoint
                    rel_time = self._current_result.relative_time_array[self._current_time_index]
                    for metric_name in self._current_result.metric_dfs:
                        if metric_name in ['radius', 'area']:
                            self._current_result.metric_dfs[metric_name].loc[rel_time, self._current_well] = getattr(image, metric_name)
                
                # Update display
                self._update_result_view()
            except Exception as e:
                print(f"AI segmentation failed: {str(e)}")


        def delete_contour(b):
            if not self._current_result or not self._current_well:
                return
            spheroid = self._current_result.spheroid_dict[self._current_well]
            image = spheroid.spheroid_image_dict[self._current_result.time_points_array[self._current_time_index]]
            
            # Delete contour
            image.contour = None
            image.contour_touches_border = False
            
            # Remove contour from HDF5
            with h5py.File(self._current_result.hdf5_path, 'a') as hdf_file:
                spheroid_image_group = hdf_file[image.hdf5_key]
                if 'contour' in spheroid_image_group:
                    del spheroid_image_group['contour']
                if 'touches_border' in spheroid_image_group.attrs:
                    del spheroid_image_group.attrs['touches_border']
            
            # Update metrics for current timepoint with NaN
            rel_time = self._current_result.relative_time_array[self._current_time_index]
            for metric_name in self._current_result.metric_dfs:
                if metric_name in ['radius', 'area']:
                    self._current_result.metric_dfs[metric_name].loc[rel_time, self._current_well] = np.nan
            
            # Update display
            self._update_result_view()

        # Connect handlers to buttons
        manual_btn.on_click(manual_segmentation)
        thresholding_btn.on_click(thresholding_segmentation)
        ai_btn.on_click(ai_segmentation)
        delete_btn.on_click(delete_contour)

        # Layout the controls
        controls = widgets.HBox([
            manual_btn,
            widgets.HBox([thresholding_btn, thresholding_input]),
            widgets.HBox([ai_btn, ai_input]),
            delete_btn
        ], layout=widgets.Layout(justify_content='space-around'))

        return widgets.VBox([header, controls], 
                           layout=widgets.Layout(
                               margin='20px 0',
                               padding='10px',
                               border='1px solid #ccc',
                               border_radius='5px'
                           ))

    def _on_result_well_click(self, well: str):
        """Handle well selection in results view."""
        self._current_well = well
        self._update_result_view()

    def _create_analyses_tab(self) -> widgets.VBox:
        """Create analyses tab with analysis selection and visualization."""
        # Analysis selector
        analysis_dropdown = widgets.Dropdown(
            options=list(self.experiment.analyses_dict.keys()),
            description='Analysis:',
            layout=widgets.Layout(width='300px')
        )
        
        # Create analysis viewer
        analysis_viewer = widgets.Output()
        
        def on_analysis_change(change):
            analysis_viewer.clear_output()
            with analysis_viewer:
                if change.new:
                    self._current_analysis = self.experiment.analyses_dict[change.new]
                    # Display analysis results
                    self._display_analysis(self._current_analysis)
        
        analysis_dropdown.observe(on_analysis_change, names='value')
        
        return widgets.VBox([analysis_dropdown, analysis_viewer])

    def _create_plate_layout(self) -> widgets.VBox:
        """Create interactive plate layout visualization."""
        # Header
        header = widgets.HTML(
            value="<h3 style='color: #2c3e50;'>Plate Layout</h3>"
        )
        
        # Create grid of well buttons
        rows = 'ABCDEFGHIJKLMNOP'[:self.experiment.row]
        cols = range(1, self.experiment.col + 1)
        
        well_buttons = []
        for row in rows:
            row_buttons = []
            for col in cols:
                well = f"{row}{col}"
                btn = widgets.Button(
                    description=well,
                    layout=widgets.Layout(width='40px', height='40px'),
                    style=widgets.ButtonStyle(button_color=self._get_well_color(well))
                )
                btn.on_click(lambda b, well=well: self._on_well_click(well))
                row_buttons.append(btn)
            well_buttons.append(widgets.HBox(row_buttons))
        
        grid = widgets.VBox(well_buttons)
        
        # Add legend
        legend = self._create_plate_legend()
        
        return widgets.VBox([header, grid, legend])

    def _get_well_color(self, well: str, result=None) -> str:
        """Get color for well button based on well status.
        
        Args:
            well: Well identifier
            result: Specific result to check (if None, uses current result)
        """
        # Use provided result or fall back to current result
        check_result = result or self._current_result
        
        # Highlight well if:
        # 1. It's in Results tab (result is None) and matches current well
        # 2. It's in Overview tab (result is not None) and matches current well and result
        if well == self._current_well and (
            (result is None) or  # Results tab
            (result == self._current_result)  # Overview tab, matching result
        ):
            return '#FFD700'  # Gold
        
        # Check if well has data
        has_data = (check_result and 
                    well in check_result.spheroid_dict)
        
        if has_data:
            return '#ADD8E6'  # Light blue
        return '#F0F0F0'  # Light gray

    def _create_plate_legend(self) -> widgets.HBox:
        """Create legend for plate layout colors."""
        legend_items = [
            ('Data', '#ADD8E6'),
            ('Empty', '#F0F0F0')
        ]
        
        legend_widgets = []
        for label, color in legend_items:
            box = widgets.ColorPicker(
                concise=True,
                value=color,
                disabled=True,
                layout=widgets.Layout(
                    width='20px',      # Slightly larger
                    height='20px'      # Square shape
                )
            )
            text = widgets.Label(
                value=label,
                layout=widgets.Layout(
                    font_size='8px'    # Keep small font
                )
            )
            legend_widgets.extend([box, text])
        
        return widgets.HBox(
            legend_widgets,
            layout=widgets.Layout(
                margin='2px 0',
                justify_content='center'
            )
        )

    def _create_metrics_summary(self) -> widgets.VBox:
        """Create summary of available metrics."""
        if not self.experiment.results_dict:
            return widgets.HTML(value="<p>No metrics available</p>")
            
        metrics_html = ["<h3 style='color: #2c3e50;'>Available Metrics</h3>"]
        
        for result_name, result in self.experiment.results_dict.items():
            if hasattr(result, 'metric_dfs'):
                metrics = list(result.metric_dfs.keys())
                metrics_html.append(f"<p><b>{result_name}:</b> {', '.join(metrics)}</p>")
        
        return widgets.HTML(value="\n".join(metrics_html))

    def _on_well_click(self, well: str):
        """Handle well button clicks."""
        self._current_well = well
        
        # Switch to Results tab if we have data for this well
        if any(well in result.spheroid_dict 
               for result in self.experiment.results_dict.values()):
            self._viewer.selected_index = 1  # Results tab
            
            # Update well selection in results view if possible
            if hasattr(self, '_current_result') and self._current_result:
                # Update well selection in result viewer
                pass

    def _display_analysis(self, analysis) -> None:
        """Display analysis results with interactive plots."""
        # Create metric plots
        if hasattr(analysis, 'metrics'):
            for metric_name, metric_data in analysis.metrics.items():
                plt.figure(figsize=(10, 6))
                plt.plot(metric_data)
                plt.title(f"{metric_name} Analysis")
                plt.xlabel('Time')
                plt.ylabel(metric_name)
                plt.show()

    def plot_metrics(self, metrics: list[str] | None = None):
        """Plot selected metrics across all results.
        
        Args:
            metrics: List of metric names to plot. If None, plots all available metrics.
        """
        if not metrics:
            metrics = ['radius', 'area']  # Default metrics
            
        for metric in metrics:
            plt.figure(figsize=(10, 6))
            for result_name, result in self.experiment.results_dict.items():
                try:
                    df = result.metric(metric, mean=True)
                    plt.plot(df.index/24, df, label=result_name)
                except:
                    continue
            
            plt.xlabel('Time [days]')
            plt.ylabel(metric)
            plt.title(f'{metric} over time')
            plt.legend()
            plt.show()

    def _update_result_view(self):
        """Update all components of the result view."""
        if not self._current_result or not self._current_well or not self._result_view:
            return
            
        # Update result info
        self._result_view['update_info']()
        
        # Update plate layout
        self._result_view['plate_layout'].clear_output()
        with self._result_view['plate_layout']:
            display(self._create_result_plate_layout())
            
        # Update dummy plot
        self._result_view['update_plot']()
        
        # Update image viewer
        self._result_view['image_viewer'].clear_output()
        with self._result_view['image_viewer']:
            display(self._create_image_viewer(2))  # Default to 2 image panels

    def _on_overview_well_click(self, well: str, result):
        """Handle well clicks in overview tab."""
        # Set the current result to the one containing this well
        if well in result.spheroid_dict:
            self._current_result = result
            self._current_well = well
            
            # Get the Results tab content
            results_tab = self._viewer.children[1]  # Index 1 is Results tab
            
            # Find and update the result dropdown in the header section
            header_section = results_tab.children[0]  # First VBox in Results tab
            dropdown_box = header_section.children[1]  # HBox containing the dropdown
            result_dropdown = dropdown_box.children[1]  # The actual dropdown widget
            
            # Update dropdown value to match current result
            result_dropdown.value = result.name
            
            # Force a complete update of the Results view
            def complete_update():
                self._update_result_view()
                # Update plot and images explicitly
                if self._result_view:
                    self._result_view['update_plot']()
                    with self._result_view['image_viewer']:
                        self._result_view['image_viewer'].clear_output(wait=True)
                        display(self._create_image_viewer(2))
            
            # Switch to Results tab and update after a short delay
            self._viewer.selected_index = 1
            import time
            time.sleep(0.1)  # Small delay to ensure tab switch is complete
            complete_update()