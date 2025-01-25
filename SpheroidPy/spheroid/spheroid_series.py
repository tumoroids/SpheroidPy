from datetime import datetime

import cv2
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
import numpy as np
from tqdm import tqdm
from matplotlib.colors import Normalize
import h5py

from SpheroidPy.spheroid.spheroid_image import SpheroidImage
from SpheroidPy.spheroid.time_period import TimePeriod

class SpheroidSeries:
    """A series of spheroid images over time.
    
    Attributes:
        name: Name of this spheroid series
        spheroid_image_dict: Dictionary mapping timepoints to SpheroidImage instances
        time_periods: Dictionary mapping names to TimePeriod instances
        _result: reference to the result of the last calculation
    """
    name: str
    spheroid_image_dict: dict   # {timepoint: SpheroidImage}
    time_periods: dict[str, TimePeriod] = {}  # {name: TimePeriod}

    def __init__(self, name: str):
        self.name = name
        self.spheroid_image_dict = {}
        self.time_periods = {}
        self._result = None

    @property
    def hdf5_path(self) -> str:
        """Get HDF5 file path from first spheroid image."""
        if not self.spheroid_image_dict:
            raise ValueError("No spheroid images available")
        return next(iter(self.spheroid_image_dict.values())).hdf5_path

    @property
    def hdf5_key(self) -> str:
        """Get HDF5 key from first spheroid image."""
        if not self.spheroid_image_dict:
            raise ValueError("No spheroid images available")
        # Get the base key (remove the timepoint part)
        full_key = next(iter(self.spheroid_image_dict.values())).hdf5_key
        return '/'.join(full_key.split('/')[:-1])

    def add_spheroid_image(self, spheroid: SpheroidImage, timepoint: datetime):
        """Add a spheroid image at a specific timepoint.
        
        Args:
            spheroid: SpheroidImage instance to add
            timepoint: When the image was taken
        """
        self.spheroid_image_dict[timepoint] = spheroid
        self.spheroid_image_dict = dict(sorted(self.spheroid_image_dict.items()))

    def save_time_periods_to_hdf5(self):
        """Save time periods to HDF5 file."""
        with h5py.File(self.hdf5_path, 'a') as hdf_file:
            series_group = hdf_file[self.hdf5_key]
            
            # Remove existing time periods if any
            if 'time_periods' in series_group:
                del series_group['time_periods']
                
            # Create time periods group
            periods_group = series_group.create_group('time_periods')
            
            # Save each time period
            for name, period in self.time_periods.items():
                period_group = periods_group.create_group(name)
                period_group.attrs['start_time'] = period.start_time.isoformat()
                period_group.attrs['end_time'] = period.end_time.isoformat()
                period_group.attrs['created_at'] = period.created_at.isoformat()
                period_group.attrs['modified_at'] = period.modified_at.isoformat()
                if period.description:
                    period_group.attrs['description'] = period.description

    def load_time_periods_from_hdf5(self):
        """Load time periods from HDF5 file."""
        with h5py.File(self.hdf5_path, 'r') as hdf_file:
            series_group = hdf_file[self.hdf5_key]
            self.time_periods = {}
            
            if 'time_periods' in series_group:
                periods_group = series_group['time_periods']
                
                for name in periods_group:
                    period_group = periods_group[name]
                    
                    # Load time period attributes
                    start_time = datetime.fromisoformat(period_group.attrs['start_time'])
                    end_time = datetime.fromisoformat(period_group.attrs['end_time'])
                    created_at = datetime.fromisoformat(period_group.attrs['created_at'])
                    modified_at = datetime.fromisoformat(period_group.attrs['modified_at'])
                    description = period_group.attrs.get('description', None)
                    
                    # Create TimePeriod instance
                    period = TimePeriod(
                        name=name,
                        start_time=start_time,
                        end_time=end_time,
                        description=description
                    )
                    period.created_at = created_at
                    period.modified_at = modified_at
                    
                    self.time_periods[name] = period

    def time_period(self, name: str,
                   start_time: str | datetime | None = None,
                   end_time: str | datetime | None = None,
                   description: str | None = None) -> TimePeriod:
        """Smart method to add or update a time period.
        
        If a time period with the given name exists:
            - Updates the existing period
        If no time period exists:
            - Creates a new time period
            
        Args:
            name: Unique name for this time period
            start_time: Start datetime, or None to use first timepoint
            end_time: End datetime, or None to use last timepoint
            description: Optional description
            
        Returns:
            Created or updated TimePeriod
        """
        if name in self.time_periods:
            return self.update_time_period(
                name=name,
                start_time=start_time,
                end_time=end_time,
                description=description
            )
        else:
            return self.add_time_period(
                name=name,
                start_time=start_time,
                end_time=end_time,
                description=description
            )

    def add_time_period(self, name: str,
                       start_time: str | datetime | None = None,
                       end_time: str | datetime | None = None,
                       description: str | None = None) -> TimePeriod:
        """Add a new time period.
        
        Args:
            name: Unique name for this time period
            start_time: Start datetime, or None to use first timepoint
            end_time: End datetime, or None to use last timepoint
            description: Optional description
            
        Returns:
            Created TimePeriod
        """
        # Convert string times to datetime if needed
        if isinstance(start_time, str):
            start_time = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
        if isinstance(end_time, str):
            end_time = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')
            
        # Use first/last timepoint if not specified
        timepoints = sorted(self.spheroid_image_dict.keys())
        if not timepoints:
            raise ValueError("No timepoints available")
            
        if start_time is None:
            start_time = datetime.strptime(timepoints[0], '%Y-%m-%d %H:%M:%S')
        if end_time is None:
            end_time = datetime.strptime(timepoints[-1], '%Y-%m-%d %H:%M:%S')
            
        # Create time period
        time_period = TimePeriod(
            name=name,
            start_time=start_time,
            end_time=end_time,
            description=description
        )
                
        self.time_periods[name] = time_period
        return time_period
        
    def update_time_period(self, name: str,
                          start_time: str | datetime | None = None,
                          end_time: str | datetime | None = None,
                          description: str | None = None) -> TimePeriod:
        """Update an existing time period."""
        if name not in self.time_periods:
            raise KeyError(f"No time period found with name '{name}'")
            
        time_period = self.time_periods[name]
        
        if isinstance(start_time, str):
            start_time = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
        if isinstance(end_time, str):
            end_time = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')
            
        time_period.update_time(start_time, end_time)
        if description:
            time_period.description = description
            
        return time_period
            
    def remove_time_period(self, name: str):
        """Remove a time period."""
        if name in self.time_periods:
            del self.time_periods[name]
            
    def get_images_in_period(self, name: str) -> dict:
        """Get spheroid images within a time period."""
        if name not in self.time_periods:
            raise KeyError(f"No time period found with name '{name}'")
            
        time_period = self.time_periods[name]
        return {
            date: image 
            for date, image in self.spheroid_image_dict.items()
            if time_period.start_time <= datetime.strptime(date, '%Y-%m-%d %H:%M:%S') <= time_period.end_time
        }

    def export_video(self, channel: str = 'brightfield', overlay_channels: list[str] = [], increases: list[str] = [], video_name: str = 'test.mp4', scalebar: bool = True, fps: int = 5, contour: bool = True, time: bool = True, exclude_out_of_contour: bool = True, diffusion_dict: dict = {}):
        """
        Exports the spheroid series images to a video.

        :param parent_folder:
        :param channel: Channel to export (default: brightfield)
        :param overlay: other channels/contours/fields, that get overlayed (default: none)
        :param format: format of the video (default: ???)
        :return:
        """
        # Initialisierung
        video = None
        time_0 = None

        # Iteriere durch die Zeitstempel und Bilder
        for i, (time_key, spheroid) in enumerate(tqdm(self.spheroid_image_dict.items(), desc="Processing Images")):
            # Lade das Bild basierend auf dem Kanal
            spheroid = self.spheroid_image_dict[time_key]
            base_image_array = cv2.imread(str(spheroid.image_path_dict[channel]))[1:-1]

            if base_image_array is None:
                print(f"Warnung: Bild konnte nicht geladen werden: {spheroid.image_path_dict[channel]}")
                continue

            # Konvertiere den Zeitstempel in ein datetime-Objekt
            time_key_dt = datetime.strptime(time_key, "%Y-%m-%d %H:%M:%S")

            # Initialisiere das Video beim ersten Durchlauf
            if video is None:
                height, width = base_image_array.shape[:2]
                video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))
                time_0 = time_key_dt

            # Erstelle eine Kopie des Basisbilds für Overlays
            combined_image = base_image_array.copy() #todo:ändern!!!

            if time:
                # Berechne die Zeitdifferenz
                time_diff = time_key_dt - time_0
                days = time_diff.days
                hours = time_diff.seconds // 3600
                time_difference_in_hours_and_days = f"{days}d {hours}h"

                # Füge den Zeitstempel als Text ins Bild ein
                cv2.putText(combined_image, time_difference_in_hours_and_days, (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 3.5, (255, 255, 255), 2)

            if contour and spheroid.contour is not None:
                # Mit cv2.drawContours zeichnen (grün, Linienstärke 2)
                cv2.drawContours(combined_image, [spheroid.contour.reshape((-1, 1, 2)).astype(np.int32)], -1, (255, 255, 255), 2)

            ''' Overlay '''
            color = {'red':(0,0,255), 'green':(0,255,0), 'blue':(255,0,0)}
            # Iteriere durch die Overlay-Kanäle
            for overlay_channel, increase in zip(overlay_channels, increases):
                # in case of fluorscence
                if overlay_channel.split('_')[0]=='fluorescence':
                    # Lade das Overlay-Bild (Fluoreszenzkanal)
                    try:
                        overlay_image = cv2.imread(str(spheroid.image_path_dict[overlay_channel]), cv2.IMREAD_GRAYSCALE)[:len(combined_image)]
                    except:
                        print(f"Warnung: Overlay-Bild konnte nicht geladen werden: {overlay_channel}")
                        continue

                    # Erstelle ein farbiges Overlay basierend auf der angegebenen Farbe
                    overlay_colored = np.stack( [np.zeros_like(overlay_image, dtype=np.uint8)] * 3, axis=-1)
                    for c, value in enumerate(color[overlay_channel.split('_')[1]]):
                        overlay_colored[:, :, c] = (overlay_image * value).astype(np.uint8)
                        overlay_colored[:, :, c] = np.clip(overlay_colored[:, :, c], 0, 255)

                        # fluoreszenz außerhalb von contour ausblenden
                        if exclude_out_of_contour==True and spheroid.contour is not None:
                            # Maske erstellen (gleiche Größe wie das Bild)
                            mask = np.zeros(overlay_colored[:, :, c].shape, dtype=np.uint8)
                            # Kontur auf der Maske zeichnen (gefüllt)
                            cv2.drawContours(mask, [spheroid.contour.reshape((-1, 1, 2)).astype(np.int32)], -1, 255, thickness=cv2.FILLED)

                            # Maske auf das Bild anwenden
                            overlay_colored[:, :, c] = cv2.bitwise_and(overlay_colored[:, :, c], overlay_colored[:, :, c], mask=mask)


                    # Kombiniere das Overlay-Bild mit dem Basisbild
                    alpha = overlay_image / 255.0 * increase  # Transparenz basierend auf Helligkeit
                    alpha = np.clip(alpha, 0, 1)

                    for c in range(3):
                        #combined_image[:, :, c] = (1 - alpha) * combined_image[:, :, c] #+ alpha * overlay_colored[:, :, c]
                        combined_image[:, :, c] = (1 - alpha) * combined_image[:, :,c] + alpha * overlay_colored[:, :, c]

                if overlay_channel == 'diffusion' and spheroid.contour is not None:
                    # solve Diffusion PDE
                    points, triangles, solution = spheroid.diffusion_stationary(diffusion_rate=diffusion_dict['diffusion_rate'], reaction_rate=diffusion_dict['reaction_rate'], boundary_value=diffusion_dict['boundary_value'], plot_full=False, plot=False, accuracy=30)

                    # Create a figure without axes
                    fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
                    ax.axis('off')  # Turn off axes
                    fig.tight_layout(pad=0)  # Keine Ränder

                    ax.imshow(cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB))  # Konvertiere ARGB zu BGRA
                    if 'cmap' in diffusion_dict:
                        ax.tricontourf(points[:, 0], points[:, 1], triangles, solution, levels=100, cmap=diffusion_dict['cmap'], alpha=.1*increase,
                                       norm=Normalize(vmin=0, vmax=diffusion_dict['boundary_value'], clip=False))
                    if 'contour_line_value' in diffusion_dict:
                        ax.tricontour(points[:, 0], points[:, 1], triangles, solution,
                                                        levels=[diffusion_dict['contour_line_value']], colors='blue', linestyle='dotted',
                                                        linewidths=1)
                        if 'contour_line_name' in diffusion_dict:
                            ax.plot([], [], color='blue', linestyle='solid', linewidth=1, label=diffusion_dict['contour_line_name'])
                            ax.legend(prop={'size': 20})

                    plt.xlim(0, width)
                    plt.ylim(height, 0)

                    # Matplotlib-Plot in ein OpenCV-kompatibles Bild umwandeln
                    fig.canvas.draw()
                    plot_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                    plot_image = plot_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                    combined_image = cv2.cvtColor(plot_image, cv2.COLOR_RGB2BGR)

            if scalebar:
                # Länge und Breite der Skalierungsleiste definieren
                scale_bar_length = int(width/spheroid.image_size[0]*300)  # Länge in Pixeln (entspricht z.B. 300µm)
                scale_bar_thickness = 15  # Dicke der Skalierungsleiste

                # Position der Skalierungsleiste festlegen
                x_start = 50  # Abstand vom linken Rand
                y_start = height - 50  # Abstand vom unteren Rand

                # Rechteck zeichnen (Skalierungsleiste)
                x_end = x_start + scale_bar_length
                y_end = y_start - scale_bar_thickness
                cv2.rectangle(combined_image, (x_start, y_start), (x_end, y_end), (255, 255, 255), -1)  # Weiße Leiste

                # Beschriftung hinzufügen
                # -*- coding: utf-8 -*-
                scale_text = "300 um"  # Beispieltext
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.5
                font_thickness = 2
                text_size = cv2.getTextSize(scale_text, font, font_scale, font_thickness)[0]
                text_x = x_start + (scale_bar_length - text_size[0]) // 2
                text_y = y_start - 30
                cv2.putText(combined_image, scale_text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)

            # Schreibe das Bild ins Video
            video.write(combined_image)

        # Beende das Video und gebe Ressourcen frei
        if video is not None:
            print("Video abgeschlossen. Datei wurde gespeichert unter:", video_name)
            video.release()
        else:
            print("Kein Video erstellt. Möglicherweise wurden keine Bilder geladen.")

    def __repr__(self):
        return self.name + (f'\n> Number of Timepoints: {len(self.spheroid_image_dict.keys())}'
                            f'\n> Channels: {list(self.spheroid_image_dict.values())[4].image_path_dict.keys()}\n\n')