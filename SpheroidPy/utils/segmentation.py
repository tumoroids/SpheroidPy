import tkinter as tk
from tkinter import ttk
import numpy as np
from PIL import Image, ImageTk, ImageDraw
import cv2

from scipy.ndimage import binary_closing
from skimage.morphology import convex_hull_image

class ManualSegmentation:
    def __init__(self, root, images: dict | None = None, initial_channel: str = 'brightfield'):
        """
        images: dict with optional keys: 'brightfield', 'green', 'red', 'blue'
                values are numpy arrays (OpenCV BGR or grayscale); sizes must match.
        initial_channel: which channel to show first
        """
        self.root = root
        self.images = images or {}
        # Determine original size from provided images or fallback
        if 'brightfield' in self.images and self.images['brightfield'] is not None:
            h0, w0 = self.images['brightfield'].shape[:2]
        else:
            # fallback
            h0, w0 = 400, 400

        # Resize images to max width 500 px (keeping aspect ratio)
        max_w = 500
        if w0 > max_w:
            self.resize_ratio = max_w / float(w0)
        else:
            self.resize_ratio = 1.0

        self.w = int(round(w0 * self.resize_ratio))
        self.h = int(round(h0 * self.resize_ratio))

        # Apply resizing to provided images so UI draws smaller canvas
        if self.resize_ratio != 1.0:
            resized = {}
            for k, img in self.images.items():
                if img is None:
                    resized[k] = None
                    continue
                resized[k] = cv2.resize(img, (self.w, self.h), interpolation=cv2.INTER_AREA)
            self.images = resized
        
        self.root.title("SpheroidPy")
        self.root.configure(bg='#f0f2f5') # Heller Hintergrund für das Hauptfenster

        self.last_x, self.last_y = None, None
        
        # Derived outputs
        self.contour = None
        self.contour_touches_border = None
        # completion flag for controlled shutdown
        self._done = tk.BooleanVar(self.root, value=False)

        # initial background
        self.background_img = self._make_background(initial_channel)
        
        # Maske für die finale, ausgefüllte Form
        self.mask_img = Image.new('L', (self.w, self.h), 0)
        self.draw_mask = ImageDraw.Draw(self.mask_img)

        # Eigene Ebene für die Linie, die der Nutzer gerade zeichnet
        self.line_img = Image.new('L', (self.w, self.h), 0)
        self.draw_line = ImageDraw.Draw(self.line_img)

        self.composite_tk = None
        
        self._setup_widgets()
        self._update_composite()

    def _setup_widgets(self):
        # Frame für den Titel
        title_frame = tk.Frame(self.root, bg='#f0f2f5')
        title_frame.pack(pady=(10, 5), fill='x', padx=20)

        # Überschrift
        lbl_title = tk.Label(title_frame, text="Manual Segmentation", font=('Arial', 20, 'bold'), bg='#f0f2f5', fg='#D97706') # Orange Farbe
        lbl_title.pack(anchor='w')

        # Sub-Überschrift
        lbl_subtitle = tk.Label(title_frame, text="Draw Contour at Border of Spheroid", font=('Arial', 12), bg='#f0f2f5', fg='#6B7280') # Graue Farbe
        lbl_subtitle.pack(anchor='w')

        # Canvas zum Zeichnen
        self.canvas = tk.Canvas(self.root, width=self.w, height=self.h)
        self.canvas.pack(padx=20, pady=(0, 10))

        self.canvas.bind("<ButtonPress-1>", self._start_draw)
        self.canvas.bind("<B1-Motion>", self._draw)
        self.canvas.bind("<ButtonRelease-1>", self._stop_draw)

        # Frame für die Steuerelemente (Buttons, Dropdown)
        control_frame = tk.Frame(self.root, bg='#f0f2f5')
        control_frame.pack(fill='x', pady=10, padx=20)

        # Dropdown-Menü
        self.dropdown_var = tk.StringVar(value='brightfield')
        dropdown = ttk.Combobox(
            control_frame, 
            textvariable=self.dropdown_var,
            values=['brightfield', 'green', 'red', 'blue'], 
            state='readonly'
        )
        dropdown.grid(row=0, column=0, sticky='ew')
        dropdown.bind('<<ComboboxSelected>>', self._update_background)

        # Button zum Löschen der Maske
        delete_button = tk.Button(control_frame, text="delete", command=self._clear_canvas)
        delete_button.grid(row=0, column=1, sticky='ew', padx=(10, 0))

        # Button zum Exportieren und Vervollständigen
        complete_button = tk.Button(control_frame, text="complete", command=self.export_and_complete_mask)
        complete_button.grid(row=0, column=2, sticky='ew', padx=(10, 0))

        # Grid-Konfiguration für responsive Spalten
        control_frame.grid_columnconfigure(0, weight=2) # Dropdown bekommt mehr Platz
        control_frame.grid_columnconfigure(1, weight=1)
        control_frame.grid_columnconfigure(2, weight=1)
        
    def _clear_canvas(self):
        """ Löscht die gezeichnete Maske und die gezeichnete Linie. """
        #print("Maske wird gelöscht...")
        self.mask_img = Image.new('L', (self.w, self.h), 0)
        self.draw_mask = ImageDraw.Draw(self.mask_img)
        self.line_img = Image.new('L', (self.w, self.h), 0)
        self.draw_line = ImageDraw.Draw(self.line_img)
        self._update_composite()

    def _make_background(self, mode):
        # Build RGB PIL image from provided data; fallback to neutral canvas
        if mode == 'brightfield' and 'brightfield' in self.images and self.images['brightfield'] is not None:
            img = self.images['brightfield']
            if img.ndim == 2:
                rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb)

        if mode in ['green', 'red', 'blue'] and mode in self.images and self.images[mode] is not None:
            img = self.images[mode]
            if img.ndim == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
            # map grayscale into colored RGB channel for visibility
            rgb = np.zeros((self.h, self.w, 3), dtype=np.uint8)
            idx = {'red': 0, 'green': 1, 'blue': 2}[mode]
            rgb[..., idx] = gray
            return Image.fromarray(rgb)

        # neutral fallback
        return Image.new('RGB', (self.w, self.h), (240, 240, 240))

    def _update_composite(self):
        # Basishintergrund nach RGBA konvertieren, um Transparenz zu ermöglichen
        bg_rgba = self.background_img.copy().convert('RGBA')
        
        # Ein komplett transparentes Overlay erstellen, auf dem wir malen
        overlay = Image.new('RGBA', (self.w, self.h), (0, 0, 0, 0))

        # Helles Orange mit ~50% Transparenz für die Füllung
        fill_color_rgba = (254, 243, 199, 128) # Entspricht #FEF3C7 mit Alpha
        # Dunkles, sattes Orange (opak) für die Zeichenlinie
        line_color_rgba = (180, 83, 9, 255) # Entspricht #B45309 mit Alpha

        # Die Füllfarbe auf das Overlay "pasten", gesteuert durch die Füllmaske
        overlay.paste(fill_color_rgba, (0, 0), self.mask_img)
        # Die Linienfarbe auf das Overlay "pasten", gesteuert durch die Linienmaske
        overlay.paste(line_color_rgba, (0, 0), self.line_img)

        # Das Overlay mit dem Hintergrundbild alpha-blenden
        composite_img = Image.alpha_composite(bg_rgba, overlay)
        
        # Für Tkinter zurück nach RGB konvertieren, da es sonst Probleme geben kann
        composite_img_rgb = composite_img.convert('RGB')

        self.composite_tk = ImageTk.PhotoImage(composite_img_rgb)
        self.canvas.create_image(0, 0, anchor='nw', image=self.composite_tk)

    def _update_background(self, event=None):
        mode = self.dropdown_var.get()
        self.background_img = self._make_background(mode)
        self._update_composite()

    def _start_draw(self, event):
        self.last_x, self.last_y = event.x, event.y

    def _draw(self, event):
        if self.last_x is not None:
            # Auf der separaten Ebene für die Linie zeichnen
            self.draw_line.line([self.last_x, self.last_y, event.x, event.y], fill=255, width=3)
            self.last_x, self.last_y = event.x, event.y
            self._update_composite()

    def _stop_draw(self, event):
        self.last_x, self.last_y = None, None

    def export_and_complete_mask(self):
        """
        Vervollständigt die gezeichnete Kontur und gibt die Maske aus.
        """
        #print("Vervollständige Kontur...")
        # Die gezeichnete Linie als Basis für die Vervollständigung nehmen
        mask_array = np.array(self.line_img) > 0

        if not np.any(mask_array):
            print("Keine Linie gezeichnet. Nichts zu vervollständigen.")
            return

        # Kontur vervollständigen
        closed_mask = binary_closing(mask_array, structure=np.ones((20, 20)))
        completed_mask_array = convex_hull_image(closed_mask)
        completed_mask_pil = Image.fromarray(completed_mask_array.astype(np.uint8) * 255)

        # Das Ergebnis in die Maske für die Füllung schreiben
        self.mask_img = completed_mask_pil
        self.draw_mask = ImageDraw.Draw(self.mask_img)

        # Die ursprüngliche Zeichenlinie löschen, damit nur die Füllung bleibt
        self.line_img = Image.new('L', (self.w, self.h), 0)
        self.draw_line = ImageDraw.Draw(self.line_img)

        # Das Kompositbild aktualisieren, um das Ergebnis anzuzeigen
        self._update_composite()

        # Kontur berechnen und Randkontakt prüfen
        mask_uint8 = (completed_mask_array.astype(np.uint8) * 255)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            main = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
            # scale contour back to original image coordinates if we resized
            if hasattr(self, 'resize_ratio') and self.resize_ratio not in (None, 0, 1.0):
                scale_back = 1.0 / float(self.resize_ratio)
                main = main * scale_back
            self.contour = main
            x = main[:, 0]
            y = main[:, 1]
            # Evaluate border contact relative to the displayed canvas bounds
            self.contour_touches_border = bool(
                np.any(x <= 0) or np.any(y <= 0) or np.any(x >= (self.w - 1) / max(1e-9, self.resize_ratio)) or np.any(y >= (self.h - 1) / max(1e-9, self.resize_ratio))
            )
        else:
            self.contour = None
            self.contour_touches_border = None

        drawn_pixels = np.count_nonzero(completed_mask_array)
        #print(f"Vervollständigte Maske: Shape={completed_mask_array.shape}, Pixel gezeichnet={drawn_pixels}")

        # Fenster sauber schließen: zuerst quit, dann destroy asynchron
        # Do not close automatically; keep window open like the example.
        # User can close the window manually when done reviewing the result.

