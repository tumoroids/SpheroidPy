import tkinter as tk
from tkinter import ttk
import numpy as np
import cv2
from PIL import Image, ImageTk


class ImageHistogramViewer:
    """
    ImageHistogramViewer
    ====================
    Displays a grayscale image alongside a vertical histogram panel.

    Histogram features
    ------------------
    - Pixel intensity on the Y-axis (0 = black at bottom, 255 = white at top)
    - Frequency (count) on the X-axis, √-scaled so tall bins don't dominate
    - Light-gray band highlights the active spectrum range [low, high]
    - Red horizontal line marks the mean intensity
    - Small arrowhead handles on every interactive line

    Interactive controls (mouse on histogram)
    -----------------------------------------
    - Drag the **low / high boundary** of the gray band
          → range is adjusted **symmetrically around the mean**
            (new_half = |dragged_edge - mean|; both edges mirror each other)
    - Drag the **red mean line**
          → shifts the mean while keeping the half-width constant

    Image display
    -------------
      pixel <= low   → 0  (black)
      pixel >= high  → highlighted in RED (overexposed)
      otherwise      → linear stretch to [0, 255]

    Parameters
    ----------
    root :
        Tkinter root or Toplevel window.
    images : dict | None
        Optional dict with grayscale/BGR numpy arrays.
        Keys can be arbitrary channel names; values are numpy arrays
        (grayscale or BGR uint8). If None, synthetic test images are generated.
    """

    # Layout constants
    HIST_W = 160      # width of the histogram canvas (px)
    IMG_MAX_W = 420   # max display width of the image (px)
    HANDLE_TOL = 9    # mouse hit-test tolerance in canvas pixels

    def __init__(self, root, images: dict | None = None, initial_channel: str | None = None):
        self.root = root
        self.root.title("SpheroidPy - Histogram Viewer")
        self.root.configure(bg="#f0f2f5")

        raw = images or {}
        self._channel_images: dict[str, np.ndarray] = {}
        # Derive channel list from provided images (stable order)
        if raw:
            self.CHANNELS = tuple(raw.keys())
        else:
            self.CHANNELS = ("brightfield", "fluorescence_green")

        for ch in self.CHANNELS:
            img = raw.get(ch, None)
            if img is None:
                img = self._generate_test_image(ch)
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            self._channel_images[ch] = img.astype(np.uint8)

        # Active channel
        if initial_channel is not None and initial_channel in self.CHANNELS:
            start_channel = initial_channel
        else:
            start_channel = self.CHANNELS[0]
        self._channel_var = tk.StringVar(value=start_channel)

        # Load first channel
        self._load_channel(start_channel)

        # Drag state
        self._drag_mode: str | None = None
        self._img_tk = None  # keep reference to avoid GC

        self._setup_widgets()
        self._update_display()

    # ------------------------------------------------------------------ #
    #  Channel loading
    # ------------------------------------------------------------------ #

    def _load_channel(self, channel: str):
        """Resize, compute histogram, and set initial mean/range for a channel."""
        raw = self._channel_images[channel]

        h0, w0 = raw.shape[:2]
        ratio = min(1.0, self.IMG_MAX_W / float(w0))
        self.w = max(1, int(round(w0 * ratio)))
        self.h = max(1, int(round(h0 * ratio)))

        if ratio < 1.0:
            self.gray_orig = cv2.resize(
                raw, (self.w, self.h), interpolation=cv2.INTER_AREA
            )
        else:
            self.gray_orig = raw.copy()

        self.hist = np.bincount(self.gray_orig.flatten(), minlength=256).astype(float)

        # sqrt-scale: compresses tall dominant bins so the shape is more readable
        self._hist_scaled = np.sqrt(self.hist)

        self.mean_val = float(np.mean(self.gray_orig))
        half = max(10.0, 2.0 * float(np.std(self.gray_orig)))
        self.low = float(np.clip(self.mean_val - half, 0, 253))
        self.high = float(np.clip(self.mean_val + half, 2, 255))

        self._init_mean = self.mean_val
        self._init_low = self.low
        self._init_high = self.high

    # ------------------------------------------------------------------ #
    #  Widget layout
    # ------------------------------------------------------------------ #

    def _setup_widgets(self):
        # Header
        hdr = tk.Frame(self.root, bg="#f0f2f5")
        hdr.pack(pady=(12, 4), fill="x", padx=20)
        tk.Label(
            hdr,
            text="Image Histogram Viewer",
            font=("Arial", 20, "bold"),
            bg="#f0f2f5",
            fg="#D97706",
        ).pack(anchor="w")
        tk.Label(
            hdr,
            text=(
                "Drag band edges (symmetric) or the red mean line  |  "
                "Red pixels = overexposed"
            ),
            font=("Arial", 11),
            bg="#f0f2f5",
            fg="#6B7280",
        ).pack(anchor="w")

        # Body: image | gap | histogram
        body = tk.Frame(self.root, bg="#f0f2f5")
        body.pack(padx=20, pady=(0, 6))

        self.img_canvas = tk.Canvas(body, width=self.w, height=self.h, highlightthickness=0)
        self.img_canvas.pack(side="left")

        tk.Frame(body, width=14, bg="#f0f2f5").pack(side="left")

        self.hist_canvas = tk.Canvas(
            body,
            width=self.HIST_W,
            height=self.h,
            bg="white",
            highlightthickness=1,
            highlightbackground="#D1D5DB",
            cursor="sb_v_double_arrow",
        )
        self.hist_canvas.pack(side="left")

        self.hist_canvas.bind("<ButtonPress-1>", self._hist_press)
        self.hist_canvas.bind("<B1-Motion>", self._hist_drag)
        self.hist_canvas.bind("<ButtonRelease-1>", self._hist_release)

        # Status line
        self.info_var = tk.StringVar()
        tk.Label(
            self.root,
            textvariable=self.info_var,
            bg="#f0f2f5",
            fg="#6B7280",
            font=("Arial", 10),
        ).pack(pady=(2, 0))

        # Controls row: [Channel dropdown]  [Reset]  [Close]
        ctrl = tk.Frame(self.root, bg="#f0f2f5")
        ctrl.pack(pady=(6, 14))

        # Channel dropdown (left)
        ch_dropdown = ttk.Combobox(
            ctrl,
            textvariable=self._channel_var,
            values=list(self.CHANNELS),
            state="readonly",
            width=20,
            font=("Arial", 10),
        )
        ch_dropdown.grid(row=0, column=0, padx=(0, 8), ipady=2)
        ch_dropdown.bind("<<ComboboxSelected>>", self._on_channel_change)

        # Reset button (center)
        tk.Button(
            ctrl,
            text="Reset",
            command=self._reset,
            font=("Arial", 10),
            padx=14,
            pady=2,
            bg="#E5E7EB",
            activebackground="#D1D5DB",
            relief="flat",
        ).grid(row=0, column=1, padx=(0, 8))

        # Close button (right)
        tk.Button(
            ctrl,
            text="Close",
            command=self.root.destroy,
            font=("Arial", 10, "bold"),
            padx=14,
            pady=2,
            bg="#D97706",
            fg="white",
            activebackground="#B45309",
            activeforeground="white",
            relief="flat",
        ).grid(row=0, column=2)

    # ------------------------------------------------------------------ #
    #  Coordinate helpers  (canvas-y <-> intensity)
    # ------------------------------------------------------------------ #

    def _iy(self, intensity: float) -> int:
        """Intensity [0, 255] -> canvas-y  (y=0 = bright/255; y=h-1 = dark/0)."""
        return int(round((1.0 - intensity / 255.0) * (self.h - 1)))

    def _yi(self, y: int) -> float:
        """Canvas-y -> intensity [0, 255]."""
        return float(np.clip((1.0 - y / max(1, self.h - 1)) * 255.0, 0.0, 255.0))

    # ------------------------------------------------------------------ #
    #  Rendering
    # ------------------------------------------------------------------ #

    def _update_display(self):
        self._render_image()
        self._render_histogram()
        self.info_var.set(
            f"Mean: {self.mean_val:.1f}   |   "
            f"Range: [{self.low:.1f} - {self.high:.1f}]   |   "
            f"Width: {self.high - self.low:.1f}"
        )

    def _make_lut(self) -> np.ndarray:
        """Build a uint8 look-up table for the current [low, high] stretch."""
        lo, hi = max(0.0, self.low), min(255.0, self.high)
        t = np.arange(256, dtype=np.float32)
        if hi > lo:
            t = (t - lo) / (hi - lo) * 255.0
        return np.clip(t, 0, 255).astype(np.uint8)

    def _render_image(self):
        """Render the stretched image; overexposed pixels (>= high) in red."""
        lut = self._make_lut()
        gray_s = lut[self.gray_orig]  # uint8 stretched grayscale

        # RGB from grey
        rgb = np.stack([gray_s, gray_s, gray_s], axis=-1)

        # Overexposed mask: original pixel >= high threshold -> paint red
        over = self.gray_orig >= int(np.ceil(self.high))
        rgb[over] = (220, 50, 50)

        self._img_tk = ImageTk.PhotoImage(Image.fromarray(rgb.astype(np.uint8)))
        self.img_canvas.create_image(0, 0, anchor="nw", image=self._img_tk)

    def _render_histogram(self):
        c = self.hist_canvas
        c.delete("all")
        W, H = self.HIST_W, self.h

        # 1. Light-gray spectrum band
        y_hi = self._iy(self.high)
        y_lo = self._iy(self.low)
        c.create_rectangle(0, y_hi, W, y_lo, fill="#E9EAEC", outline="")

        # 2. Histogram polygon (sqrt-scaled -> flatter appearance)
        hs = self._hist_scaled
        mx = hs.max()
        bar_zone = W - 22  # right margin for handles/labels

        if mx > 0:
            pts = [0, self._iy(0)]
            for i in range(256):
                x = int(hs[i] / mx * bar_zone)
                y = self._iy(i)
                pts += [x, y]
            pts += [0, self._iy(255)]
            c.create_polygon(pts, fill="#9CA3AF", outline="", smooth=False)

        # 3. Spectrum boundary lines + arrowhead handles
        for canvas_y, label_val, txt_anchor in [
            (y_lo, self.low, "sw"),
            (y_hi, self.high, "nw"),
        ]:
            c.create_line(0, canvas_y, W, canvas_y, fill="#4B5563", width=2, dash=(6, 3))
            c.create_polygon(
                W - 12,
                canvas_y - 5,
                W,
                canvas_y,
                W - 12,
                canvas_y + 5,
                fill="#4B5563",
                outline="",
            )
            c.create_text(
                4,
                canvas_y,
                text=f"{label_val:.0f}",
                anchor=txt_anchor,
                fill="#374151",
                font=("Arial", 8),
            )

        # 4. Mean line (red) + handle
        y_m = self._iy(self.mean_val)
        c.create_line(0, y_m, W, y_m, fill="#DC2626", width=2)
        c.create_polygon(
            W - 12,
            y_m - 5,
            W,
            y_m,
            W - 12,
            y_m + 5,
            fill="#DC2626",
            outline="",
        )
        c.create_text(
            4,
            y_m,
            text=f"  {self.mean_val:.0f}",
            anchor="w",
            fill="#DC2626",
            font=("Arial", 8, "bold"),
        )

        # 5. Axis labels
        c.create_text(
            W // 2,
            3,
            text="255",
            anchor="n",
            fill="#9CA3AF",
            font=("Arial", 7),
        )
        c.create_text(
            W // 2,
            H - 3,
            text="0",
            anchor="s",
            fill="#9CA3AF",
            font=("Arial", 7),
        )

    # ------------------------------------------------------------------ #
    #  Mouse interaction
    # ------------------------------------------------------------------ #

    def _hist_press(self, event):
        y = event.y
        y_m = self._iy(self.mean_val)
        y_lo = self._iy(self.low)
        y_hi = self._iy(self.high)
        tol = self.HANDLE_TOL

        if abs(y - y_m) <= tol:
            self._drag_mode = "mean"
        elif abs(y - y_lo) <= tol:
            self._drag_mode = "low"
        elif abs(y - y_hi) <= tol:
            self._drag_mode = "high"
        else:
            self._drag_mode = None

    def _hist_drag(self, event):
        if self._drag_mode is None:
            return

        new_i = self._yi(event.y)
        half = (self.high - self.low) / 2.0

        if self._drag_mode == "mean":
            self.mean_val = float(np.clip(new_i, 0, 255))
            self.low = float(np.clip(self.mean_val - half, 0, 253))
            self.high = float(np.clip(self.mean_val + half, 2, 255))

        elif self._drag_mode == "low":
            new_half = self.mean_val - new_i
            if new_half > 2.0:
                self.low = float(np.clip(self.mean_val - new_half, 0, 253))
                self.high = float(np.clip(self.mean_val + new_half, 2, 255))

        elif self._drag_mode == "high":
            new_half = new_i - self.mean_val
            if new_half > 2.0:
                self.low = float(np.clip(self.mean_val - new_half, 0, 253))
                self.high = float(np.clip(self.mean_val + new_half, 2, 255))

        self._update_display()

    def _hist_release(self, _event):
        self._drag_mode = None

    # ------------------------------------------------------------------ #
    #  Controls
    # ------------------------------------------------------------------ #

    def _on_channel_change(self, _event=None):
        ch = self._channel_var.get()
        self._load_channel(ch)
        self.img_canvas.config(width=self.w, height=self.h)
        self.hist_canvas.config(height=self.h)
        self._update_display()

    def _reset(self):
        self.mean_val = self._init_mean
        self.low = self._init_low
        self.high = self._init_high
        self._update_display()

    # ------------------------------------------------------------------ #
    #  Synthetic test images (fallback)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _generate_test_image(channel: str) -> np.ndarray:
        """Generate a 300x300 synthetic grayscale test image."""
        size = 300
        rng = np.random.default_rng(0 if channel == "brightfield" else 7)

        yi, xi = np.mgrid[0:size, 0:size]
        cx = cy = size // 2
        d = np.sqrt((xi - cx) ** 2 + (yi - cy) ** 2)

        if channel == "brightfield":
            img = np.clip(255 - d * 0.82, 20, 255).astype(np.uint8)
            cv2.circle(img, (80, 80), 52, 210, -1)
            cv2.circle(img, (220, 220), 52, 45, -1)
            cv2.circle(img, (80, 220), 38, 155, -1)
            cv2.circle(img, (220, 80), 38, 95, -1)
        else:
            # fluorescence: dark background, bright Gaussian blobs
            img = np.clip(d * 0.15, 5, 40).astype(np.uint8)
            for (cx_, cy_, r, v) in [
                (150, 150, 45, 255),
                (80, 200, 28, 230),
                (210, 100, 22, 245),
                (190, 210, 18, 200),
            ]:
                cv2.circle(img, (cx_, cy_), r, v, -1)
            img = cv2.GaussianBlur(img, (11, 11), 4)

        noise = rng.integers(-18, 19, img.shape, dtype=np.int16)
        noise = cv2.blur(noise.astype(np.float32), (3, 3)).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return img

