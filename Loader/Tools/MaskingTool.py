import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import numpy as np
import pickle
import os
from pathlib import Path
import copy
import traceback
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Polygon


class HyperspectralMaskingApp:
    """
    Interactive application for creating and applying masks to hyperspectral data.
    """

    def __init__(self, root):
        """Initialize the application interface."""
        self.root = root
        self.root.title("Hyperspectral Data Masking Tool")
        self.root.geometry("1200x700")

        # Data variables
        self.data_dict = None
        self.file_path = None
        self.rgb_image = None
        self.mask = None
        self.height = None
        self.width = None
        self.cubes = {}  # Store found data cubes

        # Polygon drawing variables
        self.polygon_points = []
        self.polygon_patch = None
        self.drawing_active = False
        self.drawing_mode = "polygon"

        self.create_widgets()

    def create_widgets(self):
        """Create the application widgets and layout."""
        # Main layout with two frames
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left side - Control panel
        self.control_frame = ttk.LabelFrame(self.main_frame, text="Controls")
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        # File controls
        self.file_frame = ttk.LabelFrame(self.control_frame, text="File")
        self.file_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(self.file_frame, text="Open Pickle File", command=self.open_file).pack(fill=tk.X, padx=5, pady=2)

        # Visualization controls
        self.viz_frame = ttk.LabelFrame(self.control_frame, text="Visualization")
        self.viz_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(self.viz_frame, text="RGB Method:").pack(anchor=tk.W, padx=5, pady=2)
        self.rgb_method = tk.StringVar(value="rgb")
        methods = ["rgb", "max", "mean"]
        for method in methods:
            ttk.Radiobutton(self.viz_frame, text=method, variable=self.rgb_method,
                            value=method, command=self.update_visualization).pack(anchor=tk.W, padx=15, pady=1)

        # Excitation wavelength selection
        self.ex_frame = ttk.Frame(self.viz_frame)
        self.ex_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(self.ex_frame, text="Excitation (nm):").pack(side=tk.LEFT, padx=5)
        self.ex_var = tk.StringVar()
        self.ex_combo = ttk.Combobox(self.ex_frame, textvariable=self.ex_var, state="readonly", width=10)
        self.ex_combo.pack(side=tk.LEFT, padx=5)
        self.ex_combo.bind("<<ComboboxSelected>>", self.update_visualization)

        # Display enhancement controls
        self.enhance_frame = ttk.Frame(self.control_frame)
        self.enhance_frame.pack(fill=tk.X, pady=5)
        ttk.Label(self.enhance_frame, text="Contrast Enhancement:").pack(anchor=tk.W, padx=5, pady=2)

        self.auto_contrast_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.enhance_frame, text="Auto Contrast", variable=self.auto_contrast_var,
                        command=self.update_visualization).pack(anchor=tk.W, padx=15, pady=1)

        # Min/Max controls for manual contrast
        self.min_max_frame = ttk.Frame(self.control_frame)
        self.min_max_frame.pack(fill=tk.X, pady=5)

        ttk.Label(self.min_max_frame, text="Min:").pack(side=tk.LEFT, padx=5)
        self.min_var = tk.DoubleVar(value=0.0)
        self.min_entry = ttk.Entry(self.min_max_frame, textvariable=self.min_var, width=8)
        self.min_entry.pack(side=tk.LEFT)

        ttk.Label(self.min_max_frame, text="Max:").pack(side=tk.LEFT, padx=5)
        self.max_var = tk.DoubleVar(value=100.0)
        self.max_entry = ttk.Entry(self.min_max_frame, textvariable=self.max_var, width=8)
        self.max_entry.pack(side=tk.LEFT)

        ttk.Button(self.min_max_frame, text="Apply", command=self.update_visualization).pack(side=tk.LEFT, padx=5)

        # Drawing tools
        self.tools_frame = ttk.LabelFrame(self.control_frame, text="Masking Tools")
        self.tools_frame.pack(fill=tk.X, padx=5, pady=5)

        # Drawing type selection
        ttk.Label(self.tools_frame, text="Selection Type:").pack(anchor=tk.W, padx=5, pady=2)
        self.drawing_mode_var = tk.StringVar(value="polygon")
        tools = [("Polygon Selection", "polygon"), ("Rectangle Selection", "rectangle")]
        for text, value in tools:
            ttk.Radiobutton(self.tools_frame, text=text, variable=self.drawing_mode_var,
                            value=value).pack(anchor=tk.W, padx=15, pady=1)

        # Mask operation buttons
        ttk.Button(self.tools_frame, text="Create Mask", command=self.start_drawing).pack(fill=tk.X, padx=5, pady=2)

        # Polygon editing controls
        self.edit_frame = ttk.LabelFrame(self.control_frame, text="Drawing Controls")
        self.edit_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(self.edit_frame, text="Remove Last Point", command=self.remove_last_point).pack(
            fill=tk.X, padx=5, pady=2)
        ttk.Button(self.edit_frame, text="Clear Points", command=self.clear_points).pack(
            fill=tk.X, padx=5, pady=2)
        ttk.Button(self.edit_frame, text="Finish Drawing", command=self.finish_drawing).pack(
            fill=tk.X, padx=5, pady=2)
        ttk.Button(self.edit_frame, text="Cancel Drawing", command=self.cancel_drawing).pack(
            fill=tk.X, padx=5, pady=2)

        # Mask modification buttons
        self.mask_ops_frame = ttk.LabelFrame(self.control_frame, text="Mask Operations")
        self.mask_ops_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(self.mask_ops_frame, text="Add to Mask", command=self.add_to_mask).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(self.mask_ops_frame, text="Subtract from Mask", command=self.subtract_from_mask).pack(
            fill=tk.X, padx=5, pady=2)
        ttk.Button(self.mask_ops_frame, text="Clear Mask", command=self.clear_mask).pack(fill=tk.X, padx=5, pady=2)

        # Save controls
        self.save_frame = ttk.LabelFrame(self.control_frame, text="Save")
        self.save_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(self.save_frame, text="Apply & Save", command=self.apply_and_save).pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(self.save_frame, text="Save Mask Only", command=self.save_mask_only).pack(fill=tk.X, padx=5, pady=2)

        # Status information
        self.status_frame = ttk.LabelFrame(self.control_frame, text="Status")
        self.status_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(self.status_frame, text="Mask Status:").pack(anchor=tk.W, padx=5)
        self.status_label = ttk.Label(self.status_frame, text="No mask created", foreground="red")
        self.status_label.pack(anchor=tk.W, padx=5, pady=2)

        # Position and value display
        ttk.Label(self.status_frame, text="Cursor Position:").pack(anchor=tk.W, padx=5)
        self.position_label = ttk.Label(self.status_frame, text="")
        self.position_label.pack(anchor=tk.W, padx=5, pady=2)

        ttk.Label(self.status_frame, text="Pixel Value:").pack(anchor=tk.W, padx=5)
        self.value_label = ttk.Label(self.status_frame, text="")
        self.value_label.pack(anchor=tk.W, padx=5, pady=2)

        # Right side - Image display
        self.display_frame = ttk.Frame(self.main_frame)
        self.display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create notebook with tabs
        self.notebook = ttk.Notebook(self.display_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Original image tab
        self.orig_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.orig_tab, text="Original Image")

        # Mask editing tab
        self.mask_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.mask_tab, text="Mask Editing")

        # Preview tab
        self.preview_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.preview_tab, text="Preview")

        # Setup matplotlib figure and canvas
        self.figure = Figure(figsize=(10, 8), dpi=100)
        self.figure.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        self.ax = self.figure.add_subplot(111)

        # Create canvases for each tab
        self.orig_canvas = FigureCanvasTkAgg(self.figure, master=self.orig_tab)
        self.orig_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.mask_canvas = FigureCanvasTkAgg(self.figure, master=self.mask_tab)
        self.mask_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.preview_canvas = FigureCanvasTkAgg(self.figure, master=self.preview_tab)
        self.preview_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Connect mouse events to each canvas
        for canvas in [self.orig_canvas, self.mask_canvas, self.preview_canvas]:
            canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
            canvas.mpl_connect('button_press_event', self.on_mouse_click)
            canvas.mpl_connect('key_press_event', self.on_key_press)

        # Initial message
        self.ax.text(0.5, 0.5, "Load a pickle file to begin",
                     ha='center', va='center', fontsize=14,
                     transform=self.ax.transAxes)

        # Update all canvases
        self.draw_all_canvases()

        # Initially hide editing controls
        self.edit_frame.pack_forget()

        # Disable controls until a file is loaded
        self.toggle_controls(False)

    def toggle_controls(self, enable=True):
        """Enable or disable controls based on whether data is loaded."""
        state = "normal" if enable else "disabled"

        for widget in self.viz_frame.winfo_children():
            try:
                widget.configure(state=state)
            except:
                pass

        for widget in self.tools_frame.winfo_children():
            try:
                widget.configure(state=state)
            except:
                pass

        for widget in self.mask_ops_frame.winfo_children():
            try:
                widget.configure(state=state)
            except:
                pass

        for widget in self.save_frame.winfo_children():
            try:
                widget.configure(state=state)
            except:
                pass

    def open_file(self):
        """Open a hyperspectral data pickle file."""
        file_path = filedialog.askopenfilename(
            title="Select Pickle File",
            filetypes=[("Pickle Files", "*.pkl"), ("All Files", "*.*")]
        )

        if not file_path:
            return

        try:
            self.status_label.config(text=f"Loading {os.path.basename(file_path)}...", foreground="blue")
            self.root.update_idletasks()

            # Load the data
            with open(file_path, 'rb') as f:
                self.data_dict = pickle.load(f)

            self.file_path = file_path

            # Print structure for debugging
            print(f"Data structure of {os.path.basename(file_path)}:")
            self._print_nested_structure(self.data_dict)

            # Find data cubes in the structure
            self._find_data_cubes(self.data_dict)

            if not self.cubes:
                raise ValueError("No suitable data cubes found in the pickle file")

            # Extract dimensions from the first cube
            first_key = next(iter(self.cubes))
            first_cube = self.cubes[first_key]
            self.height, self.width = first_cube.shape[0], first_cube.shape[1]

            # Initialize mask
            self._initialize_mask()

            # Populate excitation dropdown
            self._populate_excitation_dropdown()

            # Update visualization
            self.update_visualization()

            # Enable controls
            self.toggle_controls(True)

            self.status_label.config(text=f"Loaded {os.path.basename(file_path)} successfully", foreground="green")

        except Exception as e:
            error_msg = f"Failed to load file: {str(e)}"
            print(error_msg)
            print(traceback.format_exc())  # Print full traceback for debugging
            messagebox.showerror("Error", error_msg)
            self.status_label.config(text="Error loading file", foreground="red")

    def _print_nested_structure(self, obj, prefix='', max_depth=3, current_depth=0):
        """Print the structure of a nested object for debugging."""
        if current_depth >= max_depth:
            return

        if isinstance(obj, dict):
            for key, value in obj.items():
                value_info = f"{type(value).__name__}"
                if isinstance(value, np.ndarray):
                    value_info += f" shape={value.shape} dtype={value.dtype}"
                print(f"{prefix}{key}: {value_info}")

                if isinstance(value, (dict, list)) and current_depth < max_depth - 1:
                    self._print_nested_structure(value, prefix + '  ', max_depth, current_depth + 1)

        elif isinstance(obj, list) and obj:
            print(f"{prefix}list (length: {len(obj)})")
            if obj and current_depth < max_depth - 1:
                self._print_nested_structure(obj[0], prefix + '  ', max_depth, current_depth + 1)

    def _find_data_cubes(self, data, path='', max_depth=5, current_depth=0):
        """Recursively find all 3D data cubes in the structure."""
        if current_depth >= max_depth:
            return

        if isinstance(data, dict):
            for key, value in data.items():
                current_path = f"{path}.{key}" if path else key

                # Check if this is a data cube (3D numpy array with dimensions [bands, height, width])
                if isinstance(value, np.ndarray) and len(value.shape) == 3:
                    # Store the cube
                    self.cubes[current_path] = value
                    print(f"Found data cube at {current_path}: shape={value.shape}")

                # Recursively search nested structures
                if isinstance(value, (dict, list)):
                    self._find_data_cubes(value, current_path, max_depth, current_depth + 1)

        elif isinstance(data, list) and len(data) > 0:
            # For lists, only check the first item as an example
            current_path = f"{path}[0]"
            if isinstance(data[0], (dict, list, np.ndarray)):
                self._find_data_cubes(data[0], current_path, max_depth, current_depth + 1)

    def _initialize_mask(self):
        """Initialize the mask to all zeros (nothing masked)."""
        if self.height is None or self.width is None:
            raise ValueError("Image dimensions not set")

        self.mask = np.zeros((self.height, self.width), dtype=np.uint8)
        print(f"Initialized mask with dimensions {self.mask.shape}")

    def _populate_excitation_dropdown(self):
        """Populate the excitation wavelength dropdown with available cube keys."""
        # Clear previous values
        self.ex_combo['values'] = []

        if not self.cubes:
            return

        # Use the keys from found cubes
        cube_keys = list(self.cubes.keys())
        self.ex_combo['values'] = cube_keys

        # Select the first one
        if cube_keys:
            self.ex_var.set(cube_keys[0])

    def update_visualization(self, event=None):
        """Update the visualization based on current settings."""
        if not self.cubes:
            return

        # Get the selected excitation key and RGB method
        excitation_key = self.ex_var.get()
        rgb_method = self.rgb_method.get()

        if not excitation_key or excitation_key not in self.cubes:
            print(f"Invalid excitation key: {excitation_key}")
            return

        # Create RGB image
        self.rgb_image = self.create_rgb_image(excitation_key, rgb_method)

        # Clear previous plot
        self.ax.clear()

        # Apply contrast enhancement
        if self.auto_contrast_var.get():
            vmin = np.percentile(self.rgb_image, 2)  # 2nd percentile to avoid outliers
            vmax = np.percentile(self.rgb_image, 98)  # 98th percentile to avoid outliers
        else:
            vmin = self.min_var.get() / 100.0  # Convert to 0-1 range
            vmax = self.max_var.get() / 100.0

        # Print image shape for debugging
        print(f"RGB image shape: {self.rgb_image.shape}")

        # Display the image with proper aspect ratio
        self.ax.imshow(self.rgb_image, vmin=vmin, vmax=vmax, aspect='equal')

        # If mask exists, overlay it
        if self.mask is not None and np.any(self.mask):
            # Create RGBA mask overlay: red with alpha channel
            mask_overlay = np.zeros((*self.mask.shape, 4), dtype=np.float32)
            # Red color with 50% transparency where mask is 1
            mask_overlay[self.mask == 1, 0] = 1.0  # Red channel
            mask_overlay[self.mask == 1, 3] = 0.5  # Alpha channel

            self.ax.imshow(mask_overlay, interpolation='nearest', aspect='equal')

            # Update status
            masked_pixels = np.sum(self.mask)
            total_pixels = self.mask.size
            percent_masked = (masked_pixels / total_pixels) * 100
            self.status_label.config(text=f"Masked: {masked_pixels} pixels ({percent_masked:.1f}%)", foreground="green")
        else:
            self.status_label.config(text="No mask created", foreground="red")

        # Set title
        self.ax.set_title(f"Excitation: {excitation_key}")

        # Draw polygon points if in drawing mode
        if self.drawing_active and self.polygon_points:
            self.draw_polygon_preview()

        # Update all canvases
        self.draw_all_canvases()

    def draw_all_canvases(self):
        """Update all matplotlib canvases."""
        for canvas in [self.orig_canvas, self.mask_canvas, self.preview_canvas]:
            canvas.draw()

    def create_rgb_image(self, excitation_key, method='rgb', percentile=99):
        """
        Create an RGB representation of the selected data cube.

        Args:
            excitation_key: Key to the data cube in self.cubes
            method: Method for creating RGB ('rgb', 'max', 'mean')
            percentile: Percentile for scaling (to avoid outliers)

        Returns:
            RGB image as numpy array with shape (height, width, 3)
        """
        # Get the data cube
        cube = self.cubes[excitation_key]

        # Print cube shape for debugging
        print(f"Original cube shape: {cube.shape}")

        # Check if the cube might be transposed (common issue)
        # We expect the cube to have shape (bands, height, width) or (height, width, bands)
        if len(cube.shape) == 3:
            # Identify which dimension is likely the spectral dimension
            if cube.shape[0] < cube.shape[1] and cube.shape[0] < cube.shape[2]:
                # First dimension is smallest, likely spectral bands
                bands_dim = 0
                height_dim = 1
                width_dim = 2
                print("Cube format: (bands, height, width)")
            elif cube.shape[2] < cube.shape[0] and cube.shape[2] < cube.shape[1]:
                # Last dimension is smallest, likely spectral bands
                bands_dim = 2
                height_dim = 0
                width_dim = 1
                print("Cube format: (height, width, bands)")
            else:
                # Hard to tell, assume standard (bands, height, width)
                bands_dim = 0
                height_dim = 1
                width_dim = 2
                print("Assuming cube format: (bands, height, width)")
        else:
            raise ValueError(f"Unexpected cube shape: {cube.shape}. Expected 3D array.")

        # Replace any NaN values
        cube = np.nan_to_num(cube)

        # Create RGB based on method
        if method == 'rgb':
            # Use three wavelengths as RGB channels
            num_bands = cube.shape[bands_dim]
            if num_bands >= 3:
                indices = [int(num_bands * 0.2), int(num_bands * 0.5), int(num_bands * 0.8)]
                r_idx, g_idx, b_idx = indices

                # Extract band images based on identified dimensions
                if bands_dim == 0:
                    r_band = cube[r_idx, :, :]
                    g_band = cube[g_idx, :, :]
                    b_band = cube[b_idx, :, :]
                else:  # bands_dim == 2
                    r_band = cube[:, :, r_idx]
                    g_band = cube[:, :, g_idx]
                    b_band = cube[:, :, b_idx]

                # Normalize to range [0, 1]
                r_scaled = r_band / np.percentile(r_band, percentile)
                g_scaled = g_band / np.percentile(g_band, percentile)
                b_scaled = b_band / np.percentile(b_band, percentile)

                # Clip to [0, 1] range
                r_scaled = np.clip(r_scaled, 0, 1)
                g_scaled = np.clip(g_scaled, 0, 1)
                b_scaled = np.clip(b_scaled, 0, 1)

                # Create RGB image
                rgb = np.stack([r_scaled, g_scaled, b_scaled], axis=2)

                print(f"Created RGB image using bands at indices {r_idx}, {g_idx}, {b_idx}")
            else:
                # Default to max projection if not enough bands
                method = 'max'

        if method in ['max', 'mean']:
            # Generate a projection based on the identified dimensions
            if bands_dim == 0:
                if method == 'max':
                    proj = np.max(cube, axis=0)
                else:  # mean
                    proj = np.mean(cube, axis=0)
            else:  # bands_dim == 2
                if method == 'max':
                    proj = np.max(cube, axis=2)
                else:  # mean
                    proj = np.mean(cube, axis=2)

            # Scale and clip
            max_val = np.percentile(proj[~np.isnan(proj)], percentile)  # Exclude NaNs for percentile
            scaled = proj / max_val if max_val > 0 else proj
            scaled = np.clip(scaled, 0, 1)

            # Replace NaNs with zeros
            scaled = np.nan_to_num(scaled)

            # Convert to RGB by duplicating the channel
            rgb = np.stack([scaled, scaled, scaled], axis=2)
            print(f"Created RGB image using {method} projection")

        print(f"Final RGB image shape: {rgb.shape}")
        return rgb

    def on_mouse_move(self, event):
        """Handle mouse movement over the image."""
        if event.inaxes != self.ax or self.rgb_image is None:
            self.position_label.config(text="")
            self.value_label.config(text="")
            return

        x, y = int(event.xdata) if event.xdata is not None else -1, int(event.ydata) if event.ydata is not None else -1

        # Check bounds
        if 0 <= y < self.height and 0 <= x < self.width:
            # Update position label
            self.position_label.config(text=f"X: {x}, Y: {y}")

            # Update value label
            if self.mask is not None:
                mask_state = "Masked" if self.mask[y, x] == 1 else "Unmasked"
                self.value_label.config(text=f"Status: {mask_state}")

    def on_mouse_click(self, event):
        """Handle mouse click events for drawing."""
        if not self.drawing_active or event.inaxes != self.ax:
            return

        print(f"Mouse click detected at: {event.xdata}, {event.ydata}")

        # Get click coordinates
        x, y = event.xdata, event.ydata

        if x is None or y is None:
            return

        # Append to polygon points
        self.polygon_points.append((x, y))

        # Update status
        point_count = len(self.polygon_points)
        self.status_label.config(text=f"Drawing: {point_count} points", foreground="blue")

        # Draw the updated polygon preview
        self.draw_polygon_preview()

        # If we're in rectangle mode and have 2 points, automatically finish
        if self.drawing_mode_var.get() == "rectangle" and len(self.polygon_points) == 2:
            self.finish_drawing()

    def draw_polygon_preview(self):
        """Draw the current polygon as a preview."""
        # Remove previous polygon patch if it exists
        if self.polygon_patch in self.ax.patches:
            self.polygon_patch.remove()
            self.polygon_patch = None

        if len(self.polygon_points) < 2:
            # Just draw points if we have fewer than 2 vertices
            self.ax.plot([p[0] for p in self.polygon_points],
                         [p[1] for p in self.polygon_points],
                         'ro-', linewidth=2, markersize=8)
        else:
            # Draw polygon
            self.polygon_patch = Polygon(self.polygon_points, fill=False,
                                         edgecolor='red', linewidth=2, alpha=0.8)
            self.ax.add_patch(self.polygon_patch)

            # Add markers for each point
            self.ax.plot([p[0] for p in self.polygon_points],
                         [p[1] for p in self.polygon_points],
                         'ro', markersize=8)

        # Update canvases
        self.draw_all_canvases()

    def on_key_press(self, event):
        """Handle keyboard shortcuts during polygon editing."""
        if not self.drawing_active:
            return

        if event.key == 'backspace':
            self.remove_last_point()
        elif event.key == 'escape':
            self.cancel_drawing()
        elif event.key == 'enter':
            self.finish_drawing()

    def start_drawing(self):
        """Start drawing a new mask."""
        if self.rgb_image is None:
            messagebox.showwarning("Warning", "Load an image first!")
            return

        # Switch to the mask editing tab
        self.notebook.select(self.mask_tab)

        # Get drawing mode
        self.drawing_mode = self.drawing_mode_var.get()

        # Reset polygon points
        self.polygon_points = []
        self.polygon_patch = None

        # Activate drawing mode
        self.drawing_active = True

        # Show editing controls
        self.edit_frame.pack(after=self.tools_frame, fill=tk.X, padx=5, pady=5)

        # Force canvas to grab focus
        self.mask_canvas.get_tk_widget().focus_force()

        # Update status
        self.status_label.config(text=f"Drawing {self.drawing_mode}: 0 points", foreground="blue")

        # Rectangle mode instructions
        if self.drawing_mode == "rectangle":
            messagebox.showinfo("Rectangle Selection",
                                "Click two points to define the rectangle:\n"
                                "- First click: top-left corner\n"
                                "- Second click: bottom-right corner\n\n"
                                "The rectangle will be created automatically after the second point.")
        else:
            # Polygon mode instructions
            messagebox.showinfo("Polygon Selection",
                                "Click to add polygon vertices.\n"
                                "When finished, click 'Finish Drawing' or press Enter.\n\n"
                                "Keyboard shortcuts:\n"
                                "Enter: Finish polygon\n"
                                "Backspace: Remove last point\n"
                                "Escape: Cancel drawing")

    def remove_last_point(self):
        """Remove the last point from the polygon."""
        if not self.drawing_active or not self.polygon_points:
            return

        # Remove last point
        self.polygon_points.pop()

        # Update status
        point_count = len(self.polygon_points)
        self.status_label.config(text=f"Drawing: {point_count} points", foreground="blue")

        # Redraw
        self.update_visualization()

    def clear_points(self):
        """Clear all polygon points."""
        if not self.drawing_active:
            return

        # Clear points
        self.polygon_points = []

        # Update status
        self.status_label.config(text="Drawing: 0 points", foreground="blue")

        # Redraw
        self.update_visualization()

    def finish_drawing(self):
        """Finalize the drawing and create a mask."""
        if not self.drawing_active:
            return

        drawing_mode = self.drawing_mode

        if drawing_mode == "polygon":
            if len(self.polygon_points) < 3:
                messagebox.showwarning("Warning", "Need at least 3 points to create a polygon mask")
                return

            # Create mask from polygon
            self.create_mask_from_polygon()

        elif drawing_mode == "rectangle":
            if len(self.polygon_points) < 2:
                messagebox.showwarning("Warning", "Need 2 points to define a rectangle")
                return

            # Create mask from rectangle
            self.create_mask_from_rectangle()

        # Hide editing controls
        self.edit_frame.pack_forget()

        # End drawing mode
        self.drawing_active = False

        # Clear polygon preview
        self.polygon_points = []
        if self.polygon_patch in self.ax.patches:
            self.polygon_patch.remove()
            self.polygon_patch = None

        # Update visualization
        self.update_visualization()

    def create_mask_from_polygon(self):
        """Create a mask from the current polygon points."""
        if not self.polygon_points or len(self.polygon_points) < 3:
            return

        # Convert vertices to numpy array
        vertices = np.array(self.polygon_points)

        # Create a grid of pixel coordinates
        y, x = np.mgrid[:self.height, :self.width]
        points = np.vstack((x.flatten(), y.flatten())).T

        # Create mask from polygon
        path = MplPath(vertices)
        mask = path.contains_points(points)
        mask = mask.reshape(self.height, self.width)

        # Set this as the mask (replacing any existing mask)
        self.mask = mask.astype(np.uint8)

        # Show confirmation
        self.status_label.config(text=f"Mask created with {len(vertices)} vertices", foreground="green")

    def create_mask_from_rectangle(self):
        """Create a mask from the current rectangle points."""
        if len(self.polygon_points) < 2:
            return

        # Extract rectangle corners
        x1, y1 = self.polygon_points[0]
        x2, y2 = self.polygon_points[1]

        # Convert to integers and ensure proper bounds
        x1, x2 = int(min(x1, x2)), int(max(x1, x2))
        y1, y2 = int(min(y1, y2)), int(max(y1, y2))

        # Clamp to image boundaries
        x1 = max(0, min(x1, self.width - 1))
        x2 = max(0, min(x2, self.width - 1))
        y1 = max(0, min(y1, self.height - 1))
        y2 = max(0, min(y2, self.height - 1))

        # Create rectangle mask
        self.mask = np.zeros((self.height, self.width), dtype=np.uint8)
        self.mask[y1:y2 + 1, x1:x2 + 1] = 1

        # Show confirmation
        width, height = x2 - x1 + 1, y2 - y1 + 1
        self.status_label.config(text=f"Rectangle mask created ({width}x{height})", foreground="green")

    def cancel_drawing(self):
        """Cancel the current drawing operation."""
        if not self.drawing_active:
            return

        # Hide editing controls
        self.edit_frame.pack_forget()

        # End drawing mode
        self.drawing_active = False

        # Clear polygon preview
        self.polygon_points = []
        if self.polygon_patch in self.ax.patches:
            self.polygon_patch.remove()
            self.polygon_patch = None

        # Update visualization
        self.update_visualization()

        # Show message
        self.status_label.config(text="Drawing cancelled", foreground="red")

    def add_to_mask(self):
        """Add the current selection to existing mask."""
        if not self.mask.any():
            messagebox.showinfo("Info", "No mask exists. Create a mask first.")
            return

        # Start a new drawing
        self.start_drawing()

        # Change button behavior to add to mask when finished
        self.edit_frame.children["!button3"].config(
            text="Add to Mask",
            command=lambda: self.finish_add_to_mask()
        )

    def finish_add_to_mask(self):
        """Finish adding to the mask."""
        if not self.drawing_active:
            return

        # Create a temporary mask
        temp_mask = np.zeros((self.height, self.width), dtype=np.uint8)

        # Fill the temporary mask
        if self.drawing_mode == "polygon":
            if len(self.polygon_points) < 3:
                messagebox.showwarning("Warning", "Need at least 3 points to create a polygon")
                return

            # Create polygon mask
            vertices = np.array(self.polygon_points)
            y, x = np.mgrid[:self.height, :self.width]
            points = np.vstack((x.flatten(), y.flatten())).T
            path = MplPath(vertices)
            mask = path.contains_points(points)
            temp_mask = mask.reshape(self.height, self.width).astype(np.uint8)

        elif self.drawing_mode == "rectangle":
            if len(self.polygon_points) < 2:
                messagebox.showwarning("Warning", "Need 2 points to define a rectangle")
                return

            # Create rectangle mask
            x1, y1 = self.polygon_points[0]
            x2, y2 = self.polygon_points[1]

            # Convert to integers and ensure proper bounds
            x1, x2 = int(min(x1, x2)), int(max(x1, x2))
            y1, y2 = int(min(y1, y2)), int(max(y1, y2))

            # Clamp to image boundaries
            x1 = max(0, min(x1, self.width - 1))
            x2 = max(0, min(x2, self.width - 1))
            y1 = max(0, min(y1, self.height - 1))
            y2 = max(0, min(y2, self.height - 1))

            temp_mask[y1:y2 + 1, x1:x2 + 1] = 1

        # Add to existing mask (logical OR)
        added_pixels = np.logical_and(temp_mask, ~self.mask).sum()
        self.mask = np.logical_or(self.mask, temp_mask).astype(np.uint8)

        # Hide editing controls
        self.edit_frame.pack_forget()

        # End drawing mode
        self.drawing_active = False

        # Clear polygon preview
        self.polygon_points = []
        if self.polygon_patch in self.ax.patches:
            self.polygon_patch.remove()
            self.polygon_patch = None

        # Update visualization
        self.update_visualization()

        # Reset button text and command
        self.edit_frame.children["!button3"].config(
            text="Finish Drawing",
            command=self.finish_drawing
        )

        # Show confirmation
        self.status_label.config(text=f"Added {added_pixels} pixels to mask", foreground="green")

    def subtract_from_mask(self):
        """Subtract the current selection from existing mask."""
        if not self.mask.any():
            messagebox.showinfo("Info", "No mask exists. Create a mask first.")
            return

        # Start a new drawing
        self.start_drawing()

        # Change button behavior to subtract from mask when finished
        self.edit_frame.children["!button3"].config(
            text="Subtract from Mask",
            command=lambda: self.finish_subtract_from_mask()
        )

    def finish_subtract_from_mask(self):
        """Finish subtracting from the mask."""
        if not self.drawing_active:
            return

        # Create a temporary mask
        temp_mask = np.zeros((self.height, self.width), dtype=np.uint8)

        # Fill the temporary mask
        if self.drawing_mode == "polygon":
            if len(self.polygon_points) < 3:
                messagebox.showwarning("Warning", "Need at least 3 points to create a polygon")
                return

            # Create polygon mask
            vertices = np.array(self.polygon_points)
            y, x = np.mgrid[:self.height, :self.width]
            points = np.vstack((x.flatten(), y.flatten())).T
            path = MplPath(vertices)
            mask = path.contains_points(points)
            temp_mask = mask.reshape(self.height, self.width).astype(np.uint8)

        elif self.drawing_mode == "rectangle":
            if len(self.polygon_points) < 2:
                messagebox.showwarning("Warning", "Need 2 points to define a rectangle")
                return

            # Create rectangle mask
            x1, y1 = self.polygon_points[0]
            x2, y2 = self.polygon_points[1]

            # Convert to integers and ensure proper bounds
            x1, x2 = int(min(x1, x2)), int(max(x1, x2))
            y1, y2 = int(min(y1, y2)), int(max(y1, y2))

            # Clamp to image boundaries
            x1 = max(0, min(x1, self.width - 1))
            x2 = max(0, min(x2, self.width - 1))
            y1 = max(0, min(y1, self.height - 1))
            y2 = max(0, min(y2, self.height - 1))

            temp_mask[y1:y2 + 1, x1:x2 + 1] = 1

        # Subtract from existing mask (logical AND with NOT)
        removed_pixels = np.logical_and(temp_mask, self.mask).sum()
        self.mask = np.logical_and(self.mask, ~temp_mask).astype(np.uint8)

        # Hide editing controls
        self.edit_frame.pack_forget()

        # End drawing mode
        self.drawing_active = False

        # Clear polygon preview
        self.polygon_points = []
        if self.polygon_patch in self.ax.patches:
            self.polygon_patch.remove()
            self.polygon_patch = None

        # Update visualization
        self.update_visualization()

        # Reset button text and command
        self.edit_frame.children["!button3"].config(
            text="Finish Drawing",
            command=self.finish_drawing
        )

        # Show confirmation
        self.status_label.config(text=f"Removed {removed_pixels} pixels from mask", foreground="green")

    def clear_mask(self):
        """Clear the current mask."""
        if self.mask is not None:
            # Confirm with user
            if messagebox.askyesno("Clear Mask", "Are you sure you want to clear the mask?"):
                self.mask = np.zeros((self.height, self.width), dtype=np.uint8)
                self.update_visualization()
                self.status_label.config(text="Mask cleared", foreground="red")

    def apply_and_save(self):
        """Apply the mask to the data and save the results."""
        if not self.data_dict or not self.mask is not None:
            return

        # Ask for save location
        output_dir = filedialog.askdirectory(
            title="Select directory to save masked data",
            initialdir=os.path.dirname(self.file_path) if self.file_path else None
        )

        if not output_dir:
            return  # User cancelled

        try:
            # Apply the mask
            self.status_label.config(text="Applying mask to hyperspectral data...", foreground="blue")
            self.root.update_idletasks()

            masked_data = self.apply_mask()

            # Create output file names
            base_name = os.path.basename(self.file_path) if self.file_path else "hyperspectral_data"
            base_name = os.path.splitext(base_name)[0]

            # Save masked data
            masked_file_path = os.path.join(output_dir, f"{base_name}_masked.pkl")
            with open(masked_file_path, 'wb') as f:
                pickle.dump(masked_data, f)

            # Save mask separately
            mask_file_path = os.path.join(output_dir, f"{base_name}_mask.npy")
            np.save(mask_file_path, self.mask)

            self.status_label.config(text=f"Saved masked data and mask", foreground="green")
            messagebox.showinfo("Success",
                                f"Saved masked data to:\n{masked_file_path}\n\nSaved mask to:\n{mask_file_path}")

        except Exception as e:
            error_msg = f"Failed to save data: {str(e)}"
            print(error_msg)
            print(traceback.format_exc())
            messagebox.showerror("Error", error_msg)
            self.status_label.config(text=f"Error saving data", foreground="red")

    def save_mask_only(self):
        """Save only the mask to a file."""
        if not self.mask is not None:
            return

        # Ask for save location
        file_path = filedialog.asksaveasfilename(
            title="Save Mask",
            defaultextension=".npy",
            filetypes=[("NumPy Files", "*.npy"), ("All Files", "*.*")],
            initialdir=os.path.dirname(self.file_path) if self.file_path else None,
            initialfile="mask.npy"
        )

        if not file_path:
            return  # User cancelled

        try:
            # Save the mask
            np.save(file_path, self.mask)
            self.status_label.config(text=f"Saved mask to {os.path.basename(file_path)}", foreground="green")
            messagebox.showinfo("Success", f"Saved mask to:\n{file_path}")

        except Exception as e:
            error_msg = f"Failed to save mask: {str(e)}"
            print(error_msg)
            messagebox.showerror("Error", error_msg)
            self.status_label.config(text=f"Error saving mask", foreground="red")

    def apply_mask(self):
        """
        Apply the mask to the hyperspectral data.

        Returns:
            Dictionary with masked hyperspectral data
        """
        if not self.data_dict:
            raise ValueError("Data not loaded")

        if self.mask is None:
            raise ValueError("Mask not created")

        # Create a deep copy to avoid modifying the original
        masked_data = copy.deepcopy(self.data_dict)

        # Add mask to metadata
        if isinstance(masked_data, dict):
            if 'metadata' not in masked_data:
                masked_data['metadata'] = {}

            masked_data['metadata']['mask_applied'] = True
            masked_data['metadata']['mask_shape'] = self.mask.shape
            masked_data['metadata']['mask_sum'] = int(np.sum(self.mask))
            masked_data['metadata']['mask_percentage'] = float(np.sum(self.mask) / (self.height * self.width) * 100)

        # Apply mask to all data cubes found
        for cube_path, original_cube in self.cubes.items():
            # Get the nested path to this cube
            path_parts = cube_path.split('.')

            # Navigate to the parent container
            current = masked_data
            for part in path_parts[:-1]:
                # Handle list indices
                if '[' in part and ']' in part:
                    name, idx = part.split('[')
                    idx = int(idx.strip('[]'))
                    if name:
                        current = current[name][idx]
                    else:
                        current = current[idx]
                else:
                    current = current[part]

            # Get the last part (the key containing the cube)
            last_part = path_parts[-1]

            # Handle list indices in the last part
            if '[' in last_part and ']' in last_part:
                name, idx = last_part.split('[')
                idx = int(idx.strip('[]'))
                if name:
                    cube = current[name][idx]
                    # Create masked version
                    masked_cube = np.copy(cube)
                    # Set values outside mask to NaN
                    for i in range(cube.shape[0]):  # Assuming band is first dimension
                        masked_cube[i][~self.mask] = np.nan
                    current[name][idx] = masked_cube
                else:
                    cube = current[idx]
                    # Create masked version
                    masked_cube = np.copy(cube)
                    # Set values outside mask to NaN
                    for i in range(cube.shape[0]):
                        masked_cube[i][~self.mask] = np.nan
                    current[idx] = masked_cube
            else:
                cube = current[last_part]
                # Create masked version
                masked_cube = np.copy(cube)
                # Set values outside mask to NaN
                for i in range(cube.shape[0]):  # Assuming band is first dimension
                    masked_cube[i][~self.mask] = np.nan
                current[last_part] = masked_cube

        print(f"Applied mask: kept {np.sum(self.mask)} pixels, masked out {np.sum(~self.mask)} pixels")
        return masked_data


def main():
    """Launch the application."""
    root = tk.Tk()
    app = HyperspectralMaskingApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()