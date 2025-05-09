import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import sys
import traceback
from pathlib import Path

# Add parent directory to path to import HyperspectralDataLoader
sys.path.append(str(Path(__file__).parent.parent))
try:
    from HyperspectralDataLoader import HyperspectralDataLoader
except ImportError:
    print("Could not import HyperspectralDataLoader. Make sure it's in the parent directory.")


class HyperspectralViewer:
    """
    Interactive application for visualizing and exploring hyperspectral data from pickle files.
    """

    def __init__(self, root):
        """Initialize the application interface."""
        self.root = root
        self.root.title("Hyperspectral Data-raw Viewer")
        self.root.geometry("1200x800")

        # Data-raw variables
        self.data_dict = None
        self.file_path = None
        self.loader = None
        self.cubes = {}  # Store found data cubes
        self.current_excitation = None
        self.current_cube = None
        self.current_wavelengths = None
        self.selected_pixel = None
        self.rgb_image = None

        # Create the UI
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
        ttk.Button(self.file_frame, text="Show Data-raw Structure", command=self.show_data_structure).pack(fill=tk.X, padx=5, pady=2)

        # Visualization controls
        self.viz_frame = ttk.LabelFrame(self.control_frame, text="Visualization")
        self.viz_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(self.viz_frame, text="Visualization Method:").pack(anchor=tk.W, padx=5, pady=2)
        self.viz_method = tk.StringVar(value="rgb")
        methods = [("RGB", "rgb"), ("Max Projection", "max"), ("Mean Projection", "mean")]
        for text, value in methods:
            ttk.Radiobutton(self.viz_frame, text=text, variable=self.viz_method,
                            value=value, command=self.update_visualization).pack(anchor=tk.W, padx=15, pady=1)

        # Excitation wavelength selection
        self.ex_frame = ttk.Frame(self.viz_frame)
        self.ex_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(self.ex_frame, text="Excitation (nm):").pack(side=tk.LEFT, padx=5)
        self.ex_var = tk.StringVar()
        self.ex_combo = ttk.Combobox(self.ex_frame, textvariable=self.ex_var, state="readonly", width=30)
        self.ex_combo.pack(side=tk.LEFT, padx=5)
        self.ex_combo.bind("<<ComboboxSelected>>", self.on_excitation_changed)

        # Display enhancement controls
        self.enhance_frame = ttk.Frame(self.viz_frame)
        self.enhance_frame.pack(fill=tk.X, pady=5)
        ttk.Label(self.enhance_frame, text="Contrast Enhancement:").pack(anchor=tk.W, padx=5, pady=2)

        self.auto_contrast_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.enhance_frame, text="Auto Contrast", variable=self.auto_contrast_var,
                        command=self.update_visualization).pack(anchor=tk.W, padx=15, pady=1)

        # Percentile slider for contrast
        self.percentile_frame = ttk.Frame(self.viz_frame)
        self.percentile_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(self.percentile_frame, text="Percentile:").pack(side=tk.LEFT, padx=5)
        self.percentile_var = tk.IntVar(value=99)
        self.percentile_scale = ttk.Scale(self.percentile_frame, from_=50, to=100, 
                                         variable=self.percentile_var, orient=tk.HORIZONTAL,
                                         command=lambda _: self.update_visualization())
        self.percentile_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Label(self.percentile_frame, textvariable=self.percentile_var).pack(side=tk.LEFT, padx=5)

        # Spectrum display controls
        self.spectrum_frame = ttk.LabelFrame(self.control_frame, text="Spectrum")
        self.spectrum_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(self.spectrum_frame, text="Click on the image to view spectrum at that point").pack(padx=5, pady=2)
        
        # Status information
        self.status_frame = ttk.LabelFrame(self.control_frame, text="Status")
        self.status_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(self.status_frame, text="File:").pack(anchor=tk.W, padx=5)
        self.file_label = ttk.Label(self.status_frame, text="No file loaded", foreground="red")
        self.file_label.pack(anchor=tk.W, padx=5, pady=2)

        ttk.Label(self.status_frame, text="Cursor Position:").pack(anchor=tk.W, padx=5)
        self.position_label = ttk.Label(self.status_frame, text="")
        self.position_label.pack(anchor=tk.W, padx=5, pady=2)

        ttk.Label(self.status_frame, text="Pixel Value:").pack(anchor=tk.W, padx=5)
        self.value_label = ttk.Label(self.status_frame, text="")
        self.value_label.pack(anchor=tk.W, padx=5, pady=2)

        # Right side - Display area with tabs
        self.display_frame = ttk.Frame(self.main_frame)
        self.display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create notebook with tabs
        self.notebook = ttk.Notebook(self.display_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Image tab
        self.image_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.image_tab, text="Image")

        # Spectrum tab
        self.spectrum_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.spectrum_tab, text="Spectrum")

        # Setup matplotlib figures and canvases
        self.image_figure = Figure(figsize=(8, 6), dpi=100)
        self.image_ax = self.image_figure.add_subplot(111)
        self.image_canvas = FigureCanvasTkAgg(self.image_figure, master=self.image_tab)
        self.image_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.image_canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.image_canvas.mpl_connect('button_press_event', self.on_mouse_click)

        self.spectrum_figure = Figure(figsize=(8, 6), dpi=100)
        self.spectrum_ax = self.spectrum_figure.add_subplot(111)
        self.spectrum_canvas = FigureCanvasTkAgg(self.spectrum_figure, master=self.spectrum_tab)
        self.spectrum_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Initial message
        self.image_ax.text(0.5, 0.5, "Load a pickle file to begin",
                     ha='center', va='center', fontsize=14,
                     transform=self.image_ax.transAxes)
        self.image_canvas.draw()

        self.spectrum_ax.text(0.5, 0.5, "Click on the image to view spectrum",
                        ha='center', va='center', fontsize=14,
                        transform=self.spectrum_ax.transAxes)
        self.spectrum_canvas.draw()

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

        for widget in self.spectrum_frame.winfo_children():
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
            self.file_label.config(text=f"Loading {os.path.basename(file_path)}...", foreground="blue")
            self.root.update_idletasks()

            # Load the data
            with open(file_path, 'rb') as f:
                self.data_dict = pickle.load(f)

            self.file_path = file_path

            # Try to use HyperspectralDataLoader if available
            try:
                self.loader = HyperspectralDataLoader()
                self.loader.load_from_pkl(file_path)
                print("Successfully loaded data with HyperspectralDataLoader")
                
                # Populate excitation dropdown
                self.ex_combo['values'] = [str(ex) for ex in self.loader.excitation_wavelengths]
                if self.loader.excitation_wavelengths:
                    self.ex_var.set(str(self.loader.excitation_wavelengths[0]))
                    self.current_excitation = self.loader.excitation_wavelengths[0]
                
                # Enable controls
                self.toggle_controls(True)
                
                # Update visualization
                self.on_excitation_changed()
                
                self.file_label.config(text=f"Loaded {os.path.basename(file_path)}", foreground="green")
                return
                
            except Exception as e:
                print(f"Could not use HyperspectralDataLoader: {str(e)}")
                print("Falling back to manual data structure parsing")
                self.loader = None

            # Find data cubes in the structure
            self.find_data_cubes()

            if not self.cubes:
                raise ValueError("No suitable data cubes found in the pickle file")

            # Populate excitation dropdown
            self.ex_combo['values'] = list(self.cubes.keys())
            if self.cubes:
                self.ex_var.set(list(self.cubes.keys())[0])

            # Enable controls
            self.toggle_controls(True)

            # Update visualization
            self.on_excitation_changed()

            self.file_label.config(text=f"Loaded {os.path.basename(file_path)}", foreground="green")

        except Exception as e:
            error_msg = f"Failed to load file: {str(e)}"
            print(error_msg)
            print(traceback.format_exc())  # Print full traceback for debugging
            messagebox.showerror("Error", error_msg)
            self.file_label.config(text="Error loading file", foreground="red")

    def find_data_cubes(self):
        """Find all 3D data cubes in the data structure."""
        self.cubes = {}

        def _find_cubes(data, path='', max_depth=5, current_depth=0):
            if current_depth >= max_depth:
                return

            if isinstance(data, dict):
                for key, value in data.items():
                    current_path = f"{path}.{key}" if path else key

                    # Check if this is a data cube (3D numpy array)
                    if isinstance(value, np.ndarray) and len(value.shape) == 3:
                        # Store the cube
                        self.cubes[current_path] = value
                        print(f"Found data cube at {current_path}: shape={value.shape}")

                    # Recursively search nested structures
                    if isinstance(value, (dict, list)):
                        _find_cubes(value, current_path, max_depth, current_depth + 1)

            elif isinstance(data, list) and len(data) > 0:
                # For lists, only check the first item as an example
                current_path = f"{path}[0]"
                if isinstance(data[0], (dict, list, np.ndarray)):
                    _find_cubes(data[0], current_path, max_depth, current_depth + 1)

        _find_cubes(self.data_dict)

    def on_excitation_changed(self, event=None):
        """Handle excitation wavelength change."""
        if self.loader:
            # Using HyperspectralDataLoader
            excitation = float(self.ex_var.get())
            self.current_excitation = excitation
            try:
                self.current_cube, self.current_wavelengths = self.loader.get_cube(excitation)
                print(f"Loaded cube for excitation {excitation}nm with shape {self.current_cube.shape}")
                self.update_visualization()
            except Exception as e:
                print(f"Error loading cube for excitation {excitation}nm: {str(e)}")
                messagebox.showerror("Error", f"Could not load data for excitation {excitation}nm: {str(e)}")
        else:
            # Manual cube handling
            excitation_key = self.ex_var.get()
            if excitation_key in self.cubes:
                self.current_cube = self.cubes[excitation_key]
                self.current_excitation = excitation_key
                print(f"Selected cube: {excitation_key} with shape {self.current_cube.shape}")
                self.update_visualization()

    def update_visualization(self):
        """Update the visualization based on current settings."""
        if self.current_cube is None:
            return

        # Get visualization method
        method = self.viz_method.get()
        percentile = self.percentile_var.get()

        # Create RGB image
        self.rgb_image = self.create_rgb_image(self.current_cube, method, percentile)

        # Clear previous plot
        self.image_ax.clear()

        # Apply contrast enhancement
        if self.auto_contrast_var.get():
            vmin = np.percentile(self.rgb_image, 2)  # 2nd percentile to avoid outliers
            vmax = np.percentile(self.rgb_image, 98)  # 98th percentile to avoid outliers
        else:
            vmin = 0
            vmax = 1

        # Display the image with proper aspect ratio
        self.image_ax.imshow(self.rgb_image, vmin=vmin, vmax=vmax, aspect='equal')

        # Set title
        if self.loader:
            self.image_ax.set_title(f"Excitation: {self.current_excitation}nm")
        else:
            self.image_ax.set_title(f"Data-raw: {self.current_excitation}")

        # Update canvas
        self.image_canvas.draw()

        # If a pixel was previously selected, update its spectrum
        if self.selected_pixel:
            self.update_spectrum(*self.selected_pixel)

    def create_rgb_image(self, cube, method='rgb', percentile=99):
        """
        Create an RGB representation from a hyperspectral data cube.
        
        Args:
            cube: The data cube
            method: Method for creating RGB ('rgb', 'max', 'mean')
            percentile: Percentile for scaling
            
        Returns:
            RGB image as numpy array
        """
        # Check cube dimensions
        if len(cube.shape) == 3:
            # Identify which dimension is the spectral dimension
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
                # Assume standard format
                bands_dim = 0
                height_dim = 1
                width_dim = 2
                print("Assuming cube format: (bands, height, width)")
        else:
            raise ValueError(f"Unexpected cube shape: {cube.shape}. Expected 3D array.")

        # Replace NaN values
        cube = np.nan_to_num(cube)

        # Create RGB based on method
        if method == 'rgb':
            # Use three wavelengths as RGB channels
            num_bands = cube.shape[bands_dim]
            if num_bands >= 3:
                indices = [int(num_bands * 0.2), int(num_bands * 0.5), int(num_bands * 0.8)]
                r_idx, g_idx, b_idx = indices

                # Extract band images
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
            else:
                # Default to max projection if not enough bands
                method = 'max'

        if method in ['max', 'mean']:
            # Generate a projection
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

            # Create RGB by duplicating the channel
            rgb = np.stack([scaled, scaled, scaled], axis=2)

        return rgb

    def on_mouse_move(self, event):
        """Handle mouse movement over the image."""
        if event.inaxes != self.image_ax or self.rgb_image is None:
            self.position_label.config(text="")
            self.value_label.config(text="")
            return

        x, y = int(event.xdata) if event.xdata is not None else -1, int(event.ydata) if event.ydata is not None else -1

        # Check bounds
        if 0 <= y < self.rgb_image.shape[0] and 0 <= x < self.rgb_image.shape[1]:
            # Update position label
            self.position_label.config(text=f"X: {x}, Y: {y}")

            # Update value label with RGB values
            rgb_val = self.rgb_image[y, x]
            self.value_label.config(text=f"RGB: ({rgb_val[0]:.2f}, {rgb_val[1]:.2f}, {rgb_val[2]:.2f})")

    def on_mouse_click(self, event):
        """Handle mouse click to show spectrum at that point."""
        if event.inaxes != self.image_ax or self.current_cube is None:
            return

        x, y = int(event.xdata) if event.xdata is not None else -1, int(event.ydata) if event.ydata is not None else -1

        # Check bounds
        if 0 <= y < self.rgb_image.shape[0] and 0 <= x < self.rgb_image.shape[1]:
            self.selected_pixel = (x, y)
            self.update_spectrum(x, y)
            # Switch to spectrum tab
            self.notebook.select(self.spectrum_tab)

    def update_spectrum(self, x, y):
        """Update the spectrum plot for the selected pixel."""
        if self.current_cube is None:
            return

        # Clear previous plot
        self.spectrum_ax.clear()

        # Determine which dimension is the spectral dimension
        if self.current_cube.shape[0] < self.current_cube.shape[1] and self.current_cube.shape[0] < self.current_cube.shape[2]:
            # First dimension is smallest, likely spectral bands - (bands, height, width)
            spectrum = self.current_cube[:, y, x]
            bands_dim = 0
        elif self.current_cube.shape[2] < self.current_cube.shape[0] and self.current_cube.shape[2] < self.current_cube.shape[1]:
            # Last dimension is smallest, likely spectral bands - (height, width, bands)
            spectrum = self.current_cube[y, x, :]
            bands_dim = 2
        else:
            # Assume standard (bands, height, width)
            spectrum = self.current_cube[:, y, x]
            bands_dim = 0

        # Get wavelengths if available
        if self.loader and self.current_wavelengths:
            wavelengths = self.current_wavelengths
            self.spectrum_ax.plot(wavelengths, spectrum, 'b-', linewidth=2)
            self.spectrum_ax.set_xlabel('Emission Wavelength (nm)')
        else:
            # Use band indices if wavelengths not available
            band_indices = np.arange(len(spectrum))
            self.spectrum_ax.plot(band_indices, spectrum, 'b-', linewidth=2)
            self.spectrum_ax.set_xlabel('Band Index')

        self.spectrum_ax.set_ylabel('Intensity')
        self.spectrum_ax.set_title(f'Spectrum at Pixel ({x}, {y})')
        self.spectrum_ax.grid(True, alpha=0.3)

        # Update canvas
        self.spectrum_canvas.draw()

    def show_data_structure(self):
        """Show the structure of the loaded data."""
        if self.data_dict is None:
            messagebox.showinfo("Info", "No data loaded. Load a pickle file first.")
            return

        # Create a new window to display the structure
        structure_window = tk.Toplevel(self.root)
        structure_window.title("Data-raw Structure")
        structure_window.geometry("800x600")

        # Add a text widget with scrollbars
        frame = ttk.Frame(structure_window)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        scrollbar_y = ttk.Scrollbar(frame)
        scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)

        scrollbar_x = ttk.Scrollbar(frame, orient=tk.HORIZONTAL)
        scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)

        text = tk.Text(frame, wrap=tk.NONE, yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)
        text.pack(fill=tk.BOTH, expand=True)

        scrollbar_y.config(command=text.yview)
        scrollbar_x.config(command=text.xview)

        # Print the structure to the text widget
        text.insert(tk.END, "Data-raw Structure:\n\n")
        self._print_structure_to_text(self.data_dict, text)

    def _print_structure_to_text(self, obj, text, prefix='', max_depth=4, current_depth=0):
        """Print the structure of a nested object to a text widget."""
        if current_depth >= max_depth:
            text.insert(tk.END, f"{prefix}...\n")
            return

        if isinstance(obj, dict):
            for key, value in obj.items():
                value_info = f"{type(value).__name__}"
                if isinstance(value, np.ndarray):
                    value_info += f" shape={value.shape} dtype={value.dtype}"
                elif isinstance(value, (list, tuple)):
                    value_info += f" length={len(value)}"
                text.insert(tk.END, f"{prefix}{key}: {value_info}\n")

                if isinstance(value, (dict, list, tuple)) and current_depth < max_depth - 1:
                    self._print_structure_to_text(value, text, prefix + '    ', max_depth, current_depth + 1)

        elif isinstance(obj, (list, tuple)) and obj:
            text.insert(tk.END, f"{prefix}{type(obj).__name__} (length: {len(obj)})\n")
            if len(obj) > 0 and current_depth < max_depth - 1:
                # Print first item as example
                text.insert(tk.END, f"{prefix}    [0]: {type(obj[0]).__name__}\n")
                self._print_structure_to_text(obj[0], text, prefix + '        ', max_depth, current_depth + 1)
                if len(obj) > 1:
                    text.insert(tk.END, f"{prefix}    ... ({len(obj)-1} more items)\n")


def main():
    """Launch the application."""
    root = tk.Tk()
    app = HyperspectralViewer(root)
    root.mainloop()


if __name__ == "__main__":
    main()