import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector
from matplotlib.path import Path
import h5py
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import cv2
import datetime


class MaskCreationTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Hyperspectral Image Mask Creator")
        self.root.geometry("1200x800")

        # Initialize variables
        self.h5_file = None
        self.current_image = None
        self.mask = None
        self.image_shape = None
        self.polygon_vertices = []
        self.excitation_wavelengths = []
        self.current_excitation_idx = 0
        self.current_band_idx = 0
        self.current_colorbar = None
        self.current_img_plot = None
        self.current_mask_overlay = None
        self.view_average = True  # Default to viewing average cube

        # Create main frame
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create left panel for controls
        self.control_frame = tk.Frame(self.main_frame, width=200)
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        # Create right panel for image display
        self.image_frame = tk.Frame(self.main_frame)
        self.image_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Add controls to left panel
        tk.Button(self.control_frame, text="Load HDF5 File", command=self.load_h5_file).pack(fill=tk.X, pady=5)

        # Excitation wavelength selection
        self.excitation_frame = tk.Frame(self.control_frame)
        self.excitation_frame.pack(fill=tk.X, pady=5)
        tk.Label(self.excitation_frame, text="Excitation Wavelength:").pack(side=tk.LEFT)
        self.excitation_var = tk.StringVar()
        self.excitation_menu = tk.OptionMenu(self.excitation_frame, self.excitation_var, "")
        self.excitation_menu.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.excitation_var.trace("w", self.on_excitation_change)

        # Sample cube selection
        self.sample_frame = tk.Frame(self.control_frame)
        self.sample_frame.pack(fill=tk.X, pady=5)

        # Radio buttons for average vs individual cubes
        self.cube_type_var = tk.StringVar(value="average")
        tk.Radiobutton(self.sample_frame, text="Average Cube",
                       variable=self.cube_type_var, value="average",
                       command=self.on_cube_type_change).pack(anchor=tk.W)
        tk.Radiobutton(self.sample_frame, text="Individual Sample",
                       variable=self.cube_type_var, value="individual",
                       command=self.on_cube_type_change).pack(anchor=tk.W)

        # Sample index selection (initially hidden)
        self.sample_index_frame = tk.Frame(self.control_frame)
        self.sample_index_frame.pack(fill=tk.X, pady=5)
        tk.Label(self.sample_index_frame, text="Sample Index:").pack(side=tk.LEFT)
        self.sample_index_var = tk.StringVar()
        self.sample_index_menu = tk.OptionMenu(self.sample_index_frame, self.sample_index_var, "")
        self.sample_index_menu.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.sample_index_var.trace("w", self.on_sample_index_change)
        self.sample_index_frame.pack_forget()  # Initially hidden

        # Band selection
        self.band_frame = tk.Frame(self.control_frame)
        self.band_frame.pack(fill=tk.X, pady=5)
        tk.Label(self.band_frame, text="Emission Band:").pack(side=tk.LEFT)
        self.band_var = tk.StringVar()
        self.band_menu = tk.OptionMenu(self.band_frame, self.band_var, "")
        self.band_menu.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.band_var.trace("w", self.on_band_change)

        # Display enhancement controls
        self.enhance_frame = tk.Frame(self.control_frame)
        self.enhance_frame.pack(fill=tk.X, pady=5)
        tk.Label(self.enhance_frame, text="Contrast Enhancement:").pack(anchor=tk.W)

        self.auto_contrast_var = tk.BooleanVar(value=True)
        tk.Checkbutton(self.enhance_frame, text="Auto Contrast", variable=self.auto_contrast_var,
                       command=self.update_display).pack(anchor=tk.W)

        # Min/Max controls for manual contrast
        self.min_max_frame = tk.Frame(self.control_frame)
        self.min_max_frame.pack(fill=tk.X, pady=5)

        tk.Label(self.min_max_frame, text="Min:").pack(side=tk.LEFT)
        self.min_var = tk.DoubleVar(value=0.0)
        self.min_entry = tk.Entry(self.min_max_frame, textvariable=self.min_var, width=8)
        self.min_entry.pack(side=tk.LEFT)

        tk.Label(self.min_max_frame, text="Max:").pack(side=tk.LEFT)
        self.max_var = tk.DoubleVar(value=100.0)
        self.max_entry = tk.Entry(self.min_max_frame, textvariable=self.max_var, width=8)
        self.max_entry.pack(side=tk.LEFT)

        tk.Button(self.min_max_frame, text="Apply", command=self.update_display).pack(side=tk.LEFT, padx=5)

        # Drawing mode controls
        self.drawing_frame = tk.Frame(self.control_frame)
        self.drawing_frame.pack(fill=tk.X, pady=10)

        tk.Label(self.drawing_frame, text="Masking Tools:").pack(anchor=tk.W)

        self.drawing_mode_var = tk.StringVar(value="polygon")
        tk.Radiobutton(self.drawing_frame, text="Polygon Selection",
                       variable=self.drawing_mode_var, value="polygon",
                       command=self.set_drawing_mode).pack(anchor=tk.W)

        tk.Radiobutton(self.drawing_frame, text="Rectangle Selection",
                       variable=self.drawing_mode_var, value="rectangle",
                       command=self.set_drawing_mode).pack(anchor=tk.W)

        # Add mask operation buttons
        self.mask_ops_frame = tk.Frame(self.control_frame)
        self.mask_ops_frame.pack(fill=tk.X, pady=5)

        tk.Button(self.mask_ops_frame, text="Create Mask", command=self.start_drawing).pack(fill=tk.X, pady=2)
        tk.Button(self.mask_ops_frame, text="Add to Mask", command=self.add_to_mask).pack(fill=tk.X, pady=2)
        tk.Button(self.mask_ops_frame, text="Subtract from Mask", command=self.subtract_from_mask).pack(fill=tk.X,
                                                                                                        pady=2)
        tk.Button(self.mask_ops_frame, text="Clear Mask", command=self.clear_mask).pack(fill=tk.X, pady=2)

        # Polygon editing controls (initially hidden)
        self.polygon_edit_frame = tk.Frame(self.control_frame)
        tk.Label(self.polygon_edit_frame, text="Polygon Editing:").pack(anchor=tk.W)
        tk.Button(self.polygon_edit_frame, text="Remove Last Point", command=self.remove_last_polygon_point).pack(
            fill=tk.X, pady=2)
        tk.Button(self.polygon_edit_frame, text="Clear All Points", command=self.clear_polygon_points).pack(fill=tk.X,
                                                                                                            pady=2)
        tk.Button(self.polygon_edit_frame, text="Finish Polygon", command=self.finish_polygon).pack(fill=tk.X, pady=2)
        tk.Button(self.polygon_edit_frame, text="Cancel Drawing", command=self.cancel_polygon_drawing).pack(fill=tk.X,
                                                                                                            pady=2)
        # Initially hidden

        # Add save/load mask buttons
        self.save_load_frame = tk.Frame(self.control_frame)
        self.save_load_frame.pack(fill=tk.X, pady=10)

        tk.Button(self.save_load_frame, text="Save Mask", command=self.save_mask).pack(fill=tk.X, pady=2)
        tk.Button(self.save_load_frame, text="Load Mask", command=self.load_mask).pack(fill=tk.X, pady=2)
        tk.Button(self.save_load_frame, text="Add Mask to HDF5", command=self.add_mask_to_h5).pack(fill=tk.X, pady=2)

        # Status information
        self.status_frame = tk.Frame(self.control_frame)
        self.status_frame.pack(fill=tk.X, pady=10)

        tk.Label(self.status_frame, text="Mask Status:").pack(anchor=tk.W)
        self.status_label = tk.Label(self.status_frame, text="No mask created", fg="red")
        self.status_label.pack(anchor=tk.W)

        # Value display (for pixel under cursor)
        self.value_frame = tk.Frame(self.control_frame)
        self.value_frame.pack(fill=tk.X, pady=5)

        tk.Label(self.value_frame, text="Cursor Position:").pack(anchor=tk.W)
        self.position_label = tk.Label(self.value_frame, text="")
        self.position_label.pack(anchor=tk.W)

        tk.Label(self.value_frame, text="Pixel Value:").pack(anchor=tk.W)
        self.value_label = tk.Label(self.value_frame, text="")
        self.value_label.pack(anchor=tk.W)

        # Setup matplotlib figure for image display
        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.image_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Connect mouse events
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.canvas.mpl_connect('key_press_event', self.on_key_press)

        # Initialize polygon selector (will be activated later)
        self.polygon_selector = None
        self.drawing_active = False

        # Display initial message
        self.ax.text(0.5, 0.5, "Load an HDF5 file to begin",
                     ha='center', va='center', fontsize=12,
                     transform=self.ax.transAxes)
        self.canvas.draw()

    def load_h5_file(self):
        """Open an HDF5 file and load data"""
        file_path = filedialog.askopenfilename(
            title="Select HDF5 File",
            filetypes=[("HDF5 files", "*.h5"), ("All files", "*.*")]
        )

        if not file_path:
            return

        try:
            self.h5_file = h5py.File(file_path, 'r')

            # Get list of excitation wavelengths
            self.excitation_wavelengths = []
            for group_name in self.h5_file.keys():
                if group_name.startswith('excitation_'):
                    self.excitation_wavelengths.append(int(group_name.split('_')[1]))

            self.excitation_wavelengths.sort()

            # Update excitation menu
            menu = self.excitation_menu["menu"]
            menu.delete(0, "end")
            for excitation in self.excitation_wavelengths:
                menu.add_command(label=str(excitation),
                                 command=lambda value=excitation: self.excitation_var.set(str(value)))

            if self.excitation_wavelengths:
                self.excitation_var.set(str(self.excitation_wavelengths[0]))
                self.on_excitation_change()

            # Show a message
            messagebox.showinfo("Success", f"Loaded HDF5 file: {os.path.basename(file_path)}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load HDF5 file: {str(e)}")

    def on_excitation_change(self, *args):
        """Handle change in excitation wavelength selection"""
        if self.h5_file is None or not self.excitation_var.get():
            return

        excitation = int(self.excitation_var.get())
        self.current_excitation_idx = self.excitation_wavelengths.index(excitation)

        # Get group for this excitation
        group_name = f'excitation_{excitation}'
        group = self.h5_file[group_name]

        # Get wavelengths for this excitation
        wavelengths = group['wavelengths'][:]

        # Update band menu
        menu = self.band_menu["menu"]
        menu.delete(0, "end")
        for i, wavelength in enumerate(wavelengths):
            menu.add_command(label=f"{wavelength:.1f}",
                             command=lambda value=i, wl=wavelength:
                             self.band_var.set(f"{wl:.1f} (Band {value})"))

        # Select middle band by default
        middle_band = len(wavelengths) // 2
        self.current_band_idx = middle_band
        self.band_var.set(f"{wavelengths[middle_band]:.1f} (Band {middle_band})")

        # Update sample index menu
        sample_indices = []
        for key in group.keys():
            if key.startswith('cube_'):
                sample_indices.append(int(key.split('_')[1]))

        sample_indices.sort()

        menu = self.sample_index_menu["menu"]
        menu.delete(0, "end")
        for idx in sample_indices:
            menu.add_command(label=str(idx),
                             command=lambda value=idx: self.sample_index_var.set(str(value)))

        if sample_indices:
            self.sample_index_var.set(str(sample_indices[0]))

        # Load data based on current view setting
        self.load_current_image()

    def on_cube_type_change(self):
        """Handle change between average and individual cube views"""
        cube_type = self.cube_type_var.get()

        if cube_type == "average":
            self.view_average = True
            self.sample_index_frame.pack_forget()  # Hide sample index selector
        else:
            self.view_average = False
            self.sample_index_frame.pack(fill=tk.X, pady=5, after=self.sample_frame)  # Show sample index selector

        # Load appropriate data
        self.load_current_image()

    def on_sample_index_change(self, *args):
        """Handle change in sample index selection"""
        if self.view_average:
            return  # Ignore if we're viewing the average cube

        self.load_current_image()

    def load_current_image(self):
        """Load the current image based on selections"""
        if self.h5_file is None or not self.excitation_var.get():
            return

        excitation = int(self.excitation_var.get())
        group_name = f'excitation_{excitation}'
        group = self.h5_file[group_name]

        if self.view_average:
            # Load average cube for this excitation
            if 'average_cube' in group:
                self.current_image = group['average_cube'][:]
                self.image_shape = self.current_image.shape[1:3]  # Height, Width

                # Initialize mask if not created yet
                if self.mask is None:
                    self.mask = np.zeros(self.image_shape, dtype=np.uint8)

                # Update display
                self.update_display()
            else:
                messagebox.showwarning("Warning", "No average cube found for this excitation!")
        else:
            # Load individual sample cube
            if not self.sample_index_var.get():
                return

            sample_index = int(self.sample_index_var.get())
            cube_name = f'cube_{sample_index}'

            if cube_name in group:
                self.current_image = group[cube_name]['data'][:]
                self.image_shape = self.current_image.shape[1:3]  # Height, Width

                # Initialize mask if not created yet
                if self.mask is None:
                    self.mask = np.zeros(self.image_shape, dtype=np.uint8)

                # Update display
                self.update_display()
            else:
                messagebox.showwarning("Warning", f"No cube found with index {sample_index}!")

    def on_band_change(self, *args):
        """Handle change in emission band selection"""
        if self.current_image is None or not self.band_var.get():
            return

        # Extract band index from the selection
        band_str = self.band_var.get()
        if 'Band' in band_str:
            band_idx = int(band_str.split('Band ')[1].rstrip(')'))
            self.current_band_idx = band_idx
            self.update_display()

    def safe_remove_colorbar(self):
        """Safely remove the colorbar with proper error handling"""
        if hasattr(self, 'current_colorbar') and self.current_colorbar is not None:
            try:
                # Try to remove the colorbar
                self.current_colorbar.remove()
            except (AttributeError, ValueError, TypeError) as e:
                # If there's an error, just log it and continue
                print(f"Warning: Could not remove colorbar: {str(e)}")
            finally:
                # Always set to None after attempting removal
                self.current_colorbar = None

    def update_display(self):
        """Update the displayed image"""
        if self.current_image is None:
            return

        # Get current band image
        band_image = self.current_image[self.current_band_idx]

        # Apply contrast enhancement
        if self.auto_contrast_var.get():
            vmin = np.percentile(band_image, 2)  # 2nd percentile to avoid outliers
            vmax = np.percentile(band_image, 98)  # 98th percentile to avoid outliers
        else:
            vmin = self.min_var.get()
            vmax = self.max_var.get()

        # Safely remove existing colorbar and clear axis
        self.safe_remove_colorbar()

        # Clear previous plot
        self.ax.clear()

        # Display the image
        self.current_img_plot = self.ax.imshow(band_image, cmap='viridis', vmin=vmin, vmax=vmax)

        # Add colorbar
        self.current_colorbar = self.figure.colorbar(self.current_img_plot, ax=self.ax, orientation='vertical',
                                                     label='Intensity')

        # If mask exists, overlay it
        if self.mask is not None and np.any(self.mask):
            # Create RGBA mask overlay: red with alpha channel
            mask_overlay = np.zeros((*self.mask.shape, 4), dtype=np.float32)
            # Red color with 50% transparency where mask is 1
            mask_overlay[self.mask == 1, 0] = 1.0  # Red channel
            mask_overlay[self.mask == 1, 3] = 0.5  # Alpha channel

            self.current_mask_overlay = self.ax.imshow(mask_overlay, interpolation='nearest')

            # Update status
            masked_pixels = np.sum(self.mask)
            total_pixels = self.mask.size
            percent_masked = (masked_pixels / total_pixels) * 100
            self.status_label.config(text=f"Masked: {masked_pixels} pixels ({percent_masked:.1f}%)", fg="green")
        else:
            self.status_label.config(text="No mask created", fg="red")

        # Set title
        excitation = self.excitation_var.get()
        band_str = self.band_var.get()
        cube_type = "Average Cube" if self.view_average else f"Sample {self.sample_index_var.get()}"
        self.ax.set_title(f"Excitation: {excitation} nm, Emission: {band_str}\n{cube_type}")

        # Refresh canvas
        self.canvas.draw()

    def on_mouse_move(self, event):
        """Handle mouse movement over the image"""
        if event.inaxes != self.ax or self.current_image is None:
            self.position_label.config(text="")
            self.value_label.config(text="")
            return

        x, y = int(event.xdata), int(event.ydata)

        # Check bounds
        if 0 <= y < self.current_image.shape[1] and 0 <= x < self.current_image.shape[2]:
            # Update position and value labels
            self.position_label.config(text=f"X: {x}, Y: {y}")
            value = self.current_image[self.current_band_idx, y, x]
            self.value_label.config(text=f"Value: {value:.2f}")

            # Show mask state if mask exists
            if self.mask is not None:
                mask_state = "Masked" if self.mask[y, x] == 1 else "Unmasked"
                self.value_label.config(text=f"Value: {value:.2f} ({mask_state})")

    def on_key_press(self, event):
        """Handle keyboard shortcuts during polygon editing"""
        if not self.drawing_active or self.drawing_mode_var.get() != "polygon":
            return

        if event.key == 'backspace' or event.key == 'delete':
            self.remove_last_polygon_point()
        elif event.key == 'escape':
            self.cancel_polygon_drawing()
        elif event.key == 'enter':
            # Finalize polygon and create mask when Enter is pressed
            self.finish_polygon()

    def set_drawing_mode(self):
        """Set the current drawing mode"""
        # This will be used when starting drawing
        pass

    def start_drawing(self):
        """Start drawing a new mask"""
        if self.current_image is None:
            messagebox.showwarning("Warning", "Load an image first!")
            return

        # Clear existing mask
        self.mask = np.zeros(self.image_shape, dtype=np.uint8)
        self.drawing_active = True

        # Choose drawing method based on mode
        drawing_mode = self.drawing_mode_var.get()

        if drawing_mode == "polygon":
            # Remove existing selector if any
            if self.polygon_selector is not None:
                self.polygon_selector.disconnect_events()

            # Clear polygon vertices
            self.polygon_vertices = []

            # Create new polygon selector
            self.polygon_selector = PolygonSelector(
                self.ax, self.on_polygon_selected,
                useblit=True,
                props=dict(color='red', linestyle='-', linewidth=2, alpha=0.8),
                handle_props=dict(color='red', markersize=8, mfc='white')
            )

            # Show polygon editing controls
            self.polygon_edit_frame.pack(fill=tk.X, pady=5, after=self.mask_ops_frame)

            messagebox.showinfo("Draw Polygon",
                                "Click to add polygon vertices.\n"
                                "When finished, click 'Finish Polygon' or press Enter.\n"
                                "Keyboard shortcuts:\n"
                                "Enter: Finish polygon\n"
                                "Backspace/Delete: Remove last point\n"
                                "Escape: Cancel drawing")

        elif drawing_mode == "rectangle":
            # Create rectangle selection dialog
            self.draw_rectangle_dialog()

    def on_polygon_selected(self, vertices):
        """Handle polygon selection"""
        self.polygon_vertices = vertices

        # Display current vertices count
        num_vertices = len(vertices)
        self.status_label.config(text=f"Drawing polygon: {num_vertices} points", fg="blue")

    def remove_last_polygon_point(self):
        """Remove the last point added to the polygon"""
        if not self.drawing_active or self.drawing_mode_var.get() != "polygon":
            return

        if self.polygon_selector is not None and hasattr(self.polygon_selector, 'verts') and len(
                self.polygon_selector.verts) > 0:
            # Remove last point from the selector
            self.polygon_selector.verts.pop()

            # Force redraw
            self.polygon_selector._update_verts()
            self.canvas.draw_idle()

            # Also update our cached vertices
            if self.polygon_vertices and len(self.polygon_vertices) > 0:
                self.polygon_vertices = self.polygon_vertices[:-1]

            # Update status
            num_vertices = len(self.polygon_selector.verts)
            self.status_label.config(text=f"Drawing polygon: {num_vertices} points", fg="blue")

    def clear_polygon_points(self):
        """Clear all points in the current polygon"""
        if not self.drawing_active or self.drawing_mode_var.get() != "polygon":
            return

        if self.polygon_selector is not None:
            # Clear all vertices
            self.polygon_selector.verts = []

            # Force redraw
            self.polygon_selector._update_verts()
            self.canvas.draw_idle()

            # Also clear our cached vertices
            self.polygon_vertices = []

            # Update status
            self.status_label.config(text="Drawing polygon: 0 points", fg="blue")

    def finish_polygon(self):
        """Finalize polygon and create mask"""
        if not self.drawing_active or self.drawing_mode_var.get() != "polygon":
            return

        if not self.polygon_vertices or len(self.polygon_vertices) < 3:
            messagebox.showwarning("Warning", "Need at least 3 points to create a polygon mask")
            return

        # Convert vertices to numpy array
        vertices = np.array(self.polygon_vertices)

        # Create a grid of pixel coordinates
        y, x = np.mgrid[:self.image_shape[0], :self.image_shape[1]]
        points = np.vstack((x.flatten(), y.flatten())).T

        # Create mask from polygon
        path = Path(vertices)
        mask = path.contains_points(points)
        mask = mask.reshape(self.image_shape)

        # Set this as the mask (replacing any existing mask)
        self.mask = mask.astype(np.uint8)

        # Clean up polygon selector
        if self.polygon_selector is not None:
            self.polygon_selector.disconnect_events()
            self.polygon_selector = None

        # Hide polygon editing controls
        self.polygon_edit_frame.pack_forget()

        # End drawing mode
        self.drawing_active = False

        # Update display
        self.update_display()

        # Show confirmation
        messagebox.showinfo("Success", "Polygon mask created successfully")

    def cancel_polygon_drawing(self):
        """Cancel the current polygon drawing operation"""
        if not self.drawing_active or self.drawing_mode_var.get() != "polygon":
            return

        # Clear the polygon
        self.clear_polygon_points()

        # Disconnect the selector
        if self.polygon_selector is not None:
            self.polygon_selector.disconnect_events()
            self.polygon_selector = None

        # Hide polygon editing controls
        self.polygon_edit_frame.pack_forget()

        # End drawing mode
        self.drawing_active = False

        # Reset vertices
        self.polygon_vertices = []

        # Update display
        self.update_display()

    def add_to_mask(self):
        """Add current selection to mask"""
        if not self.drawing_active:
            messagebox.showinfo("Info", "Start drawing first!")
            return

        drawing_mode = self.drawing_mode_var.get()

        if drawing_mode == "polygon" and self.polygon_vertices:
            if len(self.polygon_vertices) < 3:
                messagebox.showwarning("Warning", "Need at least 3 points to create a polygon mask")
                return

            # Convert vertices to numpy array
            vertices = np.array(self.polygon_vertices)

            # Create a grid of pixel coordinates
            y, x = np.mgrid[:self.image_shape[0], :self.image_shape[1]]
            points = np.vstack((x.flatten(), y.flatten())).T

            # Create mask from polygon
            path = Path(vertices)
            new_mask = path.contains_points(points)
            new_mask = new_mask.reshape(self.image_shape)

            # Add to existing mask (logical OR)
            self.mask = np.logical_or(self.mask, new_mask).astype(np.uint8)

            # Clear polygon selection
            self.polygon_vertices = []
            if self.polygon_selector is not None:
                self.polygon_selector.disconnect_events()
                self.polygon_selector = None

            # Hide polygon editing controls
            self.polygon_edit_frame.pack_forget()

            # Update display
            self.drawing_active = False
            self.update_display()

            # Show confirmation
            messagebox.showinfo("Success", "Region added to mask")

    def subtract_from_mask(self):
        """Subtract current selection from mask"""
        if not self.drawing_active:
            messagebox.showinfo("Info", "Start drawing first!")
            return

        drawing_mode = self.drawing_mode_var.get()

        if drawing_mode == "polygon" and self.polygon_vertices:
            if len(self.polygon_vertices) < 3:
                messagebox.showwarning("Warning", "Need at least 3 points to create a polygon mask")
                return

            # Convert vertices to numpy array
            vertices = np.array(self.polygon_vertices)

            # Create a grid of pixel coordinates
            y, x = np.mgrid[:self.image_shape[0], :self.image_shape[1]]
            points = np.vstack((x.flatten(), y.flatten())).T

            # Create mask from polygon
            path = Path(vertices)
            subtract_mask = path.contains_points(points)
            subtract_mask = subtract_mask.reshape(self.image_shape)

            # Subtract from existing mask (logical AND with NOT)
            self.mask = np.logical_and(self.mask, ~subtract_mask).astype(np.uint8)

            # Clear polygon selection
            self.polygon_vertices = []
            if self.polygon_selector is not None:
                self.polygon_selector.disconnect_events()
                self.polygon_selector = None

            # Hide polygon editing controls
            self.polygon_edit_frame.pack_forget()

            # Update display
            self.drawing_active = False
            self.update_display()

            # Show confirmation
            messagebox.showinfo("Success", "Region subtracted from mask")

    def draw_rectangle_dialog(self):
        """Open dialog for rectangle selection"""
        rect_dialog = tk.Toplevel(self.root)
        rect_dialog.title("Rectangle Selection")
        rect_dialog.geometry("300x200")

        tk.Label(rect_dialog, text="Enter rectangle coordinates:").pack(pady=5)

        # X start
        x_start_frame = tk.Frame(rect_dialog)
        x_start_frame.pack(fill=tk.X, pady=2)
        tk.Label(x_start_frame, text="X start:").pack(side=tk.LEFT)
        x_start_var = tk.IntVar(value=0)
        tk.Entry(x_start_frame, textvariable=x_start_var).pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Y start
        y_start_frame = tk.Frame(rect_dialog)
        y_start_frame.pack(fill=tk.X, pady=2)
        tk.Label(y_start_frame, text="Y start:").pack(side=tk.LEFT)
        y_start_var = tk.IntVar(value=0)
        tk.Entry(y_start_frame, textvariable=y_start_var).pack(side=tk.LEFT, fill=tk.X, expand=True)

        # X end
        x_end_frame = tk.Frame(rect_dialog)
        x_end_frame.pack(fill=tk.X, pady=2)
        tk.Label(x_end_frame, text="X end:").pack(side=tk.LEFT)
        x_end_var = tk.IntVar(value=self.image_shape[1] - 1 if self.image_shape else 0)
        tk.Entry(x_end_frame, textvariable=x_end_var).pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Y end
        y_end_frame = tk.Frame(rect_dialog)
        y_end_frame.pack(fill=tk.X, pady=2)
        tk.Label(y_end_frame, text="Y end:").pack(side=tk.LEFT)
        y_end_var = tk.IntVar(value=self.image_shape[0] - 1 if self.image_shape else 0)
        tk.Entry(y_end_frame, textvariable=y_end_var).pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Function to create rectangle mask
        def create_rect_mask():
            x_start = max(0, min(x_start_var.get(), self.image_shape[1] - 1))
            y_start = max(0, min(y_start_var.get(), self.image_shape[0] - 1))
            x_end = max(0, min(x_end_var.get(), self.image_shape[1] - 1))
            y_end = max(0, min(y_end_var.get(), self.image_shape[0] - 1))

            # Ensure start < end
            x_start, x_end = min(x_start, x_end), max(x_start, x_end)
            y_start, y_end = min(y_start, y_end), max(y_start, y_end)

            # Create rectangle mask
            rect_mask = np.zeros(self.image_shape, dtype=np.uint8)
            rect_mask[y_start:y_end + 1, x_start:x_end + 1] = 1

            # Add to existing mask
            self.mask = np.logical_or(self.mask, rect_mask).astype(np.uint8)

            # Update display
            self.drawing_active = False
            self.update_display()
            rect_dialog.destroy()

            # Show confirmation
            messagebox.showinfo("Success", "Rectangle mask created successfully")

        # Buttons
        buttons_frame = tk.Frame(rect_dialog)
        buttons_frame.pack(fill=tk.X, pady=10)
        tk.Button(buttons_frame, text="Cancel", command=rect_dialog.destroy).pack(side=tk.LEFT, padx=5)
        tk.Button(buttons_frame, text="Create", command=create_rect_mask).pack(side=tk.RIGHT, padx=5)

    def clear_mask(self):
        """Clear the current mask"""
        if self.mask is not None:
            self.mask = np.zeros(self.image_shape, dtype=np.uint8)
            self.update_display()
            self.drawing_active = False
            messagebox.showinfo("Info", "Mask cleared")

    def save_mask(self):
        """Save the current mask to a file"""
        if self.mask is None or not np.any(self.mask):
            messagebox.showwarning("Warning", "No mask to save!")
            return

        file_path = filedialog.asksaveasfilename(
            title="Save Mask",
            defaultextension=".npy",
            filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")]
        )

        if file_path:
            try:
                np.save(file_path, self.mask)
                messagebox.showinfo("Success", f"Mask saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save mask: {str(e)}")

    def load_mask(self):
        """Load a mask from a file"""
        file_path = filedialog.askopenfilename(
            title="Load Mask",
            filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")]
        )

        if file_path:
            try:
                loaded_mask = np.load(file_path)

                # Check if mask size matches image size
                if loaded_mask.shape != self.image_shape:
                    messagebox.showerror("Error",
                                         f"Mask size ({loaded_mask.shape}) doesn't match image size ({self.image_shape})")
                    return

                self.mask = loaded_mask
                self.update_display()
                messagebox.showinfo("Success", f"Mask loaded from {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load mask: {str(e)}")

    def add_mask_to_h5(self):
        """Add the current mask to the HDF5 file"""
        if self.mask is None or not np.any(self.mask):
            messagebox.showwarning("Warning", "No mask to add!")
            return

        # Ask for the output HDF5 file
        file_path = filedialog.asksaveasfilename(
            title="Save HDF5 with Mask",
            defaultextension=".h5",
            filetypes=[("HDF5 files", "*.h5"), ("All files", "*.*")]
        )

        if not file_path:
            return

        try:
            # Open the input HDF5 file for reading
            if self.h5_file is None:
                messagebox.showerror("Error", "No HDF5 file loaded!")
                return

            # Create a new HDF5 file
            with h5py.File(file_path, 'w') as out_h5:
                # Copy all data from original file
                for key in self.h5_file.keys():
                    self.h5_file.copy(key, out_h5)

                # Add mask as a new group
                mask_group = out_h5.create_group('mask')
                mask_group.create_dataset('data', data=self.mask, compression='gzip')

                # Add metadata for the mask - using strings instead of np.string_
                mask_group.attrs['description'] = 'Binary mask for region of interest'
                mask_group.attrs['creation_date'] = datetime.datetime.now().isoformat()
                mask_group.attrs['num_masked_pixels'] = np.sum(self.mask)
                mask_group.attrs['total_pixels'] = self.mask.size
                mask_group.attrs['percent_masked'] = (np.sum(self.mask) / self.mask.size) * 100

            messagebox.showinfo("Success", f"Mask added to HDF5 file: {file_path}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to add mask to HDF5: {str(e)}")


def main():
    root = tk.Tk()
    app = MaskCreationTool(root)
    root.mainloop()


if __name__ == "__main__":
    main()