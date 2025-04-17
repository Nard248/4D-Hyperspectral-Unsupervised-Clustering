# eem_toolkit.py
"""
EEM Visualization Toolkit

A module for creating interactive visualizations of Excitation-Emission Matrix (EEM) data.
Provides tools for processing, visualizing, and comparing EEM datasets.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not installed. Install with: pip install plotly")


class EEMVisualizer:
    """Toolkit for visualizing and comparing Excitation-Emission Matrix data"""

    def __init__(self):
        """Initialize the EEM Visualizer"""
        self.check_dependencies()

    def check_dependencies(self):
        """Check if required dependencies are installed"""
        if not PLOTLY_AVAILABLE:
            print("Warning: Plotly is required for interactive visualizations.")
            print("Install with: pip install plotly")

    def process_hyperspectral_data(self, data, cutoff_offset=30, resolution=2,
                                   reflectance_range=(400, 500), exclude_keys=None):
        """
        Process hyperspectral data into an EEM format with cutoff masks

        Args:
            data: Dictionary containing hyperspectral data (from pickle file)
            cutoff_offset: Offset in nm for cutoff calculations
            resolution: Resolution in nm for the emission grid
            reflectance_range: Valid range for reflectance extraction
            exclude_keys: List of keys to exclude (e.g., ['Reflectance'])

        Returns:
            dict: Processed EEM data with visualization-ready arrays
        """
        # Handle input variations
        if isinstance(data, dict) and 'data' in data and 'metadata' in data:
            # Data is in the format saved by our processor
            hyperspectral_data = data['data']
            metadata = data['metadata']
        else:
            # Assume data is already the hyperspectral data dict
            hyperspectral_data = data
            metadata = {'cutoff_offset': cutoff_offset, 'reflectance_range': reflectance_range}

        # Exclude specified keys
        if exclude_keys is None:
            exclude_keys = ['Reflectance']

        # Get excitation wavelengths (excluding specified keys)
        excitation_keys = [key for key in hyperspectral_data.keys() if key not in exclude_keys]
        excitation_wavelengths = [float(key) for key in excitation_keys]
        excitation_wavelengths.sort()

        # Dictionary to store emission data for each excitation
        emission_data = {}

        for ex_key in excitation_keys:
            ex_data = hyperspectral_data[ex_key]

            # Handle different data formats
            if isinstance(ex_data, dict) and 'cube' in ex_data:
                wavelengths = ex_data['wavelengths']
                cube = ex_data['cube']
            else:
                # Assume ex_data is the cube itself
                # This would need adjusting based on your actual data structure
                wavelengths = np.arange(ex_data.shape[0])  # Placeholder
                cube = ex_data

            # Calculate mean spectrum
            mean_spectrum = np.mean(cube, axis=(1, 2))

            emission_data[float(ex_key)] = {
                'wavelengths': wavelengths,
                'intensity': mean_spectrum
            }

        # Find min and max emission wavelengths
        all_emission_wavelengths = []
        for ex, data in emission_data.items():
            all_emission_wavelengths.extend(data['wavelengths'])

        min_emission = min(all_emission_wavelengths)
        max_emission = max(all_emission_wavelengths)

        # Create a regular grid for the EEM
        emission_grid = np.arange(min_emission, max_emission + resolution, resolution)

        # Create the EEM matrix and cutoff mask
        eem_matrix = np.zeros((len(excitation_wavelengths), len(emission_grid)))
        cutoff_mask = np.ones_like(eem_matrix, dtype=bool)

        # Fill the EEM matrix
        for i, ex in enumerate(excitation_wavelengths):
            ex_key = str(ex)
            ex_data = emission_data[ex]

            # Get this excitation's emission wavelengths and intensity
            ex_emission_wavelengths = ex_data['wavelengths']
            ex_intensity = ex_data['intensity']

            # Interpolate onto the common grid
            for j, em in enumerate(emission_grid):
                # If this emission wavelength is within the available range
                if em >= ex_emission_wavelengths[0] and em <= ex_emission_wavelengths[-1]:
                    # Find the two closest points for interpolation
                    idx = np.searchsorted(ex_emission_wavelengths, em)
                    if idx == 0:
                        # At the beginning
                        eem_matrix[i, j] = ex_intensity[0]
                    elif idx == len(ex_emission_wavelengths):
                        # At the end
                        eem_matrix[i, j] = ex_intensity[-1]
                    else:
                        # Interpolate between two points
                        weight = (em - ex_emission_wavelengths[idx - 1]) / (
                                    ex_emission_wavelengths[idx] - ex_emission_wavelengths[idx - 1])
                        eem_matrix[i, j] = ex_intensity[idx - 1] * (1 - weight) + ex_intensity[idx] * weight
                else:
                    eem_matrix[i, j] = 0  # Outside the available range

                # Calculate cutoff mask
                # 1. Primary cutoff: emission < excitation + cutoff_offset
                if em < (ex + cutoff_offset):
                    cutoff_mask[i, j] = False

                # 2. Second-order cutoff: emission within ±cutoff_offset of 2*excitation
                second_order_min = 2 * ex - cutoff_offset
                second_order_max = 2 * ex + cutoff_offset
                if em >= second_order_min and em <= second_order_max:
                    cutoff_mask[i, j] = False

        # Prepare the result dictionary
        result = {
            'eem_matrix': eem_matrix,
            'cutoff_mask': cutoff_mask,
            'excitation_wavelengths': excitation_wavelengths,
            'emission_grid': emission_grid,
            'cutoff_offset': cutoff_offset,
            'resolution': resolution
        }

        return result

    def create_interactive_3d_eem(self, eem_data, title=None, width=950, height=750,
                                  include_cutoff=True, colorscale='Viridis', show_instructions=True):
        """
        Create an interactive 3D visualization of EEM data

        Args:
            eem_data: Dict containing processed EEM data
            title: Title for the plot (default: "Interactive 3D EEM")
            width, height: Dimensions of the plot
            include_cutoff: Whether to show the cutoff regions
            colorscale: Colorscale to use for the main surface
            show_instructions: Whether to show usage instructions

        Returns:
            fig: Plotly figure object with the interactive 3D surface
        """
        if not PLOTLY_AVAILABLE:
            print("Error: Plotly is required for interactive visualizations.")
            print("Install with: pip install plotly")
            return None

        # Extract data from the EEM data dict
        eem_matrix = eem_data['eem_matrix']
        cutoff_mask = eem_data['cutoff_mask']
        excitation_wavelengths = eem_data['excitation_wavelengths']
        emission_grid = eem_data['emission_grid']
        cutoff_offset = eem_data.get('cutoff_offset', 30)

        # Set default title if not provided
        if title is None:
            title = "Interactive 3D Excitation-Emission Matrix (EEM)"

        # Apply cutoff mask to the data if requested
        if include_cutoff:
            masked_eem = np.copy(eem_matrix)
            masked_eem[~cutoff_mask] = np.nan  # Set cutoff regions to NaN
        else:
            masked_eem = eem_matrix

        # Create meshgrid for 3D plot
        X, Y = np.meshgrid(emission_grid, excitation_wavelengths)

        # Create the 3D surface plot
        fig = go.Figure()

        # Add the main surface with valid data
        fig.add_trace(go.Surface(
            x=X,
            y=Y,
            z=masked_eem,
            colorscale=colorscale,
            colorbar=dict(title='Intensity'),
            opacity=0.9,
            name='Fluorescence Intensity',
            hovertemplate='Emission: %{x:.1f} nm<br>Excitation: %{y:.1f} nm<br>Intensity: %{z:.4f}<extra></extra>'
        ))

        # Add cutoff regions as a separate surface if requested
        if include_cutoff:
            cutoff_eem = np.zeros_like(eem_matrix)
            cutoff_level = np.min(eem_matrix[eem_matrix > 0]) * 0.2  # Low level for visualization
            cutoff_eem[~cutoff_mask] = cutoff_level

            fig.add_trace(go.Surface(
                x=X,
                y=Y,
                z=cutoff_eem,
                colorscale=[[0, 'rgba(255,0,0,0.3)'], [1, 'rgba(255,0,0,0.3)']],
                showscale=False,
                opacity=0.5,
                name='Cutoff Regions',
                hovertemplate='Emission: %{x:.1f} nm<br>Excitation: %{y:.1f} nm<br>Status: Cutoff Region<extra></extra>'
            ))

        # Update layout
        fig.update_layout(
            title={
                'text': title,
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 20}
            },
            scene=dict(
                xaxis_title='Emission Wavelength (nm)',
                yaxis_title='Excitation Wavelength (nm)',
                zaxis_title='Intensity',
                aspectratio=dict(x=1.5, y=1, z=0.7),
                camera=dict(
                    eye=dict(x=1.8, y=1.8, z=1.2)
                )
            ),
            width=width,
            height=height,
            margin=dict(l=65, r=50, b=65, t=90),
        )

        # Add explanatory text for cutoff regions
        if include_cutoff:
            fig.add_annotation(
                x=0.01,
                y=0.01,
                xref="paper",
                yref="paper",
                text=f"<b>Cutoff regions (in red):</b><br>1. Primary: Em < Ex + {cutoff_offset}nm<br>" +
                     f"2. Second-order: Em within ±{cutoff_offset}nm of 2·Ex",
                showarrow=False,
                font=dict(size=12),
                bgcolor="white",
                opacity=0.8,
                bordercolor="black",
                borderwidth=1,
                borderpad=4
            )

        # Add usage instructions if requested
        if show_instructions:
            fig.add_annotation(
                x=0.99,
                y=0.99,
                xref="paper",
                yref="paper",
                text="<b>Interactive Controls:</b><br>• Click and drag to rotate<br>• Scroll to zoom<br>" +
                     "• Right-click and drag to pan<br>• Double-click to reset view<br>• Hover for data values",
                showarrow=False,
                font=dict(size=12),
                align="right",
                bgcolor="white",
                opacity=0.8,
                bordercolor="black",
                borderwidth=1,
                borderpad=4
            )

        return fig

    def create_interactive_2d_eem(self, eem_data, title=None, width=950, height=750,
                                  include_cutoff=True, show_diagonal_lines=True, colorscale='Viridis'):
        """
        Create an interactive 2D heatmap of EEM data

        Args:
            eem_data: Dict containing processed EEM data
            title: Title for the plot
            width, height: Dimensions of the plot
            include_cutoff: Whether to show the cutoff regions
            show_diagonal_lines: Whether to show diagonal cutoff lines
            colorscale: Colorscale to use for the main heatmap

        Returns:
            fig: Plotly figure object with the interactive 2D heatmap
        """
        if not PLOTLY_AVAILABLE:
            print("Error: Plotly is required for interactive visualizations.")
            print("Install with: pip install plotly")
            return None

        # Extract data from the EEM data dict
        eem_matrix = eem_data['eem_matrix']
        cutoff_mask = eem_data['cutoff_mask']
        excitation_wavelengths = eem_data['excitation_wavelengths']
        emission_grid = eem_data['emission_grid']
        cutoff_offset = eem_data.get('cutoff_offset', 30)

        # Set default title if not provided
        if title is None:
            title = "Interactive EEM Heatmap"

        # Apply cutoff mask to the data if requested
        if include_cutoff:
            masked_eem = np.copy(eem_matrix)
            masked_eem[~cutoff_mask] = np.nan  # Set cutoff regions to NaN
        else:
            masked_eem = eem_matrix

        # Create the 2D heatmap
        fig = go.Figure(data=go.Heatmap(
            z=masked_eem,
            x=emission_grid,
            y=excitation_wavelengths,
            colorscale=colorscale,
            colorbar=dict(title='Intensity'),
            hovertemplate='Excitation: %{y:.1f} nm<br>Emission: %{x:.1f} nm<br>Intensity: %{z:.4f}<extra></extra>'
        ))

        # Mark cutoff regions if requested
        if include_cutoff:
            cutoff_display = np.where(~cutoff_mask, 1, np.nan)

            fig.add_trace(go.Heatmap(
                z=cutoff_display,
                x=emission_grid,
                y=excitation_wavelengths,
                colorscale=[[0, 'rgba(255,0,0,0.3)'], [1, 'rgba(255,0,0,0.3)']],
                showscale=False,
                hovertemplate='Excitation: %{y:.1f} nm<br>Emission: %{x:.1f} nm<br>Status: Cutoff region<extra></extra>'
            ))

        # Add diagonal lines showing the cutoff boundaries if requested
        if show_diagonal_lines and include_cutoff:
            ex_range = np.linspace(min(excitation_wavelengths), max(excitation_wavelengths), 100)

            # Primary cutoff line
            fig.add_trace(go.Scatter(
                x=ex_range + cutoff_offset,
                y=ex_range,
                mode='lines',
                line=dict(color='red', width=2, dash='solid'),
                name=f'Primary Cutoff (Ex + {cutoff_offset}nm)'
            ))

            # Second-order cutoff lines
            fig.add_trace(go.Scatter(
                x=2 * ex_range - cutoff_offset,
                y=ex_range,
                mode='lines',
                line=dict(color='red', width=2, dash='dash'),
                name=f'Second-Order (2·Ex - {cutoff_offset}nm)'
            ))

            fig.add_trace(go.Scatter(
                x=2 * ex_range + cutoff_offset,
                y=ex_range,
                mode='lines',
                line=dict(color='red', width=2, dash='dash'),
                name=f'Second-Order (2·Ex + {cutoff_offset}nm)'
            ))

        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title='Emission Wavelength (nm)',
            yaxis_title='Excitation Wavelength (nm)',
            width=width,
            height=height
        )

        return fig

    def compare_eems(self, eem_data_list, titles=None, layout='grid',
                     view_type='3d', width=1600, height=800, colorscale='Viridis'):
        """
        Compare multiple EEM datasets side by side with consistent scaling

        Args:
            eem_data_list: List of processed EEM data dicts
            titles: List of titles for each EEM plot
            layout: 'grid' or 'tabs'
            view_type: '3d' or '2d'
            width: Width of the plot (default: 1600)
            height: Height of the plot (default: 800)
            colorscale: Colorscale to use for plots

        Returns:
            fig: Plotly figure object with the comparison visualization
        """
        if not PLOTLY_AVAILABLE:
            print("Error: Plotly is required for interactive visualizations.")
            print("Install with: pip install plotly")
            return None

        num_datasets = len(eem_data_list)

        # Set default titles if not provided
        if titles is None:
            titles = [f"Dataset {i + 1}" for i in range(num_datasets)]

        # Calculate global min and max intensity values across all datasets
        global_min = float('inf')
        global_max = float('-inf')

        for eem_data in eem_data_list:
            eem_matrix = eem_data['eem_matrix']

            # Find min/max values that aren't NaN
            valid_data = eem_matrix[~np.isnan(eem_matrix)]
            if len(valid_data) > 0:
                dataset_min = np.min(valid_data)
                dataset_max = np.max(valid_data)

                global_min = min(global_min, dataset_min)
                global_max = max(global_max, dataset_max)

        if global_min == float('inf') or global_max == float('-inf'):
            global_min, global_max = 0, 1  # Fallback if no valid data

        # Add a small buffer to the range for better visualization
        z_range_padding = (global_max - global_min) * 0.05
        z_min = global_min - z_range_padding
        z_max = global_max + z_range_padding

        # Adjust height based on number of datasets for 'grid' layout
        if layout == 'grid':
            if num_datasets > 2:
                rows = (num_datasets + 1) // 2
                height = max(height, rows * 600)  # Ensure adequate height

        if layout == 'grid':
            # Calculate grid dimensions
            if num_datasets <= 2:
                rows, cols = 1, num_datasets
            else:
                cols = 2
                rows = (num_datasets + 1) // 2

            # Create subplot grid with increased spacing
            if view_type == '3d':
                horizontal_spacing = 0.08
                vertical_spacing = 0.15
            else:
                horizontal_spacing = 0.06
                vertical_spacing = 0.10

            fig = make_subplots(
                rows=rows, cols=cols,
                subplot_titles=titles,
                specs=[[{'type': 'scene' if view_type == '3d' else 'xy'} for _ in range(cols)] for _ in range(rows)],
                horizontal_spacing=horizontal_spacing,
                vertical_spacing=vertical_spacing
            )

            # Add each dataset to the grid
            for i, eem_data in enumerate(eem_data_list):
                row = i // cols + 1
                col = i % cols + 1

                if view_type == '3d':
                    # Extract data
                    eem_matrix = eem_data['eem_matrix']
                    cutoff_mask = eem_data['cutoff_mask']
                    excitation_wavelengths = eem_data['excitation_wavelengths']
                    emission_grid = eem_data['emission_grid']

                    # Apply cutoff mask
                    masked_eem = np.copy(eem_matrix)
                    masked_eem[~cutoff_mask] = np.nan

                    # Create meshgrid
                    X, Y = np.meshgrid(emission_grid, excitation_wavelengths)

                    # Add 3D surface with improved colorbar
                    surf = go.Surface(
                        x=X, y=Y, z=masked_eem,
                        colorscale=colorscale,
                        showscale=i == 0,  # Only show colorbar for first plot
                        cmin=global_min,  # Use global min for consistent color scale
                        cmax=global_max,  # Use global max for consistent color scale
                        opacity=0.9,
                        colorbar=dict(
                            title="Intensity",
                            # titleside="right",
                            thickness=20,
                            len=0.6,
                            x=1.02 if cols == 1 else 0.97  # Adjust position based on number of columns
                        ),
                        hovertemplate='Emission: %{x:.1f} nm<br>Excitation: %{y:.1f} nm<br>Intensity: %{z:.4f}<extra></extra>'
                    )

                    fig.add_trace(surf, row=row, col=col)

                    # Add cutoff regions (simplified for grid view)
                    cutoff_eem = np.zeros_like(eem_matrix)
                    cutoff_level = global_min + (global_max - global_min) * 0.01  # Very low level for visualization
                    cutoff_eem[~cutoff_mask] = cutoff_level

                    fig.add_trace(
                        go.Surface(
                            x=X, y=Y, z=cutoff_eem,
                            colorscale=[[0, 'rgba(255,0,0,0.3)'], [1, 'rgba(255,0,0,0.3)']],
                            showscale=False,
                            opacity=0.5
                        ),
                        row=row, col=col
                    )

                    # Configure 3D scene with consistent z-axis range
                    scene_key = f'scene{i + 1}' if i > 0 else 'scene'
                    scene_config = {
                        scene_key: dict(
                            xaxis_title='Emission Wavelength (nm)',
                            yaxis_title='Excitation Wavelength (nm)',
                            zaxis_title='Intensity',
                            zaxis=dict(
                                range=[z_min, z_max],  # Set consistent z range for all plots
                                autorange=False  # Disable autorange to enforce our range
                            ),
                            aspectratio=dict(x=1.5, y=1, z=0.7),
                            camera=dict(
                                eye=dict(x=1.8, y=1.8, z=1.2)
                            ),
                            bgcolor='rgb(245, 245, 245)'  # Set proper background color
                        )
                    }
                    fig.update_layout(**scene_config)

                else:  # 2D view
                    # Extract data
                    eem_matrix = eem_data['eem_matrix']
                    cutoff_mask = eem_data['cutoff_mask']
                    excitation_wavelengths = eem_data['excitation_wavelengths']
                    emission_grid = eem_data['emission_grid']

                    # Apply cutoff mask
                    masked_eem = np.copy(eem_matrix)
                    masked_eem[~cutoff_mask] = np.nan

                    # Add 2D heatmap with improved colorbar
                    hmap = go.Heatmap(
                        z=masked_eem,
                        x=emission_grid,
                        y=excitation_wavelengths,
                        colorscale=colorscale,
                        showscale=i == 0,  # Only show colorbar for first plot
                        zmin=global_min,  # Use global min for consistent color scale
                        zmax=global_max,  # Use global max for consistent color scale
                        colorbar=dict(
                            title="Intensity",
                            # titleside="right",
                            thickness=20,
                            len=0.6,
                            x=1.02 if cols == 1 else 0.97  # Adjust position based on number of columns
                        ),
                        hovertemplate='Excitation: %{y:.1f} nm<br>Emission: %{x:.1f} nm<br>Intensity: %{z:.4f}<extra></extra>'
                    )

                    fig.add_trace(hmap, row=row, col=col)

                    # Mark cutoff regions
                    cutoff_display = np.where(~cutoff_mask, 1, np.nan)

                    fig.add_trace(
                        go.Heatmap(
                            z=cutoff_display,
                            x=emission_grid,
                            y=excitation_wavelengths,
                            colorscale=[[0, 'rgba(255,0,0,0.3)'], [1, 'rgba(255,0,0,0.3)']],
                            showscale=False
                        ),
                        row=row, col=col
                    )

                    # Configure 2D axes
                    fig.update_xaxes(title_text='Emission Wavelength (nm)', row=row, col=col)
                    if col == 1:  # Only add y-axis title for leftmost plots
                        fig.update_yaxes(title_text='Excitation Wavelength (nm)', row=row, col=col)

            # Add annotation about color and z-axis scale
            fig.add_annotation(
                x=0.01,
                y=0.01,
                xref="paper",
                yref="paper",
                text=f"<b>Intensity range:</b> {global_min:.2f} to {global_max:.2f}",
                showarrow=False,
                font=dict(size=12),
                bgcolor="white",
                opacity=0.8,
                bordercolor="black",
                borderwidth=1,
                borderpad=4
            )

            # Update layout with better margins
            fig.update_layout(
                title="EEM Comparison",
                width=width,
                height=height,
                margin=dict(l=65, r=85, b=65, t=90),  # Increased right margin for colorbar
            )

        else:  # tabs layout - not directly supported in Plotly
            # For tabs, we'll return individual figures and provide guidance
            print("Tab layout is not directly supported in Plotly.")
            print("To create a tabbed view in Jupyter, use ipywidgets or separate cells.")

            figures = []
            for i, eem_data in enumerate(eem_data_list):
                if view_type == '3d':
                    fig = self.create_interactive_3d_eem(
                        eem_data,
                        title=titles[i],
                        width=width,
                        height=height,
                        colorscale=colorscale
                    )
                    # Set consistent color scale and z-axis range
                    for trace in fig.data:
                        if isinstance(trace, go.Surface) and not (trace.showscale is False):
                            trace.cmin = global_min
                            trace.cmax = global_max

                    # Update scene with consistent z-axis range
                    fig.update_layout(
                        scene=dict(
                            zaxis=dict(
                                range=[z_min, z_max],
                                autorange=False
                            )
                        )
                    )
                else:
                    fig = self.create_interactive_2d_eem(
                        eem_data,
                        title=titles[i],
                        width=width,
                        height=height,
                        colorscale=colorscale
                    )
                    # Set consistent color scale
                    for trace in fig.data:
                        if isinstance(trace, go.Heatmap) and not (trace.showscale is False):
                            trace.zmin = global_min
                            trace.zmax = global_max
                figures.append(fig)

            # Return list of figures
            return figures

        return fig

    def create_eem_from_arrays(self, emission_wavelengths, excitation_wavelengths, intensity_values,
                               cutoff_offset=30, resolution=2):
        """
        Create an EEM dataset from arrays of wavelengths and intensity values

        Args:
            emission_wavelengths: List of emission wavelengths
            excitation_wavelengths: List of excitation wavelengths
            intensity_values: 2D array of intensity values [excitation, emission]
            cutoff_offset: Offset in nm for cutoff calculations
            resolution: Resolution in nm for the emission grid

        Returns:
            dict: Processed EEM data
        """
        # Create a regular grid for the emission wavelengths
        min_emission = min(emission_wavelengths)
        max_emission = max(emission_wavelengths)
        emission_grid = np.arange(min_emission, max_emission + resolution, resolution)

        # Create the EEM matrix and cutoff mask
        eem_matrix = np.zeros((len(excitation_wavelengths), len(emission_grid)))
        cutoff_mask = np.ones_like(eem_matrix, dtype=bool)

        # Interpolate the input data onto the regular grid
        for i, ex in enumerate(excitation_wavelengths):
            # For each point in the emission grid
            for j, em in enumerate(emission_grid):
                # Find the closest emission wavelength index
                closest_idx = np.argmin(np.abs(np.array(emission_wavelengths) - em))

                # Assign the intensity (this assumes intensity_values is indexed by excitation first)
                eem_matrix[i, j] = intensity_values[i, closest_idx]

                # Calculate cutoff mask
                # 1. Primary cutoff: emission < excitation + cutoff_offset
                if em < (ex + cutoff_offset):
                    cutoff_mask[i, j] = False

                # 2. Second-order cutoff: emission within ±cutoff_offset of 2*excitation
                second_order_min = 2 * ex - cutoff_offset
                second_order_max = 2 * ex + cutoff_offset
                if em >= second_order_min and em <= second_order_max:
                    cutoff_mask[i, j] = False

        # Prepare the result dictionary
        result = {
            'eem_matrix': eem_matrix,
            'cutoff_mask': cutoff_mask,
            'excitation_wavelengths': excitation_wavelengths,
            'emission_grid': emission_grid,
            'cutoff_offset': cutoff_offset,
            'resolution': resolution
        }

        return result

    def load_eem_from_pickle(self, pickle_file):
        """
        Load EEM data from a pickle file

        Args:
            pickle_file: Path to the pickle file

        Returns:
            dict: Processed EEM data
        """
        import pickle

        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)

        return self.process_hyperspectral_data(data)

    def save_processed_eem(self, eem_data, output_file):
        """
        Save processed EEM data to a file

        Args:
            eem_data: Processed EEM data dict
            output_file: Path to the output file (CSV or pickle)
        """
        if output_file.endswith('.csv'):
            # Extract data
            eem_matrix = eem_data['eem_matrix']
            excitation_wavelengths = eem_data['excitation_wavelengths']
            emission_grid = eem_data['emission_grid']

            # Create a DataFrame
            df = pd.DataFrame(eem_matrix, index=excitation_wavelengths, columns=emission_grid)
            df.index.name = 'Excitation (nm)'
            df.columns.name = 'Emission (nm)'

            # Save to CSV
            df.to_csv(output_file)
            print(f"EEM data saved to {output_file}")

        elif output_file.endswith('.pkl') or output_file.endswith('.pickle'):
            # Save the entire dict
            import pickle
            with open(output_file, 'wb') as f:
                pickle.dump(eem_data, f)
            print(f"EEM data saved to {output_file}")

        else:
            print("Unsupported file format. Please use .csv, .pkl, or .pickle")


# Create a widget-based EEM comparison tool (for use in Jupyter)
def create_eem_comparison_widget(eem_data_list, titles=None):
    """
    Create an interactive widget for comparing EEMs in Jupyter

    Args:
        eem_data_list: List of processed EEM data dicts
        titles: List of titles for each EEM

    Returns:
        Widget for EEM comparison
    """
    try:
        import ipywidgets as widgets
        from IPython.display import display, clear_output
    except ImportError:
        print("Error: ipywidgets is required for interactive widgets.")
        print("Install with: pip install ipywidgets")
        return None

    if not PLOTLY_AVAILABLE:
        print("Error: Plotly is required for interactive visualizations.")
        print("Install with: pip install plotly")
        return None

    # Set default titles if not provided
    if titles is None:
        titles = [f"Dataset {i + 1}" for i in range(len(eem_data_list))]

    # Create a visualizer instance
    visualizer = EEMVisualizer()

    # Create widgets
    dataset_dropdown = widgets.Dropdown(
        options=[(title, i) for i, title in enumerate(titles)],
        value=0,
        description='Dataset:',
    )

    view_radio = widgets.RadioButtons(
        options=['3D View', '2D View'],
        value='3D View',
        description='View:',
        layout={'width': 'max-content'}
    )

    colorscale_dropdown = widgets.Dropdown(
        options=['Viridis', 'Plasma', 'Inferno', 'Magma', 'Cividis', 'Turbo', 'Jet'],
        value='Viridis',
        description='Colorscale:',
    )

    cutoff_checkbox = widgets.Checkbox(
        value=True,
        description='Show Cutoff Regions',
    )

    # Create output widget
    output = widgets.Output()

    # Define update function
    def update_plot(change=None):
        with output:
            clear_output(wait=True)

            # Get selected dataset
            dataset_idx = dataset_dropdown.value
            eem_data = eem_data_list[dataset_idx]
            title = titles[dataset_idx]

            # Create the appropriate plot
            if view_radio.value == '3D View':
                fig = visualizer.create_interactive_3d_eem(
                    eem_data,
                    title=title,
                    include_cutoff=cutoff_checkbox.value,
                    colorscale=colorscale_dropdown.value
                )
            else:
                fig = visualizer.create_interactive_2d_eem(
                    eem_data,
                    title=title,
                    include_cutoff=cutoff_checkbox.value,
                    colorscale=colorscale_dropdown.value
                )

            # Display the plot
            fig.show()

    # Connect the widgets to the update function
    dataset_dropdown.observe(update_plot, names='value')
    view_radio.observe(update_plot, names='value')
    colorscale_dropdown.observe(update_plot, names='value')
    cutoff_checkbox.observe(update_plot, names='value')

    # Create the UI layout
    controls = widgets.VBox([dataset_dropdown, view_radio, colorscale_dropdown, cutoff_checkbox])
    ui = widgets.HBox([controls, output])

    # Initialize the plot
    update_plot()

    return ui