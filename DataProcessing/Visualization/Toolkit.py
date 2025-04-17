import base64
import io
import os
import time
import uuid
import numpy as np
import pandas as pd
import h5py
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import ndimage
import dash
from dash import dcc, html, Input, Output, State, callback, ctx, dash_table
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Define color scales
COLORSCALES = ['Viridis', 'Plasma', 'Inferno', 'Magma', 'Cividis', 'Turbo',
               'Greys', 'YlGnBu', 'Greens', 'YlOrRd', 'Bluered', 'RdBu',
               'Reds', 'Blues', 'Jet', 'Rainbow', 'Hot', 'Blackbody']

# Initialize the Dash app with Bootstrap
app = dash.Dash(__name__,
                external_stylesheets=[dbc.themes.BOOTSTRAP],
                suppress_callback_exceptions=True)

# Server definition for deployment
server = app.server

# Define cache directory for uploaded files
CACHE_DIR = "./uploaded_data"
os.makedirs(CACHE_DIR, exist_ok=True)


# ------------------------------- Helper Functions ------------------------------- #

def parse_hyperspectral_data(contents, filename):
    """
    Parse uploaded hyperspectral data file

    Args:
        contents: Contents of the uploaded file
        filename: Name of the uploaded file

    Returns:
        data_dict: Dictionary containing the parsed data
        error: Error message if any
    """
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    error = None
    data_dict = {}

    try:
        if filename.endswith('.h5') or filename.endswith('.hdf5'):
            # Save to temporary file to work with h5py
            temp_file = os.path.join(CACHE_DIR, f"{uuid.uuid4().hex}.h5")
            with open(temp_file, 'wb') as f:
                f.write(decoded)

            # Extract data using h5py
            with h5py.File(temp_file, 'r') as f:
                # Build metadata
                metadata = {}
                if 'metadata' in f:
                    for key in f['metadata'].attrs:
                        metadata[key] = f['metadata'].attrs[key]

                # Get excitation wavelengths
                excitation_wavelengths = []
                for group_name in f.keys():
                    if group_name.startswith('excitation_'):
                        excitation_wavelength = int(group_name.split('_')[1])
                        excitation_wavelengths.append(excitation_wavelength)

                excitation_wavelengths.sort()
                data_dict['excitation_wavelengths'] = excitation_wavelengths
                data_dict['metadata'] = metadata
                data_dict['file_path'] = temp_file
                data_dict['filename'] = filename

        elif filename.endswith('.nc'):
            try:
                import xarray as xr
                # Save to temporary file to work with xarray
                temp_file = os.path.join(CACHE_DIR, f"{uuid.uuid4().hex}.nc")
                with open(temp_file, 'wb') as f:
                    f.write(decoded)

                # Open with xarray
                ds = xr.open_dataset(temp_file)

                # Extract data and metadata
                data_dict['xarray_dataset'] = ds
                data_dict['metadata'] = ds.attrs
                data_dict['file_path'] = temp_file
                data_dict['filename'] = filename

                # Try to find excitation wavelengths
                excitation_wavelengths = []
                for var_name in ds.variables:
                    if var_name.startswith('average_cube_') or var_name.startswith('sum_cube_'):
                        # Extract excitation wavelength from variable name
                        parts = var_name.split('_')
                        if len(parts) > 2 and parts[-1].isdigit():
                            excitation_wavelengths.append(int(parts[-1]))

                if not excitation_wavelengths and 'excitation_wavelengths' in ds.attrs:
                    # Try to parse from metadata
                    try:
                        excitation_str = ds.attrs['excitation_wavelengths']
                        # Handle string formats like "[300, 350, 400]"
                        if isinstance(excitation_str, str):
                            excitation_str = excitation_str.strip('[]')
                            excitation_wavelengths = [int(x.strip()) for x in excitation_str.split(',') if
                                                      x.strip().isdigit()]
                    except:
                        pass

                excitation_wavelengths = sorted(list(set(excitation_wavelengths)))
                data_dict['excitation_wavelengths'] = excitation_wavelengths

            except ImportError:
                error = "xarray is required to load NetCDF files. Please install with 'pip install xarray'."
            except Exception as e:
                error = f"Error reading NetCDF file: {str(e)}"

        elif filename.endswith('.pkl'):
            try:
                # Use pandas to read pickle file
                temp_file = os.path.join(CACHE_DIR, f"{uuid.uuid4().hex}.pkl")
                with open(temp_file, 'wb') as f:
                    f.write(decoded)

                # Load pickle file
                data = pd.read_pickle(temp_file)

                # Check if it's our expected format
                if isinstance(data, dict) and 'average_cubes' in data:
                    data_dict['pandas_data'] = data
                    data_dict['file_path'] = temp_file
                    data_dict['filename'] = filename

                    # Extract excitation wavelengths
                    if 'average_cubes' in data:
                        excitation_wavelengths = sorted(list(data['average_cubes'].keys()))
                        data_dict['excitation_wavelengths'] = excitation_wavelengths

                    # Extract metadata
                    if 'metadata' in data:
                        data_dict['metadata'] = data['metadata']
                else:
                    error = "Unsupported pickle file format. Expected dictionary with 'average_cubes' key."
            except Exception as e:
                error = f"Error reading pickle file: {str(e)}"
        else:
            error = f"Unsupported file format: {filename}. Please upload .h5, .nc, or .pkl files."

    except Exception as e:
        error = f"Error parsing file: {str(e)}"

    return data_dict, error


def get_hypercube_for_excitation(data_dict, excitation, cube_type='average_cube'):
    """
    Extract hypercube data for a specific excitation wavelength

    Args:
        data_dict: Dictionary containing the data
        excitation: Excitation wavelength
        cube_type: Type of cube to extract ('average_cube', 'sum_cube', 'filtered_average_cube', etc.)

    Returns:
        cube_data: Hypercube data as numpy array
        wavelengths: Emission wavelengths
    """
    # Default return values
    cube_data = None
    wavelengths = None

    try:
        # For H5 files
        if 'file_path' in data_dict and data_dict['file_path'].endswith(('.h5', '.hdf5')):
            with h5py.File(data_dict['file_path'], 'r') as f:
                group_name = f'excitation_{excitation}'
                if group_name in f:
                    # Get wavelengths
                    if cube_type.startswith('filtered_'):
                        if 'filtered_wavelengths' in f[group_name]:
                            wavelengths = f[group_name]['filtered_wavelengths'][:]
                        else:
                            print(f"No filtered wavelengths found for excitation {excitation}")
                            return None, None
                    else:
                        wavelengths = f[group_name]['wavelengths'][:]

                    # Get cube data
                    if cube_type in f[group_name]:
                        cube_data = f[group_name][cube_type][:]
                    else:
                        print(f"{cube_type} not found for excitation {excitation}")
                        return None, None
                else:
                    print(f"Excitation {excitation} not found in file")
                    return None, None

        # For NetCDF/xarray
        elif 'xarray_dataset' in data_dict:
            ds = data_dict['xarray_dataset']
            var_name = f"{cube_type}_{excitation}"

            if var_name in ds:
                cube_data = ds[var_name].values
                wavelengths = ds[var_name].emission.values
            else:
                print(f"Variable {var_name} not found in NetCDF file")
                return None, None

        # For pandas pickle
        elif 'pandas_data' in data_dict:
            pd_data = data_dict['pandas_data']

            if cube_type == 'average_cube' and 'average_cubes' in pd_data and excitation in pd_data['average_cubes']:
                df = pd_data['average_cubes'][excitation]
                # Get first level of MultiIndex columns (emission wavelengths)
                wavelengths = np.array([col[0] for col in df.columns if isinstance(col, tuple)])

                # Reshape DataFrame to 3D cube (bands, height, width)
                height = len(df.index.get_level_values('y').unique())
                width = len(df.index.get_level_values('x').unique())

                # Extract intensity values
                intensity_cols = [col for col in df.columns if isinstance(col, tuple) and col[1] == 'intensity']
                cube_data = df[intensity_cols].values
                cube_data = cube_data.reshape((height, width, len(wavelengths)))
                # Transpose to (bands, height, width)
                cube_data = np.transpose(cube_data, (2, 0, 1))

            elif cube_type == 'sum_cube' and 'sum_cubes' in pd_data and excitation in pd_data['sum_cubes']:
                # Similar approach for sum_cube
                df = pd_data['sum_cubes'][excitation]
                wavelengths = np.array([col[0] for col in df.columns if isinstance(col, tuple)])

                height = len(df.index.get_level_values('y').unique())
                width = len(df.index.get_level_values('x').unique())

                intensity_cols = [col for col in df.columns if isinstance(col, tuple) and col[1] == 'intensity']
                cube_data = df[intensity_cols].values
                cube_data = cube_data.reshape((height, width, len(wavelengths)))
                cube_data = np.transpose(cube_data, (2, 0, 1))

            elif cube_type == 'filtered_average_cube' and 'filtered_average_cubes' in pd_data and excitation in pd_data[
                'filtered_average_cubes']:
                # Similar approach for filtered_average_cube
                df = pd_data['filtered_average_cubes'][excitation]
                wavelengths = np.array([col[0] for col in df.columns if isinstance(col, tuple)])

                height = len(df.index.get_level_values('y').unique())
                width = len(df.index.get_level_values('x').unique())

                intensity_cols = [col for col in df.columns if isinstance(col, tuple) and col[1] == 'intensity']
                cube_data = df[intensity_cols].values
                cube_data = cube_data.reshape((height, width, len(wavelengths)))
                cube_data = np.transpose(cube_data, (2, 0, 1))

            elif cube_type == 'filtered_sum_cube' and 'filtered_sum_cubes' in pd_data and excitation in pd_data[
                'filtered_sum_cubes']:
                # Similar approach for filtered_sum_cube
                df = pd_data['filtered_sum_cubes'][excitation]
                wavelengths = np.array([col[0] for col in df.columns if isinstance(col, tuple)])

                height = len(df.index.get_level_values('y').unique())
                width = len(df.index.get_level_values('x').unique())

                intensity_cols = [col for col in df.columns if isinstance(col, tuple) and col[1] == 'intensity']
                cube_data = df[intensity_cols].values
                cube_data = cube_data.reshape((height, width, len(wavelengths)))
                cube_data = np.transpose(cube_data, (2, 0, 1))
            else:
                print(f"Cube type {cube_type} not found for excitation {excitation}")
                return None, None

    except Exception as e:
        print(f"Error extracting hypercube data: {str(e)}")
        return None, None

    return cube_data, wavelengths


def get_nearest_wavelength_index(wavelengths, target_wavelength):
    """
    Find the index of the nearest wavelength in an array

    Args:
        wavelengths: Array of wavelengths
        target_wavelength: Target wavelength to find

    Returns:
        index: Index of the nearest wavelength
    """
    return np.abs(wavelengths - target_wavelength).argmin()


def get_emission_slice(data_dict, excitation, emission, cube_type='average_cube'):
    """
    Extract a 2D slice for specific excitation-emission combination

    Args:
        data_dict: Dictionary containing the data
        excitation: Excitation wavelength
        emission: Emission wavelength (or approximate)
        cube_type: Type of cube to extract from

    Returns:
        slice_data: 2D slice as numpy array
        actual_emission: Actual emission wavelength used
    """
    # Get the hypercube for the specified excitation
    cube_data, wavelengths = get_hypercube_for_excitation(data_dict, excitation, cube_type)

    if cube_data is None or wavelengths is None:
        return None, None

    # Find the closest emission wavelength
    emission_idx = get_nearest_wavelength_index(wavelengths, emission)
    actual_emission = wavelengths[emission_idx]

    # Extract the slice
    slice_data = cube_data[emission_idx, :, :]

    return slice_data, actual_emission


def generate_eem_heatmap(data_dict, cube_type='average_cube'):
    """
    Generate an Excitation-Emission Matrix (EEM) heatmap

    Args:
        data_dict: Dictionary containing the data
        cube_type: Type of cube to use

    Returns:
        fig: Plotly figure object
    """
    # Get excitation wavelengths
    excitation_wavelengths = data_dict.get('excitation_wavelengths', [])

    if not excitation_wavelengths:
        return go.Figure().update_layout(title="No excitation wavelengths found")

    # Initialize data matrices
    emission_sets = []
    intensity_matrices = []

    # For each excitation, get the mean spectrum
    for excitation in excitation_wavelengths:
        cube_data, wavelengths = get_hypercube_for_excitation(data_dict, excitation, cube_type)

        if cube_data is not None and wavelengths is not None:
            # Calculate the mean across spatial dimensions
            mean_spectrum = np.mean(cube_data, axis=(1, 2))
            emission_sets.append(wavelengths)
            intensity_matrices.append(mean_spectrum)

    if not emission_sets:
        return go.Figure().update_layout(title="No data found for EEM heatmap")

    # Find the common set of emission wavelengths (intersection)
    common_emissions = emission_sets[0]
    for emissions in emission_sets[1:]:
        common_emissions = np.intersect1d(common_emissions, emissions)

    if len(common_emissions) == 0:
        # No common wavelengths, use union with interpolation
        all_emissions = np.unique(np.concatenate(emission_sets))

        # Create a uniform grid for the EEM
        excitation_grid, emission_grid = np.meshgrid(excitation_wavelengths, all_emissions)
        intensity_grid = np.zeros(excitation_grid.shape)

        # Fill the grid using interpolation
        for i, excitation in enumerate(excitation_wavelengths):
            # Get original data
            cube_data, wavelengths = get_hypercube_for_excitation(data_dict, excitation, cube_type)
            if cube_data is not None and wavelengths is not None:
                mean_spectrum = np.mean(cube_data, axis=(1, 2))

                # Interpolate to the common grid
                for j, emission in enumerate(all_emissions):
                    # Find the nearest index in the original wavelengths
                    idx = np.abs(wavelengths - emission).argmin()
                    intensity_grid[j, i] = mean_spectrum[idx]

        z_data = intensity_grid
        y_data = all_emissions
    else:
        # Use common wavelengths
        # Create the EEM matrix
        eem_matrix = np.zeros((len(common_emissions), len(excitation_wavelengths)))

        for i, excitation in enumerate(excitation_wavelengths):
            # Get original data
            cube_data, wavelengths = get_hypercube_for_excitation(data_dict, excitation, cube_type)
            if cube_data is not None and wavelengths is not None:
                mean_spectrum = np.mean(cube_data, axis=(1, 2))

                # Map to common emissions
                for j, emission in enumerate(common_emissions):
                    # Find this emission in the original wavelengths
                    idx = np.abs(wavelengths - emission).argmin()
                    eem_matrix[j, i] = mean_spectrum[idx]

        z_data = eem_matrix
        y_data = common_emissions

    # Create the figure with a log color scale for better visualization
    fig = px.imshow(
        z_data,
        x=excitation_wavelengths,
        y=y_data,
        labels=dict(x="Excitation Wavelength (nm)", y="Emission Wavelength (nm)"),
        title="Excitation-Emission Matrix (EEM)",
        color_continuous_scale="Viridis"
    )

    # Add a diagonal line showing the excitation=emission boundary
    max_wavelength = max(np.max(excitation_wavelengths), np.max(y_data))
    min_wavelength = min(np.min(excitation_wavelengths), np.min(y_data))

    fig.add_trace(
        go.Scatter(
            x=[min_wavelength, max_wavelength],
            y=[min_wavelength, max_wavelength],
            mode='lines',
            line=dict(color='red', width=2, dash='dash'),
            name='Ex = Em'
        )
    )

    # Add another line showing a cutoff (e.g., Ex+20 nm)
    cutoff_offset = 20  # default value, could be made adjustable
    fig.add_trace(
        go.Scatter(
            x=[min_wavelength, max_wavelength],
            y=[min_wavelength + cutoff_offset, max_wavelength + cutoff_offset],
            mode='lines',
            line=dict(color='orange', width=2, dash='dot'),
            name=f'Ex + {cutoff_offset}nm'
        )
    )

    # Update layout for better visualization
    fig.update_layout(
        xaxis_title="Excitation Wavelength (nm)",
        yaxis_title="Emission Wavelength (nm)",
        coloraxis_colorbar_title="Intensity",
        height=600
    )

    return fig


def generate_spectral_profile(data_dict, excitation, x_coord, y_coord, cube_type='average_cube'):
    """
    Generate a spectral profile for a specific pixel

    Args:
        data_dict: Dictionary containing the data
        excitation: Excitation wavelength
        x_coord, y_coord: Pixel coordinates
        cube_type: Type of cube to extract from

    Returns:
        fig: Plotly figure object
    """
    # Get the hypercube for the specified excitation
    cube_data, wavelengths = get_hypercube_for_excitation(data_dict, excitation, cube_type)

    if cube_data is None or wavelengths is None:
        return go.Figure().update_layout(title="No data found for spectral profile")

    # Check if coordinates are valid
    if x_coord >= cube_data.shape[2] or y_coord >= cube_data.shape[1] or x_coord < 0 or y_coord < 0:
        return go.Figure().update_layout(title=f"Invalid pixel coordinates: ({x_coord}, {y_coord})")

    # Extract the spectral profile for the pixel
    spectral_profile = cube_data[:, y_coord, x_coord]

    # Create the figure
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=wavelengths,
            y=spectral_profile,
            mode='lines+markers',
            name=f'Pixel ({x_coord}, {y_coord})'
        )
    )

    # Add a vertical line at the excitation wavelength
    fig.add_vline(
        x=excitation,
        line_width=2,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Excitation: {excitation}nm",
        annotation_position="top right"
    )

    # Add a vertical line at the cutoff (e.g., Ex+20 nm)
    cutoff_offset = 20  # default value, could be made adjustable
    cutoff = excitation + cutoff_offset
    fig.add_vline(
        x=cutoff,
        line_width=2,
        line_dash="dot",
        line_color="orange",
        annotation_text=f"Cutoff: {cutoff}nm",
        annotation_position="top right"
    )

    # Update layout
    fig.update_layout(
        title=f"Spectral Profile for Excitation {excitation}nm, Pixel ({x_coord}, {y_coord})",
        xaxis_title="Emission Wavelength (nm)",
        yaxis_title="Intensity",
        template="plotly_white"
    )

    return fig


def generate_multiple_image_comparison(data_dict, excitation_emission_pairs, cube_type='average_cube',
                                       color_scale='Viridis', scale_type='independent', contrast_enhancement=0):
    """
    Generate a comparison of multiple images for different excitation-emission pairs

    Args:
        data_dict: Dictionary containing the data
        excitation_emission_pairs: List of (excitation, emission) tuples
        cube_type: Type of cube to extract from
        color_scale: Color scale to use
        scale_type: 'independent' or 'linked' scaling
        contrast_enhancement: Amount of contrast enhancement (0 = none)

    Returns:
        fig: Plotly figure object
    """
    if not excitation_emission_pairs:
        return go.Figure().update_layout(title="No excitation-emission pairs selected")

    # Determine the number of rows and columns for the subplot grid
    n_images = len(excitation_emission_pairs)
    n_cols = min(3, n_images)  # Max 3 columns
    n_rows = (n_images + n_cols - 1) // n_cols  # Ceiling division

    # Create subplot figure
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[f"Ex:{ex}nm, Em:{em}nm" for ex, em in excitation_emission_pairs]
    )

    # Collect all slices to determine global min/max if using linked scaling
    all_slices = []
    for ex, em in excitation_emission_pairs:
        slice_data, actual_em = get_emission_slice(data_dict, ex, em, cube_type)
        if slice_data is not None:
            # Apply contrast enhancement if requested
            if contrast_enhancement > 0:
                p_low = contrast_enhancement
                p_high = 100 - contrast_enhancement
                vmin, vmax = np.percentile(slice_data, [p_low, p_high])
                # Clip the data
                slice_data_enhanced = np.clip(slice_data, vmin, vmax)
                all_slices.append(slice_data_enhanced)
            else:
                all_slices.append(slice_data)

    # Determine global scale if linked
    if scale_type == 'linked' and all_slices:
        global_min = min(np.min(s) for s in all_slices)
        global_max = max(np.max(s) for s in all_slices)

    # Add each image to the subplot
    for i, (ex, em) in enumerate(excitation_emission_pairs):
        row = i // n_cols + 1
        col = i % n_cols + 1

        slice_data, actual_em = get_emission_slice(data_dict, ex, em, cube_type)

        if slice_data is not None:
            # Apply contrast enhancement if requested
            if contrast_enhancement > 0:
                p_low = contrast_enhancement
                p_high = 100 - contrast_enhancement
                vmin, vmax = np.percentile(slice_data, [p_low, p_high])
                # Clip the data
                slice_data = np.clip(slice_data, vmin, vmax)

            # Set z-scale range
            if scale_type == 'linked' and all_slices:
                zmin, zmax = global_min, global_max
            else:
                zmin, zmax = np.min(slice_data), np.max(slice_data)

            fig.add_trace(
                go.Heatmap(
                    z=slice_data,
                    colorscale=color_scale,
                    zmin=zmin,
                    zmax=zmax,
                    showscale=(col == n_cols),  # Only show colorbar for rightmost column
                    colorbar=dict(
                        len=1 / n_rows,
                        yanchor="top",
                        y=1 - (row - 1) / n_rows - 0.5 / n_rows,
                        title="Intensity"
                    ) if col == n_cols else None
                ),
                row=row, col=col
            )

            # Update subplot title with actual emission wavelength
            fig.layout.annotations[i].text = f"Ex:{ex}nm, Em:{actual_em:.1f}nm"
        else:
            # Add empty subplot with error message
            fig.add_trace(
                go.Scatter(
                    x=[],
                    y=[],
                    text=["No data available"],
                    mode="text"
                ),
                row=row, col=col
            )

    # Update layout
    fig.update_layout(
        title=f"Multi-Slice Comparison ({cube_type}, {scale_type} scaling)",
        height=300 * n_rows,
        width=300 * n_cols + 100,  # Add space for colorbar
        margin=dict(l=20, r=20, t=40, b=20)
    )

    # Update axes to have the same range and disable ticks
    for i in range(1, n_rows * n_cols + 1):
        fig.update_xaxes(showticklabels=False, row=(i - 1) // n_cols + 1, col=(i - 1) % n_cols + 1)
        fig.update_yaxes(showticklabels=False, row=(i - 1) // n_cols + 1, col=(i - 1) % n_cols + 1)

    return fig


def generate_pca_analysis(data_dict, excitation, cube_type='average_cube', n_components=3):
    """
    Perform PCA analysis on hyperspectral data for a specific excitation

    Args:
        data_dict: Dictionary containing the data
        excitation: Excitation wavelength
        cube_type: Type of cube to analyze
        n_components: Number of PCA components to extract

    Returns:
        fig_components: Figure showing PCA component images
        fig_loadings: Figure showing PCA component loadings
        fig_variance: Figure showing explained variance
    """
    # Get the hypercube for the specified excitation
    cube_data, wavelengths = get_hypercube_for_excitation(data_dict, excitation, cube_type)

    if cube_data is None or wavelengths is None:
        fig = go.Figure().update_layout(title="No data found for PCA analysis")
        return fig, fig, fig

    # Reshape the cube to a 2D matrix: (bands, pixels)
    n_bands, height, width = cube_data.shape
    X = cube_data.reshape(n_bands, -1).T  # Each row is a pixel, each column is a band

    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply PCA
    n_components = min(n_components, min(X_scaled.shape))
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(X_scaled)

    # Create figure for PCA component images
    n_cols = min(3, n_components)
    n_rows = (n_components + n_cols - 1) // n_cols

    fig_components = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[f"PC{i + 1} ({pca.explained_variance_ratio_[i]:.1%})" for i in range(n_components)]
    )

    for i in range(n_components):
        row = i // n_cols + 1
        col = i % n_cols + 1

        # Reshape the component back to image shape
        component_img = transformed[:, i].reshape(height, width)

        fig_components.add_trace(
            go.Heatmap(
                z=component_img,
                colorscale='RdBu_r',
                zmid=0,  # Center the colorscale at zero
                showscale=(col == n_cols),
                colorbar=dict(
                    len=1 / n_rows,
                    yanchor="top",
                    y=1 - (row - 1) / n_rows - 0.5 / n_rows,
                    title="Score"
                ) if col == n_cols else None
            ),
            row=row, col=col
        )

    # Update axes to have the same range and disable ticks
    for i in range(1, n_rows * n_cols + 1):
        fig_components.update_xaxes(showticklabels=False, row=(i - 1) // n_cols + 1, col=(i - 1) % n_cols + 1)
        fig_components.update_yaxes(showticklabels=False, row=(i - 1) // n_cols + 1, col=(i - 1) % n_cols + 1)

    # Update layout
    fig_components.update_layout(
        title=f"PCA Components for Excitation {excitation}nm ({cube_type})",
        height=300 * n_rows,
        width=300 * n_cols + 100
    )

    # Create figure for PCA loadings
    fig_loadings = go.Figure()

    for i in range(n_components):
        fig_loadings.add_trace(
            go.Scatter(
                x=wavelengths,
                y=pca.components_[i],
                mode='lines',
                name=f'PC{i + 1} ({pca.explained_variance_ratio_[i]:.1%})'
            )
        )

    # Add a vertical line at the excitation wavelength
    fig_loadings.add_vline(
        x=excitation,
        line_width=2,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Excitation: {excitation}nm",
        annotation_position="top right"
    )

    # Add a vertical line at the cutoff (e.g., Ex+20 nm)
    cutoff_offset = 20  # default value, could be made adjustable
    cutoff = excitation + cutoff_offset
    fig_loadings.add_vline(
        x=cutoff,
        line_width=2,
        line_dash="dot",
        line_color="orange",
        annotation_text=f"Cutoff: {cutoff}nm",
        annotation_position="top right"
    )

    # Update layout
    fig_loadings.update_layout(
        title=f"PCA Loadings by Emission Wavelength (Excitation {excitation}nm)",
        xaxis_title="Emission Wavelength (nm)",
        yaxis_title="Loading",
        template="plotly_white"
    )

    # Create figure for explained variance
    fig_variance = go.Figure()

    # Explained variance ratio
    fig_variance.add_trace(
        go.Bar(
            x=[f'PC{i + 1}' for i in range(n_components)],
            y=pca.explained_variance_ratio_,
            name='Explained Variance'
        )
    )

    # Cumulative explained variance
    fig_variance.add_trace(
        go.Scatter(
            x=[f'PC{i + 1}' for i in range(n_components)],
            y=np.cumsum(pca.explained_variance_ratio_),
            mode='lines+markers',
            name='Cumulative',
            yaxis='y2'
        )
    )

    # Add a horizontal line at 95% for the cumulative plot
    fig_variance.add_hline(
        y=0.95,
        line_width=2,
        line_dash="dot",
        line_color="red",
        annotation_text="95%",
        annotation_position="right",
        yref='y2'
    )

    # Update layout
    fig_variance.update_layout(
        title=f"PCA Explained Variance (Excitation {excitation}nm)",
        xaxis_title="Principal Component",
        yaxis_title="Explained Variance Ratio",
        yaxis=dict(range=[0, 1]),
        yaxis2=dict(
            title="Cumulative Explained Variance",
            overlaying='y',
            side='right',
            range=[0, 1],
            showgrid=False
        ),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        template="plotly_white"
    )

    return fig_components, fig_loadings, fig_variance


def generate_rgb_composite(data_dict, excitation, emission_red, emission_green, emission_blue,
                           cube_type='average_cube', contrast_enhancement=0):
    """
    Generate an RGB composite image from three emission wavelengths

    Args:
        data_dict: Dictionary containing the data
        excitation: Excitation wavelength
        emission_red, emission_green, emission_blue: Emission wavelengths for RGB channels
        cube_type: Type of cube to extract from
        contrast_enhancement: Amount of contrast enhancement (0 = none)

    Returns:
        fig: Plotly figure object
    """
    # Get the three slices
    red_slice, actual_red = get_emission_slice(data_dict, excitation, emission_red, cube_type)
    green_slice, actual_green = get_emission_slice(data_dict, excitation, emission_green, cube_type)
    blue_slice, actual_blue = get_emission_slice(data_dict, excitation, emission_blue, cube_type)

    if red_slice is None or green_slice is None or blue_slice is None:
        return go.Figure().update_layout(title="Could not extract one or more slices for RGB composite")

    # Check if dimensions match
    if red_slice.shape != green_slice.shape or red_slice.shape != blue_slice.shape:
        return go.Figure().update_layout(title="Slice dimensions do not match")

    # Apply contrast enhancement if requested
    if contrast_enhancement > 0:
        p_low = contrast_enhancement
        p_high = 100 - contrast_enhancement

        # Enhance each channel independently
        for slice_data in [red_slice, green_slice, blue_slice]:
            vmin, vmax = np.percentile(slice_data, [p_low, p_high])
            # Scale to 0-1 range
            slice_data = np.clip((slice_data - vmin) / (vmax - vmin) if vmax > vmin else 0, 0, 1)
    else:
        # Normalize each channel to 0-1 range
        red_slice = (red_slice - np.min(red_slice)) / (np.max(red_slice) - np.min(red_slice)) if np.max(
            red_slice) > np.min(red_slice) else np.zeros_like(red_slice)
        green_slice = (green_slice - np.min(green_slice)) / (np.max(green_slice) - np.min(green_slice)) if np.max(
            green_slice) > np.min(green_slice) else np.zeros_like(green_slice)
        blue_slice = (blue_slice - np.min(blue_slice)) / (np.max(blue_slice) - np.min(blue_slice)) if np.max(
            blue_slice) > np.min(blue_slice) else np.zeros_like(blue_slice)

    # Create RGB array
    rgb = np.stack([red_slice, green_slice, blue_slice], axis=2)

    # Create the figure
    fig = px.imshow(
        rgb,
        title=f"RGB Composite (Ex: {excitation}nm, R: {actual_red:.1f}nm, G: {actual_green:.1f}nm, B: {actual_blue:.1f}nm)"
    )

    # Update layout
    fig.update_layout(
        coloraxis_showscale=False,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    # Disable axis ticks
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    return fig


def generate_spatial_profile(data_dict, excitation, emission, axis='horizontal', position=None,
                             cube_type='average_cube', smoothing=0):
    """
    Generate a spatial profile (line plot) through a slice

    Args:
        data_dict: Dictionary containing the data
        excitation: Excitation wavelength
        emission: Emission wavelength
        axis: 'horizontal' or 'vertical'
        position: Position for the line (row or column index)
        cube_type: Type of cube to extract from
        smoothing: Gaussian smoothing sigma (0 = none)

    Returns:
        fig: Plotly figure object
    """
    # Get the slice
    slice_data, actual_emission = get_emission_slice(data_dict, excitation, emission, cube_type)

    if slice_data is None:
        return go.Figure().update_layout(title="Could not extract slice for spatial profile")

    # If position is not specified, use the middle of the image
    if position is None:
        if axis == 'horizontal':
            position = slice_data.shape[0] // 2
        else:  # vertical
            position = slice_data.shape[1] // 2

    # Check if position is valid
    if axis == 'horizontal' and (position < 0 or position >= slice_data.shape[0]):
        return go.Figure().update_layout(title=f"Invalid position {position} for horizontal profile")
    elif axis == 'vertical' and (position < 0 or position >= slice_data.shape[1]):
        return go.Figure().update_layout(title=f"Invalid position {position} for vertical profile")

    # Extract the profile
    if axis == 'horizontal':
        profile = slice_data[position, :]
        x_values = np.arange(len(profile))
        profile_label = f"Row {position}"
    else:  # vertical
        profile = slice_data[:, position]
        x_values = np.arange(len(profile))
        profile_label = f"Column {position}"

    # Apply Gaussian smoothing if requested
    if smoothing > 0:
        profile = ndimage.gaussian_filter1d(profile, sigma=smoothing)

    # Create the figure
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=profile,
            mode='lines',
            name=profile_label
        )
    )

    # Update layout
    fig.update_layout(
        title=f"{axis.capitalize()} Profile (Ex: {excitation}nm, Em: {actual_emission:.1f}nm, {profile_label})",
        xaxis_title="Pixel Position",
        yaxis_title="Intensity",
        template="plotly_white"
    )

    return fig


# ------------------------------- App Layout ------------------------------- #

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Hyperspectral Data Visualization Toolkit", className="text-center mb-4"),
            html.P("Upload and explore 4D hyperspectral data (spatial x, y + excitation + emission wavelengths)",
                   className="text-center mb-4")
        ])
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Data Loading"),
                dbc.CardBody([
                    dcc.Upload(
                        id='upload-data',
                        children=html.Div([
                            'Drag and Drop or ',
                            html.A('Select a File', className="fw-bold")
                        ]),
                        style={
                            'width': '100%',
                            'height': '60px',
                            'lineHeight': '60px',
                            'borderWidth': '1px',
                            'borderStyle': 'dashed',
                            'borderRadius': '5px',
                            'textAlign': 'center',
                            'margin': '10px'
                        },
                        multiple=False
                    ),
                    html.Div(id='upload-status', className="mt-2"),

                    html.Hr(),

                    html.H5("Data Summary"),
                    html.Div(id='data-summary')
                ])
            ], className="mb-4")
        ])
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("EEM Heatmap & Slice Selection"),
                dbc.CardBody([
                    dcc.Loading(
                        id="loading-eem",
                        type="circle",
                        children=[
                            dcc.Graph(id='eem-heatmap', className="mb-3"),
                        ]
                    ),

                    html.Hr(),

                    html.H5("Slice Selection"),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Cube Type"),
                            dcc.Dropdown(
                                id='cube-type-selector',
                                options=[
                                    {'label': 'Average Cube', 'value': 'average_cube'},
                                    {'label': 'Sum Cube', 'value': 'sum_cube'},
                                    {'label': 'Filtered Average Cube', 'value': 'filtered_average_cube'},
                                    {'label': 'Filtered Sum Cube', 'value': 'filtered_sum_cube'}
                                ],
                                value='average_cube'
                            )
                        ], width=6),
                        dbc.Col([
                            html.Label("Color Scale"),
                            dcc.Dropdown(
                                id='colorscale-selector',
                                options=[{'label': cs, 'value': cs} for cs in COLORSCALES],
                                value='Viridis'
                            )
                        ], width=6)
                    ], className="mb-3"),

                    dbc.Row([
                        dbc.Col([
                            html.Label("Excitation Wavelength"),
                            dcc.Dropdown(id='excitation-selector')
                        ], width=6),
                        dbc.Col([
                            html.Label("Emission Wavelength"),
                            dcc.Dropdown(id='emission-selector')
                        ], width=6)
                    ], className="mb-3"),

                    html.Label("Contrast Enhancement"),
                    dcc.Slider(
                        id='contrast-slider',
                        min=0,
                        max=5,
                        step=0.5,
                        value=0,
                        marks={i: f"{i}%" for i in range(0, 6)}
                    ),

                    html.Hr(),

                    html.H5("Multi-Slice Selection"),
                    html.Div([
                        html.Label("Select multiple excitation-emission pairs:"),
                        dcc.Dropdown(
                            id='multi-slice-selector',
                            options=[],
                            value=[],
                            multi=True
                        )
                    ]),
                    dbc.RadioItems(
                        id='scale-type-selector',
                        options=[
                            {'label': 'Independent scaling', 'value': 'independent'},
                            {'label': 'Linked scaling', 'value': 'linked'}
                        ],
                        value='independent',
                        inline=True,
                        className="mt-2"
                    )
                ])
            ], className="mb-4")
        ], width=12)
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Tabs([
                dbc.Tab(label="Single Slice View", children=[
                    dbc.Row([
                        dbc.Col([
                            dcc.Loading(
                                id="loading-single-slice",
                                type="circle",
                                children=[
                                    dcc.Graph(id='single-slice-image')
                                ]
                            )
                        ], width=6),
                        dbc.Col([
                            dcc.Loading(
                                id="loading-histogram",
                                type="circle",
                                children=[
                                    dcc.Graph(id='histogram-plot')
                                ]
                            )
                        ], width=6)
                    ], className="mb-3"),

                    dbc.Row([
                        dbc.Col([
                            html.H5("Spatial Profile"),
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Profile Type"),
                                    dcc.Dropdown(
                                        id='profile-type-selector',
                                        options=[
                                            {'label': 'Horizontal', 'value': 'horizontal'},
                                            {'label': 'Vertical', 'value': 'vertical'}
                                        ],
                                        value='horizontal'
                                    )
                                ], width=6),
                                dbc.Col([
                                    html.Label("Position"),
                                    dcc.Input(
                                        id='profile-position',
                                        type='number',
                                        min=0,
                                        step=1,
                                        value=None,
                                        placeholder='Auto (center)'
                                    )
                                ], width=6)
                            ], className="mb-2"),

                            html.Label("Smoothing"),
                            dcc.Slider(
                                id='profile-smoothing',
                                min=0,
                                max=5,
                                step=0.5,
                                value=0,
                                marks={i: str(i) for i in range(0, 6)}
                            ),

                            dcc.Loading(
                                id="loading-profile",
                                type="circle",
                                children=[
                                    dcc.Graph(id='spatial-profile-plot')
                                ]
                            )
                        ], width=12)
                    ])
                ]),

                dbc.Tab(label="Multi-Slice Comparison", children=[
                    dcc.Loading(
                        id="loading-multi-slice",
                        type="circle",
                        children=[
                            dcc.Graph(id='multi-slice-comparison')
                        ]
                    )
                ]),

                dbc.Tab(label="Spectral Profiles", children=[
                    dbc.Row([
                        dbc.Col([
                            html.H5("Pixel Selection"),
                            dbc.Row([
                                dbc.Col([
                                    html.Label("X Coordinate"),
                                    dcc.Input(
                                        id='x-coordinate',
                                        type='number',
                                        min=0,
                                        step=1,
                                        value=0
                                    )
                                ], width=6),
                                dbc.Col([
                                    html.Label("Y Coordinate"),
                                    dcc.Input(
                                        id='y-coordinate',
                                        type='number',
                                        min=0,
                                        step=1,
                                        value=0
                                    )
                                ], width=6)
                            ], className="mb-3"),

                            dcc.Loading(
                                id="loading-spectral-profile",
                                type="circle",
                                children=[
                                    dcc.Graph(id='spectral-profile-plot')
                                ]
                            )
                        ], width=12)
                    ])
                ]),

                dbc.Tab(label="RGB Composite", children=[
                    dbc.Row([
                        dbc.Col([
                            html.H5("RGB Channel Selection"),
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Red Channel (nm)"),
                                    dcc.Dropdown(id='emission-red-selector')
                                ], width=4),
                                dbc.Col([
                                    html.Label("Green Channel (nm)"),
                                    dcc.Dropdown(id='emission-green-selector')
                                ], width=4),
                                dbc.Col([
                                    html.Label("Blue Channel (nm)"),
                                    dcc.Dropdown(id='emission-blue-selector')
                                ], width=4)
                            ], className="mb-3"),

                            html.Label("Contrast Enhancement"),
                            dcc.Slider(
                                id='rgb-contrast-slider',
                                min=0,
                                max=10,
                                step=1,
                                value=0,
                                marks={i: f"{i}%" for i in range(0, 11, 2)}
                            ),

                            dcc.Loading(
                                id="loading-rgb",
                                type="circle",
                                children=[
                                    dcc.Graph(id='rgb-composite')
                                ]
                            )
                        ], width=12)
                    ])
                ]),

                dbc.Tab(label="PCA Analysis", children=[
                    dbc.Row([
                        dbc.Col([
                            html.H5("PCA Settings"),
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Number of Components"),
                                    dcc.Input(
                                        id='n-components',
                                        type='number',
                                        min=1,
                                        max=10,
                                        step=1,
                                        value=3
                                    )
                                ], width=6)
                            ], className="mb-3"),

                            dcc.Loading(
                                id="loading-pca",
                                type="circle",
                                children=[
                                    html.Div([
                                        html.H6("PCA Components", className="mt-3"),
                                        dcc.Graph(id='pca-components'),

                                        html.H6("PCA Loadings", className="mt-3"),
                                        dcc.Graph(id='pca-loadings'),

                                        html.H6("Explained Variance", className="mt-3"),
                                        dcc.Graph(id='pca-variance')
                                    ])
                                ]
                            )
                        ], width=12)
                    ])
                ])
            ])
        ], width=12)
    ]),

    # Hidden div to store the uploaded data
    dcc.Store(id='uploaded-data'),

    dbc.Row([
        dbc.Col([
            html.Hr(),
            html.P("Created by Claude 3.5 Sonnet - Anthropic", className="text-center text-muted")
        ])
    ])
], fluid=True)


# ------------------------------- Callbacks ------------------------------- #

@app.callback(
    [Output('uploaded-data', 'data'),
     Output('upload-status', 'children'),
     Output('data-summary', 'children')],
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename')]
)
def update_data(contents, filename):
    if contents is None:
        return None, None, "No data uploaded yet."

    # Parse the uploaded file
    data_dict, error = parse_hyperspectral_data(contents, filename)

    if error:
        return None, html.Div([
            html.I(className="bi bi-exclamation-triangle-fill me-2", style={"color": "red"}),
            f"Error: {error}"
        ], style={"color": "red"}), "Upload failed."

    # Create a summary of the data
    summary = []

    # File info
    summary.append(html.P([
        html.Strong("File: "), filename
    ]))

    # Excitation wavelengths
    if 'excitation_wavelengths' in data_dict:
        excitation_wavelengths = data_dict['excitation_wavelengths']
        summary.append(html.P([
            html.Strong("Excitation Wavelengths: "),
            f"{len(excitation_wavelengths)} values from {min(excitation_wavelengths)} to {max(excitation_wavelengths)} nm"
        ]))

    # Metadata
    if 'metadata' in data_dict and data_dict['metadata']:
        metadata_items = []
        for key, value in data_dict['metadata'].items():
            metadata_items.append(html.Li(f"{key}: {value}"))

        summary.append(html.Div([
            html.Strong("Metadata:"),
            html.Ul(metadata_items, className="list-unstyled ms-3")
        ]))

    return data_dict, html.Div([
        html.I(className="bi bi-check-circle-fill me-2", style={"color": "green"}),
        f"Successfully loaded {filename}"
    ], style={"color": "green"}), summary


@app.callback(
    [Output('excitation-selector', 'options'),
     Output('excitation-selector', 'value')],
    [Input('uploaded-data', 'data')]
)
def update_excitation_options(data):
    if data is None:
        return [], None

    excitation_wavelengths = data.get('excitation_wavelengths', [])
    options = [{'label': f"{ex} nm", 'value': ex} for ex in excitation_wavelengths]

    # Set default value to the first excitation
    default_value = excitation_wavelengths[0] if excitation_wavelengths else None

    return options, default_value


@app.callback(
    [Output('emission-selector', 'options'),
     Output('emission-selector', 'value')],
    [Input('uploaded-data', 'data'),
     Input('excitation-selector', 'value'),
     Input('cube-type-selector', 'value')]
)
def update_emission_options(data, excitation, cube_type):
    if data is None or excitation is None:
        return [], None

    # Get emission wavelengths for the selected excitation
    cube_data, wavelengths = get_hypercube_for_excitation(data, excitation, cube_type)

    if wavelengths is None:
        return [], None

    options = [{'label': f"{em:.1f} nm", 'value': em} for em in wavelengths]

    # Set default value to the middle emission wavelength
    default_value = wavelengths[len(wavelengths) // 2] if len(wavelengths) > 0 else None

    return options, default_value


@app.callback(
    Output('multi-slice-selector', 'options'),
    [Input('uploaded-data', 'data'),
     Input('cube-type-selector', 'value')]
)
def update_multi_slice_options(data, cube_type):
    if data is None:
        return []

    options = []
    excitation_wavelengths = data.get('excitation_wavelengths', [])

    for ex in excitation_wavelengths:
        # Get emission wavelengths for this excitation
        cube_data, wavelengths = get_hypercube_for_excitation(data, ex, cube_type)

        if wavelengths is not None:
            # Create options for each excitation-emission pair
            for em in wavelengths:
                label = f"Ex:{ex}nm, Em:{em:.1f}nm"
                value = f"{ex},{em}"
                options.append({'label': label, 'value': value})

    return options


@app.callback(
    [Output('emission-red-selector', 'options'),
     Output('emission-red-selector', 'value'),
     Output('emission-green-selector', 'options'),
     Output('emission-green-selector', 'value'),
     Output('emission-blue-selector', 'options'),
     Output('emission-blue-selector', 'value')],
    [Input('uploaded-data', 'data'),
     Input('excitation-selector', 'value'),
     Input('cube-type-selector', 'value')]
)
def update_rgb_options(data, excitation, cube_type):
    if data is None or excitation is None:
        return [], None, [], None, [], None

    # Get emission wavelengths for the selected excitation
    cube_data, wavelengths = get_hypercube_for_excitation(data, excitation, cube_type)

    if wavelengths is None:
        return [], None, [], None, [], None

    options = [{'label': f"{em:.1f} nm", 'value': em} for em in wavelengths]

    # Set default values to span the emission range
    if len(wavelengths) >= 3:
        idx_red = len(wavelengths) - 1  # Longest wavelength
        idx_blue = 0  # Shortest wavelength
        idx_green = len(wavelengths) // 2  # Middle wavelength

        red_value = wavelengths[idx_red]
        green_value = wavelengths[idx_green]
        blue_value = wavelengths[idx_blue]
    else:
        red_value = wavelengths[0] if len(wavelengths) > 0 else None
        green_value = wavelengths[0] if len(wavelengths) > 0 else None
        blue_value = wavelengths[0] if len(wavelengths) > 0 else None

    return options, red_value, options, green_value, options, blue_value


@app.callback(
    Output('eem-heatmap', 'figure'),
    [Input('uploaded-data', 'data'),
     Input('cube-type-selector', 'value')]
)
def update_eem_heatmap(data, cube_type):
    if data is None:
        return go.Figure().update_layout(title="Upload data to visualize")

    # Generate EEM heatmap
    fig = generate_eem_heatmap(data, cube_type)
    return fig


@app.callback(
    Output('single-slice-image', 'figure'),
    [Input('uploaded-data', 'data'),
     Input('excitation-selector', 'value'),
     Input('emission-selector', 'value'),
     Input('cube-type-selector', 'value'),
     Input('colorscale-selector', 'value'),
     Input('contrast-slider', 'value')]
)
def update_slice_image(data, excitation, emission, cube_type, colorscale, contrast):
    if data is None or excitation is None or emission is None:
        return go.Figure().update_layout(title="Select excitation and emission wavelengths")

    # Get the slice
    slice_data, actual_emission = get_emission_slice(data, excitation, emission, cube_type)

    if slice_data is None:
        return go.Figure().update_layout(title="Could not extract slice")

    # Apply contrast enhancement if requested
    if contrast > 0:
        p_low = contrast
        p_high = 100 - contrast
        vmin, vmax = np.percentile(slice_data, [p_low, p_high])
        # Clip the data for visualization (don't modify the original)
        slice_data_viz = np.clip(slice_data, vmin, vmax)
    else:
        slice_data_viz = slice_data

    # Create the figure
    fig = px.imshow(
        slice_data_viz,
        color_continuous_scale=colorscale,
        title=f"Ex: {excitation}nm, Em: {actual_emission:.1f}nm ({cube_type})"
    )

    # Add hover information
    fig.update_traces(
        hovertemplate="X: %{x}<br>Y: %{y}<br>Intensity: %{z:.4f}"
    )

    # Update layout
    fig.update_layout(
        coloraxis_colorbar_title="Intensity",
        margin=dict(l=20, r=20, t=40, b=20)
    )

    # Add statistics annotation
    stats_text = (
        f"Min: {np.min(slice_data):.4f}<br>"
        f"Max: {np.max(slice_data):.4f}<br>"
        f"Mean: {np.mean(slice_data):.4f}<br>"
        f"Std: {np.std(slice_data):.4f}"
    )

    fig.add_annotation(
        x=1,
        y=0,
        xref="paper",
        yref="paper",
        text=stats_text,
        showarrow=False,
        font=dict(size=10),
        bgcolor="rgba(255, 255, 255, 0.7)",
        bordercolor="black",
        borderwidth=1,
        borderpad=4,
        align="left"
    )

    return fig


@app.callback(
    Output('histogram-plot', 'figure'),
    [Input('uploaded-data', 'data'),
     Input('excitation-selector', 'value'),
     Input('emission-selector', 'value'),
     Input('cube-type-selector', 'value'),
     Input('contrast-slider', 'value')]
)
def update_histogram(data, excitation, emission, cube_type, contrast):
    if data is None or excitation is None or emission is None:
        return go.Figure().update_layout(title="Select excitation and emission wavelengths")

    # Get the slice
    slice_data, actual_emission = get_emission_slice(data, excitation, emission, cube_type)

    if slice_data is None:
        return go.Figure().update_layout(title="Could not extract slice")

    # Create the histogram
    fig = px.histogram(
        slice_data.flatten(),
        nbins=50,
        title=f"Histogram (Ex: {excitation}nm, Em: {actual_emission:.1f}nm)"
    )

    # Add vertical lines for percentiles if contrast enhancement is active
    if contrast > 0:
        p_low = contrast
        p_high = 100 - contrast
        vmin, vmax = np.percentile(slice_data, [p_low, p_high])

        fig.add_vline(
            x=vmin,
            line_width=2,
            line_dash="dash",
            line_color="red",
            annotation_text=f"{p_low}%",
            annotation_position="top right"
        )

        fig.add_vline(
            x=vmax,
            line_width=2,
            line_dash="dash",
            line_color="red",
            annotation_text=f"{p_high}%",
            annotation_position="top right"
        )

    # Update layout
    fig.update_layout(
        xaxis_title="Intensity",
        yaxis_title="Count",
        template="plotly_white"
    )

    return fig


@app.callback(
    Output('multi-slice-comparison', 'figure'),
    [Input('uploaded-data', 'data'),
     Input('multi-slice-selector', 'value'),
     Input('cube-type-selector', 'value'),
     Input('colorscale-selector', 'value'),
     Input('scale-type-selector', 'value'),
     Input('contrast-slider', 'value')]
)
def update_multi_slice_comparison(data, selected_pairs, cube_type, colorscale, scale_type, contrast):
    if data is None or not selected_pairs:
        return go.Figure().update_layout(title="Select excitation-emission pairs to compare")

    # Parse the selected pairs
    excitation_emission_pairs = []
    for pair_str in selected_pairs:
        parts = pair_str.split(',')
        if len(parts) == 2:
            try:
                ex = int(parts[0])
                em = float(parts[1])
                excitation_emission_pairs.append((ex, em))
            except ValueError:
                pass

    # Generate the multi-slice comparison
    fig = generate_multiple_image_comparison(
        data, excitation_emission_pairs, cube_type, colorscale, scale_type, contrast
    )

    return fig


@app.callback(
    Output('spectral-profile-plot', 'figure'),
    [Input('uploaded-data', 'data'),
     Input('excitation-selector', 'value'),
     Input('x-coordinate', 'value'),
     Input('y-coordinate', 'value'),
     Input('cube-type-selector', 'value')]
)
def update_spectral_profile(data, excitation, x_coord, y_coord, cube_type):
    if data is None or excitation is None or x_coord is None or y_coord is None:
        return go.Figure().update_layout(title="Select excitation and pixel coordinates")

    # Generate the spectral profile
    fig = generate_spectral_profile(data, excitation, x_coord, y_coord, cube_type)
    return fig


@app.callback(
    Output('spatial-profile-plot', 'figure'),
    [Input('uploaded-data', 'data'),
     Input('excitation-selector', 'value'),
     Input('emission-selector', 'value'),
     Input('profile-type-selector', 'value'),
     Input('profile-position', 'value'),
     Input('cube-type-selector', 'value'),
     Input('profile-smoothing', 'value')]
)
def update_spatial_profile(data, excitation, emission, profile_type, position, cube_type, smoothing):
    if data is None or excitation is None or emission is None:
        return go.Figure().update_layout(title="Select excitation and emission wavelengths")

    # Generate the spatial profile
    fig = generate_spatial_profile(
        data, excitation, emission, profile_type, position, cube_type, smoothing
    )

    return fig


@app.callback(
    [Output('pca-components', 'figure'),
     Output('pca-loadings', 'figure'),
     Output('pca-variance', 'figure')],
    [Input('uploaded-data', 'data'),
     Input('excitation-selector', 'value'),
     Input('cube-type-selector', 'value'),
     Input('n-components', 'value')]
)
def update_pca_analysis(data, excitation, cube_type, n_components):
    if data is None or excitation is None or n_components is None:
        empty_fig = go.Figure().update_layout(title="Select excitation and PCA parameters")
        return empty_fig, empty_fig, empty_fig

    # Generate the PCA analysis
    fig_components, fig_loadings, fig_variance = generate_pca_analysis(
        data, excitation, cube_type, n_components
    )

    return fig_components, fig_loadings, fig_variance


@app.callback(
    Output('rgb-composite', 'figure'),
    [Input('uploaded-data', 'data'),
     Input('excitation-selector', 'value'),
     Input('emission-red-selector', 'value'),
     Input('emission-green-selector', 'value'),
     Input('emission-blue-selector', 'value'),
     Input('cube-type-selector', 'value'),
     Input('rgb-contrast-slider', 'value')]
)
def update_rgb_composite(data, excitation, emission_red, emission_green, emission_blue, cube_type, contrast):
    if (data is None or excitation is None or
            emission_red is None or emission_green is None or emission_blue is None):
        return go.Figure().update_layout(title="Select excitation and emission wavelengths for RGB channels")

    # Generate the RGB composite
    fig = generate_rgb_composite(
        data, excitation, emission_red, emission_green, emission_blue, cube_type, contrast
    )

    return fig


# Run the server
if __name__ == '__main__':
    app.run(debug=True)