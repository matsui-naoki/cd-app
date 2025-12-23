"""
CD Analyzer - Streamlit Web Application
Charge-Discharge Curve Analysis Tool for Battery Research
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import io
import sys
import os
import re
import matplotlib.pyplot as plt
import matplotlib.colors as plt_colors

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

from tools.data_loader import (
    load_biologic_mpt, load_biologic_mpr, parse_biologic_header, get_supported_formats,
    load_uploaded_file, validate_cd_data
)
from tools.mps_parser import (
    parse_mps_file, get_technique_summary, load_gcpl_data_from_session,
    load_peis_data_from_session, MeasurementSession
)
from components.plots import (
    create_cd_plot, create_capacity_voltage_plot, create_dqdv_plot,
    create_cycle_summary_plot, create_multi_file_vq_plot,
    get_publication_config, COLORS, apply_axis_range
)
from components.styles import inject_custom_css as inject_external_css
from utils.helpers import (
    calculate_capacity, calculate_dqdv, split_charge_discharge,
    calculate_coulombic_efficiency
)
from utils.igor_export import generate_igor_file, generate_csv_export

# Page configuration
st.set_page_config(
    page_title="CD Analyzer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)


def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'files' not in st.session_state:
        st.session_state.files = {}
    if 'selected_file' not in st.session_state:
        st.session_state.selected_file = None
    if 'sample_info' not in st.session_state:
        st.session_state.sample_info = {
            'name': '',
            'mass_mg': 10.0,  # Total cathode mass (mg)
            'active_ratio': 0.7,  # Active material ratio (0-1)
            'area_cm2': 1.0,  # Electrode area (cm²) - default 1.0
            'diameter_cm': 1.0,  # Electrode diameter (cm) - default 1.0
            'area_input_mode': 'area',  # 'area' or 'diameter'
            'theoretical_capacity': 140.0,  # mAh/g
            'capacity_unit': 'mAh/g',  # 'mAh/g' or 'mAh/cm²'
            # Theoretical capacity calculator fields
            'composition': '',  # e.g., 'LiCoO2'
            'electron_number': 1,  # Number of electrons in reaction
        }
    if 'plot_settings' not in st.session_state:
        st.session_state.plot_settings = {
            'tick_font_size': 22,
            'axis_label_font_size': 22,
            'axis_line_width': 1,
            'line_width': 1,
            'marker_size': 0,
            'charge_color': '#E63946',
            'discharge_color': '#457B9D',
            'voltage_label': 'Voltage / V',
            'time_label': 'Time / h',
            'capacity_label': 'Capacity / mAh g⁻¹',
            'dqdv_label': 'dQ/dV / mAh g⁻¹ V⁻¹',
            'show_legend': True,
            'legend_font_size': 18,
            'legend_position': 'middle right',
            'legend_by_color': True,
            'show_electron_number': False,  # Show electron number on upper x-axis
        }
    if 'view_mode' not in st.session_state:
        st.session_state.view_mode = 'CD'
    if 'selected_cycles' not in st.session_state:
        st.session_state.selected_cycles = []
    if 'show_all_cycles' not in st.session_state:
        st.session_state.show_all_cycles = True
    if 'color_mode' not in st.session_state:
        st.session_state.color_mode = 'cycle'  # 'cycle' or 'charge_discharge'
    if 'selected_files' not in st.session_state:
        st.session_state.selected_files = []  # For multi-file V-Q plot
    # File colors and cycle info for multi-file mode
    if 'file_colors' not in st.session_state:
        st.session_state.file_colors = {}  # {filename: color}
    if 'file_cycle_info' not in st.session_state:
        st.session_state.file_cycle_info = {}  # {filename: {cycle_num, is_charge, is_discharge, current_mA}}
    # Cycle performance settings
    if 'capacity_display' not in st.session_state:
        st.session_state.capacity_display = 'discharge'  # 'charge' or 'discharge'
    if 'show_rate_labels' not in st.session_state:
        st.session_state.show_rate_labels = False
    # MPS session support
    if 'mps_session' not in st.session_state:
        st.session_state.mps_session = None
    if 'eis_data' not in st.session_state:
        st.session_state.eis_data = []
    # Axis range settings
    if 'axis_range' not in st.session_state:
        st.session_state.axis_range = {
            'x_min': None,
            'x_max': None,
            'y_min': None,
            'y_max': None,
            'enabled': False
        }
    # Track which files the axis range was calculated from
    if 'axis_range_files_hash' not in st.session_state:
        st.session_state.axis_range_files_hash = None
    # Graph offset settings
    if 'graph_offset' not in st.session_state:
        st.session_state.graph_offset = {
            'x_offset': 0.0,
            'y_offset': 0.0,
            'enabled': False
        }


def inject_custom_css():
    """Inject custom CSS for better UI"""
    inject_external_css(st)


def sidebar_header():
    """Render sidebar header"""
    st.markdown('<div class="sidebar-title">CD Analyzer</div>', unsafe_allow_html=True)


def process_uploaded_files(uploaded_files):
    """Process uploaded battery data files"""
    if not uploaded_files:
        return

    for uploaded_file in uploaded_files:
        filename = uploaded_file.name
        base_name = os.path.splitext(filename)[0]
        file_ext = os.path.splitext(filename)[1].lower()

        # Check if already loaded
        if base_name in st.session_state.files:
            continue

        # Handle MPS files specially
        if file_ext == '.mps':
            process_mps_file(uploaded_file)
            continue

        try:
            data, error = load_uploaded_file(uploaded_file, file_ext)

            if error:
                st.error(f"Error loading {filename}: {error}")
                continue

            if data is not None:
                is_valid, msg = validate_cd_data(data)
                if is_valid:
                    st.session_state.files[base_name] = data
                    st.success(f"Loaded: {filename}")
                else:
                    st.warning(f"Invalid data in {filename}: {msg}")

        except Exception as e:
            st.error(f"Error loading {filename}: {str(e)}")


def process_mps_file(uploaded_file):
    """Process MPS settings file and load related data files"""
    import tempfile

    try:
        # Save MPS file temporarily
        bytes_data = uploaded_file.read()

        with tempfile.TemporaryDirectory() as tmp_dir:
            # We need the original folder structure, so this won't work with just upload
            # Instead, inform user to upload all files together
            st.warning("MPS file detected. Please upload the entire measurement folder or use folder path input.")
            return

    except Exception as e:
        st.error(f"Error processing MPS file: {str(e)}")


def sidebar_file_upload():
    """File upload section"""
    st.markdown("### Upload Files")

    # Add MPS to supported formats
    supported_formats = get_supported_formats() + ['mps']

    uploaded_files = st.file_uploader(
        "Upload battery data files",
        type=supported_formats,
        accept_multiple_files=True,
        help="Supported: BioLogic (.mpt, .mpr, .mps), CSV, TXT",
        label_visibility="collapsed"
    )

    if uploaded_files:
        process_uploaded_files(uploaded_files)


def load_mps_session(mps_path: str):
    """Load measurement session from MPS file"""
    if not os.path.exists(mps_path):
        st.error(f"File not found: {mps_path}")
        return

    session = parse_mps_file(mps_path)
    if session is None:
        st.error("Failed to parse MPS file")
        return

    st.session_state.mps_session = session

    # Update sample info from MPS
    if session.sample_info:
        if 'mass_mg' in session.sample_info and session.sample_info['mass_mg'] > 0:
            st.session_state.sample_info['mass_mg'] = session.sample_info['mass_mg']
        if 'electrode_area_cm2' in session.sample_info and session.sample_info['electrode_area_cm2'] > 0:
            st.session_state.sample_info['area_cm2'] = session.sample_info['electrode_area_cm2']

    # Set sample name from base name
    st.session_state.sample_info['name'] = session.base_name

    # Load GCPL data
    gcpl_data = load_gcpl_data_from_session(session)
    if gcpl_data:
        st.session_state.files['GCPL_combined'] = gcpl_data
        st.session_state.selected_file = 'GCPL_combined'

    # Try to load EIS data
    try:
        eis_list = load_peis_data_from_session(session)
        st.session_state.eis_data = eis_list
    except Exception as e:
        st.warning(f"Could not load EIS data: {e}")

    st.success(f"Loaded session: {session.base_name}")
    st.rerun()


def sidebar_sample_info():
    """Sample information input section"""
    st.markdown("### Sample Information")

    st.session_state.sample_info['name'] = st.text_input(
        "Sample name",
        value=st.session_state.sample_info.get('name', ''),
        placeholder="Enter sample name"
    )

    # Mass input
    col1, col2 = st.columns(2)
    with col1:
        mass = st.number_input(
            "Cathode mass (mg)",
            value=st.session_state.sample_info.get('mass_mg', 10.0),
            min_value=0.001,
            step=0.1,
            format="%.3f",
            help="Total cathode composite mass"
        )
        st.session_state.sample_info['mass_mg'] = mass

    with col2:
        ratio = st.number_input(
            "Active ratio",
            value=st.session_state.sample_info.get('active_ratio', 0.7),
            min_value=0.01,
            max_value=1.0,
            step=0.05,
            format="%.2f",
            help="Active material ratio in composite (0-1)"
        )
        st.session_state.sample_info['active_ratio'] = ratio

    # Calculate and display active material mass
    active_mass_mg = mass * ratio
    st.caption(f"Active material: **{active_mass_mg:.3f} mg**")

    # Area input mode toggle
    area_mode = st.radio(
        "Electrode size input",
        options=['area', 'diameter'],
        format_func=lambda x: 'Area (cm²)' if x == 'area' else 'Diameter (cm)',
        index=0 if st.session_state.sample_info.get('area_input_mode', 'area') == 'area' else 1,
        horizontal=True
    )
    st.session_state.sample_info['area_input_mode'] = area_mode

    if area_mode == 'area':
        area = st.number_input(
            "Electrode area (cm²)",
            value=st.session_state.sample_info.get('area_cm2', 1.0),
            min_value=0.001,
            step=0.01,
            format="%.3f",
            help="Default: 1.0 cm²"
        )
        st.session_state.sample_info['area_cm2'] = area
        # Calculate diameter from area
        diameter = np.sqrt(4 * area / np.pi)
        st.session_state.sample_info['diameter_cm'] = diameter
    else:
        diameter = st.number_input(
            "Electrode diameter (cm)",
            value=st.session_state.sample_info.get('diameter_cm', 1.0),
            min_value=0.001,
            step=0.01,
            format="%.3f",
            help="Default: 1.0 cm"
        )
        st.session_state.sample_info['diameter_cm'] = diameter
        # Calculate area from diameter
        area = np.pi * (diameter / 2) ** 2
        st.session_state.sample_info['area_cm2'] = area
        st.caption(f"Area: **{area:.4f} cm²**")

    # Calculate loading
    loading_mg_cm2 = active_mass_mg / area
    st.caption(f"Loading: **{loading_mg_cm2:.3f} mg/cm²**")

    # Theoretical capacity (optional, collapsed by default)
    with st.expander("Theoretical Capacity"):
        # Formula input first
        composition = st.text_input(
            "Formula",
            value=st.session_state.sample_info.get('composition', ''),
            placeholder="e.g., LiCoO2, LiFePO4",
            help="Q = nF / (M × 3.6) [mAh/g], where n = reaction electrons, F = 96485 C/mol, M = molar mass"
        )
        st.session_state.sample_info['composition'] = composition

        # Number of reaction electrons (float support)
        electron_n = st.number_input(
            "Number of reaction electrons",
            value=float(st.session_state.sample_info.get('electron_number', 1.0)),
            min_value=0.1,
            max_value=10.0,
            step=0.1,
            format="%.1f",
            help="Number of electrons transferred in redox reaction (e.g., 0.5, 1, 2)"
        )
        st.session_state.sample_info['electron_number'] = electron_n

        # Calculate button
        if composition:
            if st.button("Calculate", use_container_width=True):
                try:
                    from utils.theocapacity import calculate_theoretical_capacity
                    calc_cap, molar_mass = calculate_theoretical_capacity(composition, electron_n)
                    if calc_cap is not None:
                        st.session_state.sample_info['theoretical_capacity'] = calc_cap
                        st.session_state.sample_info['calculated_molar_mass'] = molar_mass
                        st.rerun()
                except ImportError:
                    st.error("theocapacity module not available")
                except Exception as e:
                    st.error(f"Calculation error: {e}")

        # Show calculated or manual theoretical capacity
        theo_cap = st.number_input(
            "Theoretical capacity (mAh/g)",
            value=st.session_state.sample_info.get('theoretical_capacity', 140.0),
            min_value=0.1,
            step=10.0,
            format="%.1f"
        )
        st.session_state.sample_info['theoretical_capacity'] = theo_cap

        # Show molar mass if calculated
        if 'calculated_molar_mass' in st.session_state.sample_info:
            mw = st.session_state.sample_info['calculated_molar_mass']
            st.caption(f"Molar mass: {mw:.2f} g/mol")


def sidebar_file_manager():
    """File management section"""
    if len(st.session_state.files) == 0:
        st.info("No files loaded")
        return

    st.markdown("### Loaded Files")

    # Multi-select mode enabled for all plot views
    view_mode = st.session_state.view_mode
    multi_select_views = ['CD', 'CV/LSV', 'Nyquist', 'CCD', 'Custom']

    if view_mode in multi_select_views and len(st.session_state.files) > 1:
        st.caption("Select files for comparison")
        # Multi-select checkboxes
        selected = []
        for i, filename in enumerate(list(st.session_state.files.keys())):
            is_checked = filename in st.session_state.selected_files
            if st.checkbox(filename, value=is_checked, key=f"multi_{i}"):
                selected.append(filename)
        st.session_state.selected_files = selected

        # Also update single selection to first selected
        if selected and st.session_state.selected_file not in selected:
            st.session_state.selected_file = selected[0]
    else:
        # Single selection mode
        for i, filename in enumerate(list(st.session_state.files.keys())):
            is_selected = (filename == st.session_state.selected_file)

            col1, col2 = st.columns([4, 1])

            with col1:
                btn_type = "primary" if is_selected else "secondary"
                if st.button(filename, key=f"select_{i}", type=btn_type, use_container_width=True):
                    st.session_state.selected_file = filename
                    st.session_state.selected_files = [filename]
                    st.rerun()

            with col2:
                if st.button("✕", key=f"delete_{i}", help="Delete file"):
                    del st.session_state.files[filename]
                    if st.session_state.selected_file == filename:
                        st.session_state.selected_file = None
                    if filename in st.session_state.selected_files:
                        st.session_state.selected_files.remove(filename)
                    if filename in st.session_state.file_colors:
                        del st.session_state.file_colors[filename]
                    st.rerun()

    st.markdown("---")
    if st.button("Clear All", key="clear_all", use_container_width=True):
        st.session_state.files = {}
        st.session_state.selected_file = None
        st.session_state.selected_files = []
        st.session_state.file_colors = {}
        st.session_state.file_cycle_info = {}
        st.session_state.mps_session = None
        st.session_state.eis_data = []
        st.rerun()


def sidebar_view_mode():
    """View mode selection"""
    st.markdown("### View Mode")

    # Ordered view options (no emojis)
    view_options = {
        'CD': 'Charge-Discharge curve',
        'CV/LSV': 'CV / LSV',
        'Nyquist': 'Nyquist Plot',
        'CCD': 'Critical Current Density',
        'Custom': 'Custom Plot',
        'DataFrame': 'Data Table',
    }

    # Add Session view if MPS is loaded
    if st.session_state.mps_session:
        view_options['Session'] = 'Session Info'

    selected = st.radio(
        "Select view",
        options=list(view_options.keys()),
        format_func=lambda x: view_options[x],
        index=list(view_options.keys()).index(st.session_state.view_mode) if st.session_state.view_mode in view_options else 0,
        label_visibility="collapsed"
    )

    if selected != st.session_state.view_mode:
        st.session_state.view_mode = selected
        st.rerun()


def sidebar_cycle_selection():
    """Cycle selection and color mode settings"""
    # Only show for views that use cycles
    if st.session_state.view_mode not in ['CD', 'CCD']:
        return

    if st.session_state.selected_file is None:
        return

    if st.session_state.selected_file not in st.session_state.files:
        return

    data = st.session_state.files[st.session_state.selected_file]
    cycles = data.get('cycles', [])

    if not cycles:
        return

    st.markdown("### Cycle Selection")

    # Get unique cycle numbers
    cycle_numbers = sorted(set(c.get('cycle_number', 0) for c in cycles))
    n_cycles = len(cycle_numbers)

    if n_cycles == 0:
        return

    # Show all cycles checkbox
    show_all = st.checkbox(
        "Show all cycles",
        value=st.session_state.show_all_cycles,
        key='show_all_checkbox'
    )
    st.session_state.show_all_cycles = show_all

    if not show_all:
        # Cycle range slider
        if n_cycles > 1:
            cycle_range = st.slider(
                "Cycle range",
                min_value=min(cycle_numbers) + 1,
                max_value=max(cycle_numbers) + 1,
                value=(min(cycle_numbers) + 1, max(cycle_numbers) + 1),
                step=1
            )
            # Convert to 0-indexed
            st.session_state.selected_cycles = list(range(cycle_range[0] - 1, cycle_range[1]))
        else:
            st.session_state.selected_cycles = cycle_numbers
    else:
        st.session_state.selected_cycles = None  # None means all cycles

    # Color mode selection (only for CD view)
    if st.session_state.view_mode == 'CD':
        st.markdown("##### Color mode")
        color_options = [
            'cycle',           # Rainbow by cycle number
            'charge_discharge', # Red=charge, Blue=discharge
            'first_last',      # 1st=red, middle=black, last=blue
            'grayscale',       # Black to gray gradient
            'single_black',    # All black (for publication)
        ]
        color_labels = {
            'cycle': 'Rainbow (cycle)',
            'charge_discharge': 'Charge(red)/Discharge(blue)',
            'first_last': '1st=red, mid=black, last=blue',
            'grayscale': 'Grayscale gradient',
            'single_black': 'All black',
        }
        current_index = color_options.index(st.session_state.color_mode) if st.session_state.color_mode in color_options else 0
        color_mode = st.selectbox(
            "Color by",
            options=color_options,
            format_func=lambda x: color_labels.get(x, x),
            index=current_index,
            label_visibility="collapsed"
        )
        st.session_state.color_mode = color_mode


def sort_files_by_time_and_assign_cycles(files_data: dict) -> list:
    """
    Sort files by their start time and assign cycle numbers.
    Pairs charge+discharge as one cycle.

    Returns list of dicts: [{filename, data, start_time, is_charge, is_discharge, cycle_num, current_mA, color}]
    """
    sorted_files = []

    for filename, data in files_data.items():
        # Get start time
        if 'time' in data and data['time'] is not None and len(data['time']) > 0:
            start_time = data['time'][0]
        else:
            start_time = 0

        # Determine if charge or discharge from current
        is_charge = False
        is_discharge = False
        current_mA = 0

        if 'current' in data and data['current'] is not None and len(data['current']) > 0:
            avg_current = np.mean(data['current'])
            current_mA = abs(avg_current)
            if avg_current > 0.01:
                is_charge = True
            elif avg_current < -0.01:
                is_discharge = True

        # Check cycles for charge/discharge info
        if 'cycles' in data and len(data['cycles']) > 0:
            cycle = data['cycles'][0]
            if cycle.get('is_charge'):
                is_charge = True
            if cycle.get('is_discharge'):
                is_discharge = True

        sorted_files.append({
            'filename': filename,
            'data': data,
            'start_time': start_time,
            'is_charge': is_charge,
            'is_discharge': is_discharge,
            'current_mA': current_mA,
        })

    # Sort by start time
    sorted_files.sort(key=lambda x: x['start_time'])

    # Assign cycle numbers (pair charge+discharge as one cycle)
    cycle_num = 0
    half_cycle_in_pair = 0

    for i, file_info in enumerate(sorted_files):
        file_info['cycle_num'] = cycle_num
        half_cycle_in_pair += 1

        # After 2 half-cycles, increment cycle number
        if half_cycle_in_pair >= 2:
            cycle_num += 1
            half_cycle_in_pair = 0

    # Assign default colors: cycle1=red, middle=black, last=blue
    n_cycles = cycle_num + 1 if half_cycle_in_pair > 0 else cycle_num

    for file_info in sorted_files:
        filename = file_info['filename']
        cyc = file_info['cycle_num']

        # Check if user has set custom color
        if filename in st.session_state.file_colors:
            file_info['color'] = st.session_state.file_colors[filename]
        else:
            # Default: first=red, last=blue, middle=black
            if cyc == 0:
                file_info['color'] = '#E63946'  # Red
            elif cyc == n_cycles - 1:
                file_info['color'] = '#457B9D'  # Blue
            else:
                file_info['color'] = '#000000'  # Black

    return sorted_files


def render_data_list_panel(sorted_files: list):
    """Render data list panel below plot for per-cycle color settings"""
    # Initialize cycle_colors in session state if not present
    if 'cycle_colors' not in st.session_state:
        st.session_state.cycle_colors = {}

    # Count total cycles from all files (D+C pairs)
    # First, collect all half-cycles with their start times
    all_half_cycles = []
    for file_info in sorted_files:
        data = file_info['data']
        filename = file_info['filename']
        is_charge = file_info.get('is_charge', False)
        is_discharge = file_info.get('is_discharge', False)

        if 'cycles' in data and len(data['cycles']) > 0:
            for i, cycle in enumerate(data['cycles']):
                start_time = cycle['time'][0] if 'time' in cycle and len(cycle['time']) > 0 else 0
                all_half_cycles.append({
                    'start_time': start_time,
                    'type': 'C' if is_charge else ('D' if is_discharge else ''),
                })
        elif 'capacity' in data and data['capacity'] is not None:
            start_time = data['time'][0] if 'time' in data and len(data['time']) > 0 else 0
            all_half_cycles.append({
                'start_time': start_time,
                'type': 'C' if is_charge else ('D' if is_discharge else ''),
            })

    # Sort by time and assign cycle numbers (D+C = 1 cycle)
    all_half_cycles.sort(key=lambda x: x['start_time'])
    n_half_cycles = len(all_half_cycles)
    n_cycles = (n_half_cycles + 1) // 2  # Round up for odd number of half-cycles

    if n_cycles == 0:
        return

    # Define color schemes
    COLOR_SCHEMES = {
        'Default (Red-Black-Blue)': lambda i, n: '#E63946' if i == 1 else ('#009BFF' if i == n else '#000000'),
        'Rainbow': lambda i, n: [
            '#E63946', '#F77F00', '#FCBF49', '#90BE6D', '#43AA8B',
            '#577590', '#6D6875', '#9B5DE5', '#F15BB5', '#00BBF9'
        ][(i - 1) % 10],
        'Viridis': lambda i, n: plt_colors.rgb2hex(plt.cm.viridis((i - 1) / max(1, n - 1))) if n > 1 else '#440154',
        'Plasma': lambda i, n: plt_colors.rgb2hex(plt.cm.plasma((i - 1) / max(1, n - 1))) if n > 1 else '#0D0887',
        'Inferno': lambda i, n: plt_colors.rgb2hex(plt.cm.inferno((i - 1) / max(1, n - 1))) if n > 1 else '#000004',
        'Cool': lambda i, n: plt_colors.rgb2hex(plt.cm.cool((i - 1) / max(1, n - 1))) if n > 1 else '#00FFFF',
        'Grayscale': lambda i, n: f'#{int(255 * (1 - (i - 1) / max(1, n - 1))):02x}{int(255 * (1 - (i - 1) / max(1, n - 1))):02x}{int(255 * (1 - (i - 1) / max(1, n - 1))):02x}' if n > 1 else '#000000',
        'Custom': None,  # Use individual color pickers
    }

    # Initialize color_scheme in session state
    if 'color_scheme' not in st.session_state:
        st.session_state.color_scheme = 'Default (Red-Black-Blue)'

    # Cycle Colors in collapsible expander
    with st.expander(f"Cycle Colors ({n_cycles} cycles)", expanded=False):
        # Color scheme selector
        color_scheme = st.selectbox(
            "Color scheme",
            options=list(COLOR_SCHEMES.keys()),
            index=list(COLOR_SCHEMES.keys()).index(st.session_state.color_scheme),
            key='color_scheme_selector'
        )

        # Update colors when scheme changes
        if color_scheme != st.session_state.color_scheme:
            st.session_state.color_scheme = color_scheme
            if color_scheme != 'Custom' and COLOR_SCHEMES[color_scheme]:
                # Apply scheme to all cycles
                for cycle_num in range(1, n_cycles + 1):
                    st.session_state.cycle_colors[cycle_num] = COLOR_SCHEMES[color_scheme](cycle_num, n_cycles)
            st.rerun()

        # Show color pickers for all cycles (clickable to change)
        # Display in rows of 10 cycles
        cycles_per_row = 10
        for row_start in range(1, n_cycles + 1, cycles_per_row):
            row_end = min(row_start + cycles_per_row - 1, n_cycles)
            cols = st.columns(min(cycles_per_row, n_cycles - row_start + 1))
            for i, col in enumerate(cols):
                cycle_num = row_start + i
                if cycle_num > n_cycles:
                    break

                with col:
                    # Get current color from session state or scheme default
                    scheme_color = COLOR_SCHEMES[color_scheme](cycle_num, n_cycles) if COLOR_SCHEMES[color_scheme] else '#000000'
                    current_color = st.session_state.cycle_colors.get(cycle_num, scheme_color)

                    new_color = st.color_picker(
                        f"{cycle_num}",
                        value=current_color,
                        key=f"cycle_color_{cycle_num}",
                    )
                    if new_color != current_color:
                        st.session_state.cycle_colors[cycle_num] = new_color
                        # Switch to Custom mode when user manually changes a color
                        if color_scheme != 'Custom':
                            st.session_state.color_scheme = 'Custom'
                        st.rerun()

        # Add caption below color palette
        st.caption(f"Cycles 1-{n_cycles} (click to change color)")


def render_main_plot():
    """Render the main plot area"""
    view_mode = st.session_state.view_mode

    # Handle Session Info view
    if view_mode == 'Session' and st.session_state.mps_session:
        render_session_info()
        return

    # Handle Nyquist view (EIS)
    if view_mode == 'Nyquist':
        render_nyquist_plot()
        return

    # Handle DataFrame view
    if view_mode == 'DataFrame':
        render_dataframe_view()
        return

    # Handle CV/LSV view
    if view_mode == 'CV/LSV':
        render_cv_lsv_plot()
        return

    # Handle CCD (Critical Current Density) view
    if view_mode == 'CCD':
        render_ccd_plot()
        return

    # Handle Custom view
    if view_mode == 'Custom':
        render_custom_plot()
        return

    # Standard CD view (Charge-Discharge curve with dQ/dV and Cycle Performance)
    if st.session_state.selected_file is None:
        st.info("Select a file from the sidebar to view data")
        return

    if st.session_state.selected_file not in st.session_state.files:
        st.warning("Selected file not found")
        return

    data = st.session_state.files[st.session_state.selected_file]
    settings = st.session_state.plot_settings
    sample_info = st.session_state.sample_info
    selected_cycles = st.session_state.selected_cycles
    color_mode = st.session_state.color_mode

    # Display file info
    st.markdown(f"### {st.session_state.selected_file}")

    # Get plot config for export
    plot_config = get_publication_config()

    if view_mode == 'CD':
        # Main Charge-Discharge curve (V vs Q)
        selected_files = st.session_state.selected_files
        sorted_files = None

        # Create hash of selected files to detect changes
        current_files_hash = hash(tuple(sorted(selected_files))) if selected_files else None

        # Reset axis range if selected files changed
        if current_files_hash != st.session_state.axis_range_files_hash:
            st.session_state.axis_range = {
                'x_min': None,
                'x_max': None,
                'y_min': None,
                'y_max': None,
                'enabled': st.session_state.axis_range.get('enabled', False)
            }
            st.session_state.axis_range_files_hash = current_files_hash

        if len(selected_files) > 1:
            # Multi-file mode: sort by time and assign cycles
            files_data = {fn: st.session_state.files[fn] for fn in selected_files if fn in st.session_state.files}
            sorted_files = sort_files_by_time_and_assign_cycles(files_data)

            # Calculate axis range from data with 5% margin (before plot creation)
            if sorted_files and len(sorted_files) > 0:
                all_x, all_y = [], []
                for file_info in sorted_files:
                    file_data = file_info['data']
                    if 'cycles' in file_data:
                        for cycle in file_data['cycles']:
                            if 'capacity' in cycle and 'voltage' in cycle:
                                cap = np.abs(np.array(cycle['capacity']))
                                cap = cap - cap[0]  # Start from 0
                                # Apply capacity normalization
                                mass_mg = sample_info.get('mass_mg', 10.0)
                                active_ratio = sample_info.get('active_ratio', 1.0)
                                active_mass_g = mass_mg * active_ratio / 1000
                                if active_mass_g > 0:
                                    cap = cap / active_mass_g
                                all_x.extend(cap.tolist())
                                all_y.extend(cycle['voltage'].tolist() if hasattr(cycle['voltage'], 'tolist') else list(cycle['voltage']))
                    elif 'capacity' in file_data and file_data['capacity'] is not None:
                        cap = np.abs(np.array(file_data['capacity']))
                        cap = cap - cap[0]
                        mass_mg = sample_info.get('mass_mg', 10.0)
                        active_ratio = sample_info.get('active_ratio', 1.0)
                        active_mass_g = mass_mg * active_ratio / 1000
                        if active_mass_g > 0:
                            cap = cap / active_mass_g
                        all_x.extend(cap.tolist())
                        all_y.extend(file_data['voltage'].tolist() if hasattr(file_data['voltage'], 'tolist') else list(file_data['voltage']))

                if all_x and all_y:
                    x_range = max(all_x) - min(all_x)
                    y_range = max(all_y) - min(all_y)
                    margin = 0.05  # 5% margin
                    data_x_min = min(all_x) - x_range * margin
                    data_x_max = max(all_x) + x_range * margin
                    data_y_min = min(all_y) - y_range * margin
                    data_y_max = max(all_y) + y_range * margin
                    # Round for cleaner values
                    data_x_min = round(data_x_min, 1)
                    data_x_max = round(data_x_max, 1)
                    data_y_min = round(data_y_min, 2)
                    data_y_max = round(data_y_max, 2)

                    # Set default axis range values from data
                    if st.session_state.axis_range.get('x_min') is None:
                        st.session_state.axis_range['x_min'] = data_x_min
                    if st.session_state.axis_range.get('x_max') is None:
                        st.session_state.axis_range['x_max'] = data_x_max
                    if st.session_state.axis_range.get('y_min') is None:
                        st.session_state.axis_range['y_min'] = data_y_min
                    if st.session_state.axis_range.get('y_max') is None:
                        st.session_state.axis_range['y_max'] = data_y_max

            # Create multi-file plot with sorted files and custom colors
            axis_range = st.session_state.axis_range
            show_electron_number = settings.get('show_electron_number', False)
            fig = create_multi_file_cd_plot(sorted_files, settings, sample_info, axis_range, show_electron_number)
        else:
            # Get show_electron_number setting
            show_electron_number = settings.get('show_electron_number', False)
            fig = create_capacity_voltage_plot(
                data, settings, sample_info, selected_cycles, color_mode,
                show_electron_number=show_electron_number
            )
            # Apply axis range for single-file mode
            axis_range = st.session_state.axis_range
            if axis_range:
                fig = apply_axis_range(fig, axis_range)

        # Use width_mode setting: Responsive (default) = use_container_width, Fixed = specified width
        use_responsive = st.session_state.plot_settings.get('width_mode', 'Responsive') == 'Responsive'
        st.plotly_chart(fig, use_container_width=use_responsive, config=plot_config)

        # Plot Settings below the chart (moved from sidebar)
        with st.expander("Plot Settings", expanded=False):
            # Row 1: Line and font sizes
            col1, col2, col3 = st.columns(3)
            with col1:
                new_line_width = st.number_input(
                    "Line width",
                    min_value=0.5, max_value=5.0, step=0.5,
                    value=float(st.session_state.plot_settings.get('line_width', 2.0)),
                    key='cd_line_width'
                )
                if new_line_width != st.session_state.plot_settings.get('line_width'):
                    st.session_state.plot_settings['line_width'] = new_line_width
                    st.rerun()
            with col2:
                new_tick_size = st.slider(
                    "Tick font size", 8, 24,
                    st.session_state.plot_settings.get('tick_font_size', 14),
                    key='cd_tick_font_size'
                )
                if new_tick_size != st.session_state.plot_settings.get('tick_font_size'):
                    st.session_state.plot_settings['tick_font_size'] = new_tick_size
                    st.rerun()
            with col3:
                new_label_size = st.slider(
                    "Axis label size", 10, 28,
                    st.session_state.plot_settings.get('axis_label_font_size', 16),
                    key='cd_axis_label_font_size'
                )
                if new_label_size != st.session_state.plot_settings.get('axis_label_font_size'):
                    st.session_state.plot_settings['axis_label_font_size'] = new_label_size
                    st.rerun()

            # Row 2: Axis and tick settings
            col4, col5, col6 = st.columns(3)
            with col4:
                new_axis_width = st.number_input(
                    "Axis width",
                    min_value=0.5, max_value=4.0, step=0.5,
                    value=float(st.session_state.plot_settings.get('axis_width', 1.0)),
                    key='cd_axis_width'
                )
                if new_axis_width != st.session_state.plot_settings.get('axis_width'):
                    st.session_state.plot_settings['axis_width'] = new_axis_width
                    st.rerun()
            with col5:
                new_tick_length = st.slider(
                    "Tick length", 2, 12,
                    st.session_state.plot_settings.get('tick_length', 5),
                    key='cd_tick_length'
                )
                if new_tick_length != st.session_state.plot_settings.get('tick_length'):
                    st.session_state.plot_settings['tick_length'] = new_tick_length
                    st.rerun()
            with col6:
                new_tick_width = st.number_input(
                    "Tick width",
                    min_value=0.5, max_value=4.0, step=0.5,
                    value=float(st.session_state.plot_settings.get('tick_width', 1.0)),
                    key='cd_tick_width'
                )
                if new_tick_width != st.session_state.plot_settings.get('tick_width'):
                    st.session_state.plot_settings['tick_width'] = new_tick_width
                    st.rerun()

            # Row 3: Legend settings
            col_leg1, col_leg2, col_leg3, col_leg4 = st.columns(4)
            with col_leg1:
                new_show_legend = st.checkbox(
                    "Show legend",
                    value=st.session_state.plot_settings.get('show_legend', True),
                    key='cd_show_legend'
                )
                if new_show_legend != st.session_state.plot_settings.get('show_legend'):
                    st.session_state.plot_settings['show_legend'] = new_show_legend
                    st.rerun()
            with col_leg2:
                new_legend_by_color = st.checkbox(
                    "Legend by color",
                    value=st.session_state.plot_settings.get('legend_by_color', True),
                    key='cd_legend_by_color',
                    help="Group cycles by color in legend"
                )
                if new_legend_by_color != st.session_state.plot_settings.get('legend_by_color'):
                    st.session_state.plot_settings['legend_by_color'] = new_legend_by_color
                    st.rerun()
            with col_leg3:
                legend_positions = [
                    'top right', 'middle right', 'bottom right',
                    'top left', 'middle left', 'bottom left',
                    'top center', 'bottom center'
                ]
                current_pos = st.session_state.plot_settings.get('legend_position', 'middle right')
                new_legend_position = st.selectbox(
                    "Legend position",
                    options=legend_positions,
                    index=legend_positions.index(current_pos) if current_pos in legend_positions else 0,
                    key='cd_legend_position'
                )
                if new_legend_position != st.session_state.plot_settings.get('legend_position'):
                    st.session_state.plot_settings['legend_position'] = new_legend_position
                    st.rerun()
            with col_leg4:
                new_legend_font_size = st.slider(
                    "Legend font size",
                    min_value=8, max_value=24,
                    value=st.session_state.plot_settings.get('legend_font_size', 18),
                    key='cd_legend_font_size'
                )
                if new_legend_font_size != st.session_state.plot_settings.get('legend_font_size'):
                    st.session_state.plot_settings['legend_font_size'] = new_legend_font_size
                    st.rerun()

            # Row 4: Figure size
            col_wm, col_fw, col_fh = st.columns(3)
            with col_wm:
                width_mode = st.selectbox(
                    "Width mode",
                    options=['Responsive', 'Fixed'],
                    index=0 if st.session_state.plot_settings.get('width_mode', 'Responsive') == 'Responsive' else 1,
                    key='cd_width_mode',
                    help="Responsive: follows window size, Fixed: uses specified width"
                )
                if width_mode != st.session_state.plot_settings.get('width_mode'):
                    st.session_state.plot_settings['width_mode'] = width_mode
                    st.rerun()
            with col_fw:
                new_fig_width = st.number_input(
                    "Fig width (px)",
                    min_value=300, max_value=1200,
                    value=st.session_state.plot_settings.get('fig_width', 700),
                    step=50,
                    key='cd_fig_width',
                    disabled=(st.session_state.plot_settings.get('width_mode', 'Responsive') == 'Responsive')
                )
                if new_fig_width != st.session_state.plot_settings.get('fig_width'):
                    st.session_state.plot_settings['fig_width'] = new_fig_width
                    st.rerun()
            with col_fh:
                new_fig_height = st.number_input(
                    "Fig height (px)",
                    min_value=300, max_value=1000,
                    value=st.session_state.plot_settings.get('fig_height', 500),
                    step=50,
                    key='cd_fig_height'
                )
                if new_fig_height != st.session_state.plot_settings.get('fig_height'):
                    st.session_state.plot_settings['fig_height'] = new_fig_height
                    st.rerun()

            # Axis Range section - calculate defaults from data with 5% margin
            st.markdown("**Axis Range**")

            # Calculate data range for defaults
            data_x_min, data_x_max = 0.0, 200.0
            data_y_min, data_y_max = 0.0, 5.0
            all_x, all_y = [], []

            if sorted_files and len(sorted_files) > 0:
                # Multi-file mode
                for file_info in sorted_files:
                    file_data = file_info['data']
                    if 'cycles' in file_data:
                        for cycle in file_data['cycles']:
                            if 'capacity' in cycle and 'voltage' in cycle:
                                cap = np.abs(np.array(cycle['capacity']))
                                cap = cap - cap[0]  # Start from 0
                                # Apply capacity normalization
                                mass_mg = sample_info.get('mass_mg', 10.0)
                                active_ratio = sample_info.get('active_ratio', 1.0)
                                active_mass_g = mass_mg * active_ratio / 1000
                                if active_mass_g > 0:
                                    cap = cap / active_mass_g
                                all_x.extend(cap.tolist())
                                all_y.extend(cycle['voltage'].tolist() if hasattr(cycle['voltage'], 'tolist') else list(cycle['voltage']))
                    elif 'capacity' in file_data and file_data['capacity'] is not None:
                        cap = np.abs(np.array(file_data['capacity']))
                        cap = cap - cap[0]
                        mass_mg = sample_info.get('mass_mg', 10.0)
                        active_ratio = sample_info.get('active_ratio', 1.0)
                        active_mass_g = mass_mg * active_ratio / 1000
                        if active_mass_g > 0:
                            cap = cap / active_mass_g
                        all_x.extend(cap.tolist())
                        all_y.extend(file_data['voltage'].tolist() if hasattr(file_data['voltage'], 'tolist') else list(file_data['voltage']))
            else:
                # Single-file mode
                if 'cycles' in data:
                    for cycle in data['cycles']:
                        if 'capacity' in cycle and 'voltage' in cycle:
                            cap = np.abs(np.array(cycle['capacity']))
                            cap = cap - cap[0]
                            mass_mg = sample_info.get('mass_mg', 10.0)
                            active_ratio = sample_info.get('active_ratio', 1.0)
                            active_mass_g = mass_mg * active_ratio / 1000
                            if active_mass_g > 0:
                                cap = cap / active_mass_g
                            all_x.extend(cap.tolist())
                            all_y.extend(cycle['voltage'].tolist() if hasattr(cycle['voltage'], 'tolist') else list(cycle['voltage']))
                elif 'capacity' in data and data['capacity'] is not None:
                    cap = np.abs(np.array(data['capacity']))
                    cap = cap - cap[0]
                    mass_mg = sample_info.get('mass_mg', 10.0)
                    active_ratio = sample_info.get('active_ratio', 1.0)
                    active_mass_g = mass_mg * active_ratio / 1000
                    if active_mass_g > 0:
                        cap = cap / active_mass_g
                    all_x.extend(cap.tolist())
                    all_y.extend(data['voltage'].tolist() if hasattr(data['voltage'], 'tolist') else list(data['voltage']))

            if all_x and all_y:
                x_range = max(all_x) - min(all_x)
                y_range = max(all_y) - min(all_y)
                margin = 0.05  # 5% margin
                data_x_min = min(all_x) - x_range * margin
                data_x_max = max(all_x) + x_range * margin
                data_y_min = min(all_y) - y_range * margin
                data_y_max = max(all_y) + y_range * margin
                # Round for cleaner values
                data_x_min = round(data_x_min, 1)
                data_x_max = round(data_x_max, 1)
                data_y_min = round(data_y_min, 2)
                data_y_max = round(data_y_max, 2)

            # Use calculated defaults if not already set
            if st.session_state.axis_range.get('x_min') is None:
                st.session_state.axis_range['x_min'] = data_x_min
            if st.session_state.axis_range.get('x_max') is None:
                st.session_state.axis_range['x_max'] = data_x_max
            if st.session_state.axis_range.get('y_min') is None:
                st.session_state.axis_range['y_min'] = data_y_min
            if st.session_state.axis_range.get('y_max') is None:
                st.session_state.axis_range['y_max'] = data_y_max

            col_y1, col_y2, col_x1, col_x2, col_reset = st.columns([1, 1, 1, 1, 0.6])
            with col_y1:
                y_min = st.number_input(
                    "V min", value=float(st.session_state.axis_range.get('y_min', data_y_min)),
                    step=0.1, format="%.2f", key="cd_y_min"
                )
                st.session_state.axis_range['y_min'] = y_min
            with col_y2:
                y_max = st.number_input(
                    "V max", value=float(st.session_state.axis_range.get('y_max', data_y_max)),
                    step=0.1, format="%.2f", key="cd_y_max"
                )
                st.session_state.axis_range['y_max'] = y_max
            with col_x1:
                x_min = st.number_input(
                    "Q min", value=float(st.session_state.axis_range.get('x_min', data_x_min)),
                    step=10.0, format="%.1f", key="cd_x_min"
                )
                st.session_state.axis_range['x_min'] = x_min
            with col_x2:
                x_max = st.number_input(
                    "Q max", value=float(st.session_state.axis_range.get('x_max', data_x_max)),
                    step=10.0, format="%.1f", key="cd_x_max"
                )
                st.session_state.axis_range['x_max'] = x_max
            with col_reset:
                st.write("")  # Add vertical spacing to align button
                if st.button("Reset", key="reset_axis_range", help="Reset axis range to auto-calculated values from data"):
                    st.session_state.axis_range['x_min'] = data_x_min
                    st.session_state.axis_range['x_max'] = data_x_max
                    st.session_state.axis_range['y_min'] = data_y_min
                    st.session_state.axis_range['y_max'] = data_y_max
                    st.rerun()

            # Capacity unit selection inside Plot Settings
            st.markdown("**Capacity Unit**")
            capacity_unit = st.radio(
                "Unit",
                options=['mAh/g', 'mAh/cm²'],
                index=0 if sample_info.get('capacity_unit', 'mAh/g') == 'mAh/g' else 1,
                horizontal=True,
                key='capacity_unit_cd',
                label_visibility='collapsed'
            )
            if capacity_unit != sample_info.get('capacity_unit'):
                st.session_state.sample_info['capacity_unit'] = capacity_unit
                st.rerun()

            # Show electron number on upper x-axis (only available when:
            # - theoretical capacity is calculated
            # - composition is set
            # - capacity unit is mAh/g)
            theo_cap = sample_info.get('theoretical_capacity', 0)
            composition = sample_info.get('composition', '')
            cap_unit = sample_info.get('capacity_unit', 'mAh/g')

            # Check if conditions are met to show the toggle
            can_show_electron_toggle = (theo_cap > 0 and composition and cap_unit == 'mAh/g')

            if can_show_electron_toggle:
                st.markdown("**Upper X-axis**")
                show_electron = st.checkbox(
                    "Show electron number",
                    value=st.session_state.plot_settings.get('show_electron_number', False),
                    key='cd_show_electron_number',
                    help=f"Display electron (*n*) in {composition} on upper x-axis"
                )
                if show_electron != st.session_state.plot_settings.get('show_electron_number'):
                    st.session_state.plot_settings['show_electron_number'] = show_electron
                    st.rerun()

        # Data list panel for multi-file mode (Cycle Colors)
        if sorted_files and len(sorted_files) > 1:
            render_data_list_panel(sorted_files)

        # Sub-panels: dQ/dV curve and Cycle Performance (stacked vertically)
        with st.expander("dQ/dV curve", expanded=False):
            if sorted_files and len(sorted_files) > 1:
                fig_dqdv = create_multi_file_dqdv_plot(sorted_files, settings, sample_info)
            else:
                fig_dqdv = create_dqdv_plot(data, settings, sample_info, selected_cycles)
            st.plotly_chart(fig_dqdv, use_container_width=True, config=plot_config)

            # Figure settings for dQ/dV (same style as main Plot Settings)
            st.markdown("**Figure Settings**")
            # Row 1: Line and font sizes
            col1, col2, col3 = st.columns(3)
            with col1:
                st.slider("Line width", 1, 5, settings.get('line_width', 1), key='dqdv_line_width')
            with col2:
                st.slider("Tick font size", 8, 24, settings.get('tick_font_size', 22), key='dqdv_tick_size')
            with col3:
                st.slider("Axis label size", 10, 28, settings.get('axis_label_font_size', 22), key='dqdv_label_size')

            # Row 2: Axis and tick settings
            col4, col5, col6 = st.columns(3)
            with col4:
                st.slider("Axis width", 1, 4, settings.get('axis_width', 1), key='dqdv_axis_width')
            with col5:
                st.slider("Tick length", 2, 12, settings.get('tick_length', 5), key='dqdv_tick_length')
            with col6:
                st.slider("Tick width", 1, 4, settings.get('tick_width', 1), key='dqdv_tick_width')

            # Row 3: Legend settings
            col_leg1, col_leg2 = st.columns(2)
            with col_leg1:
                st.checkbox("Show legend", value=settings.get('show_legend', True), key='dqdv_show_legend')
            with col_leg2:
                st.slider("Legend font size", 8, 24, settings.get('legend_font_size', 12), key='dqdv_legend_font_size')

            # Row 4: Figure size
            col_wm, col_fw, col_fh = st.columns(3)
            with col_wm:
                st.selectbox("Width mode", options=['Responsive', 'Fixed'],
                             index=0 if settings.get('width_mode', 'Responsive') == 'Responsive' else 1,
                             key='dqdv_width_mode')
            with col_fw:
                st.number_input("Fig width (px)", min_value=300, max_value=1200,
                                value=settings.get('fig_width', 700), step=50, key='dqdv_fig_width')
            with col_fh:
                st.number_input("Fig height (px)", min_value=300, max_value=1000,
                                value=settings.get('fig_height', 500), step=50, key='dqdv_fig_height')

        with st.expander("Cycle Performance", expanded=False):
            # Capacity display selector
            cap_display = st.radio(
                "Display",
                options=['Discharge', 'Charge'],
                index=0 if st.session_state.capacity_display == 'discharge' else 1,
                horizontal=True,
                key='capacity_display_selector'
            )
            st.session_state.capacity_display = cap_display.lower()

            # Rate label toggle
            show_rate = st.checkbox(
                "Show rate labels",
                value=st.session_state.show_rate_labels,
                key='show_rate_toggle'
            )
            st.session_state.show_rate_labels = show_rate

            if sorted_files and len(sorted_files) > 1:
                fig_perf = create_multi_file_cycle_performance(
                    sorted_files, settings, sample_info,
                    st.session_state.capacity_display,
                    st.session_state.show_rate_labels
                )
            else:
                fig_perf = create_cycle_summary_plot(data, settings, sample_info)
            st.plotly_chart(fig_perf, use_container_width=True, config=plot_config)

            # Figure settings for Cycle Performance (same style as main Plot Settings)
            st.markdown("**Figure Settings**")
            # Row 1: Line and font sizes
            col1, col2, col3 = st.columns(3)
            with col1:
                st.slider("Line width", 1, 5, settings.get('line_width', 1), key='cp_line_width')
            with col2:
                st.slider("Tick font size", 8, 24, settings.get('tick_font_size', 22), key='cp_tick_size')
            with col3:
                st.slider("Axis label size", 10, 28, settings.get('axis_label_font_size', 22), key='cp_label_size')

            # Row 2: Axis and tick settings
            col4, col5, col6 = st.columns(3)
            with col4:
                st.slider("Axis width", 1, 4, settings.get('axis_width', 1), key='cp_axis_width')
            with col5:
                st.slider("Tick length", 2, 12, settings.get('tick_length', 5), key='cp_tick_length')
            with col6:
                st.slider("Tick width", 1, 4, settings.get('tick_width', 1), key='cp_tick_width')

            # Row 3: Legend settings
            col_leg1, col_leg2 = st.columns(2)
            with col_leg1:
                st.checkbox("Show legend", value=settings.get('show_legend', True), key='cp_show_legend')
            with col_leg2:
                st.slider("Legend font size", 8, 24, settings.get('legend_font_size', 12), key='cp_legend_font_size')

            # Row 4: Figure size
            col_wm, col_fw, col_fh = st.columns(3)
            with col_wm:
                st.selectbox("Width mode", options=['Responsive', 'Fixed'],
                             index=0 if settings.get('width_mode', 'Responsive') == 'Responsive' else 1,
                             key='cp_width_mode')
            with col_fw:
                st.number_input("Fig width (px)", min_value=300, max_value=1200,
                                value=settings.get('fig_width', 700), step=50, key='cp_fig_width')
            with col_fh:
                st.number_input("Fig height (px)", min_value=300, max_value=1000,
                                value=settings.get('fig_height', 500), step=50, key='cp_fig_height')

    # Show data summary
    render_data_summary(data, sample_info)


def render_session_info():
    """Render measurement session information"""
    session = st.session_state.mps_session

    st.markdown(f"### Session: {session.base_name}")

    # Device info
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Device Info")
        for key, val in session.device_info.items():
            st.text(f"{key}: {val}")

    with col2:
        st.markdown("##### Sample Info")
        for key, val in session.sample_info.items():
            if val:
                st.text(f"{key}: {val}")

    st.markdown("---")

    # Technique table
    st.markdown("##### Measurement History")
    summary = get_technique_summary(session)
    df = pd.DataFrame(summary)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Data file status
    st.markdown("---")
    st.markdown("##### Data Files")
    for tech in session.techniques:
        if tech.has_data:
            st.markdown(f"[+] **{tech.index}. {tech.short_name}**: `{os.path.basename(tech.data_file)}`")
        else:
            st.markdown(f"[-] **{tech.index}. {tech.short_name}**: No data file")


def render_dataframe_view():
    """Render raw data as DataFrame table"""
    if st.session_state.selected_file is None:
        st.info("Select a file from the sidebar to view data")
        return

    if st.session_state.selected_file not in st.session_state.files:
        st.warning("Selected file not found")
        return

    data = st.session_state.files[st.session_state.selected_file]
    st.markdown(f"### Data Table: {st.session_state.selected_file}")

    # Priority 1: raw_df from mpt_loader
    if 'raw_df' in data and data['raw_df'] is not None:
        df = data['raw_df']
        st.dataframe(df, use_container_width=True, height=600)
        st.caption(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
        st.caption(f"Columns: {', '.join(df.columns.tolist())}")

        # Download CSV button
        csv_data = df.to_csv(index=False)
        st.download_button(
            "Download CSV",
            data=csv_data,
            file_name=f"{st.session_state.selected_file}_raw.csv",
            mime="text/csv"
        )
        return

    # Priority 2: df from other loaders
    if 'df' in data and data['df'] is not None:
        df = data['df']
        st.dataframe(df, use_container_width=True, height=600)
        st.caption(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
        st.caption(f"Columns: {', '.join(df.columns.tolist())}")

        csv_data = df.to_csv(index=False)
        st.download_button(
            "Download CSV",
            data=csv_data,
            file_name=f"{st.session_state.selected_file}_data.csv",
            mime="text/csv"
        )
        return

    # Priority 3: Build from arrays
    df_dict = {}
    array_keys = ['time', 'voltage', 'current', 'capacity', 'ns', 'cycle_number',
                  'freq', 're_z', 'im_z', 'z_abs', 'phase_z']

    for key in array_keys:
        if key in data and data[key] is not None:
            df_dict[key] = data[key]

    if df_dict:
        df = pd.DataFrame(df_dict)
        st.dataframe(df, use_container_width=True, height=600)
        st.caption(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")

        csv_data = df.to_csv(index=False)
        st.download_button(
            "Download CSV",
            data=csv_data,
            file_name=f"{st.session_state.selected_file}_data.csv",
            mime="text/csv"
        )
    else:
        st.warning("No raw data available for this file")


def render_nyquist_plot():
    """Render EIS Nyquist plot"""
    settings = st.session_state.plot_settings

    st.markdown("### EIS Data (Nyquist Plot)")

    # Check for EIS data from multiple sources
    eis_list = []

    # Source 1: st.session_state.eis_data (from MPS session)
    if st.session_state.eis_data:
        eis_list.extend(st.session_state.eis_data)

    # Source 2: Currently selected file (if it contains EIS data)
    if st.session_state.selected_file and st.session_state.selected_file in st.session_state.files:
        data = st.session_state.files[st.session_state.selected_file]
        # Check if this file has EIS data
        if 'Z_real' in data and data['Z_real'] is not None and 'Z_imag' in data and data['Z_imag'] is not None:
            # Add as EIS data
            eis_entry = {
                'Z_real': data['Z_real'],
                'Z_imag': data['Z_imag'],
                'technique_index': 1,
            }
            if 'freq' in data and data['freq'] is not None:
                eis_entry['freq'] = data['freq']
            eis_list.append(eis_entry)

    if not eis_list:
        st.info("No EIS data available. Load an EIS file (.mpr with PEIS data) to view Nyquist plot.")
        return

    # Create Nyquist plot
    fig = go.Figure()

    for i, eis in enumerate(eis_list):
        if 'Z_real' not in eis or 'Z_imag' not in eis:
            continue

        Z_real = eis['Z_real']
        Z_imag = eis['Z_imag']

        color = COLORS[i % len(COLORS)]
        name = f"PEIS {eis.get('technique_index', i+1)}"

        fig.add_trace(go.Scatter(
            x=Z_real,
            y=Z_imag,
            mode='markers',
            name=name,
            marker=dict(
                size=6,
                color=color,
                symbol='circle',
                line=dict(width=0.5, color='black')
            ),
            hovertemplate="Z' = %{x:.1f} Ω<br>-Z'' = %{y:.1f} Ω<extra></extra>"
        ))

    # Calculate axis range for 1:1 aspect ratio
    all_zr = np.concatenate([eis['Z_real'] for eis in eis_list if 'Z_real' in eis])
    all_zi = np.concatenate([eis['Z_imag'] for eis in eis_list if 'Z_imag' in eis])

    max_val = max(np.max(all_zr), np.max(all_zi)) * 1.1
    min_val = min(0, np.min(all_zr), np.min(all_zi))

    tick_size = settings.get('tick_font_size', 14)
    label_size = settings.get('axis_label_font_size', 16)

    fig.update_layout(
        font={'family': 'Arial', 'color': 'black'},
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=500,
        xaxis=dict(
            title="Z' / Ω",
            title_font=dict(size=label_size),
            tickfont=dict(size=tick_size),
            showgrid=False,
            showline=True,
            linewidth=1.5,
            linecolor='black',
            mirror=True,
            ticks='inside',
            range=[min_val, max_val],
        ),
        yaxis=dict(
            title="-Z'' / Ω",
            title_font=dict(size=label_size),
            tickfont=dict(size=tick_size),
            showgrid=False,
            showline=True,
            linewidth=1.5,
            linecolor='black',
            mirror=True,
            ticks='inside',
            scaleanchor='x',
            scaleratio=1,
            range=[min_val, max_val],
        ),
        showlegend=settings.get('show_legend', True),
        legend=dict(
            yanchor="top", y=0.99, xanchor="right", x=0.99,
            font=dict(size=settings.get('legend_font_size', 12)),
            bgcolor='rgba(255,255,255,0.8)'
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    # Show EIS summary table
    with st.expander("EIS Data Summary", expanded=False):
        for eis in eis_list:
            if 'freq' in eis:
                st.text(f"PEIS {eis.get('technique_index', '?')}: "
                       f"Freq range: {eis['freq'].min():.2e} - {eis['freq'].max():.2e} Hz, "
                       f"Points: {len(eis['freq'])}")


def render_bode_plot():
    """Render EIS Bode plot (|Z| and Phase vs Frequency)"""
    settings = st.session_state.plot_settings

    st.markdown("### EIS Data (Bode Plot)")

    # Check for EIS data from multiple sources (same as Nyquist)
    eis_list = []

    # Source 1: st.session_state.eis_data (from MPS session)
    if st.session_state.eis_data:
        eis_list.extend(st.session_state.eis_data)

    # Source 2: Currently selected file (if it contains EIS data)
    if st.session_state.selected_file and st.session_state.selected_file in st.session_state.files:
        data = st.session_state.files[st.session_state.selected_file]
        if 'Z_real' in data and data['Z_real'] is not None and 'Z_imag' in data and data['Z_imag'] is not None:
            eis_entry = {
                'Z_real': data['Z_real'],
                'Z_imag': data['Z_imag'],
                'technique_index': 1,
            }
            if 'freq' in data and data['freq'] is not None:
                eis_entry['freq'] = data['freq']
            eis_list.append(eis_entry)

    if not eis_list:
        st.info("No EIS data available. Load an EIS file (.mpr with PEIS data) to view Bode plot.")
        return

    # Bode plot options
    col1, col2 = st.columns([1, 3])
    with col1:
        bode_type = st.selectbox(
            "Plot type",
            options=['both', 'impedance', 'phase'],
            format_func=lambda x: {'both': '|Z| + Phase', 'impedance': '|Z| only', 'phase': 'Phase only'}[x],
            index=0
        )

    # Create Bode plot
    fig = create_bode_plot(eis_list, settings, plot_type=bode_type)
    st.plotly_chart(fig, use_container_width=True)

    # Show EIS summary table
    with st.expander("EIS Data Summary", expanded=False):
        for eis in eis_list:
            if 'freq' in eis and 'Z_real' in eis and 'Z_imag' in eis:
                Z_mag = np.sqrt(eis['Z_real']**2 + eis['Z_imag']**2)
                st.text(f"PEIS {eis.get('technique_index', '?')}: "
                       f"Freq: {eis['freq'].min():.2e} - {eis['freq'].max():.2e} Hz, "
                       f"|Z|: {Z_mag.min():.1f} - {Z_mag.max():.1f} Ω")


def render_cv_lsv_plot():
    """Render CV/LSV plot (Current vs Voltage)"""
    if st.session_state.selected_file is None:
        st.info("Select a file from the sidebar to view data")
        return

    if st.session_state.selected_file not in st.session_state.files:
        st.warning("Selected file not found")
        return

    data = st.session_state.files[st.session_state.selected_file]
    settings = st.session_state.plot_settings
    sample_info = st.session_state.sample_info

    st.markdown(f"### CV/LSV: {st.session_state.selected_file}")

    # Check if we have voltage and current data
    if 'voltage' not in data or 'current' not in data:
        st.warning("No voltage/current data available for CV/LSV plot")
        return

    voltage = data['voltage']
    current = data['current']

    if voltage is None or current is None:
        st.warning("Voltage or current data is missing")
        return

    # Normalize current to current density if area is available
    area = sample_info.get('area_cm2', 1.0)

    # Current density option
    use_density = st.checkbox("Show current density (mA/cm²)", value=True)

    if use_density:
        current_display = current / area
        y_label = 'Current density / mA cm⁻²'
    else:
        current_display = current
        y_label = 'Current / mA'

    # Create figure
    fig = go.Figure()

    tick_size = settings.get('tick_font_size', 22)
    label_size = settings.get('axis_label_font_size', 22)
    line_width = settings.get('line_width', 1)

    fig.add_trace(go.Scatter(
        x=voltage,
        y=current_display,
        mode='lines',
        line=dict(width=line_width, color='#1f77b4'),
        hovertemplate='E = %{x:.3f} V<br>I = %{y:.3f}<extra></extra>'
    ))

    fig.update_layout(
        font={'family': 'Arial', 'color': 'black'},
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=500,
        xaxis=dict(
            title="Potential / V",
            title_font=dict(size=label_size),
            tickfont=dict(size=tick_size),
            showgrid=False,
            showline=True,
            linewidth=1,
            linecolor='black',
            mirror=True,
            ticks='inside',
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor='gray',
        ),
        yaxis=dict(
            title=y_label,
            title_font=dict(size=label_size),
            tickfont=dict(size=tick_size),
            showgrid=False,
            showline=True,
            linewidth=1,
            linecolor='black',
            mirror=True,
            ticks='inside',
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor='gray',
        ),
        showlegend=False,
    )

    plot_config = get_publication_config()
    st.plotly_chart(fig, use_container_width=True, config=plot_config)

    # Data summary
    with st.expander("CV/LSV Data Summary", expanded=False):
        st.text(f"Voltage range: {voltage.min():.3f} - {voltage.max():.3f} V")
        st.text(f"Current range: {current.min():.3f} - {current.max():.3f} mA")
        if use_density:
            st.text(f"Current density range: {current_display.min():.3f} - {current_display.max():.3f} mA/cm²")
        st.text(f"Data points: {len(voltage)}")


def render_ccd_plot():
    """Render Critical Current Density (CCD) plot
    Left Y-axis (black): Overpotential / mV
    X-axis: Time / h
    Right Y-axis (blue): Applied current / mA
    """
    if st.session_state.selected_file is None:
        st.info("Select a file from the sidebar to view data")
        return

    if st.session_state.selected_file not in st.session_state.files:
        st.warning("Selected file not found")
        return

    data = st.session_state.files[st.session_state.selected_file]
    settings = st.session_state.plot_settings

    st.markdown(f"### Critical Current Density: {st.session_state.selected_file}")

    # Check for required data
    if 'voltage' not in data or data['voltage'] is None:
        st.warning("No voltage data available for CCD plot")
        return
    if 'time' not in data or data['time'] is None:
        st.warning("No time data available for CCD plot")
        return

    voltage = data['voltage']
    time_s = data['time']
    time_h = time_s / 3600  # Convert to hours

    current = data.get('current', None)

    # Calculate overpotential (deviation from OCV or reference)
    # For CCD, overpotential is typically voltage - OCV
    # Here we'll use the voltage directly as overpotential proxy, converting to mV
    overpotential_mV = voltage * 1000  # V to mV (adjust as needed)

    tick_size = settings.get('tick_font_size', 22)
    label_size = settings.get('axis_label_font_size', 22)
    line_width = settings.get('line_width', 1)

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Left Y-axis: Overpotential (black)
    fig.add_trace(
        go.Scatter(
            x=time_h,
            y=overpotential_mV,
            mode='lines',
            name='Overpotential',
            line=dict(width=line_width, color='black'),
            hovertemplate='t = %{x:.2f} h<br>η = %{y:.1f} mV<extra></extra>'
        ),
        secondary_y=False
    )

    # Right Y-axis: Applied current (blue)
    if current is not None:
        fig.add_trace(
            go.Scatter(
                x=time_h,
                y=current,
                mode='lines',
                name='Current',
                line=dict(width=line_width, color='#1f77b4'),
                hovertemplate='t = %{x:.2f} h<br>I = %{y:.3f} mA<extra></extra>'
            ),
            secondary_y=True
        )

    fig.update_layout(
        font={'family': 'Arial', 'color': 'black'},
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=500,
        showlegend=True,
        legend=dict(
            yanchor="top", y=0.99, xanchor="right", x=0.99,
            font=dict(size=settings.get('legend_font_size', 12)),
            bgcolor='rgba(255,255,255,0.8)'
        ),
    )

    # Left Y-axis (Overpotential)
    fig.update_yaxes(
        title_text="Overpotential / mV",
        title_font=dict(size=label_size, color='black'),
        tickfont=dict(size=tick_size, color='black'),
        showgrid=False,
        showline=True,
        linewidth=1,
        linecolor='black',
        mirror=False,
        ticks='inside',
        secondary_y=False
    )

    # Right Y-axis (Current)
    fig.update_yaxes(
        title_text="Applied current / mA",
        title_font=dict(size=label_size, color='#1f77b4'),
        tickfont=dict(size=tick_size, color='#1f77b4'),
        showgrid=False,
        showline=True,
        linewidth=1,
        linecolor='#1f77b4',
        mirror=False,
        ticks='inside',
        secondary_y=True
    )

    # X-axis (Time)
    fig.update_xaxes(
        title_text="Time / h",
        title_font=dict(size=label_size),
        tickfont=dict(size=tick_size),
        showgrid=False,
        showline=True,
        linewidth=1,
        linecolor='black',
        mirror=True,
        ticks='inside',
    )

    plot_config = get_publication_config()
    st.plotly_chart(fig, use_container_width=True, config=plot_config)

    # Data summary
    with st.expander("CCD Data Summary", expanded=False):
        st.text(f"Time range: {time_h.min():.2f} - {time_h.max():.2f} h")
        st.text(f"Voltage range: {voltage.min():.3f} - {voltage.max():.3f} V")
        if current is not None:
            st.text(f"Current range: {current.min():.3f} - {current.max():.3f} mA")
        st.text(f"Data points: {len(voltage)}")


def render_custom_plot():
    """Render custom plot with user-selectable X and Y axes"""
    if st.session_state.selected_file is None:
        st.info("Select a file from the sidebar to view data")
        return

    if st.session_state.selected_file not in st.session_state.files:
        st.warning("Selected file not found")
        return

    data = st.session_state.files[st.session_state.selected_file]
    settings = st.session_state.plot_settings

    st.markdown(f"### Custom Plot: {st.session_state.selected_file}")

    # Get available columns
    available_cols = []
    col_data = {}

    # Check for standard arrays
    standard_keys = ['time', 'voltage', 'current', 'capacity', 'freq', 'Z_real', 'Z_imag']
    for key in standard_keys:
        if key in data and data[key] is not None and len(data[key]) > 0:
            available_cols.append(key)
            col_data[key] = data[key]

    # Also check raw_df columns
    if 'raw_df' in data and data['raw_df'] is not None:
        df = data['raw_df']
        for col in df.columns:
            if col not in available_cols and df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                available_cols.append(col)
                col_data[col] = df[col].values

    if len(available_cols) < 2:
        st.warning("Not enough numeric columns for custom plot")
        return

    # Column selection
    col1, col2 = st.columns(2)
    with col1:
        x_col = st.selectbox("X-axis", options=available_cols, index=0, key='custom_x')
    with col2:
        y_col = st.selectbox("Y-axis", options=available_cols, index=min(1, len(available_cols)-1), key='custom_y')

    x_data = col_data[x_col]
    y_data = col_data[y_col]

    # Ensure same length
    min_len = min(len(x_data), len(y_data))
    x_data = x_data[:min_len]
    y_data = y_data[:min_len]

    tick_size = settings.get('tick_font_size', 22)
    label_size = settings.get('axis_label_font_size', 22)
    line_width = settings.get('line_width', 1)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x_data,
        y=y_data,
        mode='lines',
        line=dict(width=line_width, color='#1f77b4'),
        hovertemplate=f'{x_col} = %{{x:.3f}}<br>{y_col} = %{{y:.3f}}<extra></extra>'
    ))

    fig.update_layout(
        font={'family': 'Arial', 'color': 'black'},
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=500,
        xaxis=dict(
            title=x_col,
            title_font=dict(size=label_size),
            tickfont=dict(size=tick_size),
            showgrid=False,
            showline=True,
            linewidth=1,
            linecolor='black',
            mirror=True,
            ticks='inside',
        ),
        yaxis=dict(
            title=y_col,
            title_font=dict(size=label_size),
            tickfont=dict(size=tick_size),
            showgrid=False,
            showline=True,
            linewidth=1,
            linecolor='black',
            mirror=True,
            ticks='inside',
        ),
        showlegend=False,
    )

    plot_config = get_publication_config()
    st.plotly_chart(fig, use_container_width=True, config=plot_config)

    # Data summary
    with st.expander("Custom Plot Data Summary", expanded=False):
        st.text(f"{x_col} range: {x_data.min():.4g} - {x_data.max():.4g}")
        st.text(f"{y_col} range: {y_data.min():.4g} - {y_data.max():.4g}")
        st.text(f"Data points: {len(x_data)}")


def get_legend_position_config(position: str) -> dict:
    """Convert legend position string to Plotly legend config"""
    positions = {
        'top right': {'yanchor': 'top', 'y': 0.99, 'xanchor': 'right', 'x': 0.99},
        'top left': {'yanchor': 'top', 'y': 0.99, 'xanchor': 'left', 'x': 0.01},
        'bottom right': {'yanchor': 'bottom', 'y': 0.01, 'xanchor': 'right', 'x': 0.99},
        'bottom left': {'yanchor': 'bottom', 'y': 0.01, 'xanchor': 'left', 'x': 0.01},
        'top center': {'yanchor': 'top', 'y': 0.99, 'xanchor': 'center', 'x': 0.5},
        'bottom center': {'yanchor': 'bottom', 'y': 0.01, 'xanchor': 'center', 'x': 0.5},
        'middle right': {'yanchor': 'middle', 'y': 0.5, 'xanchor': 'right', 'x': 0.99},
        'middle left': {'yanchor': 'middle', 'y': 0.5, 'xanchor': 'left', 'x': 0.01},
    }
    return positions.get(position, positions['top right'])


def create_multi_file_cd_plot(sorted_files: list, settings: dict, sample_info: dict, axis_range: dict = None, show_electron_number: bool = False) -> go.Figure:
    """Create Charge-Discharge plot for multiple files with custom colors

    Parameters
    ----------
    show_electron_number : bool
        If True and theoretical capacity is available with mAh/g unit,
        show electron number on upper x-axis
    """
    fig = go.Figure()

    # Calculate active mass considering active_ratio (same as single-file version)
    mass_mg = sample_info.get('mass_mg', 10.0)
    active_ratio = sample_info.get('active_ratio', 1.0)
    active_mass_g = mass_mg * active_ratio / 1000  # g
    if active_mass_g <= 0:
        active_mass_g = 0.001
    area_cm2 = sample_info.get('area_cm2', 1.0)
    capacity_unit = sample_info.get('capacity_unit', 'mAh/g')

    tick_size = settings.get('tick_font_size', 22)
    label_size = settings.get('axis_label_font_size', 22)
    line_width = settings.get('line_width', 1)
    axis_width = settings.get('axis_width', 1)
    tick_length = settings.get('tick_length', 5)
    tick_width = settings.get('tick_width', 1)
    show_legend = settings.get('show_legend', True)

    x_label = 'Capacity / mAh g⁻¹' if capacity_unit == 'mAh/g' else 'Capacity / mAh cm⁻²'

    # Collect all cycles from all files with their start times for sorting
    all_cycles_data = []

    for file_info in sorted_files:
        data = file_info['data']
        filename = file_info['filename']
        is_charge = file_info.get('is_charge', False)
        is_discharge = file_info.get('is_discharge', False)
        type_label = 'C' if is_charge else ('D' if is_discharge else '')

        if 'cycles' in data and len(data['cycles']) > 0:
            for i, cycle in enumerate(data['cycles']):
                if 'voltage' not in cycle or 'capacity' not in cycle:
                    continue
                # Get start time for sorting
                start_time = cycle['time'][0] if 'time' in cycle and len(cycle['time']) > 0 else 0

                # Normalize capacity: absolute value, then start from 0
                cap_raw = np.array(cycle['capacity'])
                cap_abs = np.abs(cap_raw)
                cap_normalized = cap_abs - cap_abs[0]  # Start from 0

                all_cycles_data.append({
                    'filename': filename,
                    'voltage': cycle['voltage'],
                    'capacity': cap_normalized,
                    'start_time': start_time,
                    'type_label': type_label,
                })
        elif 'capacity' in data and data['capacity'] is not None and 'voltage' in data:
            start_time = data['time'][0] if 'time' in data and len(data['time']) > 0 else 0
            # Normalize capacity: absolute value, then start from 0
            cap_raw = np.array(data['capacity'])
            cap_abs = np.abs(cap_raw)
            cap_normalized = cap_abs - cap_abs[0]  # Start from 0
            all_cycles_data.append({
                'filename': filename,
                'voltage': data['voltage'],
                'capacity': cap_normalized,
                'start_time': start_time,
                'type_label': type_label,
            })

    # Sort all cycles by start time
    all_cycles_data.sort(key=lambda x: x['start_time'])

    # Assign cycle numbers: D+C pair = 1 cycle
    # Pattern: D1, C1, D2, C2, ... -> Cycle 1 (D), Cycle 1 (C), Cycle 2 (D), Cycle 2 (C), ...
    cycle_number = 1
    half_cycle_count = 0
    for cyc_data in all_cycles_data:
        cyc_data['cycle_number'] = cycle_number
        half_cycle_count += 1
        # After 2 half-cycles (D+C), increment cycle number
        if half_cycle_count >= 2:
            cycle_number += 1
            half_cycle_count = 0

    # Calculate total real cycles correctly
    # After loop: cycle_number points to next cycle, so we need cycle_number - 1 if complete
    # But if half_cycle_count > 0, the last cycle is still in progress
    total_real_cycles = cycle_number - 1 if half_cycle_count == 0 else cycle_number
    if total_real_cycles < 1:
        total_real_cycles = 1
    total_half_cycles = len(all_cycles_data)

    # Color palette for cycles (first=red, middle=black, last=blue)
    # Check for user-defined cycle colors in session state
    cycle_colors_from_state = st.session_state.get('cycle_colors', {})

    def get_cycle_color(cycle_num: int, total: int) -> str:
        # First check session state for user-defined color
        if cycle_num in cycle_colors_from_state:
            return cycle_colors_from_state[cycle_num]
        # Otherwise use default color scheme
        if total <= 1:
            return '#E63946'  # Red
        elif cycle_num == 1:
            return '#E63946'  # Red for first cycle
        elif cycle_num == total:
            return '#009BFF'  # Blue for last cycle
        else:
            return '#000000'  # Black for middle cycles

    # Get legend_by_color setting
    legend_by_color = settings.get('legend_by_color', True)

    # For legend_by_color mode, track which colors have been added to legend
    color_cycles_map = {}  # color -> list of cycle numbers
    colors_added_to_legend = set()

    # First pass: collect color -> cycles mapping if legend_by_color is enabled
    if legend_by_color:
        for cyc_data in all_cycles_data:
            cycle_num = cyc_data['cycle_number']
            cycle_color = get_cycle_color(cycle_num, total_real_cycles)
            if cycle_color not in color_cycles_map:
                color_cycles_map[cycle_color] = []
            if cycle_num not in color_cycles_map[cycle_color]:
                color_cycles_map[cycle_color].append(cycle_num)

    # Add traces in time-sorted order
    for idx, cyc_data in enumerate(all_cycles_data):
        filename = cyc_data['filename']
        voltage = cyc_data['voltage']
        capacity = cyc_data['capacity']
        type_label = cyc_data['type_label']
        cycle_num = cyc_data['cycle_number']

        # Convert capacity (using active_mass_g for mAh/g)
        if capacity_unit == 'mAh/g':
            cap_display = capacity / active_mass_g
        else:
            cap_display = capacity / area_cm2  # mAh/cm²

        # Get color based on cycle number (D+C pair has same color)
        cycle_color = get_cycle_color(cycle_num, total_real_cycles)

        # Determine legend behavior
        if legend_by_color:
            # Group cycles by color in legend
            if cycle_color not in colors_added_to_legend:
                # First trace with this color - show in legend with grouped name
                cycles_with_color = color_cycles_map[cycle_color]
                if len(cycles_with_color) == 1:
                    cycle_label = f"Cycle {cycles_with_color[0]}"
                elif len(cycles_with_color) == 2:
                    cycle_label = f"Cycle {cycles_with_color[0]}, {cycles_with_color[-1]}"
                else:
                    cycle_label = f"Cycle {cycles_with_color[0]}-{cycles_with_color[-1]}"
                show_legend_for_trace = True
                colors_added_to_legend.add(cycle_color)
            else:
                # Subsequent traces with same color - hide from legend
                cycle_label = f"Cycle {cycle_num}"
                show_legend_for_trace = False
        else:
            # Normal mode: each cycle has its own legend entry
            cycle_label = f"Cycle {cycle_num}"
            if type_label:
                cycle_label += f" ({type_label})"
            show_legend_for_trace = True

        # Build hover label with cycle and C/D info
        cd_label = type_label if type_label else ''
        hover_cycle_info = f'Cycle {cycle_num}'
        if cd_label:
            hover_cycle_info += f' ({cd_label})'

        fig.add_trace(go.Scatter(
            x=cap_display,
            y=voltage,
            mode='lines',
            name=cycle_label,
            legendgroup=cycle_color if legend_by_color else None,
            showlegend=show_legend_for_trace,
            line=dict(width=line_width, color=cycle_color),
            hovertemplate=f'{hover_cycle_info}<br>Q = %{{x:.2f}}<br>V = %{{y:.3f}} V<extra></extra>'
        ))

    # Get figure size from settings
    fig_width = settings.get('fig_width', 700)
    fig_height = settings.get('fig_height', 500)

    # Build xaxis and yaxis settings
    xaxis_config = dict(
        title=x_label,
        title_font=dict(size=label_size, color='black'),
        tickfont=dict(size=tick_size, color='black'),
        tickcolor='black',
        showgrid=False,
        showline=True,
        linewidth=axis_width,
        linecolor='black',
        mirror=True,
        ticks='inside',
        ticklen=tick_length,
        tickwidth=tick_width,
    )
    yaxis_config = dict(
        title='Voltage / V',
        title_font=dict(size=label_size, color='black'),
        tickfont=dict(size=tick_size, color='black'),
        tickcolor='black',
        showgrid=False,
        showline=True,
        linewidth=axis_width,
        linecolor='black',
        mirror=True,
        ticks='inside',
        ticklen=tick_length,
        tickwidth=tick_width,
    )

    # Apply custom axis range
    if axis_range:
        if axis_range.get('x_min') is not None and axis_range.get('x_max') is not None:
            xaxis_config['range'] = [axis_range['x_min'], axis_range['x_max']]
        if axis_range.get('y_min') is not None and axis_range.get('y_max') is not None:
            yaxis_config['range'] = [axis_range['y_min'], axis_range['y_max']]

    # Check if we should show electron number on upper x-axis
    composition = sample_info.get('composition', '')
    can_show_electron = (
        show_electron_number and
        capacity_unit == 'mAh/g' and
        composition
    )

    # Prepare layout dict
    top_margin = 30
    xaxis2_config = None

    if can_show_electron:
        # Import helper functions
        from utils.helpers import format_formula_subscript
        from utils.theocapacity import calculate_molar_mass, FARADAY_CONSTANT

        try:
            molar_mass = calculate_molar_mass(composition)
            if molar_mass and molar_mass > 0:
                formatted_composition = format_formula_subscript(composition)

                # Upper x-axis title: number of reaction electron (n) in Composition
                upper_x_title = f'number of reaction electron (<i>n</i>) in {formatted_composition}'

                # Get x-axis range from data
                if fig.data:
                    all_x = []
                    all_y = []
                    for trace in fig.data:
                        if trace.x is not None and len(trace.x) > 0:
                            all_x.extend(trace.x)
                        if trace.y is not None and len(trace.y) > 0:
                            all_y.extend([v for v in trace.y if v is not None])

                    if all_x:
                        x_min, x_max = min(all_x), max(all_x)
                        # Convert capacity (mAh/g) to electron numbers
                        # n = capacity * M * 3.6 / F
                        # where M = molar mass (g/mol), F = Faraday constant (C/mol)
                        n_min = x_min * molar_mass * 3.6 / FARADAY_CONSTANT
                        n_max = x_max * molar_mass * 3.6 / FARADAY_CONSTANT

                        # Get y range for the invisible trace
                        y_mid = (min(all_y) + max(all_y)) / 2 if all_y else 3.5

                        # Add invisible trace for upper x-axis
                        fig.add_trace(go.Scatter(
                            x=[n_min, n_max],
                            y=[y_mid, y_mid],
                            mode='lines',
                            line=dict(width=0, color='rgba(0,0,0,0)'),
                            showlegend=False,
                            hoverinfo='skip',
                            xaxis='x2',
                            yaxis='y'
                        ))

                        xaxis2_config = dict(
                            overlaying='x',
                            side='top',
                            showgrid=False,
                            showline=True,
                            linewidth=axis_width,
                            linecolor='black',
                            tickcolor='black',
                            tickfont=dict(family='Arial', color='black', size=tick_size),
                            title=dict(
                                text=upper_x_title,
                                font=dict(family='Arial', color='black', size=label_size),
                                standoff=5  # Reduce gap between title and axis
                            ),
                            ticks='inside',
                            ticklen=tick_length,
                            tickwidth=tick_width,
                            mirror=False,
                            zeroline=False,
                            range=[n_min, n_max],
                        )
                        top_margin = 70  # Adjusted margin for upper x-axis title
        except Exception:
            pass  # If formula parsing fails, just skip the upper axis

    # Get legend position configuration
    legend_position = settings.get('legend_position', 'middle right')
    legend_pos_config = get_legend_position_config(legend_position)

    layout_dict = dict(
        font={'family': 'Arial', 'color': 'black'},
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=fig_width,
        height=fig_height,
        margin=dict(l=70, r=40, t=top_margin, b=60),
        xaxis=xaxis_config,
        yaxis=yaxis_config,
        showlegend=show_legend,
        legend=dict(
            **legend_pos_config,
            font=dict(size=settings.get('legend_font_size', 18)),
            bgcolor='rgba(255,255,255,0.8)'
        ),
    )

    if xaxis2_config:
        layout_dict['xaxis2'] = xaxis2_config

    fig.update_layout(**layout_dict)

    return fig


def create_multi_file_dqdv_plot(sorted_files: list, settings: dict, sample_info: dict) -> go.Figure:
    """Create dQ/dV plot for multiple files with time-based cycle pairing"""
    fig = go.Figure()

    # Calculate active mass considering active_ratio (same as single-file version)
    mass_mg = sample_info.get('mass_mg', 10.0)
    active_ratio = sample_info.get('active_ratio', 1.0)
    active_mass_g = mass_mg * active_ratio / 1000  # g
    if active_mass_g <= 0:
        active_mass_g = 0.001

    tick_size = settings.get('tick_font_size', 22)
    label_size = settings.get('axis_label_font_size', 22)
    line_width = settings.get('line_width', 1)

    # Collect all half-cycles from all files with their start times
    all_cycles_data = []
    for file_info in sorted_files:
        data = file_info['data']
        is_charge = file_info.get('is_charge', False)
        is_discharge = file_info.get('is_discharge', False)
        type_label = 'C' if is_charge else ('D' if is_discharge else '')

        if 'cycles' in data and len(data['cycles']) > 0:
            for i, cycle in enumerate(data['cycles']):
                if 'voltage' not in cycle or 'capacity' not in cycle:
                    continue
                start_time = cycle['time'][0] if 'time' in cycle and len(cycle['time']) > 0 else 0
                all_cycles_data.append({
                    'voltage': cycle['voltage'],
                    'capacity': cycle['capacity'],
                    'start_time': start_time,
                    'type_label': type_label,
                })
        elif 'capacity' in data and data['capacity'] is not None and 'voltage' in data:
            start_time = data['time'][0] if 'time' in data and len(data['time']) > 0 else 0
            all_cycles_data.append({
                'voltage': data['voltage'],
                'capacity': data['capacity'],
                'start_time': start_time,
                'type_label': type_label,
            })

    # Sort by start time
    all_cycles_data.sort(key=lambda x: x['start_time'])

    # Assign cycle numbers (D+C pair = 1 cycle)
    cycle_number = 1
    half_cycle_count = 0
    for cyc_data in all_cycles_data:
        cyc_data['cycle_number'] = cycle_number
        half_cycle_count += 1
        if half_cycle_count >= 2:
            cycle_number += 1
            half_cycle_count = 0

    # Calculate total real cycles correctly
    total_real_cycles = cycle_number - 1 if half_cycle_count == 0 else cycle_number
    if total_real_cycles < 1:
        total_real_cycles = 1

    # Get cycle colors from session state or use defaults
    cycle_colors_from_state = st.session_state.get('cycle_colors', {})

    def get_cycle_color(cycle_num: int, total: int) -> str:
        if cycle_num in cycle_colors_from_state:
            return cycle_colors_from_state[cycle_num]
        if total <= 1:
            return '#E63946'
        elif cycle_num == 1:
            return '#E63946'
        elif cycle_num == total:
            return '#009BFF'
        else:
            return '#000000'

    # Add traces
    for cyc_data in all_cycles_data:
        voltage = np.array(cyc_data['voltage'])
        capacity = np.array(cyc_data['capacity'])
        cycle_num = cyc_data['cycle_number']
        type_label = cyc_data['type_label']

        # Calculate dQ/dV
        if len(voltage) > 2:
            dv = np.diff(voltage)
            dq = np.diff(capacity)

            # Avoid division by zero
            dv[dv == 0] = 1e-10
            dqdv = dq / dv / active_mass_g

            v_mid = (voltage[:-1] + voltage[1:]) / 2

            cycle_color = get_cycle_color(cycle_num, total_real_cycles)
            label = f"Cycle {cycle_num}"
            if type_label:
                label += f" ({type_label})"

            fig.add_trace(go.Scatter(
                x=v_mid,
                y=dqdv,
                mode='lines',
                name=label,
                line=dict(width=line_width, color=cycle_color),
            ))

    axis_width = settings.get('axis_width', 1)
    tick_length = settings.get('tick_length', 6)
    tick_width = settings.get('tick_width', 1)

    fig.update_layout(
        font={'family': 'Arial', 'color': 'black'},
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=400,
        margin=dict(l=70, r=40, t=40, b=60),
        xaxis=dict(
            title='Voltage / V',
            title_font=dict(size=label_size, color='black'),
            tickfont=dict(size=tick_size, color='black'),
            showgrid=False,
            showline=True,
            linewidth=axis_width,
            linecolor='black',
            mirror=True,
            ticks='inside',
            ticklen=tick_length,
            tickwidth=tick_width,
            tickcolor='black',
            zeroline=False,
        ),
        yaxis=dict(
            title='dQ/dV / mAh g⁻¹ V⁻¹',
            title_font=dict(size=label_size, color='black'),
            tickfont=dict(size=tick_size, color='black'),
            showgrid=False,
            showline=True,
            linewidth=axis_width,
            linecolor='black',
            mirror=True,
            ticks='inside',
            ticklen=tick_length,
            tickwidth=tick_width,
            tickcolor='black',
            zeroline=False,
        ),
        showlegend=True,
        legend=dict(font=dict(size=10)),
    )

    return fig


def create_multi_file_cycle_performance(
    sorted_files: list, settings: dict, sample_info: dict,
    capacity_display: str = 'discharge', show_rate_labels: bool = False
) -> go.Figure:
    """
    Create cycle performance plot for multiple files with CE calculation.
    Uses time-based sorting and D+C pairing (same as CD plot).

    CE calculation:
    - If discharge display: CE = discharge_cap / charge_cap (same cycle)
    - If charge display: CE = charge_cap / discharge_cap (same cycle)
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Calculate active mass considering active_ratio (same as single-file version)
    mass_mg = sample_info.get('mass_mg', 10.0)
    active_ratio = sample_info.get('active_ratio', 1.0)
    active_mass_g = mass_mg * active_ratio / 1000  # g
    if active_mass_g <= 0:
        active_mass_g = 0.001

    tick_size = settings.get('tick_font_size', 22)
    label_size = settings.get('axis_label_font_size', 22)

    # Collect all half-cycles from all files with their start times
    all_half_cycles = []
    for file_info in sorted_files:
        data = file_info['data']
        filename = file_info['filename']
        is_charge = file_info.get('is_charge', False)
        is_discharge = file_info.get('is_discharge', False)
        current_mA = file_info.get('current_mA', 0)

        if 'cycles' in data and len(data['cycles']) > 0:
            for i, cycle in enumerate(data['cycles']):
                start_time = cycle['time'][0] if 'time' in cycle and len(cycle['time']) > 0 else 0
                # Calculate capacity for this half-cycle
                cap_raw = np.array(cycle.get('capacity', []))
                if len(cap_raw) > 0:
                    cap_abs = np.abs(cap_raw)
                    cap = (cap_abs[-1] - cap_abs[0]) / active_mass_g  # Normalized capacity
                else:
                    cap = 0
                all_half_cycles.append({
                    'start_time': start_time,
                    'type': 'C' if is_charge else ('D' if is_discharge else ''),
                    'capacity': cap,
                    'current_mA': current_mA,
                })
        elif 'capacity' in data and data['capacity'] is not None:
            start_time = data['time'][0] if 'time' in data and len(data['time']) > 0 else 0
            cap_raw = np.array(data['capacity'])
            if len(cap_raw) > 0:
                cap_abs = np.abs(cap_raw)
                cap = (cap_abs[-1] - cap_abs[0]) / active_mass_g
            else:
                cap = 0
            all_half_cycles.append({
                'start_time': start_time,
                'type': 'C' if is_charge else ('D' if is_discharge else ''),
                'capacity': cap,
                'current_mA': current_mA,
            })

    # Sort by start time
    all_half_cycles.sort(key=lambda x: x['start_time'])

    # Assign cycle numbers (D+C pair = 1 cycle)
    cycle_number = 1
    half_cycle_count = 0
    for hc in all_half_cycles:
        hc['cycle_number'] = cycle_number
        half_cycle_count += 1
        if half_cycle_count >= 2:
            cycle_number += 1
            half_cycle_count = 0

    # Group by cycle number
    cycle_data = {}  # {cycle_num: {'charge_cap': None, 'discharge_cap': None, 'current_mA': 0}}
    for hc in all_half_cycles:
        cn = hc['cycle_number']
        if cn not in cycle_data:
            cycle_data[cn] = {'charge_cap': None, 'discharge_cap': None, 'current_mA': 0}

        if hc['type'] == 'C':
            cycle_data[cn]['charge_cap'] = hc['capacity']
        elif hc['type'] == 'D':
            cycle_data[cn]['discharge_cap'] = hc['capacity']

        if hc['current_mA'] > cycle_data[cn]['current_mA']:
            cycle_data[cn]['current_mA'] = hc['current_mA']

    # Build plot arrays
    cycle_nums = sorted(cycle_data.keys())
    x_data = list(cycle_nums)  # Already 1-indexed from pairing

    if capacity_display == 'discharge':
        y_data = [cycle_data[c].get('discharge_cap', 0) or 0 for c in cycle_nums]
        y_label = 'Discharge Capacity / mAh g⁻¹'
    else:
        y_data = [cycle_data[c].get('charge_cap', 0) or 0 for c in cycle_nums]
        y_label = 'Charge Capacity / mAh g⁻¹'

    # Calculate CE
    ce_data = []
    for c in cycle_nums:
        charge_cap = cycle_data[c].get('charge_cap')
        discharge_cap = cycle_data[c].get('discharge_cap')

        if charge_cap and discharge_cap and charge_cap > 0:
            # CE = discharge / charge (same cycle pair)
            ce = (discharge_cap / charge_cap) * 100
        else:
            ce = 0
        ce_data.append(ce)

    # Get current values for rate labels
    current_data = [cycle_data[c].get('current_mA', 0) for c in cycle_nums]

    # Plot capacity (left axis, black)
    fig.add_trace(
        go.Scatter(
            x=x_data,
            y=y_data,
            mode='lines+markers',
            name='Capacity',
            line=dict(width=2, color='black'),
            marker=dict(size=8, color='black'),
        ),
        secondary_y=False
    )

    # Plot CE (right axis, blue)
    fig.add_trace(
        go.Scatter(
            x=x_data,
            y=ce_data,
            mode='lines+markers',
            name='CE',
            line=dict(width=2, color='#009BFF'),
            marker=dict(size=8, color='#009BFF'),
        ),
        secondary_y=True
    )

    # Add rate labels if enabled
    if show_rate_labels:
        annotations = []
        for i, (x, y, curr) in enumerate(zip(x_data, y_data, current_data)):
            if curr > 0:
                annotations.append(dict(
                    x=x, y=y,
                    text=f"{curr:.1f} mA",
                    showarrow=False,
                    yshift=15,
                    font=dict(size=10, color='gray')
                ))
        fig.update_layout(annotations=annotations)

    axis_width = settings.get('axis_width', 1)
    tick_length = settings.get('tick_length', 6)
    tick_width = settings.get('tick_width', 1)

    fig.update_layout(
        font={'family': 'Arial', 'color': 'black'},
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=400,
        margin=dict(l=70, r=70, t=40, b=60),
        showlegend=True,
        legend=dict(
            yanchor="top", y=0.99, xanchor="left", x=0.01,
            font=dict(size=10)
        ),
    )

    # Left Y-axis (Capacity, black)
    fig.update_yaxes(
        title_text=y_label,
        title_font=dict(size=label_size, color='black'),
        tickfont=dict(size=tick_size, color='black'),
        showgrid=False,
        showline=True,
        linewidth=axis_width,
        linecolor='black',
        ticks='inside',
        ticklen=tick_length,
        tickwidth=tick_width,
        tickcolor='black',
        zeroline=False,
        secondary_y=False
    )

    # Right Y-axis (CE, blue)
    fig.update_yaxes(
        title_text='Coulombic Efficiency / %',
        title_font=dict(size=label_size, color='#009BFF'),
        tickfont=dict(size=tick_size, color='#009BFF'),
        showgrid=False,
        showline=True,
        linewidth=axis_width,
        linecolor='#009BFF',
        ticks='inside',
        ticklen=tick_length,
        tickwidth=tick_width,
        tickcolor='#009BFF',
        range=[0, 105],
        zeroline=False,
        secondary_y=True
    )

    # X-axis (black)
    fig.update_xaxes(
        title_text='Cycle Number',
        title_font=dict(size=label_size, color='black'),
        tickfont=dict(size=tick_size, color='black'),
        showgrid=False,
        showline=True,
        linewidth=axis_width,
        linecolor='black',
        mirror=True,
        ticks='inside',
        ticklen=tick_length,
        tickwidth=tick_width,
        tickcolor='black',
        zeroline=False,
    )

    return fig


def build_cycle_history_data(sorted_files, data, sample_info):
    """
    Build cycle history data with numeric values for proper sorting.
    Returns list of dicts with cycle info and stores in session state.
    """
    mass_mg = sample_info.get('mass_mg', 10.0)
    active_ratio = sample_info.get('active_ratio', 1.0)
    active_mass_g = mass_mg * active_ratio / 1000  # g
    area_cm2 = sample_info.get('area_cm2', 1.0)

    cycle_history = []

    if sorted_files and len(sorted_files) > 0:
        # Multi-file mode
        all_half_cycles = []
        for file_info in sorted_files:
            fdata = file_info['data']
            is_charge = file_info.get('is_charge', False)
            is_discharge = file_info.get('is_discharge', False)
            current_mA = file_info.get('current_mA', 0)

            if 'cycles' in fdata and len(fdata['cycles']) > 0:
                for cycle in fdata['cycles']:
                    start_time = cycle['time'][0] if 'time' in cycle and len(cycle['time']) > 0 else 0
                    cap_raw = np.array(cycle.get('capacity', []))
                    if len(cap_raw) > 0:
                        cap_abs = np.abs(cap_raw)
                        cap_mAh = cap_abs[-1] - cap_abs[0]
                    else:
                        cap_mAh = 0
                    all_half_cycles.append({
                        'start_time': float(start_time),
                        'type': 'C' if is_charge else ('D' if is_discharge else ''),
                        'capacity_mAh': float(cap_mAh),
                        'current_mA': float(current_mA),
                    })
            elif 'capacity' in fdata and fdata['capacity'] is not None:
                start_time = fdata['time'][0] if 'time' in fdata and len(fdata['time']) > 0 else 0
                cap_raw = np.array(fdata['capacity'])
                if len(cap_raw) > 0:
                    cap_abs = np.abs(cap_raw)
                    cap_mAh = cap_abs[-1] - cap_abs[0]
                else:
                    cap_mAh = 0
                all_half_cycles.append({
                    'start_time': float(start_time),
                    'type': 'C' if is_charge else ('D' if is_discharge else ''),
                    'capacity_mAh': float(cap_mAh),
                    'current_mA': float(current_mA),
                })

        # Sort by start time
        all_half_cycles.sort(key=lambda x: x['start_time'])

        # Assign cycle numbers (D+C pair = 1 cycle)
        cycle_number = 1
        half_cycle_count = 0
        for hc in all_half_cycles:
            hc['cycle_number'] = cycle_number
            half_cycle_count += 1
            if half_cycle_count >= 2:
                cycle_number += 1
                half_cycle_count = 0

        # Build history with CE calculation
        prev_charge_cap = None
        for hc in all_half_cycles:
            cap_mAh = hc['capacity_mAh']
            cap_mAh_g = cap_mAh / active_mass_g if active_mass_g > 0 else 0
            cap_mAh_cm2 = cap_mAh / area_cm2 if area_cm2 > 0 else 0

            ce = None
            if hc['type'] == 'D' and prev_charge_cap is not None and prev_charge_cap > 0:
                ce = (cap_mAh / prev_charge_cap) * 100

            if hc['type'] == 'C':
                prev_charge_cap = cap_mAh

            cycle_history.append({
                'time_s': hc['start_time'],
                'cycle': hc['cycle_number'],
                'type': hc['type'],
                'current_mA': hc['current_mA'],
                'cap_mAh_g': cap_mAh_g,
                'cap_mAh_cm2': cap_mAh_cm2,
                'ce': ce,
            })

    elif data and 'cycles' in data and len(data['cycles']) > 0:
        # Single-file mode
        cycles = data['cycles']
        prev_charge_cap = None

        for i, cycle in enumerate(cycles):
            is_charge = cycle.get('is_charge', False)
            is_discharge = cycle.get('is_discharge', False)
            c_d = 'C' if is_charge else ('D' if is_discharge else '')

            cap_mAh = 0
            if 'capacity_mAh' in cycle:
                cap_mAh = cycle['capacity_mAh']
            elif 'capacity' in cycle:
                cap_raw = np.array(cycle['capacity'])
                if len(cap_raw) > 0:
                    cap_abs = np.abs(cap_raw)
                    cap_mAh = cap_abs[-1] - cap_abs[0]

            cap_mAh_g = cap_mAh / active_mass_g if active_mass_g > 0 else 0
            cap_mAh_cm2 = cap_mAh / area_cm2 if area_cm2 > 0 else 0

            curr = cycle.get('current_mA', 0)
            time_s = cycle['time'][0] if 'time' in cycle and len(cycle['time']) > 0 else 0

            ce = None
            if c_d == 'D' and prev_charge_cap is not None and prev_charge_cap > 0:
                ce = (cap_mAh / prev_charge_cap) * 100

            if c_d == 'C':
                prev_charge_cap = cap_mAh

            cycle_history.append({
                'time_s': float(time_s),
                'cycle': cycle.get('cycle_number', i + 1),
                'type': c_d,
                'current_mA': float(curr),
                'cap_mAh_g': float(cap_mAh_g),
                'cap_mAh_cm2': float(cap_mAh_cm2),
                'ce': ce,
            })

    # Store in session state for use by other components
    st.session_state.cycle_history = cycle_history
    return cycle_history


def render_data_summary(data, sample_info):
    """Render data summary metrics and cycle history table"""
    with st.expander("Data Summary", expanded=True):
        # Get sorted files for multi-file mode
        selected_files = st.session_state.selected_files
        sorted_files = None

        if len(selected_files) > 1:
            files_data = {fn: st.session_state.files[fn] for fn in selected_files if fn in st.session_state.files}
            sorted_files = sort_files_by_time_and_assign_cycles(files_data)

        # Build cycle history data (stores in session state)
        cycle_history = build_cycle_history_data(sorted_files, data, sample_info)

        # Display table with numeric values for proper sorting
        if cycle_history:
            st.markdown("**Cycle History**")

            # Create DataFrame with numeric columns
            df_data = []
            for row in cycle_history:
                # Format current with sign for display
                curr = row['current_mA']
                if row['type'] == 'D':
                    curr_display = f"-{abs(curr):.2f}" if curr != 0 else "0"
                else:
                    curr_display = f"+{abs(curr):.2f}" if curr != 0 else "0"

                df_data.append({
                    'Time (s)': round(row['time_s'], 1),  # Numeric for sorting
                    'Cycle': row['cycle'],
                    'C/D': row['type'],
                    'I (mA)': curr_display,
                    'Q (mAh/g)': round(row['cap_mAh_g'], 2),  # Numeric for sorting
                    'Q (mAh/cm²)': round(row['cap_mAh_cm2'], 4),  # Numeric for sorting
                    'CE (%)': round(row['ce'], 1) if row['ce'] is not None else None,  # Numeric for sorting
                })

            df_summary = pd.DataFrame(df_data)
            st.dataframe(df_summary, use_container_width=True, hide_index=True)
        else:
            # Fallback: show basic data info
            col1, col2 = st.columns(2)
            with col1:
                if 'time' in data and data['time'] is not None:
                    duration = data['time'][-1] - data['time'][0]
                    st.metric("Duration", f"{duration/3600:.2f} h")
            with col2:
                if 'voltage' in data and data['voltage'] is not None:
                    st.metric("Voltage Range", f"{data['voltage'].min():.3f} - {data['voltage'].max():.3f} V")


def render_export_section():
    """Render export options in sidebar"""
    if len(st.session_state.files) == 0:
        return

    st.markdown("### Export")

    sample_name = st.session_state.sample_info.get('name', 'cd_data')
    safe_name = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in sample_name) if sample_name else 'cd_data'
    date_str = datetime.now().strftime('%Y%m%d')

    col1, col2 = st.columns(2)

    with col1:
        if st.session_state.selected_file:
            data = st.session_state.files[st.session_state.selected_file]
            csv_data = generate_csv_export(data, st.session_state.sample_info)
            st.download_button(
                "CSV",
                data=csv_data,
                file_name=f"{safe_name}_{date_str}.csv",
                mime="text/csv",
                use_container_width=True
            )

    with col2:
        igor_str = generate_igor_file(
            st.session_state.files,
            st.session_state.sample_info
        )
        st.download_button(
            "Igor",
            data=igor_str,
            file_name=f"{safe_name}_{date_str}.itx",
            mime="text/plain",
            use_container_width=True,
            help="Igor Text File with publication-ready plots"
        )


def main():
    """Main application entry point"""
    initialize_session_state()
    inject_custom_css()

    # Sidebar
    with st.sidebar:
        sidebar_header()
        sidebar_file_upload()
        st.markdown("---")
        sidebar_sample_info()
        st.markdown("---")
        sidebar_file_manager()
        st.markdown("---")
        sidebar_view_mode()
        sidebar_cycle_selection()
        st.markdown("---")
        render_export_section()

    # Main content
    render_main_plot()


if __name__ == "__main__":
    main()
