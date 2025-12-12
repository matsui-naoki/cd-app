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
            'legend_font_size': 12,
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


def sidebar_plot_settings():
    """Plot settings section"""
    with st.expander("Plot Settings", expanded=False):
        st.session_state.plot_settings['line_width'] = st.slider(
            "Line width", 1, 5, st.session_state.plot_settings.get('line_width', 2)
        )

        st.session_state.plot_settings['tick_font_size'] = st.slider(
            "Tick font size", 8, 24, st.session_state.plot_settings.get('tick_font_size', 14)
        )

        st.session_state.plot_settings['axis_label_font_size'] = st.slider(
            "Axis label size", 10, 28, st.session_state.plot_settings.get('axis_label_font_size', 16)
        )

        col1, col2 = st.columns(2)
        with col1:
            st.session_state.plot_settings['charge_color'] = st.color_picker(
                "Charge color",
                value=st.session_state.plot_settings.get('charge_color', '#E63946')
            )
        with col2:
            st.session_state.plot_settings['discharge_color'] = st.color_picker(
                "Discharge color",
                value=st.session_state.plot_settings.get('discharge_color', '#457B9D')
            )

        st.session_state.plot_settings['show_legend'] = st.checkbox(
            "Show legend",
            value=st.session_state.plot_settings.get('show_legend', True)
        )

    # Axis Range Controls (inspired by Igor IPF)
    with st.expander("Axis Range", expanded=False):
        st.session_state.axis_range['enabled'] = st.checkbox(
            "Enable custom axis range",
            value=st.session_state.axis_range.get('enabled', False)
        )

        if st.session_state.axis_range['enabled']:
            st.caption("Y-axis (Voltage)")
            col1, col2 = st.columns(2)
            with col1:
                y_min = st.number_input(
                    "Y min",
                    value=st.session_state.axis_range.get('y_min') or 0.0,
                    step=0.1,
                    format="%.2f",
                    key="y_min_input"
                )
                st.session_state.axis_range['y_min'] = y_min if y_min != 0.0 or st.session_state.axis_range.get('y_min') else None
            with col2:
                y_max = st.number_input(
                    "Y max",
                    value=st.session_state.axis_range.get('y_max') or 5.0,
                    step=0.1,
                    format="%.2f",
                    key="y_max_input"
                )
                st.session_state.axis_range['y_max'] = y_max if y_max != 5.0 or st.session_state.axis_range.get('y_max') else None

            st.caption("X-axis (Time/Capacity)")
            col3, col4 = st.columns(2)
            with col3:
                x_min = st.number_input(
                    "X min",
                    value=st.session_state.axis_range.get('x_min') or 0.0,
                    step=1.0,
                    format="%.1f",
                    key="x_min_input"
                )
                st.session_state.axis_range['x_min'] = x_min if x_min != 0.0 or st.session_state.axis_range.get('x_min') else None
            with col4:
                x_max = st.number_input(
                    "X max",
                    value=st.session_state.axis_range.get('x_max') or 100.0,
                    step=1.0,
                    format="%.1f",
                    key="x_max_input"
                )
                st.session_state.axis_range['x_max'] = x_max if x_max != 100.0 or st.session_state.axis_range.get('x_max') else None

            if st.button("Reset to Auto", key="reset_axis"):
                st.session_state.axis_range = {
                    'x_min': None, 'x_max': None,
                    'y_min': None, 'y_max': None,
                    'enabled': False
                }
                st.rerun()

    # Graph Offset Controls (inspired by Igor IPF)
    with st.expander("Trace Offset", expanded=False):
        st.caption("Apply offset between traces for better comparison")
        st.session_state.graph_offset['enabled'] = st.checkbox(
            "Enable trace offset",
            value=st.session_state.graph_offset.get('enabled', False)
        )

        if st.session_state.graph_offset['enabled']:
            col1, col2 = st.columns(2)
            with col1:
                st.session_state.graph_offset['x_offset'] = st.number_input(
                    "X offset",
                    value=st.session_state.graph_offset.get('x_offset', 0.0),
                    step=1.0,
                    format="%.1f",
                    key="x_offset_input",
                    help="Shift traces along X-axis"
                )
            with col2:
                st.session_state.graph_offset['y_offset'] = st.number_input(
                    "Y offset",
                    value=st.session_state.graph_offset.get('y_offset', 0.0),
                    step=0.1,
                    format="%.2f",
                    key="y_offset_input",
                    help="Shift traces along Y-axis"
                )


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
    """Render data list panel below plot for per-file color settings"""
    st.markdown("---")
    st.markdown("##### Data Files")

    for i, file_info in enumerate(sorted_files):
        filename = file_info['filename']
        cycle_num = file_info['cycle_num']
        is_charge = file_info['is_charge']
        is_discharge = file_info['is_discharge']
        current_mA = file_info['current_mA']

        # Type label
        if is_charge:
            type_label = "Charge"
        elif is_discharge:
            type_label = "Discharge"
        else:
            type_label = "Unknown"

        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])

        with col1:
            st.text(f"{filename}")

        with col2:
            st.text(f"Cycle {cycle_num + 1}")

        with col3:
            st.text(f"{type_label}")

        with col4:
            # Color picker
            new_color = st.color_picker(
                "Color",
                value=file_info['color'],
                key=f"color_{filename}_{i}",
                label_visibility="collapsed"
            )
            if new_color != file_info['color']:
                st.session_state.file_colors[filename] = new_color
                st.rerun()


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

        if len(selected_files) > 1:
            # Multi-file mode: sort by time and assign cycles
            files_data = {fn: st.session_state.files[fn] for fn in selected_files if fn in st.session_state.files}
            sorted_files = sort_files_by_time_and_assign_cycles(files_data)

            # Create multi-file plot with sorted files and custom colors
            fig = create_multi_file_cd_plot(sorted_files, settings, sample_info)
        else:
            fig = create_capacity_voltage_plot(data, settings, sample_info, selected_cycles, color_mode)

        st.plotly_chart(fig, use_container_width=True, config=plot_config)

        # Capacity unit selection below the plot
        st.markdown("---")
        capacity_unit = st.radio(
            "Capacity unit",
            options=['mAh/g', 'mAh/cm²'],
            index=0 if sample_info.get('capacity_unit', 'mAh/g') == 'mAh/g' else 1,
            horizontal=True,
            key='capacity_unit_cd'
        )
        if capacity_unit != sample_info.get('capacity_unit'):
            st.session_state.sample_info['capacity_unit'] = capacity_unit
            st.rerun()

        # Data list panel for multi-file mode
        if sorted_files and len(sorted_files) > 1:
            render_data_list_panel(sorted_files)

        # Sub-panels: dQ/dV curve and Cycle Performance
        col1, col2 = st.columns(2)

        with col1:
            with st.expander("dQ/dV curve", expanded=False):
                if sorted_files and len(sorted_files) > 1:
                    fig_dqdv = create_multi_file_dqdv_plot(sorted_files, settings, sample_info)
                else:
                    fig_dqdv = create_dqdv_plot(data, settings, sample_info, selected_cycles)
                st.plotly_chart(fig_dqdv, use_container_width=True, config=plot_config)

        with col2:
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


def create_multi_file_cd_plot(sorted_files: list, settings: dict, sample_info: dict) -> go.Figure:
    """Create Charge-Discharge plot for multiple files with custom colors"""
    fig = go.Figure()

    mass_g = sample_info.get('mass_mg', 1.0) / 1000
    if mass_g <= 0:
        mass_g = 0.001
    area_cm2 = sample_info.get('area_cm2', 1.0)
    capacity_unit = sample_info.get('capacity_unit', 'mAh/g')

    tick_size = settings.get('tick_font_size', 22)
    label_size = settings.get('axis_label_font_size', 22)
    line_width = settings.get('line_width', 1)

    x_label = 'Capacity / mAh g⁻¹' if capacity_unit == 'mAh/g' else 'Capacity / mAh cm⁻²'

    for file_info in sorted_files:
        data = file_info['data']
        color = file_info['color']
        filename = file_info['filename']
        file_cycle_num = file_info['cycle_num']  # Cycle number assigned to this file
        is_charge = file_info.get('is_charge', False)
        is_discharge = file_info.get('is_discharge', False)

        # Determine type label
        type_label = 'C' if is_charge else ('D' if is_discharge else '')

        # Get capacity and voltage from cycles or raw data
        if 'cycles' in data and len(data['cycles']) > 0:
            for i, cycle in enumerate(data['cycles']):
                if 'voltage' not in cycle or 'capacity' not in cycle:
                    continue
                voltage = cycle['voltage']
                capacity = np.abs(cycle['capacity'])  # Use absolute value

                # Convert capacity
                if capacity_unit == 'mAh/g':
                    cap_display = capacity / mass_g
                else:
                    cap_display = capacity * mass_g * 1000 / area_cm2

                # Create trace name with file cycle number and internal cycle number
                cycle_label = f"Cycle {file_cycle_num + 1}"
                if len(data['cycles']) > 1:
                    cycle_label += f"-{i + 1}"
                if type_label:
                    cycle_label += f" ({type_label})"

                fig.add_trace(go.Scatter(
                    x=cap_display,
                    y=voltage,
                    mode='lines',
                    name=cycle_label,
                    line=dict(width=line_width, color=color),
                    hovertemplate=f'{filename}<br>Q = %{{x:.2f}}<br>V = %{{y:.3f}} V<extra></extra>'
                ))
        elif 'capacity' in data and data['capacity'] is not None and 'voltage' in data:
            capacity = np.abs(data['capacity'])  # Use absolute value
            voltage = data['voltage']

            if capacity_unit == 'mAh/g':
                cap_display = capacity / mass_g
            else:
                cap_display = capacity * mass_g * 1000 / area_cm2

            # Create trace name
            cycle_label = f"Cycle {file_cycle_num + 1}"
            if type_label:
                cycle_label += f" ({type_label})"

            fig.add_trace(go.Scatter(
                x=cap_display,
                y=voltage,
                mode='lines',
                name=cycle_label,
                line=dict(width=line_width, color=color),
                hovertemplate=f'{filename}<br>Q = %{{x:.2f}}<br>V = %{{y:.3f}} V<extra></extra>'
            ))

    fig.update_layout(
        font={'family': 'Arial', 'color': 'black'},
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=500,
        xaxis=dict(
            title=x_label,
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
            title='Voltage / V',
            title_font=dict(size=label_size),
            tickfont=dict(size=tick_size),
            showgrid=False,
            showline=True,
            linewidth=1,
            linecolor='black',
            mirror=True,
            ticks='inside',
        ),
        showlegend=True,
        legend=dict(
            yanchor="top", y=0.99, xanchor="right", x=0.99,
            font=dict(size=settings.get('legend_font_size', 12)),
            bgcolor='rgba(255,255,255,0.8)'
        ),
    )

    return fig


def create_multi_file_dqdv_plot(sorted_files: list, settings: dict, sample_info: dict) -> go.Figure:
    """Create dQ/dV plot for multiple files"""
    fig = go.Figure()

    mass_g = sample_info.get('mass_mg', 1.0) / 1000
    if mass_g <= 0:
        mass_g = 0.001

    tick_size = settings.get('tick_font_size', 22)
    label_size = settings.get('axis_label_font_size', 22)
    line_width = settings.get('line_width', 1)

    for file_info in sorted_files:
        data = file_info['data']
        color = file_info['color']
        cycle_num = file_info['cycle_num']

        if 'cycles' in data and len(data['cycles']) > 0:
            for cycle in data['cycles']:
                if 'voltage' not in cycle or 'capacity' not in cycle:
                    continue
                voltage = cycle['voltage']
                capacity = cycle['capacity']

                # Calculate dQ/dV
                if len(voltage) > 2:
                    dv = np.diff(voltage)
                    dq = np.diff(capacity)

                    # Avoid division by zero
                    dv[dv == 0] = 1e-10
                    dqdv = dq / dv / mass_g

                    v_mid = (voltage[:-1] + voltage[1:]) / 2

                    fig.add_trace(go.Scatter(
                        x=v_mid,
                        y=dqdv,
                        mode='lines',
                        name=f"Cycle {cycle_num + 1}",
                        line=dict(width=line_width, color=color),
                    ))

    fig.update_layout(
        font={'family': 'Arial', 'color': 'black'},
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=400,
        xaxis=dict(
            title='Voltage / V',
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
            title='dQ/dV / mAh g⁻¹ V⁻¹',
            title_font=dict(size=label_size),
            tickfont=dict(size=tick_size),
            showgrid=False,
            showline=True,
            linewidth=1,
            linecolor='black',
            mirror=True,
            ticks='inside',
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

    CE calculation:
    - If discharge display: CE = discharge_cap / charge_cap_before_discharge
    - If charge display: CE = charge_cap / discharge_cap_before_charge
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    mass_g = sample_info.get('mass_mg', 1.0) / 1000
    if mass_g <= 0:
        mass_g = 0.001

    tick_size = settings.get('tick_font_size', 22)
    label_size = settings.get('axis_label_font_size', 22)

    # Collect capacity data per cycle
    cycle_data = {}  # {cycle_num: {'charge_cap': [], 'discharge_cap': [], 'current_mA': []}}

    for file_info in sorted_files:
        data = file_info['data']
        cycle_num = file_info['cycle_num']
        is_charge = file_info['is_charge']
        is_discharge = file_info['is_discharge']
        current_mA = file_info['current_mA']

        if cycle_num not in cycle_data:
            cycle_data[cycle_num] = {'charge_cap': None, 'discharge_cap': None, 'current_mA': current_mA}

        # Get capacity from cycles
        cap = 0
        if 'cycles' in data and len(data['cycles']) > 0:
            for cycle in data['cycles']:
                if 'capacity_mAh' in cycle:
                    cap = cycle['capacity_mAh'] / mass_g
                elif 'capacity' in cycle and len(cycle['capacity']) > 0:
                    cap = (np.max(np.abs(cycle['capacity'])) - np.min(np.abs(cycle['capacity']))) / mass_g

        if is_charge:
            cycle_data[cycle_num]['charge_cap'] = cap
        elif is_discharge:
            cycle_data[cycle_num]['discharge_cap'] = cap

        # Update current if higher
        if current_mA > cycle_data[cycle_num].get('current_mA', 0):
            cycle_data[cycle_num]['current_mA'] = current_mA

    # Build plot arrays
    cycle_nums = sorted(cycle_data.keys())
    x_data = [c + 1 for c in cycle_nums]  # 1-indexed

    if capacity_display == 'discharge':
        y_data = [cycle_data[c].get('discharge_cap', 0) or 0 for c in cycle_nums]
        y_label = 'Discharge Capacity / mAh g⁻¹'
    else:
        y_data = [cycle_data[c].get('charge_cap', 0) or 0 for c in cycle_nums]
        y_label = 'Charge Capacity / mAh g⁻¹'

    # Calculate CE
    ce_data = []
    for i, c in enumerate(cycle_nums):
        charge_cap = cycle_data[c].get('charge_cap')
        discharge_cap = cycle_data[c].get('discharge_cap')

        if capacity_display == 'discharge' and charge_cap and discharge_cap:
            # CE = discharge / charge (same cycle)
            ce = (discharge_cap / charge_cap) * 100 if charge_cap > 0 else 0
        elif capacity_display == 'charge' and i > 0:
            # CE = charge / previous discharge
            prev_discharge = cycle_data[cycle_nums[i-1]].get('discharge_cap')
            if prev_discharge and charge_cap:
                ce = (charge_cap / prev_discharge) * 100 if prev_discharge > 0 else 0
            else:
                ce = 0
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
            line=dict(width=2, color='#457B9D'),
            marker=dict(size=8, color='#457B9D'),
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

    fig.update_layout(
        font={'family': 'Arial', 'color': 'black'},
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=400,
        showlegend=True,
        legend=dict(
            yanchor="top", y=0.99, xanchor="left", x=0.01,
            font=dict(size=10)
        ),
    )

    # Left Y-axis (Capacity)
    fig.update_yaxes(
        title_text=y_label,
        title_font=dict(size=label_size, color='black'),
        tickfont=dict(size=tick_size, color='black'),
        showgrid=False,
        showline=True,
        linewidth=1,
        linecolor='black',
        ticks='inside',
        secondary_y=False
    )

    # Right Y-axis (CE)
    fig.update_yaxes(
        title_text='Coulombic Efficiency / %',
        title_font=dict(size=label_size, color='#457B9D'),
        tickfont=dict(size=tick_size, color='#457B9D'),
        showgrid=False,
        showline=True,
        linewidth=1,
        linecolor='#457B9D',
        ticks='inside',
        range=[0, 105],
        secondary_y=True
    )

    # X-axis
    fig.update_xaxes(
        title_text='Cycle Number',
        title_font=dict(size=label_size),
        tickfont=dict(size=tick_size),
        showgrid=False,
        showline=True,
        linewidth=1,
        linecolor='black',
        mirror=True,
        ticks='inside',
    )

    return fig


def render_data_summary(data, sample_info):
    """Render data summary metrics"""
    with st.expander("Data Summary", expanded=True):
        if 'cycles' in data and len(data['cycles']) > 0:
            cycles = data['cycles']
            n_cycles = len(cycles)

            mass_g = sample_info['mass_mg'] / 1000

            capacities_charge = []
            capacities_discharge = []
            efficiencies = []

            for cycle in cycles:
                if 'capacity_charge_mAh' in cycle and cycle['capacity_charge_mAh'] is not None:
                    cap_charge = cycle['capacity_charge_mAh'] / mass_g if mass_g > 0 else 0
                    capacities_charge.append(cap_charge)
                if 'capacity_discharge_mAh' in cycle and cycle['capacity_discharge_mAh'] is not None:
                    cap_discharge = cycle['capacity_discharge_mAh'] / mass_g if mass_g > 0 else 0
                    capacities_discharge.append(cap_discharge)
                # Also check capacity_mAh from half_cycle parsing
                if 'capacity_mAh' in cycle and cycle.get('is_discharge'):
                    cap = cycle['capacity_mAh'] / mass_g if mass_g > 0 else cycle['capacity_mAh']
                    capacities_discharge.append(cap)
                elif 'capacity_mAh' in cycle and cycle.get('is_charge'):
                    cap = cycle['capacity_mAh'] / mass_g if mass_g > 0 else cycle['capacity_mAh']
                    capacities_charge.append(cap)

                if 'coulombic_efficiency' in cycle and cycle['coulombic_efficiency'] is not None:
                    efficiencies.append(cycle['coulombic_efficiency'] * 100)

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Cycles", n_cycles)

            with col2:
                if capacities_charge:
                    st.metric("First Charge", f"{capacities_charge[0]:.2f} mAh/g")
                elif capacities_discharge:
                    st.metric("First Cap", f"{capacities_discharge[0]:.4f} mAh/g")

            with col3:
                if capacities_discharge:
                    st.metric("First Discharge", f"{capacities_discharge[0]:.4f} mAh/g")

            with col4:
                if efficiencies:
                    st.metric("Avg. CE", f"{np.mean(efficiencies):.1f}%")

        else:
            # Raw data without cycle info
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
        sidebar_plot_settings()
        st.markdown("---")
        render_export_section()

    # Main content
    render_main_plot()


if __name__ == "__main__":
    main()
