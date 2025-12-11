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
    create_cycle_summary_plot, create_capacity_retention_plot,
    create_multi_file_vq_plot, get_publication_config, COLORS,
    create_bode_plot, create_cumulative_capacity_plot, apply_axis_range
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
    page_icon="🔋",
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
        st.session_state.view_mode = 'V-t'
    if 'selected_cycles' not in st.session_state:
        st.session_state.selected_cycles = []
    if 'show_all_cycles' not in st.session_state:
        st.session_state.show_all_cycles = True
    if 'color_mode' not in st.session_state:
        st.session_state.color_mode = 'cycle'  # 'cycle' or 'charge_discharge'
    if 'selected_files' not in st.session_state:
        st.session_state.selected_files = []  # For multi-file V-Q plot
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
    st.markdown('<div class="sidebar-title">🔋 CD Analyzer</div>', unsafe_allow_html=True)


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

    # Multi-select mode for V-Q view
    view_mode = st.session_state.view_mode
    is_vq_view = view_mode == 'V-Q'

    if is_vq_view and len(st.session_state.files) > 1:
        st.caption("Select multiple files for comparison")
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
                    st.rerun()

    st.markdown("---")
    if st.button("Clear All", key="clear_all", use_container_width=True):
        st.session_state.files = {}
        st.session_state.selected_file = None
        st.session_state.selected_files = []
        st.session_state.mps_session = None
        st.session_state.eis_data = []
        st.rerun()


def sidebar_view_mode():
    """View mode selection"""
    st.markdown("### View Mode")

    view_options = {
        'V-t': '⏱️ Voltage vs Time',
        'V-Q': '⚡ Voltage vs Capacity',
        'dQ/dV': '📊 dQ/dV Analysis',
        'CV/LSV': '📈 CV / LSV',
        'Summary': '📈 Cycle Summary',
        'Retention': '📉 Capacity Retention',
        'Cumulative': '📊 Cumulative Capacity',
        'Nyquist': '🔬 Nyquist Plot',
        'Bode': '📈 Bode Plot',
        'DataFrame': '📋 Data Table',
    }

    # Add Session view if MPS is loaded
    if st.session_state.mps_session:
        view_options['Session'] = '📋 Session Info'

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
    if st.session_state.view_mode not in ['V-t', 'V-Q', 'dQ/dV']:
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

    # Color mode selection (only for V-Q view)
    if st.session_state.view_mode == 'V-Q':
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

    # Handle Bode plot view
    if view_mode == 'Bode':
        render_bode_plot()
        return

    # Handle DataFrame view
    if view_mode == 'DataFrame':
        render_dataframe_view()
        return

    # Handle CV/LSV view
    if view_mode == 'CV/LSV':
        render_cv_lsv_plot()
        return

    # Standard CD views
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

    if view_mode == 'V-t':
        fig = create_cd_plot(data, settings, sample_info, selected_cycles)
        st.plotly_chart(fig, use_container_width=True, config=plot_config)

    elif view_mode == 'V-Q':
        # Multi-file support for V-Q view
        selected_files = st.session_state.selected_files
        if len(selected_files) > 1:
            # Multi-file comparison mode
            files_data = {fn: st.session_state.files[fn] for fn in selected_files if fn in st.session_state.files}
            fig = create_multi_file_vq_plot(files_data, settings, sample_info, selected_cycles, color_mode)
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
            key='capacity_unit_vq'
        )
        if capacity_unit != sample_info.get('capacity_unit'):
            st.session_state.sample_info['capacity_unit'] = capacity_unit
            st.rerun()

    elif view_mode == 'dQ/dV':
        fig = create_dqdv_plot(data, settings, sample_info, selected_cycles)
        st.plotly_chart(fig, use_container_width=True, config=plot_config)

    elif view_mode == 'Summary':
        fig = create_cycle_summary_plot(data, settings, sample_info)
        st.plotly_chart(fig, use_container_width=True, config=plot_config)

        # Capacity unit selection below the plot
        st.markdown("---")
        capacity_unit = st.radio(
            "Capacity unit",
            options=['mAh/g', 'mAh/cm²'],
            index=0 if sample_info.get('capacity_unit', 'mAh/g') == 'mAh/g' else 1,
            horizontal=True,
            key='capacity_unit_summary'
        )
        if capacity_unit != sample_info.get('capacity_unit'):
            st.session_state.sample_info['capacity_unit'] = capacity_unit
            st.rerun()

    elif view_mode == 'Retention':
        fig = create_capacity_retention_plot(data, settings, sample_info)
        st.plotly_chart(fig, use_container_width=True, config=plot_config)

        # Capacity unit selection below the plot
        st.markdown("---")
        capacity_unit = st.radio(
            "Capacity unit",
            options=['mAh/g', 'mAh/cm²'],
            index=0 if sample_info.get('capacity_unit', 'mAh/g') == 'mAh/g' else 1,
            horizontal=True,
            key='capacity_unit_retention'
        )
        if capacity_unit != sample_info.get('capacity_unit'):
            st.session_state.sample_info['capacity_unit'] = capacity_unit
            st.rerun()

    elif view_mode == 'Cumulative':
        fig = create_cumulative_capacity_plot(data, settings, sample_info, capacity_type='both')
        st.plotly_chart(fig, use_container_width=True, config=plot_config)

        # Capacity unit selection below the plot
        st.markdown("---")
        capacity_unit = st.radio(
            "Capacity unit",
            options=['mAh/g', 'mAh/cm²'],
            index=0 if sample_info.get('capacity_unit', 'mAh/g') == 'mAh/g' else 1,
            horizontal=True,
            key='capacity_unit_cumulative'
        )
        if capacity_unit != sample_info.get('capacity_unit'):
            st.session_state.sample_info['capacity_unit'] = capacity_unit
            st.rerun()

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
            st.markdown(f"✅ **{tech.index}. {tech.short_name}**: `{os.path.basename(tech.data_file)}`")
        else:
            st.markdown(f"⬜ **{tech.index}. {tech.short_name}**: No data file")


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
    eis_list = st.session_state.eis_data
    settings = st.session_state.plot_settings

    st.markdown("### EIS Data (Nyquist Plot)")

    if not eis_list:
        st.info("No EIS data available")
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
    eis_list = st.session_state.eis_data
    settings = st.session_state.plot_settings

    st.markdown("### EIS Data (Bode Plot)")

    if not eis_list:
        st.info("No EIS data available")
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
