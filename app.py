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
    load_biologic_mpt, parse_biologic_header, get_supported_formats,
    load_uploaded_file, validate_cd_data
)
from components.plots import (
    create_cd_plot, create_capacity_voltage_plot, create_dqdv_plot,
    create_cycle_summary_plot, COLORS
)
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
            'mass_mg': 10.0,  # Active material mass in mg
            'area_cm2': 0.636,  # Electrode area in cm² (default: φ9mm)
            'theoretical_capacity': 140.0,  # mAh/g
        }
    if 'plot_settings' not in st.session_state:
        st.session_state.plot_settings = {
            # Font settings
            'tick_font_size': 14,
            'axis_label_font_size': 16,
            # Marker/Line settings
            'line_width': 2,
            'marker_size': 0,  # 0 means no markers
            # Colors
            'charge_color': '#E63946',
            'discharge_color': '#457B9D',
            # Axis labels
            'voltage_label': 'Voltage / V',
            'time_label': 'Time / h',
            'capacity_label': 'Capacity / mAh g⁻¹',
            'dqdv_label': 'dQ/dV / mAh g⁻¹ V⁻¹',
            # Legend
            'show_legend': True,
            'legend_font_size': 12,
        }
    if 'view_mode' not in st.session_state:
        st.session_state.view_mode = 'V-t'  # 'V-t', 'V-Q', 'dQ/dV', 'Summary'
    if 'selected_cycles' not in st.session_state:
        st.session_state.selected_cycles = []
    if 'show_all_cycles' not in st.session_state:
        st.session_state.show_all_cycles = True


def inject_custom_css():
    """Inject custom CSS for better UI"""
    st.markdown("""
    <style>
    .sidebar-title {
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
        color: #1f77b4;
    }
    .stButton > button {
        width: 100%;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)


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


def sidebar_file_upload():
    """File upload section"""
    st.markdown("### Upload Files")

    supported_formats = get_supported_formats()
    uploaded_files = st.file_uploader(
        "Upload battery data files",
        type=supported_formats,
        accept_multiple_files=True,
        help="Supported formats: BioLogic (.mpt), CSV, TXT",
        label_visibility="collapsed"
    )

    if uploaded_files:
        process_uploaded_files(uploaded_files)


def sidebar_sample_info():
    """Sample information input section"""
    st.markdown("### Sample Information")

    st.session_state.sample_info['name'] = st.text_input(
        "Sample name",
        value=st.session_state.sample_info.get('name', ''),
        placeholder="Enter sample name"
    )

    col1, col2 = st.columns(2)
    with col1:
        mass = st.number_input(
            "Active mass (mg)",
            value=st.session_state.sample_info.get('mass_mg', 10.0),
            min_value=0.001,
            step=0.1,
            format="%.3f"
        )
        st.session_state.sample_info['mass_mg'] = mass

    with col2:
        area = st.number_input(
            "Area (cm²)",
            value=st.session_state.sample_info.get('area_cm2', 0.636),
            min_value=0.001,
            step=0.01,
            format="%.3f"
        )
        st.session_state.sample_info['area_cm2'] = area

    theo_cap = st.number_input(
        "Theoretical capacity (mAh/g)",
        value=st.session_state.sample_info.get('theoretical_capacity', 140.0),
        min_value=1.0,
        step=10.0
    )
    st.session_state.sample_info['theoretical_capacity'] = theo_cap


def sidebar_file_manager():
    """File management section"""
    if len(st.session_state.files) == 0:
        st.info("No files loaded")
        return

    st.markdown("### Loaded Files")

    for i, filename in enumerate(list(st.session_state.files.keys())):
        is_selected = (filename == st.session_state.selected_file)

        col1, col2 = st.columns([4, 1])

        with col1:
            btn_type = "primary" if is_selected else "secondary"
            if st.button(filename, key=f"select_{i}", type=btn_type, use_container_width=True):
                st.session_state.selected_file = filename
                st.rerun()

        with col2:
            if st.button("✕", key=f"delete_{i}", help="Delete file"):
                del st.session_state.files[filename]
                if st.session_state.selected_file == filename:
                    st.session_state.selected_file = None
                st.rerun()

    st.markdown("---")
    if st.button("Clear All", key="clear_all", use_container_width=True):
        st.session_state.files = {}
        st.session_state.selected_file = None
        st.rerun()


def sidebar_view_mode():
    """View mode selection"""
    st.markdown("### View Mode")

    view_options = {
        'V-t': '⏱️ Voltage vs Time',
        'V-Q': '⚡ Voltage vs Capacity',
        'dQ/dV': '📊 dQ/dV Analysis',
        'Summary': '📈 Cycle Summary'
    }

    selected = st.radio(
        "Select view",
        options=list(view_options.keys()),
        format_func=lambda x: view_options[x],
        index=list(view_options.keys()).index(st.session_state.view_mode),
        label_visibility="collapsed"
    )

    if selected != st.session_state.view_mode:
        st.session_state.view_mode = selected
        st.rerun()


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


def render_main_plot():
    """Render the main plot area"""
    if st.session_state.selected_file is None:
        st.info("Select a file from the sidebar to view data")
        return

    if st.session_state.selected_file not in st.session_state.files:
        st.warning("Selected file not found")
        return

    data = st.session_state.files[st.session_state.selected_file]
    settings = st.session_state.plot_settings
    sample_info = st.session_state.sample_info

    # Display file info
    st.markdown(f"### {st.session_state.selected_file}")

    view_mode = st.session_state.view_mode

    if view_mode == 'V-t':
        fig = create_cd_plot(data, settings, sample_info)
        st.plotly_chart(fig, use_container_width=True)

    elif view_mode == 'V-Q':
        fig = create_capacity_voltage_plot(data, settings, sample_info)
        st.plotly_chart(fig, use_container_width=True)

    elif view_mode == 'dQ/dV':
        fig = create_dqdv_plot(data, settings, sample_info)
        st.plotly_chart(fig, use_container_width=True)

    elif view_mode == 'Summary':
        fig = create_cycle_summary_plot(data, settings, sample_info)
        st.plotly_chart(fig, use_container_width=True)

    # Show data summary
    render_data_summary(data, sample_info)


def render_data_summary(data, sample_info):
    """Render data summary metrics"""
    with st.expander("Data Summary", expanded=True):
        if 'cycles' in data and len(data['cycles']) > 0:
            cycles = data['cycles']
            n_cycles = len(cycles)

            # Calculate metrics
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
                if 'coulombic_efficiency' in cycle and cycle['coulombic_efficiency'] is not None:
                    efficiencies.append(cycle['coulombic_efficiency'] * 100)

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Cycles", n_cycles)

            with col2:
                if capacities_charge:
                    st.metric("First Charge", f"{capacities_charge[0]:.1f} mAh/g")

            with col3:
                if capacities_discharge:
                    st.metric("First Discharge", f"{capacities_discharge[0]:.1f} mAh/g")

            with col4:
                if efficiencies:
                    st.metric("Avg. CE", f"{np.mean(efficiencies):.1f}%")

        else:
            # Raw data without cycle info
            if 'time' in data:
                duration = data['time'][-1] - data['time'][0]
                st.metric("Duration", f"{duration/3600:.2f} h")
            if 'voltage' in data:
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
        # CSV export
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
        # Igor export
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


def export_to_csv(data):
    """Export data to CSV format"""
    if 'time' in data and 'voltage' in data:
        df = pd.DataFrame({
            'Time (s)': data['time'],
            'Voltage (V)': data['voltage']
        })
        if 'current' in data:
            df['Current (mA)'] = data['current']
        if 'capacity' in data:
            df['Capacity (mAh)'] = data['capacity']

        return df.to_csv(index=False)
    return ""


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
        sidebar_plot_settings()
        st.markdown("---")
        render_export_section()

    # Main content
    render_main_plot()


if __name__ == "__main__":
    main()
