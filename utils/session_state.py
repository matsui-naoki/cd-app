"""
Session State Management for CD Analyzer
"""

import streamlit as st


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
            'area_cm2': 1.0,  # Electrode area (cm2) - default 1.0
            'diameter_cm': 1.0,  # Electrode diameter (cm) - default 1.0
            'area_input_mode': 'area',  # 'area' or 'diameter'
            'theoretical_capacity': 140.0,  # mAh/g
            'capacity_unit': 'mAh/g',  # 'mAh/g' or 'mAh/cm2'
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
            'capacity_label': 'Capacity / mAh g-1',
            'dqdv_label': 'dQ/dV / mAh g-1 V-1',
            'show_legend': True,
            'legend_font_size': 12,
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
    # Cycle colors for multi-file mode
    if 'cycle_colors' not in st.session_state:
        st.session_state.cycle_colors = {}
    # Color scheme selection
    if 'color_scheme' not in st.session_state:
        st.session_state.color_scheme = 'Default (Red-Black-Blue)'
    # Cycle history data for table and cycle performance
    if 'cycle_history' not in st.session_state:
        st.session_state.cycle_history = []
