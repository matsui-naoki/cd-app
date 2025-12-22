"""
Sidebar Components for CD Analyzer
"""

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.colors as plt_colors

from tools.data_loader import get_supported_formats
from utils.file_processing import process_uploaded_files, load_mps_session


def sidebar_header():
    """Sidebar header with logo and title"""
    st.markdown('<p class="sidebar-title">CD Analyzer</p>', unsafe_allow_html=True)


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
        format_func=lambda x: 'Area (cm2)' if x == 'area' else 'Diameter (cm)',
        index=0 if st.session_state.sample_info.get('area_input_mode', 'area') == 'area' else 1,
        horizontal=True
    )
    st.session_state.sample_info['area_input_mode'] = area_mode

    if area_mode == 'area':
        area = st.number_input(
            "Electrode area (cm2)",
            value=st.session_state.sample_info.get('area_cm2', 1.0),
            min_value=0.001,
            step=0.01,
            format="%.3f",
            help="Default: 1.0 cm2"
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
        st.caption(f"Area: **{area:.4f} cm2**")

    # Calculate loading
    loading_mg_cm2 = active_mass_mg / area
    st.caption(f"Loading: **{loading_mg_cm2:.3f} mg/cm2**")

    # Theoretical capacity (optional, collapsed by default)
    with st.expander("Theoretical Capacity"):
        # Formula input first
        composition = st.text_input(
            "Formula",
            value=st.session_state.sample_info.get('composition', ''),
            placeholder="e.g., LiCoO2, LiFePO4",
            help="Q = nF / (M x 3.6) [mAh/g], where n = reaction electrons, F = 96485 C/mol, M = molar mass"
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
                if st.button("X", key=f"delete_{i}", help="Delete file"):
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


def render_data_list_panel(sorted_files: list):
    """Render data list panel below plot for per-cycle color settings"""
    # Get number of cycles
    n_cycles = max((f['cycle_num'] for f in sorted_files), default=0) + 1

    # Initialize cycle_colors with default colors if not present
    if not st.session_state.cycle_colors:
        for i in range(1, n_cycles + 1):
            if i == 1:
                st.session_state.cycle_colors[i] = '#E63946'  # Red
            elif i == n_cycles:
                st.session_state.cycle_colors[i] = '#457B9D'  # Blue
            else:
                st.session_state.cycle_colors[i] = '#000000'  # Black

    # Color schemes with various options
    COLOR_SCHEMES = {
        'Default (Red-Black-Blue)': lambda i, n: '#E63946' if i == 1 else ('#457B9D' if i == n else '#000000'),
        'Rainbow': lambda i, n: plt_colors.rgb2hex(plt.cm.rainbow((i - 1) / max(1, n - 1))) if n > 1 else '#FF0000',
        'Viridis': lambda i, n: plt_colors.rgb2hex(plt.cm.viridis((i - 1) / max(1, n - 1))) if n > 1 else '#440154',
        'Plasma': lambda i, n: plt_colors.rgb2hex(plt.cm.plasma((i - 1) / max(1, n - 1))) if n > 1 else '#0D0887',
        'Cool': lambda i, n: plt_colors.rgb2hex(plt.cm.cool((i - 1) / max(1, n - 1))) if n > 1 else '#00FFFF',
        'Grayscale': lambda i, n: f'#{int(255 * (1 - (i - 1) / max(1, n - 1))):02x}{int(255 * (1 - (i - 1) / max(1, n - 1))):02x}{int(255 * (1 - (i - 1) / max(1, n - 1))):02x}' if n > 1 else '#000000',
        'Custom': None,  # Use individual color pickers
    }

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

        # Show individual color pickers only for Custom mode
        if color_scheme == 'Custom':
            st.markdown("**Individual Cycle Colors**")
            # Display in rows of 5 cycles
            cycles_per_row = 5
            for row_start in range(1, n_cycles + 1, cycles_per_row):
                cols = st.columns(min(cycles_per_row, n_cycles - row_start + 1))
                for i, col in enumerate(cols):
                    cycle_num = row_start + i
                    if cycle_num <= n_cycles:
                        with col:
                            current_color = st.session_state.cycle_colors.get(
                                cycle_num,
                                '#E63946' if cycle_num == 1 else ('#457B9D' if cycle_num == n_cycles else '#000000')
                            )

                            new_color = st.color_picker(
                                f"Cycle {cycle_num}",
                                value=current_color,
                                key=f"cycle_color_{cycle_num}",
                            )
                            if new_color != current_color:
                                st.session_state.cycle_colors[cycle_num] = new_color
                                st.rerun()
        else:
            # Show preview of the color scheme in rows of 10
            if n_cycles <= 30:
                cycles_per_row = 10
                for row_start in range(0, n_cycles, cycles_per_row):
                    row_end = min(row_start + cycles_per_row, n_cycles)
                    preview_cols = st.columns(row_end - row_start)
                    for i, col in enumerate(preview_cols):
                        cycle_num = row_start + i + 1
                        color = st.session_state.cycle_colors.get(cycle_num, COLOR_SCHEMES[color_scheme](cycle_num, n_cycles) if COLOR_SCHEMES[color_scheme] else '#000000')
                        col.markdown(f'<div style="background-color:{color};width:100%;height:20px;border-radius:3px;" title="Cycle {cycle_num}"></div>', unsafe_allow_html=True)

        # Add caption below color palette to prevent expander cutoff
        st.caption(f"Cycles 1-{n_cycles}")
