"""
Plotly-based plotting functions for battery charge-discharge data visualization
Publication-ready figures for academic papers and presentations
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, List, Dict, Tuple, Any

# Color palette optimized for publications
COLORS = [
    '#000000',  # Black
    '#E63946',  # Red
    '#457B9D',  # Blue
    '#2A9D8F',  # Teal
    '#F4A261',  # Orange
    '#9B5DE5',  # Purple
    '#00B4D8',  # Cyan
    '#6B705C',  # Gray-green
    '#D62828',  # Dark red
    '#1D3557',  # Dark blue
]

# Rainbow color palette for multi-cycle display
RAINBOW_COLORS = [
    '#000000', '#FF0000', '#FF6600', '#FFCC00', '#66FF00',
    '#00FF66', '#00FFFF', '#0066FF', '#6600FF', '#FF00FF'
]


def get_cycle_color(
    cycle_idx: int,
    total_cycles: int,
    color_mode: str,
    is_charge: bool = None,
    is_discharge: bool = None,
    settings: dict = None
) -> Tuple[str, str]:
    """
    Get color and name suffix for a cycle based on color mode

    Parameters
    ----------
    cycle_idx : int
        Index of current cycle (0-based)
    total_cycles : int
        Total number of cycles
    color_mode : str
        Color mode: 'cycle', 'charge_discharge', 'first_last', 'grayscale', 'single_black'
    is_charge : bool
        Whether this is a charge cycle
    is_discharge : bool
        Whether this is a discharge cycle
    settings : dict
        Plot settings (for custom colors)

    Returns
    -------
    color : str
        Hex color code
    name_suffix : str
        Suffix for trace name (e.g., 'Chg', 'Dchg', '')
    """
    if settings is None:
        settings = {}

    charge_color = settings.get('charge_color', '#E63946')
    discharge_color = settings.get('discharge_color', '#457B9D')

    if color_mode == 'charge_discharge':
        if is_charge:
            return charge_color, 'Chg'
        elif is_discharge:
            return discharge_color, 'Dchg'
        else:
            return '#888888', ''

    elif color_mode == 'first_last':
        # 1st cycle = red, middle cycles = black, last cycle = blue
        if cycle_idx == 0:
            return '#E63946', '1st'  # Red for first
        elif cycle_idx == total_cycles - 1:
            return '#457B9D', 'last'  # Blue for last
        else:
            return '#333333', ''  # Dark gray/black for middle

    elif color_mode == 'grayscale':
        # Gradient from black to light gray
        if total_cycles > 1:
            gray_val = int(50 + (cycle_idx / (total_cycles - 1)) * 150)  # 50-200 range
        else:
            gray_val = 50
        return f'#{gray_val:02x}{gray_val:02x}{gray_val:02x}', ''

    elif color_mode == 'single_black':
        return '#000000', ''

    else:  # 'cycle' - rainbow by cycle number
        color_idx = cycle_idx % len(RAINBOW_COLORS)
        return RAINBOW_COLORS[color_idx], ''


def common_layout(settings: dict = None) -> dict:
    """Common layout settings for publication-ready plots"""
    if settings is None:
        settings = {}

    return {
        'font': {'family': 'Arial', 'color': 'black'},
        'plot_bgcolor': 'white',
        'paper_bgcolor': 'white',
        'margin': {'l': 80, 'r': 20, 't': 40, 'b': 60},
    }


def common_axis_settings(settings: dict = None) -> dict:
    """Common axis settings for publication-ready plots"""
    if settings is None:
        settings = {}

    tick_size = settings.get('tick_font_size', 14)
    label_size = settings.get('axis_label_font_size', 16)

    return {
        'showgrid': False,
        'showline': True,
        'linewidth': 1.5,
        'linecolor': 'black',
        'tickcolor': 'black',
        'tickfont': {'family': 'Arial', 'color': 'black', 'size': tick_size},
        'title_font': {'family': 'Arial', 'color': 'black', 'size': label_size},
        'mirror': True,
        'ticks': 'inside',
        'ticklen': 6,
        'title_standoff': 10,
        'zeroline': False,
    }


def create_cd_plot(
    data: Dict,
    settings: dict = None,
    sample_info: dict = None,
    selected_cycles: List[int] = None,
    show_current: bool = True
) -> go.Figure:
    """
    Create Voltage vs Time plot

    Parameters
    ----------
    data : dict
        Battery data dictionary with 'time', 'voltage', 'current'
    settings : dict
        Plot settings
    sample_info : dict
        Sample information (mass, area, etc.)
    selected_cycles : list
        List of cycle numbers to display (None = all)
    show_current : bool
        Whether to show current on secondary y-axis

    Returns
    -------
    fig : go.Figure
        Plotly figure
    """
    if settings is None:
        settings = {}
    if sample_info is None:
        sample_info = {}

    # Create figure with secondary y-axis for current
    fig = make_subplots(specs=[[{"secondary_y": show_current and 'current' in data}]])

    # Get data arrays
    time = data.get('time', np.array([]))
    voltage = data.get('voltage', np.array([]))
    current = data.get('current', np.array([]))

    if len(time) == 0 or len(voltage) == 0:
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color='black')
        )
        return fig

    # Convert time to hours
    time_h = time / 3600

    # Plot settings
    line_width = settings.get('line_width', 2)
    charge_color = settings.get('charge_color', '#E63946')
    discharge_color = settings.get('discharge_color', '#457B9D')

    # Check if we have cycle information
    cycles = data.get('cycles', [])

    if cycles and len(cycles) > 0:
        # Plot by cycle with different colors
        for i, cycle in enumerate(cycles):
            if selected_cycles is not None and cycle['cycle_number'] not in selected_cycles:
                continue

            if 'time' in cycle and 'voltage' in cycle:
                t = cycle['time'] / 3600
                v = cycle['voltage']

                # Color by cycle number
                color = COLORS[i % len(COLORS)]

                fig.add_trace(go.Scatter(
                    x=t,
                    y=v,
                    mode='lines',
                    name=f'Cycle {cycle["cycle_number"] + 1}',
                    line=dict(width=line_width, color=color),
                    hovertemplate='<i>t</i> = %{x:.2f} h<br><i>V</i> = %{y:.3f} V<extra></extra>'
                ))

    else:
        # Single trace - color by current direction if available
        if len(current) > 0:
            # Split by current sign for coloring
            positive_mask = current >= 0
            negative_mask = current < 0

            # Charge (positive current)
            if np.any(positive_mask):
                fig.add_trace(go.Scatter(
                    x=time_h[positive_mask],
                    y=voltage[positive_mask],
                    mode='lines',
                    name='Charge',
                    line=dict(width=line_width, color=charge_color),
                    hovertemplate='<i>t</i> = %{x:.2f} h<br><i>V</i> = %{y:.3f} V<extra></extra>'
                ))

            # Discharge (negative current)
            if np.any(negative_mask):
                fig.add_trace(go.Scatter(
                    x=time_h[negative_mask],
                    y=voltage[negative_mask],
                    mode='lines',
                    name='Discharge',
                    line=dict(width=line_width, color=discharge_color),
                    hovertemplate='<i>t</i> = %{x:.2f} h<br><i>V</i> = %{y:.3f} V<extra></extra>'
                ))
        else:
            # No current info - single line
            fig.add_trace(go.Scatter(
                x=time_h,
                y=voltage,
                mode='lines',
                name='Voltage',
                line=dict(width=line_width, color='black'),
                hovertemplate='<i>t</i> = %{x:.2f} h<br><i>V</i> = %{y:.3f} V<extra></extra>'
            ))

    # Add current trace on secondary axis
    if show_current and len(current) > 0:
        area = sample_info.get('area_cm2', 1.0)
        current_density = current / area

        fig.add_trace(go.Scatter(
            x=time_h,
            y=current_density,
            mode='lines',
            name='Current',
            line=dict(width=1, color='gray', dash='dot'),
            hovertemplate='<i>t</i> = %{x:.2f} h<br><i>i</i> = %{y:.2f} mA/cm²<extra></extra>',
            opacity=0.7
        ), secondary_y=True)

    # Update layout
    axis_settings = common_axis_settings(settings)
    label_size = settings.get('axis_label_font_size', 16)
    label_font = dict(family='Arial', color='black', size=label_size)
    legend_font_size = settings.get('legend_font_size', 12)
    show_legend = settings.get('show_legend', True)

    voltage_label = settings.get('voltage_label', 'Voltage / V')
    time_label = settings.get('time_label', 'Time / h')

    fig.update_layout(
        **common_layout(settings),
        height=450,
        uirevision='cd_plot',
        xaxis=dict(
            **axis_settings,
            title=dict(text=time_label, font=label_font),
        ),
        yaxis=dict(
            **axis_settings,
            title=dict(text=voltage_label, font=label_font),
        ),
        showlegend=show_legend,
        legend=dict(
            yanchor="top", y=0.99, xanchor="right", x=0.99,
            font=dict(size=legend_font_size), bgcolor='rgba(255,255,255,0.8)'
        )
    )

    # Secondary y-axis for current
    if show_current and len(current) > 0:
        fig.update_yaxes(
            title=dict(text='Current density / mA cm⁻²', font=label_font),
            **{k: v for k, v in axis_settings.items() if k not in ['title']},
            secondary_y=True
        )

    return fig


def get_capacity_normalization(sample_info: dict, capacity_mah: np.ndarray) -> Tuple[np.ndarray, str]:
    """
    Normalize capacity based on sample_info settings

    Parameters
    ----------
    sample_info : dict
        Sample information with 'capacity_unit', 'mass_mg', 'active_ratio', 'area_cm2'
    capacity_mah : np.ndarray
        Capacity in mAh

    Returns
    -------
    normalized_capacity : np.ndarray
        Capacity normalized according to settings
    unit_label : str
        Unit label for axis
    """
    capacity_unit = sample_info.get('capacity_unit', 'mAh/g')
    mass_mg = sample_info.get('mass_mg', 10.0)
    active_ratio = sample_info.get('active_ratio', 1.0)
    area_cm2 = sample_info.get('area_cm2', 0.636)

    active_mass_g = mass_mg * active_ratio / 1000  # g

    if capacity_unit == 'mAh/g':
        if active_mass_g > 0:
            return capacity_mah / active_mass_g, 'mAh g⁻¹'
        else:
            return capacity_mah, 'mAh'
    elif capacity_unit == 'mAh/cm²':
        if area_cm2 > 0:
            return capacity_mah / area_cm2, 'mAh cm⁻²'
        else:
            return capacity_mah, 'mAh'
    else:
        return capacity_mah, 'mAh'


def create_capacity_voltage_plot(
    data: Dict,
    settings: dict = None,
    sample_info: dict = None,
    selected_cycles: List[int] = None,
    color_mode: str = 'cycle'
) -> go.Figure:
    """
    Create Voltage vs Capacity plot

    Parameters
    ----------
    data : dict
        Battery data dictionary
    settings : dict
        Plot settings
    sample_info : dict
        Sample information (with capacity_unit, mass_mg, active_ratio, area_cm2)
    selected_cycles : list
        List of cycle numbers to display (None = all)
    color_mode : str
        'cycle': color by cycle number (rainbow)
        'charge_discharge': red for charge, blue for discharge

    Returns
    -------
    fig : go.Figure
        Plotly figure
    """
    if settings is None:
        settings = {}
    if sample_info is None:
        sample_info = {}

    fig = go.Figure()

    cycles = data.get('cycles', [])
    line_width = settings.get('line_width', 2)
    charge_color = settings.get('charge_color', '#E63946')
    discharge_color = settings.get('discharge_color', '#457B9D')

    if cycles and len(cycles) > 0:
        # Count total cycles for rainbow coloring
        cycle_numbers = sorted(set(c.get('cycle_number', 0) for c in cycles))
        n_cycles = len(cycle_numbers)

        for i, cycle in enumerate(cycles):
            cycle_num = cycle.get('cycle_number', 0)

            if selected_cycles is not None and cycle_num not in selected_cycles:
                continue

            # Skip cycles with zero current (relaxation process)
            if 'current' in cycle:
                avg_current = np.mean(np.abs(cycle['current']))
                if avg_current < 0.01:  # Skip relaxation process (low current < 0.01 mA)
                    continue

            # Get data
            voltage = cycle.get('voltage', np.array([]))
            if len(voltage) == 0:
                continue

            # Calculate or get capacity
            if 'capacity' in cycle:
                capacity_mah = np.abs(cycle['capacity'])
            elif 'current' in cycle and 'time' in cycle:
                time = cycle['time']
                current = cycle['current']
                dt = np.diff(time)
                capacity_mah = np.zeros(len(time))
                capacity_mah[1:] = np.cumsum(np.abs(current[:-1]) * dt) / 3600
            else:
                continue

            # Normalize capacity
            capacity_norm, cap_unit = get_capacity_normalization(sample_info, capacity_mah)

            # Determine color using helper function
            is_charge = cycle.get('is_charge', False)
            is_discharge = cycle.get('is_discharge', False)
            cycle_idx = cycle_numbers.index(cycle_num) if cycle_num in cycle_numbers else 0

            color, name_suffix = get_cycle_color(
                cycle_idx=cycle_idx,
                total_cycles=n_cycles,
                color_mode=color_mode,
                is_charge=is_charge,
                is_discharge=is_discharge,
                settings=settings
            )

            # Create trace name
            half_cycle = cycle.get('half_cycle', i)
            if name_suffix:
                trace_name = f'Cyc{cycle_num + 1} {name_suffix}'
            else:
                trace_name = f'Cycle {cycle_num + 1}'
                if is_charge:
                    trace_name += ' (C)'
                elif is_discharge:
                    trace_name += ' (D)'

            fig.add_trace(go.Scatter(
                x=capacity_norm,
                y=voltage,
                mode='lines',
                name=trace_name,
                line=dict(width=line_width, color=color),
                hovertemplate=f'{trace_name}<br>' + '<i>Q</i> = %{x:.1f}<br><i>V</i> = %{y:.3f} V<extra></extra>'
            ))

    else:
        # Use raw data (no cycle information)
        if 'capacity' in data and 'voltage' in data:
            capacity_mah = np.abs(data['capacity'])
        elif 'current' in data and 'time' in data:
            time = data['time']
            current = data['current']
            dt = np.diff(time)
            capacity_mah = np.zeros(len(time))
            capacity_mah[1:] = np.cumsum(np.abs(current[:-1]) * dt) / 3600
        else:
            capacity_mah = np.array([])

        if len(capacity_mah) > 0 and 'voltage' in data:
            capacity_norm, cap_unit = get_capacity_normalization(sample_info, capacity_mah)

            fig.add_trace(go.Scatter(
                x=capacity_norm,
                y=data['voltage'],
                mode='lines',
                name='Voltage',
                line=dict(width=line_width, color='black'),
                hovertemplate='<i>Q</i> = %{x:.1f}<br><i>V</i> = %{y:.3f} V<extra></extra>'
            ))
            cap_unit = cap_unit  # Use the determined unit
        else:
            cap_unit = 'mAh g⁻¹'

    # Update layout
    axis_settings = common_axis_settings(settings)
    label_size = settings.get('axis_label_font_size', 16)
    label_font = dict(family='Arial', color='black', size=label_size)
    legend_font_size = settings.get('legend_font_size', 12)
    show_legend = settings.get('show_legend', True)

    # Get capacity unit from sample_info
    capacity_unit = sample_info.get('capacity_unit', 'mAh/g')
    if capacity_unit == 'mAh/g':
        cap_label = 'Capacity / mAh g⁻¹'
    else:
        cap_label = 'Capacity / mAh cm⁻²'

    voltage_label = settings.get('voltage_label', 'Voltage / V')

    fig.update_layout(
        **common_layout(settings),
        height=450,
        uirevision='vq_plot',
        xaxis=dict(
            **axis_settings,
            title=dict(text=cap_label, font=label_font),
        ),
        yaxis=dict(
            **axis_settings,
            title=dict(text=voltage_label, font=label_font),
        ),
        showlegend=show_legend,
        legend=dict(
            yanchor="top", y=0.99, xanchor="right", x=0.99,
            font=dict(size=legend_font_size), bgcolor='rgba(255,255,255,0.8)'
        )
    )

    return fig


def create_dqdv_plot(
    data: Dict,
    settings: dict = None,
    sample_info: dict = None,
    selected_cycles: List[int] = None,
    smooth_window: int = 5
) -> go.Figure:
    """
    Create dQ/dV vs Voltage plot

    Parameters
    ----------
    data : dict
        Battery data dictionary
    settings : dict
        Plot settings
    sample_info : dict
        Sample information (with capacity_unit, mass_mg, active_ratio, area_cm2)
    selected_cycles : list
        List of cycle numbers to display
    smooth_window : int
        Window size for smoothing dQ/dV

    Returns
    -------
    fig : go.Figure
        Plotly figure
    """
    if settings is None:
        settings = {}
    if sample_info is None:
        sample_info = {}

    fig = go.Figure()

    cycles = data.get('cycles', [])
    line_width = settings.get('line_width', 2)
    charge_color = settings.get('charge_color', '#E63946')
    discharge_color = settings.get('discharge_color', '#457B9D')

    # Get normalization parameters
    capacity_unit = sample_info.get('capacity_unit', 'mAh/g')
    mass_mg = sample_info.get('mass_mg', 10.0)
    active_ratio = sample_info.get('active_ratio', 1.0)
    area_cm2 = sample_info.get('area_cm2', 0.636)
    active_mass_g = mass_mg * active_ratio / 1000  # g

    def calculate_dqdv(voltage, current, time, smooth=True):
        """Calculate dQ/dV from current and time data"""
        # Calculate capacity
        dt = np.diff(time)
        dq = np.abs(current[:-1]) * dt / 3600  # mAh

        # Calculate dV
        dv = np.diff(voltage)

        # Calculate dQ/dV, avoiding division by very small values
        dqdv = np.zeros_like(dv)
        valid = np.abs(dv) > 1e-6
        dqdv[valid] = dq[valid] / dv[valid]

        # Normalize based on capacity unit
        if capacity_unit == 'mAh/g':
            if active_mass_g > 0:
                dqdv = dqdv / active_mass_g
        elif capacity_unit == 'mAh/cm²':
            if area_cm2 > 0:
                dqdv = dqdv / area_cm2

        # Smooth
        if smooth and smooth_window > 1 and len(dqdv) > smooth_window:
            kernel = np.ones(smooth_window) / smooth_window
            dqdv = np.convolve(dqdv, kernel, mode='same')

        # Return mid-point voltages
        v_mid = (voltage[:-1] + voltage[1:]) / 2

        return v_mid, dqdv

    if 'dqdv' in data and 'voltage' in data:
        # Use pre-calculated dQ/dV
        dqdv = data['dqdv'].copy()
        voltage = data['voltage']

        # Filter out zeros and very large values
        valid = (np.abs(dqdv) > 0) & (np.abs(dqdv) < 1e10)

        # Normalize based on capacity unit
        if capacity_unit == 'mAh/g':
            if active_mass_g > 0:
                dqdv = dqdv / active_mass_g
        elif capacity_unit == 'mAh/cm²':
            if area_cm2 > 0:
                dqdv = dqdv / area_cm2

        fig.add_trace(go.Scatter(
            x=voltage[valid],
            y=dqdv[valid],
            mode='lines',
            name='dQ/dV',
            line=dict(width=line_width, color='black'),
            hovertemplate='<i>V</i> = %{x:.3f} V<br>dQ/dV = %{y:.1f}<extra></extra>'
        ))

    elif cycles and len(cycles) > 0:
        for i, cycle in enumerate(cycles):
            if selected_cycles is not None and cycle['cycle_number'] not in selected_cycles:
                continue

            if 'current' in cycle and 'time' in cycle and 'voltage' in cycle:
                current = cycle['current']
                time = cycle['time']
                voltage = cycle['voltage']

                if len(voltage) < 3:
                    continue

                # Split by charge/discharge
                charge_mask = current > 0
                discharge_mask = current < 0

                color = COLORS[i % len(COLORS)]

                # Charge dQ/dV
                if np.sum(charge_mask) > 3:
                    idx = np.where(charge_mask)[0]
                    v_c, dqdv_c = calculate_dqdv(
                        voltage[idx], current[idx], time[idx], smooth=True
                    )
                    fig.add_trace(go.Scatter(
                        x=v_c,
                        y=dqdv_c,
                        mode='lines',
                        name=f'Cycle {cycle["cycle_number"] + 1} (charge)',
                        line=dict(width=line_width, color=color),
                        hovertemplate='<i>V</i> = %{x:.3f} V<br>dQ/dV = %{y:.1f}<extra></extra>'
                    ))

                # Discharge dQ/dV
                if np.sum(discharge_mask) > 3:
                    idx = np.where(discharge_mask)[0]
                    v_d, dqdv_d = calculate_dqdv(
                        voltage[idx], current[idx], time[idx], smooth=True
                    )
                    fig.add_trace(go.Scatter(
                        x=v_d,
                        y=-dqdv_d,  # Invert for discharge
                        mode='lines',
                        name=f'Cycle {cycle["cycle_number"] + 1} (discharge)',
                        line=dict(width=line_width, color=color, dash='dash'),
                        hovertemplate='<i>V</i> = %{x:.3f} V<br>dQ/dV = %{y:.1f}<extra></extra>'
                    ))

    else:
        # Use raw data
        if 'current' in data and 'time' in data and 'voltage' in data:
            current = data['current']
            time = data['time']
            voltage = data['voltage']

            if len(voltage) > 3:
                v_mid, dqdv = calculate_dqdv(voltage, current, time, smooth=True)

                fig.add_trace(go.Scatter(
                    x=v_mid,
                    y=dqdv,
                    mode='lines',
                    name='dQ/dV',
                    line=dict(width=line_width, color='black'),
                    hovertemplate='<i>V</i> = %{x:.3f} V<br>dQ/dV = %{y:.1f}<extra></extra>'
                ))

    # Update layout
    axis_settings = common_axis_settings(settings)
    label_size = settings.get('axis_label_font_size', 16)
    label_font = dict(family='Arial', color='black', size=label_size)
    legend_font_size = settings.get('legend_font_size', 12)
    show_legend = settings.get('show_legend', True)

    voltage_label = settings.get('voltage_label', 'Voltage / V')

    # Set dQ/dV label based on capacity unit
    if capacity_unit == 'mAh/g':
        dqdv_label = 'dQ/dV / mAh g⁻¹ V⁻¹'
    else:
        dqdv_label = 'dQ/dV / mAh cm⁻² V⁻¹'

    fig.update_layout(
        **common_layout(settings),
        height=450,
        uirevision='dqdv_plot',
        xaxis=dict(
            **axis_settings,
            title=dict(text=voltage_label, font=label_font),
        ),
        yaxis=dict(
            **axis_settings,
            title=dict(text=dqdv_label, font=label_font),
        ),
        showlegend=show_legend,
        legend=dict(
            yanchor="top", y=0.99, xanchor="right", x=0.99,
            font=dict(size=legend_font_size), bgcolor='rgba(255,255,255,0.8)'
        )
    )

    return fig


def create_cycle_summary_plot(
    data: Dict,
    settings: dict = None,
    sample_info: dict = None
) -> go.Figure:
    """
    Create cycle summary plot showing capacity and efficiency vs cycle number

    Parameters
    ----------
    data : dict
        Battery data dictionary
    settings : dict
        Plot settings
    sample_info : dict
        Sample information (with capacity_unit, mass_mg, active_ratio, area_cm2)

    Returns
    -------
    fig : go.Figure
        Plotly figure with capacity and efficiency
    """
    if settings is None:
        settings = {}
    if sample_info is None:
        sample_info = {}

    # Create figure with secondary y-axis for efficiency
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    cycles = data.get('cycles', [])

    if not cycles or len(cycles) == 0:
        fig.add_annotation(
            text="No cycle data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color='black')
        )
        return fig

    # Get normalization parameters
    capacity_unit = sample_info.get('capacity_unit', 'mAh/g')
    mass_mg = sample_info.get('mass_mg', 10.0)
    active_ratio = sample_info.get('active_ratio', 1.0)
    area_cm2 = sample_info.get('area_cm2', 0.636)
    active_mass_g = mass_mg * active_ratio / 1000  # g

    cycle_numbers = []
    charge_capacities = []
    discharge_capacities = []
    efficiencies = []

    for cycle in cycles:
        cn = cycle.get('cycle_number', 0) + 1
        cycle_numbers.append(cn)

        # Get capacities
        cap_charge = cycle.get('capacity_charge_mAh', None)
        cap_discharge = cycle.get('capacity_discharge_mAh', None)
        ce = cycle.get('coulombic_efficiency', None)

        # Normalize based on capacity unit
        if cap_charge is not None:
            if capacity_unit == 'mAh/g' and active_mass_g > 0:
                charge_capacities.append(cap_charge / active_mass_g)
            elif capacity_unit == 'mAh/cm²' and area_cm2 > 0:
                charge_capacities.append(cap_charge / area_cm2)
            else:
                charge_capacities.append(cap_charge)
        else:
            charge_capacities.append(None)

        if cap_discharge is not None:
            if capacity_unit == 'mAh/g' and active_mass_g > 0:
                discharge_capacities.append(cap_discharge / active_mass_g)
            elif capacity_unit == 'mAh/cm²' and area_cm2 > 0:
                discharge_capacities.append(cap_discharge / area_cm2)
            else:
                discharge_capacities.append(cap_discharge)
        else:
            discharge_capacities.append(None)

        if ce is not None:
            efficiencies.append(ce * 100)
        else:
            efficiencies.append(None)

    # Plot settings
    marker_size = settings.get('marker_size', 8) if settings.get('marker_size', 0) > 0 else 8
    line_width = settings.get('line_width', 2)
    charge_color = settings.get('charge_color', '#E63946')
    discharge_color = settings.get('discharge_color', '#457B9D')

    # Charge capacity
    if any(c is not None for c in charge_capacities):
        fig.add_trace(go.Scatter(
            x=cycle_numbers,
            y=charge_capacities,
            mode='markers+lines',
            name='Charge',
            marker=dict(size=marker_size, color=charge_color, symbol='circle'),
            line=dict(width=line_width, color=charge_color),
            hovertemplate='Cycle %{x}<br>Charge: %{y:.1f} mAh/g<extra></extra>'
        ))

    # Discharge capacity
    if any(c is not None for c in discharge_capacities):
        fig.add_trace(go.Scatter(
            x=cycle_numbers,
            y=discharge_capacities,
            mode='markers+lines',
            name='Discharge',
            marker=dict(size=marker_size, color=discharge_color, symbol='square'),
            line=dict(width=line_width, color=discharge_color),
            hovertemplate='Cycle %{x}<br>Discharge: %{y:.1f} mAh/g<extra></extra>'
        ))

    # Coulombic efficiency on secondary axis
    if any(e is not None for e in efficiencies):
        fig.add_trace(go.Scatter(
            x=cycle_numbers,
            y=efficiencies,
            mode='markers+lines',
            name='CE',
            marker=dict(size=marker_size, color='gray', symbol='diamond'),
            line=dict(width=1, color='gray', dash='dot'),
            hovertemplate='Cycle %{x}<br>CE: %{y:.1f}%<extra></extra>'
        ), secondary_y=True)

    # Update layout
    axis_settings = common_axis_settings(settings)
    label_size = settings.get('axis_label_font_size', 16)
    label_font = dict(family='Arial', color='black', size=label_size)
    legend_font_size = settings.get('legend_font_size', 12)
    show_legend = settings.get('show_legend', True)

    # Set capacity label based on capacity unit
    if capacity_unit == 'mAh/g':
        cap_label = 'Capacity / mAh g⁻¹'
    else:
        cap_label = 'Capacity / mAh cm⁻²'

    fig.update_layout(
        **common_layout(settings),
        height=450,
        uirevision='summary_plot',
        xaxis=dict(
            **axis_settings,
            title=dict(text='Cycle number', font=label_font),
        ),
        yaxis=dict(
            **axis_settings,
            title=dict(text=cap_label, font=label_font),
        ),
        showlegend=show_legend,
        legend=dict(
            yanchor="top", y=0.99, xanchor="right", x=0.99,
            font=dict(size=legend_font_size), bgcolor='rgba(255,255,255,0.8)'
        )
    )

    # Secondary y-axis for efficiency
    fig.update_yaxes(
        title=dict(text='Coulombic efficiency / %', font=label_font),
        range=[90, 105],  # Typical CE range
        **{k: v for k, v in axis_settings.items() if k not in ['title']},
        secondary_y=True
    )

    return fig


def create_capacity_retention_plot(
    data: Dict,
    settings: dict = None,
    sample_info: dict = None
) -> go.Figure:
    """
    Create capacity retention plot showing capacity retention % vs cycle number

    Parameters
    ----------
    data : dict
        Battery data dictionary
    settings : dict
        Plot settings
    sample_info : dict
        Sample information (with capacity_unit, mass_mg, active_ratio, area_cm2)

    Returns
    -------
    fig : go.Figure
        Plotly figure with capacity retention
    """
    if settings is None:
        settings = {}
    if sample_info is None:
        sample_info = {}

    fig = go.Figure()

    cycles = data.get('cycles', [])

    if not cycles or len(cycles) == 0:
        fig.add_annotation(
            text="No cycle data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color='black')
        )
        return fig

    # Get normalization parameters
    capacity_unit = sample_info.get('capacity_unit', 'mAh/g')
    mass_mg = sample_info.get('mass_mg', 10.0)
    active_ratio = sample_info.get('active_ratio', 1.0)
    area_cm2 = sample_info.get('area_cm2', 0.636)
    active_mass_g = mass_mg * active_ratio / 1000  # g

    cycle_numbers = []
    discharge_capacities = []
    retentions = []

    # Collect discharge capacities
    for cycle in cycles:
        cn = cycle.get('cycle_number', 0) + 1

        cap_discharge = cycle.get('capacity_discharge_mAh', None)
        if cap_discharge is None and 'capacity_mAh' in cycle and cycle.get('is_discharge'):
            cap_discharge = cycle.get('capacity_mAh')

        if cap_discharge is not None:
            # Normalize based on capacity unit
            if capacity_unit == 'mAh/g' and active_mass_g > 0:
                cycle_numbers.append(cn)
                discharge_capacities.append(cap_discharge / active_mass_g)
            elif capacity_unit == 'mAh/cm²' and area_cm2 > 0:
                cycle_numbers.append(cn)
                discharge_capacities.append(cap_discharge / area_cm2)
            elif cap_discharge is not None:
                cycle_numbers.append(cn)
                discharge_capacities.append(cap_discharge)

    if len(discharge_capacities) == 0:
        fig.add_annotation(
            text="No discharge capacity data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color='black')
        )
        return fig

    # Calculate retention (reference = first cycle)
    ref_cap = discharge_capacities[0]
    if ref_cap > 0:
        retentions = [cap / ref_cap * 100 for cap in discharge_capacities]
    else:
        retentions = [0] * len(discharge_capacities)

    # Plot settings
    marker_size = settings.get('marker_size', 8) if settings.get('marker_size', 0) > 0 else 8
    line_width = settings.get('line_width', 2)
    discharge_color = settings.get('discharge_color', '#457B9D')

    fig.add_trace(go.Scatter(
        x=cycle_numbers,
        y=retentions,
        mode='markers+lines',
        name='Retention',
        marker=dict(size=marker_size, color=discharge_color, symbol='circle'),
        line=dict(width=line_width, color=discharge_color),
        hovertemplate='Cycle %{x}<br>Retention: %{y:.1f}%<extra></extra>'
    ))

    # Update layout
    axis_settings = common_axis_settings(settings)
    label_size = settings.get('axis_label_font_size', 16)
    label_font = dict(family='Arial', color='black', size=label_size)
    legend_font_size = settings.get('legend_font_size', 12)

    fig.update_layout(
        **common_layout(settings),
        height=450,
        uirevision='retention_plot',
        xaxis=dict(
            **axis_settings,
            title=dict(text='Cycle number', font=label_font),
        ),
        yaxis=dict(
            **axis_settings,
            title=dict(text='Capacity retention / %', font=label_font),
            range=[0, 105],
        ),
        showlegend=False,
    )

    return fig


def create_multi_file_cd_plot(
    files_data: Dict[str, Dict],
    settings: dict = None,
    sample_info: dict = None,
    plot_type: str = 'V-t'
) -> go.Figure:
    """
    Create plot comparing multiple files

    Parameters
    ----------
    files_data : dict
        Dictionary of {filename: data}
    settings : dict
        Plot settings
    sample_info : dict
        Sample information
    plot_type : str
        'V-t', 'V-Q', or 'dQ/dV'

    Returns
    -------
    fig : go.Figure
        Plotly figure
    """
    if settings is None:
        settings = {}
    if sample_info is None:
        sample_info = {}

    fig = go.Figure()

    line_width = settings.get('line_width', 2)

    for i, (filename, data) in enumerate(files_data.items()):
        color = COLORS[i % len(COLORS)]

        if plot_type == 'V-t':
            time = data.get('time', np.array([]))
            voltage = data.get('voltage', np.array([]))

            if len(time) > 0 and len(voltage) > 0:
                time_h = time / 3600

                fig.add_trace(go.Scatter(
                    x=time_h,
                    y=voltage,
                    mode='lines',
                    name=filename,
                    line=dict(width=line_width, color=color),
                    hovertemplate=f'{filename}<br>' + '<i>t</i> = %{x:.2f} h<br><i>V</i> = %{y:.3f} V<extra></extra>'
                ))

        elif plot_type == 'V-Q':
            mass_g = sample_info.get('mass_mg', 1.0) / 1000

            if 'capacity' in data and 'voltage' in data:
                cap = np.abs(data['capacity'])
                if mass_g > 0:
                    cap = cap / mass_g

                fig.add_trace(go.Scatter(
                    x=cap,
                    y=data['voltage'],
                    mode='lines',
                    name=filename,
                    line=dict(width=line_width, color=color),
                    hovertemplate=f'{filename}<br>' + '<i>Q</i> = %{x:.1f}<br><i>V</i> = %{y:.3f} V<extra></extra>'
                ))

    # Update layout
    axis_settings = common_axis_settings(settings)
    label_size = settings.get('axis_label_font_size', 16)
    label_font = dict(family='Arial', color='black', size=label_size)
    legend_font_size = settings.get('legend_font_size', 12)

    if plot_type == 'V-t':
        x_label = 'Time / h'
    else:
        x_label = 'Capacity / mAh g⁻¹'

    fig.update_layout(
        **common_layout(settings),
        height=450,
        xaxis=dict(
            **axis_settings,
            title=dict(text=x_label, font=label_font),
        ),
        yaxis=dict(
            **axis_settings,
            title=dict(text='Voltage / V', font=label_font),
        ),
        showlegend=True,
        legend=dict(
            yanchor="top", y=0.99, xanchor="right", x=0.99,
            font=dict(size=legend_font_size), bgcolor='rgba(255,255,255,0.8)'
        )
    )

    return fig


def create_multi_file_vq_plot(
    files_data: Dict[str, Dict],
    settings: dict = None,
    sample_info: dict = None,
    selected_cycles: List[int] = None,
    color_mode: str = 'cycle',
    combine_by_time: bool = True
) -> go.Figure:
    """
    Create Voltage vs Capacity plot for multiple files

    Parameters
    ----------
    files_data : dict
        Dictionary of {filename: data}
    settings : dict
        Plot settings
    sample_info : dict
        Sample information
    selected_cycles : list
        List of cycle numbers to display
    color_mode : str
        'cycle': color by cycle number (rainbow)
        'charge_discharge': color by charge/discharge
        'first_last': 1st=red, middle=black, last=blue
        'grayscale': black to gray gradient
        'single_black': all black
    combine_by_time : bool
        If True, combine cycles from all files by time order

    Returns
    -------
    fig : go.Figure
        Plotly figure
    """
    if settings is None:
        settings = {}
    if sample_info is None:
        sample_info = {}

    fig = go.Figure()

    line_width = settings.get('line_width', 2)

    # Collect all cycles with time info for ordering
    all_cycles = []

    for filename, data in files_data.items():
        cycles = data.get('cycles', [])
        if not cycles:
            continue

        for cycle in cycles:
            cycle_num = cycle.get('cycle_number', 0)

            if selected_cycles is not None and cycle_num not in selected_cycles:
                continue

            # Skip cycles with zero current (relaxation)
            if 'current' in cycle:
                avg_current = np.mean(np.abs(cycle['current']))
                if avg_current < 0.01:
                    continue

            # Get data
            voltage = cycle.get('voltage', np.array([]))
            if len(voltage) == 0:
                continue

            # Calculate or get capacity
            if 'capacity' in cycle:
                capacity_mah = np.abs(cycle['capacity'])
            elif 'current' in cycle and 'time' in cycle:
                time_arr = cycle['time']
                current = cycle['current']
                dt = np.diff(time_arr)
                capacity_mah = np.zeros(len(time_arr))
                capacity_mah[1:] = np.cumsum(np.abs(current[:-1]) * dt) / 3600
            else:
                continue

            # Get start time for ordering
            start_time = cycle.get('time', [0])[0] if 'time' in cycle and len(cycle['time']) > 0 else 0

            all_cycles.append({
                'filename': filename,
                'cycle_num': cycle_num,
                'voltage': voltage,
                'capacity_mah': capacity_mah,
                'is_charge': cycle.get('is_charge', False),
                'is_discharge': cycle.get('is_discharge', False),
                'start_time': start_time,
                'half_cycle': cycle.get('half_cycle', 0),
            })

    # Sort by start time (combines data from multiple files in time order)
    if combine_by_time:
        all_cycles.sort(key=lambda x: x['start_time'])

    # Assign global cycle index for coloring
    total_cycles = len(all_cycles)

    for global_idx, cyc_data in enumerate(all_cycles):
        filename = cyc_data['filename']
        cycle_num = cyc_data['cycle_num']
        voltage = cyc_data['voltage']
        capacity_mah = cyc_data['capacity_mah']
        is_charge = cyc_data['is_charge']
        is_discharge = cyc_data['is_discharge']

        # Normalize capacity
        capacity_norm, cap_unit = get_capacity_normalization(sample_info, capacity_mah)

        # Get color using helper function (use global index for color)
        color, name_suffix = get_cycle_color(
            cycle_idx=global_idx,
            total_cycles=total_cycles,
            color_mode=color_mode,
            is_charge=is_charge,
            is_discharge=is_discharge,
            settings=settings
        )

        # Create trace name
        if name_suffix:
            trace_name = f'{filename} Cyc{cycle_num + 1} {name_suffix}'
        else:
            trace_name = f'{filename} Cyc{cycle_num + 1}'
            if is_charge:
                trace_name += '(C)'
            elif is_discharge:
                trace_name += '(D)'

        fig.add_trace(go.Scatter(
            x=capacity_norm,
            y=voltage,
            mode='lines',
            name=trace_name,
            line=dict(width=line_width, color=color),
            hovertemplate=f'{trace_name}<br>' + '<i>Q</i> = %{x:.1f}<br><i>V</i> = %{y:.3f} V<extra></extra>'
        ))

    # Update layout
    axis_settings = common_axis_settings(settings)
    label_size = settings.get('axis_label_font_size', 16)
    label_font = dict(family='Arial', color='black', size=label_size)
    legend_font_size = settings.get('legend_font_size', 12)
    show_legend = settings.get('show_legend', True)

    # Get capacity unit from sample_info
    capacity_unit = sample_info.get('capacity_unit', 'mAh/g')
    if capacity_unit == 'mAh/g':
        cap_label = 'Capacity / mAh g⁻¹'
    else:
        cap_label = 'Capacity / mAh cm⁻²'

    voltage_label = settings.get('voltage_label', 'Voltage / V')

    fig.update_layout(
        **common_layout(settings),
        height=450,
        uirevision='multi_vq_plot',
        xaxis=dict(
            **axis_settings,
            title=dict(text=cap_label, font=label_font),
        ),
        yaxis=dict(
            **axis_settings,
            title=dict(text=voltage_label, font=label_font),
        ),
        showlegend=show_legend,
        legend=dict(
            yanchor="top", y=0.99, xanchor="right", x=0.99,
            font=dict(size=legend_font_size), bgcolor='rgba(255,255,255,0.8)'
        )
    )

    return fig


def create_bode_plot(
    eis_data: List[Dict],
    settings: dict = None,
    plot_type: str = 'both'
) -> go.Figure:
    """
    Create Bode plot (frequency response) from EIS data
    Shows |Z| vs frequency and Phase vs frequency

    Parameters
    ----------
    eis_data : list
        List of EIS data dictionaries with 'freq', 'Z_real', 'Z_imag'
    settings : dict
        Plot settings
    plot_type : str
        'impedance', 'phase', or 'both'

    Returns
    -------
    fig : go.Figure
        Plotly figure with Bode plot
    """
    if settings is None:
        settings = {}

    # Create subplots if showing both
    if plot_type == 'both':
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=('Impedance Magnitude', 'Phase Angle')
        )
    else:
        fig = go.Figure()

    line_width = settings.get('line_width', 2)

    for i, eis in enumerate(eis_data):
        if 'freq' not in eis or 'Z_real' not in eis or 'Z_imag' not in eis:
            continue

        freq = eis['freq']
        Z_real = eis['Z_real']
        Z_imag = eis['Z_imag']

        # Calculate impedance magnitude and phase
        Z_mag = np.sqrt(Z_real**2 + Z_imag**2)
        Z_phase = np.arctan2(-Z_imag, Z_real) * 180 / np.pi  # degrees

        color = COLORS[i % len(COLORS)]
        name = f"PEIS {eis.get('technique_index', i+1)}"

        if plot_type == 'both':
            # Impedance magnitude on row 1
            fig.add_trace(go.Scatter(
                x=freq,
                y=Z_mag,
                mode='lines+markers',
                name=f"{name} |Z|",
                line=dict(width=line_width, color=color),
                marker=dict(size=4, color=color),
                hovertemplate=f'{name}<br>f = %{{x:.2e}} Hz<br>|Z| = %{{y:.1f}} Ω<extra></extra>'
            ), row=1, col=1)

            # Phase on row 2
            fig.add_trace(go.Scatter(
                x=freq,
                y=Z_phase,
                mode='lines+markers',
                name=f"{name} Phase",
                line=dict(width=line_width, color=color, dash='dash'),
                marker=dict(size=4, color=color),
                hovertemplate=f'{name}<br>f = %{{x:.2e}} Hz<br>Phase = %{{y:.1f}}°<extra></extra>',
                showlegend=False
            ), row=2, col=1)
        elif plot_type == 'impedance':
            fig.add_trace(go.Scatter(
                x=freq,
                y=Z_mag,
                mode='lines+markers',
                name=f"{name}",
                line=dict(width=line_width, color=color),
                marker=dict(size=4, color=color),
                hovertemplate=f'{name}<br>f = %{{x:.2e}} Hz<br>|Z| = %{{y:.1f}} Ω<extra></extra>'
            ))
        else:  # phase
            fig.add_trace(go.Scatter(
                x=freq,
                y=Z_phase,
                mode='lines+markers',
                name=f"{name}",
                line=dict(width=line_width, color=color),
                marker=dict(size=4, color=color),
                hovertemplate=f'{name}<br>f = %{{x:.2e}} Hz<br>Phase = %{{y:.1f}}°<extra></extra>'
            ))

    # Update layout
    axis_settings = common_axis_settings(settings)
    label_size = settings.get('axis_label_font_size', 16)
    label_font = dict(family='Arial', color='black', size=label_size)
    legend_font_size = settings.get('legend_font_size', 12)
    show_legend = settings.get('show_legend', True)

    if plot_type == 'both':
        fig.update_layout(
            **common_layout(settings),
            height=600,
            showlegend=show_legend,
            legend=dict(
                yanchor="top", y=0.99, xanchor="right", x=0.99,
                font=dict(size=legend_font_size), bgcolor='rgba(255,255,255,0.8)'
            )
        )

        # Update x-axis (bottom - row 2)
        fig.update_xaxes(
            type='log',
            title=dict(text='Frequency / Hz', font=label_font),
            **{k: v for k, v in axis_settings.items() if k not in ['title']},
            row=2, col=1
        )

        # Update y-axes
        fig.update_yaxes(
            type='log',
            title=dict(text='|Z| / Ω', font=label_font),
            **{k: v for k, v in axis_settings.items() if k not in ['title']},
            row=1, col=1
        )
        fig.update_yaxes(
            title=dict(text='Phase / °', font=label_font),
            **{k: v for k, v in axis_settings.items() if k not in ['title']},
            row=2, col=1
        )
    else:
        y_title = '|Z| / Ω' if plot_type == 'impedance' else 'Phase / °'
        fig.update_layout(
            **common_layout(settings),
            height=450,
            xaxis=dict(
                **axis_settings,
                type='log',
                title=dict(text='Frequency / Hz', font=label_font),
            ),
            yaxis=dict(
                **axis_settings,
                type='log' if plot_type == 'impedance' else 'linear',
                title=dict(text=y_title, font=label_font),
            ),
            showlegend=show_legend,
            legend=dict(
                yanchor="top", y=0.99, xanchor="right", x=0.99,
                font=dict(size=legend_font_size), bgcolor='rgba(255,255,255,0.8)'
            )
        )

    return fig


def create_cumulative_capacity_plot(
    data: Dict,
    settings: dict = None,
    sample_info: dict = None,
    capacity_type: str = 'discharge'
) -> go.Figure:
    """
    Create cumulative capacity plot showing total capacity delivered over cycles

    Parameters
    ----------
    data : dict
        Battery data dictionary with cycles
    settings : dict
        Plot settings
    sample_info : dict
        Sample information
    capacity_type : str
        'discharge', 'charge', or 'both'

    Returns
    -------
    fig : go.Figure
        Plotly figure with cumulative capacity
    """
    if settings is None:
        settings = {}
    if sample_info is None:
        sample_info = {}

    fig = go.Figure()

    cycles = data.get('cycles', [])

    if not cycles or len(cycles) == 0:
        fig.add_annotation(
            text="No cycle data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color='black')
        )
        return fig

    # Get normalization parameters
    capacity_unit = sample_info.get('capacity_unit', 'mAh/g')
    mass_mg = sample_info.get('mass_mg', 10.0)
    active_ratio = sample_info.get('active_ratio', 1.0)
    area_cm2 = sample_info.get('area_cm2', 0.636)
    active_mass_g = mass_mg * active_ratio / 1000  # g

    cycle_numbers = []
    cumulative_discharge = []
    cumulative_charge = []

    running_discharge = 0
    running_charge = 0

    for cycle in cycles:
        cn = cycle.get('cycle_number', 0) + 1
        cycle_numbers.append(cn)

        # Get capacities
        cap_charge = cycle.get('capacity_charge_mAh', 0) or 0
        cap_discharge = cycle.get('capacity_discharge_mAh', 0) or 0

        # Also check capacity_mAh from half_cycle parsing
        if cap_discharge == 0 and 'capacity_mAh' in cycle and cycle.get('is_discharge'):
            cap_discharge = cycle.get('capacity_mAh', 0)
        if cap_charge == 0 and 'capacity_mAh' in cycle and cycle.get('is_charge'):
            cap_charge = cycle.get('capacity_mAh', 0)

        # Normalize based on capacity unit
        if capacity_unit == 'mAh/g' and active_mass_g > 0:
            cap_discharge /= active_mass_g
            cap_charge /= active_mass_g
        elif capacity_unit == 'mAh/cm²' and area_cm2 > 0:
            cap_discharge /= area_cm2
            cap_charge /= area_cm2

        running_discharge += cap_discharge
        running_charge += cap_charge

        cumulative_discharge.append(running_discharge)
        cumulative_charge.append(running_charge)

    # Plot settings
    marker_size = settings.get('marker_size', 8) if settings.get('marker_size', 0) > 0 else 8
    line_width = settings.get('line_width', 2)
    charge_color = settings.get('charge_color', '#E63946')
    discharge_color = settings.get('discharge_color', '#457B9D')

    if capacity_type in ('discharge', 'both'):
        fig.add_trace(go.Scatter(
            x=cycle_numbers,
            y=cumulative_discharge,
            mode='markers+lines',
            name='Cumulative Discharge',
            marker=dict(size=marker_size, color=discharge_color, symbol='circle'),
            line=dict(width=line_width, color=discharge_color),
            hovertemplate='Cycle %{x}<br>Cum. Discharge: %{y:.1f}<extra></extra>'
        ))

    if capacity_type in ('charge', 'both'):
        fig.add_trace(go.Scatter(
            x=cycle_numbers,
            y=cumulative_charge,
            mode='markers+lines',
            name='Cumulative Charge',
            marker=dict(size=marker_size, color=charge_color, symbol='square'),
            line=dict(width=line_width, color=charge_color),
            hovertemplate='Cycle %{x}<br>Cum. Charge: %{y:.1f}<extra></extra>'
        ))

    # Update layout
    axis_settings = common_axis_settings(settings)
    label_size = settings.get('axis_label_font_size', 16)
    label_font = dict(family='Arial', color='black', size=label_size)
    legend_font_size = settings.get('legend_font_size', 12)

    # Set capacity label based on capacity unit
    if capacity_unit == 'mAh/g':
        cap_label = 'Cumulative Capacity / mAh g⁻¹'
    else:
        cap_label = 'Cumulative Capacity / mAh cm⁻²'

    fig.update_layout(
        **common_layout(settings),
        height=450,
        uirevision='cumulative_plot',
        xaxis=dict(
            **axis_settings,
            title=dict(text='Cycle number', font=label_font),
        ),
        yaxis=dict(
            **axis_settings,
            title=dict(text=cap_label, font=label_font),
        ),
        showlegend=True,
        legend=dict(
            yanchor="top", y=0.99, xanchor="left", x=0.01,
            font=dict(size=legend_font_size), bgcolor='rgba(255,255,255,0.8)'
        )
    )

    return fig


def apply_axis_range(fig: go.Figure, axis_range: Dict) -> go.Figure:
    """
    Apply custom axis ranges to a figure

    Parameters
    ----------
    fig : go.Figure
        Plotly figure to modify
    axis_range : dict
        Dictionary with 'x_min', 'x_max', 'y_min', 'y_max' (any can be None for auto)

    Returns
    -------
    fig : go.Figure
        Modified figure with custom axis ranges
    """
    x_min = axis_range.get('x_min')
    x_max = axis_range.get('x_max')
    y_min = axis_range.get('y_min')
    y_max = axis_range.get('y_max')

    if x_min is not None or x_max is not None:
        fig.update_xaxes(range=[x_min, x_max])

    if y_min is not None or y_max is not None:
        fig.update_yaxes(range=[y_min, y_max])

    return fig


def apply_trace_offset(
    fig: go.Figure,
    x_offset: float = 0,
    y_offset: float = 0,
    cumulative: bool = True
) -> go.Figure:
    """
    Apply offset to traces for comparison visualization

    Parameters
    ----------
    fig : go.Figure
        Plotly figure to modify
    x_offset : float
        X-axis offset between traces
    y_offset : float
        Y-axis offset between traces
    cumulative : bool
        If True, offset increases for each trace

    Returns
    -------
    fig : go.Figure
        Modified figure with offset traces
    """
    for i, trace in enumerate(fig.data):
        offset_multiplier = i if cumulative else 1

        if hasattr(trace, 'x') and trace.x is not None:
            trace.x = np.array(trace.x) + (x_offset * offset_multiplier)

        if hasattr(trace, 'y') and trace.y is not None:
            trace.y = np.array(trace.y) + (y_offset * offset_multiplier)

    return fig


def get_publication_config(width_px: int = 800, height_px: int = 600, scale: float = 2) -> dict:
    """
    Get configuration for high-resolution figure export

    Parameters
    ----------
    width_px : int
        Width in pixels
    height_px : int
        Height in pixels
    scale : float
        Scale factor for high DPI export

    Returns
    -------
    config : dict
        Plotly config dictionary for export
    """
    return {
        'toImageButtonOptions': {
            'format': 'svg',  # svg, png, jpeg, webp
            'filename': 'cd_plot',
            'width': width_px,
            'height': height_px,
            'scale': scale
        },
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'eraseshape'],
        'modeBarButtonsToRemove': ['lasso2d', 'select2d']
    }
