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


def create_capacity_voltage_plot(
    data: Dict,
    settings: dict = None,
    sample_info: dict = None,
    selected_cycles: List[int] = None,
    normalize_capacity: bool = True
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
        Sample information
    selected_cycles : list
        List of cycle numbers to display
    normalize_capacity : bool
        Whether to normalize capacity by active mass (mAh/g)

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

    # Mass for normalization
    mass_g = sample_info.get('mass_mg', 1.0) / 1000

    if cycles and len(cycles) > 0:
        for i, cycle in enumerate(cycles):
            if selected_cycles is not None and cycle['cycle_number'] not in selected_cycles:
                continue

            # Calculate capacity for this cycle
            if 'current' in cycle and 'time' in cycle and 'voltage' in cycle:
                time = cycle['time']
                current = cycle['current']
                voltage = cycle['voltage']

                # Calculate cumulative capacity
                dt = np.diff(time)
                capacity = np.zeros(len(time))
                capacity[1:] = np.cumsum(np.abs(current[:-1]) * dt) / 3600  # mAh

                if normalize_capacity and mass_g > 0:
                    capacity = capacity / mass_g  # mAh/g

                # Color by cycle
                color = COLORS[i % len(COLORS)]

                fig.add_trace(go.Scatter(
                    x=capacity,
                    y=voltage,
                    mode='lines',
                    name=f'Cycle {cycle["cycle_number"] + 1}',
                    line=dict(width=line_width, color=color),
                    hovertemplate='<i>Q</i> = %{x:.1f}<br><i>V</i> = %{y:.3f} V<extra></extra>'
                ))

            elif 'capacity' in cycle and 'voltage' in cycle:
                # Use pre-calculated capacity
                cap = np.abs(cycle['capacity'])
                if normalize_capacity and mass_g > 0:
                    cap = cap / mass_g

                color = COLORS[i % len(COLORS)]

                fig.add_trace(go.Scatter(
                    x=cap,
                    y=cycle['voltage'],
                    mode='lines',
                    name=f'Cycle {cycle["cycle_number"] + 1}',
                    line=dict(width=line_width, color=color),
                    hovertemplate='<i>Q</i> = %{x:.1f}<br><i>V</i> = %{y:.3f} V<extra></extra>'
                ))

    else:
        # Use raw data
        if 'capacity' in data and 'voltage' in data:
            cap = np.abs(data['capacity'])
            if normalize_capacity and mass_g > 0:
                cap = cap / mass_g

            fig.add_trace(go.Scatter(
                x=cap,
                y=data['voltage'],
                mode='lines',
                name='Voltage',
                line=dict(width=line_width, color='black'),
                hovertemplate='<i>Q</i> = %{x:.1f}<br><i>V</i> = %{y:.3f} V<extra></extra>'
            ))

        elif 'current' in data and 'time' in data and 'voltage' in data:
            # Calculate capacity
            time = data['time']
            current = data['current']
            dt = np.diff(time)
            capacity = np.zeros(len(time))
            capacity[1:] = np.cumsum(np.abs(current[:-1]) * dt) / 3600

            if normalize_capacity and mass_g > 0:
                capacity = capacity / mass_g

            fig.add_trace(go.Scatter(
                x=capacity,
                y=data['voltage'],
                mode='lines',
                name='Voltage',
                line=dict(width=line_width, color='black'),
                hovertemplate='<i>Q</i> = %{x:.1f}<br><i>V</i> = %{y:.3f} V<extra></extra>'
            ))

    # Update layout
    axis_settings = common_axis_settings(settings)
    label_size = settings.get('axis_label_font_size', 16)
    label_font = dict(family='Arial', color='black', size=label_size)
    legend_font_size = settings.get('legend_font_size', 12)
    show_legend = settings.get('show_legend', True)

    cap_unit = 'mAh g⁻¹' if normalize_capacity else 'mAh'
    capacity_label = f'Capacity / {cap_unit}'
    voltage_label = settings.get('voltage_label', 'Voltage / V')

    fig.update_layout(
        **common_layout(settings),
        height=450,
        uirevision='vq_plot',
        xaxis=dict(
            **axis_settings,
            title=dict(text=capacity_label, font=label_font),
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
        Sample information
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

    # Mass for normalization
    mass_g = sample_info.get('mass_mg', 1.0) / 1000

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

        # Normalize by mass
        if mass_g > 0:
            dqdv = dqdv / mass_g

        # Smooth
        if smooth and smooth_window > 1 and len(dqdv) > smooth_window:
            kernel = np.ones(smooth_window) / smooth_window
            dqdv = np.convolve(dqdv, kernel, mode='same')

        # Return mid-point voltages
        v_mid = (voltage[:-1] + voltage[1:]) / 2

        return v_mid, dqdv

    if 'dqdv' in data and 'voltage' in data:
        # Use pre-calculated dQ/dV
        dqdv = data['dqdv']
        voltage = data['voltage']

        # Filter out zeros and very large values
        valid = (np.abs(dqdv) > 0) & (np.abs(dqdv) < 1e10)

        if mass_g > 0:
            dqdv = dqdv / mass_g

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
    dqdv_label = settings.get('dqdv_label', 'dQ/dV / mAh g⁻¹ V⁻¹')

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
        Sample information

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

    mass_g = sample_info.get('mass_mg', 1.0) / 1000

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

        if cap_charge is not None and mass_g > 0:
            charge_capacities.append(cap_charge / mass_g)
        else:
            charge_capacities.append(None)

        if cap_discharge is not None and mass_g > 0:
            discharge_capacities.append(cap_discharge / mass_g)
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
            title=dict(text='Capacity / mAh g⁻¹', font=label_font),
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
