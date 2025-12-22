"""
Components module for CD Analyzer
Plot generation and UI components
"""

from .plots import (
    create_cd_plot,
    create_capacity_voltage_plot,
    create_dqdv_plot,
    create_cycle_summary_plot,
    create_multi_file_vq_plot,
    create_multi_file_cd_plot,
    create_bode_plot,
    create_capacity_retention_plot,
    create_cumulative_capacity_plot,
    get_publication_config,
    apply_axis_range,
    apply_trace_offset,
    get_cycle_color,
    get_capacity_normalization,
    common_layout,
    common_axis_settings,
    COLORS,
    RAINBOW_COLORS
)

from .sidebar import (
    sidebar_header,
    sidebar_file_upload,
    sidebar_sample_info,
    sidebar_file_manager,
    sidebar_view_mode,
    sidebar_cycle_selection,
    sidebar_plot_settings,
    render_data_list_panel
)

__all__ = [
    # Plots
    'create_cd_plot',
    'create_capacity_voltage_plot',
    'create_dqdv_plot',
    'create_cycle_summary_plot',
    'create_multi_file_vq_plot',
    'create_multi_file_cd_plot',
    'create_bode_plot',
    'create_capacity_retention_plot',
    'create_cumulative_capacity_plot',
    'get_publication_config',
    'apply_axis_range',
    'apply_trace_offset',
    'get_cycle_color',
    'get_capacity_normalization',
    'common_layout',
    'common_axis_settings',
    'COLORS',
    'RAINBOW_COLORS',
    # Sidebar
    'sidebar_header',
    'sidebar_file_upload',
    'sidebar_sample_info',
    'sidebar_file_manager',
    'sidebar_view_mode',
    'sidebar_cycle_selection',
    'sidebar_plot_settings',
    'render_data_list_panel',
]
