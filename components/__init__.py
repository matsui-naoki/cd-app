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

__all__ = [
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
    'RAINBOW_COLORS'
]
