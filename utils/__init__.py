"""
Utilities module for CD Analyzer
Helper functions and calculations
"""

from .helpers import (
    calculate_capacity,
    calculate_dqdv,
    split_charge_discharge,
    calculate_coulombic_efficiency,
    smooth_data,
    format_formula_subscript,
    format_formula_html
)

from .session_state import initialize_session_state

from .file_processing import (
    process_uploaded_files,
    process_mps_file,
    load_mps_session,
    sort_files_by_time_and_assign_cycles
)

from .theocapacity import (
    calculate_theoretical_capacity,
    calculate_molar_mass,
    parse_composition
)

__all__ = [
    # Helpers
    'calculate_capacity',
    'calculate_dqdv',
    'split_charge_discharge',
    'calculate_coulombic_efficiency',
    'smooth_data',
    'format_formula_subscript',
    'format_formula_html',
    # Session state
    'initialize_session_state',
    # File processing
    'process_uploaded_files',
    'process_mps_file',
    'load_mps_session',
    'sort_files_by_time_and_assign_cycles',
    # Theoretical capacity
    'calculate_theoretical_capacity',
    'calculate_molar_mass',
    'parse_composition',
]
