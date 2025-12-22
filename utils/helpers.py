"""
Helper functions for battery data analysis
"""

import re
import numpy as np
from typing import Tuple, Optional, List, Dict, Any
from scipy import signal
from scipy.ndimage import uniform_filter1d


def format_formula_subscript(composition: str) -> str:
    """
    Convert numbers in chemical formula to subscript format for display.

    For Plotly/HTML: Uses Unicode subscript characters (₀₁₂₃₄₅₆₇₈₉)

    Parameters
    ----------
    composition : str
        Chemical formula (e.g., 'LiCoO2', 'Li0.5FePO4')

    Returns
    -------
    str
        Formula with subscript numbers (e.g., 'LiCoO₂', 'Li₀.₅FePO₄')

    Examples
    --------
    >>> format_formula_subscript('LiCoO2')
    'LiCoO₂'
    >>> format_formula_subscript('Li0.5FePO4')
    'Li₀.₅FePO₄'
    """
    if not composition:
        return composition

    # Unicode subscript digits mapping
    subscript_map = {
        '0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄',
        '5': '₅', '6': '₆', '7': '₇', '8': '₈', '9': '₉',
        '.': '.'  # Keep decimal point as-is for now
    }

    result = []
    i = 0
    while i < len(composition):
        char = composition[i]

        # Check if this is a number or decimal (subscript candidate)
        if char.isdigit() or (char == '.' and i + 1 < len(composition) and composition[i + 1].isdigit()):
            # Collect the entire number (including decimals)
            num_str = ''
            while i < len(composition) and (composition[i].isdigit() or composition[i] == '.'):
                num_str += composition[i]
                i += 1
            # Convert to subscript
            for c in num_str:
                if c in subscript_map:
                    result.append(subscript_map[c])
                else:
                    result.append(c)
        else:
            result.append(char)
            i += 1

    return ''.join(result)


def format_formula_html(composition: str) -> str:
    """
    Convert numbers in chemical formula to HTML subscript format.

    Parameters
    ----------
    composition : str
        Chemical formula (e.g., 'LiCoO2', 'Li0.5FePO4')

    Returns
    -------
    str
        Formula with HTML subscript tags (e.g., 'LiCoO<sub>2</sub>')
    """
    if not composition:
        return composition

    # Use regex to wrap numbers in <sub> tags
    return re.sub(r'(\d+\.?\d*)', r'<sub>\1</sub>', composition)


def calculate_capacity(
    current: np.ndarray,
    time: np.ndarray,
    mass_g: float = None
) -> np.ndarray:
    """
    Calculate cumulative capacity from current and time

    Parameters
    ----------
    current : np.ndarray
        Current in mA
    time : np.ndarray
        Time in seconds
    mass_g : float, optional
        Active material mass in grams for normalization

    Returns
    -------
    capacity : np.ndarray
        Cumulative capacity in mAh (or mAh/g if mass is provided)
    """
    if len(current) != len(time):
        raise ValueError("Current and time arrays must have same length")

    if len(current) < 2:
        return np.array([0.0])

    # Calculate time differences
    dt = np.diff(time)

    # Calculate capacity increments
    # Using absolute current to always get positive capacity
    dq = np.abs(current[:-1]) * dt / 3600  # mA * s / (s/h) = mAh

    # Cumulative capacity
    capacity = np.zeros(len(current))
    capacity[1:] = np.cumsum(dq)

    # Normalize by mass if provided
    if mass_g is not None and mass_g > 0:
        capacity = capacity / mass_g

    return capacity


def calculate_dqdv(
    voltage: np.ndarray,
    current: np.ndarray,
    time: np.ndarray,
    smooth_window: int = 5,
    mass_g: float = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate dQ/dV (differential capacity)

    Parameters
    ----------
    voltage : np.ndarray
        Voltage in V
    current : np.ndarray
        Current in mA
    time : np.ndarray
        Time in seconds
    smooth_window : int
        Window size for smoothing
    mass_g : float, optional
        Active material mass in grams

    Returns
    -------
    v_mid : np.ndarray
        Mid-point voltages
    dqdv : np.ndarray
        dQ/dV values
    """
    if len(voltage) < 3:
        return np.array([]), np.array([])

    # Calculate dQ
    dt = np.diff(time)
    dq = np.abs(current[:-1]) * dt / 3600  # mAh

    # Calculate dV
    dv = np.diff(voltage)

    # Calculate dQ/dV, avoiding division by very small values
    dqdv = np.zeros_like(dv)
    valid = np.abs(dv) > 1e-7
    dqdv[valid] = dq[valid] / dv[valid]

    # Normalize by mass
    if mass_g is not None and mass_g > 0:
        dqdv = dqdv / mass_g

    # Mid-point voltages
    v_mid = (voltage[:-1] + voltage[1:]) / 2

    # Smooth if requested
    if smooth_window > 1 and len(dqdv) > smooth_window:
        dqdv = smooth_data(dqdv, window=smooth_window, method='savgol')

    # Remove extreme outliers
    threshold = np.percentile(np.abs(dqdv[np.isfinite(dqdv)]), 99)
    dqdv = np.clip(dqdv, -threshold * 10, threshold * 10)

    return v_mid, dqdv


def split_charge_discharge(
    voltage: np.ndarray,
    current: np.ndarray,
    time: np.ndarray,
    capacity: np.ndarray = None
) -> Tuple[Dict, Dict]:
    """
    Split data into charge and discharge portions

    Parameters
    ----------
    voltage : np.ndarray
        Voltage data
    current : np.ndarray
        Current data
    time : np.ndarray
        Time data
    capacity : np.ndarray, optional
        Capacity data

    Returns
    -------
    charge_data : dict
        Dictionary with charge data
    discharge_data : dict
        Dictionary with discharge data
    """
    # Positive current = charge, negative = discharge
    # (This convention may vary depending on the instrument)
    charge_mask = current >= 0
    discharge_mask = current < 0

    charge_data = {
        'voltage': voltage[charge_mask],
        'current': current[charge_mask],
        'time': time[charge_mask],
    }

    discharge_data = {
        'voltage': voltage[discharge_mask],
        'current': current[discharge_mask],
        'time': time[discharge_mask],
    }

    if capacity is not None:
        charge_data['capacity'] = capacity[charge_mask]
        discharge_data['capacity'] = capacity[discharge_mask]

    return charge_data, discharge_data


def calculate_coulombic_efficiency(
    capacity_charge: float,
    capacity_discharge: float
) -> float:
    """
    Calculate Coulombic efficiency

    Parameters
    ----------
    capacity_charge : float
        Charge capacity (mAh)
    capacity_discharge : float
        Discharge capacity (mAh)

    Returns
    -------
    ce : float
        Coulombic efficiency (0-1 scale)
    """
    if capacity_charge <= 0:
        return 0.0

    return capacity_discharge / capacity_charge


def smooth_data(
    data: np.ndarray,
    window: int = 5,
    method: str = 'savgol'
) -> np.ndarray:
    """
    Smooth data using various methods

    Parameters
    ----------
    data : np.ndarray
        Input data
    window : int
        Window size for smoothing
    method : str
        'savgol' (Savitzky-Golay), 'uniform' (moving average), 'median'

    Returns
    -------
    smoothed : np.ndarray
        Smoothed data
    """
    if len(data) < window:
        return data

    if method == 'savgol':
        # Savitzky-Golay filter
        # Window must be odd
        if window % 2 == 0:
            window += 1
        # Polynomial order (must be less than window)
        polyorder = min(3, window - 1)
        return signal.savgol_filter(data, window, polyorder)

    elif method == 'uniform':
        # Uniform (moving average) filter
        return uniform_filter1d(data, size=window)

    elif method == 'median':
        # Median filter
        return signal.medfilt(data, kernel_size=window if window % 2 == 1 else window + 1)

    else:
        return data


def find_peaks_in_dqdv(
    voltage: np.ndarray,
    dqdv: np.ndarray,
    prominence: float = 0.1,
    height: float = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find peaks in dQ/dV curve

    Parameters
    ----------
    voltage : np.ndarray
        Voltage values
    dqdv : np.ndarray
        dQ/dV values
    prominence : float
        Required prominence of peaks
    height : float, optional
        Minimum height of peaks

    Returns
    -------
    peak_voltages : np.ndarray
        Voltages at peak positions
    peak_heights : np.ndarray
        Heights of peaks
    """
    if len(dqdv) < 3:
        return np.array([]), np.array([])

    # Find peaks
    peaks, properties = signal.find_peaks(
        dqdv,
        prominence=prominence * np.max(np.abs(dqdv)),
        height=height
    )

    if len(peaks) == 0:
        return np.array([]), np.array([])

    peak_voltages = voltage[peaks]
    peak_heights = dqdv[peaks]

    return peak_voltages, peak_heights


def calculate_rate_capability(
    capacities: List[float],
    c_rates: List[float],
    reference_c_rate: float = 0.1
) -> List[float]:
    """
    Calculate rate capability as percentage of low-rate capacity

    Parameters
    ----------
    capacities : list
        List of capacities at different C-rates
    c_rates : list
        List of corresponding C-rates
    reference_c_rate : float
        Reference C-rate for 100% capacity

    Returns
    -------
    rate_capability : list
        Rate capability as percentage
    """
    # Find reference capacity
    ref_idx = None
    for i, c in enumerate(c_rates):
        if abs(c - reference_c_rate) < 0.01:
            ref_idx = i
            break

    if ref_idx is None:
        # Use minimum C-rate as reference
        ref_idx = np.argmin(c_rates)

    ref_capacity = capacities[ref_idx]

    if ref_capacity <= 0:
        return [0.0] * len(capacities)

    return [cap / ref_capacity * 100 for cap in capacities]


def calculate_capacity_retention(
    capacities: List[float],
    reference_cycle: int = 0
) -> List[float]:
    """
    Calculate capacity retention as percentage of reference cycle

    Parameters
    ----------
    capacities : list
        List of capacities for each cycle
    reference_cycle : int
        Index of reference cycle (default: first cycle)

    Returns
    -------
    retention : list
        Capacity retention as percentage
    """
    if len(capacities) == 0:
        return []

    ref_capacity = capacities[reference_cycle]

    if ref_capacity <= 0:
        return [0.0] * len(capacities)

    return [cap / ref_capacity * 100 for cap in capacities]


def interpolate_to_voltage(
    voltage: np.ndarray,
    values: np.ndarray,
    target_voltages: np.ndarray
) -> np.ndarray:
    """
    Interpolate values to target voltage points

    Parameters
    ----------
    voltage : np.ndarray
        Original voltage array
    values : np.ndarray
        Values to interpolate (e.g., capacity)
    target_voltages : np.ndarray
        Target voltage points

    Returns
    -------
    interpolated : np.ndarray
        Interpolated values at target voltages
    """
    return np.interp(target_voltages, voltage, values)


def calculate_energy(
    voltage: np.ndarray,
    current: np.ndarray,
    time: np.ndarray
) -> float:
    """
    Calculate energy (Wh) from voltage, current, and time

    Parameters
    ----------
    voltage : np.ndarray
        Voltage in V
    current : np.ndarray
        Current in mA
    time : np.ndarray
        Time in seconds

    Returns
    -------
    energy : float
        Energy in mWh
    """
    if len(voltage) < 2:
        return 0.0

    dt = np.diff(time)
    power = voltage[:-1] * current[:-1]  # mW
    energy = np.sum(power * dt) / 3600  # mWh

    return energy
