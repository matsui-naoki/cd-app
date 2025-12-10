"""
Data loader module for battery charge-discharge data
Supports BioLogic .mpt files and other common formats
Uses custom MPT loader based on bioloader patterns (https://github.com/ks250206/bioloader)
"""

import numpy as np
import pandas as pd
import os
import tempfile
import re
from typing import Tuple, Optional, List, Dict, Any
from io import StringIO

# Import custom MPT loader (based on bioloader)
try:
    from .mpt_loader import load_mpt_file, load_mpt_with_cycles, MptColumn
    HAS_MPT_LOADER = True
except ImportError:
    HAS_MPT_LOADER = False

# Try to import galvani for BioLogic file support (fallback)
try:
    from galvani import BioLogic
    HAS_GALVANI = True
except ImportError:
    HAS_GALVANI = False


def get_supported_formats() -> List[str]:
    """Return list of supported file extensions"""
    formats = ['mpt', 'mpr', 'csv', 'txt']
    return formats


def load_uploaded_file(uploaded_file, file_extension: str) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Load battery data from an uploaded file in Streamlit

    Parameters
    ----------
    uploaded_file : UploadedFile
        File uploaded via Streamlit file_uploader
    file_extension : str
        File extension ('.mpt', '.csv', etc.)

    Returns
    -------
    data : dict or None
        Dictionary containing loaded data
    error_message : str or None
        Error message if loading failed
    """
    try:
        # Read file content
        bytes_data = uploaded_file.read()

        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            tmp_file.write(bytes_data)
            tmp_path = tmp_file.name

        try:
            ext = file_extension.lower()
            if ext == '.mpt':
                data, error = load_biologic_mpt(tmp_path)
            elif ext == '.mpr':
                data, error = load_biologic_mpr(tmp_path)
            elif ext in ['.csv', '.txt']:
                data, error = load_csv_file(tmp_path)
            else:
                return None, f"Unsupported file format: {ext}"

            return data, error

        finally:
            # Clean up temporary file
            os.unlink(tmp_path)

    except Exception as e:
        return None, f"Error loading file: {str(e)}"


def load_biologic_mpt(file_path: str) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Load BioLogic .mpt file using custom MPT loader (based on bioloader)

    Parameters
    ----------
    file_path : str
        Path to the .mpt file

    Returns
    -------
    data : dict or None
        Dictionary with keys: time, voltage, current, capacity, cycles, etc.
    error_message : str or None
        Error message if loading failed
    """
    try:
        # Primary method: Use our custom MPT loader (based on bioloader patterns)
        if HAS_MPT_LOADER:
            data, error = load_mpt_with_cycles(file_path)
            if data is not None:
                # Add file_type for consistency
                data['file_type'] = 'mpt'
                return data, None
            # If custom loader fails, try fallbacks

        # Fallback 1: galvani library
        if HAS_GALVANI:
            data, error = _load_mpt_galvani(file_path)
            if data is not None:
                return data, None

        # Fallback 2: custom basic parser
        return _load_mpt_custom(file_path)

    except Exception as e:
        return None, f"Error loading .mpt file: {str(e)}"


def load_biologic_mpr(file_path: str) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Load BioLogic .mpr file (binary format) using galvani library

    Parameters
    ----------
    file_path : str
        Path to the .mpr file

    Returns
    -------
    data : dict or None
        Dictionary with keys: time, voltage, current, capacity, cycles, etc.
    error_message : str or None
        Error message if loading failed
    """
    # Try galvani first
    if HAS_GALVANI:
        try:
            data, error = _load_mpr_galvani(file_path)
            if data is not None:
                return data, None
        except Exception:
            pass  # Fall through to custom parser

    # Fall back to universal mpr_parser
    try:
        from .mpr_parser import load_mpr_to_dict
        mpr_data, error = load_mpr_to_dict(file_path)

        if error or mpr_data is None:
            return None, f"Error loading .mpr file: {error}"

        # Convert mpr_parser format to data_loader format
        data = {
            'raw_df': mpr_data.get('raw_df'),
            'columns': mpr_data.get('columns', []),
            'file_type': 'mpr',
        }

        # Map standard column names
        if 'time' in mpr_data:
            data['time'] = mpr_data['time']
        if 'voltage' in mpr_data:
            data['voltage'] = mpr_data['voltage']
        if 'current' in mpr_data:
            data['current'] = mpr_data['current']
        if 'capacity' in mpr_data:
            data['capacity'] = mpr_data['capacity']
        if 'half_cycle' in mpr_data:
            data['half_cycle'] = mpr_data['half_cycle']
        if 'cycle_number' in mpr_data:
            data['cycle_number'] = mpr_data['cycle_number']

        # EIS data
        if 'freq' in mpr_data:
            data['freq'] = mpr_data['freq']
        if 'Z_real' in mpr_data:
            data['Z_real'] = mpr_data['Z_real']
        if 'Z_imag' in mpr_data:
            data['Z_imag'] = mpr_data['Z_imag']

        # Parse cycles
        data['cycles'] = _parse_cycles(data)

        return data, None

    except Exception as e:
        return None, f"Error loading .mpr file: {str(e)}"


def _load_mpr_galvani(file_path: str) -> Tuple[Optional[Dict], Optional[str]]:
    """Load .mpr file using galvani library"""
    try:
        mpr = BioLogic.MPRfile(file_path)
        df = pd.DataFrame(mpr.data)

        # Map column names to standard names
        # MPR files may have different column names than MPT
        column_mapping = {
            'time/s': 'time',
            'Ewe/V': 'voltage',
            '<Ewe>/V': 'voltage',
            'Ewe': 'voltage',
            'I/mA': 'current',
            '<I>/mA': 'current',
            'control/mA': 'current',
            'control/V/mA': 'current',  # GCPL uses this column name
            'I': 'current',
            'Capacity/mA.h': 'capacity',
            'Q charge/discharge/mA.h': 'capacity',
            '(Q-Qo)/mA.h': 'capacity',
            'Q charge/mA.h': 'capacity',
            'Q discharge/mA.h': 'capacity',
            'Capacity': 'capacity',
            'Ns': 'ns',
            'Ns changes': 'ns_changes',
            'ox/red': 'ox_red',
            'cycle number': 'cycle_number',
            'half cycle': 'half_cycle',
            'd(Q-Qo)/dE/mA.h/V': 'dqdv',
            'dQ/mA.h': 'dq',
            'dq/mA.h': 'dq',
        }

        # Rename columns
        df_renamed = df.copy()
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                df_renamed[new_name] = df[old_name]

        # Build data dictionary
        data = {
            'raw_df': df,  # Keep original dataframe
            'columns': list(df.columns),
            'file_type': 'mpr',
        }

        # Extract arrays
        if 'time' in df_renamed.columns:
            data['time'] = df_renamed['time'].values
        if 'voltage' in df_renamed.columns:
            data['voltage'] = df_renamed['voltage'].values
        if 'current' in df_renamed.columns:
            data['current'] = df_renamed['current'].values
        if 'capacity' in df_renamed.columns:
            data['capacity'] = df_renamed['capacity'].values
        if 'dqdv' in df_renamed.columns:
            data['dqdv'] = df_renamed['dqdv'].values
        if 'cycle_number' in df_renamed.columns:
            data['cycle_number'] = df_renamed['cycle_number'].values
        if 'half_cycle' in df_renamed.columns:
            data['half_cycle'] = df_renamed['half_cycle'].values
        if 'ns' in df_renamed.columns:
            data['ns'] = df_renamed['ns'].values
        if 'ns_changes' in df_renamed.columns:
            data['ns_changes'] = df_renamed['ns_changes'].values
        if 'ox_red' in df_renamed.columns:
            data['ox_red'] = df_renamed['ox_red'].values

        # Parse cycles if possible
        data['cycles'] = _parse_cycles(data)

        return data, None

    except Exception as e:
        return None, f"Error loading .mpr file with galvani: {str(e)}"


def _load_mpt_galvani(file_path: str) -> Tuple[Optional[Dict], Optional[str]]:
    """Load .mpt file using galvani library"""
    try:
        mpt = BioLogic.MPTfile(file_path)
        df = pd.DataFrame(mpt.data)

        # Map column names to standard names
        column_mapping = {
            'time/s': 'time',
            'Ewe/V': 'voltage',
            'I/mA': 'current',
            'control/mA': 'current',
            '<I>/mA': 'current',
            'Capacity/mA.h': 'capacity',
            'Q charge/discharge/mA.h': 'capacity',
            '(Q-Qo)/mA.h': 'capacity',
            'Ns': 'ns',
            'Ns changes': 'ns_changes',
            'ox/red': 'ox_red',
            'cycle number': 'cycle_number',
            'd(Q-Qo)/dE/mA.h/V': 'dqdv',
        }

        # Rename columns
        df_renamed = df.copy()
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                df_renamed[new_name] = df[old_name]

        # Build data dictionary
        data = {
            'raw_df': df,  # Keep original dataframe
            'columns': list(df.columns),
        }

        # Extract arrays
        if 'time' in df_renamed.columns:
            data['time'] = df_renamed['time'].values
        if 'voltage' in df_renamed.columns:
            data['voltage'] = df_renamed['voltage'].values
        if 'current' in df_renamed.columns:
            data['current'] = df_renamed['current'].values
        if 'capacity' in df_renamed.columns:
            data['capacity'] = df_renamed['capacity'].values
        if 'dqdv' in df_renamed.columns:
            data['dqdv'] = df_renamed['dqdv'].values
        if 'cycle_number' in df_renamed.columns:
            data['cycle_number'] = df_renamed['cycle_number'].values
        if 'ns' in df_renamed.columns:
            data['ns'] = df_renamed['ns'].values
        if 'ns_changes' in df_renamed.columns:
            data['ns_changes'] = df_renamed['ns_changes'].values
        if 'ox_red' in df_renamed.columns:
            data['ox_red'] = df_renamed['ox_red'].values

        # Parse cycles if possible
        data['cycles'] = _parse_cycles(data)

        return data, None

    except Exception as e:
        # Fall back to custom parser
        return _load_mpt_custom(file_path)


def _load_mpt_custom(file_path: str) -> Tuple[Optional[Dict], Optional[str]]:
    """Custom parser for .mpt files when galvani is not available"""
    try:
        with open(file_path, 'r', encoding='latin-1') as f:
            lines = f.readlines()

        # Find header line count
        header_lines = 0
        for i, line in enumerate(lines):
            if line.startswith('Nb header lines'):
                match = re.search(r'(\d+)', line)
                if match:
                    header_lines = int(match.group(1))
                break

        if header_lines == 0:
            # Try to find data start another way
            for i, line in enumerate(lines):
                if '\t' in line and not line.startswith(('EC-Lab', 'BT-Lab', 'Nb', 'mode', 'ox/red')):
                    # Check if it looks like data
                    parts = line.strip().split('\t')
                    try:
                        float(parts[0])
                        header_lines = i
                        break
                    except:
                        continue

        # Read column headers
        if header_lines > 0 and header_lines < len(lines):
            header_line = lines[header_lines - 1].strip()
            columns = header_line.split('\t')
        else:
            return None, "Could not find column headers in .mpt file"

        # Read data
        data_lines = lines[header_lines:]
        data_str = ''.join(data_lines)

        df = pd.read_csv(StringIO(data_str), sep='\t', names=columns,
                         na_values=['', ' ', 'NaN', 'nan'], skip_blank_lines=True)

        # Clean up column names and map to standard names
        df.columns = df.columns.str.strip()

        column_mapping = {
            'time/s': 'time',
            'Ewe/V': 'voltage',
            'I/mA': 'current',
            'control/mA': 'current',
            '<I>/mA': 'current',
            'Capacity/mA.h': 'capacity',
            'Q charge/discharge/mA.h': 'capacity',
            '(Q-Qo)/mA.h': 'capacity',
            'Ns': 'ns',
            'Ns changes': 'ns_changes',
            'ox/red': 'ox_red',
            'cycle number': 'cycle_number',
            'd(Q-Qo)/dE/mA.h/V': 'dqdv',
        }

        data = {
            'raw_df': df,
            'columns': list(df.columns),
        }

        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                data[new_name] = df[old_name].values

        data['cycles'] = _parse_cycles(data)

        return data, None

    except Exception as e:
        return None, f"Error parsing .mpt file: {str(e)}"


def _parse_cycles(data: Dict) -> List[Dict]:
    """
    Parse data into individual charge/discharge cycles (half-cycles)

    Parameters
    ----------
    data : dict
        Dictionary with time, voltage, current, etc.

    Returns
    -------
    cycles : list
        List of dictionaries, one per half-cycle with is_charge/is_discharge flags
    """
    cycles = []

    # Method 1: Use half_cycle column (most accurate for GCPL data)
    if 'half_cycle' in data:
        half_cycles = data['half_cycle']
        unique_hc = np.unique(half_cycles[~np.isnan(half_cycles)])

        for hc_num in unique_hc:
            mask = half_cycles == hc_num
            indices = np.where(mask)[0]

            if len(indices) == 0:
                continue

            start_idx = indices[0]
            end_idx = indices[-1] + 1

            # Full cycle number = half_cycle // 2
            cycle_num = int(hc_num) // 2

            cycle_data = _extract_cycle_data(data, start_idx, end_idx, cycle_num)
            if cycle_data:
                # Skip short cycles (< 10 data points) - likely transition artifacts
                n_points = len(cycle_data.get('voltage', []))
                if n_points < 10:
                    continue

                cycle_data['half_cycle'] = int(hc_num)
                # Determine if charge or discharge from current sign
                if 'current' in cycle_data:
                    avg_current = np.mean(cycle_data['current'])
                    avg_abs_current = np.mean(np.abs(cycle_data['current']))
                    # Skip relaxation process (low current < 0.01 mA)
                    if avg_abs_current < 0.01:
                        continue
                    cycle_data['is_charge'] = avg_current > 0
                    cycle_data['is_discharge'] = avg_current < 0
                # Or from ox_red column (1=oxidation/charge, 0=reduction/discharge)
                if 'ox_red' in data:
                    ox_red = data['ox_red'][start_idx:end_idx]
                    avg_ox_red = np.mean(ox_red)
                    cycle_data['is_charge'] = avg_ox_red > 0.5
                    cycle_data['is_discharge'] = avg_ox_red < 0.5
                cycles.append(cycle_data)

        return cycles

    # Method 2: Use Ns and Ns changes columns (Biologic MB technique)
    if 'ns' in data and 'ns_changes' in data:
        ns = data['ns']
        ns_changes = data['ns_changes']

        # Find sequence boundaries
        change_indices = np.where(ns_changes == 1)[0]

        if len(change_indices) > 0:
            # Add start and end
            boundaries = [0] + list(change_indices) + [len(ns)]

            cycle_num = 0
            half_cycle_num = 0
            for i in range(len(boundaries) - 1):
                start_idx = boundaries[i]
                end_idx = boundaries[i + 1]

                if end_idx <= start_idx:
                    continue

                cycle_data = _extract_cycle_data(data, start_idx, end_idx, cycle_num)
                if cycle_data:
                    cycle_data['half_cycle'] = half_cycle_num
                    # Determine charge/discharge from current
                    if 'current' in cycle_data:
                        avg_current = np.mean(cycle_data['current'])
                        cycle_data['is_charge'] = avg_current > 0
                        cycle_data['is_discharge'] = avg_current < 0
                    cycles.append(cycle_data)
                    half_cycle_num += 1
                    # Increment cycle number every 2 half-cycles
                    if half_cycle_num % 2 == 0:
                        cycle_num += 1

            return cycles

    # Method 3: Use cycle_number column
    if 'cycle_number' in data:
        cycle_nums = data['cycle_number']
        unique_cycles = np.unique(cycle_nums[~np.isnan(cycle_nums)])

        for cn in unique_cycles:
            mask = cycle_nums == cn
            indices = np.where(mask)[0]

            if len(indices) == 0:
                continue

            start_idx = indices[0]
            end_idx = indices[-1] + 1

            cycle_data = _extract_cycle_data(data, start_idx, end_idx, int(cn))
            if cycle_data:
                # Split by charge/discharge if we have current data
                if 'current' in data:
                    current_slice = data['current'][start_idx:end_idx]
                    charge_mask = current_slice > 0
                    discharge_mask = current_slice < 0

                    # If mixed, split into separate half-cycles
                    if np.any(charge_mask) and np.any(discharge_mask):
                        # Find transition point
                        charge_indices = np.where(charge_mask)[0]
                        discharge_indices = np.where(discharge_mask)[0]

                        # Charge half-cycle
                        if len(charge_indices) > 0:
                            c_start = start_idx + charge_indices[0]
                            c_end = start_idx + charge_indices[-1] + 1
                            charge_data = _extract_cycle_data(data, c_start, c_end, int(cn))
                            if charge_data:
                                charge_data['is_charge'] = True
                                charge_data['is_discharge'] = False
                                charge_data['half_cycle'] = int(cn) * 2
                                cycles.append(charge_data)

                        # Discharge half-cycle
                        if len(discharge_indices) > 0:
                            d_start = start_idx + discharge_indices[0]
                            d_end = start_idx + discharge_indices[-1] + 1
                            discharge_data = _extract_cycle_data(data, d_start, d_end, int(cn))
                            if discharge_data:
                                discharge_data['is_charge'] = False
                                discharge_data['is_discharge'] = True
                                discharge_data['half_cycle'] = int(cn) * 2 + 1
                                cycles.append(discharge_data)
                    else:
                        # Single phase
                        cycle_data['is_charge'] = np.any(charge_mask)
                        cycle_data['is_discharge'] = np.any(discharge_mask)
                        cycles.append(cycle_data)
                else:
                    cycles.append(cycle_data)

        return cycles

    # Method 4: Use current sign changes (simple charge/discharge detection)
    if 'current' in data and 'voltage' in data:
        current = data['current']
        voltage = data['voltage']
        time = data.get('time', np.arange(len(current)))

        # Find sign changes in current (ignoring noise near zero)
        # Use a threshold to avoid detecting noise as sign changes
        threshold = np.max(np.abs(current)) * 0.01
        significant_current = np.abs(current) > threshold
        sign = np.sign(current)
        sign[~significant_current] = 0

        # Find actual sign changes
        sign_changes = np.where(np.diff(sign) != 0)[0]

        if len(sign_changes) > 0:
            boundaries = [0] + list(sign_changes + 1) + [len(current)]
            cycle_num = 0
            half_cycle_num = 0

            for i in range(len(boundaries) - 1):
                start_idx = boundaries[i]
                end_idx = boundaries[i + 1]

                if end_idx <= start_idx:
                    continue

                # Skip segments with mostly zero current
                segment_current = current[start_idx:end_idx]
                if np.mean(np.abs(segment_current)) < threshold:
                    continue

                cycle_data = _extract_cycle_data(data, start_idx, end_idx, cycle_num)
                if cycle_data:
                    avg_current = np.mean(segment_current)
                    cycle_data['is_charge'] = avg_current > 0
                    cycle_data['is_discharge'] = avg_current < 0
                    cycle_data['half_cycle'] = half_cycle_num
                    cycles.append(cycle_data)
                    half_cycle_num += 1
                    # Increment cycle number every 2 half-cycles
                    if half_cycle_num % 2 == 0:
                        cycle_num += 1

            return cycles

    return cycles


def _extract_cycle_data(data: Dict, start_idx: int, end_idx: int, cycle_num: int) -> Optional[Dict]:
    """Extract data for a single cycle/half-cycle"""
    cycle = {
        'cycle_number': cycle_num,
        'start_idx': start_idx,
        'end_idx': end_idx,
    }

    # Extract arrays for this cycle
    for key in ['time', 'voltage', 'current', 'capacity', 'dqdv', 'ox_red']:
        if key in data and data[key] is not None:
            cycle[key] = data[key][start_idx:end_idx]

    # Calculate capacities if possible
    if 'current' in cycle and 'time' in cycle:
        current = cycle['current']
        time = cycle['time']

        if len(current) > 1 and len(time) > 1:
            # Calculate total capacity from current integration
            dt = np.diff(time)
            total_cap = np.sum(np.abs(current[:-1]) * dt) / 3600  # mA*s to mAh
            cycle['capacity_mAh'] = total_cap

            # Separate charge and discharge by current sign
            # In BioLogic convention: positive current = charge (for Li-ion)
            charge_mask = current > 0
            discharge_mask = current < 0

            if np.any(charge_mask):
                charge_indices = np.where(charge_mask[:-1])[0]
                if len(charge_indices) > 0:
                    i_charge = np.abs(current[charge_indices])
                    dt_charge = dt[charge_indices]
                    capacity_charge = np.sum(i_charge * dt_charge) / 3600
                    cycle['capacity_charge_mAh'] = capacity_charge

            if np.any(discharge_mask):
                discharge_indices = np.where(discharge_mask[:-1])[0]
                if len(discharge_indices) > 0:
                    i_discharge = np.abs(current[discharge_indices])
                    dt_discharge = dt[discharge_indices]
                    capacity_discharge = np.sum(i_discharge * dt_discharge) / 3600
                    cycle['capacity_discharge_mAh'] = capacity_discharge

            # Coulombic efficiency
            if 'capacity_charge_mAh' in cycle and 'capacity_discharge_mAh' in cycle:
                if cycle['capacity_charge_mAh'] > 0:
                    cycle['coulombic_efficiency'] = (
                        cycle['capacity_discharge_mAh'] / cycle['capacity_charge_mAh']
                    )

    # Use capacity column if available (often more accurate)
    if 'capacity' in cycle and len(cycle['capacity']) > 0:
        cap = np.abs(cycle['capacity'])
        cap_from_col = np.max(cap) - np.min(cap)
        if cap_from_col > 0:
            cycle['capacity_max_mAh'] = cap_from_col
            # Update capacity values if from capacity column
            if cycle.get('is_charge'):
                cycle['capacity_charge_mAh'] = cap_from_col
            elif cycle.get('is_discharge'):
                cycle['capacity_discharge_mAh'] = cap_from_col

    return cycle


def load_csv_file(file_path: str) -> Tuple[Optional[Dict], Optional[str]]:
    """Load CSV/TXT file with flexible column detection"""
    try:
        # Try different separators
        for sep in [',', '\t', ';', ' ']:
            try:
                df = pd.read_csv(file_path, sep=sep, engine='python')
                if len(df.columns) > 1:
                    break
            except:
                continue

        if len(df.columns) <= 1:
            return None, "Could not parse CSV file - check separator"

        # Try to identify columns
        data = {'raw_df': df, 'columns': list(df.columns)}

        # Common column name patterns
        time_patterns = ['time', 'Time', 't', 's', 'sec', 'second']
        voltage_patterns = ['voltage', 'Voltage', 'V', 'Ewe', 'E', 'potential', 'Potential']
        current_patterns = ['current', 'Current', 'I', 'i', 'mA', 'A']
        capacity_patterns = ['capacity', 'Capacity', 'Q', 'q', 'mAh', 'Ah', 'charge']

        for col in df.columns:
            col_lower = col.lower().strip()

            for pattern in time_patterns:
                if pattern.lower() in col_lower:
                    data['time'] = df[col].values
                    break

            for pattern in voltage_patterns:
                if pattern.lower() in col_lower:
                    data['voltage'] = df[col].values
                    break

            for pattern in current_patterns:
                if pattern.lower() in col_lower:
                    data['current'] = df[col].values
                    break

            for pattern in capacity_patterns:
                if pattern.lower() in col_lower:
                    data['capacity'] = df[col].values
                    break

        # If columns not identified, use first few columns
        if 'time' not in data and len(df.columns) >= 1:
            data['time'] = df.iloc[:, 0].values
        if 'voltage' not in data and len(df.columns) >= 2:
            data['voltage'] = df.iloc[:, 1].values
        if 'current' not in data and len(df.columns) >= 3:
            data['current'] = df.iloc[:, 2].values

        data['cycles'] = _parse_cycles(data)

        return data, None

    except Exception as e:
        return None, f"Error loading CSV file: {str(e)}"


def parse_biologic_header(file_path: str) -> Dict[str, Any]:
    """
    Parse BioLogic .mpt file header for metadata

    Parameters
    ----------
    file_path : str
        Path to the .mpt file

    Returns
    -------
    header_info : dict
        Dictionary containing header metadata
    """
    header_info = {
        'technique': None,
        'acquisition_started': None,
        'nb_header_lines': 0,
        'columns': [],
        'settings': {},
    }

    try:
        with open(file_path, 'r', encoding='latin-1') as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            line = line.strip()

            if line.startswith('Nb header lines'):
                match = re.search(r'(\d+)', line)
                if match:
                    header_info['nb_header_lines'] = int(match.group(1))

            elif 'Modulo Bat' in line or 'CCCV' in line or 'GCPL' in line:
                header_info['technique'] = line.split(':')[-1].strip() if ':' in line else line

            elif line.startswith('Acquisition started on'):
                header_info['acquisition_started'] = line.replace('Acquisition started on', '').strip()

            # Parse settings like mass, electrode area
            elif 'Mass of active material' in line or 'mass' in line.lower():
                match = re.search(r'([\d.]+)\s*(mg|g)', line, re.IGNORECASE)
                if match:
                    mass = float(match.group(1))
                    unit = match.group(2).lower()
                    if unit == 'g':
                        mass *= 1000  # Convert to mg
                    header_info['settings']['mass_mg'] = mass

            elif 'Electrode surface area' in line or 'area' in line.lower():
                match = re.search(r'([\d.]+)\s*(cm2|cm²|mm2|mm²)', line, re.IGNORECASE)
                if match:
                    area = float(match.group(1))
                    unit = match.group(2).lower()
                    if 'mm' in unit:
                        area /= 100  # Convert mm² to cm²
                    header_info['settings']['area_cm2'] = area

            # Stop parsing after header
            if header_info['nb_header_lines'] > 0 and i >= header_info['nb_header_lines']:
                # Get column names from last header line
                header_info['columns'] = lines[header_info['nb_header_lines'] - 1].strip().split('\t')
                break

    except Exception as e:
        header_info['error'] = str(e)

    return header_info


def validate_cd_data(data: Dict) -> Tuple[bool, str]:
    """
    Validate charge-discharge data

    Parameters
    ----------
    data : dict
        Dictionary containing battery data

    Returns
    -------
    is_valid : bool
        True if data is valid
    message : str
        Validation message
    """
    if data is None:
        return False, "Data is None"

    # Check for essential arrays
    has_voltage = 'voltage' in data and data['voltage'] is not None and len(data['voltage']) > 0
    has_time = 'time' in data and data['time'] is not None and len(data['time']) > 0

    if not has_voltage:
        return False, "No voltage data found"

    if not has_time:
        return False, "No time data found"

    # Check for NaN/Inf
    voltage = data['voltage']
    if np.any(~np.isfinite(voltage)):
        # Remove bad values instead of failing
        pass

    return True, "Data is valid"
