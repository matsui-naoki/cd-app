"""
BioLogic MPT file loader
Simplified MPT parser inspired by bioloader (https://github.com/ks250206/bioloader)
Supports GCPL, PEIS, OCV and other EC-Lab techniques

MIT License - Based on bioloader by ks250206
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from enum import StrEnum
from dataclasses import dataclass


class MptColumn(StrEnum):
    """MPT file column names (matching EC-Lab format)"""
    # Time and cycle
    TIME_S = "time/s"
    STEP_TIME_S = "step time/s"
    NS = "Ns"
    NS_CHANGES = "Ns changes"
    CYCLE_NUMBER = "cycle number"
    HALF_CYCLE = "half cycle"
    OX_RED = "ox/red"

    # Voltage and current
    EWE_V = "Ewe/V"
    I_MA = "I/mA"
    EWE_V_AVG = "<Ewe>/V"
    I_MA_AVG = "<I>/mA"

    # Capacity
    CAPACITY_MA_H = "Capacity/mA.h"
    Q_CHARGE_MA_H = "Q charge/mA.h"
    Q_DISCHARGE_MA_H = "Q discharge/mA.h"
    Q_CHARGE_DISCHARGE_MA_H = "Q charge/discharge/mA.h"
    DQ_MA_H = "dq/mA.h"

    # Energy
    ENERGY_W_H = "Energy/W.h"
    ENERGY_CHARGE_W_H = "Energy charge/W.h"
    ENERGY_DISCHARGE_W_H = "Energy discharge/W.h"
    P_W = "P/W"

    # EIS (Impedance)
    FREQ_HZ = "freq/Hz"
    RE_Z_OHM = "Re(Z)/Ohm"
    IM_Z_OHM = "-Im(Z)/Ohm"
    Z_OHM = "|Z|/Ohm"
    PHASE_Z_DEG = "Phase(Z)/deg"

    # Control
    CONTROL_V = "control/V"
    CONTROL_MA = "control/mA"
    MODE = "mode"
    ERROR = "error"
    I_RANGE = "I Range"


@dataclass
class MptMetadata:
    """MPT file metadata"""
    filename: str
    header_lines: int
    technique: str
    acquisition_start: str
    device: str
    all_columns: List[str]
    raw_metadata: List[str]


def parse_mpt_metadata(filepath: Path) -> Tuple[MptMetadata, int]:
    """
    Parse MPT file metadata without loading data

    Parameters
    ----------
    filepath : Path
        Path to the MPT file

    Returns
    -------
    metadata : MptMetadata
        Parsed metadata
    skip_rows : int
        Number of rows to skip when loading data (excluding header row)
    """
    metadata_lines = []
    header_count = 0
    technique = ""
    acquisition_start = ""
    device = ""
    all_columns = []

    with open(filepath, 'r', encoding='latin-1') as f:
        # First line should be "EC-Lab ASCII FILE"
        first_line = f.readline().strip()
        if not first_line.startswith("EC-Lab"):
            raise ValueError(f"Not an EC-Lab MPT file: {filepath}")

        # Second line: "Nb header lines : XX"
        nb_header_line = f.readline().strip()
        if "Nb header lines" in nb_header_line:
            try:
                header_count = int(nb_header_line.split(":")[-1].strip())
            except ValueError:
                header_count = 50  # Default fallback

        # Read remaining metadata lines (header_count - 3 because:
        # line 1 = EC-Lab header, line 2 = Nb header lines, last line = column headers)
        # So metadata lines are from line 3 to line (header_count - 1)
        for i in range(header_count - 3):
            line = f.readline()
            metadata_lines.append(line.strip())

            # Extract technique (look for specific patterns)
            line_stripped = line.strip()
            if line_stripped.startswith("Potentio Electrochemical Impedance"):
                technique = "PEIS"
            elif line_stripped.startswith("Galvanostatic Cycling"):
                technique = "GCPL"
            elif line_stripped.startswith("Modulo Bat"):
                technique = "MB"
            elif "CCCV" in line_stripped:
                technique = "CCCV"
            elif "OCV" in line_stripped and "Open Circuit" in line_stripped:
                technique = "OCV"
            elif "Acquisition started on" in line:
                # Parse the date/time which comes after the colon
                parts = line.split(":", 1)
                if len(parts) > 1:
                    acquisition_start = parts[1].strip()
            elif line.startswith("Device :"):
                device = line.split(":")[-1].strip()

        # The last header line is the column names (line header_count)
        header_line = f.readline().strip()
        all_columns = header_line.split('\t')

    metadata = MptMetadata(
        filename=filepath.name,
        header_lines=header_count,
        technique=technique,
        acquisition_start=acquisition_start,
        device=device,
        all_columns=all_columns,
        raw_metadata=metadata_lines
    )

    # Return header_count - 1 because pandas skiprows should skip
    # all lines BEFORE the column header, and the header is read separately
    return metadata, header_count - 1


def load_mpt_file(filepath: str | Path) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Load BioLogic MPT file

    Parameters
    ----------
    filepath : str or Path
        Path to the MPT file

    Returns
    -------
    data : dict or None
        Dictionary with keys: raw_df, columns, metadata, time, voltage, current, etc.
    error : str or None
        Error message if loading failed
    """
    filepath = Path(filepath)

    if not filepath.exists():
        return None, f"File not found: {filepath}"

    try:
        # Parse metadata first
        metadata, skip_rows = parse_mpt_metadata(filepath)

        # Load data with pandas
        df = pd.read_csv(
            filepath,
            sep='\t',
            skiprows=skip_rows,
            encoding='latin-1',
            low_memory=False
        )

        # Clean column names (remove leading/trailing spaces)
        df.columns = [col.strip() for col in df.columns]

        # Build result dictionary
        result = {
            'raw_df': df,
            'columns': list(df.columns),
            'metadata': metadata,
            'file_type': 'mpt',
            'technique': metadata.technique,
        }

        # Extract common columns
        column_mapping = {
            'time': [MptColumn.TIME_S, 'time/s'],
            'voltage': [MptColumn.EWE_V, 'Ewe/V', '<Ewe>/V'],
            'current': [MptColumn.I_MA, 'I/mA', '<I>/mA'],
            'capacity': [MptColumn.CAPACITY_MA_H, 'Capacity/mA.h'],
            'ns': [MptColumn.NS, 'Ns'],
            'ns_changes': [MptColumn.NS_CHANGES, 'Ns changes'],
            'cycle_number': [MptColumn.CYCLE_NUMBER, 'cycle number'],
            'half_cycle': [MptColumn.HALF_CYCLE, 'half cycle'],
            'ox_red': [MptColumn.OX_RED, 'ox/red'],
            'q_charge': [MptColumn.Q_CHARGE_MA_H, 'Q charge/mA.h'],
            'q_discharge': [MptColumn.Q_DISCHARGE_MA_H, 'Q discharge/mA.h'],
            # EIS columns
            'freq': [MptColumn.FREQ_HZ, 'freq/Hz'],
            're_z': [MptColumn.RE_Z_OHM, 'Re(Z)/Ohm'],
            'im_z': [MptColumn.IM_Z_OHM, '-Im(Z)/Ohm'],
            'z_abs': [MptColumn.Z_OHM, '|Z|/Ohm'],
            'phase_z': [MptColumn.PHASE_Z_DEG, 'Phase(Z)/deg'],
        }

        for key, col_names in column_mapping.items():
            for col_name in col_names:
                if col_name in df.columns:
                    result[key] = df[col_name].values
                    break

        # Parse cycles
        cycles = _parse_cycles_from_mpt(result)
        if cycles:
            result['cycles'] = cycles

        return result, None

    except Exception as e:
        return None, f"Error loading MPT file: {str(e)}"


def _parse_cycles_from_mpt(data: Dict) -> List[Dict]:
    """
    Parse cycle information from MPT data

    Uses half_cycle column if available, otherwise falls back to
    cycle_number or Ns columns.

    Also splits on current=0 (relaxation periods) and excludes
    data points where current is effectively zero.
    """
    cycles = []

    # Method 1: Use half_cycle column (most accurate for GCPL)
    if 'half_cycle' in data and data['half_cycle'] is not None:
        half_cycles = data['half_cycle']
        unique_half_cycles = sorted(set(half_cycles))

        for hc_num in unique_half_cycles:
            mask = half_cycles == hc_num
            indices = np.where(mask)[0]

            if len(indices) < 10:  # Skip short artifacts
                continue

            start_idx = indices[0]
            end_idx = indices[-1] + 1

            # Calculate cycle number (full cycle = half_cycle // 2)
            cycle_num = int(hc_num) // 2

            cycle = _extract_cycle_data(data, start_idx, end_idx, cycle_num)
            if cycle:
                cycle['half_cycle'] = int(hc_num)
                _determine_charge_discharge(cycle)
                # Skip relaxation periods (current ~0)
                if not cycle.get('is_relaxation', False):
                    cycles.append(cycle)

        return cycles

    # Method 2: Use Ns column with diff() method (bioloader pattern)
    # Also add current=0 detection for additional splitting
    if 'ns' in data and data['ns'] is not None:
        ns = data['ns']

        # bioloader pattern: detect where Ns changes
        ns_diff = np.diff(ns, prepend=ns[0])
        ns_change_indices = set(np.where(ns_diff != 0)[0])

        # Also detect current=0 boundaries (relaxation periods)
        current_change_indices = set()
        if 'current' in data and data['current'] is not None:
            current = data['current']
            # Detect transitions to/from zero current
            # Consider "zero" as |current| < threshold (0.01 mA)
            is_zero = np.abs(current) < 0.01
            zero_transitions = np.diff(is_zero.astype(int))
            current_change_indices = set(np.where(zero_transitions != 0)[0] + 1)

        # Combine all boundaries
        all_change_indices = sorted(ns_change_indices | current_change_indices)

        if len(all_change_indices) > 0:
            boundaries = [0] + list(all_change_indices) + [len(ns)]

            half_cycle_counter = 0
            for i in range(len(boundaries) - 1):
                start_idx = boundaries[i]
                end_idx = boundaries[i + 1]

                if end_idx - start_idx < 10:
                    continue

                cycle_num = half_cycle_counter // 2
                cycle = _extract_cycle_data(data, start_idx, end_idx, cycle_num)
                if cycle:
                    cycle['half_cycle'] = half_cycle_counter
                    _determine_charge_discharge(cycle)

                    # Skip relaxation periods (current ~0)
                    if not cycle.get('is_relaxation', False):
                        cycles.append(cycle)
                        half_cycle_counter += 1

            return cycles

    # Method 2b: Use Ns changes column if available
    if 'ns_changes' in data and data['ns_changes'] is not None:
        ns_changes = data['ns_changes']
        ns_change_indices = set(np.where(ns_changes == 1)[0])

        # Also detect current=0 boundaries
        current_change_indices = set()
        if 'current' in data and data['current'] is not None:
            current = data['current']
            is_zero = np.abs(current) < 0.01
            zero_transitions = np.diff(is_zero.astype(int))
            current_change_indices = set(np.where(zero_transitions != 0)[0] + 1)

        all_change_indices = sorted(ns_change_indices | current_change_indices)

        if len(all_change_indices) > 0:
            boundaries = [0] + list(all_change_indices) + [len(ns_changes)]

            half_cycle_counter = 0
            for i in range(len(boundaries) - 1):
                start_idx = boundaries[i]
                end_idx = boundaries[i + 1]

                if end_idx - start_idx < 10:
                    continue

                cycle_num = half_cycle_counter // 2
                cycle = _extract_cycle_data(data, start_idx, end_idx, cycle_num)
                if cycle:
                    cycle['half_cycle'] = half_cycle_counter
                    _determine_charge_discharge(cycle)

                    if not cycle.get('is_relaxation', False):
                        cycles.append(cycle)
                        half_cycle_counter += 1

            return cycles

    # Method 3: Use cycle_number column
    if 'cycle_number' in data:
        cycle_numbers = data['cycle_number']
        unique_cycles = sorted(set(cycle_numbers))

        for cyc_num in unique_cycles:
            mask = cycle_numbers == cyc_num
            indices = np.where(mask)[0]

            if len(indices) < 10:
                continue

            start_idx = indices[0]
            end_idx = indices[-1] + 1

            cycle = _extract_cycle_data(data, start_idx, end_idx, int(cyc_num))
            if cycle:
                _determine_charge_discharge(cycle)
                cycles.append(cycle)

        return cycles

    # Method 4: Use ox_red to split charge/discharge
    if 'ox_red' in data and 'time' in data:
        ox_red = data['ox_red']
        ox_red_changes = np.where(np.diff(ox_red) != 0)[0] + 1

        if len(ox_red_changes) > 0:
            boundaries = [0] + list(ox_red_changes) + [len(ox_red)]

            half_cycle = 0
            for i in range(len(boundaries) - 1):
                start_idx = boundaries[i]
                end_idx = boundaries[i + 1]

                if end_idx - start_idx < 10:
                    continue

                cycle_num = half_cycle // 2
                cycle = _extract_cycle_data(data, start_idx, end_idx, cycle_num)
                if cycle:
                    cycle['half_cycle'] = half_cycle
                    # Use ox_red value to determine charge/discharge
                    avg_ox_red = np.mean(ox_red[start_idx:end_idx])
                    cycle['is_charge'] = avg_ox_red > 0.5
                    cycle['is_discharge'] = avg_ox_red < 0.5
                    cycles.append(cycle)

                half_cycle += 1

            return cycles

    # No cycle information available - return single pseudo-cycle
    if 'time' in data and 'voltage' in data:
        cycle = _extract_cycle_data(data, 0, len(data['time']), 0)
        if cycle:
            cycles.append(cycle)

    return cycles


def _extract_cycle_data(data: Dict, start_idx: int, end_idx: int, cycle_num: int) -> Optional[Dict]:
    """Extract data for a single cycle"""
    cycle = {
        'cycle_number': cycle_num,
        'start_idx': start_idx,
        'end_idx': end_idx,
    }

    # Extract arrays for this cycle
    for key in ['time', 'voltage', 'current', 'capacity', 'q_charge', 'q_discharge']:
        if key in data and data[key] is not None:
            arr = data[key][start_idx:end_idx]
            if len(arr) > 0:
                cycle[key] = arr

    # Skip if no voltage data
    if 'voltage' not in cycle or len(cycle['voltage']) == 0:
        return None

    return cycle


def _determine_charge_discharge(cycle: Dict) -> None:
    """Determine if cycle is charge or discharge based on current"""
    if 'current' in cycle and len(cycle['current']) > 0:
        avg_current = np.mean(cycle['current'])
        avg_abs_current = np.mean(np.abs(cycle['current']))

        # Skip relaxation process (very low current)
        if avg_abs_current < 0.01:
            cycle['is_charge'] = False
            cycle['is_discharge'] = False
            cycle['is_relaxation'] = True
        else:
            cycle['is_charge'] = avg_current > 0
            cycle['is_discharge'] = avg_current < 0
            cycle['is_relaxation'] = False
    else:
        cycle['is_charge'] = False
        cycle['is_discharge'] = False


def detect_technique(metadata: MptMetadata) -> str:
    """
    Detect measurement technique from metadata

    Returns
    -------
    technique : str
        One of: GCPL, PEIS, OCV, CV, CA, CP, MB, CCCV, Unknown
    """
    technique = metadata.technique.upper() if metadata.technique else ""

    if "GCPL" in technique or "GALVANOSTATIC CYCLING" in technique:
        return "GCPL"
    elif "PEIS" in technique or "POTENTIO" in technique and "IMPEDANCE" in technique:
        return "PEIS"
    elif "GEIS" in technique or "GALVANO" in technique and "IMPEDANCE" in technique:
        return "GEIS"
    elif "OCV" in technique or "OPEN CIRCUIT" in technique:
        return "OCV"
    elif "CV" in technique or "CYCLIC VOLTAMMETRY" in technique:
        return "CV"
    elif "CA" in technique or "CHRONOAMPEROMETRY" in technique:
        return "CA"
    elif "CP" in technique or "CHRONOPOTENTIOMETRY" in technique:
        return "CP"
    elif "MB" in technique or "MODULO BAT" in technique:
        return "MB"
    elif "CC" in technique or "CCCV" in technique:
        return "CCCV"
    elif "LSV" in technique or "LINEAR SWEEP" in technique:
        return "LSV"
    else:
        return "Unknown"


def load_mpt_with_cycles(filepath: str | Path) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Load MPT file and ensure cycle data is properly parsed

    This is a convenience function that loads the file and verifies
    cycle information is available.

    Parameters
    ----------
    filepath : str or Path
        Path to the MPT file

    Returns
    -------
    data : dict or None
        Data dictionary with 'cycles' key populated
    error : str or None
        Error message if failed
    """
    data, error = load_mpt_file(filepath)

    if error:
        return None, error

    if not data:
        return None, "No data loaded"

    # Verify cycles were parsed
    if 'cycles' not in data or len(data['cycles']) == 0:
        # Try alternative parsing methods
        cycles = _parse_cycles_from_mpt(data)
        if cycles:
            data['cycles'] = cycles
        else:
            # Create a single pseudo-cycle from raw data
            if 'time' in data and 'voltage' in data:
                data['cycles'] = [{
                    'cycle_number': 0,
                    'time': data['time'],
                    'voltage': data['voltage'],
                    'current': data.get('current', np.zeros_like(data['voltage'])),
                    'capacity': data.get('capacity', np.zeros_like(data['voltage'])),
                    'is_charge': False,
                    'is_discharge': True,
                }]

    return data, None


def is_impedance_data(data: Dict) -> bool:
    """
    Detect if loaded data contains impedance (EIS) measurements

    Parameters
    ----------
    data : dict
        Data dictionary from load_mpt_file

    Returns
    -------
    is_eis : bool
        True if data contains impedance data
    """
    # Check technique type
    technique = data.get('technique', '').upper()
    if 'PEIS' in technique or 'GEIS' in technique or 'EIS' in technique:
        return True

    # Check for EIS columns
    eis_columns = ['freq', 're_z', 'im_z', 'z_abs', 'phase_z']
    has_eis_cols = any(col in data and data[col] is not None for col in eis_columns)

    # Check raw_df for EIS columns
    if 'raw_df' in data:
        df = data['raw_df']
        eis_df_cols = ['freq/Hz', 'Re(Z)/Ohm', '-Im(Z)/Ohm', '|Z|/Ohm', 'Phase(Z)/deg']
        has_eis_cols = has_eis_cols or any(col in df.columns for col in eis_df_cols)

    return has_eis_cols


def is_charge_discharge_data(data: Dict) -> bool:
    """
    Detect if loaded data contains charge-discharge (GCPL) measurements

    Parameters
    ----------
    data : dict
        Data dictionary from load_mpt_file

    Returns
    -------
    is_cd : bool
        True if data contains charge-discharge data
    """
    # Check technique type
    technique = data.get('technique', '').upper()
    if 'GCPL' in technique or 'CCCV' in technique or 'MB' in technique:
        return True

    # Check for CD columns
    cd_columns = ['time', 'voltage', 'current', 'capacity']
    has_cd_cols = sum(1 for col in cd_columns if col in data and data[col] is not None) >= 3

    # Check for cycle data
    has_cycles = 'cycles' in data and len(data.get('cycles', [])) > 0

    return has_cd_cols or has_cycles


def detect_data_type(data: Dict) -> str:
    """
    Detect the type of measurement data

    Parameters
    ----------
    data : dict
        Data dictionary from load_mpt_file

    Returns
    -------
    data_type : str
        One of: 'EIS', 'CD' (charge-discharge), 'CV', 'OCV', 'Unknown'
    """
    if is_impedance_data(data):
        return 'EIS'

    if is_charge_discharge_data(data):
        return 'CD'

    technique = data.get('technique', '').upper()
    if 'CV' in technique or 'CYCLIC' in technique:
        return 'CV'
    if 'OCV' in technique or 'OPEN CIRCUIT' in technique:
        return 'OCV'

    return 'Unknown'
