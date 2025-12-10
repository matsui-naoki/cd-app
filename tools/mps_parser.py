"""
BioLogic MPS (Settings) file parser
Parses measurement sequence and auto-detects related data files
"""

import os
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TechniqueInfo:
    """Information about a single technique in the measurement sequence"""
    index: int
    name: str
    short_name: str  # OCV, PEIS, GCPL, etc.
    parameters: Dict[str, Any]
    data_file: Optional[str] = None
    has_data: bool = False


@dataclass
class MeasurementSession:
    """Complete measurement session parsed from mps file"""
    mps_path: str
    base_name: str
    folder: str
    num_techniques: int
    techniques: List[TechniqueInfo]
    sample_info: Dict[str, Any]
    device_info: Dict[str, Any]
    data_files: Dict[str, str]  # technique_key -> file_path


# Technique name mappings
TECHNIQUE_MAP = {
    'Open Circuit Voltage': 'OCV',
    'Potentio Electrochemical Impedance Spectroscopy': 'PEIS',
    'Galvanostatic Electrochemical Impedance Spectroscopy': 'GEIS',
    'Galvanostatic Cycling with Potential Limitation': 'GCPL',
    'Constant Current': 'CC',
    'Constant Voltage': 'CV',
    'Cyclic Voltammetry': 'CV',
    'Linear Sweep Voltammetry': 'LSV',
    'Chronoamperometry / Chronocoulometry': 'CA',
    'Chronopotentiometry': 'CP',
    'Modulo Bat': 'MB',
    'Loop': 'Loop',
    'Wait': 'Wait',
}


def parse_mps_file(mps_path: str) -> Optional[MeasurementSession]:
    """
    Parse BioLogic .mps settings file

    Parameters
    ----------
    mps_path : str
        Path to the .mps file

    Returns
    -------
    session : MeasurementSession or None
        Parsed measurement session
    """
    if not os.path.exists(mps_path):
        return None

    try:
        with open(mps_path, 'r', encoding='latin-1') as f:
            content = f.read()

        lines = content.split('\n')

        # Extract basic info
        folder = os.path.dirname(mps_path)
        base_name = os.path.splitext(os.path.basename(mps_path))[0]

        # Parse number of techniques
        num_techniques = 0
        for line in lines:
            if 'Number of linked techniques' in line:
                match = re.search(r'(\d+)', line)
                if match:
                    num_techniques = int(match.group(1))
                break

        # Parse sample info
        sample_info = _parse_sample_info(lines)

        # Parse device info
        device_info = _parse_device_info(lines)

        # Parse techniques
        techniques = _parse_techniques(lines)

        # Find related data files
        data_files = _find_data_files(folder, base_name, techniques)

        # Update technique info with data file paths
        for tech in techniques:
            key = f"{tech.index:02d}_{tech.short_name}"
            if key in data_files:
                tech.data_file = data_files[key]
                tech.has_data = True

        return MeasurementSession(
            mps_path=mps_path,
            base_name=base_name,
            folder=folder,
            num_techniques=num_techniques,
            techniques=techniques,
            sample_info=sample_info,
            device_info=device_info,
            data_files=data_files
        )

    except Exception as e:
        print(f"Error parsing mps file: {e}")
        return None


def _parse_sample_info(lines: List[str]) -> Dict[str, Any]:
    """Extract sample information from mps file"""
    info = {}

    patterns = {
        'mass_mg': (r'Mass of active material\s*:\s*([\d.]+)\s*(mg|g)', lambda m: float(m.group(1)) * (1000 if m.group(2) == 'g' else 1)),
        'electrode_area_cm2': (r'Electrode surface area\s*:\s*([\d.]+)\s*cm', lambda m: float(m.group(1))),
        'electrode_material': (r'Electrode material\s*:\s*(.+)', lambda m: m.group(1).strip()),
        'electrolyte': (r'Electrolyte\s*:\s*(.+)', lambda m: m.group(1).strip()),
        'comments': (r'Comments\s*:\s*(.+)', lambda m: m.group(1).strip()),
        'molecular_weight': (r'Molecular weight.*:\s*([\d.]+)', lambda m: float(m.group(1))),
        'battery_capacity_Ah': (r'Battery capacity\s*:\s*([\d.]+)\s*A\.h', lambda m: float(m.group(1))),
    }

    for line in lines:
        for key, (pattern, extractor) in patterns.items():
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                try:
                    info[key] = extractor(match)
                except:
                    pass

    return info


def _parse_device_info(lines: List[str]) -> Dict[str, Any]:
    """Extract device information from mps file"""
    info = {}

    patterns = {
        'device': (r'Device\s*:\s*(.+)', lambda m: m.group(1).strip()),
        'software_version': (r'EC-LAB.*v([\d.]+)', lambda m: m.group(1)),
        'firmware_version': (r'Internet server v([\d.]+)', lambda m: m.group(1)),
        'channel': (r'Channel\s*:\s*(.+)', lambda m: m.group(1).strip()),
        'electrode_connection': (r'Electrode connection\s*:\s*(.+)', lambda m: m.group(1).strip()),
    }

    for line in lines:
        for key, (pattern, extractor) in patterns.items():
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                try:
                    info[key] = extractor(match)
                except:
                    pass

    return info


def _parse_techniques(lines: List[str]) -> List[TechniqueInfo]:
    """Parse technique definitions from mps file"""
    techniques = []
    current_technique = None
    current_params = {}

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Check for technique header
        match = re.match(r'Technique\s*:\s*(\d+)', line)
        if match:
            # Save previous technique
            if current_technique is not None:
                techniques.append(current_technique)

            tech_index = int(match.group(1))

            # Next line should be technique name
            if i + 1 < len(lines):
                tech_name = lines[i + 1].strip()
                short_name = TECHNIQUE_MAP.get(tech_name, tech_name[:4].upper())

                current_technique = TechniqueInfo(
                    index=tech_index,
                    name=tech_name,
                    short_name=short_name,
                    parameters={}
                )
                current_params = {}
                i += 1

        elif current_technique is not None and line:
            # Parse parameter line
            # Format: "param_name    value1    value2    ..."
            parts = re.split(r'\s{2,}', line)
            if len(parts) >= 2:
                param_name = parts[0].strip()
                param_values = [p.strip() for p in parts[1:] if p.strip()]
                if param_values:
                    current_params[param_name] = param_values[0] if len(param_values) == 1 else param_values

            current_technique.parameters = current_params

        i += 1

    # Save last technique
    if current_technique is not None:
        techniques.append(current_technique)

    return techniques


def _find_data_files(folder: str, base_name: str, techniques: List[TechniqueInfo]) -> Dict[str, str]:
    """Find related .mpr data files in the same folder"""
    data_files = {}

    if not os.path.exists(folder):
        return data_files

    # List all files in folder
    for filename in os.listdir(folder):
        if not filename.endswith('.mpr'):
            continue

        # Check if filename matches pattern: basename_XX_TECHNIQUE_*.mpr
        if not filename.startswith(base_name):
            continue

        # Extract technique index and type from filename
        # Pattern: base_name_01_OCV_*.mpr
        pattern = re.escape(base_name) + r'_(\d+)_([A-Z]+)_'
        match = re.search(pattern, filename)

        if match:
            tech_index = int(match.group(1))
            tech_type = match.group(2)

            key = f"{tech_index:02d}_{tech_type}"
            data_files[key] = os.path.join(folder, filename)

    return data_files


def get_technique_summary(session: MeasurementSession) -> List[Dict[str, Any]]:
    """
    Get summary table of techniques in measurement session

    Returns list of dicts for DataFrame creation
    """
    rows = []

    for tech in session.techniques:
        row = {
            'No.': tech.index,
            'Technique': tech.short_name,
            'Full Name': tech.name,
            'Has Data': 'Yes' if tech.has_data else 'No',
            'Data File': os.path.basename(tech.data_file) if tech.data_file else '-',
        }

        # Add key parameters based on technique type
        if tech.short_name == 'GCPL':
            if 'Is' in tech.parameters:
                row['Current'] = tech.parameters.get('Is', '-')
            if 'EM (V)' in tech.parameters:
                row['E limit'] = tech.parameters.get('EM (V)', '-')

        elif tech.short_name == 'PEIS':
            if 'fi' in tech.parameters:
                row['Freq. range'] = f"{tech.parameters.get('fi', '?')} - {tech.parameters.get('ff', '?')}"
            if 'Va (mV)' in tech.parameters:
                row['Amplitude'] = f"{tech.parameters.get('Va (mV)', '-')} mV"

        elif tech.short_name == 'OCV':
            if 'tR (h:m:s)' in tech.parameters:
                row['Duration'] = tech.parameters.get('tR (h:m:s)', '-')

        rows.append(row)

    return rows


def load_gcpl_data_from_session(
    session: MeasurementSession,
    technique_indices: List[int] = None
) -> Dict[str, Any]:
    """
    Load GCPL (charge-discharge) data from session

    Parameters
    ----------
    session : MeasurementSession
        Parsed measurement session
    technique_indices : list of int, optional
        Specific technique indices to load. If None, load all GCPL.

    Returns
    -------
    data : dict
        Combined charge-discharge data
    """
    from .data_loader import load_biologic_mpr
    import numpy as np

    # Find GCPL techniques
    gcpl_techs = [t for t in session.techniques if t.short_name == 'GCPL' and t.has_data]

    if technique_indices is not None:
        gcpl_techs = [t for t in gcpl_techs if t.index in technique_indices]

    if not gcpl_techs:
        return None

    # Load and combine data
    all_time = []
    all_voltage = []
    all_current = []
    all_capacity = []
    all_half_cycle = []
    technique_labels = []

    time_offset = 0

    for tech in gcpl_techs:
        if not tech.data_file:
            continue

        data, error = load_biologic_mpr(tech.data_file)

        if error or data is None:
            continue

        if 'time' in data:
            # Adjust time to be continuous
            t = data['time'] + time_offset
            all_time.extend(t)
            time_offset = t[-1] if len(t) > 0 else time_offset

        if 'voltage' in data:
            all_voltage.extend(data['voltage'])

        if 'current' in data:
            all_current.extend(data['current'])
        elif 'control/V/mA' in data.get('raw_df', {}).columns if hasattr(data.get('raw_df'), 'columns') else False:
            all_current.extend(data['raw_df']['control/V/mA'].values)

        if 'capacity' in data:
            all_capacity.extend(data['capacity'])

        if 'half_cycle' in data:
            all_half_cycle.extend(data['half_cycle'])

        technique_labels.extend([tech.index] * len(data.get('time', [])))

    if not all_time:
        return None

    combined = {
        'time': np.array(all_time),
        'voltage': np.array(all_voltage) if all_voltage else None,
        'current': np.array(all_current) if all_current else None,
        'capacity': np.array(all_capacity) if all_capacity else None,
        'half_cycle': np.array(all_half_cycle) if all_half_cycle else None,
        'technique_index': np.array(technique_labels) if technique_labels else None,
        'source_session': session,
        'cycles': [],
    }

    # Parse cycles from half_cycle
    if combined['half_cycle'] is not None:
        combined['cycles'] = _parse_cycles_from_half_cycle(combined)

    return combined


def _parse_cycles_from_half_cycle(data: Dict) -> List[Dict]:
    """Parse charge/discharge cycles from half_cycle column"""
    import numpy as np

    cycles = []
    half_cycle = data['half_cycle']

    if half_cycle is None or len(half_cycle) == 0:
        return cycles

    unique_half_cycles = np.unique(half_cycle[~np.isnan(half_cycle)])

    # Group half-cycles into full cycles (charge + discharge)
    cycle_num = 0

    for hc in unique_half_cycles:
        mask = half_cycle == hc
        indices = np.where(mask)[0]

        if len(indices) == 0:
            continue

        start_idx = indices[0]
        end_idx = indices[-1] + 1

        cycle = {
            'cycle_number': cycle_num,
            'half_cycle': int(hc),
            'start_idx': start_idx,
            'end_idx': end_idx,
        }

        # Extract data for this half-cycle
        for key in ['time', 'voltage', 'current', 'capacity']:
            if key in data and data[key] is not None:
                cycle[key] = data[key][start_idx:end_idx]

        # Skip short cycles (< 10 data points) - likely transition artifacts
        n_points = len(cycle.get('voltage', []))
        if n_points < 10:
            continue

        # Determine if charge or discharge based on current sign
        if 'current' in cycle and len(cycle['current']) > 0:
            avg_current = np.mean(cycle['current'])
            avg_abs_current = np.mean(np.abs(cycle['current']))

            # Skip relaxation process (low current < 0.01 mA)
            if avg_abs_current < 0.01:
                continue

            cycle['is_charge'] = avg_current > 0
            cycle['is_discharge'] = avg_current < 0

            # Calculate capacity from current integration
            if 'time' in cycle and len(cycle['time']) > 1:
                time = cycle['time']
                current = cycle['current']
                dt = np.diff(time)

                # Total capacity in mAh
                total_cap = np.sum(np.abs(current[:-1]) * dt) / 3600
                cycle['capacity_mAh'] = total_cap

                if cycle['is_charge']:
                    cycle['capacity_charge_mAh'] = total_cap
                elif cycle['is_discharge']:
                    cycle['capacity_discharge_mAh'] = total_cap

        # Also use capacity column if available
        if 'capacity' in cycle and len(cycle['capacity']) > 0:
            cap = np.abs(cycle['capacity'])
            cap_from_col = np.max(cap) - np.min(cap)
            if cap_from_col > 0:
                cycle['capacity_mAh'] = cap_from_col
                if cycle.get('is_charge'):
                    cycle['capacity_charge_mAh'] = cap_from_col
                elif cycle.get('is_discharge'):
                    cycle['capacity_discharge_mAh'] = cap_from_col

        cycles.append(cycle)

        # Increment cycle number every 2 half-cycles
        if int(hc) % 2 == 1:
            cycle_num += 1

    return cycles


def load_peis_data_from_session(
    session: MeasurementSession,
    technique_indices: List[int] = None
) -> List[Dict[str, Any]]:
    """
    Load PEIS (EIS) data from session

    Parameters
    ----------
    session : MeasurementSession
        Parsed measurement session
    technique_indices : list of int, optional
        Specific technique indices to load. If None, load all PEIS.

    Returns
    -------
    eis_data_list : list of dict
        List of EIS data dictionaries
    """
    from .mpr_parser import load_mpr_to_dict

    peis_techs = [t for t in session.techniques if t.short_name == 'PEIS' and t.has_data]

    if technique_indices is not None:
        peis_techs = [t for t in peis_techs if t.index in technique_indices]

    eis_data_list = []

    for tech in peis_techs:
        if not tech.data_file:
            continue

        data, error = load_mpr_to_dict(tech.data_file)

        if error or data is None:
            print(f"Could not load PEIS from {tech.data_file}: {error}")
            continue

        eis_data = {
            'technique_index': tech.index,
            'file': os.path.basename(tech.data_file),
        }

        # Extract EIS columns
        if 'freq' in data:
            eis_data['freq'] = data['freq']
        if 'Z_real' in data:
            eis_data['Z_real'] = data['Z_real']
        if 'Z_imag' in data:
            eis_data['Z_imag'] = data['Z_imag']
        if 'Z_mag' in data:
            eis_data['Z_mag'] = data['Z_mag']
        if 'Z_phase' in data:
            eis_data['Z_phase'] = data['Z_phase']

        if 'freq' in eis_data and 'Z_real' in eis_data:
            eis_data_list.append(eis_data)

    return eis_data_list
