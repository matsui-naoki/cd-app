"""
Universal BioLogic MPR file parser
Reads binary MPR files directly without external dependencies

Based on EC-Lab file format analysis and eclabfiles/galvani references
"""

import os
import re
import struct
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd


# Column ID to (numpy_dtype, column_name, unit) mapping
# Based on galvani and eclabfiles mappings
COLUMN_MAP = {
    # Time and control
    4: ('<f8', 'time/s', 's'),
    5: ('<f4', 'control/V/mA', 'V/mA'),
    6: ('<f4', 'Ewe/V', 'V'),
    7: ('<f8', 'dq/mA.h', 'mA.h'),
    8: ('<f4', 'I/mA', 'mA'),
    9: ('<f4', 'Ece/V', 'V'),
    11: ('<f8', '<I>/mA', 'mA'),
    13: ('<f8', '(Q-Qo)/mA.h', 'mA.h'),
    16: ('<f4', 'Analog IN 1/V', 'V'),
    17: ('<f4', 'Analog IN 2/V', 'V'),
    19: ('<f4', 'control/V', 'V'),
    20: ('<f4', 'control/mA', 'mA'),
    23: ('<f8', 'dQ/mA.h', 'mA.h'),
    24: ('<f8', 'cycle number', ''),
    26: ('<f4', 'Rapp/Ohm', 'Ohm'),
    27: ('<f4', 'Ewe-Ece/V', 'V'),

    # EIS columns
    32: ('<f4', 'freq/Hz', 'Hz'),
    33: ('<f4', '|Ewe|/V', 'V'),
    34: ('<f4', '|I|/A', 'A'),
    35: ('<f4', 'Phase(Z)/deg', 'deg'),
    36: ('<f4', '|Z|/Ohm', 'Ohm'),
    37: ('<f4', 'Re(Z)/Ohm', 'Ohm'),
    38: ('<f4', '-Im(Z)/Ohm', 'Ohm'),
    39: ('<u2', 'I Range', ''),

    # Resistance/Power
    69: ('<f4', 'R/Ohm', 'Ohm'),
    70: ('<f4', 'P/W', 'W'),
    74: ('<f8', '|Energy|/W.h', 'W.h'),
    75: ('<f4', 'Analog OUT/V', 'V'),
    76: ('<f4', '<I>/mA', 'mA'),
    77: ('<f4', '<Ewe>/V', 'V'),
    78: ('<f4', 'Cs-2/µF-2', 'µF-2'),

    # Counter electrode EIS
    96: ('<f4', '|Ece|/V', 'V'),
    98: ('<f4', 'Phase(Zce)/deg', 'deg'),
    99: ('<f4', '|Zce|/Ohm', 'Ohm'),
    100: ('<f4', 'Re(Zce)/Ohm', 'Ohm'),
    101: ('<f4', '-Im(Zce)/Ohm', 'Ohm'),

    # Energy/Capacity
    123: ('<f8', 'Energy charge/W.h', 'W.h'),
    124: ('<f8', 'Energy discharge/W.h', 'W.h'),
    125: ('<f8', 'Capacitance charge/µF', 'µF'),
    126: ('<f8', 'Capacitance discharge/µF', 'µF'),
    131: ('<u2', 'Ns', ''),

    # Additional
    163: ('<f4', '|Estack|/V', 'V'),
    168: ('<f4', 'Rcmp/Ohm', 'Ohm'),
    169: ('<f4', 'Cs/µF', 'µF'),
    172: ('<f4', 'Cp/µF', 'µF'),
    173: ('<f4', 'Cp-2/µF-2', 'µF-2'),
    174: ('<f4', '<Ewe>/V', 'V'),
    178: ('<f4', '(Q-Qo)/C', 'C'),
    179: ('<f4', 'dQ/C', 'C'),

    # Additional columns found in EC-Lab files
    65: ('<u1', 'Ns_2', ''),  # Secondary Ns counter

    # Cycle info
    211: ('<f8', 'Q charge/discharge/mA.h', 'mA.h'),
    212: ('<u4', 'half cycle', ''),
    213: ('<u4', 'z cycle', ''),
    214: ('<f4', 'Ewe-Ece/V', 'V'),
    215: ('<f4', 'dQ/mA.h', 'mA.h'),  # Differential capacity (alternative ID)
    216: ('<f4', 'dq/mA.h', 'mA.h'),

    # THD/NSD/NSR
    217: ('<f4', 'THD Ewe/%', '%'),
    218: ('<f4', 'THD I/%', '%'),
    220: ('<f4', 'NSD Ewe/%', '%'),
    221: ('<f4', 'NSD I/%', '%'),
    223: ('<f4', 'NSR Ewe/%', '%'),
    224: ('<f4', 'NSR I/%', '%'),

    # Harmonics
    230: ('<f4', '|Ewe h2|/V', 'V'),
    231: ('<f4', '|Ewe h3|/V', 'V'),
    232: ('<f4', '|Ewe h4|/V', 'V'),
    233: ('<f4', '|Ewe h5|/V', 'V'),
    234: ('<f4', '|Ewe h6|/V', 'V'),
    235: ('<f4', '|Ewe h7|/V', 'V'),
    236: ('<f4', '|I h2|/A', 'A'),
    237: ('<f4', '|I h3|/A', 'A'),
    238: ('<f4', '|I h4|/A', 'A'),
    239: ('<f4', '|I h5|/A', 'A'),
    240: ('<f4', '|I h6|/A', 'A'),
    241: ('<f4', '|I h7|/A', 'A'),
    242: ('<f4', '|E2|/V', 'V'),

    # Multi-channel EIS
    271: ('<f4', 'Phase(Z1)/deg', 'deg'),
    272: ('<f4', 'Phase(Z2)/deg', 'deg'),
    301: ('<f4', '|Z1|/Ohm', 'Ohm'),
    302: ('<f4', '|Z2|/Ohm', 'Ohm'),
    331: ('<f4', 'Re(Z1)/Ohm', 'Ohm'),
    332: ('<f4', 'Re(Z2)/Ohm', 'Ohm'),
    361: ('<f4', '-Im(Z1)/Ohm', 'Ohm'),
    362: ('<f4', '-Im(Z2)/Ohm', 'Ohm'),
    391: ('<f4', '<E1>/V', 'V'),
    392: ('<f4', '<E2>/V', 'V'),

    # Stack EIS
    422: ('<f4', 'Phase(Zstack)/deg', 'deg'),
    423: ('<f4', '|Zstack|/Ohm', 'Ohm'),
    424: ('<f4', 'Re(Zstack)/Ohm', 'Ohm'),
    425: ('<f4', '-Im(Zstack)/Ohm', 'Ohm'),
    426: ('<f4', '<Estack>/V', 'V'),
    430: ('<f4', 'Phase(Zwe-ce)/deg', 'deg'),
    431: ('<f4', '|Zwe-ce|/Ohm', 'Ohm'),
    432: ('<f4', 'Re(Zwe-ce)/Ohm', 'Ohm'),
    433: ('<f4', '-Im(Zwe-ce)/Ohm', 'Ohm'),
    434: ('<f4', '(Q-Qo)/C', 'C'),
    435: ('<f4', 'dQ/C', 'C'),
    438: ('<f8', 'step time/s', 's'),
    441: ('<f4', '<Ecv>/V', 'V'),

    # Temperature
    462: ('<f4', 'Temperature/°C', '°C'),

    # Extended cycle info (newer versions)
    467: ('<f8', 'Q charge/discharge/mA.h', 'mA.h'),
    468: ('<u4', 'half cycle', ''),
    469: ('<u4', 'z cycle', ''),
    471: ('<f4', '<Ece>/V', 'V'),
    473: ('<f4', 'THD Ewe/%', '%'),
    474: ('<f4', 'THD I/%', '%'),
    476: ('<f4', 'NSD Ewe/%', '%'),
    477: ('<f4', 'NSD I/%', '%'),
    479: ('<f4', 'NSR Ewe/%', '%'),
    480: ('<f4', 'NSR I/%', '%'),
    486: ('<f4', '|Ewe h2|/V', 'V'),
    487: ('<f4', '|Ewe h3|/V', 'V'),
    488: ('<f4', '|Ewe h4|/V', 'V'),
    489: ('<f4', '|Ewe h5|/V', 'V'),
    490: ('<f4', '|Ewe h6|/V', 'V'),
    491: ('<f4', '|Ewe h7|/V', 'V'),
    492: ('<f4', '|I h2|/A', 'A'),
    493: ('<f4', '|I h3|/A', 'A'),
    494: ('<f4', '|I h4|/A', 'A'),
    495: ('<f4', '|I h5|/A', 'A'),
    496: ('<f4', '|I h6|/A', 'A'),
    497: ('<f4', '|I h7|/A', 'A'),
    498: ('<f8', 'Q charge/mA.h', 'mA.h'),
    499: ('<f8', 'Q discharge/mA.h', 'mA.h'),
    500: ('<f8', 'step time/s', 's'),
    501: ('<f8', 'Efficiency/%', '%'),
    502: ('<f8', 'Capacity/mA.h', 'mA.h'),
    505: ('<f4', 'Rdc/Ohm', 'Ohm'),
    509: ('<u1', 'Acir/Dcir Control', ''),
}

# Flag column IDs (packed in single byte)
FLAG_COLUMNS = {
    1: ('mode', 0x03, np.uint8),
    2: ('ox/red', 0x04, np.bool_),
    3: ('error', 0x08, np.bool_),
    21: ('control changes', 0x10, np.bool_),
    31: ('Ns changes', 0x20, np.bool_),
}


class MPRData:
    """Container for parsed MPR data"""
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.columns: List[str] = []
        self.n_points: int = 0
        self.technique: str = ''
        self.settings: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {}


def parse_mpr(file_path: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Parse BioLogic MPR binary file

    Uses galvani library as primary parser, falls back to custom parser if needed.

    Parameters
    ----------
    file_path : str
        Path to the .mpr file

    Returns
    -------
    df : pandas.DataFrame or None
        Parsed data as DataFrame
    error : str or None
        Error message if parsing failed
    """
    if not os.path.exists(file_path):
        return None, f"File not found: {file_path}"

    # Try galvani first (more reliable for most files)
    try:
        from galvani import BioLogic
        mpr = BioLogic.MPRfile(file_path)
        df = pd.DataFrame(mpr.data)
        if len(df) > 0:
            return df, None
    except Exception as galvani_error:
        # galvani failed, try custom parser
        pass

    # Fall back to custom parser
    try:
        with open(file_path, 'rb') as f:
            content = f.read()

        # Verify file header
        if not content[:21].decode('latin-1', errors='ignore').startswith('BIO-LOGIC MODULAR'):
            return None, "Not a valid BioLogic MPR file"

        # Find and parse modules
        modules = _find_modules(content)

        if not modules:
            return None, "No data modules found in file"

        # Find VMP data module
        data_module = None
        for mod in modules:
            if 'data' in mod['short_name'].lower():
                data_module = mod
                break

        if not data_module:
            return None, "No VMP data module found"

        # Parse data module
        df = _parse_data_module(content, data_module)

        if df is None or len(df) == 0:
            return None, f"Could not parse data from file (galvani error: {galvani_error})"

        return df, None

    except Exception as e:
        return None, f"Error parsing MPR file: {str(e)}"


def _find_modules(content: bytes) -> List[Dict[str, Any]]:
    """Find all MODULE sections in the file"""
    modules = []

    # Find all MODULE markers
    pattern = re.compile(b'MODULE')

    for match in pattern.finditer(content):
        pos = match.start()

        # Parse module header
        # MODULE (6) + short_name (10) + long_name (25) + padding + version/date/length (variable)
        try:
            short_name = content[pos+6:pos+16].decode('latin-1', errors='ignore').rstrip('\x00 ')
            long_name = content[pos+16:pos+41].decode('latin-1', errors='ignore').rstrip('\x00 ')

            # Length is stored at different positions depending on version
            # Try common positions
            length = None
            for len_offset in [45, 49, 58, 60, 62]:
                if pos + len_offset + 4 <= len(content):
                    test_len = struct.unpack('<I', content[pos+len_offset:pos+len_offset+4])[0]
                    # Sanity check - length should be reasonable
                    if 100 < test_len < len(content):
                        length = test_len
                        break

            # For VMP data module, header ends where column count starts
            # This is typically around offset 69 from module start
            # Search for valid n_cols pattern (small int followed by column IDs)
            header_end = pos + 69  # Default approximate

            modules.append({
                'pos': pos,
                'short_name': short_name,
                'long_name': long_name,
                'length': length,
                'header_end': header_end
            })
        except Exception:
            continue

    return modules


def _parse_data_module(content: bytes, module: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """Parse the VMP data module to extract measurement data"""

    # Search for column definitions after module header
    search_start = module['header_end']
    search_end = min(search_start + 300, len(content))

    # Find column count and IDs
    col_info = _find_column_info(content, search_start, search_end)

    if col_info is None:
        return None

    n_cols, col_ids, col_start = col_info

    # Build numpy dtype from column IDs
    # Note: EC-Lab MPR files may or may not have flag bytes depending on version
    dtype_fields = []
    flag_ids = [cid for cid in col_ids if cid in FLAG_COLUMNS]

    # Add flags byte if any flag columns present
    if flag_ids:
        dtype_fields.append(('flags', '<u1'))

    for cid in col_ids:
        if cid in FLAG_COLUMNS:
            # Already handled above
            continue
        elif cid in COLUMN_MAP:
            dt, name, _ = COLUMN_MAP[cid]
            # Handle duplicate column names
            base_name = name
            counter = 2
            while any(f[0] == name for f in dtype_fields):
                name = f"{base_name}_{counter}"
                counter += 1
            dtype_fields.append((name, dt))
        else:
            # Skip unknown columns - galvani also skips them
            # This is safer than guessing the dtype
            continue

    if not dtype_fields:
        return None

    record_dtype = np.dtype(dtype_fields)
    record_size = record_dtype.itemsize

    # Column info ends here
    col_info_end = col_start + 2 + n_cols * 2

    # Find end of data (next MODULE)
    next_module = content.find(b'MODULE', col_info_end)
    data_end = next_module if next_module > 0 else len(content)

    # Find actual data start by searching for valid float data
    # There's often padding (zeros) between column info and actual data
    data_start = _find_data_start(content, col_info_end, data_end, record_dtype)

    if data_start is None:
        # Fall back to right after column info
        data_start = col_info_end

    # Calculate number of records
    data_length = data_end - data_start
    n_records = data_length // record_size

    if n_records < 1:
        return None

    # Read data
    try:
        data_bytes = content[data_start:data_start + n_records * record_size]
        records = np.frombuffer(data_bytes, dtype=record_dtype)
        df = pd.DataFrame(records)
        return df
    except Exception as e:
        print(f"Error reading data: {e}")
        return None


def _find_data_start(content: bytes, start: int, end: int, record_dtype: np.dtype) -> Optional[int]:
    """
    Find the actual start of data records by searching for valid float values

    BioLogic MPR files often have padding/metadata between column info and data
    """
    record_size = record_dtype.itemsize

    # Get first few field names and types to validate
    first_field = record_dtype.names[0]
    first_dtype = record_dtype.fields[first_field][0]

    # Known reasonable ranges for EC-Lab data
    # freq: 1e-3 to 1e8 Hz
    # Z: 1e-6 to 1e12 Ohm
    # time: 0 to 1e8 s
    # voltage: -100 to 100 V
    # current: -1e3 to 1e3 mA

    def is_reasonable_value(val, field_name):
        """Check if value is in reasonable range for electrochemical data"""
        if np.isnan(val) or np.isinf(val):
            return False

        field_lower = field_name.lower()

        if 'freq' in field_lower:
            return 1e-4 < abs(val) < 1e9
        elif 'z' in field_lower or 'ohm' in field_lower:
            return abs(val) < 1e12
        elif 'time' in field_lower:
            return 0 <= val < 1e8
        elif 'ewe' in field_lower or 'ece' in field_lower or '/v' in field_lower:
            return -100 < val < 100
        elif 'i' in field_lower and '/ma' in field_lower:
            return abs(val) < 1e6
        else:
            # Generic check - value should be in float32 reasonable range
            return abs(val) < 1e15 and abs(val) > 1e-15 if val != 0 else True

    # Search starting from 4-byte aligned position after start
    # Data in BioLogic files is typically 4-byte aligned from file start
    aligned_start = ((start + 3) // 4) * 4  # Round up to next 4-byte boundary
    for offset in range(aligned_start, min(start + 2000, end - record_size), 4):
        try:
            # Read first record at this offset
            sample = np.frombuffer(
                content[offset:offset + record_size],
                dtype=record_dtype
            )

            # Check first field value
            first_val = sample[first_field][0]

            if first_val == 0:
                continue

            if not is_reasonable_value(first_val, first_field):
                continue

            # Additional validation: check multiple fields and records
            if offset + record_size * 3 < end:
                test_data = np.frombuffer(
                    content[offset:offset + record_size * 3],
                    dtype=record_dtype
                )

                # Check first 3 fields for reasonable values
                all_valid = True
                for name in record_dtype.names[:3]:
                    vals = test_data[name]
                    for v in vals:
                        if not is_reasonable_value(float(v), name):
                            all_valid = False
                            break
                    if not all_valid:
                        break

                if not all_valid:
                    continue

            return offset

        except Exception:
            continue

    return None


def _find_column_info(content: bytes, start: int, end: int) -> Optional[Tuple[int, Tuple[int, ...], int]]:
    """
    Find column count and IDs in the data module

    Returns (n_cols, col_ids, position) or None
    """
    # Also try common columns for different techniques
    common_cols = {
        4,    # time/s
        5,    # control/V/mA
        6,    # Ewe/V
        8,    # I/mA
        13,   # (Q-Qo)/mA.h
        32,   # freq/Hz
        37,   # Re(Z)/Ohm
        38,   # -Im(Z)/Ohm
        212,  # half cycle
        468,  # half cycle (newer)
    }

    for offset in range(start, end, 2):
        if offset + 2 > len(content):
            break

        n_cols = struct.unpack('<H', content[offset:offset+2])[0]

        # Valid column count range for EC-Lab data
        if not (5 <= n_cols <= 60):
            continue

        # Check if we can read all column IDs
        col_end = offset + 2 + n_cols * 2
        if col_end > len(content):
            continue

        col_ids = struct.unpack(f'<{n_cols}H', content[offset+2:col_end])

        # Validate: count known columns
        known_count = sum(1 for c in col_ids if c in COLUMN_MAP or c in FLAG_COLUMNS)

        # At least 30% should be known columns
        if known_count >= max(3, n_cols * 0.3):
            # Additional validation: check for common columns
            has_common = any(c in col_ids for c in common_cols)

            if has_common:
                return (n_cols, col_ids, offset)

    return None


def load_mpr_to_dict(file_path: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Load MPR file and return as dictionary with standard column names

    Returns
    -------
    data : dict or None
        Dictionary with keys: time, voltage, current, capacity, etc.
    error : str or None
        Error message if loading failed
    """
    df, error = parse_mpr(file_path)

    if error or df is None:
        return None, error

    # Map to standard names
    data = {
        'raw_df': df,
        'columns': list(df.columns),
    }

    # Standard column mappings
    mappings = {
        'time': ['time/s', 'time'],
        'voltage': ['Ewe/V', '<Ewe>/V', 'Ewe', 'voltage'],
        'current': ['I/mA', '<I>/mA', 'control/V/mA', 'current'],
        'capacity': ['(Q-Qo)/mA.h', 'Q charge/discharge/mA.h', 'Capacity/mA.h', 'capacity'],
        'half_cycle': ['half cycle'],
        'cycle_number': ['cycle number'],
        # EIS specific
        'freq': ['freq/Hz', 'frequency'],
        'Z_real': ['Re(Z)/Ohm', "Z'/Ohm"],
        'Z_imag': ['-Im(Z)/Ohm', "Z''/Ohm"],
        'Z_mag': ['|Z|/Ohm'],
        'Z_phase': ['Phase(Z)/deg'],
    }

    for key, candidates in mappings.items():
        for col in candidates:
            if col in df.columns:
                data[key] = df[col].values
                break

    return data, None


def get_mpr_info(file_path: str) -> Dict[str, Any]:
    """
    Get basic information about an MPR file without fully parsing

    Returns dict with: technique, n_points, columns, etc.
    """
    info = {
        'file': os.path.basename(file_path),
        'valid': False,
        'technique': '',
        'n_points': 0,
        'columns': [],
    }

    try:
        with open(file_path, 'rb') as f:
            content = f.read()

        if not content[:21].decode('latin-1', errors='ignore').startswith('BIO-LOGIC MODULAR'):
            return info

        info['valid'] = True

        # Find modules
        modules = _find_modules(content)

        for mod in modules:
            if 'set' in mod['short_name'].lower():
                # Try to extract technique name from settings
                pass
            elif 'data' in mod['short_name'].lower():
                # Get column info
                col_info = _find_column_info(
                    content,
                    mod['header_end'],
                    min(mod['header_end'] + 200, len(content))
                )
                if col_info:
                    n_cols, col_ids, _ = col_info
                    info['columns'] = [
                        COLUMN_MAP.get(c, (None, f'unknown_{c}', ''))[1]
                        for c in col_ids
                    ]

        return info

    except Exception:
        return info
