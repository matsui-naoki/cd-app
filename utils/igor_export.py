"""
Igor Pro export functionality for CD Analyzer.
Generates Igor Text Files (.itx) with publication-quality plot settings for battery data.
"""

import numpy as np
from typing import Dict, List, Optional


def make_wave_name(base_name: str, suffix: str, common_prefix: str = "", max_len: int = 31) -> str:
    """Create Igor-compatible wave name (max 31 chars, alphanumeric + underscore)"""
    name = base_name
    if common_prefix and name.startswith(common_prefix):
        name = name[len(common_prefix):]

    # Remove file extension
    name = name.rsplit('.', 1)[0] if '.' in name else name

    # Sanitize: only alphanumeric and underscore
    name = "".join(c if c.isalnum() or c == '_' else '_' for c in name)
    name = name.lstrip('_')

    # If starts with digit, add prefix
    if name and name[0].isdigit():
        name = 'w' + name

    # Combine with suffix
    full_name = f"{name}_{suffix}" if name else suffix

    # Truncate to max length
    if len(full_name) > max_len:
        suffix_len = len(suffix) + 1
        name_len = max_len - suffix_len
        if name_len > 0:
            full_name = f"{name[:name_len]}_{suffix}"
        else:
            full_name = suffix[:max_len]

    return full_name


def find_common_prefix(filenames: List[str]) -> str:
    """Find common prefix among filenames"""
    if len(filenames) <= 1:
        return ""

    common_prefix = ""
    min_len = min(len(f) for f in filenames)
    for i in range(min_len):
        chars = set(f[i] for f in filenames)
        if len(chars) == 1:
            common_prefix += filenames[0][i]
        else:
            break
    return common_prefix


def generate_cd_style(y_min: float, y_max: float, x_max: float) -> List[str]:
    """Generate charge-discharge plot style commands"""
    return [
        "X ModifyGraph mode=0,lsize=1.5",
        "X ModifyGraph tick=2,minor=0,btLen=5",
        "X ModifyGraph mirror=2,gfSize=14,gFont=\"Arial\"",
        "X ModifyGraph width=283,height=198",
        "X ModifyGraph margin(left)=50,margin(bottom)=45,margin(top)=10,margin(right)=45",
        "X Label left \"\\\\f02V\\\\f00 / V\"",
        "X Label bottom \"Time / h\"",
        f"X SetAxis left {y_min:.4f},{y_max:.4f}",
        f"X SetAxis bottom 0,{x_max:.4f}",
    ]


def generate_capacity_voltage_style(v_min: float, v_max: float, cap_max: float) -> List[str]:
    """Generate V vs Q plot style commands"""
    return [
        "X ModifyGraph mode=0,lsize=1.5",
        "X ModifyGraph tick=2,minor=0,btLen=5",
        "X ModifyGraph mirror=2,gfSize=14,gFont=\"Arial\"",
        "X ModifyGraph width=198,height=170",
        "X ModifyGraph margin(left)=50,margin(bottom)=45,margin(top)=10,margin(right)=10",
        "X Label left \"\\\\f02V\\\\f00 / V\"",
        "X Label bottom \"Capacity / mAh g\\\\S–1\\\\M\"",
        f"X SetAxis left {v_min:.4f},{v_max:.4f}",
        f"X SetAxis bottom 0,{cap_max:.4f}",
    ]


def generate_dqdv_style(v_min: float, v_max: float, dqdv_max: float) -> List[str]:
    """Generate dQ/dV plot style commands"""
    return [
        "X ModifyGraph mode=0,lsize=1.5",
        "X ModifyGraph tick=2,minor=0,btLen=5",
        "X ModifyGraph mirror=2,gfSize=14,gFont=\"Arial\"",
        "X ModifyGraph width=198,height=170",
        "X ModifyGraph margin(left)=50,margin(bottom)=45,margin(top)=10,margin(right)=10",
        "X Label left \"d\\\\f02Q\\\\f00/d\\\\f02V\\\\f00 / mAh g\\\\S–1\\\\M V\\\\S–1\\\\M\"",
        "X Label bottom \"\\\\f02V\\\\f00 / V\"",
        f"X SetAxis bottom {v_min:.4f},{v_max:.4f}",
        f"X SetAxis left {-dqdv_max:.4f},{dqdv_max:.4f}",
    ]


def generate_cycle_summary_style(n_cycles: int, cap_max: float) -> List[str]:
    """Generate cycle summary plot style commands"""
    return [
        "X ModifyGraph mode=3,marker=19,msize=4",
        "X ModifyGraph mrkThick=0.5,useMrkStrokeRGB=1",
        "X ModifyGraph tick=2,minor=0,btLen=5",
        "X ModifyGraph mirror=2,gfSize=14,gFont=\"Arial\"",
        "X ModifyGraph width=283,height=170",
        "X ModifyGraph margin(left)=50,margin(bottom)=45,margin(top)=10,margin(right)=50",
        "X Label left \"Capacity / mAh g\\\\S–1\\\\M\"",
        "X Label bottom \"Cycle number\"",
        f"X SetAxis bottom 0,{n_cycles + 1}",
        f"X SetAxis left 0,{cap_max * 1.1:.4f}",
        "X Label right \"Coulombic efficiency / %\"",
        "X SetAxis right 90,105",
    ]


def generate_igor_file(
    files_data: Dict,
    sample_info: Dict,
    export_type: str = 'all'
) -> str:
    """
    Generate Igor Text File (.itx) for battery data.

    Args:
        files_data: Dictionary of {filename: file_data} from session state
        sample_info: Sample information dictionary
        export_type: 'all', 'raw', 'cycles', 'summary'

    Returns:
        Igor Text File content as string
    """
    lines = ["IGOR"]

    filenames = list(files_data.keys())
    if not filenames:
        return "IGOR\n"

    common_prefix = find_common_prefix(filenames)
    sample_name = sample_info.get('name', '').strip()
    mass_g = sample_info.get('mass_mg', 1.0) / 1000

    # Color palette
    colors = [
        (0, 0, 0),           # black
        (65535, 0, 0),       # red
        (0, 0, 65535),       # blue
        (0, 39321, 0),       # green
        (65535, 32768, 0),   # orange
        (0, 65535, 65535),   # cyan
        (65535, 0, 65535),   # magenta
    ]

    # Collect data for each file
    all_waves = []
    v_min_global, v_max_global = float('inf'), float('-inf')
    t_max_global = 0
    cap_max_global = 0

    for fname in filenames:
        fdata = files_data[fname]

        time = fdata.get('time')
        voltage = fdata.get('voltage')
        current = fdata.get('current')
        capacity = fdata.get('capacity')

        if time is None or voltage is None:
            continue

        time_h = time / 3600  # Convert to hours

        # Track global ranges
        v_min_global = min(v_min_global, np.min(voltage))
        v_max_global = max(v_max_global, np.max(voltage))
        t_max_global = max(t_max_global, np.max(time_h))

        # Create wave names
        wave_time = make_wave_name(fname, 'time', common_prefix)
        wave_volt = make_wave_name(fname, 'V', common_prefix)

        all_waves.append({
            'fname': fname,
            'wave_time': wave_time,
            'wave_volt': wave_volt,
        })

        # Write time and voltage waves
        lines.append(f"WAVES/O {wave_time}, {wave_volt}")
        lines.append("BEGIN")
        for i in range(len(time)):
            lines.append(f"  {time_h[i]:.6E}  {voltage[i]:.6E}")
        lines.append("END")
        lines.append("")

        # Write current if available
        if current is not None:
            wave_curr = make_wave_name(fname, 'I', common_prefix)
            all_waves[-1]['wave_curr'] = wave_curr

            lines.append(f"WAVES/O {wave_curr}")
            lines.append("BEGIN")
            for i in range(len(current)):
                lines.append(f"  {current[i]:.6E}")
            lines.append("END")
            lines.append("")

        # Write capacity if available
        if capacity is not None:
            cap_normalized = np.abs(capacity) / mass_g if mass_g > 0 else np.abs(capacity)
            cap_max_global = max(cap_max_global, np.max(cap_normalized))

            wave_cap = make_wave_name(fname, 'Q', common_prefix)
            all_waves[-1]['wave_cap'] = wave_cap

            lines.append(f"WAVES/O {wave_cap}")
            lines.append("BEGIN")
            for i in range(len(capacity)):
                lines.append(f"  {cap_normalized[i]:.6E}")
            lines.append("END")
            lines.append("")

        # Process cycles
        cycles = fdata.get('cycles', [])
        cycle_nums = []
        charge_caps = []
        discharge_caps = []
        efficiencies = []

        for cycle in cycles:
            cn = cycle.get('cycle_number', 0) + 1
            cycle_nums.append(cn)

            cap_c = cycle.get('capacity_charge_mAh')
            cap_d = cycle.get('capacity_discharge_mAh')
            ce = cycle.get('coulombic_efficiency')

            if cap_c is not None and mass_g > 0:
                charge_caps.append(cap_c / mass_g)
            else:
                charge_caps.append(0)

            if cap_d is not None and mass_g > 0:
                discharge_caps.append(cap_d / mass_g)
                cap_max_global = max(cap_max_global, cap_d / mass_g)
            else:
                discharge_caps.append(0)

            if ce is not None:
                efficiencies.append(ce * 100)
            else:
                efficiencies.append(0)

        # Write cycle summary if available
        if cycle_nums:
            wave_cn = make_wave_name(fname, 'cn', common_prefix)
            wave_qc = make_wave_name(fname, 'Qc', common_prefix)
            wave_qd = make_wave_name(fname, 'Qd', common_prefix)
            wave_ce = make_wave_name(fname, 'CE', common_prefix)

            all_waves[-1].update({
                'wave_cn': wave_cn,
                'wave_qc': wave_qc,
                'wave_qd': wave_qd,
                'wave_ce': wave_ce,
                'n_cycles': len(cycle_nums),
            })

            lines.append(f"WAVES/O {wave_cn}, {wave_qc}, {wave_qd}, {wave_ce}")
            lines.append("BEGIN")
            for i in range(len(cycle_nums)):
                lines.append(f"  {cycle_nums[i]}  {charge_caps[i]:.6E}  {discharge_caps[i]:.6E}  {efficiencies[i]:.4f}")
            lines.append("END")
            lines.append("")

    # =========================================
    # Generate plots
    # =========================================

    # V-t plot (all data combined)
    if all_waves:
        first = all_waves[0]
        lines.append(f"X Display {first['wave_volt']} vs {first['wave_time']} as \"CD_Voltage_vs_Time\"")

        for wave_info in all_waves[1:]:
            lines.append(f"X AppendToGraph {wave_info['wave_volt']} vs {wave_info['wave_time']}")

        # Add margin to voltage range
        v_margin = (v_max_global - v_min_global) * 0.05
        lines.extend(generate_cd_style(v_min_global - v_margin, v_max_global + v_margin, t_max_global))

        # Apply colors
        for i, wave_info in enumerate(all_waves):
            color = colors[i % len(colors)]
            lines.append(f"X ModifyGraph rgb({wave_info['wave_volt']})=({color[0]},{color[1]},{color[2]})")

        lines.append("X Legend/C/N=text0/F=0/B=1")
        lines.append("")

    # V-Q plot for files with capacity
    waves_with_cap = [w for w in all_waves if 'wave_cap' in w]
    if waves_with_cap:
        first = waves_with_cap[0]
        lines.append(f"X Display {first['wave_volt']} vs {first['wave_cap']} as \"CD_Voltage_vs_Capacity\"")

        for wave_info in waves_with_cap[1:]:
            lines.append(f"X AppendToGraph {wave_info['wave_volt']} vs {wave_info['wave_cap']}")

        v_margin = (v_max_global - v_min_global) * 0.05
        lines.extend(generate_capacity_voltage_style(
            v_min_global - v_margin, v_max_global + v_margin, cap_max_global * 1.05
        ))

        for i, wave_info in enumerate(waves_with_cap):
            color = colors[i % len(colors)]
            lines.append(f"X ModifyGraph rgb({wave_info['wave_volt']})=({color[0]},{color[1]},{color[2]})")

        lines.append("X Legend/C/N=text0/F=0/B=1")
        lines.append("")

    # Cycle summary plot for files with cycle data
    waves_with_cycles = [w for w in all_waves if 'wave_cn' in w]
    if waves_with_cycles:
        first = waves_with_cycles[0]
        n_cycles = first.get('n_cycles', 10)

        lines.append(f"X Display {first['wave_qd']} vs {first['wave_cn']} as \"CD_Cycle_Summary\"")
        lines.append(f"X AppendToGraph {first['wave_qc']} vs {first['wave_cn']}")
        lines.append(f"X AppendToGraph/R {first['wave_ce']} vs {first['wave_cn']}")

        lines.extend(generate_cycle_summary_style(n_cycles, cap_max_global))

        # Set colors and markers
        lines.append(f"X ModifyGraph rgb({first['wave_qd']})=(0,0,65535)")
        lines.append(f"X ModifyGraph rgb({first['wave_qc']})=(65535,0,0)")
        lines.append(f"X ModifyGraph rgb({first['wave_ce']})=(39321,39321,39321)")
        lines.append(f"X ModifyGraph marker({first['wave_qd']})=19")
        lines.append(f"X ModifyGraph marker({first['wave_qc']})=16")
        lines.append(f"X ModifyGraph marker({first['wave_ce']})=17")

        lines.append("X Legend/C/N=text0/F=0/B=1")
        lines.append("")

    return "\n".join(lines) + "\n"


def generate_csv_export(
    data: Dict,
    sample_info: Dict,
    include_cycles: bool = True
) -> str:
    """
    Generate CSV export of battery data.

    Args:
        data: Battery data dictionary
        sample_info: Sample information
        include_cycles: Whether to include cycle-by-cycle summary

    Returns:
        CSV content as string
    """
    import io

    lines = []
    mass_g = sample_info.get('mass_mg', 1.0) / 1000

    # Header with sample info
    lines.append(f"# Sample: {sample_info.get('name', '')}")
    lines.append(f"# Mass: {sample_info.get('mass_mg', 0)} mg")
    lines.append(f"# Area: {sample_info.get('area_cm2', 0)} cm^2")
    lines.append("")

    # Raw data section
    lines.append("# Raw Data")
    headers = ['Time (s)', 'Voltage (V)']

    time = data.get('time', [])
    voltage = data.get('voltage', [])
    current = data.get('current')
    capacity = data.get('capacity')

    if current is not None:
        headers.append('Current (mA)')
    if capacity is not None:
        headers.append('Capacity (mAh)')
        headers.append('Capacity (mAh/g)')

    lines.append(','.join(headers))

    for i in range(len(time)):
        row = [f"{time[i]:.6f}", f"{voltage[i]:.6f}"]
        if current is not None:
            row.append(f"{current[i]:.6f}")
        if capacity is not None:
            row.append(f"{capacity[i]:.6f}")
            row.append(f"{capacity[i]/mass_g:.6f}" if mass_g > 0 else "0")
        lines.append(','.join(row))

    lines.append("")

    # Cycle summary section
    if include_cycles:
        cycles = data.get('cycles', [])
        if cycles:
            lines.append("# Cycle Summary")
            lines.append("Cycle,Charge Capacity (mAh/g),Discharge Capacity (mAh/g),Coulombic Efficiency (%)")

            for cycle in cycles:
                cn = cycle.get('cycle_number', 0) + 1
                cap_c = cycle.get('capacity_charge_mAh', 0)
                cap_d = cycle.get('capacity_discharge_mAh', 0)
                ce = cycle.get('coulombic_efficiency', 0)

                cap_c_norm = cap_c / mass_g if mass_g > 0 else cap_c
                cap_d_norm = cap_d / mass_g if mass_g > 0 else cap_d

                lines.append(f"{cn},{cap_c_norm:.2f},{cap_d_norm:.2f},{ce*100:.2f}")

    return '\n'.join(lines)
