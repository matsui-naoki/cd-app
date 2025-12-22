"""
File Processing Functions for CD Analyzer
"""

import os
import numpy as np
import streamlit as st

from tools.data_loader import (
    load_uploaded_file, validate_cd_data, get_supported_formats
)
from tools.mps_parser import (
    parse_mps_file, load_gcpl_data_from_session, load_peis_data_from_session
)


def process_uploaded_files(uploaded_files):
    """Process uploaded battery data files"""
    if not uploaded_files:
        return

    for uploaded_file in uploaded_files:
        filename = uploaded_file.name
        base_name = os.path.splitext(filename)[0]
        file_ext = os.path.splitext(filename)[1].lower()

        # Check if already loaded
        if base_name in st.session_state.files:
            continue

        # Handle MPS files specially
        if file_ext == '.mps':
            process_mps_file(uploaded_file)
            continue

        try:
            data, error = load_uploaded_file(uploaded_file, file_ext)

            if error:
                st.error(f"Error loading {filename}: {error}")
                continue

            if data is not None:
                is_valid, msg = validate_cd_data(data)
                if is_valid:
                    st.session_state.files[base_name] = data
                    st.success(f"Loaded: {filename}")
                else:
                    st.warning(f"Invalid data in {filename}: {msg}")

        except Exception as e:
            st.error(f"Error loading {filename}: {str(e)}")


def process_mps_file(uploaded_file):
    """Process MPS settings file and load related data files"""
    import tempfile

    try:
        # Save MPS file temporarily
        bytes_data = uploaded_file.read()

        with tempfile.TemporaryDirectory() as tmp_dir:
            # We need the original folder structure, so this won't work with just upload
            # Instead, inform user to upload all files together
            st.warning("MPS file detected. Please upload the entire measurement folder or use folder path input.")
            return

    except Exception as e:
        st.error(f"Error processing MPS file: {str(e)}")


def load_mps_session(mps_path: str):
    """Load measurement session from MPS file"""
    if not os.path.exists(mps_path):
        st.error(f"File not found: {mps_path}")
        return

    session = parse_mps_file(mps_path)
    if session is None:
        st.error("Failed to parse MPS file")
        return

    st.session_state.mps_session = session

    # Update sample info from MPS
    if session.sample_info:
        if 'mass_mg' in session.sample_info and session.sample_info['mass_mg'] > 0:
            st.session_state.sample_info['mass_mg'] = session.sample_info['mass_mg']
        if 'electrode_area_cm2' in session.sample_info and session.sample_info['electrode_area_cm2'] > 0:
            st.session_state.sample_info['area_cm2'] = session.sample_info['electrode_area_cm2']

    # Set sample name from base name
    st.session_state.sample_info['name'] = session.base_name

    # Load GCPL data
    gcpl_data = load_gcpl_data_from_session(session)
    if gcpl_data:
        st.session_state.files['GCPL_combined'] = gcpl_data
        st.session_state.selected_file = 'GCPL_combined'

    # Try to load EIS data
    try:
        eis_list = load_peis_data_from_session(session)
        st.session_state.eis_data = eis_list
    except Exception as e:
        st.warning(f"Could not load EIS data: {e}")

    st.success(f"Loaded session: {session.base_name}")
    st.rerun()


def sort_files_by_time_and_assign_cycles(files_data: dict) -> list:
    """
    Sort files by their start time and assign cycle numbers.
    Pairs charge+discharge as one cycle.

    Returns list of dicts: [{filename, data, start_time, is_charge, is_discharge, cycle_num, current_mA, color}]
    """
    sorted_files = []

    for filename, data in files_data.items():
        # Get start time
        if 'time' in data and data['time'] is not None and len(data['time']) > 0:
            start_time = data['time'][0]
        else:
            start_time = 0

        # Determine if charge or discharge from current
        is_charge = False
        is_discharge = False
        current_mA = 0

        if 'current' in data and data['current'] is not None and len(data['current']) > 0:
            avg_current = np.mean(data['current'])
            current_mA = abs(avg_current)
            if avg_current > 0.01:
                is_charge = True
            elif avg_current < -0.01:
                is_discharge = True

        # Check cycles for charge/discharge info
        if 'cycles' in data and len(data['cycles']) > 0:
            cycle = data['cycles'][0]
            if cycle.get('is_charge'):
                is_charge = True
            if cycle.get('is_discharge'):
                is_discharge = True

        sorted_files.append({
            'filename': filename,
            'data': data,
            'start_time': start_time,
            'is_charge': is_charge,
            'is_discharge': is_discharge,
            'current_mA': current_mA,
        })

    # Sort by start time
    sorted_files.sort(key=lambda x: x['start_time'])

    # Assign cycle numbers (pair charge+discharge as one cycle)
    cycle_num = 0
    half_cycle_in_pair = 0

    for i, file_info in enumerate(sorted_files):
        file_info['cycle_num'] = cycle_num
        half_cycle_in_pair += 1

        # After 2 half-cycles, increment cycle number
        if half_cycle_in_pair >= 2:
            cycle_num += 1
            half_cycle_in_pair = 0

    # Assign default colors: cycle1=red, middle=black, last=blue
    n_cycles = cycle_num + 1 if half_cycle_in_pair > 0 else cycle_num

    for file_info in sorted_files:
        filename = file_info['filename']
        cyc = file_info['cycle_num']

        # Check if user has set custom color
        if filename in st.session_state.file_colors:
            file_info['color'] = st.session_state.file_colors[filename]
        else:
            # Default: first=red, last=blue, middle=black
            if cyc == 0:
                file_info['color'] = '#E63946'  # Red
            elif cyc == n_cycles - 1:
                file_info['color'] = '#457B9D'  # Blue
            else:
                file_info['color'] = '#000000'  # Black

    return sorted_files
