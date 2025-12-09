# CD Analyzer

Streamlit-based web application for battery charge-discharge curve analysis and visualization.

## Features

- **Data Import**: Support for BioLogic .mpt files, CSV, and TXT formats
- **Multiple View Modes**:
  - Voltage vs Time (V-t)
  - Voltage vs Capacity (V-Q)
  - dQ/dV Analysis
  - Cycle Summary
- **Automatic Cycle Detection**: Parses charge/discharge cycles from data
- **Capacity Calculation**: Automatic calculation of specific capacity (mAh/g)
- **Coulombic Efficiency**: Automatic calculation of CE for each cycle
- **Publication-Ready Figures**: Plotly-based interactive plots with customizable styling
- **Export Options**:
  - CSV export with cycle summary
  - Igor Pro export (.itx) with pre-configured publication-quality plots

## Installation

```bash
# Clone the repository
git clone https://github.com/matsui-naoki/cd-app.git
cd cd-app

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

## Usage

1. Upload your battery data file (BioLogic .mpt, CSV, or TXT)
2. Enter sample information (mass, electrode area)
3. Select view mode from sidebar
4. Customize plot settings as needed
5. Export data or figures

## Supported File Formats

- **BioLogic .mpt**: Native support via galvani library
- **CSV**: Comma-separated with headers (time, voltage, current, capacity)
- **TXT**: Tab-separated data files

## Requirements

- Python 3.8+
- Streamlit >= 1.28.0
- NumPy >= 1.24.0
- Pandas >= 2.0.0
- Plotly >= 5.18.0
- SciPy >= 1.11.0
- galvani >= 0.5.0

## Project Structure

```
cd-app/
├── app.py                 # Main Streamlit application
├── tools/
│   ├── __init__.py
│   └── data_loader.py     # Data loading and parsing
├── components/
│   ├── __init__.py
│   └── plots.py           # Plotly visualization functions
├── utils/
│   ├── __init__.py
│   ├── helpers.py         # Helper functions
│   └── igor_export.py     # Igor Pro export
├── requirements.txt
├── .gitignore
└── README.md
```

## License

MIT License
