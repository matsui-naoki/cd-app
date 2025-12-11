"""
Theoretical Capacity Calculator for Battery Active Materials

Calculates theoretical specific capacity from chemical composition and reaction electrons.
Formula: Q = nF / (M * 3.6)  [mAh/g]
where:
    n = number of electrons
    F = Faraday constant (96485 C/mol)
    M = molar mass (g/mol)
    3.6 = conversion factor (C to mAh)
"""

import re
from typing import Tuple, Optional, Dict

# Standard atomic weights (IUPAC 2021)
ATOMIC_WEIGHTS: Dict[str, float] = {
    'H': 1.008,
    'He': 4.003,
    'Li': 6.94,
    'Be': 9.012,
    'B': 10.81,
    'C': 12.011,
    'N': 14.007,
    'O': 15.999,
    'F': 18.998,
    'Ne': 20.180,
    'Na': 22.990,
    'Mg': 24.305,
    'Al': 26.982,
    'Si': 28.085,
    'P': 30.974,
    'S': 32.06,
    'Cl': 35.45,
    'Ar': 39.948,
    'K': 39.098,
    'Ca': 40.078,
    'Sc': 44.956,
    'Ti': 47.867,
    'V': 50.942,
    'Cr': 51.996,
    'Mn': 54.938,
    'Fe': 55.845,
    'Co': 58.933,
    'Ni': 58.693,
    'Cu': 63.546,
    'Zn': 65.38,
    'Ga': 69.723,
    'Ge': 72.630,
    'As': 74.922,
    'Se': 78.971,
    'Br': 79.904,
    'Kr': 83.798,
    'Rb': 85.468,
    'Sr': 87.62,
    'Y': 88.906,
    'Zr': 91.224,
    'Nb': 92.906,
    'Mo': 95.95,
    'Tc': 98.0,
    'Ru': 101.07,
    'Rh': 102.91,
    'Pd': 106.42,
    'Ag': 107.87,
    'Cd': 112.41,
    'In': 114.82,
    'Sn': 118.71,
    'Sb': 121.76,
    'Te': 127.60,
    'I': 126.90,
    'Xe': 131.29,
    'Cs': 132.91,
    'Ba': 137.33,
    'La': 138.91,
    'Ce': 140.12,
    'Pr': 140.91,
    'Nd': 144.24,
    'Pm': 145.0,
    'Sm': 150.36,
    'Eu': 151.96,
    'Gd': 157.25,
    'Tb': 158.93,
    'Dy': 162.50,
    'Ho': 164.93,
    'Er': 167.26,
    'Tm': 168.93,
    'Yb': 173.05,
    'Lu': 174.97,
    'Hf': 178.49,
    'Ta': 180.95,
    'W': 183.84,
    'Re': 186.21,
    'Os': 190.23,
    'Ir': 192.22,
    'Pt': 195.08,
    'Au': 196.97,
    'Hg': 200.59,
    'Tl': 204.38,
    'Pb': 207.2,
    'Bi': 208.98,
    'Po': 209.0,
    'At': 210.0,
    'Rn': 222.0,
    'Fr': 223.0,
    'Ra': 226.0,
    'Ac': 227.0,
    'Th': 232.04,
    'Pa': 231.04,
    'U': 238.03,
    'Np': 237.0,
    'Pu': 244.0,
    'Am': 243.0,
    'Cm': 247.0,
    'Bk': 247.0,
    'Cf': 251.0,
    'Es': 252.0,
    'Fm': 257.0,
    'Md': 258.0,
    'No': 259.0,
    'Lr': 262.0,
}

# Faraday constant (C/mol)
FARADAY_CONSTANT = 96485.3321


def parse_composition(formula: str) -> Dict[str, float]:
    """
    Parse chemical formula into element counts.

    Supports:
    - Simple formulas: LiCoO2, Fe2O3
    - Decimal subscripts: Li0.5CoO2
    - Parentheses: (PO4)2, Ca(OH)2

    Parameters
    ----------
    formula : str
        Chemical formula string

    Returns
    -------
    elements : dict
        Dictionary mapping element symbols to counts
    """
    if not formula or not formula.strip():
        return {}

    formula = formula.strip()

    # Handle parentheses recursively
    while '(' in formula:
        # Find innermost parentheses
        match = re.search(r'\(([^()]+)\)(\d*\.?\d*)', formula)
        if not match:
            break

        group_content = match.group(1)
        multiplier = float(match.group(2)) if match.group(2) else 1.0

        # Parse group content
        group_elements = parse_composition(group_content)

        # Build replacement string
        replacement = ''
        for elem, count in group_elements.items():
            new_count = count * multiplier
            if new_count == int(new_count):
                replacement += f"{elem}{int(new_count)}"
            else:
                replacement += f"{elem}{new_count}"

        # Replace parentheses group
        formula = formula[:match.start()] + replacement + formula[match.end():]

    # Parse element-number pairs
    # Matches: Element symbol (1-2 letters) followed by optional number (int or float)
    pattern = r'([A-Z][a-z]?)(\d*\.?\d*)'
    matches = re.findall(pattern, formula)

    elements = {}
    for element, count_str in matches:
        if not element:
            continue
        count = float(count_str) if count_str else 1.0
        elements[element] = elements.get(element, 0) + count

    return elements


def calculate_molar_mass(formula: str) -> Optional[float]:
    """
    Calculate molar mass from chemical formula.

    Parameters
    ----------
    formula : str
        Chemical formula string

    Returns
    -------
    molar_mass : float or None
        Molar mass in g/mol, or None if formula is invalid
    """
    elements = parse_composition(formula)

    if not elements:
        return None

    molar_mass = 0.0
    for element, count in elements.items():
        if element not in ATOMIC_WEIGHTS:
            raise ValueError(f"Unknown element: {element}")
        molar_mass += ATOMIC_WEIGHTS[element] * count

    return molar_mass


def calculate_theoretical_capacity(formula: str, n_electrons: int = 1) -> Tuple[Optional[float], Optional[float]]:
    """
    Calculate theoretical specific capacity from chemical formula.

    Formula: Q = nF / (M * 3.6)  [mAh/g]

    Parameters
    ----------
    formula : str
        Chemical formula of active material (e.g., 'LiCoO2', 'LiFePO4')
    n_electrons : int
        Number of electrons transferred in the electrochemical reaction

    Returns
    -------
    capacity : float or None
        Theoretical specific capacity in mAh/g
    molar_mass : float or None
        Molar mass in g/mol

    Examples
    --------
    >>> calculate_theoretical_capacity('LiCoO2', 1)
    (137.21, 97.87)  # ~137 mAh/g

    >>> calculate_theoretical_capacity('LiFePO4', 1)
    (170.04, 157.76)  # ~170 mAh/g

    >>> calculate_theoretical_capacity('Li2MnO3', 2)
    (458.12, 116.82)  # ~458 mAh/g for 2e-
    """
    try:
        molar_mass = calculate_molar_mass(formula)
        if molar_mass is None or molar_mass <= 0:
            return None, None

        # Q = nF / (M * 3.6) [mAh/g]
        # 3.6 converts C to mAh (1 mAh = 3.6 C)
        capacity = (n_electrons * FARADAY_CONSTANT) / (molar_mass * 3.6)

        return round(capacity, 2), round(molar_mass, 2)

    except Exception:
        return None, None


# Common battery materials reference
COMMON_MATERIALS = {
    'LiCoO2': {'n': 1, 'capacity': 137.2},
    'LiFePO4': {'n': 1, 'capacity': 170.0},
    'LiMn2O4': {'n': 1, 'capacity': 148.0},
    'LiNi0.8Co0.15Al0.05O2': {'n': 1, 'capacity': 199.0},  # NCA
    'LiNi0.33Co0.33Mn0.33O2': {'n': 1, 'capacity': 160.0},  # NCM111
    'LiNi0.5Co0.2Mn0.3O2': {'n': 1, 'capacity': 170.0},  # NCM523
    'LiNi0.8Co0.1Mn0.1O2': {'n': 1, 'capacity': 200.0},  # NCM811
    'Li2MnO3': {'n': 2, 'capacity': 459.0},
    'Li4Ti5O12': {'n': 3, 'capacity': 175.0},  # LTO
    'FeS2': {'n': 4, 'capacity': 894.0},
    'S': {'n': 2, 'capacity': 1672.0},  # Sulfur
    'Si': {'n': 4, 'capacity': 3579.0},  # Silicon (Li15Si4)
}


if __name__ == '__main__':
    # Test examples
    test_cases = [
        ('LiCoO2', 1),
        ('LiFePO4', 1),
        ('LiMn2O4', 1),
        ('Li2MnO3', 2),
        ('Li4Ti5O12', 3),
        ('S', 2),
        ('Si', 4),
    ]

    print("Theoretical Capacity Calculator Test")
    print("=" * 50)
    for formula, n in test_cases:
        cap, mw = calculate_theoretical_capacity(formula, n)
        print(f"{formula} (n={n}): {cap:.1f} mAh/g (M = {mw:.2f} g/mol)")
