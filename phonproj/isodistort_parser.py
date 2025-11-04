"""
Parser for ISODISTORT output files containing phonon mode definitions.

ISODISTORT (https://stokes.byu.edu/iso/isodistort.php) is a tool for analyzing
structural distortions in terms of symmetry modes. This module provides utilities
to parse ISODISTORT output files that contain:
- Parent structure definition
- Supercell structures (undistorted and distorted)
- Phonon mode definitions with displacement patterns
"""

from pathlib import Path
from typing import Dict, List, Tuple, Union

from ase import Atoms


def _dict_to_atoms(structure_dict: dict, lattice_key: str = "lattice_params") -> Atoms:
    """
    Converts a dictionary representation of a structure from the ISODISTORT parser
    into an ase.Atoms object.
    """
    if lattice_key not in structure_dict:
        raise ValueError(f"Lattice key '{lattice_key}' not found in structure_dict.")

    lattice_params = structure_dict[lattice_key]
    a = lattice_params.get("a")
    b = lattice_params.get("b")
    c = lattice_params.get("c")
    alpha = lattice_params.get("alpha", 90)
    beta = lattice_params.get("beta", 90)
    gamma = lattice_params.get("gamma", 90)

    if a is None or b is None or c is None:
        raise ValueError("Missing a, b, or c lattice parameters.")

    # Use ASE's cellpar parameter (angles are already in degrees from ISODISTORT)
    cellpar = [a, b, c, alpha, beta, gamma]

    symbols = []
    scaled_positions = []
    for atom in structure_dict["atoms"]:
        symbols.append(atom["element"])
        scaled_positions.append([atom["x"], atom["y"], atom["z"]])

    return Atoms(
        symbols=symbols, scaled_positions=scaled_positions, cell=cellpar, pbc=True
    )


def parse_isodistort_file(filepath: Union[str, Path]) -> Dict[str, Union[Atoms, Dict]]:
    """
    Parse ISODISTORT file containing phonon mode definitions.

    Parameters
    ----------
    filepath : Union[str, Path]
        Path to the ISODISTORT file

    Returns
    -------
    dict
        Dictionary containing:
        - parent_structure: dict with lattice and atomic positions
        - supercell_structure: dict with undistorted and distorted structures
        - modes: dict of phonon modes with displacement patterns

    Raises
    ------
    FileNotFoundError
        If the ISODISTORT file does not exist

    Examples
    --------
    >>> result = parse_isodistort_file('P4mmm-ref.txt')
    >>> parent = result['parent_structure']
    >>> modes = result['modes']
    >>> print(f"Found {len(modes)} phonon modes")
    """
    if isinstance(filepath, str):
        filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"ISODISTORT file not found: {filepath}")

    with open(filepath) as f:
        lines = f.readlines()

    result: Dict = {"parent_structure": {}, "supercell_structure": {}, "modes": {}}
    parent_structure_raw = {}
    undistorted_supercell_raw = {"lattice_params": {}, "atoms": []}
    distorted_supercell_raw = {"lattice_params": {}, "atoms": []}

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Parse parent structure
        if line.startswith("Parent structure"):
            i += 1
            # Parse lattice parameters
            lattice_line = lines[i].strip()
            if lattice_line.startswith("a="):
                params = {}
                for part in lattice_line.split(","):
                    key, val = part.strip().split("=")
                    params[key] = float(val)
                parent_structure_raw["lattice_params"] = params
            i += 1

            # Skip header line
            i += 1

            # Parse atoms
            atoms = []
            while (
                i < len(lines)
                and lines[i].strip()
                and not lines[i].strip().startswith("Subgroup")
            ):
                parts = lines[i].strip().split()
                if len(parts) >= 5:
                    atoms.append(
                        {
                            "element": parts[0].rstrip("0123456789"),
                            "label": parts[0],
                            "site": parts[1],
                            "x": float(parts[2]),
                            "y": float(parts[3]),
                            "z": float(parts[4]),
                            "occupancy": float(parts[5]),
                        }
                    )
                i += 1
            parent_structure_raw["atoms"] = atoms

        # Parse undistorted superstructure
        elif line.startswith("Undistorted superstructure"):
            i += 1
            # Parse lattice
            lattice_line = lines[i].strip()
            if lattice_line.startswith("a="):
                params = {}
                for part in lattice_line.split(","):
                    key, val = part.strip().split("=")
                    params[key] = float(val)
                undistorted_supercell_raw["lattice_params"] = params
            i += 1

            # Skip header
            i += 1

            # Parse atoms
            atoms = []
            while (
                i < len(lines)
                and lines[i].strip()
                and not lines[i].strip().startswith("Distorted")
            ):
                parts = lines[i].strip().split()
                if len(parts) >= 6:
                    atoms.append(
                        {
                            "element": parts[0].rstrip("0123456789"),
                            "label": parts[0],
                            "site": parts[1],
                            "x": float(parts[2]),
                            "y": float(parts[3]),
                            "z": float(parts[4]),
                            "occupancy": float(parts[5]),
                            "displ": float(parts[6]),
                        }
                    )
                i += 1
            undistorted_supercell_raw["atoms"] = atoms

        # Parse distorted superstructure
        elif line.startswith("Distorted superstructure"):
            i += 1
            # Parse lattice
            lattice_line = lines[i].strip()
            if lattice_line.startswith("a="):
                params = {}
                for part in lattice_line.split(","):
                    key, val = part.strip().split("=")
                    params[key] = float(val)
                distorted_supercell_raw["lattice_params"] = params
            i += 1

            # Skip header
            i += 1

            # Parse atoms
            atoms = []
            while (
                i < len(lines)
                and lines[i].strip()
                and not lines[i].strip().startswith("Displacive")
            ):
                parts = lines[i].strip().split()
                if len(parts) >= 6:
                    atoms.append(
                        {
                            "element": parts[0].rstrip("0123456789"),
                            "label": parts[0],
                            "site": parts[1],
                            "x": float(parts[2]),
                            "y": float(parts[3]),
                            "z": float(parts[4]),
                            "occupancy": float(parts[5]),
                            "displ": float(parts[6]),
                        }
                    )
                i += 1
            distorted_supercell_raw["atoms"] = atoms

        # Parse displacive mode definitions
        elif line.startswith("Displacive mode definitions"):
            i += 1
            # Skip blank line
            i += 1

            # Parse modes
            while i < len(lines):
                line = lines[i].strip()

                # Check if this is a mode header
                if "normfactor" in line:
                    # Extract mode information from header
                    mode_info = line.split()[
                        0
                    ]  # e.g., "P1[0,0,0]GM1(a)[Pb1:a:dsp]A_1(a)"
                    normfactor_part = (
                        line.split("normfactor")[1].strip().split("=")[1].strip()
                    )
                    normfactor = float(normfactor_part)

                    i += 1

                    # Parse displacement vectors for this mode
                    displacements = []
                    while i < len(lines):
                        line = lines[i].strip()
                        if not line:
                            # End of this mode
                            break
                        if "normfactor" in line:
                            # Start of next mode
                            break

                        parts = line.split()
                        if len(parts) >= 7:
                            displacements.append(
                                {
                                    "atom_label": parts[0],
                                    "x": float(parts[1]),
                                    "y": float(parts[2]),
                                    "z": float(parts[3]),
                                    "dx": float(parts[4]),
                                    "dy": float(parts[5]),
                                    "dz": float(parts[6]),
                                }
                            )
                        i += 1

                    # Store mode
                    if mode_info not in result["modes"]:
                        result["modes"][mode_info] = {
                            "normfactor": normfactor,
                            "displacements": displacements,
                        }
                else:
                    i += 1

        else:
            i += 1

    # Convert raw dictionaries to ase.Atoms objects
    if parent_structure_raw.get("atoms"):
        result["parent_structure"] = _dict_to_atoms(
            parent_structure_raw, lattice_key="lattice_params"
        )

    if undistorted_supercell_raw.get("atoms"):
        result["supercell_structure"]["undistorted"] = _dict_to_atoms(
            undistorted_supercell_raw, lattice_key="lattice_params"
        )

    if distorted_supercell_raw.get("atoms"):
        result["supercell_structure"]["distorted"] = _dict_to_atoms(
            distorted_supercell_raw, lattice_key="lattice_params"
        )

    return result


def extract_qpoint_from_mode_label(mode_label: str) -> Tuple[float, float, float]:
    """
    Extract q-point coordinates from ISODISTORT mode label.

    Parameters
    ----------
    mode_label : str
        Mode label in format P1[qx,qy,qz]...

    Returns
    -------
    Tuple[float, float, float]
        Q-point coordinates (qx, qy, qz)

    Examples
    --------
    >>> qpoint = extract_qpoint_from_mode_label('P1[0,0,0]GM1(a)[Pb1:a:dsp]A_1(a)')
    >>> print(qpoint)
    (0.0, 0.0, 0.0)

    >>> qpoint = extract_qpoint_from_mode_label('P1[1/2,0,0]X1(a)[Pb1:a:dsp]A_1(a)')
    >>> print(qpoint)
    (0.5, 0.0, 0.0)
    """
    import re

    # Pattern to match q-point in format [qx,qy,qz]
    pattern = r"P1\[([^,]+),([^,]+),([^\]]+)\]"
    match = re.search(pattern, mode_label)

    if not match:
        raise ValueError(f"Could not extract q-point from mode label: {mode_label}")

    qx_str, qy_str, qz_str = match.groups()

    # Evaluate fractions (e.g., "1/2" -> 0.5)
    def eval_fraction(s: str) -> float:
        s = s.strip()
        if "/" in s:
            num, den = s.split("/")
            return float(num) / float(den)
        return float(s)

    qx = eval_fraction(qx_str)
    qy = eval_fraction(qy_str)
    qz = eval_fraction(qz_str)

    return (qx, qy, qz)


def group_modes_by_qpoint(modes: Dict) -> Dict[Tuple[float, float, float], List[str]]:
    """
    Group ISODISTORT modes by their q-point.

    Parameters
    ----------
    modes : dict
        Dictionary of modes from parse_isodistort_file

    Returns
    -------
    dict
        Dictionary mapping q-point tuples to lists of mode labels

    Examples
    --------
    >>> result = parse_isodistort_file('P4mmm-ref.txt')
    >>> qpoint_groups = group_modes_by_qpoint(result['modes'])
    >>> for qpoint, mode_list in sorted(qpoint_groups.items()):
    ...     print(f"Q-point {qpoint}: {len(mode_list)} modes")
    """
    qpoint_groups: Dict[Tuple[float, float, float], List[str]] = {}

    for mode_label in modes.keys():
        try:
            qpoint = extract_qpoint_from_mode_label(mode_label)
            if qpoint not in qpoint_groups:
                qpoint_groups[qpoint] = []
            qpoint_groups[qpoint].append(mode_label)
        except ValueError:
            # Skip modes that don't have extractable q-points
            continue

    return qpoint_groups
