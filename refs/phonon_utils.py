"""
Phonopy utility functions.

This module provides functions for working with Phonopy calculations,
including loading YAML files, creating Phonopy objects, and calculating
phonon band structures directly from Phonopy data.
"""

import numpy as np
from typing import Union, Optional, Tuple
from pathlib import Path
from ase import Atoms

try:
    from phonopy import Phonopy, load
    PHONOPY_AVAILABLE = True
except ImportError:
    PHONOPY_AVAILABLE = False

from .core.band_structure import PhononBand


def load_from_phonopy_files(
    directory: str,
    force_sets_file: str = "FORCE_SETS",
    phonopy_yaml_file: str = "phonopy.yaml",
    poscar_file: str = "POSCAR"
) -> dict:
    """
    Load Phonopy calculation from individual files using Phonopy I/O.

    This function provides an alternative to load_yaml_file() for cases where
    Phonopy calculations are stored in separate files in a directory.
    It relies on Phonopy's built-in error handling and file discovery.

    Parameters
    ----------
    directory : str
        Path to the directory containing the Phonopy calculation files
    force_sets_file : str, optional
        Name of the FORCE_SETS file (default: "FORCE_SETS")
    phonopy_yaml_file : str, optional
        Name of the phonopy.yaml file (default: "phonopy.yaml")
    poscar_file : str, optional
        Name of the POSCAR file (default: "POSCAR")

    Returns
    -------
    dict
        Dictionary containing loaded data with keys:
        - 'phonopy': Phonopy object
        - 'primitive_cell': ASE Atoms object for primitive cell
        - 'unitcell': ASE Atoms object for unit cell
        - 'supercell': ASE Atoms object for supercell
        - 'directory': Path to the calculation directory

    Raises
    ------
    ImportError
        If Phonopy is not available
    """
    if not PHONOPY_AVAILABLE:
        raise ImportError("Phonopy is required for loading Phonopy files. Install with: pip install phonopy")

    from pathlib import Path
    from phonopy import load

    directory = Path(directory)
    phonopy_yaml_path = directory / phonopy_yaml_file

    # Load Phonopy object from yaml file with FORCE_SETS
    force_sets_path = directory / force_sets_file
    if force_sets_path.exists():
        # Use Phonopy's built-in force_sets_filename parameter
        phonopy = load(
            str(phonopy_yaml_path),
            force_sets_filename=str(force_sets_path)
        )
    else:
        # Load without FORCE_SETS if file doesn't exist
        phonopy = load(str(phonopy_yaml_path))

    # Convert Phonopy structures to ASE Atoms for consistency
    try:
        from .core.io import phonopy_to_ase
        primitive_cell_ase = phonopy_to_ase(phonopy.primitive)
        unitcell_ase = phonopy_to_ase(phonopy.unitcell)
        supercell_ase = phonopy_to_ase(phonopy.supercell)
    except ImportError:
        # Fallback: create ASE Atoms directly
        from ase import Atoms

        def phonopy_to_ase_fallback(phonopy_cell):
            """Convert Phonopy cell to ASE Atoms."""
            return Atoms(
                symbols=phonopy_cell.symbols,
                positions=phonopy_cell.positions,
                cell=phonopy_cell.cell,
                pbc=True
            )

        primitive_cell_ase = phonopy_to_ase_fallback(phonopy.primitive)
        unitcell_ase = phonopy_to_ase_fallback(phonopy.unitcell)
        supercell_ase = phonopy_to_ase_fallback(phonopy.supercell)

    return {
        'phonopy': phonopy,
        'primitive_cell': primitive_cell_ase,
        'unitcell': unitcell_ase,
        'supercell': supercell_ase,
        'directory': str(directory)
    }


def load_yaml_file(yaml_path: str) -> dict:
    """
    Load Phonopy YAML file and return parsed data.

    Parameters
    ----------
    yaml_path : str
        Path to the Phonopy YAML file

    Returns
    -------
    dict
        Parsed YAML data

    Raises
    ------
    FileNotFoundError
        If YAML file doesn't exist
    ImportError
        If Phonopy is not available
    """
    if not PHONOPY_AVAILABLE:
        raise ImportError("Phonopy is required for loading YAML files. Install with: pip install phonopy")

    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")

    # Load the Phonopy object from YAML
    phonopy = load(str(yaml_path))

    return {
        'phonopy': phonopy,
        'primitive_cell': phonopy.primitive,
        'unitcell': phonopy.unitcell,
        'supercell': phonopy.supercell
    }


def create_phonopy_object(yaml_path: str) -> 'Phonopy':
    """
    Create Phonopy object from YAML file.

    Parameters
    ----------
    yaml_path : str
        Path to the Phonopy YAML file

    Returns
    -------
    Phonopy
        Phonopy object

    Raises
    ------
    FileNotFoundError
        If YAML file doesn't exist
    ImportError
        If Phonopy is not available
    """
    yaml_data = load_yaml_file(yaml_path)
    return yaml_data['phonopy']


def calculate_band_structure_from_phonopy(
    data_source: Union[str, 'Phonopy'],
    path: str = 'GMXMG',
    npoints: int = 50,
    units: str = 'cm-1'
) -> PhononBand:
    """
    Calculate phonon band structure from Phonopy data.

    This function can accept either:
    - A path to a Phonopy YAML file
    - A Phonopy object directly

    Parameters
    ----------
    data_source : str or Phonopy
        Either a path to a Phonopy YAML file or a Phonopy object
    path : str, optional
        High-symmetry k-path (e.g., 'GMXMG', 'Γ-X-M-Γ-R')
    npoints : int, optional
        Number of k-points along the path
    units : str, optional
        Frequency units ('THz', 'cm-1', 'meV')

    Returns
    -------
    PhononBand
        Calculated band structure

    Raises
    ------
    ValueError
        If data_source is not a valid path or Phonopy object
    ImportError
        If Phonopy is not available
    """
    if not PHONOPY_AVAILABLE:
        raise ImportError("Phonopy is required for band structure calculations. Install with: pip install phonopy")

    # Get Phonopy object from data source
    if isinstance(data_source, str):
        # Load from YAML file
        phonopy = create_phonopy_object(data_source)
    else:
        # Assume it's already a Phonopy object
        phonopy = data_source

    # Generate k-path using existing kpath functionality
    from .core.kpath import auto_kpath

    # Convert Phonopy primitive cell to ASE Atoms
    try:
        from .core.io import phonopy_to_ase
        primitive_cell = phonopy_to_ase(phonopy.primitive)
    except ImportError:
        # Fallback: create ASE Atoms directly from Phonopy data
        from ase import Atoms
        primitive_cell = Atoms(
            symbols=phonopy.primitive.symbols,
            positions=phonopy.primitive.positions,
            cell=phonopy.primitive.cell,
            pbc=True
        )

    # Generate k-path
    xlist, kptlist, Xs, knames, spk = auto_kpath(
        primitive_cell.get_cell(),
        path=path,
        npoints=npoints
    )

    # Combine all k-points
    kpoints = np.concatenate(kptlist)

    # Calculate phonons directly at each k-point
    frequencies, eigenvectors = _calculate_phonons_at_kpoints(phonopy, kpoints)

    # Convert units if needed
    if units != 'THz':
        frequencies = _convert_frequencies(frequencies, units)

    # Create k-path labels
    kpath_labels = _create_kpath_labels(xlist, kptlist, knames, spk)

    # Create k-path data
    kpath_data = {
        'path': path,
        'npoints': npoints,
        'special_points': spk,
        'xcoords': xlist,
        'segments': kptlist,
        'kpath_labels': kpath_labels,
        'units': units
    }

    # Create atomic masses
    atomic_masses = np.array(phonopy.primitive.masses)

    return PhononBand(
        primitive_cell=primitive_cell,
        qpoints=kpoints,
        frequencies=frequencies,
        eigenvectors=eigenvectors,
        atomic_masses=atomic_masses,
        kpath_data=kpath_data
    )


def _calculate_phonons_at_kpoints(phonopy: 'Phonopy', kpoints: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate phonon frequencies and eigenvectors at specified k-points.

    Parameters
    ----------
    phonopy : Phonopy
        Phonopy object
    kpoints : np.ndarray
        Array of k-point coordinates, shape (n_kpoints, 3)

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Frequencies (THz) and eigenvectors for all k-points
    """
    n_kpoints = len(kpoints)
    n_atoms = len(phonopy.primitive)
    n_modes = n_atoms * 3

    frequencies = np.zeros((n_kpoints, n_modes))
    eigenvectors = np.zeros((n_kpoints, n_modes, n_modes), dtype=complex)

    for i, q in enumerate(kpoints):
        # Get dynamical matrix at q-point
        dm = phonopy.get_dynamical_matrix_at_q(q)

        # Solve eigenvalue problem
        eigenvalues, eigenvectors_q = np.linalg.eigh(dm)

        # Convert eigenvalues to frequencies
        frequencies[i] = np.sqrt(np.abs(eigenvalues)) * np.sign(eigenvalues) * phonopy.unit_conversion_factor

        # Transpose eigenvectors to get correct shape
        eigenvectors[i] = eigenvectors_q.T

    return frequencies, eigenvectors


def _convert_frequencies(frequencies: np.ndarray, target_units: str) -> np.ndarray:
    """Convert frequencies between different units."""
    if target_units == 'THz':
        return frequencies
    elif target_units == 'cm-1':
        return frequencies * 33.356
    elif target_units == 'meV':
        return frequencies * 4.1357
    else:
        raise ValueError(f"Unsupported units: {target_units}")


def _create_kpath_labels(xlist, kptlist, knames, spk):
    """Create k-path labels for plotting."""
    current_pos = 0
    kpath_labels = []

    for i, (x, k) in enumerate(zip(xlist, kptlist)):
        for name in knames:
            # Find matches for this special point in the current segment
            matches = np.where((k == spk[name]).all(axis=1))[0]
            if matches.size > 0:
                # For each match, add the global index
                for match_idx in matches:
                    global_idx = match_idx + current_pos
                    kpath_labels.append((global_idx, name))
        current_pos += len(k)

    # Sort labels by index and remove duplicates at the same index
    kpath_labels.sort(key=lambda x: x[0])

    # Remove duplicate labels at the same index, keeping the first occurrence
    seen_indices = set()
    filtered_labels = []
    for idx, name in kpath_labels:
        if idx not in seen_indices:
            seen_indices.add(idx)
            filtered_labels.append((idx, name))

    return filtered_labels