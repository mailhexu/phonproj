import numpy as np
from typing import Union, Optional, Tuple, TYPE_CHECKING, Any
from pathlib import Path

try:
    from phonopy import Phonopy, load as phonopy_load

    PHONOPY_AVAILABLE = True
except ImportError:
    PHONOPY_AVAILABLE = False
    phonopy_load = None  # type: ignore
    Phonopy = None  # type: ignore


def load_yaml_file(yaml_path: Path) -> dict:
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
        raise ImportError(
            "Phonopy is required for loading YAML files. Install with: pip install phonopy"
        )

    if isinstance(yaml_path, str):
        yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")

    # Load the Phonopy object from YAML
    phonopy = phonopy_load(str(yaml_path))  # type: ignore

    # Convert Phonopy cells to ASE Atoms objects
    primitive_cell_ase = phonopy_to_ase(phonopy.primitive)
    unitcell_ase = phonopy_to_ase(phonopy.unitcell)
    supercell_ase = phonopy_to_ase(phonopy.supercell)

    return {
        "phonopy": phonopy,
        "primitive_cell": primitive_cell_ase,
        "unitcell": unitcell_ase,
        "supercell": supercell_ase,
    }


def create_phonopy_object(yaml_path: Path) -> Any:
    if isinstance(yaml_path, str):
        yaml_path = Path(yaml_path)
    if not PHONOPY_AVAILABLE:
        raise ImportError(
            "Phonopy is required for loading YAML files. Install with: pip install phonopy"
        )
    phonopy = phonopy_load(str(yaml_path))  # type: ignore
    return phonopy


def phonopy_to_ase(phonopy_cell: Any) -> Any:
    """
    Convert a Phonopy cell to an ASE Atoms object.
    """
    from ase import Atoms

    return Atoms(
        symbols=phonopy_cell.symbols,
        positions=phonopy_cell.positions,
        cell=phonopy_cell.cell,
        pbc=True,
        masses=getattr(phonopy_cell, "masses", None),
    )


def load_from_phonopy_files(
    directory: Path,
    force_sets_file: str = "FORCE_SETS",
    phonopy_yaml_file: str = "phonopy.yaml",
    poscar_file: str = "POSCAR",
) -> dict:
    if not PHONOPY_AVAILABLE:
        raise ImportError(
            "Phonopy is required for loading Phonopy files. Install with: pip install phonopy"
        )
    if isinstance(directory, str):
        directory = Path(directory)
    phonopy_yaml_path = directory / phonopy_yaml_file
    force_sets_path = directory / force_sets_file
    if force_sets_path.exists():
        phonopy = phonopy_load(
            str(phonopy_yaml_path), force_sets_filename=str(force_sets_path)
        )  # type: ignore
    else:
        phonopy = phonopy_load(str(phonopy_yaml_path))  # type: ignore
    primitive_cell_ase = phonopy_to_ase(phonopy.primitive)
    unitcell_ase = phonopy_to_ase(phonopy.unitcell)
    supercell_ase = phonopy_to_ase(phonopy.supercell)
    return {
        "phonopy": phonopy,
        "primitive_cell": primitive_cell_ase,
        "unitcell": unitcell_ase,
        "supercell": supercell_ase,
        "directory": str(directory),
    }


def _calculate_phonons_at_kpoints(
    phonopy: Any, kpoints: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
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
    masses = np.array(phonopy.primitive.masses)
    sqrt_masses: np.ndarray = np.repeat(np.sqrt(masses), 3)
    for i, q in enumerate(kpoints):
        dm = phonopy.get_dynamical_matrix_at_q(q)
        eigenvalues, eigenvectors_q = np.linalg.eigh(dm)
        frequencies[i] = (
            np.sqrt(np.abs(eigenvalues))
            * np.sign(eigenvalues)
            * phonopy.unit_conversion_factor
        )
        # Store eigenvectors directly without mass-weighting or individual normalization
        # The eigenvectors from eigh are already orthonormal in the appropriate space
        eigenvectors[i] = eigenvectors_q.T  # Transpose to get shape (n_modes, n_modes)
    return frequencies, eigenvectors
