"""
Structure Analysis and Displacement Projection

Provides functions for analyzing and comparing crystal structures,
finding atomic correspondences, and projecting displacement patterns
between different structural configurations.
"""

import numpy as np
from typing import Tuple, Optional, List
from ase import Atoms
from ase.geometry import get_distances
import copy


def find_nearest_atoms(
    structure1: Atoms,
    structure2: Atoms,
    cell1: Optional[np.ndarray] = None,
    cell2: Optional[np.ndarray] = None,
    symprec: float = 1e-5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if cell1 is None:
        cell1 = np.array(structure1.get_cell())
    if cell2 is None:
        cell2 = np.array(structure2.get_cell())

    indices, distances = get_distances(
        structure1.positions, structure2.positions, cell=cell2, pbc=True
    )

    min_indices = np.argmin(distances, axis=1)
    min_distances = np.min(distances, axis=1)

    pos1 = structure1.positions
    pos2 = structure2.positions[min_indices]

    periodic_vectors = []
    for i, (idx, p1, p2) in enumerate(zip(min_indices, pos1, pos2)):
        disp = p2 - p1
        cell2_recip = np.linalg.inv(cell2)
        frac_disp = np.dot(disp, cell2_recip)
        frac_disp = frac_disp - np.round(frac_disp)
        periodic_vec = np.dot(frac_disp, cell2)
        periodic_vectors.append(periodic_vec)

    periodic_vectors = np.array(periodic_vectors)

    return min_indices, min_distances, periodic_vectors


def create_atom_mapping(
    structure1: Atoms,
    structure2: Atoms,
    method: str = "distance",
    max_cost: float = 1e-3,
    symprec: float = 1e-5,
) -> Tuple[np.ndarray, float]:
    if method != "distance":
        raise ValueError("Only 'distance' method is currently implemented")

    if len(structure1) != len(structure2):
        raise ValueError("Structures must have same number of atoms")

    indices, distances, _ = find_nearest_atoms(structure1, structure2, symprec=symprec)
    total_cost = np.sum(distances)

    if total_cost > max_cost:
        raise ValueError(
            f"Mapping cost {total_cost:.6f} exceeds maximum allowed {max_cost}"
        )

    mapping = indices
    return mapping, total_cost


def reorder_structure_by_mapping(structure: Atoms, mapping: np.ndarray) -> Atoms:
    if len(mapping) != len(structure):
        raise ValueError("Mapping length must equal structure length")

    reordered = copy.deepcopy(structure)
    original_indices = np.arange(len(structure))
    reordered_indices = np.argsort(mapping)

    reordered.positions = reordered.positions[reordered_indices]
    reordered.numbers = reordered.numbers[reordered_indices]

    for key, array in reordered.arrays.items():
        if key not in ["numbers", "positions"]:
            reordered.arrays[key] = array[reordered_indices]

    return reordered


def project_displacement(
    source_modes: np.ndarray,
    target_displacement: np.ndarray,
    source_supercell: Atoms,
    target_supercell: Atoms,
    atom_mapping: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    n_source_atoms = len(source_supercell)
    n_target_atoms = len(target_supercell)

    if atom_mapping is None:
        atom_mapping, _ = create_atom_mapping(
            source_supercell, target_supercell, max_cost=10.0
        )

    if target_displacement.shape[0] != n_target_atoms:
        raise ValueError("Target displacement must match target supercell atom count")

    if source_modes.shape[1] % 3 != 0:
        raise ValueError(
            "Source modes must have shape (n_atoms, n_modes*3) - second dimension must be divisible by 3"
        )

    n_modes = source_modes.shape[1] // 3
    A = np.zeros((n_target_atoms, 3, n_modes))

    inverse_mapping = np.zeros(n_target_atoms, dtype=int)
    for source_idx, target_idx in enumerate(atom_mapping):
        inverse_mapping[target_idx] = source_idx

    for target_idx in range(n_target_atoms):
        source_idx = inverse_mapping[target_idx]
        atom_modes = source_modes[source_idx, :].reshape(n_modes, 3)
        A[target_idx, :, :] = atom_modes.T

    coefficients = np.zeros(n_modes)
    target_atomic_masses = target_supercell.get_masses()
    target_masses = np.repeat(target_atomic_masses, 3)
    target_flat = target_displacement.ravel()
    target_mass_norm = np.sqrt(np.sum(target_masses * target_flat.conj() * target_flat))

    for mode_idx in range(n_modes):
        mode_displacement = A[:, :, mode_idx]
        mode_flat = mode_displacement.ravel()
        mode_mass_norm = np.sqrt(np.sum(target_masses * mode_flat.conj() * mode_flat))
        projection = np.sum(target_masses * mode_flat.conj() * target_flat)
        if mode_mass_norm > 0 and target_mass_norm > 0:
            coeff = projection / (mode_mass_norm * target_mass_norm)
        else:
            coeff = 0.0
        coefficients[mode_idx] = coeff

    projected_displacement = np.zeros_like(target_displacement)
    for mode_idx in range(n_modes):
        mode_contribution = coefficients[mode_idx] * A[:, :, mode_idx]
        projected_displacement += mode_contribution

    return coefficients, projected_displacement


def project_displacement_with_phase_scan(
    source_modes,  # PhononModes object
    target_displacement: np.ndarray,
    source_supercell: Atoms,
    target_supercell: Atoms,
    supercell_matrix: np.ndarray,
    atom_mapping: Optional[np.ndarray] = None,
    n_phases: int = 360,
) -> Tuple[np.ndarray, np.ndarray]:
    phases = np.linspace(0, 2 * np.pi, n_phases)
    coefficients = np.zeros(n_phases)
    q_index = 0
    mode_index = 0
    for i, phase in enumerate(phases):
        source_displacement = source_modes.generate_mode_displacement(
            q_index=q_index,
            mode_index=mode_index,
            supercell_matrix=supercell_matrix,
            argument=phase,
        )
        coeff, _ = project_displacement(
            source_displacement,
            target_displacement,
            source_supercell,
            target_supercell,
            atom_mapping,
        )
        if len(coeff) == 1:
            coefficients[i] = coeff[0]
        else:
            coefficients[i] = np.max(np.abs(coeff))
    return phases, coefficients


def find_maximum_projection(
    phases: np.ndarray, coefficients: np.ndarray
) -> Tuple[float, float]:
    abs_coefficients = np.abs(coefficients)
    max_idx = np.argmax(abs_coefficients)
    max_coefficient = abs_coefficients[max_idx]
    optimal_phase = phases[max_idx]
    return max_coefficient, optimal_phase
