"""
Structure Analysis and Displacement Projection

Provides functions for analyzing and comparing crystal structures,
finding atomic correspondences, and projecting displacement patterns
between different structural configurations.

Functions:
    find_nearest_atoms: Find nearest atoms between structures with periodic boundaries
    create_atom_mapping: Create optimal atom correspondence between structures
    reorder_structure_by_mapping: Reorder structures based on atom mapping
    project_displacement: Project eigenvectors from one supercell to another

This implementation strictly follows user requirements using ASE functions
and exact mathematical formulations.
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
    symprec: float = 1e-5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Find nearest atoms in structure2 for each atom in structure1 with periodic boundaries.

    Uses ASE ase.geometry.get_distances with mic=True for minimum image convention.

    Args:
        structure1: Reference structure
        structure2: Target structure to search in
        cell1: Unit cell for structure1 (default: structure1.get_cell())
        cell2: Unit cell for structure2 (default: structure2.get_cell())
        symprec: Symmetry precision (default: 1e-5)

    Returns:
        Tuple of (indices, distances, vectors):
            indices: Array of indices of nearest atoms in structure2
            distances: Array of distances to nearest atoms
            vectors: Periodic image vectors from structure1 to structure2

    Example:
        >>> from ase.build import bulk
        >>> struct1 = bulk('Si', 'diamond', a=5.43)
        >>> struct2 = bulk('Si', 'diamond', a=5.44)  # Slightly different
        >>> indices, dists, vecs = find_nearest_atoms(struct1, struct2)
        >>> print(f"Max distance: {np.max(dists):.6f} Å")
    """
    # Use default cells if not provided
    if cell1 is None:
        cell1 = structure1.get_cell()
    if cell2 is None:
        cell2 = structure2.get_cell()

    # Use ASE's get_distances function with cell and pbc for periodic boundaries (MIC applied automatically)
    indices, distances = get_distances(
        structure1.positions,
        structure2.positions,
        cell=cell2,
        pbc=True
    )

    # Extract indices and minimum distances
    min_indices = np.argmin(distances, axis=1)
    min_distances = np.min(distances, axis=1)

    # Calculate periodic vectors
    # Vector from atom i in structure1 to its nearest image in structure2
    pos1 = structure1.positions
    pos2 = structure2.positions[min_indices]

    periodic_vectors = []
    for i, (idx, p1, p2) in enumerate(zip(min_indices, pos1, pos2)):
        # Find periodic image that gives minimum distance
        disp = p2 - p1

        # Apply periodic boundary conditions
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
    method: str = 'distance',
    max_cost: float = 1e-3,
    symprec: float = 1e-5
) -> Tuple[np.ndarray, float]:
    """
    Create optimal atom correspondence between two structures.

    Uses distance-based matching to find which atoms in structure2
    correspond to which atoms in structure1.

    Args:
        structure1: Reference structure
        structure2: Target structure
        method: Matching method ('distance' only for now)
        max_cost: Maximum allowed cost for matching
        symprec: Symmetry precision

    Returns:
        Tuple of (mapping, cost):
            mapping: Array where mapping[i] = j means atom i in struct1 maps to atom j in struct2
            cost: Total cost (sum of distances) of mapping

    Example:
        >>> mapping, cost = create_atom_mapping(struct1, struct2)
        >>> print(f"Mapping cost: {cost:.6f} Å")
    """
    if method != 'distance':
        raise ValueError("Only 'distance' method is currently implemented")

    if len(structure1) != len(structure2):
        raise ValueError("Structures must have same number of atoms")

    # Find nearest atoms
    indices, distances, _ = find_nearest_atoms(structure1, structure2, symprec=symprec)

    # Calculate total cost
    total_cost = np.sum(distances)

    if total_cost > max_cost:
        raise ValueError(f"Mapping cost {total_cost:.6f} exceeds maximum allowed {max_cost}")

    # Create mapping array
    mapping = indices

    return mapping, total_cost


def reorder_structure_by_mapping(
    structure: Atoms,
    mapping: np.ndarray
) -> Atoms:
    """
    Reorder atoms in structure according to provided mapping.

    Args:
        structure: Structure to reorder
        mapping: Array where mapping[i] = j means new position i should have
                 atom that was originally at position j

    Returns:
        Reordered Atoms object

    Example:
        >>> reordered = reorder_structure_by_mapping(structure, mapping)
        >>> print(f"Reordered {len(reordered)} atoms")
    """
    if len(mapping) != len(structure):
        raise ValueError("Mapping length must equal structure length")

    # Create reordered structure
    reordered = copy.deepcopy(structure)

    # Reorder atoms, positions, and all properties
    original_indices = np.arange(len(structure))
    reordered_indices = np.argsort(mapping)

    reordered.positions = reordered.positions[reordered_indices]
    reordered.numbers = reordered.numbers[reordered_indices]

    # Reorder all arrays in the Atoms object (except numbers and positions which are handled above)
    for key, array in reordered.arrays.items():
        if key not in ['numbers', 'positions']:
            reordered.arrays[key] = array[reordered_indices]

    return reordered


def project_displacement(
    source_modes: np.ndarray,
    target_displacement: np.ndarray,
    source_supercell: Atoms,
    target_supercell: Atoms,
    atom_mapping: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project mass-normalized eigen displacements from source supercell to target supercell displacement.

    The source_modes should contain mass-normalized eigen displacements (u_i = e_i/sqrt(m)),
    which are orthonormal under the mass-weighted inner product.

    Args:
        source_modes: Mass-normalized eigen displacement matrix (n_atoms, n_modes*3)
        target_displacement: Target displacement pattern (n_target_atoms, 3)
        source_supercell: Source supercell structure
        target_supercell: Target supercell structure
        atom_mapping: Optional mapping from source to target atoms

    Returns:
        Tuple of (coefficients, projected_displacement):
            coefficients: Projection coefficients for each mode
            projected_displacement: Reconstructed displacement pattern

    Example:
        >>> coeffs, projected = project_displacement(mass_norm_eigvecs, target_disp, supercell1, supercell2)
        >>> print(f"Coefficients shape: {coeffs.shape}")
    """
    # Get atom numbers
    n_source_atoms = len(source_supercell)
    n_target_atoms = len(target_supercell)

    # Create atom mapping if not provided
    if atom_mapping is None:
        atom_mapping, _ = create_atom_mapping(source_supercell, target_supercell, max_cost=10.0)

    # Check dimensions
    if target_displacement.shape[0] != n_target_atoms:
        raise ValueError("Target displacement must match target supercell atom count")

    if source_modes.shape[1] % 3 != 0:
        raise ValueError("Source modes must have shape (n_atoms, n_modes*3) - second dimension must be divisible by 3")

    # Calculate number of modes
    n_modes = source_modes.shape[1] // 3

    # Map mass-normalized eigen displacements from source to target atoms
    A = np.zeros((n_target_atoms, 3, n_modes))

    # Invert the mapping: atom_mapping[source_idx] = target_idx, so we need to find source_idx for each target_idx
    inverse_mapping = np.zeros(n_target_atoms, dtype=int)
    for source_idx, target_idx in enumerate(atom_mapping):
        inverse_mapping[target_idx] = source_idx

    for target_idx in range(n_target_atoms):
        source_idx = inverse_mapping[target_idx]
        # Reshape the displacement vector for this atom: (n_modes*3) -> (n_modes, 3)
        atom_modes = source_modes[source_idx, :].reshape(n_modes, 3)
        A[target_idx, :, :] = atom_modes.T  # Transpose to get (3, n_modes)

    # Calculate projection coefficients using mass-weighted inner product
    # For mass-normalized eigen displacements: coefficient = <ui|d>_M / (||ui||_M * ||d||_M)
    coefficients = np.zeros(n_modes)

    # Get atomic masses directly from target supercell
    # This is more reliable than mapping from source supercell masses
    target_atomic_masses = target_supercell.get_masses()
    target_masses = np.repeat(target_atomic_masses, 3)
    target_flat = target_displacement.ravel()

    # Calculate target mass-weighted norm once
    target_mass_norm = np.sqrt(np.sum(target_masses * target_flat.conj() * target_flat))

    for mode_idx in range(n_modes):
        # Extract mode displacement for all target atoms: shape (n_target_atoms, 3)
        mode_displacement = A[:, :, mode_idx]
        mode_flat = mode_displacement.ravel()

        # Calculate mode mass-weighted norm
        mode_mass_norm = np.sqrt(np.sum(target_masses * mode_flat.conj() * mode_flat))

        # Calculate mass-weighted projection
        projection = np.sum(target_masses * mode_flat.conj() * target_flat)

        # Normalized coefficient
        if mode_mass_norm > 0 and target_mass_norm > 0:
            coeff = projection / (mode_mass_norm * target_mass_norm)
        else:
            coeff = 0.0

        coefficients[mode_idx] = coeff

    # Calculate projected displacement: sum over all modes with their coefficients
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
    n_phases: int = 360
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scan projection coefficients as a function of phase angle.

    Generates displacement patterns with varying phase angles using the existing
    generate_mode_displacement method and calculates projection coefficients.

    Args:
        source_modes: PhononModes object containing source phonon data
        target_displacement: Target displacement pattern (n_target_atoms, 3)
        source_supercell: Source supercell structure
        target_supercell: Target supercell structure
        supercell_matrix: 3x3 supercell transformation matrix
        atom_mapping: Optional mapping from source to target atoms
        n_phases: Number of phase points to sample (default: 360)

    Returns:
        Tuple of (phases, coefficients):
            phases: Array of phase angles in radians from 0 to 2π
            coefficients: Array of projection coefficients at each phase angle

    Example:
        >>> phases, coeffs = project_displacement_with_phase_scan(
        ...     modes, target_disp, supercell1, supercell2, supercell_matrix, n_phases=180
        ... )
        >>> print(f"Generated {len(phases)} phase points")
    """
    # Generate phase array from 0 to 2π radians
    phases = np.linspace(0, 2 * np.pi, n_phases)
    coefficients = np.zeros(n_phases)

    # Need to specify which mode to use - assume first mode at first q-point
    q_index = 0
    mode_index = 0

    # Apply phase scan using existing generate_mode_displacement method
    for i, phase in enumerate(phases):
        # Generate displacement with phase using existing method
        source_displacement = source_modes.generate_mode_displacement(
            q_index=q_index,
            mode_index=mode_index,
            supercell_matrix=supercell_matrix,
            argument=phase  # Phase in radians
        )

        # Use existing projection function
        coeff, _ = project_displacement(
            source_displacement, target_displacement,
            source_supercell, target_supercell, atom_mapping
        )

        # For single mode case, take the first coefficient
        # If multiple modes, take the maximum absolute coefficient
        if len(coeff) == 1:
            coefficients[i] = coeff[0]
        else:
            coefficients[i] = np.max(np.abs(coeff))

    return phases, coefficients


def find_maximum_projection(
    phases: np.ndarray,
    coefficients: np.ndarray
) -> Tuple[float, float]:
    """
    Find the maximum absolute projection coefficient and corresponding phase angle.

    Args:
        phases: Array of phase angles in radians
        coefficients: Array of projection coefficients at each phase angle

    Returns:
        Tuple of (max_coefficient, optimal_phase):
            max_coefficient: Maximum absolute projection coefficient
            optimal_phase: Phase angle corresponding to maximum coefficient

    Example:
        >>> max_coeff, optimal_phase = find_maximum_projection(phases, coeffs)
        >>> print(f"Maximum: {max_coeff:.6f} at phase {optimal_phase:.3f} rad")
    """
    # Find maximum absolute coefficient
    abs_coefficients = np.abs(coefficients)
    max_idx = np.argmax(abs_coefficients)

    max_coefficient = abs_coefficients[max_idx]
    optimal_phase = phases[max_idx]

    return max_coefficient, optimal_phase
