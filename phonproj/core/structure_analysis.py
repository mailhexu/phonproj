import numpy as np
from typing import Tuple, Optional, List, Union
from ase import Atoms
from ase.cell import Cell
from ase.geometry import get_distances
import copy


def project_out_acoustic_modes(
    displacement: np.ndarray,
    structure: Atoms,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project out the three acoustic translation modes from a displacement vector.

    This removes all collective translations (x, y, z directions) using mass-weighted
    projection onto the three acoustic mode directions.

    Args:
        displacement: Displacement pattern (n_atoms, 3) in Cartesian coordinates
        structure: Reference structure for mass weighting

    Returns:
        Tuple of:
        - projected_displacement: Displacement with acoustic modes removed (n_atoms, 3)
        - acoustic_projections: Projections onto the 3 acoustic modes (3,)
    """
    n_atoms = len(structure)
    masses = structure.get_masses()

    if displacement.shape != (n_atoms, 3):
        raise ValueError(
            f"Displacement shape {displacement.shape} doesn't match structure with {n_atoms} atoms"
        )

    # Create the three acoustic mode eigenvectors (uniform translations in x, y, z)
    # Each acoustic mode is a uniform translation of all atoms in one direction
    # Mass-weighted normalization: each atom displacement is weighted by sqrt(m)
    acoustic_modes = []
    for direction in range(3):
        mode = np.zeros((n_atoms, 3))
        mode[:, direction] = 1.0  # Unit displacement in this direction

        # Mass-weight normalize: multiply by sqrt(mass) for each atom
        mode_mass_weighted = mode * np.sqrt(masses)[:, np.newaxis]

        # Normalize
        norm = np.linalg.norm(mode_mass_weighted)
        if norm > 0:
            mode_mass_weighted /= norm

        acoustic_modes.append(mode_mass_weighted)

    # Project displacement onto each acoustic mode and subtract
    displacement_copy = displacement.copy()
    acoustic_projections = np.zeros(3)

    # Mass-weight the displacement for projection
    displacement_mass_weighted = displacement * np.sqrt(masses)[:, np.newaxis]

    for i, acoustic_mode in enumerate(acoustic_modes):
        # Calculate projection coefficient (already normalized)
        projection = np.sum(displacement_mass_weighted * acoustic_mode)
        acoustic_projections[i] = projection

        # Subtract this component from displacement
        # Need to un-mass-weight the acoustic mode for subtraction
        acoustic_displacement = acoustic_mode / np.sqrt(masses)[:, np.newaxis]
        displacement_copy -= projection * acoustic_displacement

    return displacement_copy, acoustic_projections


def find_nearest_atoms(
    structure1: Atoms,
    structure2: Atoms,
    cell1: Optional[Union[np.ndarray, Cell]] = None,
    cell2: Optional[Union[np.ndarray, Cell]] = None,
    symprec: float = 1e-5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if cell1 is None:
        cell1 = structure1.get_cell()
    if cell2 is None:
        cell2 = structure2.get_cell()

    # Convert Cell objects to numpy arrays if needed
    if isinstance(cell1, Cell):
        cell1_array = np.array(cell1)
    else:
        cell1_array = cell1

    if isinstance(cell2, Cell):
        cell2_array = np.array(cell2)
    else:
        cell2_array = cell2

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
        cell2_recip = np.linalg.inv(cell2_array)
        frac_disp = np.dot(disp, cell2_recip)
        frac_disp = frac_disp - np.round(frac_disp)
        periodic_vec = np.dot(frac_disp, cell2_array)
        periodic_vectors.append(periodic_vec)
    periodic_vectors = np.array(periodic_vectors)
    return min_indices, min_distances, periodic_vectors


def create_atom_mapping(
    structure1: Atoms,
    structure2: Atoms,
    method: str = "distance",
    max_cost: float = 1e-3,
    symprec: float = 1e-5,
    warn_threshold: float = 0.5,
    species_map: Optional[dict] = None,
) -> Tuple[np.ndarray, float]:
    """
    Create atom mapping between two structures.

    This function uses the Hungarian algorithm (linear sum assignment) to find
    the optimal mapping that minimizes total distance while respecting atomic species.
    It will find the nearest atom of the same species for each atom in structure1.

    Args:
        structure1: First structure (source)
        structure2: Second structure (target)
        method: Method for creating mapping (only 'distance' is implemented)
        max_cost: Maximum allowed total cost (triggers warning if exceeded)
        symprec: Symmetry precision (not used currently)
        warn_threshold: Distance threshold (Angstrom) for individual atom warnings
        species_map: Optional dict mapping species in structure1 to allowed species in structure2
                     e.g., {'Pb': 'Sr', 'Sr': 'Pb'} to allow Pb-Sr substitution
                     If None, only exact species matches are allowed

    Returns:
        Tuple of (mapping, total_cost) where mapping[i] gives the index in structure2
        that corresponds to atom i in structure1
    """
    if method != "distance":
        raise ValueError("Only 'distance' method is currently implemented")
    if len(structure1) != len(structure2):
        raise ValueError("Structures must have same number of atoms")

    # Import here to avoid hard dependency
    from scipy.optimize import linear_sum_assignment

    n = len(structure1)
    species1 = structure1.get_chemical_symbols()
    species2 = structure2.get_chemical_symbols()

    # Build cost matrix with species constraint
    cost_matrix = np.full((n, n), 1e10)  # Very high cost for mismatched species

    # Get positions and cell for PBC
    pos1 = structure1.get_positions()
    pos2 = structure2.get_positions()
    cell = structure2.get_cell().array
    cell_inv = np.linalg.inv(cell)

    # Helper function to check if species are compatible
    def species_compatible(sp1: str, sp2: str) -> bool:
        if sp1 == sp2:
            return True
        if species_map is not None:
            # Check if sp1 maps to sp2
            if species_map.get(sp1) == sp2:
                return True
            # Also check list of allowed species
            if isinstance(species_map.get(sp1), list) and sp2 in species_map[sp1]:
                return True
        return False

    # Calculate minimum image distances for compatible species pairs
    for i in range(n):
        for j in range(n):
            if species_compatible(species1[i], species2[j]):
                # Calculate minimum image distance
                disp_vec = pos2[j] - pos1[i]
                disp_vec -= np.round(disp_vec @ cell_inv) @ cell
                cost_matrix[i, j] = np.linalg.norm(disp_vec)

    # Solve assignment problem
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    mapping = col_ind
    total_cost = cost_matrix[row_ind, col_ind].sum()

    # Check for species mismatches that are NOT in the species_map
    mismatched = []
    for i in range(n):
        if species1[i] != species2[mapping[i]]:
            # Only warn if this mismatch is not explicitly allowed
            if not species_compatible(species1[i], species2[mapping[i]]):
                mismatched.append((i, species1[i], species2[mapping[i]]))

    if mismatched:
        print("\n⚠️  WARNING: Species mismatch detected in atom mapping!")
        print("This likely means the structures have different chemical compositions.")
        for i, sp1, sp2 in mismatched[:10]:  # Show first 10
            print(f"  Atom {i}: {sp1} (structure1) mapped to {sp2} (structure2)")
        if len(mismatched) > 10:
            print(f"  ... and {len(mismatched) - 10} more mismatches")

    # Warn about large displacements
    large_displacements = []
    for i in range(n):
        dist = cost_matrix[i, mapping[i]]
        if dist < 1e9 and dist > warn_threshold:  # Skip the 1e10 penalties
            large_displacements.append((i, species1[i], dist))

    if large_displacements:
        print(
            f"\n⚠️  WARNING: {len(large_displacements)} atoms have large displacements (>{warn_threshold} Å):"
        )
        # Sort by distance and show top 10
        large_displacements.sort(key=lambda x: x[2], reverse=True)
        for i, species, dist in large_displacements[:10]:
            print(f"  Atom {i} ({species}): {dist:.4f} Å")
        if len(large_displacements) > 10:
            print(f"  ... and {len(large_displacements) - 10} more")

    # Warn if total cost is large
    if total_cost > max_cost:
        print(
            f"\n⚠️  WARNING: Total mapping cost {total_cost:.6f} exceeds threshold {max_cost}"
        )
        print("The structures may be significantly different or poorly aligned.")

    return mapping, total_cost


def reorder_structure_by_mapping(structure: Atoms, mapping: np.ndarray) -> Atoms:
    if len(mapping) != len(structure):
        raise ValueError("Mapping length must equal structure length")
    reordered = copy.deepcopy(structure)
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
    phonon_modes,  # PhononModes object
    target_displacement: np.ndarray,
    supercell_matrix: np.ndarray,
    q_index: int,
    mode_index: int,
    n_phases: int = 36,
) -> Tuple[float, float]:
    """
    Project a displacement onto a single phonon mode across a range of phases,
    and return the maximum projection coefficient and corresponding optimal phase.

    Args:
        phonon_modes: PhononModes object containing phonon data.
        target_displacement: The displacement to project, shape (n_atoms, 3).
        supercell_matrix: The 3x3 supercell matrix.
        q_index: The index of the q-point to use.
        mode_index: The index of the mode to use.
        n_phases: The number of phase angles to sample between 0 and π.

    Returns:
        A tuple of (max_coefficient, optimal_phase), where max_coefficient
        is the maximum absolute projection coefficient found, and optimal_phase
        is the phase angle (in radians, 0 to π) that produces it.
    """
    from ..modes import create_supercell

    phases = np.linspace(0, np.pi, n_phases, endpoint=False)
    max_abs_coeff = 0.0
    optimal_phase = 0.0

    # Create the supercell structures needed for projection
    source_supercell = create_supercell(phonon_modes.primitive_cell, supercell_matrix)
    target_supercell = source_supercell  # In this case, they are the same

    # The target displacement should be normalized for this projection
    target_masses = np.repeat(target_supercell.get_masses(), 3)
    target_flat = target_displacement.ravel()
    mass_weighted_norm = np.sqrt(
        np.sum(target_masses * target_flat.conj() * target_flat)
    )
    if mass_weighted_norm > 1e-10:
        normalized_target_displacement = target_displacement / mass_weighted_norm
    else:
        normalized_target_displacement = target_displacement

    for phase in phases:
        # Generate the complex mode displacement for the current phase
        # This displacement is mass-weighted orthonormal
        source_displacement = phonon_modes.generate_mode_displacement(
            q_index=q_index,
            mode_index=mode_index,
            supercell_matrix=supercell_matrix,
            argument=phase,
            amplitude=1.0,  # Normalized
        )

        # Project the target displacement onto the phased mode displacement
        # Since both are normalized, we expect a value between -1 and 1
        coeff = project_displacements_between_supercells(
            source_displacement=source_displacement,
            target_displacement=normalized_target_displacement,
            source_supercell=source_supercell,
            target_supercell=target_supercell,
            atom_mapping=None,  # Mapping is identity here
            normalize=False,  # Both vectors are already normalized
            use_mass_weighting=True,
        )

        # Track the maximum absolute coefficient
        abs_coeff = abs(coeff)
        if abs_coeff > max_abs_coeff:
            max_abs_coeff = abs_coeff
            optimal_phase = phase

    return max_abs_coeff, optimal_phase


def find_maximum_projection(
    phases: np.ndarray, coefficients: np.ndarray
) -> Tuple[float, float]:
    abs_coefficients = np.abs(coefficients)
    max_idx = np.argmax(abs_coefficients)
    max_coefficient = abs_coefficients[max_idx]
    optimal_phase = phases[max_idx]
    return max_coefficient, optimal_phase


def project_displacements_between_supercells(
    source_displacement: np.ndarray,
    target_displacement: np.ndarray,
    source_supercell: Atoms,
    target_supercell: Atoms,
    atom_mapping: Optional[np.ndarray] = None,
    normalize: bool = True,
    use_mass_weighting: bool = True,
) -> float:
    """
    Project displacement from source supercell onto displacement in target supercell.

    This function implements Step 8 of the project plan: projection of displacements
    between two different supercells that may have different atom ordering and
    positions due to periodic boundary conditions.

    Args:
        source_displacement: Source displacement pattern (n_source_atoms, 3)
        target_displacement: Target displacement pattern (n_target_atoms, 3)
        source_supercell: Source supercell structure
        target_supercell: Target supercell structure
        atom_mapping: Optional mapping from source to target atoms
        normalize: Whether to normalize the projection coefficient
        use_mass_weighting: Whether to use mass-weighted inner product (True) or
                          regular inner product (False). Set to False when both
                          displacements are already mass-weighted normalized.

    Returns:
        float: Projection coefficient (normalized if normalize=True)

    Example:
        >>> coeff = project_displacements_between_supercells(
        ...     source_disp, target_disp, source_supercell, target_supercell
        ... )
        >>> print(f"Projection coefficient: {coeff:.6f}")
    """
    # Get atom numbers
    n_source_atoms = len(source_supercell)
    n_target_atoms = len(target_supercell)

    # Check dimensions
    if source_displacement.shape[0] != n_source_atoms:
        raise ValueError("Source displacement must match source supercell atom count")
    if target_displacement.shape[0] != n_target_atoms:
        raise ValueError("Target displacement must match target supercell atom count")

    # Create atom mapping if not provided
    if atom_mapping is None:
        atom_mapping, _ = create_atom_mapping(
            source_supercell, target_supercell, max_cost=100.0
        )

    # Map source displacement to target supercell atom ordering
    # Always use complex dtype to handle both real and complex displacements
    mapped_source_displacement = np.zeros(target_displacement.shape, dtype=complex)

    # Create inverse mapping: from target atoms to source atoms
    inverse_mapping = np.zeros(n_target_atoms, dtype=int)
    for source_idx, target_idx in enumerate(atom_mapping):
        inverse_mapping[target_idx] = source_idx

    # Map source displacement to target ordering
    for target_idx in range(n_target_atoms):
        source_idx = inverse_mapping[target_idx]
        mapped_source_displacement[target_idx] = source_displacement[source_idx]

    # Flatten displacements for inner product calculations
    mapped_source_flat = mapped_source_displacement.ravel()
    target_flat = target_displacement.ravel()

    # Calculate projection using appropriate inner product
    if use_mass_weighting:
        # Original mass-weighted projection
        target_atomic_masses = target_supercell.get_masses()
        target_masses = np.repeat(target_atomic_masses, 3)

        # Calculate mass-weighted inner product
        projection = np.sum(target_masses * mapped_source_flat.conj() * target_flat)

        if normalize:
            # Calculate mass-weighted norms
            source_norm = np.sqrt(
                np.sum(target_masses * mapped_source_flat.conj() * mapped_source_flat)
            )
            target_norm = np.sqrt(
                np.sum(target_masses * target_flat.conj() * target_flat)
            )

            # Normalized projection coefficient
            if source_norm > 0 and target_norm > 0:
                coefficient = projection / (source_norm * target_norm)
            else:
                coefficient = 0.0
        else:
            # Unnormalized projection (just the inner product)
            coefficient = projection
    else:
        # Regular (non-mass-weighted) projection
        # This should be used when both displacements are already mass-weighted normalized

        # Calculate regular inner product
        projection = np.sum(mapped_source_flat.conj() * target_flat)

        if normalize:
            # Calculate regular norms
            source_norm = np.sqrt(
                np.sum(mapped_source_flat.conj() * mapped_source_flat)
            )
            target_norm = np.sqrt(np.sum(target_flat.conj() * target_flat))

            # Normalized projection coefficient
            if source_norm > 0 and target_norm > 0:
                coefficient = projection / (source_norm * target_norm)
            else:
                coefficient = 0.0
        else:
            # Unnormalized projection (just the inner product)
            coefficient = projection

    return float(coefficient.real)  # Return real part as float


def decompose_displacement_to_modes(
    displacement: np.ndarray,
    phonon_modes,  # PhononModes object
    supercell_matrix: np.ndarray,
    normalize: bool = True,
    tolerance: float = 1e-12,
    sort_by_contribution: bool = True,
) -> Tuple[List[dict], dict]:
    """
    Decompose an arbitrary displacement into contributions from all phonon modes
    across all commensurate q-points in a supercell.

    This function implements Step 9 of the project plan: complete mode decomposition
    that projects any displacement vector onto the complete set of phonon modes
    for all commensurate q-points in a supercell.

    Args:
        displacement: Displacement pattern to decompose (n_atoms, 3)
        phonon_modes: PhononModes object containing phonon data
        supercell_matrix: 3x3 supercell transformation matrix
        normalize: Whether to normalize displacement before decomposition
        tolerance: Numerical tolerance for completeness verification
        sort_by_contribution: Whether to sort results by contribution magnitude (default: True)

    Returns:
        Tuple of (projection_table, summary):
        - projection_table: List of dicts with projection data for each mode
        - summary: Dict with completeness verification and statistics

    Example:
        >>> table, summary = decompose_displacement_to_modes(
        ...     displacement, phonon_modes, np.eye(3)*2
        ... )
        >>> print(f"Completeness: {summary['completeness']:.6f}")
        >>> print(f"Sum of squared projections: {summary['sum_squared_projections']:.6f}")
    """
    from ..modes import PhononModes

    if not isinstance(phonon_modes, PhononModes):
        raise TypeError("phonon_modes must be a PhononModes object")

    if displacement.ndim != 2 or displacement.shape[1] != 3:
        raise ValueError("displacement must have shape (n_atoms, 3)")

    if supercell_matrix.shape != (3, 3):
        raise ValueError("supercell_matrix must be 3x3")

    # Generate target supercell for displacement
    from phonproj.modes import create_supercell

    target_supercell = create_supercell(phonon_modes.primitive_cell, supercell_matrix)

    if displacement.shape[0] != len(target_supercell):
        raise ValueError(
            f"displacement atom count {displacement.shape[0]} does not match "
            f"supercell atom count {len(target_supercell)}"
        )

    # Normalize displacement if requested
    if normalize:
        target_masses = np.repeat(target_supercell.get_masses(), 3)
        target_flat = displacement.ravel()
        mass_weighted_norm = np.sqrt(
            np.sum(target_masses * target_flat.conj() * target_flat)
        )
        if mass_weighted_norm > 0:
            displacement = displacement / mass_weighted_norm

    # Get all commensurate displacement data using PhononModes API
    # Prefer explicit grid generation and matching to avoid silently skipping
    # expected q-points. Use the detailed mode to get diagnostics about missing q-points.
    result = phonon_modes.get_commensurate_qpoints(supercell_matrix, detailed=True)

    # Backwards-compatible handling: some implementations may return a list (legacy),
    # while detailed=True should return a dict. Coerce both cases to named variables.
    if isinstance(result, list):
        matched_indices = result
        missing_qpoints = []
        all_qpoints = []
    else:
        matched_indices = result.get("matched_indices", [])
        missing_qpoints = result.get("missing_qpoints", [])
        all_qpoints = result.get("all_qpoints", [])

    if len(matched_indices) == 0:
        raise ValueError(
            f"No commensurate q-points found for the given supercell.\n"
            f"Expected q-vectors: {all_qpoints}\n"
            f"Available q-points: {phonon_modes.qpoints.tolist()}"
        )

    if len(missing_qpoints) > 0:
        # Be explicit: raising by default to avoid silent partial decompositions
        raise ValueError(
            f"Missing commensurate q-points for the given supercell:\n"
            f"Missing q-vectors (reciprocal coordinates): {missing_qpoints}\n"
            f"All expected q-vectors: {all_qpoints}\n"
            f"Present q-points: {phonon_modes.qpoints.tolist()}\n"
            f"Suggested actions: compute phonon modes for the missing q-points or use a supercell"
            f" that matches the available q-mesh."
        )

    # Use matched_indices as the list of commensurate q-points to process
    commensurate_qpoints = list(matched_indices)

    projection_table = []
    total_squared_projections = 0.0

    # Generate all commensurate displacements at once (properly mass-weighted orthonormal)
    all_commensurate_displacements = (
        phonon_modes.generate_all_commensurate_displacements(
            supercell_matrix, amplitude=1.0
        )
    )

    # Get supercell masses for projection
    source_supercell = create_supercell(phonon_modes.primitive_cell, supercell_matrix)
    det = int(np.round(np.linalg.det(supercell_matrix)))
    supercell_masses = np.tile(phonon_modes.atomic_masses, det)

    # Process each commensurate q-point
    for q_index in commensurate_qpoints:
        q_point = phonon_modes.qpoints[q_index]
        q_frequencies = phonon_modes.frequencies[q_index]

        # Get precomputed mode displacements for this q-point
        if q_index not in all_commensurate_displacements:
            continue

        mode_displacements = all_commensurate_displacements[
            q_index
        ]  # shape: (n_modes, n_atoms, 3)

        # Process each mode at this q-point
        n_modes = phonon_modes.frequencies.shape[1]
        for mode_index in range(n_modes):
            frequency = q_frequencies[mode_index]
            mode_displacement = mode_displacements[mode_index]  # shape: (n_atoms, 3)

            # Mode displacement is already mass-weighted orthonormal from generate_all_commensurate_displacements
            # Project this mode displacement onto target displacement using mass-weighted inner product
            projection_coeff = project_displacements_between_supercells(
                source_displacement=mode_displacement,
                target_displacement=displacement,
                source_supercell=source_supercell,
                target_supercell=target_supercell,
                normalize=False,  # Both are already normalized
                use_mass_weighting=True,  # Use mass-weighted inner product
            )

            squared_coeff = projection_coeff**2
            total_squared_projections += squared_coeff

            # Store projection data
            projection_data = {
                "q_index": q_index,
                "q_point": q_point.copy(),
                "mode_index": mode_index,
                "frequency": frequency,
                "projection_coefficient": projection_coeff,
                "squared_coefficient": squared_coeff,
            }
            projection_table.append(projection_data)

    # Sort table by contribution magnitude (largest first) if requested
    if sort_by_contribution:
        projection_table.sort(key=lambda x: abs(x["squared_coefficient"]), reverse=True)

    # Completeness verification
    # Calculate the expected sum based on whether displacement was normalized
    if normalize:
        # If normalized, expect sum of squared projections to equal 1.0
        expected_sum = 1.0
    else:
        # If not normalized, expect sum to equal mass-weighted norm squared of displacement
        target_masses = target_supercell.get_masses()
        expected_sum = np.sum(target_masses[:, np.newaxis] * np.abs(displacement) ** 2)

    completeness_error = abs(total_squared_projections - expected_sum)
    # For non-normalized case, use relative tolerance
    relative_tolerance = tolerance if normalize else tolerance * expected_sum
    is_complete = completeness_error < relative_tolerance

    # Summary statistics
    summary = {
        "sum_squared_projections": total_squared_projections,
        "expected_sum": expected_sum,
        "completeness_error": completeness_error,
        "is_complete": is_complete,
        "tolerance": tolerance,
        "n_modes_total": len(projection_table),
        "n_qpoints": len(commensurate_qpoints),
        "largest_contribution": projection_table[0]["squared_coefficient"]
        if projection_table
        else 0.0,
        "smallest_contribution": projection_table[-1]["squared_coefficient"]
        if projection_table
        else 0.0,
    }

    return projection_table, summary


def print_decomposition_table(
    projection_table: List[dict],
    summary: dict,
    max_entries: Optional[int] = 20,
    min_contribution: float = 1e-6,
) -> None:
    """
    Print a nicely formatted table of projection coefficients.

    Args:
        projection_table: List of projection data dicts from decompose_displacement_to_modes
        summary: Summary dict from decompose_displacement_to_modes
        max_entries: Maximum number of entries to display
        min_contribution: Minimum squared coefficient to display
    """
    print("=" * 80)
    print("DISPLACEMENT MODE DECOMPOSITION")
    print("=" * 80)

    # Summary information
    print(f"Total modes analyzed: {summary['n_modes_total']}")
    print(f"Q-points involved: {summary['n_qpoints']}")
    print(f"Sum of squared projections: {summary['sum_squared_projections']:.8f}")
    if "expected_sum" in summary:
        print(f"Expected sum (||displacement||²_M): {summary['expected_sum']:.8f}")
        completeness_ratio = (
            summary["sum_squared_projections"] / summary["expected_sum"]
            if summary["expected_sum"] > 0
            else 0.0
        )
        print(
            f"Completeness ratio: {completeness_ratio:.6f} ({completeness_ratio * 100:.2f}%)"
        )
    print(f"Completeness error: {summary['completeness_error']:.2e}")
    print(f"Completeness test: {'PASS' if summary['is_complete'] else 'FAIL'}")
    print()

    # Filter and limit entries
    filtered_table = [
        entry
        for entry in projection_table
        if abs(entry["squared_coefficient"]) >= min_contribution
    ]

    # Interpret None as "no limit" (show all)
    if max_entries is None:
        display_table = filtered_table
    else:
        if not isinstance(max_entries, int) or max_entries < 0:
            raise ValueError("max_entries must be a non-negative int or None")
        display_table = filtered_table[:max_entries]

    if len(display_table) == 0:
        print("No significant contributions found.")
        return

    # Table header
    print(
        f"{'Q-idx':<6} {'Mode':<6} {'Freq(cm⁻¹)':<12} {'Proj.Coeff':<12} {'Squared':<12} {'Q-point':<25}"
    )
    print("-" * 80)

    # Table rows
    for entry in display_table:
        q_str = f"[{entry['q_point'][0]:.3f}, {entry['q_point'][1]:.3f}, {entry['q_point'][2]:.3f}]"
        # Convert frequency from THz to cm⁻¹ (1 THz = 33.356 cm⁻¹)
        freq_cm = entry["frequency"] * 33.356
        print(
            f"{entry['q_index']:<6} "
            f"{entry['mode_index']:<6} "
            f"{freq_cm:<12.2f} "
            f"{entry['projection_coefficient']:<12.6f} "
            f"{entry['squared_coefficient']:<12.8f} "
            f"{q_str:<25}"
        )

    # Footer information
    if max_entries is not None and len(filtered_table) > max_entries:
        print(f"\n... and {len(filtered_table) - max_entries} more entries ...")

    if len(projection_table) > len(filtered_table):
        print(
            f"\n({len(projection_table) - len(filtered_table)} entries below minimum contribution threshold)"
        )

    print("=" * 80)


def print_qpoint_summary_table(
    projection_table: List[dict],
    summary: dict,
) -> None:
    """
    Print a summary table of contributions grouped by q-point.

    Args:
        projection_table: List of projection data dicts from decompose_displacement_to_modes
        summary: Summary dict from decompose_displacement_to_modes
    """
    from collections import defaultdict

    print("\n" + "=" * 90)
    print("Q-POINT CONTRIBUTION SUMMARY")
    print("=" * 90)

    # Group contributions by q-point
    qpoint_contributions = {}

    for entry in projection_table:
        q_idx = entry["q_index"]
        squared_coeff = entry["squared_coefficient"]

        if q_idx not in qpoint_contributions:
            qpoint_contributions[q_idx] = {
                "squared_sum": 0.0,
                "n_modes": 0,
                "q_point": entry["q_point"],
            }

        qpoint_contributions[q_idx]["squared_sum"] += squared_coeff
        qpoint_contributions[q_idx]["n_modes"] += 1  # type: ignore

    # Convert to sorted list (by contribution, largest first)
    qpoint_list = []
    for q_idx, data in qpoint_contributions.items():
        qpoint_list.append(
            {
                "q_index": q_idx,
                "q_point": data["q_point"],
                "squared_sum": data["squared_sum"],
                "n_modes": data["n_modes"],
            }
        )

    qpoint_list.sort(key=lambda x: x["squared_sum"], reverse=True)

    # Calculate percentage contribution for each q-point
    total_sum = summary["sum_squared_projections"]

    # Print table header
    print(
        f"{'Q-idx':<8} {'Q-point':<25} {'Total Contrib.':<16} {'% of Total':<12} {'# Modes':<10}"
    )
    print("-" * 90)

    # Print each q-point
    for entry in qpoint_list:
        q_str = f"[{entry['q_point'][0]:.3f}, {entry['q_point'][1]:.3f}, {entry['q_point'][2]:.3f}]"
        percentage = (entry["squared_sum"] / total_sum * 100) if total_sum > 0 else 0.0

        print(
            f"{entry['q_index']:<8} "
            f"{q_str:<25} "
            f"{entry['squared_sum']:<16.8f} "
            f"{percentage:<12.2f} "
            f"{entry['n_modes']:<10}"
        )

    print("-" * 90)
    print(f"Total q-points: {len(qpoint_list)}")
    print(f"Total modes: {sum(e['n_modes'] for e in qpoint_list)}")
    print(f"Sum of all contributions: {total_sum:.8f}")
    print("=" * 90)
