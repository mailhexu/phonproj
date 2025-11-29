import copy
import os
from datetime import datetime
from typing import List, Optional, Tuple, Union

import numpy as np
from ase import Atoms
from ase.cell import Cell
from ase.geometry import get_distances


def calculate_center_of_mass(structure: Atoms) -> np.ndarray:
    """
    Calculate the center of mass of a structure.

    Args:
        structure: ASE Atoms object

    Returns:
        Center of mass position as (3,) array in Cartesian coordinates
    """
    masses = structure.get_masses()
    positions = structure.get_positions()
    total_mass = np.sum(masses)

    if total_mass == 0:
        raise ValueError("Total mass is zero")

    com = np.sum(masses[:, np.newaxis] * positions, axis=0) / total_mass
    return com


def calculate_pbc_distance(
    pos1: np.ndarray, pos2: np.ndarray, cell: np.ndarray
) -> float:
    """
    Calculate minimal distance between two positions considering periodic boundary conditions.

    Uses ASE's get_distances with periodic boundary conditions to find the shortest distance
    between two points in a periodic cell.

    Args:
        pos1: First position (3,) in Cartesian coordinates
        pos2: Second position (3,) in Cartesian coordinates
        cell: Unit cell vectors (3, 3) in Cartesian coordinates

    Returns:
        Minimal distance considering periodic boundary conditions
    """
    # Reshape for ASE compatibility
    pos1_reshaped = pos1.reshape(1, 3)
    pos2_reshaped = pos2.reshape(1, 3)

    # Use ASE's get_distances with pbc=True for minimum image convention
    # Returns tuple (vectors, distances)
    vectors, distances = get_distances(
        pos1_reshaped, pos2_reshaped, cell=cell, pbc=True
    )

    return float(distances[0, 0])  # Return the single distance value as float


def find_closest_to_origin(structure: Atoms) -> Tuple[int, float, np.ndarray]:
    """
    Find the atom closest to the origin (0,0,0) considering periodic boundary conditions.

    Args:
        structure: ASE Atoms object

    Returns:
        Tuple of (atom_index, distance, position) where:
        - atom_index: Index of the atom closest to origin
        - distance: Minimal distance to origin considering PBC
        - position: The position of the closest atom (3,) in Cartesian coordinates
    """
    positions = structure.get_positions()
    cell = structure.get_cell().array

    min_distance = float("inf")
    closest_atom = 0
    closest_pos = positions[0]

    for i, pos in enumerate(positions):
        # Calculate distance to origin with PBC
        distance = calculate_pbc_distance(np.zeros(3), pos, cell)
        if distance < min_distance:
            min_distance = distance
            closest_atom = i
            closest_pos = pos

    return closest_atom, min_distance, closest_pos


def find_common_reference_atom(
    structure1: Atoms, structure2: Atoms, preferred_species: Optional[list] = None
) -> Tuple[int, int, str]:
    """
    Find a suitable reference atom present in both structures for alignment.

    Prioritizes heavy atoms (cations) over light atoms (anions) for stable alignment.
    Ensures the same species is used in both structures.

    Args:
        structure1: First structure
        structure2: Second structure
        preferred_species: Optional list of species to prefer (e.g., ['Tm', 'Pb', 'Ti', 'Fe'])
                          If None, will auto-detect heavy atoms

    Returns:
        Tuple of (index1, index2, species) where:
        - index1: Index of reference atom in structure1 (closest to origin)
        - index2: Index of reference atom in structure2 (closest to origin)
        - species: Chemical symbol of the reference species
    """
    symbols1 = structure1.get_chemical_symbols()
    symbols2 = structure2.get_chemical_symbols()

    # Find common species
    common_species = set(symbols1) & set(symbols2)

    if not common_species:
        raise ValueError("No common species found between structures")

    # Define heavy atom priority if not provided
    if preferred_species is None:
        # Prioritize by typical cation importance in perovskites and related materials
        preferred_species = [
            "Pb",
            "Sr",
            "Ba",
            "Ca",  # A-site cations
            "Ti",
            "Zr",
            "Hf",
            "Sn",  # B-site cations
            "Tm",
            "Er",
            "Y",
            "La",
            "Ce",  # Rare earths
            "Fe",
            "Mn",
            "Co",
            "Ni",
            "Cr",  # Transition metals
            "Mg",
            "Zn",
            "Al",
            "Ga",  # Other metals
        ]

    # Find the first preferred species that exists in both structures
    chosen_species = None
    for species in preferred_species:
        if species in common_species:
            chosen_species = species
            break

    # If no preferred species found, use the heaviest common species
    if chosen_species is None:
        # Get atomic masses for common species
        from ase.data import atomic_masses, atomic_numbers

        species_masses = {
            sp: atomic_masses[atomic_numbers[sp]] for sp in common_species
        }
        # Choose heaviest species
        max_mass = 0.0
        chosen_species = list(common_species)[0]
        for sp, mass in species_masses.items():
            if mass > max_mass:
                max_mass = mass
                chosen_species = sp

    # Find the atom of chosen species closest to origin in each structure
    positions1 = structure1.get_positions()
    positions2 = structure2.get_positions()
    cell1 = structure1.get_cell().array
    cell2 = structure2.get_cell().array

    # Find in structure1
    min_dist1 = float("inf")
    index1 = -1
    for i, (pos, sym) in enumerate(zip(positions1, symbols1)):
        if sym == chosen_species:
            dist = calculate_pbc_distance(np.zeros(3), pos, cell1)
            if dist < min_dist1:
                min_dist1 = dist
                index1 = i

    # Find in structure2
    min_dist2 = float("inf")
    index2 = -1
    for i, (pos, sym) in enumerate(zip(positions2, symbols2)):
        if sym == chosen_species:
            dist = calculate_pbc_distance(np.zeros(3), pos, cell2)
            if dist < min_dist2:
                min_dist2 = dist
                index2 = i

    if index1 == -1 or index2 == -1:
        raise ValueError(f"Could not find {chosen_species} atoms in both structures")

    return index1, index2, chosen_species


def shift_to_origin(structure: Atoms, reference_atom_index: int) -> Atoms:
    """
    Shift structure so that the specified atom is at the origin.

    Creates a copy of the structure and applies translation to place the
    reference atom at (0,0,0) while maintaining all relative positions.

    Args:
        structure: ASE Atoms object to shift
        reference_atom_index: Index of atom to place at origin

    Returns:
        New ASE Atoms object with the reference atom shifted to origin
    """
    shifted_structure = structure.copy()
    positions = shifted_structure.get_positions()

    # Calculate shift vector needed to move reference atom to origin
    shift_vector = -positions[reference_atom_index]

    # Apply shift to all atoms
    shifted_positions = positions + shift_vector
    shifted_structure.set_positions(shifted_positions)

    return shifted_structure


def force_near_0(structure: Atoms, threshold: float = 0.001) -> Atoms:
    """
    Force scaled positions to be closer to 0 by shifting atoms near boundaries.

    For each atom, if its scaled position is close to 1 or -1, shift it by one period
    so that it is closer to 0. This helps with mapping by ensuring atoms are
    consistently positioned within unit cell.

    Note: ASE automatically wraps scaled positions to [0,1), so this function
    works with Cartesian positions to maintain the desired shifts.

    Args:
        structure: ASE Atoms object to modify
        threshold: Distance threshold from ±1 to trigger shift (default: 0.001)

    Returns:
        New ASE Atoms object with positions adjusted to be closer to origin
    """
    shifted_structure = structure.copy()

    # Get scaled positions and cell
    scaled_positions = shifted_structure.get_scaled_positions()
    cell = shifted_structure.get_cell()

    # Count how many atoms are shifted for reporting
    shifts_applied = 0

    # Shift atoms that are close to boundaries
    for i in range(len(scaled_positions)):
        for dim in range(3):
            pos = scaled_positions[i, dim]

            # Shift if position is close to 1.0 (from either side)
            if abs(pos - 1.0) < threshold:
                scaled_positions[i, dim] -= 1.0
                shifts_applied += 1
            # Shift if position is close to -1.0 (from either side)
            elif abs(pos + 1.0) < threshold:
                scaled_positions[i, dim] += 1.0
                shifts_applied += 1

    # Convert shifted scaled positions back to Cartesian coordinates
    # This preserves the shifts since ASE wraps scaled positions automatically
    shifted_cartesian_positions = scaled_positions @ cell
    shifted_structure.set_positions(shifted_cartesian_positions)

    # Print summary if shifts were applied
    if shifts_applied > 0:
        print(
            f"Force near 0: Applied {shifts_applied} position shifts to bring atoms closer to origin"
        )

    return shifted_structure


def align_structures_by_com(
    reference: Atoms,
    displaced: Atoms,
    mapping: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the center of mass shift between reference and displaced structures
    after applying atom mapping, and return the COM shift vector.

    This function helps remove collective acoustic motion (translations) by
    calculating how much the center of mass has shifted between structures.

    Args:
        reference: Reference structure
        displaced: Displaced structure
        mapping: Optional atom mapping from reference to displaced.
                If None, assumes structures are in same order.

    Returns:
        Tuple of:
        - com_shift: Center of mass shift vector (3,) in Cartesian coordinates
        - reference_com: Reference structure center of mass (3,)
        - displaced_com: Displaced structure center of mass (3,) after mapping
    """
    # Get positions and masses from reference
    ref_positions = reference.get_positions()
    ref_masses = reference.get_masses()

    # Get displaced positions, reordered according to mapping if provided
    if mapping is not None:
        disp_positions = displaced.get_positions()[mapping]
    else:
        disp_positions = displaced.get_positions()

    # Calculate center of mass for both structures
    total_mass = np.sum(ref_masses)
    if total_mass == 0:
        raise ValueError("Total mass is zero")

    reference_com = (
        np.sum(ref_masses[:, np.newaxis] * ref_positions, axis=0) / total_mass
    )
    displaced_com = (
        np.sum(ref_masses[:, np.newaxis] * disp_positions, axis=0) / total_mass
    )

    # Calculate COM shift
    com_shift = displaced_com - reference_com

    return com_shift, reference_com, displaced_com


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
    for _i, (_idx, p1, p2) in enumerate(zip(min_indices, pos1, pos2)):
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


def create_enhanced_atom_mapping(
    structure1: Atoms,
    structure2: Atoms,
    method: str = "distance",
    max_cost: float = 1e-3,
    symprec: float = 1e-5,
    warn_threshold: float = 0.5,
    species_map: Optional[dict] = None,
    optimize_shift: bool = True,
    origin_alignment: bool = True,
    force_near_origin: bool = True,
) -> Tuple[np.ndarray, float, np.ndarray, dict]:
    """
    Create enhanced atom mapping between two structures with shift optimization.

    This extends the basic atom mapping to include optimal shift vector calculation
    and detailed quality metrics. It can align structures to a common origin and
    find the optimal translation that minimizes mapped distances.

    Args:
        structure1: First structure (source)
        structure2: Second structure (target)
        method: Method for creating mapping (only 'distance' is implemented)
        max_cost: Maximum allowed total cost (triggers warning if exceeded)
        symprec: Symmetry precision (not used currently)
        warn_threshold: Distance threshold (Angstrom) for individual atom warnings
        species_map: Optional dict mapping species in structure1 to allowed species in structure2
        optimize_shift: Whether to optimize shift vector to minimize total distance
        origin_alignment: Whether to align structures to common origin before mapping
        force_near_origin: Whether to force atoms near origin before mapping

    Returns:
        Tuple of (mapping, total_cost, shift_vector, quality_metrics) where:
        - mapping: Array where mapping[i] gives index in structure2 for atom i in structure1
        - total_cost: Total mapped distance after shift optimization
        - shift_vector: Optimal shift vector (3,) applied to structure2
        - quality_metrics: Dict with detailed quality information
    """

    # Create working copies
    work_struct1 = structure1.copy()
    work_struct2 = structure2.copy()

    # Force atoms to be closer to origin before mapping
    if force_near_origin:
        work_struct1 = force_near_0(work_struct1)
        work_struct2 = force_near_0(work_struct2)

    # Initialize shift vector
    shift_vector = np.zeros(3)

    # Origin alignment if requested
    if origin_alignment:
        # Find common reference atom (same species, preferring heavy atoms)
        closest1, closest2, ref_species = find_common_reference_atom(
            work_struct1, work_struct2
        )

        # Get positions for printing
        pos1 = work_struct1.get_positions()[closest1]
        pos2 = work_struct2.get_positions()[closest2]
        dist1 = calculate_pbc_distance(np.zeros(3), pos1, work_struct1.get_cell().array)
        dist2 = calculate_pbc_distance(np.zeros(3), pos2, work_struct2.get_cell().array)

        # Print origin alignment information
        print("Origin Alignment:")
        print(
            f"  Reference structure: Atom {closest1} ({ref_species}) at position [{pos1[0]:.6f}, {pos1[1]:.6f}, {pos1[2]:.6f}] Å, distance {dist1:.6f} Å from origin"
        )
        print(
            f"  Target structure:    Atom {closest2} ({ref_species}) at position [{pos2[0]:.6f}, {pos2[1]:.6f}, {pos2[2]:.6f}] Å, distance {dist2:.6f} Å from origin"
        )

        # Calculate and print shift vectors
        shift1 = -pos1
        shift2 = -pos2
        print(
            f"  Applied shift to reference: [{shift1[0]:.6f}, {shift1[1]:.6f}, {shift1[2]:.6f}] Å"
        )
        print(
            f"  Applied shift to target:    [{shift2[0]:.6f}, {shift2[1]:.6f}, {shift2[2]:.6f}] Å"
        )

        # Shift both structures so their reference atoms are at origin
        work_struct1 = shift_to_origin(work_struct1, closest1)
        work_struct2 = shift_to_origin(work_struct2, closest2)

    # Initial mapping without shift optimization
    mapping, base_cost = create_atom_mapping(
        work_struct1,
        work_struct2,
        method,
        max_cost,
        symprec,
        warn_threshold,
        species_map,
    )

    # Shift optimization if requested
    if optimize_shift:
        # Get positions after mapping
        pos1 = work_struct1.get_positions()
        pos2_mapped = work_struct2.get_positions()[mapping]

        # Calculate optimal shift using least squares
        # We want to find shift that minimizes sum(|pos2_mapped + shift - pos1|^2)
        # This is equivalent to shift = mean(pos1 - pos2_mapped)
        raw_shift = np.mean(pos1 - pos2_mapped, axis=0)

        # Apply shift and recalculate mapping
        shifted_pos2 = work_struct2.get_positions() + raw_shift
        work_struct2.set_positions(shifted_pos2)

        # Recalculate mapping with shifted structure
        mapping, total_cost = create_atom_mapping(
            work_struct1,
            work_struct2,
            method,
            max_cost,
            symprec,
            warn_threshold,
            species_map,
        )

        shift_vector = raw_shift
    else:
        total_cost = base_cost

    # Calculate quality metrics
    quality_metrics = _calculate_mapping_quality(
        work_struct1, work_struct2, mapping, shift_vector
    )

    return mapping, total_cost, shift_vector, quality_metrics


def _calculate_mapping_quality(
    structure1: Atoms, structure2: Atoms, mapping: np.ndarray, shift_vector: np.ndarray
) -> dict:
    """
    Calculate detailed quality metrics for atom mapping.

    Args:
        structure1: First structure (should be aligned)
        structure2: Second structure (should include shift)
        mapping: Atom mapping from structure1 to structure2
        shift_vector: Shift vector applied to structure2

    Returns:
        Dictionary with quality metrics
    """
    pos1 = structure1.get_positions()
    pos2 = structure2.get_positions()[mapping]
    cell = structure2.get_cell().array

    # Calculate individual mapped distances
    mapped_distances = []
    for i in range(len(pos1)):
        distance = calculate_pbc_distance(pos1[i], pos2[i], cell)
        mapped_distances.append(distance)

    mapped_distances = np.array(mapped_distances)

    # Calculate statistics
    quality_metrics = {
        "mean_distance": np.mean(mapped_distances),
        "max_distance": np.max(mapped_distances),
        "min_distance": np.min(mapped_distances),
        "std_distance": np.std(mapped_distances),
        "atoms_above_threshold": np.sum(mapped_distances > 0.1),
        "atoms_above_01angstrom": int(np.sum(mapped_distances > 0.1)),
        "atoms_above_05angstrom": int(np.sum(mapped_distances > 0.5)),
        "shift_magnitude": np.linalg.norm(shift_vector),
        "mapped_distances": mapped_distances.tolist(),
    }

    return quality_metrics


class MappingAnalyzer:
    """
    Class for analyzing and generating detailed output for atom mapping operations.

    Provides comprehensive analysis of mapping results including detailed tables,
    quality metrics, and shift vector information. Outputs can be saved to text files
    for further analysis.
    """

    def __init__(
        self,
        structure1: Atoms,
        structure2: Atoms,
        mapping: np.ndarray,
        shift_vector: np.ndarray,
        quality_metrics: dict,
    ):
        """
        Initialize MappingAnalyzer with mapping results.

        Args:
            structure1: First structure (source)
            structure2: Second structure (target)
            mapping: Atom mapping from structure1 to structure2
            shift_vector: Shift vector applied to structure2
            quality_metrics: Quality metrics from mapping analysis
        """
        self.structure1 = structure1
        self.structure2 = structure2
        self.mapping = mapping
        self.shift_vector = shift_vector
        self.quality_metrics = quality_metrics
        self.species1 = structure1.get_chemical_symbols()
        self.species2 = structure2.get_chemical_symbols()

    def analyze_mapping(self) -> dict:
        """
        Perform comprehensive analysis of the mapping results.

        Returns:
            Dictionary containing detailed analysis results
        """
        pos1 = self.structure1.get_positions()
        pos2_mapped = self.structure2.get_positions()[self.mapping]
        cell = self.structure2.get_cell().array

        # Calculate detailed mapping information
        mapping_details = []
        for i in range(len(pos1)):
            distance = calculate_pbc_distance(pos1[i], pos2_mapped[i], cell)
            mapping_details.append(
                {
                    "atom_index": i,
                    "target_index": int(self.mapping[i]),
                    "species_source": self.species1[i],
                    "species_target": self.species2[self.mapping[i]],
                    "source_position": pos1[i],
                    "target_position": pos2_mapped[i],
                    "distance": distance,
                    "shift_contribution": self.shift_vector.copy(),
                }
            )

        analysis = {
            "mapping_details": mapping_details,
            "quality_metrics": self.quality_metrics,
            "shift_vector": self.shift_vector,
            "total_atoms": len(mapping_details),
            "species_conservation": self._check_species_conservation(),
            "mapping_summary": self._generate_mapping_summary(mapping_details),
        }

        return analysis

    def _check_species_conservation(self) -> dict:
        """
        Check if species are conserved in the mapping.

        Returns:
            Dictionary with species conservation analysis
        """
        species_counts1 = {}
        species_counts2 = {}

        for sp in self.species1:
            species_counts1[sp] = species_counts1.get(sp, 0) + 1

        for sp in self.species2:
            species_counts2[sp] = species_counts2.get(sp, 0) + 1

        # Check mapped species
        mapped_species1 = {}
        mapped_species2 = {}

        for i, target_idx in enumerate(self.mapping):
            sp1 = self.species1[i]
            sp2 = self.species2[target_idx]

            mapped_species1[sp1] = mapped_species1.get(sp1, 0) + 1
            mapped_species2[sp2] = mapped_species2.get(sp2, 0) + 1

        return {
            "source_species": species_counts1,
            "target_species": species_counts2,
            "mapped_source_species": mapped_species1,
            "mapped_target_species": mapped_species2,
            "species_conserved": species_counts1 == species_counts2,
        }

    def _generate_mapping_summary(self, mapping_details: list) -> dict:
        """
        Generate summary statistics for the mapping.

        Args:
            mapping_details: List of detailed mapping information

        Returns:
            Dictionary with summary statistics
        """
        distances = [detail["distance"] for detail in mapping_details]

        return {
            "total_atoms": len(mapping_details),
            "perfect_mappings": sum(1 for d in distances if d < 1e-6),
            "excellent_mappings": sum(1 for d in distances if d < 0.01),
            "good_mappings": sum(1 for d in distances if d < 0.1),
            "poor_mappings": sum(1 for d in distances if d >= 0.1),
            "mean_distance": np.mean(distances),
            "max_distance": np.max(distances),
            "std_distance": np.std(distances),
        }

    def save_detailed_output(self, filepath: str) -> None:
        """
        Save detailed mapping analysis to a text file.

        Args:
            filepath: Path to save the output file
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        analysis = self.analyze_mapping()

        with open(filepath, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("ENHANCED ATOM MAPPING ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")

            # Timestamp
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Summary statistics
            f.write("MAPPING SUMMARY\n")
            f.write("-" * 40 + "\n")
            summary = analysis["mapping_summary"]
            f.write(f"Total atoms mapped: {summary['total_atoms']}\n")
            f.write(f"Perfect mappings (< 1μm): {summary['perfect_mappings']}\n")
            f.write(f"Excellent mappings (< 0.01Å): {summary['excellent_mappings']}\n")
            f.write(f"Good mappings (< 0.1Å): {summary['good_mappings']}\n")
            f.write(f"Poor mappings (≥ 0.1Å): {summary['poor_mappings']}\n")
            f.write(f"Mean distance: {summary['mean_distance']:.6f} Å\n")
            f.write(f"Maximum distance: {summary['max_distance']:.6f} Å\n")
            f.write(f"Std deviation: {summary['std_distance']:.6f} Å\n\n")

            # Shift vector information
            f.write("SHIFT VECTOR INFORMATION\n")
            f.write("-" * 40 + "\n")
            shift = analysis["shift_vector"]
            f.write(
                f"Shift vector: [{shift[0]:.6f}, {shift[1]:.6f}, {shift[2]:.6f}] Å\n"
            )
            f.write(f"Shift magnitude: {np.linalg.norm(shift):.6f} Å\n\n")

            # Quality metrics
            f.write("QUALITY METRICS\n")
            f.write("-" * 40 + "\n")
            quality = analysis["quality_metrics"]
            f.write(
                f"Atoms above 0.1Å threshold: {quality['atoms_above_01angstrom']}\n"
            )
            f.write(
                f"Atoms above 0.5Å threshold: {quality['atoms_above_05angstrom']}\n"
            )
            f.write(
                f"Species conserved: {analysis['species_conservation']['species_conserved']}\n\n"
            )

            # Detailed mapping table
            f.write("DETAILED MAPPING TABLE\n")
            f.write("-" * 40 + "\n")
            f.write(
                f"{'Idx':>4} {'Tgt':>4} {'Sp1':>3} {'Sp2':>3} {'Distance(Å)':>12} {'Shift_X':>10} {'Shift_Y':>10} {'Shift_Z':>10}\n"
            )
            f.write("-" * 80 + "\n")

            for detail in analysis["mapping_details"]:
                f.write(
                    f"{detail['atom_index']:>4} {detail['target_index']:>4} "
                    f"{detail['species_source']:>3} {detail['species_target']:>3} "
                    f"{detail['distance']:>12.6f} "
                    f"{detail['shift_contribution'][0]:>10.6f} "
                    f"{detail['shift_contribution'][1]:>10.6f} "
                    f"{detail['shift_contribution'][2]:>10.6f}\n"
                )

            f.write("\n" + "=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")


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


def project_complex_displacement(
    complex_displacement: np.ndarray,
    target_displacement: np.ndarray,
    masses: np.ndarray,
) -> complex:
    """
    Project a complex displacement onto a real target displacement.

    Both displacements must be in the same atom ordering (already mapped).

    For complex source displacement d = d_r + i*d_i and real target t:
        projection = <d, t> = sum(m_i * conj(d_i) * t_i)
                   = <d_r, t> + i*<d_i, t>

    The magnitude |projection| gives the maximum achievable projection across all phases.
    The argument arg(projection) gives the optimal phase.

    Args:
        complex_displacement: Complex displacement pattern (n_atoms, 3), already mapped
        target_displacement: Real target displacement pattern (n_atoms, 3), already mapped
        masses: Atomic masses (n_atoms,)

    Returns:
        complex: Complex projection coefficient
    """
    # Check dimensions
    if complex_displacement.shape != target_displacement.shape:
        raise ValueError(
            f"Displacement shapes must match: {complex_displacement.shape} vs {target_displacement.shape}"
        )
    if len(masses) != complex_displacement.shape[0]:
        raise ValueError(
            f"Number of masses {len(masses)} must match number of atoms {complex_displacement.shape[0]}"
        )

    # Flatten for inner product
    complex_flat = complex_displacement.ravel()
    target_flat = target_displacement.ravel()
    masses_repeated = np.repeat(masses, 3)  # Repeat for x, y, z components

    # Calculate mass-weighted complex inner product: <d, t> = sum(m_i * conj(d_i) * t_i)
    projection = np.sum(masses_repeated * complex_flat.conj() * target_flat)

    return projection


def project_complex_displacement_with_phase_scan(
    complex_displacement: np.ndarray,
    target_displacement: np.ndarray,
    masses: np.ndarray,
    n_phases: int = 36,
) -> Tuple[float, float]:
    """
    Project a complex displacement onto real target with phase scan.

    Both displacements must be in the same atom ordering (already mapped).

    For each phase θ in [0, π], computes:
        displacement(θ) = Re[exp(iθ) * complex_displacement]
        projection(θ) = <displacement(θ), target>

    Returns the maximum |projection| and optimal phase.

    This is mathematically equivalent to computing the complex projection directly:
        max|projection(θ)| = |<complex_displacement, target>|
        optimal_phase = arg(<complex_displacement, target>)

    Args:
        complex_displacement: Complex displacement pattern (n_atoms, 3), already mapped
        target_displacement: Real target displacement (n_atoms, 3), already mapped
        masses: Atomic masses (n_atoms,)
        n_phases: Number of phase points to sample in [0, π]

    Returns:
        Tuple of (max_coefficient, optimal_phase)
    """
    # Check dimensions
    if complex_displacement.shape != target_displacement.shape:
        raise ValueError(
            f"Displacement shapes must match: {complex_displacement.shape} vs {target_displacement.shape}"
        )
    if len(masses) != complex_displacement.shape[0]:
        raise ValueError(
            f"Number of masses {len(masses)} must match number of atoms {complex_displacement.shape[0]}"
        )

    phases = np.linspace(0, np.pi, n_phases, endpoint=True)
    masses_repeated = np.repeat(masses, 3)
    target_flat = target_displacement.ravel()

    max_abs_coeff = 0.0
    optimal_phase = 0.0

    for phase in phases:
        # Apply phase: Re[exp(iθ) * d_complex]
        phase_factor = np.exp(1j * phase)
        real_displacement = (complex_displacement * phase_factor).real
        real_flat = real_displacement.ravel()

        # Calculate mass-weighted projection
        projection = np.sum(masses_repeated * real_flat * target_flat)

        abs_coeff = abs(projection)
        if abs_coeff > max_abs_coeff:
            max_abs_coeff = abs_coeff
            optimal_phase = phase

    return float(max_abs_coeff), float(optimal_phase)


def project_displacement_with_phase_scan(
    phonon_modes,  # PhononModes object
    target_displacement: np.ndarray,
    supercell_matrix: np.ndarray,
    q_index: int,
    mode_index: int,
    n_phases: int = 36,
    target_supercell=None,  # Optional ASE Atoms object for target structure
    precomputed_mode_displacements=None,  # Optional pre-computed mode displacements
) -> Tuple[float, float]:
    """
    Project target onto a phonon mode with phase scan (wrapper for backward compatibility).

    NOTE: This assumes target_displacement is already in the same atom ordering as
    the phonon mode displacements (i.e., pre-mapped).

    Args:
        phonon_modes: PhononModes object containing phonon data.
        target_displacement: The displacement to project, shape (n_atoms, 3), already mapped.
        supercell_matrix: The 3x3 supercell matrix.
        q_index: The index of the q-point to use.
        mode_index: The index of the mode to use.
        n_phases: The number of phase angles to sample between 0 and π.
        target_supercell: Optional target supercell structure (for getting masses).
        precomputed_mode_displacements: Optional pre-computed mode displacements array.

    Returns:
        Tuple of (max_coefficient, optimal_phase)
    """
    from ..modes import create_supercell

    # Create supercell to get masses
    if target_supercell is None:
        target_supercell = create_supercell(
            phonon_modes.primitive_cell, supercell_matrix
        )

    # Verify dimensions
    if target_displacement.shape[0] != len(target_supercell):
        raise ValueError(
            f"Target displacement has {target_displacement.shape[0]} atoms but "
            f"target supercell has {len(target_supercell)} atoms"
        )

    # Get or generate complex mode displacement
    if precomputed_mode_displacements is not None:
        all_mode_displacements = precomputed_mode_displacements
    else:
        all_mode_displacements = phonon_modes.generate_all_mode_displacements(
            q_index=q_index,
            supercell_matrix=supercell_matrix,
            amplitude=1.0,
        )

    complex_displacement = all_mode_displacements[mode_index]  # shape: (n_atoms, 3)

    # Get masses for projection
    masses = target_supercell.get_masses()

    # Use new projection function with phase scan
    return project_complex_displacement_with_phase_scan(
        complex_displacement=complex_displacement,
        target_displacement=target_displacement,
        masses=masses,
        n_phases=n_phases,
    )


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

    # Get supercell for projection
    source_supercell = create_supercell(phonon_modes.primitive_cell, supercell_matrix)

    # Get supercell for projection
    source_supercell = create_supercell(phonon_modes.primitive_cell, supercell_matrix)

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

        # Get masses for projections
        supercell_masses = target_supercell.get_masses()

        # Process each mode at this q-point
        n_modes = phonon_modes.frequencies.shape[1]
        for mode_index in range(n_modes):
            frequency = q_frequencies[mode_index]
            mode_displacement = mode_displacements[
                mode_index
            ]  # shape: (n_atoms, 3), complex

            # Mode displacement is complex, displacement is real, both already in same ordering
            # Project complex mode displacement onto real target displacement
            projection_coeff = project_complex_displacement(
                complex_displacement=mode_displacement,
                target_displacement=displacement,
                masses=supercell_masses,
            )

            # When mode_displacement is complex and target is real, projection_coeff is complex
            # The squared coefficient is |c|^2 = c* × c
            squared_coeff = abs(projection_coeff) ** 2
            total_squared_projections += squared_coeff

            # Store projection data
            projection_data = {
                "q_index": q_index,
                "q_point": q_point.copy(),
                "mode_index": mode_index,
                "frequency": frequency,
                "projection_coefficient": abs(projection_coeff),  # Store magnitude
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
                "q_point": None,
            }

        qpoint_contributions[q_idx]["squared_sum"] += squared_coeff
        qpoint_contributions[q_idx]["n_modes"] += 1
        qpoint_contributions[q_idx]["q_point"] = entry["q_point"]

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
