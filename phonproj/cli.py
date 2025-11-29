#!/usr/bin/env python3
"""
Command-line interface for phonon mode decomposition analysis.

This tool analyzes structural displacements in terms of phonon mode contributions.
"""

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple
import os

if TYPE_CHECKING:
    from ase import Atoms

import numpy as np
from ase.io import read as ase_read

from phonproj.core import load_from_phonopy_files, load_yaml_file
from phonproj.isodistort_parser import parse_isodistort_file
from phonproj.modes import PhononModes


def parse_supercell_matrix(supercell_str: str) -> np.ndarray:
    """
    Parse supercell matrix from string.

    Args:
        supercell_str: String like "2x2x2", "16x1x1", or "16 1 1"

    Returns:
        3x3 supercell matrix

    Raises:
        ValueError: If format is invalid
    """
    try:
        # Try splitting by 'x' first, then by whitespace
        if "x" in supercell_str.lower():
            parts = supercell_str.lower().split("x")
        else:
            parts = supercell_str.split()

        if len(parts) != 3:
            raise ValueError(
                "Supercell must be in format NxMxL (e.g., 2x2x2) or 'N M L' (e.g., 16 1 1)"
            )

        n, m, k = [int(p) for p in parts]
        if n <= 0 or m <= 0 or k <= 0:
            raise ValueError("Supercell dimensions must be positive")

        return np.array([[n, 0, 0], [0, m, 0], [0, 0, k]])
    except (ValueError, IndexError) as e:
        raise ValueError(f"Invalid supercell format '{supercell_str}': {e}") from e


def load_phonopy_data(phonopy_path: str, quiet: bool = False):
    """
    Load phonopy data from file or directory.

    Args:
        phonopy_path: Path to phonopy_params.yaml file or directory containing phonopy files

    Returns:
        Dictionary with phonopy data including 'phonopy', 'primitive_cell', etc.
    """
    print(os.getcwd())
    print(phonopy_path)
    path = Path(phonopy_path)
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {phonopy_path}")

    if path.is_dir():
        if not quiet:
            print(f"Loading phonopy data from directory: {path}")
        return load_from_phonopy_files(path)
    elif path.is_file():
        if not quiet:
            print(f"Loading phonopy data from file: {path}")
        return load_yaml_file(path)
    else:
        raise ValueError(f"Invalid path type: {phonopy_path}")


def load_isodistort_structures(isodistort_path: str):
    """
    Load undistorted and distorted structures from ISODISTORT file.

    Args:
        isodistort_path: Path to ISODISTORT file

    Returns:
        Tuple of (undistorted_structure, distorted_structure) as ASE Atoms objects
    """
    try:
        isodistort_data = parse_isodistort_file(isodistort_path)
        supercell_structure = isodistort_data["supercell_structure"]

        if not isinstance(supercell_structure, dict):
            raise ValueError("Invalid supercell_structure format in ISODISTORT file")
        if "undistorted" not in supercell_structure:
            raise ValueError("Undistorted structure not found in ISODISTORT file")
        if "distorted" not in supercell_structure:
            raise ValueError("Distorted structure not found in ISODISTORT file")

        return supercell_structure["undistorted"], supercell_structure["distorted"]
    except Exception as e:
        raise ValueError(
            f"Failed to load ISODISTORT structures from {isodistort_path}: {e}"
        ) from e


def generate_commensurate_qpoints(supercell_matrix: np.ndarray) -> np.ndarray:
    """
    Generate unique commensurate q-points for a supercell using time-reversal symmetry.

    Time-reversal symmetry means k and -k (or 1-k in fractional coordinates) are equivalent.
    We only generate q-points in the range [0, 0.5] to avoid redundancy.

    Args:
        supercell_matrix: 3x3 supercell matrix (must be diagonal)

    Returns:
        Array of unique q-points in fractional coordinates
    """
    n1 = int(round(supercell_matrix[0, 0]))
    n2 = int(round(supercell_matrix[1, 1]))
    n3 = int(round(supercell_matrix[2, 2]))

    qpoints = []

    for i in range(n1):
        qi = i / n1
        # Only include q-points with qi <= 0.5
        if qi > 0.5:
            continue

        for j in range(n2):
            qj = j / n2
            if qj > 0.5:
                continue

            for k in range(n3):
                qk = k / n3
                if qk > 0.5:
                    continue

                qpoints.append([qi, qj, qk])

    return np.array(qpoints)


def load_displaced_structure(displaced_path: str):
    """
    Load displaced structure from file.

    Args:
        displaced_path: Path to displaced structure file (VASP POSCAR/CONTCAR format)

    Returns:
        ASE Atoms object
    """
    try:
        displaced = ase_read(displaced_path, format="vasp")
        # Handle case where read returns a list
        if isinstance(displaced, list):
            displaced = displaced[0]
        return displaced
    except Exception as e:
        raise ValueError(
            f"Failed to load displaced structure from {displaced_path}: {e}"
        ) from e


def calculate_displacement_vector(
    displaced_atoms,
    reference_atoms,
    normalize: bool = False,
    verbose: bool = True,
    species_map: Optional[dict] = None,
    remove_com: bool = False,
    output_structure_path: Optional[str] = None,
    input_structure: Optional["Atoms"] = None,
) -> Tuple[np.ndarray, float]:
    """
    Calculate displacement vector between two structures.

    This function handles atom reordering and periodic boundary conditions.

    If input_structure is provided, it will be used to determine the optimal
    atom mapping and PBC shifts, which are then applied to the displaced structure
    for computing displacements relative to the reference structure.

    Args:
        displaced_atoms: Displaced structure (ASE Atoms)
        reference_atoms: Reference structure (ASE Atoms)
        normalize: Whether to mass-weight normalize the displacement
        verbose: Whether to print atom mapping verification table
        species_map: Optional dict mapping species substitutions (e.g., {'Pb': 'Sr'})
        remove_com: Whether to align structures by COM and project out acoustic modes
        output_structure_path: Optional path to save the processed displaced structure
                               (after mapping and PBC shifts)
        input_structure: Optional structure used to determine atom mapping and PBC shifts.
                        If provided, mapping/shifts are computed from input→reference,
                        then applied to displaced→reference for displacement calculation.

    Returns:
        Tuple of (displacement_vector, displacement_norm)
        displacement_vector has shape (n_atoms * 3,)
    """
    from phonproj.core.structure_analysis import (
        create_enhanced_atom_mapping,
        project_out_acoustic_modes,
    )

    # Determine atom mapping and shift vectors
    if input_structure is not None:
        # Use input structure to find optimal mapping and shift vectors
        # mapping[i] gives the index in input_structure for atom i in reference_atoms
        mapping, total_cost, shift_vector, quality_metrics = (
            create_enhanced_atom_mapping(
                reference_atoms,
                input_structure,
                max_cost=100.0,
                warn_threshold=0.5,
                species_map=species_map,
                optimize_shift=True,
            )
        )

        if verbose:
            print(f"\n{'=' * 90}")
            print("ATOM MAPPING AND SHIFT VECTOR (from input structure)")
            print(f"{'=' * 90}")
            print("Using input structure to determine mapping and shift vector")
            print("Mapping and shift will be applied to displaced structure")
            print(
                f"Shift vector: [{shift_vector[0]:.6f}, {shift_vector[1]:.6f}, {shift_vector[2]:.6f}] Å"
            )
            print(f"Shift magnitude: {np.linalg.norm(shift_vector):.6f} Å")
            print(f"{'=' * 90}")

        # Apply the same mapping and shift vector to displaced structure
        # Reorder displaced atoms according to mapping found from input structure
        displaced_positions = displaced_atoms.get_positions()[mapping]
        reference_positions = reference_atoms.get_positions()

        # Apply shift vector to displaced structure in reduced coordinates
        # Convert shift vector to fractional coordinates in reference cell
        ref_cell_array = reference_atoms.get_cell().array
        shift_vector_frac = shift_vector @ np.linalg.inv(ref_cell_array)

        # Apply shift in fractional coordinates, then convert back to Cartesian
        displaced_positions_frac = displaced_positions @ np.linalg.inv(ref_cell_array)
        displaced_positions_frac_shifted = displaced_positions_frac + shift_vector_frac
        displaced_positions = displaced_positions_frac_shifted @ ref_cell_array

    else:
        # Default behavior: map displaced structure directly to reference structure
        # mapping[i] gives the index in displaced_atoms for atom i in reference_atoms
        mapping, total_cost, shift_vector, quality_metrics = (
            create_enhanced_atom_mapping(
                reference_atoms,
                displaced_atoms,
                max_cost=100.0,
                warn_threshold=0.5,
                species_map=species_map,
                optimize_shift=True,
            )
        )
        # Reorder displaced atoms according to mapping
        displaced_positions = displaced_atoms.get_positions()[mapping]
        reference_positions = reference_atoms.get_positions()

        # Apply shift vector in reduced coordinates
        # Convert shift vector to fractional coordinates in reference cell
        ref_cell_array = reference_atoms.get_cell().array
        shift_vector_frac = shift_vector @ np.linalg.inv(ref_cell_array)

        # Apply shift in fractional coordinates, then convert back to Cartesian
        displaced_positions_frac = displaced_positions @ np.linalg.inv(ref_cell_array)
        displaced_positions_frac_shifted = displaced_positions_frac + shift_vector_frac
        displaced_positions = displaced_positions_frac_shifted @ ref_cell_array

    # Get cell matrices as numpy arrays for calculations
    ref_cell_array = reference_atoms.get_cell().array
    disp_cell_array = displaced_atoms.get_cell().array

    # Convert positions to fractional coordinates (each in their own cell)
    displaced_positions_frac = displaced_positions @ np.linalg.inv(disp_cell_array)

    # Key transformation: Convert displaced structure's fractional coordinates
    # to Cartesian using the REFERENCE cell (not displaced cell)
    # This ensures both structures are in the same coordinate system
    displaced_positions_in_ref_cell = displaced_positions_frac @ ref_cell_array
    reference_positions_in_ref_cell = reference_positions  # Already in reference cell

    # Align structures by center of mass if requested
    # IMPORTANT: Do this AFTER transforming to common coordinate system
    com_shift = np.zeros(3)
    if remove_com:
        # Calculate COM in reference cell coordinate system
        ref_masses = reference_atoms.get_masses()
        total_mass = np.sum(ref_masses)

        # Reference positions are not reordered, masses stay in original order
        reference_com = (
            np.sum(ref_masses[:, np.newaxis] * reference_positions_in_ref_cell, axis=0)
            / total_mass
        )
        displaced_com = (
            np.sum(ref_masses[:, np.newaxis] * displaced_positions_in_ref_cell, axis=0)
            / total_mass
        )

        com_shift = displaced_com - reference_com

        # Shift displaced positions to align COM with reference
        displaced_positions_in_ref_cell = displaced_positions_in_ref_cell - com_shift

        if verbose:
            print(f"\n{'=' * 90}")
            print("CENTER OF MASS ALIGNMENT")
            print(f"{'=' * 90}")
            print(
                f"Reference COM:  [{reference_com[0]:12.8f}, {reference_com[1]:12.8f}, {reference_com[2]:12.8f}]"
            )
            print(
                f"Displaced COM:  [{displaced_com[0]:12.8f}, {displaced_com[1]:12.8f}, {displaced_com[2]:12.8f}]"
            )
            print(
                f"COM shift:      [{com_shift[0]:12.8f}, {com_shift[1]:12.8f}, {com_shift[2]:12.8f}]"
            )
            print(f"COM shift magnitude: {np.linalg.norm(com_shift):.8f} Å")
            print(f"{'=' * 90}")

    # Calculate displacement entirely in reference cell coordinates
    # displacement = displaced - reference (always)
    displacement = displaced_positions_in_ref_cell - reference_positions_in_ref_cell

    # Apply periodic boundary conditions by wrapping to nearest image
    # Convert displacement to fractional (in reference cell) to wrap
    displacement_frac = displacement @ np.linalg.inv(ref_cell_array)
    displacement_frac = displacement_frac - np.round(displacement_frac)
    displacement = displacement_frac @ ref_cell_array

    # Project out any remaining acoustic mode components if requested
    # Note: After COM alignment, these should be very small (numerical artifacts)
    acoustic_projections = np.zeros(3)
    if remove_com:
        # Project out all three acoustic translation modes
        # This removes any residual acoustic components after COM alignment
        displacement, acoustic_projections = project_out_acoustic_modes(
            displacement, reference_atoms
        )

        if verbose:
            print(f"\n{'=' * 90}")
            print("RESIDUAL ACOUSTIC MODE PROJECTION")
            print(f"{'=' * 90}")
            print("Projections onto acoustic translation modes (after COM alignment):")
            print(f"  X-translation: {acoustic_projections[0]:12.8f}")
            print(f"  Y-translation: {acoustic_projections[1]:12.8f}")
            print(f"  Z-translation: {acoustic_projections[2]:12.8f}")
            print(
                f"  Total residual magnitude: {np.linalg.norm(acoustic_projections):.8f}"
            )
            print("Note: These values should be close to zero after COM alignment")
            print(f"{'=' * 90}")

    # Print detailed structure information if verbose
    if verbose:
        symbols = reference_atoms.get_chemical_symbols()

        # 1. Print cell parameters
        print(f"\n{'=' * 90}")
        print("CELL PARAMETERS")
        print(f"{'=' * 90}")
        ref_cell = reference_atoms.get_cell()
        disp_cell = displaced_atoms.get_cell()

        print("Reference Cell (Å):")
        for i, row in enumerate(ref_cell.array):
            print(f"  a{i + 1} = [{row[0]:12.8f}, {row[1]:12.8f}, {row[2]:12.8f}]")

        print("\nDisplaced Cell (Å):")
        for i, row in enumerate(disp_cell.array):
            print(f"  a{i + 1} = [{row[0]:12.8f}, {row[1]:12.8f}, {row[2]:12.8f}]")
        print(f"{'=' * 90}")

        # 2. Print scaled positions - reference structure
        print(f"\n{'=' * 90}")
        print("REFERENCE STRUCTURE - SCALED POSITIONS")
        print(f"{'=' * 90}")
        print(f"{'Idx':<6} {'Species':<8} {'Scaled Position (fractional)':<40}")
        print("-" * 90)

        ref_scaled = reference_positions @ np.linalg.inv(ref_cell.array)
        for i in range(len(reference_positions)):
            print(
                f"{i:<6} {symbols[i]:<8} "
                f"[{ref_scaled[i, 0]:12.8f}, {ref_scaled[i, 1]:12.8f}, {ref_scaled[i, 2]:12.8f}]"
            )
        print(f"{'=' * 90}")

        # 3. Print scaled positions - displaced structure BEFORE mapping
        print(f"\n{'=' * 90}")
        print("DISPLACED STRUCTURE - SCALED POSITIONS (BEFORE MAPPING)")
        print(f"{'=' * 90}")
        print(f"{'Idx':<6} {'Species':<8} {'Scaled Position (fractional)':<40}")
        print("-" * 90)

        disp_symbols_original = displaced_atoms.get_chemical_symbols()
        disp_positions_original = displaced_atoms.get_positions()
        disp_scaled_original = disp_positions_original @ np.linalg.inv(disp_cell.array)
        for i in range(len(disp_positions_original)):
            print(
                f"{i:<6} {disp_symbols_original[i]:<8} "
                f"[{disp_scaled_original[i, 0]:12.8f}, {disp_scaled_original[i, 1]:12.8f}, {disp_scaled_original[i, 2]:12.8f}]"
            )
        print(f"{'=' * 90}")

        # 4. Print scaled positions - displaced structure AFTER mapping
        print(f"\n{'=' * 90}")
        print("DISPLACED STRUCTURE - SCALED POSITIONS (AFTER MAPPING)")
        print(f"{'=' * 90}")
        print(
            f"{'Idx':<6} {'Species':<8} {'Scaled Position (fractional)':<40} {'Original Idx':<12}"
        )
        print("-" * 90)

        disp_scaled_mapped = displaced_positions @ np.linalg.inv(disp_cell.array)
        for i in range(len(displaced_positions)):
            print(
                f"{i:<6} {symbols[i]:<8} "
                f"[{disp_scaled_mapped[i, 0]:12.8f}, {disp_scaled_mapped[i, 1]:12.8f}, {disp_scaled_mapped[i, 2]:12.8f}]  "
                f"({mapping[i]})"
            )
        print(f"{'=' * 90}")

        # 5. Print atom mapping verification table - FULL TABLE
        print(f"\n{'=' * 110}")
        print("ATOM MAPPING VERIFICATION - CARTESIAN COORDINATES")
        print(f"{'=' * 110}")
        print(
            f"{'Idx':<5} {'Species':<8} {'Ref Position (Å)':<30} {'Disp Position (Å)':<30} {'Displacement (Å)':<30}"
        )
        print("-" * 110)

        # Show ALL atoms
        # Calculate final displaced positions after all transformations
        final_displaced_positions = reference_positions_in_ref_cell + displacement
        for i in range(len(reference_positions)):
            ref_pos = reference_positions[i]
            disp_pos = final_displaced_positions[i]
            disp_vec = displacement[i]
            print(
                f"{i:<5} {symbols[i]:<8} "
                f"({ref_pos[0]:8.4f}, {ref_pos[1]:8.4f}, {ref_pos[2]:8.4f})   "
                f"({disp_pos[0]:8.4f}, {disp_pos[1]:8.4f}, {disp_pos[2]:8.4f})   "
                f"({disp_vec[0]:8.4f}, {disp_vec[1]:8.4f}, {disp_vec[2]:8.4f})"
            )

        print("-" * 110)
        print(f"Total atoms: {len(reference_positions)}")
        print(f"{'=' * 110}")

    # Flatten to 1D vector (keep as raw displacement, not mass-weighted)
    # The decomposition function will handle mass-weighting internally
    displacement_vector = displacement.flatten()

    # Calculate mass-weighted norm (always for consistency)
    masses = reference_atoms.get_masses()
    supercell_masses = masses
    mass_weighted_norm_sq = np.sum(
        supercell_masses[:, np.newaxis] * np.abs(displacement.reshape(-1, 3)) ** 2
    )
    displacement_norm = float(np.sqrt(mass_weighted_norm_sq))

    # Normalize if requested
    if normalize:
        # Actually normalize the displacement vector
        if displacement_norm > 1e-10:
            displacement_vector = displacement_vector / displacement_norm
        else:
            raise ValueError("Displacement norm is too small to normalize")

    # Save processed displaced structure if requested
    if output_structure_path:
        # CRITICAL: Create output structure from REFERENCE structure, not displaced!
        # This ensures the atom ordering matches the verification table
        processed_displaced = reference_atoms.copy()

        # CRITICAL: Use the EXACT same positions shown in the verification table
        # final_displaced_positions = reference_positions_in_ref_cell + displacement
        # This ensures consistency between text output and saved file
        processed_positions_final = reference_positions_in_ref_cell + displacement

        # Wrap to [0,1) for output file (standard VASP convention)
        # Convert to fractional, wrap, then back to Cartesian
        ref_cell_array = reference_atoms.get_cell().array
        processed_positions_frac = processed_positions_final @ np.linalg.inv(
            ref_cell_array
        )
        processed_positions_frac = np.mod(processed_positions_frac, 1.0)
        processed_positions_final = processed_positions_frac @ ref_cell_array

        # Update the structure with processed positions
        processed_displaced.set_positions(processed_positions_final)

        # Save to file
        try:
            from ase.io import write as ase_write

            ase_write(
                output_structure_path,
                processed_displaced,
                format="vasp",
                vasp5=True,
                sort=True,
            )
            print(
                f"\n✅ Processed displaced structure saved to: {output_structure_path}"
            )
            print("   Structure includes atom mapping, PBC shifts, and COM alignment")
        except Exception as e:
            print(f"\n⚠️  Warning: Failed to save processed structure: {e}")

    return displacement_vector, displacement_norm


def analyze_phase_scan(
    phonopy_data: dict,
    supercell_matrix: np.ndarray,
    displacement_vector: np.ndarray,
    target_structure=None,
    n_points: int = 36,
    sort_by_contribution: bool = True,
    output_file: Optional[str] = None,
    quiet: bool = False,
):
    """
    Perform phase-resolved projection analysis for all phonon modes.

    Args:
        phonopy_data: Dictionary with phonopy data
        supercell_matrix: 3x3 supercell matrix
        displacement_vector: Flat displacement vector (n_atoms * 3,)
        target_structure: Optional ASE Atoms object for target structure
        n_points: Number of phase sample points
        sort_by_contribution: Whether to sort results by contribution magnitude
        output_file: Optional file to write results to
        quiet: Whether to suppress output to stdout
    """
    from phonproj.core.structure_analysis import (
        project_displacement_with_phase_scan,
        print_decomposition_table,
        print_qpoint_summary_table,
    )
    from phonproj.core.io import _calculate_phonons_at_kpoints

    # Generate commensurate q-points for the supercell
    det = int(np.round(np.linalg.det(supercell_matrix)))

    # Generate unique commensurate q-points (exploiting time-reversal symmetry)
    qpoints = generate_commensurate_qpoints(supercell_matrix)

    n1, n2, n3 = (
        int(round(supercell_matrix[0, 0])),
        int(round(supercell_matrix[1, 1])),
        int(round(supercell_matrix[2, 2])),
    )

    if not quiet:
        print(f"\n{'=' * 90}")
        print("PHONON MODE PHASE SCAN ANALYSIS")
        print(f"{'=' * 90}")
        print(f"Supercell: {n1}×{n2}×{n3} (det={det})")
        print(f"Commensurate q-points: {len(qpoints)}")
        print(f"Phase sample points per mode: {n_points}")
        print(f"Results sorted by contribution: {sort_by_contribution}")

        print(f"\nCommensurate q-points for {n1}×{n2}×{n3} supercell:")
        for i, qpt in enumerate(qpoints):
            print(f"  {i:2d}. [{qpt[0]:.6f}, {qpt[1]:.6f}, {qpt[2]:.6f}]")

    # Load phonon modes at commensurate q-points
    if not quiet:
        print(f"\nLoading phonon modes at commensurate q-points...")

    phonopy = phonopy_data["phonopy"]
    primitive_cell = phonopy_data["primitive_cell"]

    # Calculate phonon modes at commensurate q-points
    frequencies, eigenvectors = _calculate_phonons_at_kpoints(phonopy, qpoints)

    # Create PhononModes object
    phonon_modes = PhononModes(
        primitive_cell=primitive_cell,
        qpoints=qpoints,
        frequencies=frequencies,
        eigenvectors=eigenvectors,
        atomic_masses=None,  # Will be inferred from primitive_cell
        gauge="R",
    )

    n_modes = phonon_modes.frequencies.shape[1]
    if not quiet:
        print(
            f"✅ Loaded {len(phonon_modes.qpoints)} q-points with {n_modes} modes each"
        )

    # Reshape displacement vector to (n_atoms, 3)
    n_atoms_supercell = len(displacement_vector) // 3
    displacement_reshaped = displacement_vector.reshape(n_atoms_supercell, 3)

    # DEBUG: Print displacement statistics
    print(f"\n[DEBUG PHASE-SCAN] Displacement shape: {displacement_reshaped.shape}")
    print(
        f"[DEBUG PHASE-SCAN] Displacement vector norm: {np.linalg.norm(displacement_vector):.6f}"
    )
    print(f"[DEBUG PHASE-SCAN] Displacement first 3 values: {displacement_vector[:3]}")

    if not quiet:
        print(
            f"\nPerforming phase scan for all {len(qpoints)} q-points and {n_modes} modes..."
        )

    # Perform phase scan for all q-points and modes
    projection_table = []
    total_squared_projections = 0.0
    total_calculations = len(qpoints) * n_modes

    # Pre-generate mode displacements per q-point for efficiency
    if not quiet:
        print("Pre-generating mode displacements for all q-points...")

    all_q_mode_displacements = {}
    for q_index in range(len(qpoints)):
        if not quiet and q_index % 4 == 0:
            print(f"  Generating modes for q-point {q_index + 1}/{len(qpoints)}...")
        all_q_mode_displacements[q_index] = (
            phonon_modes.generate_all_mode_displacements(
                q_index=q_index, supercell_matrix=supercell_matrix, amplitude=1.0
            )
        )

    if not quiet:
        print("✅ Mode displacements generated\n")

    for q_index in range(len(qpoints)):
        q_point = phonon_modes.qpoints[q_index]
        q_frequencies = phonon_modes.frequencies[q_index]

        for mode_index in range(n_modes):
            frequency = q_frequencies[mode_index]

            # Show progress
            current = q_index * n_modes + mode_index + 1
            if not quiet and current % 50 == 0:
                print(
                    f"  Progress: {current}/{total_calculations} ({100 * current / total_calculations:.1f}%)"
                )

            # Perform phase scan for this mode using pre-generated displacements
            max_coeff, optimal_phase = project_displacement_with_phase_scan(
                phonon_modes=phonon_modes,
                target_displacement=displacement_reshaped,
                supercell_matrix=supercell_matrix,
                q_index=q_index,
                mode_index=mode_index,
                n_phases=n_points,
                target_supercell=target_structure,
                precomputed_mode_displacements=all_q_mode_displacements.get(q_index),
            )

            squared_coeff = max_coeff**2
            total_squared_projections += squared_coeff

            # Store projection data
            projection_data = {
                "q_index": q_index,
                "q_point": q_point.copy(),
                "mode_index": mode_index,
                "frequency": frequency,
                "projection_coefficient": max_coeff,
                "squared_coefficient": squared_coeff,
                "optimal_phase": optimal_phase,
            }
            projection_table.append(projection_data)

    # Sort table by contribution magnitude (largest first) if requested
    if sort_by_contribution:
        projection_table.sort(key=lambda x: abs(x["squared_coefficient"]), reverse=True)

    # DEBUG: Print first 5 contributions and total
    print(
        f"\n[DEBUG PHASE-SCAN] Total squared projections: {total_squared_projections:.6f}"
    )
    print("[DEBUG PHASE-SCAN] First 5 contributions:")
    for i, entry in enumerate(projection_table[:5]):
        print(
            f"  {i + 1}. q={entry['q_index']}, mode={entry['mode_index']}: coeff={entry['projection_coefficient']:.6f}, squared={entry['squared_coefficient']:.6f}"
        )

    # Calculate summary statistics
    summary = {
        "sum_squared_projections": total_squared_projections,
        "expected_sum": 1.0,  # Assuming normalized displacement
        "completeness_error": abs(total_squared_projections - 1.0),
        "is_complete": abs(total_squared_projections - 1.0) < 1e-6,
        "tolerance": 1e-6,
        "n_modes_total": len(projection_table),
        "n_qpoints": len(qpoints),
        "largest_contribution": projection_table[0]["squared_coefficient"]
        if projection_table
        else 0.0,
        "smallest_contribution": projection_table[-1]["squared_coefficient"]
        if projection_table
        else 0.0,
    }

    # Print results to stdout
    if not quiet:
        print_decomposition_table(
            projection_table=projection_table,
            summary=summary,
            max_entries=None,  # Show all entries
            min_contribution=1e-10,
        )

        # Print q-point summary table
        print_qpoint_summary_table(
            projection_table=projection_table,
            summary=summary,
        )

    # Write to file if requested
    if output_file:
        import sys
        from io import StringIO

        # Capture print output to file
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        print_decomposition_table(
            projection_table=projection_table,
            summary=summary,
            max_entries=None,
            min_contribution=1e-10,
        )

        print_qpoint_summary_table(
            projection_table=projection_table,
            summary=summary,
        )

        # Restore stdout
        sys.stdout = old_stdout

        # Write to file
        with open(output_file, "w") as f:
            f.write(captured_output.getvalue())

        if not quiet:
            print(f"\n✅ Results written to: {output_file}")


def analyze_displacement(
    phonopy_data: dict,
    supercell_matrix: np.ndarray,
    displacement_vector: np.ndarray,
    normalize: bool = False,
    sort_by_contribution: bool = True,
    output_file: Optional[str] = None,
    quiet: bool = False,
    remove_com: bool = False,
):
    """
    Analyze displacement in terms of phonon mode contributions.

    Args:
        phonopy_data: Dictionary with phonopy data
        supercell_matrix: 3x3 supercell matrix
        displacement_vector: Flat displacement vector (n_atoms * 3,)
        normalize: Whether displacement is mass-weight normalized
        sort_by_contribution: Whether to sort results by contribution magnitude
        output_file: Optional file to write results to
        quiet: Whether to suppress output
        remove_com: Whether acoustic modes were removed (their contributions will be set to 0)
    """
    from phonproj.core.structure_analysis import (
        decompose_displacement_to_modes,
        print_decomposition_table,
        print_qpoint_summary_table,
    )

    # Generate commensurate q-points for the supercell
    det = int(np.round(np.linalg.det(supercell_matrix)))

    # Generate unique commensurate q-points (exploiting time-reversal symmetry)
    qpoints = generate_commensurate_qpoints(supercell_matrix)

    n1, n2, n3 = (
        int(round(supercell_matrix[0, 0])),
        int(round(supercell_matrix[1, 1])),
        int(round(supercell_matrix[2, 2])),
    )

    if not quiet:
        print(f"\n{'=' * 90}")
        print("PHONON MODE DECOMPOSITION ANALYSIS")
        print(f"{'=' * 90}")
        print(f"Supercell: {n1}×{n2}×{n3} (det={det})")
        print(f"Commensurate q-points: {len(qpoints)}")
        print(f"Displacement normalized: {normalize}")
        print(f"Results sorted by contribution: {sort_by_contribution}")

        # Print list of q-points
        print(f"\nCommensurate q-points for {n1}×{n2}×{n3} supercell:")
        for i, qpt in enumerate(qpoints):
            print(f"  {i:2d}. [{qpt[0]:.6f}, {qpt[1]:.6f}, {qpt[2]:.6f}]")

        # Load phonon modes at commensurate q-points
        print("\nLoading phonon modes at commensurate q-points...")

    # Get the phonopy object and calculate modes
    from phonproj.core.io import _calculate_phonons_at_kpoints

    phonopy = phonopy_data["phonopy"]
    primitive_cell = phonopy_data["primitive_cell"]

    # Calculate phonon modes at commensurate q-points
    frequencies, eigenvectors = _calculate_phonons_at_kpoints(phonopy, qpoints)

    # Create PhononModes object
    phonon_modes = PhononModes(
        primitive_cell=primitive_cell,
        qpoints=qpoints,
        frequencies=frequencies,
        eigenvectors=eigenvectors,
        atomic_masses=None,  # Will be inferred from primitive_cell
        gauge="R",
    )

    if not quiet:
        print(
            f"✅ Loaded {len(phonon_modes.qpoints)} q-points with {phonon_modes.frequencies.shape[1]} modes each"
        )

    # Reshape displacement vector to (n_atoms, 3)
    n_atoms_supercell = len(displacement_vector) // 3
    displacement_reshaped = displacement_vector.reshape(n_atoms_supercell, 3)

    # DEBUG: Print displacement statistics
    print(f"\n[DEBUG NON-PHASE-SCAN] Displacement shape: {displacement_reshaped.shape}")
    print(
        f"[DEBUG NON-PHASE-SCAN] Displacement vector norm: {np.linalg.norm(displacement_vector):.6f}"
    )
    print(
        f"[DEBUG NON-PHASE-SCAN] Displacement first 3 values: {displacement_vector[:3]}"
    )

    if not quiet:
        print("\nCalculating mode decomposition...")

    # Use the existing decompose_displacement_to_modes function
    # Note: If normalize=True, we already normalized in calculate_displacement_vector,
    # so we pass normalize=False here to avoid double normalization
    projection_table, summary = decompose_displacement_to_modes(
        displacement=displacement_reshaped,
        phonon_modes=phonon_modes,
        supercell_matrix=supercell_matrix,
        normalize=False,  # Already normalized if requested
        tolerance=1e-6,
        sort_by_contribution=sort_by_contribution,
    )

    # DEBUG: Print first 5 contributions and total
    print(
        f"\n[DEBUG NON-PHASE-SCAN] Total squared projections: {summary['sum_squared_projections']:.6f}"
    )
    print("[DEBUG NON-PHASE-SCAN] First 5 contributions:")
    for i, entry in enumerate(projection_table[:5]):
        print(
            f"  {i + 1}. q={entry['q_index']}, mode={entry['mode_index']}: coeff={entry['projection_coefficient']:.6f}, squared={entry['squared_coefficient']:.6f}"
        )

    # If acoustic modes were removed, set their contributions to 0
    # Acoustic modes are at q=[0,0,0] with frequency ≈ 0
    if remove_com:
        acoustic_threshold = 1e-2  # cm^-1, modes below this are considered acoustic
        for entry in projection_table:
            q_point = entry["q_point"]
            frequency = entry["frequency"]
            # Check if this is Gamma point and low frequency (acoustic mode)
            is_gamma = np.allclose(q_point, [0, 0, 0], atol=1e-6)
            is_acoustic = abs(frequency) < acoustic_threshold
            if is_gamma and is_acoustic:
                entry["projection_coefficient"] = 0.0
                entry["squared_coefficient"] = 0.0

        # Recalculate summary statistics after zeroing acoustic modes
        total_squared_projections = sum(
            e["squared_coefficient"] for e in projection_table
        )
        summary["sum_squared_projections"] = total_squared_projections

        # Re-sort if needed
        if sort_by_contribution:
            projection_table.sort(
                key=lambda x: abs(x["squared_coefficient"]), reverse=True
            )

    # Print results to stdout
    if not quiet:
        print_decomposition_table(
            projection_table=projection_table,
            summary=summary,
            max_entries=None,  # Show all entries
            min_contribution=1e-10,
        )

        # Print q-point summary table
    if not quiet:
        print_qpoint_summary_table(
            projection_table=projection_table,
            summary=summary,
        )

    # Write to file if requested
    if output_file:
        import sys
        from io import StringIO

        # Capture print output to file
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        print_decomposition_table(
            projection_table=projection_table,
            summary=summary,
            max_entries=None,
            min_contribution=1e-10,
        )

        print_qpoint_summary_table(
            projection_table=projection_table,
            summary=summary,
        )

        # Restore stdout
        sys.stdout = old_stdout

        # Write to file
        with open(output_file, "w") as f:
            f.write(captured_output.getvalue())

        if not quiet:
            print(f"\n✅ Results written to: {output_file}")


def print_supercell_displacements(
    modes: PhononModes, supercell_matrix: np.ndarray, amplitude: float
):
    """
    Print all supercell displacements for commensurate q-points.

    Args:
        modes: PhononModes object
        supercell_matrix: 3x3 supercell transformation matrix
        amplitude: Displacement amplitude
    """
    print(f"\n=== Supercell Displacements (amplitude = {amplitude}) ===")

    # Get all commensurate q-points
    commensurate_qpoints = modes.get_commensurate_qpoints(supercell_matrix)

    if not commensurate_qpoints:
        print("No commensurate q-points found for the given supercell matrix.")
        return

    print(f"Found {len(commensurate_qpoints)} commensurate q-points:")

    for q_idx in commensurate_qpoints:
        # Ensure q_idx is an integer
        q_idx_int = int(q_idx)
        qpoint = modes.qpoints[q_idx_int]
        print(
            f"\nQ-point {q_idx_int}: [{qpoint[0]:.3f}, {qpoint[1]:.3f}, {qpoint[2]:.3f}]"
        )
        print("-" * 50)

        # Generate displacements for all modes at this q-point
        for mode_idx in range(modes.n_modes):
            try:
                displacement = modes.generate_mode_displacement(
                    q_idx_int, mode_idx, supercell_matrix, amplitude=amplitude
                )

                # Print displacement info with limited precision
                freq = modes.frequencies[q_idx_int, mode_idx]
                print(f"Mode {mode_idx:2d} (freq = {freq:8.2f} cm⁻¹):")

                # Print a few representative atoms (not all to avoid too much output)
                n_atoms_to_show = min(5, len(displacement))
                for atom_idx in range(n_atoms_to_show):
                    disp = displacement[atom_idx]
                    if np.iscomplexobj(disp):
                        # Print complex displacement
                        print(
                            f"  Atom {atom_idx:2d}: ({disp.real:8.4f}, {disp.imag:8.4f}i) Å"
                        )
                    else:
                        # Print real displacement
                        print(
                            f"  Atom {atom_idx:2d}: ({disp[0]:8.4f}, {disp[1]:8.4f}, {disp[2]:8.4f}) Å"
                        )

                if len(displacement) > n_atoms_to_show:
                    print(f"  ... and {len(displacement) - n_atoms_to_show} more atoms")

            except Exception as e:
                print(f"  Mode {mode_idx:2d}: Error generating displacement - {e}")


def save_supercell_structures(
    modes: PhononModes, supercell_matrix: np.ndarray, amplitude: float, output_dir: str
):
    """
    Save all supercell structures with displacements to directory in VASP format.

    Args:
        modes: PhononModes object
        supercell_matrix: 3x3 supercell transformation matrix
        amplitude: Displacement amplitude
        output_dir: Directory to save VASP files
    """
    from pathlib import Path

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Saving Supercell Structures to {output_dir} ===")

    # Get all commensurate q-points
    commensurate_qpoints = modes.get_commensurate_qpoints(supercell_matrix)

    if not commensurate_qpoints:
        print("No commensurate q-points found for the given supercell matrix.")
        return

    print(f"Found {len(commensurate_qpoints)} commensurate q-points")

    # Save structures for each commensurate q-point and mode
    for q_idx in commensurate_qpoints:
        # Ensure q_idx is an integer
        q_idx_int = int(q_idx)
        qpoint = modes.qpoints[q_idx_int]
        q_str = f"q{q_idx_int}_{''.join(f'{c:.2f}'.replace('.', 'p').replace('-', 'm') for c in qpoint)}"

        for mode_idx in range(modes.n_modes):
            try:
                # Generate displacement
                displacement = modes.generate_mode_displacement(
                    q_idx_int, mode_idx, supercell_matrix, amplitude=amplitude
                )

                # Generate base supercell structure first
                supercell_structure = modes.generate_supercell(supercell_matrix)

                # Apply displacement to supercell
                supercell_structure.set_positions(
                    supercell_structure.get_positions() + displacement
                )

                # Create filename
                freq = modes.frequencies[q_idx_int, mode_idx]
                filename = f"{q_str}_mode{mode_idx:02d}_freq{freq:6.1f}.vasp"
                filepath = output_path / filename

                # Save in VASP format
                from ase.io import write

                write(filepath, supercell_structure, format="vasp")

                print(f"  Saved: {filename}")

            except Exception as e:
                print(f"  Error saving q{q_idx_int}_mode{mode_idx}: {e}")

    print(
        f"\n✅ Saved {len(commensurate_qpoints) * modes.n_modes} supercell structures to {output_dir}"
    )


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Phonon Mode Decomposition Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze with phonopy directory
  phonproj-decompose -p data/phonopy_dir -s 2x2x2 -d displaced.vasp

  # Analyze with phonopy_params.yaml
  phonproj-decompose -p phonopy_params.yaml -s 16x1x1 -d CONTCAR

  # Analyze with ISODISTORT file
  phonproj-decompose -p phonopy_params.yaml -s 2x2x2 --isodistort structures.txt

  # Normalize displacement and save output
  phonproj-decompose -p phonopy_params.yaml -s 2x2x2 -d displaced.vasp --normalize -o output.txt

  # Save processed displaced structure after mapping and PBC shifts
  phonproj-decompose -p phonopy_params.yaml -s 2x2x2 -d displaced.vasp --output-structure processed.vasp

  # Print all supercell displacements with custom amplitude
  phonproj-decompose -p phonopy_params.yaml -s 2x2x2 -d displaced.vasp --print-displacements --amplitude 0.05

  # Save all supercell structures with displacements to directory
  phonproj-decompose -p phonopy_params.yaml -s 2x2x2 -d displaced.vasp --save-supercells output_structures --amplitude 0.1
        """,
    )

    parser.add_argument(
        "-p",
        "--phonopy",
        required=True,
        help="Path to phonopy_params.yaml file or directory containing phonopy files",
    )

    parser.add_argument(
        "-s",
        "--supercell",
        required=True,
        help="Supercell size in format NxMxL (e.g., 2x2x2, 16x1x1) or 'N M L' (e.g., '16 1 1')",
    )

    parser.add_argument(
        "-d",
        "--displaced",
        required=False,
        help="Path to displaced structure file (VASP POSCAR/CONTCAR format)",
    )

    parser.add_argument(
        "--isodistort",
        required=False,
        help="Path to ISODISTORT file containing undistorted and distorted structures",
    )

    parser.add_argument(
        "-r",
        "--reference",
        default=None,
        help="Path to reference structure file (optional, if different from phonopy supercell)",
    )

    parser.add_argument(
        "-i",
        "--input-structure",
        default=None,
        help="Path to input structure file used to determine atom mapping and PBC shifts (optional)",
    )

    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Mass-weight normalize the displacement vector before projection",
    )

    parser.add_argument(
        "--no-sort",
        action="store_true",
        help="Do not sort projection results by contribution magnitude",
    )

    parser.add_argument(
        "--species-map",
        default=None,
        help="Species substitution mapping in format 'A:B,C:D' (e.g., 'Pb:Sr' to map Pb to Sr)",
    )

    parser.add_argument(
        "--remove-com",
        action="store_true",
        help="Align structures by center of mass and project out acoustic translations",
    )

    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output file for results (default: print to stdout only)",
    )

    parser.add_argument(
        "--output-structure",
        default=None,
        help="Output file for processed displaced structure after mapping and PBC shifts (VASP format)",
    )

    # Phase scan options
    parser.add_argument(
        "--phase-scan",
        action="store_true",
        help="Enable phase-resolved projection analysis for all modes instead of full decomposition",
    )

    parser.add_argument(
        "--phase-scan-points",
        type=int,
        default=36,
        help="Number of phase sample points between 0 and pi for phase scan (default: 36)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity",
    )

    parser.add_argument(
        "--print-displacements",
        action="store_true",
        help="Print all supercell displacements for commensurate q-points",
    )

    parser.add_argument(
        "--amplitude",
        type=float,
        default=0.1,
        help="Amplitude for displacement generation (default: 0.1)",
    )

    parser.add_argument(
        "--save-supercells",
        type=str,
        metavar="DIRECTORY",
        help="Save all supercell structures with displacements to specified directory in VASP format",
    )

    args = parser.parse_args()

    # Validate input arguments
    if not args.displaced and not args.isodistort:
        parser.error("Must specify either --displaced or --isodistort")
    if args.displaced and args.isodistort:
        parser.error("Cannot specify both --displaced and --isodistort")

    try:
        # Parse supercell matrix
        supercell_matrix = parse_supercell_matrix(args.supercell)

        # Parse species mapping if provided
        species_map = None
        if args.species_map:
            species_map = {}
            for pair in args.species_map.split(","):
                if ":" not in pair:
                    raise ValueError(
                        f"Invalid species mapping format: '{pair}'. Expected 'A:B' format."
                    )
                src, dst = pair.split(":", 1)
                species_map[src.strip()] = dst.strip()
            print(f"\nUsing species mapping: {species_map}")

        # Load phonopy data
        phonopy_data = load_phonopy_data(args.phonopy, quiet=args.quiet)

        # Load structures based on input type
        if args.isodistort:
            # Load from ISODISTORT file
            reference_structure, displaced_structure = load_isodistort_structures(
                args.isodistort
            )
            if not args.quiet:
                print(f"Loaded ISODISTORT file: {args.isodistort}")
                print(f"  Reference atoms: {len(reference_structure)}")
                print(f"  Displaced atoms: {len(displaced_structure)}")
        else:
            # Load from displaced structure file
            displaced_structure = load_displaced_structure(args.displaced)

            # Generate reference supercell
            if args.reference:
                reference_structure = load_displaced_structure(args.reference)
            else:
                # Generate reference supercell from primitive cell
                from ase.build import make_supercell

                primitive_cell = phonopy_data["primitive_cell"]
                reference_structure = make_supercell(primitive_cell, supercell_matrix)

        # Load input structure if provided
        input_structure = None
        if args.input_structure:
            input_structure = load_displaced_structure(args.input_structure)

        if not args.quiet:
            print(f"\n{'=' * 90}")
            print("INPUT SUMMARY")
            print(f"{'=' * 90}")
            print(f"Phonopy data: {args.phonopy}")
            print(f"Supercell: {args.supercell}")
            if args.isodistort:
                print(f"ISODISTORT file: {args.isodistort}")
            else:
                print(f"Displaced structure: {args.displaced}")
            print(
                f"Reference structure: {args.reference if args.reference else 'Generated from primitive cell'}"
            )
            print(
                f"Input structure: {args.input_structure if args.input_structure else 'Not provided (using direct mapping)'}"
            )
            print(f"Normalize: {args.normalize}")
            print(f"Displaced atoms: {len(displaced_structure)}")
            print(f"Reference atoms: {len(reference_structure)}")
            if input_structure:
                print(f"Input atoms: {len(input_structure)}")

        # Verify atom counts match
        if len(displaced_structure) != len(reference_structure):
            raise ValueError(
                f"Atom count mismatch: displaced={len(displaced_structure)}, "
                f"reference={len(reference_structure)}"
            )

        # Calculate displacement vector
        if not args.quiet:
            print("\nCalculating displacement vector...")
        displacement_vector, displacement_norm = calculate_displacement_vector(
            displaced_structure,
            reference_structure,
            normalize=args.normalize,
            verbose=not args.quiet,
            species_map=species_map,
            remove_com=args.remove_com,
            output_structure_path=args.output_structure,
            input_structure=input_structure,
        )

        if not args.quiet:
            print("✅ Displacement calculated")
            print(f"   Mass-weighted norm: {displacement_norm:.6f}")
            print(f"   Vector length: {len(displacement_vector)}")

        # Store directory/yaml path in phonopy_data for later use
        phonopy_path = Path(args.phonopy)
        if phonopy_path.is_dir():
            phonopy_data["phonopy_directory"] = str(phonopy_path)
        else:
            phonopy_data["phonopy_yaml"] = str(phonopy_path)

        # Analyze displacement
        if args.phase_scan:
            analyze_phase_scan(
                phonopy_data,
                supercell_matrix,
                displacement_vector,
                target_structure=displaced_structure,
                n_points=args.phase_scan_points,
                sort_by_contribution=not args.no_sort,
                output_file=args.output,
                quiet=args.quiet,
            )
        else:
            analyze_displacement(
                phonopy_data,
                supercell_matrix,
                displacement_vector,
                normalize=args.normalize,
                sort_by_contribution=not args.no_sort,
                output_file=args.output,
                quiet=args.quiet,
                remove_com=args.remove_com,
            )

        # Handle additional functionality: print displacements and/or save supercells
        if args.print_displacements or args.save_supercells:
            # Load phonon modes for the new functionality
            from phonproj.core.io import _calculate_phonons_at_kpoints

            # Generate unique commensurate q-points (exploiting time-reversal symmetry)
            qpoints = generate_commensurate_qpoints(supercell_matrix)

            n1, n2, n3 = (
                int(round(supercell_matrix[0, 0])),
                int(round(supercell_matrix[1, 1])),
                int(round(supercell_matrix[2, 2])),
            )

            # Get the phonopy object and calculate modes
            phonopy = phonopy_data["phonopy"]
            primitive_cell = phonopy_data["primitive_cell"]

            # Calculate phonon modes at commensurate q-points
            frequencies, eigenvectors = _calculate_phonons_at_kpoints(phonopy, qpoints)

            # Create PhononModes object
            phonon_modes = PhononModes(
                primitive_cell=primitive_cell,
                qpoints=qpoints,
                frequencies=frequencies,
                eigenvectors=eigenvectors,
                atomic_masses=None,  # Will be inferred from primitive_cell
                gauge="R",
            )

            # Print displacements if requested
            if args.print_displacements:
                print_supercell_displacements(
                    phonon_modes, supercell_matrix, args.amplitude
                )

            # Save supercell structures if requested
            if args.save_supercells:
                save_supercell_structures(
                    phonon_modes, supercell_matrix, args.amplitude, args.save_supercells
                )

        if not args.quiet:
            print(f"\n{'=' * 90}")
            print("✅ ANALYSIS COMPLETED SUCCESSFULLY")
            print(f"{'=' * 90}")

    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
