#!/usr/bin/env python3
"""
Command-line interface for phonon mode decomposition analysis.

This tool analyzes structural displacements in terms of phonon mode contributions.
"""

import argparse
import sys
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from ase.io import read as ase_read

from phonproj.modes import PhononModes
from phonproj.core import load_from_phonopy_files, load_yaml_file


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

        n, m, l = [int(p) for p in parts]
        if n <= 0 or m <= 0 or l <= 0:
            raise ValueError("Supercell dimensions must be positive")

        return np.array([[n, 0, 0], [0, m, 0], [0, 0, l]])
    except (ValueError, IndexError) as e:
        raise ValueError(f"Invalid supercell format '{supercell_str}': {e}")


def load_phonopy_data(phonopy_path: str):
    """
    Load phonopy data from file or directory.

    Args:
        phonopy_path: Path to phonopy_params.yaml file or directory containing phonopy files

    Returns:
        Dictionary with phonopy data including 'phonopy', 'primitive_cell', etc.
    """
    path = Path(phonopy_path)

    if not path.exists():
        raise FileNotFoundError(f"Path not found: {phonopy_path}")

    if path.is_dir():
        print(f"Loading phonopy data from directory: {path}")
        return load_from_phonopy_files(path)
    elif path.is_file():
        print(f"Loading phonopy data from file: {path}")
        return load_yaml_file(path)
    else:
        raise ValueError(f"Invalid path type: {phonopy_path}")


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
        )


def calculate_displacement_vector(
    displaced_atoms,
    reference_atoms,
    normalize: bool = False,
    verbose: bool = True,
    species_map: Optional[dict] = None,
    remove_com: bool = False,
    output_structure_path: Optional[str] = None,
) -> Tuple[np.ndarray, float]:
    """
    Calculate displacement vector between two structures.

    This function handles atom reordering and periodic boundary conditions.

    Args:
        displaced_atoms: Displaced structure (ASE Atoms)
        reference_atoms: Reference structure (ASE Atoms)
        normalize: Whether to mass-weight normalize the displacement
        verbose: Whether to print atom mapping verification table
        species_map: Optional dict mapping species substitutions (e.g., {'Pb': 'Sr'})
        remove_com: Whether to align structures by COM and project out acoustic modes
        output_structure_path: Optional path to save the processed displaced structure
                               (after mapping and PBC shifts)

    Returns:
        Tuple of (displacement_vector, displacement_norm)
        displacement_vector has shape (n_atoms * 3,)
    """
    from phonproj.core.structure_analysis import (
        create_atom_mapping,
        project_out_acoustic_modes,
    )

    # Find optimal atom mapping
    # mapping[i] gives the index in displaced_atoms for atom i in reference_atoms
    mapping, _ = create_atom_mapping(
        reference_atoms,
        displaced_atoms,
        max_cost=100.0,
        warn_threshold=0.5,
        species_map=species_map,
    )

    # Reorder displaced atoms according to mapping
    displaced_positions = displaced_atoms.get_positions()[mapping]
    reference_positions = reference_atoms.get_positions()

    # Get cell matrices as numpy arrays for calculations
    ref_cell_array = reference_atoms.get_cell().array
    disp_cell_array = displaced_atoms.get_cell().array

    # Convert positions to fractional coordinates (each in their own cell)
    reference_positions_frac = reference_positions @ np.linalg.inv(ref_cell_array)
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
        # Calculate COM in the reference cell coordinate system
        ref_masses = reference_atoms.get_masses()
        total_mass = np.sum(ref_masses)

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
            print(f"   Structure includes atom mapping, PBC shifts, and COM alignment")
        except Exception as e:
            print(f"\n⚠️  Warning: Failed to save processed structure: {e}")

    return displacement_vector, displacement_norm


def analyze_phase_scan(
    phonopy_data: dict,
    supercell_matrix: np.ndarray,
    displacement_vector: np.ndarray,
    n_points: int = 36,
    sort_by_contribution: bool = True,
    output_file: Optional[str] = None,
):
    """
    Perform phase-resolved projection analysis for all phonon modes.

    Args:
        phonopy_data: Dictionary with phonopy data
        supercell_matrix: 3x3 supercell matrix
        displacement_vector: Flat displacement vector (n_atoms * 3,)
        n_points: Number of phase sample points
        sort_by_contribution: Whether to sort results by contribution magnitude
        output_file: Optional file to write results to
    """
    from phonproj.core.structure_analysis import (
        project_displacement_with_phase_scan,
        print_decomposition_table,
        print_qpoint_summary_table,
    )
    from phonproj.core.io import _calculate_phonons_at_kpoints

    # Generate commensurate q-points for the supercell
    det = int(np.round(np.linalg.det(supercell_matrix)))

    # Generate commensurate q-points
    qpoints = []
    n1, n2, n3 = (
        int(supercell_matrix[0, 0]),
        int(supercell_matrix[1, 1]),
        int(supercell_matrix[2, 2]),
    )

    for i in range(n1):
        for j in range(n2):
            for k in range(n3):
                qpoints.append([i / n1, j / n2, k / n3])

    qpoints = np.array(qpoints)

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
    print(f"✅ Loaded {len(phonon_modes.qpoints)} q-points with {n_modes} modes each")

    # Reshape displacement vector to (n_atoms, 3)
    n_atoms_supercell = len(displacement_vector) // 3
    displacement_reshaped = displacement_vector.reshape(n_atoms_supercell, 3)

    print(
        f"\nPerforming phase scan for all {len(qpoints)} q-points and {n_modes} modes..."
    )

    # Perform phase scan for all q-points and modes
    projection_table = []
    total_squared_projections = 0.0

    for q_index in range(len(qpoints)):
        q_point = phonon_modes.qpoints[q_index]
        q_frequencies = phonon_modes.frequencies[q_index]

        for mode_index in range(n_modes):
            frequency = q_frequencies[mode_index]

            # Perform phase scan for this mode
            max_coeff, optimal_phase = project_displacement_with_phase_scan(
                phonon_modes=phonon_modes,
                target_displacement=displacement_reshaped,
                supercell_matrix=supercell_matrix,
                q_index=q_index,
                mode_index=mode_index,
                n_phases=n_points,
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

        print(f"\n✅ Results written to: {output_file}")


def analyze_displacement(
    phonopy_data: dict,
    supercell_matrix: np.ndarray,
    displacement_vector: np.ndarray,
    normalize: bool = False,
    sort_by_contribution: bool = True,
    output_file: Optional[str] = None,
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
    """
    from phonproj.core.structure_analysis import (
        decompose_displacement_to_modes,
        print_decomposition_table,
    )

    # Generate commensurate q-points for the supercell
    det = int(np.round(np.linalg.det(supercell_matrix)))

    # Generate commensurate q-points
    qpoints = []
    n1, n2, n3 = (
        int(supercell_matrix[0, 0]),
        int(supercell_matrix[1, 1]),
        int(supercell_matrix[2, 2]),
    )

    for i in range(n1):
        for j in range(n2):
            for k in range(n3):
                qpoints.append([i / n1, j / n2, k / n3])

    qpoints = np.array(qpoints)

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
    print(f"\nLoading phonon modes at commensurate q-points...")

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

    print(
        f"✅ Loaded {len(phonon_modes.qpoints)} q-points with {phonon_modes.frequencies.shape[1]} modes each"
    )

    # Reshape displacement vector to (n_atoms, 3)
    n_atoms_supercell = len(displacement_vector) // 3
    displacement_reshaped = displacement_vector.reshape(n_atoms_supercell, 3)

    print(f"\nCalculating mode decomposition...")

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

    # Print results to stdout
    print_decomposition_table(
        projection_table=projection_table,
        summary=summary,
        max_entries=None,  # Show all entries
        min_contribution=1e-10,
    )

    # Print q-point summary table
    from phonproj.core.structure_analysis import print_qpoint_summary_table

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

        print(f"\n✅ Results written to: {output_file}")


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

  # Normalize displacement and save output
  phonproj-decompose -p phonopy_params.yaml -s 2x2x2 -d displaced.vasp --normalize -o output.txt

  # Save processed displaced structure after mapping and PBC shifts
  phonproj-decompose -p phonopy_params.yaml -s 2x2x2 -d displaced.vasp --output-structure processed.vasp
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
        required=True,
        help="Path to displaced structure file (VASP POSCAR/CONTCAR format)",
    )

    parser.add_argument(
        "-r",
        "--reference",
        default=None,
        help="Path to reference structure file (optional, if different from phonopy supercell)",
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

    args = parser.parse_args()

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
        phonopy_data = load_phonopy_data(args.phonopy)

        # Load displaced structure
        displaced_structure = load_displaced_structure(args.displaced)

        # Generate reference supercell
        if args.reference:
            reference_structure = load_displaced_structure(args.reference)
        else:
            # Generate reference supercell from primitive cell
            from ase.build import make_supercell

            primitive_cell = phonopy_data["primitive_cell"]
            reference_structure = make_supercell(primitive_cell, supercell_matrix)

        print(f"\n{'=' * 90}")
        print("INPUT SUMMARY")
        print(f"{'=' * 90}")
        print(f"Phonopy data: {args.phonopy}")
        print(f"Supercell: {args.supercell}")
        print(f"Displaced structure: {args.displaced}")
        print(
            f"Reference structure: {args.reference if args.reference else 'Generated from primitive cell'}"
        )
        print(f"Normalize: {args.normalize}")
        print(f"Displaced atoms: {len(displaced_structure)}")
        print(f"Reference atoms: {len(reference_structure)}")

        # Verify atom counts match
        if len(displaced_structure) != len(reference_structure):
            raise ValueError(
                f"Atom count mismatch: displaced={len(displaced_structure)}, "
                f"reference={len(reference_structure)}"
            )

        # Calculate displacement vector
        print(f"\nCalculating displacement vector...")
        displacement_vector, displacement_norm = calculate_displacement_vector(
            displaced_structure,
            reference_structure,
            normalize=args.normalize,
            species_map=species_map,
            remove_com=args.remove_com,
            output_structure_path=args.output_structure,
        )

        print(f"✅ Displacement calculated")
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
                n_points=args.phase_scan_points,
                sort_by_contribution=not args.no_sort,
                output_file=args.output,
            )
        else:
            analyze_displacement(
                phonopy_data,
                supercell_matrix,
                displacement_vector,
                normalize=args.normalize,
                sort_by_contribution=not args.no_sort,
                output_file=args.output,
            )

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
