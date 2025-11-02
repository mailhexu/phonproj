#!/usr/bin/env python3
"""
Debug script to compare phonopy API vs direct eigenvector method
for generating supercell displacements.
"""

import numpy as np
from pathlib import Path
from phonproj.modes import PhononModes
from ase.build import make_supercell

# Data path
BATIO3_YAML_PATH = Path(__file__).parent / "data" / "BaTiO3_phonopy_params.yaml"


def compare_displacement_methods():
    """Compare phonopy API vs direct method for generating displacements."""

    # Set up the same test case as the failing test
    qpoints_2x2x2 = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                qpoints_2x2x2.append([i / 2.0, j / 2.0, k / 2.0])
    qpoints_2x2x2 = np.array(qpoints_2x2x2)

    modes = PhononModes.from_phonopy_yaml(str(BATIO3_YAML_PATH), qpoints_2x2x2)

    # Use 2x2x2 supercell
    supercell_matrix = np.eye(3, dtype=int) * 2

    # Find a non-Gamma q-point (same as the test)
    non_gamma_index = None
    for i, qpoint in enumerate(modes.qpoints):
        if not np.allclose(qpoint, [0, 0, 0], atol=1e-6):
            non_gamma_index = i
            break

    if non_gamma_index is None:
        print("No non-Gamma q-points found!")
        return

    print(f"Using q-point index {non_gamma_index}: {modes.qpoints[non_gamma_index]}")
    print(f"Q-point value: {modes.qpoints[non_gamma_index]}")

    # Test first two modes
    mode_indices = [0, 1]

    print("\n=== PHONOPY API METHOD ===")
    phonopy_displacements = []
    for mode_idx in mode_indices:
        # Use the phonopy API method directly
        displacement = modes._calculate_supercell_displacements_phonopy(
            non_gamma_index, mode_idx, supercell_matrix, amplitude=1.0
        )

        # Apply the same normalization as in generate_all_mode_displacements
        current_norm = modes.mass_weighted_norm(displacement)
        if current_norm > 1e-12:
            displacement = displacement * 1.0 / current_norm

        phonopy_displacements.append(displacement)
        print(f"Mode {mode_idx}: norm = {modes.mass_weighted_norm(displacement):.6f}")

    # Check phonopy orthogonality
    projection_phonopy = modes.mass_weighted_projection(
        phonopy_displacements[0], phonopy_displacements[1]
    )
    print(f"Phonopy orthogonality: projection = {projection_phonopy}")

    print("\n=== DIRECT EIGENVECTOR METHOD ===")
    # Now try a simple comparison of the eigenvectors themselves
    direct_displacements = []

    # Get eigenvectors for this q-point
    frequencies = modes.frequencies[non_gamma_index]
    eigenvecs = modes.eigenvectors[non_gamma_index]

    print(f"Frequencies: {frequencies[mode_indices]}")

    for mode_idx in mode_indices:
        # Get the unit cell eigen displacement
        unit_cell_displacement = modes.get_eigen_displacement(
            non_gamma_index, mode_idx, normalize=True
        )

        # Now manually create the supercell displacement using the manual method
        # This is what should happen for non-phonopy case
        qpoint = modes.qpoints[non_gamma_index]

        # Create supercell
        supercell = make_supercell(modes.primitive_cell, supercell_matrix)
        n_supercell_atoms = len(supercell)

        # Create displacement array
        displacement = np.zeros((n_supercell_atoms, 3), dtype=complex)

        # Get supercell to primitive mapping
        primitive_positions = modes.primitive_cell.get_positions()
        supercell_positions = supercell.get_positions()
        n_primitive_atoms = len(modes.primitive_cell)

        # For each supercell atom, find corresponding primitive cell
        for sc_atom_idx in range(n_supercell_atoms):
            primitive_idx = sc_atom_idx % n_primitive_atoms

            # Calculate lattice vector for this supercell site
            sc_cell_idx = sc_atom_idx // n_primitive_atoms
            lattice_vector = np.array(
                [
                    (sc_cell_idx % supercell_matrix[0, 0]),
                    ((sc_cell_idx // supercell_matrix[0, 0]) % supercell_matrix[1, 1]),
                    (sc_cell_idx // (supercell_matrix[0, 0] * supercell_matrix[1, 1])),
                ]
            )

            # Phase factor
            phase = np.exp(2.0j * np.pi * np.dot(qpoint, lattice_vector))

            # Apply displacement with phase
            displacement[sc_atom_idx] = unit_cell_displacement[primitive_idx] * phase

        # Convert to real and normalize
        displacement = np.real(displacement)
        current_norm = modes.mass_weighted_norm(displacement)
        if current_norm > 1e-12:
            displacement = displacement * 1.0 / current_norm

        direct_displacements.append(displacement)
        print(f"Mode {mode_idx}: norm = {modes.mass_weighted_norm(displacement):.6f}")

    # Check direct method orthogonality
    projection_direct = modes.mass_weighted_projection(
        direct_displacements[0], direct_displacements[1]
    )
    print(f"Direct orthogonality: projection = {projection_direct}")

    print("\n=== COMPARISON ===")
    print(f"Phonopy projection: {projection_phonopy}")
    print(f"Direct projection: {projection_direct}")
    print(f"Difference: {abs(projection_phonopy - projection_direct)}")

    # Compare displacement patterns
    print("\n=== DISPLACEMENT COMPARISON ===")
    for mode_idx in range(2):
        diff = phonopy_displacements[mode_idx] - direct_displacements[mode_idx]
        max_diff = np.max(np.abs(diff))
        rms_diff = np.sqrt(np.mean(diff**2))
        print(f"Mode {mode_idx}: max_diff = {max_diff:.6f}, rms_diff = {rms_diff:.6f}")

        # Check if they're just scaled versions of each other
        phonopy_norm = np.linalg.norm(phonopy_displacements[mode_idx])
        direct_norm = np.linalg.norm(direct_displacements[mode_idx])
        print(
            f"Mode {mode_idx}: phonopy_norm = {phonopy_norm:.6f}, direct_norm = {direct_norm:.6f}"
        )

        if phonopy_norm > 1e-12 and direct_norm > 1e-12:
            correlation = np.dot(
                phonopy_displacements[mode_idx].flatten(),
                direct_displacements[mode_idx].flatten(),
            ) / (phonopy_norm * direct_norm)
            print(f"Mode {mode_idx}: correlation = {correlation:.6f}")


if __name__ == "__main__":
    compare_displacement_methods()
