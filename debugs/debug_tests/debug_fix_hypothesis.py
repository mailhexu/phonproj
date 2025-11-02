#!/usr/bin/env python3
"""
Simple debug script to test a hypothesis: Does the phonopy API
always produce non-orthogonal results, or is it our usage?
"""

import numpy as np
from pathlib import Path
from phonproj.modes import PhononModes

# Data path
BATIO3_YAML_PATH = Path(__file__).parent / "data" / "BaTiO3_phonopy_params.yaml"


def test_fix_hypothesis():
    """Test if we can fix the orthogonality by avoiding phonopy API."""

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

    print(f"Using q-point index {non_gamma_index}: {modes.qpoints[non_gamma_index]}")

    # Test the hypothesis: modify generate_all_mode_displacements to NOT use phonopy API
    print("\n=== TESTING MODIFIED generate_all_mode_displacements ===")

    # Let's temporarily override the routing to force use of direct method
    # We'll do this by temporarily setting the supercell to 1x1x1 in the condition
    original_generate = modes.generate_all_mode_displacements

    def modified_generate_all_mode_displacements(
        q_index, supercell_matrix, amplitude=1.0
    ):
        """Modified version that forces direct eigenvector method."""
        all_displacements = []

        n_modes = modes.frequencies.shape[1]

        for mode_index in range(n_modes):
            # ALWAYS use direct method instead of phonopy API
            displacement = original_generate_mode_displacement_direct(
                modes, q_index, mode_index, supercell_matrix, amplitude
            )
            all_displacements.append(displacement)

        return np.array(all_displacements)

    # Generate displacements using direct method only
    all_displacements = modified_generate_all_mode_displacements(
        non_gamma_index, supercell_matrix, amplitude=1.0
    )

    # Check orthogonality
    mode_indices = [0, 1]
    projection = modes.mass_weighted_projection(
        all_displacements[mode_indices[0]], all_displacements[mode_indices[1]]
    )

    print(f"Direct method orthogonality: projection = {projection}")
    print(f"Is orthogonal (< 1e-6): {abs(projection) < 1e-6}")


def original_generate_mode_displacement_direct(
    modes, q_index, mode_index, supercell_matrix, amplitude
):
    """Direct method for generating mode displacements (non-phonopy)."""
    from ase.build import make_supercell

    # Get the unit cell eigen displacement
    unit_cell_displacement = modes.get_eigen_displacement(
        q_index, mode_index, normalize=True
    )

    # Get q-point
    qpoint = modes.qpoints[q_index]

    # Create supercell
    supercell = make_supercell(modes.primitive_cell, supercell_matrix)
    n_supercell_atoms = len(supercell)

    # Create displacement array
    displacement = np.zeros((n_supercell_atoms, 3), dtype=complex)

    # Get primitive cell info
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
        displacement = displacement * amplitude / current_norm

    return displacement


if __name__ == "__main__":
    test_fix_hypothesis()
