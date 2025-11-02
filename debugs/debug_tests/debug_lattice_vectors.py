#!/usr/bin/env python3
"""
Debug the lattice vector calculation in our supercell displacement method.
"""

import numpy as np
from pathlib import Path
from phonproj.modes import PhononModes

# Data path
BATIO3_YAML_PATH = Path(__file__).parent / "data" / "BaTiO3_phonopy_params.yaml"


def debug_lattice_vectors():
    """Debug lattice vector calculation and phases."""

    # Set up test case
    qpoints_2x2x2 = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                qpoints_2x2x2.append([i / 2.0, j / 2.0, k / 2.0])
    qpoints_2x2x2 = np.array(qpoints_2x2x2)

    modes = PhononModes.from_phonopy_yaml(str(BATIO3_YAML_PATH), qpoints_2x2x2)

    # Use 2x2x2 supercell
    supercell_matrix = np.eye(3, dtype=int) * 2

    # Find a non-Gamma q-point
    non_gamma_index = None
    for i, qpoint in enumerate(modes.qpoints):
        if not np.allclose(qpoint, [0, 0, 0], atol=1e-6):
            non_gamma_index = i
            break

    qpoint = modes.qpoints[non_gamma_index]
    print(f"Q-point: {qpoint}")

    # Calculate lattice vectors and phases for first 16 atoms in 2x2x2 supercell
    n_primitive_atoms = 5  # BaTiO3 has 5 atoms
    n_supercell_atoms = 8 * 5  # 2x2x2 * 5

    print(f"\n=== LATTICE VECTORS AND PHASES ===")
    print(f"Primitive atoms: {n_primitive_atoms}")
    print(f"Supercell atoms: {n_supercell_atoms}")

    for sc_atom_idx in range(min(25, n_supercell_atoms)):
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

        # Phase factor for supercell translation
        phase = np.exp(2.0j * np.pi * np.dot(qpoint, lattice_vector))

        print(
            f"Atom {sc_atom_idx:2d}: primitive_idx={primitive_idx}, "
            f"cell_idx={sc_cell_idx}, lattice_vec={lattice_vector}, "
            f"phase={phase:.4f}"
        )

    # Now check what the theoretical phases should be
    print(f"\n=== THEORETICAL PHASES ===")
    print(f"Q-point: {qpoint}")

    # For a 2x2x2 supercell, we have 8 unit cells
    expected_phases = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                lattice_vec = [i, j, k]
                phase = np.exp(2.0j * np.pi * np.dot(qpoint, lattice_vec))
                expected_phases.append(phase)
                print(f"Cell [{i},{j},{k}]: phase = {phase:.4f}")

    # The pattern should repeat for each primitive atom
    print(f"\n=== PHASE PATTERN ===")
    print(f"Expected phase pattern (should repeat for each primitive atom):")
    for i, phase in enumerate(expected_phases):
        print(f"Cell {i}: {phase:.4f}")


if __name__ == "__main__":
    debug_lattice_vectors()
