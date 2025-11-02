#!/usr/bin/env python3
"""
Test the create_supercell function for determinism with the exact same parameters as step10.
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from phonproj.modes import PhononModes, create_supercell


def test_create_supercell_determinism():
    """Test if create_supercell is deterministic."""

    print("Loading Yajun PTO phonon data...")

    # Load data exactly as in step10
    data_dir = project_root / "data" / "yajundata" / "0.02-P4mmm-PTO"

    # Generate all commensurate q-points for (16,1,1) supercell exactly as in step10
    qpoints_16x1x1 = []
    for i in range(16):
        for j in range(1):
            for k in range(1):
                qpoints_16x1x1.append([i / 16.0, j / 1.0, k / 1.0])
    qpoints = np.array(qpoints_16x1x1)

    print(f"Using {len(qpoints)} commensurate q-points for (16,1,1) supercell")

    # Load phonon modes
    phonon_modes = PhononModes.from_phonopy_directory(str(data_dir), qpoints=qpoints)

    print(
        f"Loaded {phonon_modes.n_qpoints} q-points with {phonon_modes.n_modes} modes each"
    )

    # Use exact same supercell matrix as step10
    supercell_matrix = np.array([[16, 0, 0], [0, 1, 0], [0, 0, 1]])

    print(f"Testing supercell matrix: {supercell_matrix.tolist()}")

    # Create multiple supercells and compare them
    print("\nCreating target supercells:")
    target_supercells = []
    for i in range(3):
        target_supercell = create_supercell(
            phonon_modes.primitive_cell, supercell_matrix
        )
        target_supercells.append(target_supercell)
        print(f"  Supercell {i + 1}: {len(target_supercell)} atoms")

    print("\nCreating source supercells (as done in decomposition loop):")
    source_supercells = []
    for i in range(3):
        source_supercell = create_supercell(
            phonon_modes.primitive_cell, supercell_matrix
        )
        source_supercells.append(source_supercell)
        print(f"  Supercell {i + 1}: {len(source_supercell)} atoms")

    # Compare masses between target and source supercells
    print("\nComparing target vs source supercells:")
    for i in range(3):
        target_masses = target_supercells[i].get_masses()
        source_masses = source_supercells[i].get_masses()

        masses_identical = np.allclose(target_masses, source_masses)
        print(f"  Pair {i + 1} masses identical: {masses_identical}")

        if not masses_identical:
            diff = np.abs(target_masses - source_masses)
            print(f"    Max difference: {np.max(diff):.10f}")
            print(f"    Mean difference: {np.mean(diff):.10f}")

            # Check positions too
            target_positions = target_supercells[i].get_positions()
            source_positions = source_supercells[i].get_positions()
            pos_identical = np.allclose(target_positions, source_positions)
            print(f"    Positions identical: {pos_identical}")

            if not pos_identical:
                pos_diff = np.linalg.norm(target_positions - source_positions, axis=1)
                print(f"    Max position difference: {np.max(pos_diff):.10f}")

    # Compare all target supercells among themselves
    print("\nComparing target supercells among themselves:")
    for i in range(2):
        for j in range(i + 1, 3):
            masses_i = target_supercells[i].get_masses()
            masses_j = target_supercells[j].get_masses()

            masses_identical = np.allclose(masses_i, masses_j)
            print(
                f"  Target {i + 1} vs Target {j + 1} masses identical: {masses_identical}"
            )

    # Compare all source supercells among themselves
    print("\nComparing source supercells among themselves:")
    for i in range(2):
        for j in range(i + 1, 3):
            masses_i = source_supercells[i].get_masses()
            masses_j = source_supercells[j].get_masses()

            masses_identical = np.allclose(masses_i, masses_j)
            print(
                f"  Source {i + 1} vs Source {j + 1} masses identical: {masses_identical}"
            )


if __name__ == "__main__":
    test_create_supercell_determinism()
