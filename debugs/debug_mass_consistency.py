#!/usr/bin/env python3
"""
Debug mass consistency issue in decompose_displacement_to_modes.

The issue is:
1. Target displacement normalization uses target_supercell masses (line 358)
2. Mode displacement normalization uses source_supercell masses (line 434)
3. Projection uses target_supercell masses (line 251)

But target_supercell and source_supercell are created separately and might have different
atom orderings/masses even though they represent the same structure!
"""

import numpy as np
from phonproj.modes import PhononModes, create_supercell


def debug_mass_consistency():
    """Debug mass consistency between target and source supercells."""

    # Load Yajun data (copy from step10 example)
    print("Loading Yajun PTO phonon data...")

    # Generate qpoints for a 2x2x2 supercell (simpler test)
    qpoints_2x2x2 = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                qpoints_2x2x2.append([i / 2.0, j / 2.0, k / 2.0])
    qpoints = np.array(qpoints_2x2x2)

    print(f"Using {len(qpoints)} commensurate q-points for (2,2,2) supercell")

    # Load phonon modes
    phonon_modes = PhononModes.from_phonopy_directory(
        "/Users/hexu/projects/phonproj/data/yajundata/0.02-P4mmm-PTO", qpoints=qpoints
    )

    # Load phonon modes
    phonon_modes = PhononModes.from_phonopy_directory(
        "/Users/hexu/projects/phonproj/data", qpoints=qpoints
    )

    # Use same supercell matrix as in step10 example
    supercell_matrix = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]])

    # Create target supercell (as done in decompose_displacement_to_modes, line 348)
    target_supercell = create_supercell(phonon_modes.primitive_cell, supercell_matrix)
    print(f"Target supercell: {len(target_supercell)} atoms")

    # Create source supercell (as done in decompose_displacement_to_modes, line 412-414)
    source_supercell = create_supercell(phonon_modes.primitive_cell, supercell_matrix)
    print(f"Source supercell: {len(source_supercell)} atoms")

    # Check if supercells are identical
    target_masses = target_supercell.get_masses()
    source_masses = source_supercell.get_masses()

    print(f"\nTarget supercell masses shape: {target_masses.shape}")
    print(f"Source supercell masses shape: {source_masses.shape}")
    print(f"Target masses first 10: {target_masses[:10]}")
    print(f"Source masses first 10: {source_masses[:10]}")

    # Check if masses are identical
    masses_identical = np.allclose(target_masses, source_masses)
    print(f"Masses identical: {masses_identical}")

    if not masses_identical:
        print("ERROR: Target and source supercell masses differ!")
        mass_diff = np.abs(target_masses - source_masses)
        print(f"Max mass difference: {np.max(mass_diff)}")
        print(f"Mean mass difference: {np.mean(mass_diff)}")

        # Check if it's just ordering
        target_sorted = np.sort(target_masses)
        source_sorted = np.sort(source_masses)
        ordering_issue = np.allclose(target_sorted, source_sorted)
        print(f"Same masses, different ordering: {ordering_issue}")

    # Check positions as well
    target_positions = target_supercell.get_positions()
    source_positions = source_supercell.get_positions()

    positions_identical = np.allclose(target_positions, source_positions)
    print(f"Positions identical: {positions_identical}")

    if not positions_identical:
        print("ERROR: Target and source supercell positions differ!")
        pos_diff = np.linalg.norm(target_positions - source_positions, axis=1)
        print(f"Max position difference: {np.max(pos_diff)}")
        print(f"Mean position difference: {np.mean(pos_diff)}")

    # Check chemical symbols
    target_symbols = target_supercell.get_chemical_symbols()
    source_symbols = source_supercell.get_chemical_symbols()

    symbols_identical = target_symbols == source_symbols
    print(f"Symbols identical: {np.all(symbols_identical)}")

    if not np.all(symbols_identical):
        print("ERROR: Target and source supercell symbols differ!")
        for i, (t_sym, s_sym) in enumerate(zip(target_symbols, source_symbols)):
            if t_sym != s_sym:
                print(f"Atom {i}: target={t_sym}, source={s_sym}")
                if i > 10:  # Limit output
                    print("... (more differences)")
                    break

    return target_supercell, source_supercell, masses_identical, positions_identical


if __name__ == "__main__":
    debug_mass_consistency()
