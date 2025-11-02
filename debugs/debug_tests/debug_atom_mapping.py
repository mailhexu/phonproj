#!/usr/bin/env python3
"""
Test to verify if the atom ordering in supercell is causing the non-orthogonality.
"""

import numpy as np
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from phonproj.modes import PhononModes
from ase.build import make_supercell


def test_supercell_atom_mapping():
    """Test if the supercell atom mapping is correct."""

    # Load BaTiO3 data
    BATIO3_YAML_PATH = Path("data/BaTiO3_phonopy_params.yaml")

    qpoints_2x2x2 = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                qpoints_2x2x2.append([i / 2.0, j / 2.0, k / 2.0])
    qpoints_2x2x2 = np.array(qpoints_2x2x2)

    modes = PhononModes.from_phonopy_yaml(str(BATIO3_YAML_PATH), qpoints_2x2x2)

    print("=== PRIMITIVE CELL ANALYSIS ===")
    print(f"Primitive cell has {modes._n_atoms} atoms")
    print("Atomic symbols:", [s for s in modes.primitive_cell.get_chemical_symbols()])
    print("Atomic numbers:", modes.primitive_cell.get_atomic_numbers())
    print("Atomic masses:", modes.atomic_masses)
    print("Primitive cell positions:")
    prim_positions = modes.primitive_cell.get_scaled_positions()
    for i, pos in enumerate(prim_positions):
        symbol = modes.primitive_cell.get_chemical_symbols()[i]
        print(f"  Atom {i} ({symbol}): {pos}")

    print("\n=== SUPERCELL ANALYSIS ===")
    supercell_matrix = np.eye(3, dtype=int) * 2
    supercell = make_supercell(modes.primitive_cell, supercell_matrix)

    print(f"Supercell has {len(supercell)} atoms")
    print("Supercell atomic symbols:", supercell.get_chemical_symbols())
    print("Supercell atomic numbers:", supercell.get_atomic_numbers())

    # Check the mapping assumption: i % n_atoms
    print("\n=== ATOM MAPPING VERIFICATION ===")
    prim_symbols = modes.primitive_cell.get_chemical_symbols()
    super_symbols = supercell.get_chemical_symbols()

    mapping_correct = True
    for i in range(len(supercell)):
        expected_prim_index = i % modes._n_atoms
        expected_symbol = prim_symbols[expected_prim_index]
        actual_symbol = super_symbols[i]

        if expected_symbol != actual_symbol:
            print(
                f"❌ Atom {i}: expected {expected_symbol} (from prim {expected_prim_index}), got {actual_symbol}"
            )
            mapping_correct = False
        elif i < 20:  # Show first 20 for verification
            print(f"✓ Atom {i}: {actual_symbol} (from prim {expected_prim_index})")

    if mapping_correct:
        print("✓ Atom mapping is CORRECT: i % n_atoms works")
    else:
        print("❌ Atom mapping is WRONG: i % n_atoms fails")

        # Try to find the correct mapping
        print("\n=== FINDING CORRECT MAPPING ===")
        super_positions = supercell.get_scaled_positions()
        prim_positions = modes.primitive_cell.get_scaled_positions()

        # For each supercell atom, find which primitive atom it corresponds to
        correct_mapping = []
        for i, super_pos in enumerate(super_positions):
            super_symbol = super_symbols[i]

            # Find primitive atoms with same symbol
            candidates = []
            for j, prim_symbol in enumerate(prim_symbols):
                if prim_symbol == super_symbol:
                    candidates.append(j)

            # Among candidates, find the one that matches position modulo lattice
            best_match = None
            min_distance = float("inf")

            for prim_idx in candidates:
                prim_pos = prim_positions[prim_idx]
                # Check all possible lattice translations
                for dx in range(2):
                    for dy in range(2):
                        for dz in range(2):
                            translated_pos = prim_pos + np.array(
                                [dx / 2, dy / 2, dz / 2]
                            )
                            # Bring to unit cell
                            translated_pos = translated_pos % 1.0
                            distance = np.linalg.norm(super_pos - translated_pos)
                            if distance < min_distance:
                                min_distance = distance
                                best_match = prim_idx

            correct_mapping.append(best_match)
            if i < 10:
                naive_mapping = i % modes._n_atoms
                print(
                    f"Atom {i}: naive={naive_mapping}, correct={best_match}, distance={min_distance:.6f}"
                )

        print(f"Correct mapping for first 20 atoms: {correct_mapping[:20]}")
        print(
            f"Naive mapping for first 20 atoms: {[i % modes._n_atoms for i in range(20)]}"
        )

    return mapping_correct, supercell, super_symbols


if __name__ == "__main__":
    mapping_correct, supercell, symbols = test_supercell_atom_mapping()
