#!/usr/bin/env python3
"""
Corrected Step 9 test using only available q-points from the dataset.
This should demonstrate the CORRECT behavior: incomplete decomposition when
the dataset doesn't have enough q-points for a complete basis.
"""

import numpy as np
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from phonproj.modes import PhononModes
from phonproj.core.structure_analysis import decompose_displacement_to_modes


def test_step9_correct_approach():
    """Test Step 9 with correct q-point handling"""
    print("=== Step 9 Test with Correct Q-Point Handling ===\n")

    # Load BaTiO3 data (only has Gamma point)
    gamma_qpoint = np.array([[0.0, 0.0, 0.0]])
    modes = PhononModes.from_phonopy_yaml(
        "data/BaTiO3_phonopy_params.yaml", gamma_qpoint
    )

    print(f"Available q-points in dataset: {len(modes.qpoints)}")
    for i, q in enumerate(modes.qpoints):
        print(f"  {i}: [{q[0]:6.3f}, {q[1]:6.3f}, {q[2]:6.3f}]")

    # Test 1: 1x1x1 supercell (should be complete)
    print(f"\n--- Test 1: 1x1x1 Supercell ---")
    supercell_1x1x1 = np.eye(3, dtype=int)

    commensurate_indices_1x1x1 = modes.get_commensurate_qpoints(supercell_1x1x1)
    print(f"Commensurate q-point indices: {commensurate_indices_1x1x1}")
    print(f"Number of commensurate q-points: {len(commensurate_indices_1x1x1)}")
    print(f"Expected for 1x1x1: 1 q-point → Should be COMPLETE")

    # Create test displacement using available mode
    test_displacement_1x1x1 = modes.generate_mode_displacement(
        q_index=0, mode_index=14, supercell_matrix=supercell_1x1x1, amplitude=1.0
    )

    # Decompose using only available q-points
    results_1x1x1, summary_1x1x1 = decompose_displacement_to_modes(
        test_displacement_1x1x1, modes, supercell_1x1x1
    )

    completeness_1x1x1 = summary_1x1x1["sum_squared_projections"]
    print(f"Completeness: {completeness_1x1x1:.6f}")
    print(f"Total modes used: {len(results_1x1x1)}")

    # Test 2: 2x2x2 supercell (should raise ValueError due to missing q-points)
    print(f"\n--- Test 2: 2x2x2 Supercell ---")
    supercell_2x2x2 = np.eye(3, dtype=int) * 2

    commensurate_indices_2x2x2 = modes.get_commensurate_qpoints(supercell_2x2x2)
    print(f"Commensurate q-point indices: {commensurate_indices_2x2x2}")
    print(f"Number of commensurate q-points: {len(commensurate_indices_2x2x2)}")
    print(
        f"Expected for 2x2x2: 8 q-points, but only {len(commensurate_indices_2x2x2)} available → Should raise ValueError"
    )

    # Create test displacement
    test_displacement_2x2x2 = modes.generate_mode_displacement(
        q_index=0, mode_index=14, supercell_matrix=supercell_2x2x2, amplitude=1.0
    )

    # Decompose should raise ValueError due to missing q-points
    with pytest.raises(ValueError, match="Missing commensurate q-points"):
        decompose_displacement_to_modes(test_displacement_2x2x2, modes, supercell_2x2x2)

    print(f"\n=== Summary ===")
    print(
        f"1x1x1 supercell: {completeness_1x1x1:.3f} completeness ({len(results_1x1x1)} modes)"
    )
    print(f"2x2x2 supercell: ValueError raised (as expected with missing q-points)")

    if abs(completeness_1x1x1 - 1.0) < 0.05:
        print("✅ 1x1x1 decomposition is nearly complete (as expected)")
    else:
        print("⚠️  1x1x1 decomposition is not complete (unexpected)")

    print(
        "✅ 2x2x2 decomposition correctly raises ValueError (as expected with limited q-points)"
    )


if __name__ == "__main__":
    test_step9_correct_approach()
