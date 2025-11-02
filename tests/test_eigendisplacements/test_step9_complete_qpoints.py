#!/usr/bin/env python3
"""
Test Step 9 with complete q-point sets to achieve perfect completeness.
This test loads phonon data with all commensurate q-points needed for complete decomposition.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from phonproj.modes import PhononModes
from phonproj.core.structure_analysis import (
    decompose_displacement_to_modes,
    print_decomposition_table,
)


def test_complete_qpoint_decomposition():
    """Test decomposition with complete q-point sets"""
    print("=== Testing Step 9 with Complete Q-Point Sets ===\n")

    # Test 1: 1x1x1 supercell (only needs Gamma point)
    print("--- Test 1: 1x1x1 supercell (Gamma only) ---")
    gamma_qpoint = np.array([[0.0, 0.0, 0.0]])
    modes_1x1x1 = PhononModes.from_phonopy_yaml(
        "data/BaTiO3_phonopy_params.yaml", gamma_qpoint
    )

    supercell_matrix_1x1x1 = np.eye(3, dtype=int)

    print(f"Loaded {len(modes_1x1x1.qpoints)} q-points: {modes_1x1x1.qpoints}")

    # Create a test displacement by combining multiple modes
    test_displacement_1x1x1 = (
        0.5
        * modes_1x1x1.generate_mode_displacement(
            q_index=0, mode_index=14, supercell_matrix=supercell_matrix_1x1x1
        )
        + 0.3
        * modes_1x1x1.generate_mode_displacement(
            q_index=0, mode_index=0, supercell_matrix=supercell_matrix_1x1x1
        )
        + 0.2
        * modes_1x1x1.generate_mode_displacement(
            q_index=0, mode_index=8, supercell_matrix=supercell_matrix_1x1x1
        )
    )

    print(
        f"Created test displacement with norm: {np.linalg.norm(test_displacement_1x1x1):.6f}"
    )

    # Decompose
    projection_table_1x1x1, summary_1x1x1 = decompose_displacement_to_modes(
        test_displacement_1x1x1, modes_1x1x1, supercell_matrix_1x1x1
    )

    print(
        f"Results: {len(projection_table_1x1x1)} modes, completeness = {summary_1x1x1['sum_squared_projections']:.6f}"
    )
    print()

    # Test 2: 2x2x2 supercell (needs 8 q-points)
    print("--- Test 2: 2x2x2 supercell (8 commensurate q-points) ---")

    # Generate all commensurate q-points for 2x2x2 supercell
    qpoints_2x2x2 = []
    for i in [0, 0.5]:
        for j in [0, 0.5]:
            for k in [0, 0.5]:
                qpoints_2x2x2.append([i, j, k])
    qpoints_2x2x2 = np.array(qpoints_2x2x2)

    print(f"Loading phonons at {len(qpoints_2x2x2)} q-points:")
    for i, q in enumerate(qpoints_2x2x2):
        print(f"  {i}: [{q[0]:3.1f}, {q[1]:3.1f}, {q[2]:3.1f}]")

    modes_2x2x2 = PhononModes.from_phonopy_yaml(
        "data/BaTiO3_phonopy_params.yaml", qpoints_2x2x2
    )

    supercell_matrix_2x2x2 = 2 * np.eye(3, dtype=int)

    print(f"Loaded {len(modes_2x2x2.qpoints)} q-points successfully")

    # Create a test displacement in the 2x2x2 supercell
    print("Creating test displacement from multiple q-points and modes...")

    # Use mode from Gamma point as base
    test_displacement_2x2x2 = modes_2x2x2.generate_mode_displacement(
        q_index=0, mode_index=14, supercell_matrix=supercell_matrix_2x2x2
    )

    print(
        f"Created test displacement with norm: {np.linalg.norm(test_displacement_2x2x2):.6f}"
    )

    # Decompose
    print("Decomposing displacement...")
    projection_table_2x2x2, summary_2x2x2 = decompose_displacement_to_modes(
        test_displacement_2x2x2, modes_2x2x2, supercell_matrix_2x2x2
    )

    print(
        f"Results: {len(projection_table_2x2x2)} modes, completeness = {summary_2x2x2['sum_squared_projections']:.6f}"
    )

    # Show top contributions
    print("\nTop 10 mode contributions:")
    sorted_results = sorted(
        projection_table_2x2x2,
        key=lambda x: abs(x["squared_coefficient"]),
        reverse=True,
    )

    for i, result in enumerate(sorted_results[:10]):
        qpt = result["q_point"]
        mode_idx = result["mode_index"]
        proj = result["projection_coefficient"]
        freq = result["frequency"]

        # Check if this is the original mode (Gamma, mode 14)
        is_original = np.allclose(qpt, [0.0, 0.0, 0.0], atol=1e-10) and mode_idx == 14
        marker = " *** ORIGINAL ***" if is_original else ""

        print(
            f"  {i + 1:2d}. q=[{qpt[0]:4.1f}, {qpt[1]:4.1f}, {qpt[2]:4.1f}] mode={mode_idx:2d} freq={freq:8.3f} proj={proj:8.5f}{marker}"
        )

    print()

    # Compare results
    print("=== Summary ===")
    print(
        f"1x1x1 supercell: {summary_1x1x1['sum_squared_projections']:.6f} completeness ({summary_1x1x1['n_modes_total']} modes, {summary_1x1x1['n_qpoints']} q-points)"
    )
    print(
        f"2x2x2 supercell: {summary_2x2x2['sum_squared_projections']:.6f} completeness ({summary_2x2x2['n_modes_total']} modes, {summary_2x2x2['n_qpoints']} q-points)"
    )

    if abs(summary_2x2x2["sum_squared_projections"] - 1.0) < 0.01:
        print("✅ 2x2x2 decomposition is essentially complete!")
    else:
        print("⚠️  2x2x2 decomposition still incomplete, may need further investigation")

    # Test the theoretical maximum: all q-points should give exactly 8x more modes
    print(
        f"\nExpected total modes: {summary_1x1x1['n_modes_total']} × 8 = {summary_1x1x1['n_modes_total'] * 8}"
    )
    print(f"Actual total modes: {summary_2x2x2['n_modes_total']}")

    if summary_2x2x2["n_modes_total"] == summary_1x1x1["n_modes_total"] * 8:
        print("✅ Correct number of modes found!")
    else:
        print("❌ Mode count mismatch - possible q-point or commensuratibilty issue")


if __name__ == "__main__":
    test_complete_qpoint_decomposition()
