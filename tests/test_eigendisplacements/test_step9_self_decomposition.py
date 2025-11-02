#!/usr/bin/env python3
"""
Test script to investigate Step 9 completeness issue by decomposing mode displacements back to themselves.
This should give projection coefficients of 1.0 if the algorithm is working correctly.
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


def test_self_decomposition():
    """Test decomposing mode displacements back to themselves"""
    print("=== Testing Step 9 Self-Decomposition ===\n")

    # Load BaTiO3 phonon data
    print("Loading BaTiO3 phonon data...")
    gamma_qpoint = np.array([[0.0, 0.0, 0.0]])
    modes = PhononModes.from_phonopy_yaml(
        "data/BaTiO3_phonopy_params.yaml", gamma_qpoint
    )
    qpoints = modes.qpoints
    print(f"Loaded {len(qpoints)} q-points")
    print(f"Unit cell: {modes.primitive_cell.cell}")
    print()

    # Test with 1x1x1 supercell first
    supercell_matrix = np.eye(3, dtype=int)
    print(f"Testing with supercell matrix:\n{supercell_matrix}")
    print()

    # Get a specific mode displacement to test with
    test_qpoint = [0, 0, 0]  # Gamma point
    test_mode_index = 14  # Highest frequency mode that showed up in previous tests

    print(f"Getting displacement for q-point {test_qpoint}, mode {test_mode_index}")

    # Get the mode displacement
    try:
        test_displacement = modes.generate_mode_displacement(
            q_index=0, mode_index=test_mode_index, supercell_matrix=supercell_matrix
        )
        print(f"Generated test displacement with shape: {test_displacement.shape}")
        print(f"Displacement norm: {np.linalg.norm(test_displacement):.6f}")
        print()

        # Now decompose this displacement back to modes
        print("Decomposing the mode displacement back to all modes...")
        projection_table, summary = decompose_displacement_to_modes(
            test_displacement, modes, supercell_matrix
        )

        print(f"Decomposition found {len(projection_table)} mode projections")
        print()

        # Check if we find the original mode with coefficient ~1.0
        found_original = False
        original_projection = 0.0

        print("Top 10 mode projections:")
        sorted_results = sorted(
            projection_table, key=lambda x: abs(x["squared_coefficient"]), reverse=True
        )

        for i, result in enumerate(sorted_results[:10]):
            qpt = result["q_point"]
            mode_idx = result["mode_index"]
            proj = result["projection_coefficient"]
            freq = result["frequency"]

            # Check if this is our original mode
            is_original = (
                np.allclose(qpt, test_qpoint, atol=1e-10)
                and mode_idx == test_mode_index
            )

            marker = " *** ORIGINAL ***" if is_original else ""
            print(
                f"  {i + 1:2d}. q={qpt} mode={mode_idx:2d} freq={freq:8.3f} proj={proj:8.5f}{marker}"
            )

            if is_original:
                found_original = True
                original_projection = abs(proj)

        print()
        print(f"Sum of squared projections: {summary['sum_squared_projections']:.6f}")

        if found_original:
            print(f"✅ Found original mode with projection: {original_projection:.6f}")
            if abs(original_projection - 1.0) < 0.01:
                print(
                    "✅ Projection is very close to 1.0 - algorithm working correctly!"
                )
            else:
                print(
                    "⚠️  Projection is not close to 1.0 - potential issue with algorithm"
                )
        else:
            print("❌ Original mode not found in decomposition!")

        print()
        print("=== Testing with 2x2x2 supercell ===")

        # Test with 2x2x2 supercell
        supercell_matrix_222 = 2 * np.eye(3, dtype=int)
        print(f"Supercell matrix:\n{supercell_matrix_222}")

        # Get displacement for the larger supercell
        test_displacement_222 = modes.generate_mode_displacement(
            q_index=0, mode_index=test_mode_index, supercell_matrix=supercell_matrix_222
        )
        print(f"Generated 2x2x2 displacement with shape: {test_displacement_222.shape}")
        print(f"Displacement norm: {np.linalg.norm(test_displacement_222):.6f}")
        print()

        # Decompose
        projection_table_222, summary_222 = decompose_displacement_to_modes(
            test_displacement_222, modes, supercell_matrix_222
        )

        print(f"Decomposition found {len(projection_table_222)} mode projections")

        # Find original mode again
        found_original_222 = False
        original_projection_222 = 0.0

        print("Top 5 mode projections for 2x2x2:")
        sorted_results_222 = sorted(
            projection_table_222,
            key=lambda x: abs(x["squared_coefficient"]),
            reverse=True,
        )

        for i, result in enumerate(sorted_results_222[:5]):
            qpt = result["q_point"]
            mode_idx = result["mode_index"]
            proj = result["projection_coefficient"]
            freq = result["frequency"]

            # Check if this is our original mode
            is_original = (
                np.allclose(qpt, test_qpoint, atol=1e-10)
                and mode_idx == test_mode_index
            )

            marker = " *** ORIGINAL ***" if is_original else ""
            print(
                f"  {i + 1}. q={qpt} mode={mode_idx:2d} freq={freq:8.3f} proj={proj:8.5f}{marker}"
            )

            if is_original:
                found_original_222 = True
                original_projection_222 = abs(proj)

        print()
        print(
            f"Sum of squared projections (2x2x2): {summary_222['sum_squared_projections']:.6f}"
        )

        if found_original_222:
            print(
                f"✅ Found original mode in 2x2x2 with projection: {original_projection_222:.6f}"
            )
        else:
            print("❌ Original mode not found in 2x2x2 decomposition!")

    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_self_decomposition()
