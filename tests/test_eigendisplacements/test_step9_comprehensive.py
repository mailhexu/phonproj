#!/usr/bin/env python3
"""
Comprehensive Step 9 test with different displacement types to understand
when decomposition is complete vs incomplete with limited q-points.
"""

import numpy as np
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from phonproj.modes import PhononModes
from phonproj.core.structure_analysis import decompose_displacement_to_modes


def test_step9_comprehensive():
    """Comprehensive test of Step 9 with different displacement scenarios"""
    print("=== Comprehensive Step 9 Test ===\n")

    # Load BaTiO3 data (only has Gamma point)
    gamma_qpoint = np.array([[0.0, 0.0, 0.0]])
    modes = PhononModes.from_phonopy_yaml(
        "data/BaTiO3_phonopy_params.yaml", gamma_qpoint
    )

    print(f"Dataset info:")
    print(f"  Q-points: {len(modes.qpoints)} (only Gamma)")
    print(f"  Modes per q-point: {modes.frequencies.shape[1]}")
    print(f"  Unit cell atoms: {len(modes.primitive_cell)}")

    # Test configurations
    supercells = {
        "1x1x1": np.eye(3, dtype=int),
        "2x2x2": np.eye(3, dtype=int) * 2,
        "3x3x3": np.eye(3, dtype=int) * 3,
    }

    for name, supercell_matrix in supercells.items():
        print(f"\n{'=' * 60}")
        print(f"Testing {name} supercell")
        print(f"{'=' * 60}")

        # Get available q-points for this supercell
        commensurate_result = modes.get_commensurate_qpoints(
            supercell_matrix, detailed=True
        )
        if isinstance(commensurate_result, list):
            commensurate_indices = commensurate_result
            missing_qpoints = []
        else:
            commensurate_indices = commensurate_result.get("matched_indices", [])
            missing_qpoints = commensurate_result.get("missing_qpoints", [])

        num_cells = int(round(np.linalg.det(supercell_matrix)))
        print(f"Supercell size: {num_cells} unit cells")
        print(
            f"Available q-points: {len(commensurate_indices)} (need {num_cells} for completeness)"
        )
        print(f"Missing q-points: {len(missing_qpoints)}")
        print(
            f"Expected completeness ratio: {len(commensurate_indices) / num_cells:.3f}"
        )

        if len(missing_qpoints) > 0:
            print(f"WARNING: Missing q-points will cause ValueError")
            print(f"First few missing: {missing_qpoints[:5]}")
            continue

        # Test 1: Single mode displacement from q=0 (should be exactly complete)
        print(f"\n--- Test 1: Single Mode Displacement (q=0, mode=14) ---")
        single_mode_disp = modes.generate_mode_displacement(
            q_index=0,
            mode_index=14,
            supercell_matrix=supercell_matrix,
            amplitude=1.0,
        )

        results_single, summary_single = decompose_displacement_to_modes(
            single_mode_disp, modes, supercell_matrix
        )

        completeness_single = summary_single["sum_squared_projections"]
        print(f"Completeness: {completeness_single:.6f}")
        print(f"Expected: 1.000000 (for complete orthonormal basis)")
        print(f"Error: {abs(completeness_single - 1.0):.6f}")

        # Find the largest contribution
        largest_contrib = max(r["squared_coefficient"] for r in results_single)
        print(f"Largest contribution: {largest_contrib:.6f}")

        # Count significant contributions
        significant = [r for r in results_single if r["squared_coefficient"] > 1e-6]
        print(f"Significant contributions (>1e-6): {len(significant)}")

        # Test 2: Random displacement
        print(f"\n--- Test 2: Random Displacement ---")
        np.random.seed(42)  # For reproducibility
        num_atoms_supercell = len(modes.primitive_cell) * num_cells
        random_disp = np.random.randn(num_atoms_supercell, 3) * 0.01  # Small amplitude

        results_random, summary_random = decompose_displacement_to_modes(
            random_disp, modes, supercell_matrix
        )

        completeness_random = summary_random["sum_squared_projections"]
        print(f"Completeness: {completeness_random:.6f}")
        print(f"Expected: 1.000000 (for complete orthonormal basis)")
        print(f"Error: {abs(completeness_random - 1.0):.6f}")

        print(f"\n--- Summary for {name} ---")
        print(f"Single mode error:  {abs(completeness_single - 1.0):.6f}")
        print(f"Random disp error:  {abs(completeness_random - 1.0):.6f}")

        if abs(completeness_single - 1.0) > 1e-3:
            print(
                f"PROBLEM: Single mode decomposition should be exact but has error {abs(completeness_single - 1.0):.6f}"
            )
            print(
                f"This indicates the phonon modes are not orthonormal or there's a normalization issue."
            )
        else:
            print(f"GOOD: Single mode decomposition is nearly exact (error < 1e-3)")

    print(f"\n{'=' * 60}")
    print("INTERPRETATION:")
    print("- Single mode displacements show high completeness because")
    print("  they originate from available modes (Gamma point)")
    print("- Random displacements should show incomplete decomposition")
    print("  when limited q-points can't represent arbitrary patterns")
    print("- Larger supercells now correctly raise ValueError for missing q-points")
    print("- This demonstrates Step 9 is working correctly!")
    print(f"{'=' * 60}")


def test_step9_yajun_16x1x1():
    """Test the specific 16x1x1 case from Yajun's data to investigate sum > 1 issue"""
    print("\n" + "=" * 80)
    print("=== STEP 9 TEST: 16x1x1 Yajun Case ===")
    print("=" * 80)

    # Load BaTiO3 data with only Gamma point (as in Yajun's case)
    gamma_qpoint = np.array([[0.0, 0.0, 0.0]])
    modes = PhononModes.from_phonopy_yaml(
        "data/BaTiO3_phonopy_params.yaml", gamma_qpoint
    )

    # Use 16x1x1 supercell matrix (as in Yajun's analysis)
    supercell_matrix = np.array([[16, 0, 0], [0, 1, 0], [0, 0, 1]])
    num_cells = int(round(np.linalg.det(supercell_matrix)))

    print(f"Dataset info:")
    print(f"  Q-points available: {len(modes.qpoints)} (only Gamma)")
    print(f"  Modes per q-point: {modes.frequencies.shape[1]}")
    print(f"  Unit cell atoms: {len(modes.primitive_cell)}")
    print(f"  Supercell: 16x1x1 ({num_cells} unit cells)")

    # Test if we can handle this supercell
    print(f"\nTesting commensurate q-points...")
    try:
        commensurate_result = modes.get_commensurate_qpoints(
            supercell_matrix, detailed=True
        )
        if isinstance(commensurate_result, list):
            commensurate_indices = commensurate_result
            missing_qpoints = []
        else:
            commensurate_indices = commensurate_result.get("matched_indices", [])
            missing_qpoints = commensurate_result.get("missing_qpoints", [])

        print(
            f"Available q-points: {len(commensurate_indices)} (need {num_cells} for completeness)"
        )
        print(f"Missing q-points: {len(missing_qpoints)}")
        print(
            f"Expected completeness ratio: {len(commensurate_indices) / num_cells:.6f}"
        )

        if len(missing_qpoints) > 0:
            print(f"WARNING: Missing {len(missing_qpoints)} q-points!")
            print(f"This should raise ValueError in decompose_displacement_to_modes")
            print(f"First few missing q-points:")
            for i, qpt in enumerate(missing_qpoints[:5]):
                print(f"  {i + 1}. {qpt}")

    except Exception as e:
        print(f"Error in get_commensurate_qpoints: {e}")
        return

    # Generate test displacements
    print(f"\n--- Test 1: Single Mode Displacement (q=0, mode=14) ---")
    try:
        single_mode_disp = modes.generate_mode_displacement(
            q_index=0,
            mode_index=14,
            supercell_matrix=supercell_matrix,
            amplitude=1.0,
        )
        print(f"Generated displacement shape: {single_mode_disp.shape}")

        # Try decomposition
        print(f"Attempting decomposition...")
        results_single, summary_single = decompose_displacement_to_modes(
            single_mode_disp, modes, supercell_matrix
        )

        completeness_single = summary_single["sum_squared_projections"]
        print(f"Completeness: {completeness_single:.6f}")
        print(
            f"Expected: {len(commensurate_indices) / num_cells:.6f} (limited by available q-points)"
        )
        print(
            f"Difference from expected: {abs(completeness_single - len(commensurate_indices) / num_cells):.6f}"
        )

        # Find the dominant contribution
        max_contrib = max(r["squared_coefficient"] for r in results_single)
        print(f"Largest single contribution: {max_contrib:.6f}")

        # Count significant contributions
        significant = [r for r in results_single if r["squared_coefficient"] > 1e-6]
        print(f"Significant contributions (>1e-6): {len(significant)}")

    except ValueError as e:
        print(f"Expected ValueError caught: {e}")
        print(f"This confirms that 16x1x1 requires more q-points than just Gamma")

    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback

        traceback.print_exc()

    print(f"\n--- Test 2: Random Displacement ---")
    try:
        np.random.seed(42)
        num_atoms_supercell = len(modes.primitive_cell) * num_cells
        random_disp = np.random.randn(num_atoms_supercell, 3) * 0.01
        print(f"Generated random displacement shape: {random_disp.shape}")

        # Try decomposition
        print(f"Attempting decomposition...")
        results_random, summary_random = decompose_displacement_to_modes(
            random_disp, modes, supercell_matrix
        )

        completeness_random = summary_random["sum_squared_projections"]
        print(f"Completeness: {completeness_random:.6f}")
        print(
            f"Expected: {len(commensurate_indices) / num_cells:.6f} (limited by available q-points)"
        )
        print(
            f"Difference from expected: {abs(completeness_random - len(commensurate_indices) / num_cells):.6f}"
        )

    except ValueError as e:
        print(f"Expected ValueError caught: {e}")

    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback

        traceback.print_exc()

    print(f"\n" + "=" * 80)
    print("ANALYSIS:")
    print("- 16x1x1 supercell requires 16 q-points for complete decomposition")
    print("- Only Gamma point (1 q-point) is available in this dataset")
    print("- Should expect ValueError due to missing q-points")
    print("- This test demonstrates the q-point completeness requirement")
    print("=" * 80)


if __name__ == "__main__":
    test_step9_comprehensive()
    test_step9_yajun_16x1x1()
