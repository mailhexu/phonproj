#!/usr/bin/env python3
"""
Test generate_full_commensurate_grid and get_commensurate_qpoints detailed output.
"""

import numpy as np
import sys
import os
from typing import cast, Dict, Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from phonproj.modes import PhononModes


def test_generate_full_commensurate_grid():
    """Test generate_full_commensurate_grid method"""
    print("=== Testing generate_full_commensurate_grid ===")

    # Load test data
    gamma_qpoint = np.array([[0.0, 0.0, 0.0]])
    modes = PhononModes.from_phonopy_yaml(
        "data/BaTiO3_phonopy_params.yaml", gamma_qpoint
    )

    # Test 1x1x1 supercell
    supercell_1x1x1 = np.eye(3, dtype=int)
    qpoints_1x1x1 = modes.generate_full_commensurate_grid(supercell_1x1x1)
    expected_1x1x1 = [np.array([0.0, 0.0, 0.0])]
    assert len(qpoints_1x1x1) == 1
    np.testing.assert_array_almost_equal(qpoints_1x1x1[0], expected_1x1x1[0])
    print("✓ 1x1x1 supercell: correct q-points generated")

    # Test 2x2x2 supercell
    supercell_2x2x2 = 2 * np.eye(3, dtype=int)
    qpoints_2x2x2 = modes.generate_full_commensurate_grid(supercell_2x2x2)
    expected_2x2x2 = [
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.5]),
        np.array([0.0, 0.5, 0.0]),
        np.array([0.0, 0.5, 0.5]),
        np.array([0.5, 0.0, 0.0]),
        np.array([0.5, 0.0, 0.5]),
        np.array([0.5, 0.5, 0.0]),
        np.array([0.5, 0.5, 0.5]),
    ]
    assert len(qpoints_2x2x2) == 8
    for i, q in enumerate(qpoints_2x2x2):
        np.testing.assert_array_almost_equal(q, expected_2x2x2[i])
    print("✓ 2x2x2 supercell: correct q-points generated")

    # Test 3x3x3 supercell
    supercell_3x3x3 = 3 * np.eye(3, dtype=int)
    qpoints_3x3x3 = modes.generate_full_commensurate_grid(supercell_3x3x3)
    assert len(qpoints_3x3x3) == 27
    # Check that all q-points are in [0, 1) range
    for q in qpoints_3x3x3:
        assert all(0.0 <= coord < 1.0 for coord in q)
    print("✓ 3x3x3 supercell: correct number of q-points generated")

    # Test non-diagonal supercell (should raise NotImplementedError)
    try:
        supercell_non_diag = np.array([[2, 1, 0], [0, 2, 0], [0, 0, 2]], dtype=int)
        modes.generate_full_commensurate_grid(supercell_non_diag)
        assert False, "Should have raised NotImplementedError"
    except NotImplementedError:
        print("✓ Non-diagonal supercell: correctly raises NotImplementedError")

    print("✅ generate_full_commensurate_grid tests passed\n")


def test_get_commensurate_qpoints_detailed():
    """Test get_commensurate_qpoints with detailed=True"""
    print("=== Testing get_commensurate_qpoints detailed output ===")

    # Load test data with multiple q-points
    qpoints_2x2x2 = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.5],
            [0.0, 0.5, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.0],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
            [0.5, 0.5, 0.5],
        ]
    )
    modes = PhononModes.from_phonopy_yaml(
        "data/BaTiO3_phonopy_params.yaml", qpoints_2x2x2
    )

    # Test 2x2x2 supercell - should find all q-points
    supercell_2x2x2 = 2 * np.eye(3, dtype=int)
    result = modes.get_commensurate_qpoints(supercell_2x2x2, detailed=True)
    assert isinstance(result, dict), f"Expected dict, got {type(result)}"
    result_dict = result  # type: Dict[str, Any]

    matched_indices = result_dict["matched_indices"]
    missing_qpoints = result_dict["missing_qpoints"]
    all_qpoints = result_dict["all_qpoints"]

    assert len(matched_indices) == 8  # All q-points should be matched
    assert len(missing_qpoints) == 0  # No missing q-points
    assert len(all_qpoints) == 8  # All expected q-points

    # Check that matched_indices are in correct order
    assert matched_indices == list(range(8))
    print("✓ 2x2x2 supercell: all q-points matched correctly")

    # Test 1x1x1 supercell - should find only Gamma point
    supercell_1x1x1 = np.eye(3, dtype=int)
    result_1x1x1 = modes.get_commensurate_qpoints(supercell_1x1x1, detailed=True)

    matched_indices_1x1x1 = result_1x1x1["matched_indices"]
    missing_qpoints_1x1x1 = result_1x1x1["missing_qpoints"]
    all_qpoints_1x1x1 = result_1x1x1["all_qpoints"]

    assert len(matched_indices_1x1x1) == 1  # Only Gamma point
    assert len(missing_qpoints_1x1x1) == 0  # No missing q-points
    assert len(all_qpoints_1x1x1) == 1  # Only Gamma expected
    assert matched_indices_1x1x1[0] == 0  # Gamma point is at index 0
    print("✓ 1x1x1 supercell: Gamma point matched correctly")

    # Test 3x3x3 supercell - should have missing q-points
    supercell_3x3x3 = 3 * np.eye(3, dtype=int)
    result_3x3x3 = modes.get_commensurate_qpoints(supercell_3x3x3, detailed=True)

    matched_indices_3x3x3 = result_3x3x3["matched_indices"]
    missing_qpoints_3x3x3 = result_3x3x3["missing_qpoints"]
    all_qpoints_3x3x3 = result_3x3x3["all_qpoints"]

    assert len(all_qpoints_3x3x3) == 27  # 3x3x3 = 27 q-points expected
    # Since we only have 8 q-points loaded, most will be missing
    assert len(missing_qpoints_3x3x3) == 27 - len(matched_indices_3x3x3)
    print(
        f"✓ 3x3x3 supercell: {len(matched_indices_3x3x3)} matched, {len(missing_qpoints_3x3x3)} missing"
    )

    # Test backward compatibility (detailed=False)
    indices_only = modes.get_commensurate_qpoints(supercell_2x2x2, detailed=False)
    assert isinstance(indices_only, list)
    assert indices_only == matched_indices
    print("✓ Backward compatibility: detailed=False returns list of indices")

    print("✅ get_commensurate_qpoints detailed tests passed\n")


def test_get_commensurate_qpoints_error_handling():
    """Test error handling in get_commensurate_qpoints"""
    print("=== Testing get_commensurate_qpoints error handling ===")

    # Load test data
    gamma_qpoint = np.array([[0.0, 0.0, 0.0]])
    modes = PhononModes.from_phonopy_yaml(
        "data/BaTiO3_phonopy_params.yaml", gamma_qpoint
    )

    # Test with supercell that has commensurate q-points (detailed=False)
    supercell_3x3x3 = 3 * np.eye(3, dtype=int)
    indices = modes.get_commensurate_qpoints(supercell_3x3x3, detailed=False)
    assert isinstance(indices, list)
    # Gamma point is always commensurate with any supercell
    assert len(indices) == 1  # Gamma point matches
    print("✓ Correctly finds Gamma point as commensurate for any supercell")

    # Test generate_all_commensurate_displacements error handling
    try:
        modes.generate_all_commensurate_displacements(supercell_3x3x3)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Missing commensurate q-points" in str(e)
        print(
            "✓ generate_all_commensurate_displacements correctly raises ValueError when q-points are missing"
        )

    # Test with supercell that has no commensurate q-points (detailed=True)
    result = modes.get_commensurate_qpoints(supercell_3x3x3, detailed=True)
    assert isinstance(result, dict)
    # Gamma is always commensurate, so should have 1 match
    assert len(result["matched_indices"]) == 1  # type: ignore
    print("✓ Correctly returns dict with Gamma point matched when detailed=True")

    print("✅ Error handling tests passed\n")


if __name__ == "__main__":
    test_generate_full_commensurate_grid()
    test_get_commensurate_qpoints_detailed()
    test_get_commensurate_qpoints_error_handling()
    print("🎉 All tests passed!")
