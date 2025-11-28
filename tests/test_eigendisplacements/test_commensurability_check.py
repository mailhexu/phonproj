#!/usr/bin/env python3
"""
Test the commensurability check for q-points and supercells.
"""

import numpy as np
import sys
import pytest

sys.path.insert(0, ".")

from phonproj.modes import PhononModes


def test_commensurability_check():
    """Test that non-commensurate q-points raise appropriate exceptions."""

    print("=== Q-point Commensurability Check Test ===")

    # Load test data
    BATIO3_YAML_PATH = "./data/BaTiO3_phonopy_params.yaml"

    # Test with a commensurate q-point first (should work)
    print("\n1. Testing commensurate q-point [0.5, 0.5, 0.0] with 2x2x1 supercell...")
    qpoints_good = np.array([[0.5, 0.5, 0.0]])
    supercell_matrix = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]])

    try:
        modes = PhononModes.from_phonopy_yaml(BATIO3_YAML_PATH, qpoints=qpoints_good)
        supercell_disp = modes.generate_mode_displacement(
            0, 0, supercell_matrix, amplitude=1.0
        )
        print(
            f"   ✓ SUCCESS: Commensurate q-point worked correctly (displacement shape: {supercell_disp.shape})"
        )
        assert (
            supercell_disp.shape[1] == 3
        ), f"Expected 3D displacement, got {supercell_disp.shape}"
    except Exception as e:
        print(f"   ❌ UNEXPECTED ERROR: Commensurate q-point failed: {e}")
        raise AssertionError(f"Commensurate q-point failed unexpectedly: {e}")

    # Test with a non-commensurate q-point (should raise ValueError)
    print(
        "\n2. Testing non-commensurate q-point [0.25, 0.25, 0.25] with 2x2x1 supercell..."
    )
    qpoints_bad = np.array([[0.25, 0.25, 0.25]])

    with pytest.raises(ValueError, match="not commensurate"):
        modes = PhononModes.from_phonopy_yaml(BATIO3_YAML_PATH, qpoints=qpoints_bad)
        supercell_disp = modes.generate_mode_displacement(
            0, 0, supercell_matrix, amplitude=1.0
        )
    print(f"   ✓ SUCCESS: Correctly caught ValueError for non-commensurate q-point")

    # Test with another non-commensurate case
    print(
        "\n3. Testing non-commensurate q-point [0.3, 0.0, 0.0] with 2x2x1 supercell..."
    )
    qpoints_bad2 = np.array([[0.3, 0.0, 0.0]])

    with pytest.raises(ValueError, match="not commensurate"):
        modes = PhononModes.from_phonopy_yaml(BATIO3_YAML_PATH, qpoints=qpoints_bad2)
        supercell_disp = modes.generate_mode_displacement(
            0, 0, supercell_matrix, amplitude=1.0
        )
    print(f"   ✓ SUCCESS: Correctly caught ValueError for non-commensurate q-point")

    # Test edge case: q-point that becomes commensurate with numerical tolerance
    print(
        "\n4. Testing edge case q-point [0.5000001, 0.5, 0.0] (should be treated as commensurate)..."
    )
    qpoints_edge = np.array([[0.5000001, 0.5, 0.0]])

    try:
        modes = PhononModes.from_phonopy_yaml(BATIO3_YAML_PATH, qpoints=qpoints_edge)
        supercell_disp = modes.generate_mode_displacement(
            0, 0, supercell_matrix, amplitude=1.0
        )
        print(
            f"   ✓ SUCCESS: Edge case q-point treated as commensurate (displacement shape: {supercell_disp.shape})"
        )
        assert (
            supercell_disp.shape[1] == 3
        ), f"Expected 3D displacement, got {supercell_disp.shape}"
    except Exception as e:
        print(f"   ❌ UNEXPECTED ERROR: Edge case q-point failed: {e}")
        raise AssertionError(f"Edge case q-point failed unexpectedly: {e}")

    print("\n=== Test Summary ===")
    print("✅ All commensurability tests passed!")
    print("✅ Commensurate q-points work correctly")
    print("✅ Non-commensurate q-points raise clear ValueError exceptions")
    print("✅ Numerical tolerance handling works correctly")


if __name__ == "__main__":
    success = test_commensurability_check()
    if success:
        print("\n🎯 COMMENSURABILITY CHECK: SUCCESS")
    else:
        print("\n🎯 COMMENSURABILITY CHECK: FAILURE")

    sys.exit(0 if success else 1)
