#!/usr/bin/env python3
"""
Demonstration of the commensurability check feature.
This shows how the system now prevents invalid q-point/supercell combinations.
"""

import numpy as np
import sys

sys.path.insert(0, "/Users/hexu/projects/phonproj")

from phonproj.modes import PhononModes


def demonstrate_commensurability_check():
    """Demonstrate the new commensurability checking feature."""

    print("=== Commensurability Check Demonstration ===")
    print()

    # Load test data
    BATIO3_YAML_PATH = "/Users/hexu/projects/phonproj/data/BaTiO3_phonopy_params.yaml"

    print("🎯 This demonstrates the new commensurability check that prevents")
    print(
        "   invalid q-point/supercell combinations and provides clear error messages."
    )
    print()

    # Example 1: Valid combination
    print("1️⃣ Valid combination: q=[0.5, 0.5, 0] with 2×2×1 supercell")
    try:
        qpoints = np.array([[0.5, 0.5, 0.0]])
        supercell_matrix = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]])
        modes = PhononModes.from_phonopy_yaml(BATIO3_YAML_PATH, qpoints=qpoints)
        displacement = modes.generate_mode_displacement(0, 0, supercell_matrix)
        print(
            f"   ✅ SUCCESS: Generated displacement pattern (shape: {displacement.shape})"
        )
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
    print()

    # Example 2: Invalid combination with helpful error
    print("2️⃣ Invalid combination: q=[0.25, 0.25, 0.25] with 2×2×1 supercell")
    try:
        qpoints = np.array([[0.25, 0.25, 0.25]])
        supercell_matrix = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]])
        modes = PhononModes.from_phonopy_yaml(BATIO3_YAML_PATH, qpoints=qpoints)
        displacement = modes.generate_mode_displacement(0, 0, supercell_matrix)
        print(f"   ❌ This should have failed!")
    except ValueError as e:
        print("   ✅ CORRECTLY BLOCKED: Got helpful error message:")
        print(f"   📄 {str(e)}")
    print()

    # Example 3: Another invalid combination
    print("3️⃣ Another invalid: q=[0, 0, 0.3] with 2×2×2 supercell")
    try:
        qpoints = np.array([[0.0, 0.0, 0.3]])
        supercell_matrix = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]])
        modes = PhononModes.from_phonopy_yaml(BATIO3_YAML_PATH, qpoints=qpoints)
        displacement = modes.generate_mode_displacement(0, 0, supercell_matrix)
        print(f"   ❌ This should have failed!")
    except ValueError as e:
        print("   ✅ CORRECTLY BLOCKED: Got helpful error message:")
        print(f"   📄 {str(e)}")
    print()

    # Example 4: Show how to fix it
    print("4️⃣ Fixed version: q=[0, 0, 0.3] with 1×1×10 supercell")
    try:
        qpoints = np.array([[0.0, 0.0, 0.3]])
        supercell_matrix = np.array(
            [[1, 0, 0], [0, 1, 0], [0, 0, 10]]
        )  # 10 makes 0.3 commensurate
        modes = PhononModes.from_phonopy_yaml(BATIO3_YAML_PATH, qpoints=qpoints)
        displacement = modes.generate_mode_displacement(0, 0, supercell_matrix)
        print(
            f"   ✅ SUCCESS: Fixed with larger supercell (shape: {displacement.shape})"
        )
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
    print()

    print("🎉 Summary:")
    print("   • The system now automatically checks q-point/supercell compatibility")
    print("   • Invalid combinations are blocked with clear error messages")
    print("   • The error message explains what went wrong and suggests solutions")
    print("   • This prevents confusing results from non-commensurate combinations")


if __name__ == "__main__":
    demonstrate_commensurability_check()
