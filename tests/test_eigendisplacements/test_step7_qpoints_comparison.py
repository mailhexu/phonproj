#!/usr/bin/env python3
"""
Test to compare manual q-point generation with Step 7's get_commensurate_qpoints() method.
This will help us understand if the overcomplete decomposition is due to incorrect q-point selection.
"""

import numpy as np
from phonproj.modes import PhononModes


def test_qpoint_methods():
    """Compare manual q-point generation with Step 7 method"""

    print("=== Testing Q-Point Methods Comparison ===\n")

    # Load modes
    modes = PhononModes.from_phonopy_yaml(
        "data/BaTiO3_phonopy_params.yaml",
        np.array([[0.0, 0.0, 0.0]]),  # Start with just Gamma
    )
    print(f"Loaded {len(modes.qpoints)} total q-points")
    print(f"Q-points in dataset:")
    for i, q in enumerate(modes.qpoints):
        print(f"  {i:2d}: [{q[0]:6.3f}, {q[1]:6.3f}, {q[2]:6.3f}]")

    # Test 1x1x1 supercell
    print(f"\n--- 1x1x1 Supercell ---")
    supercell_1x1x1 = np.eye(3)

    # Manual method (what we used before)
    manual_qpoints_1x1x1 = np.array([[0, 0, 0]])
    print(f"Manual q-points: {manual_qpoints_1x1x1}")

    # Step 7 method
    step7_indices_1x1x1 = modes.get_commensurate_qpoints(supercell_1x1x1)
    step7_qpoints_1x1x1 = [modes.qpoints[i] for i in step7_indices_1x1x1]
    print(f"Step 7 indices: {step7_indices_1x1x1}")
    print(f"Step 7 q-points: {step7_qpoints_1x1x1}")

    # Test 2x2x2 supercell
    print(f"\n--- 2x2x2 Supercell ---")
    supercell_2x2x2 = np.eye(3) * 2

    # Manual method (what we used before)
    manual_qpoints_2x2x2 = []
    for i in [0, 0.5]:
        for j in [0, 0.5]:
            for k in [0, 0.5]:
                manual_qpoints_2x2x2.append([i, j, k])
    manual_qpoints_2x2x2 = np.array(manual_qpoints_2x2x2)
    print(f"Manual q-points ({len(manual_qpoints_2x2x2)}):")
    for i, q in enumerate(manual_qpoints_2x2x2):
        print(f"  {i}: [{q[0]:3.1f}, {q[1]:3.1f}, {q[2]:3.1f}]")

    # Step 7 method
    step7_indices_2x2x2 = modes.get_commensurate_qpoints(supercell_2x2x2)
    step7_qpoints_2x2x2 = [modes.qpoints[i] for i in step7_indices_2x2x2]
    print(f"Step 7 indices: {step7_indices_2x2x2}")
    print(f"Step 7 q-points ({len(step7_qpoints_2x2x2)}):")
    for i, q in enumerate(step7_qpoints_2x2x2):
        print(f"  {i}: [{q[0]:6.3f}, {q[1]:6.3f}, {q[2]:6.3f}]")

    # Test if the manual points are available in the dataset
    print(f"\n--- Manual Q-Point Availability Check ---")
    for manual_q in manual_qpoints_2x2x2:
        found = False
        for i, dataset_q in enumerate(modes.qpoints):
            if np.allclose(manual_q, dataset_q, atol=1e-6):
                print(f"✅ Manual q-point {manual_q} found at index {i}")
                found = True
                break
        if not found:
            print(f"❌ Manual q-point {manual_q} NOT found in dataset")

    # Compare the sets
    print(f"\n--- Comparison ---")
    print(
        f"1x1x1: Manual={len(manual_qpoints_1x1x1)}, Step7={len(step7_indices_1x1x1)}"
    )
    print(
        f"2x2x2: Manual={len(manual_qpoints_2x2x2)}, Step7={len(step7_indices_2x2x2)}"
    )

    # Check if Step 7 finds more or fewer q-points
    if len(step7_indices_2x2x2) != len(manual_qpoints_2x2x2):
        print(
            f"⚠️  Different number of q-points! This might explain the overcomplete decomposition."
        )
        if len(step7_indices_2x2x2) > len(manual_qpoints_2x2x2):
            print(f"   Step 7 found MORE q-points than expected")
        else:
            print(f"   Step 7 found FEWER q-points than expected")
    else:
        print(f"✅ Same number of q-points")


if __name__ == "__main__":
    test_qpoint_methods()
