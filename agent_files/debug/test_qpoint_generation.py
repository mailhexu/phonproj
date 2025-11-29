"""
Test the new commensurate q-point generation with time-reversal symmetry.

This script tests that:
1. Only q-points in [0, 0.5] range are generated
2. The number of q-points is approximately half of the original
3. No redundant k and 1-k pairs exist

Run:
    uv run python agent_files/debug/test_qpoint_generation.py
"""

import numpy as np
import sys

sys.path.insert(0, ".")

from phonproj.cli import generate_commensurate_qpoints


def test_qpoint_generation():
    """Test q-point generation for various supercell sizes."""

    test_cases = [
        (np.diag([2, 2, 2]), "2x2x2"),
        (np.diag([4, 1, 1]), "4x1x1"),
        (np.diag([3, 3, 3]), "3x3x3"),
    ]

    for supercell_matrix, label in test_cases:
        print(f"\n{'=' * 60}")
        print(f"Testing supercell: {label}")
        print(f"{'=' * 60}")

        # Generate q-points
        qpoints = generate_commensurate_qpoints(supercell_matrix)

        # Original would have been n1*n2*n3 points
        n1 = int(supercell_matrix[0, 0])
        n2 = int(supercell_matrix[1, 1])
        n3 = int(supercell_matrix[2, 2])
        original_count = n1 * n2 * n3

        print(f"Original q-points: {original_count}")
        print(f"New unique q-points: {len(qpoints)}")
        print(f"Reduction: {100 * (1 - len(qpoints) / original_count):.1f}%")

        # Check all q-points are in [0, 0.5] range
        print(f"\nGenerated q-points:")
        for i, q in enumerate(qpoints):
            print(f"  {i:2d}. [{q[0]:.4f}, {q[1]:.4f}, {q[2]:.4f}]", end="")
            if any(qi > 0.5 + 1e-10 for qi in q):
                print(" ❌ OUTSIDE [0, 0.5] RANGE")
            else:
                print(" ✓")

        # Check for redundant pairs (k and 1-k)
        print(f"\nChecking for redundant pairs:")
        found_redundant = False
        for i, q1 in enumerate(qpoints):
            # Calculate 1-q (equivalent to -q in periodic BZ)
            q1_inverse = np.array([1 - q1[0], 1 - q1[1], 1 - q1[2]])
            # Wrap to [0, 1)
            q1_inverse = np.mod(q1_inverse, 1.0)

            for j, q2 in enumerate(qpoints):
                if i != j and np.allclose(q1_inverse, q2, atol=1e-6):
                    print(f"  ❌ Found redundant pair: {q1} and {q2}")
                    found_redundant = True

        if not found_redundant:
            print("  ✓ No redundant pairs found")


if __name__ == "__main__":
    test_qpoint_generation()
