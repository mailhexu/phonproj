"""
Test that time-reversal symmetry matching works correctly.

This script tests that q-points like 0.5625 can be matched to their
time-reversal partners like 0.4375.

Run:
    uv run python agent_files/debug/test_time_reversal_matching.py
"""

import numpy as np
import sys

sys.path.insert(0, ".")

from phonproj.modes import PhononModes
from ase import Atoms


def test_time_reversal_matching():
    """Test that time-reversal matching works in get_commensurate_qpoints."""

    # Create a simple primitive cell
    primitive_cell = Atoms("H", positions=[[0, 0, 0]], cell=[5.0, 5.0, 5.0], pbc=True)

    # Create q-points only in [0, 0.5] range (simulating our new behavior)
    qpoints_unique = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0625, 0.0, 0.0],
            [0.125, 0.0, 0.0],
            [0.1875, 0.0, 0.0],
            [0.25, 0.0, 0.0],
            [0.3125, 0.0, 0.0],
            [0.375, 0.0, 0.0],
            [0.4375, 0.0, 0.0],
            [0.5, 0.0, 0.0],
        ]
    )

    # Create dummy frequencies and eigenvectors
    n_modes = 3  # 3 atoms * 3 = 3 modes for 1 atom
    frequencies = np.ones((len(qpoints_unique), n_modes))
    eigenvectors = np.ones((len(qpoints_unique), n_modes, n_modes), dtype=complex)

    # Create PhononModes object
    phonon_modes = PhononModes(
        primitive_cell=primitive_cell,
        qpoints=qpoints_unique,
        frequencies=frequencies,
        eigenvectors=eigenvectors,
        atomic_masses=None,
        gauge="R",
    )

    # Test with 16x1x1 supercell
    supercell_matrix = np.diag([16, 1, 1])

    print("Testing time-reversal symmetry matching...")
    print(f"Available q-points (only [0, 0.5] range): {len(qpoints_unique)}")
    print(f"Supercell: 16x1x1 (expects 16 commensurate q-points)")

    result = phonon_modes.get_commensurate_qpoints(supercell_matrix, detailed=True)

    matched_indices = result["matched_indices"]
    missing_qpoints = result["missing_qpoints"]
    all_qpoints = result["all_qpoints"]

    print(f"\nExpected commensurate q-points: {len(all_qpoints)}")
    print(f"Matched q-points: {len(matched_indices)}")
    print(f"Unique matched q-points: {len(set(matched_indices))}")
    print(f"Missing q-points: {len(missing_qpoints)}")

    # Check for duplicates
    if len(matched_indices) != len(set(matched_indices)):
        print(f"\n❌ ERROR: Found duplicate indices in matched_indices!")
        print(f"   Indices: {matched_indices}")
        from collections import Counter

        counts = Counter(matched_indices)
        for idx, count in counts.items():
            if count > 1:
                print(f"   Index {idx} appears {count} times")
        return False

    if missing_qpoints:
        print("\n❌ Missing q-points:")
        for q in missing_qpoints:
            print(f"  {q}")
            # Check if time-reversal partner exists
            q_tr = (1.0 - q) % 1.0
            print(f"    Time-reversal partner would be: {q_tr}")
    else:
        print("\n✓ All q-points matched successfully!")
        print(f"✓ Time-reversal symmetry matching is working!")

    # Show which q-points were matched to what
    print(f"\nMatching details:")
    for i, q_expected in enumerate(all_qpoints):
        if i < len(matched_indices):
            matched_idx = matched_indices[i]
            q_matched = qpoints_unique[matched_idx]
            q_tr = (1.0 - q_expected) % 1.0

            if np.allclose(q_matched, q_expected, atol=1e-6):
                print(f"  {q_expected} → direct match to {q_matched}")
            elif np.allclose(q_matched, q_tr, atol=1e-6):
                print(
                    f"  {q_expected} → time-reversal match to {q_matched} (TR of {q_expected} is {q_tr})"
                )
            else:
                print(f"  {q_expected} → unclear match to {q_matched}")

    return len(missing_qpoints) == 0


if __name__ == "__main__":
    success = test_time_reversal_matching()
    sys.exit(0 if success else 1)
