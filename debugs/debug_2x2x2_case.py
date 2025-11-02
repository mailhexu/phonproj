#!/usr/bin/env python3
"""
Debug the working 2x2x2 case to understand correct q-point handling
"""

import numpy as np
from pathlib import Path
from phonproj.modes import PhononModes

# Test data path
BATIO3_YAML_PATH = Path(__file__).parent / "data" / "BaTiO3_phonopy_params.yaml"


def debug_2x2x2_case():
    print("=== Analyzing Working 2x2x2 Case ===")

    # Generate q-points exactly like the working test
    qpoints_2x2x2 = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                qpoints_2x2x2.append([i / 2.0, j / 2.0, k / 2.0])
    qpoints_2x2x2 = np.array(qpoints_2x2x2)

    print(f"2x2x2 q-points:")
    for i, qpt in enumerate(qpoints_2x2x2):
        print(f"  q{i}: [{qpt[0]:.3f}, {qpt[1]:.3f}, {qpt[2]:.3f}]")

    modes = PhononModes.from_phonopy_yaml(str(BATIO3_YAML_PATH), qpoints_2x2x2)

    # Use 2x2x2 supercell
    supercell_matrix = np.eye(3, dtype=int) * 2
    N = 8

    print(f"\nSupercell matrix:\n{supercell_matrix}")

    # Check commensurate q-points
    commensurate_qpoints = modes.get_commensurate_qpoints(supercell_matrix)
    print(f"\nCommensurate q-points found: {len(commensurate_qpoints)}")
    for i, qpt_idx in enumerate(commensurate_qpoints):
        qpt = modes.qpoints[qpt_idx]
        print(f"  Index {qpt_idx}: [{qpt[0]:.3f}, {qpt[1]:.3f}, {qpt[2]:.3f}]")

    # Generate displacements
    all_commensurate_displacements = modes.generate_all_commensurate_displacements(
        supercell_matrix, amplitude=1.0
    )

    # Check for zone-folding equivalent pairs
    print(f"\n=== Checking for zone-folding equivalencies ===")

    equivalent_pairs = []

    for i, qpt_idx_i in enumerate(commensurate_qpoints):
        for j, qpt_idx_j in enumerate(commensurate_qpoints):
            if i < j:  # Avoid checking the same pair twice
                qpt_i = modes.qpoints[qpt_idx_i]
                qpt_j = modes.qpoints[qpt_idx_j]

                # Check if they differ by a reciprocal lattice vector
                diff = qpt_j - qpt_i

                # Check if diff is close to integer values (reciprocal lattice vector)
                is_equivalent = np.allclose(diff, np.round(diff), atol=1e-6)

                if is_equivalent:
                    equivalent_pairs.append((qpt_idx_i, qpt_idx_j, qpt_i, qpt_j, diff))
                    print(
                        f"  Zone-folding equivalent: Q{qpt_idx_i} {qpt_i} <-> Q{qpt_idx_j} {qpt_j} (diff = {diff})"
                    )

    print(f"Found {len(equivalent_pairs)} zone-folding equivalent pairs")

    # Check orthogonality between different q-points (excluding equivalent ones)
    print(f"\n=== Checking orthogonality (excluding zone-folded pairs) ===")

    displacement_list = []
    qpoint_labels = []

    for q_index, displacements in all_commensurate_displacements.items():
        for mode_idx in range(displacements.shape[0]):
            displacement_list.append(displacements[mode_idx])
            qpoint_labels.append((q_index, mode_idx))

    supercell_masses = np.tile(modes.atomic_masses, N)
    max_non_equivalent_overlap = 0.0
    max_equivalent_overlap = 0.0

    for i in range(len(displacement_list)):
        for j in range(i + 1, len(displacement_list)):
            q_i, mode_i = qpoint_labels[i]
            q_j, mode_j = qpoint_labels[j]

            if q_i != q_j:  # Different q-points
                # Check if they're zone-folding equivalent
                is_equivalent = any(
                    (qpt_idx_i == q_i and qpt_idx_j == q_j)
                    or (qpt_idx_i == q_j and qpt_idx_j == q_i)
                    for qpt_idx_i, qpt_idx_j, _, _, _ in equivalent_pairs
                )

                projection = modes.mass_weighted_projection(
                    displacement_list[i], displacement_list[j], supercell_masses
                )
                overlap = abs(projection)

                if is_equivalent:
                    max_equivalent_overlap = max(max_equivalent_overlap, overlap)
                else:
                    max_non_equivalent_overlap = max(
                        max_non_equivalent_overlap, overlap
                    )

    print(
        f"Max overlap between non-equivalent q-points: {max_non_equivalent_overlap:.2e}"
    )
    print(f"Max overlap between equivalent q-points: {max_equivalent_overlap:.2e}")

    # The 2x2x2 case works because there are no zone-folding equivalent q-points!
    if len(equivalent_pairs) == 0:
        print("\n*** SUCCESS: No zone-folding equivalent q-points in 2x2x2 case ***")
        print("This is why the 2x2x2 test passes - all q-points are truly independent!")
    else:
        print(f"\n*** WARNING: Found zone-folding equivalent pairs in 2x2x2 case ***")


if __name__ == "__main__":
    debug_2x2x2_case()
