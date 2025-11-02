#!/usr/bin/env python3
"""
Debug zone-folding in 3x3x3 supercell case.

For 3x3x3 supercell, q-points are equivalent if they differ by reciprocal lattice vectors.
Two q-points q1 and q2 are equivalent if q1 - q2 = G where G is a reciprocal lattice vector.

For our case: q = [i/3, j/3, k/3], reciprocal lattice vectors are [1,0,0], [0,1,0], [0,0,1], etc.
"""

import numpy as np
from pathlib import Path
from phonproj.modes import PhononModes

# Test data path
BATIO3_YAML_PATH = Path(__file__).parent / "data" / "BaTiO3_phonopy_params.yaml"


def debug_3x3x3_zone_folding():
    """Debug zone-folding for 3x3x3 supercell."""

    print("=== Debug Zone-Folding for 3x3x3 ===")

    # Generate all required q-points for 3x3x3 supercell
    qpoints_3x3x3 = []
    qpoint_to_ijk = {}  # Maps qpoint index to (i,j,k) tuple
    for i in range(3):
        for j in range(3):
            for k in range(3):
                idx = len(qpoints_3x3x3)
                qpt = [i / 3.0, j / 3.0, k / 3.0]
                qpoints_3x3x3.append(qpt)
                qpoint_to_ijk[idx] = (i, j, k)

    qpoints_3x3x3 = np.array(qpoints_3x3x3)

    print(f"Generated {len(qpoints_3x3x3)} q-points:")
    for idx, qpt in enumerate(qpoints_3x3x3):
        i, j, k = qpoint_to_ijk[idx]
        print(
            f"  Q{idx}: [{qpt[0]:.4f}, {qpt[1]:.4f}, {qpt[2]:.4f}] = [{i}/3, {j}/3, {k}/3]"
        )

    modes = PhononModes.from_phonopy_yaml(str(BATIO3_YAML_PATH), qpoints_3x3x3)

    # Use 3x3x3 supercell
    supercell_matrix = np.array([[3, 0, 0], [0, 3, 0], [0, 0, 3]])

    # Get commensurate q-points
    commensurate_qpoints = modes.get_commensurate_qpoints(supercell_matrix)
    print(f"\nCommensurate q-points: {commensurate_qpoints}")

    # Check for zone-folding equivalencies using general method
    print(f"\n=== Zone-folding analysis ===")

    zone_folded_pairs = set()

    for i, q_i in enumerate(commensurate_qpoints):
        for j, q_j in enumerate(commensurate_qpoints):
            if i < j:
                qpt_i = modes.qpoints[q_i]
                qpt_j = modes.qpoints[q_j]

                # Check if qpt_i and qpt_j are equivalent under zone-folding
                # Method: q1 ≡ q2 (mod 1) means q1 - q2 = integer vector
                diff = qpt_i - qpt_j
                diff_mod = diff - np.round(diff)

                if np.allclose(diff_mod, 0.0, atol=1e-6):
                    zone_folded_pairs.add(
                        (i, j)
                    )  # Store indices into commensurate_qpoints
                    # q_i and q_j are the actual qpoint indices from the original array
                    idx_i, idx_j = qpoint_to_ijk[q_i], qpoint_to_ijk[q_j]
                    print(f"  Zone-folded pair: Q{q_i} {idx_i} ↔ Q{q_j} {idx_j}")
                    print(f"    Q{q_i}: {qpt_i}")
                    print(f"    Q{q_j}: {qpt_j}")
                    print(f"    Diff: {diff}")
                    print(f"    Rounded diff: {np.round(diff)}")

    # Check for other types of equivalencies
    print(f"\n=== Alternative equivalence analysis ===")

    for i, q_i in enumerate(commensurate_qpoints):
        for j, q_j in enumerate(commensurate_qpoints):
            if i < j:
                qpt_i = modes.qpoints[q_i]
                qpt_j = modes.qpoints[q_j]

                # Check different equivalence conditions:
                # 1. q1 + q2 ≈ integer vector (sum condition)
                sum_vec = qpt_i + qpt_j
                sum_mod = sum_vec - np.round(sum_vec)
                is_sum_equiv = np.allclose(sum_mod, 0.0, atol=1e-6)

                # 2. q1 ≈ -q2 (mod 1) (negation equivalence)
                neg_diff = qpt_i + qpt_j  # qpt_i - (-qpt_j)
                neg_diff_mod = neg_diff - np.round(neg_diff)
                is_neg_equiv = np.allclose(neg_diff_mod, 0.0, atol=1e-6)

                if is_sum_equiv or is_neg_equiv:
                    idx_i, idx_j = qpoint_to_ijk[q_i], qpoint_to_ijk[q_j]
                    print(f"  Alternative equivalence: Q{q_i} {idx_i} ↔ Q{q_j} {idx_j}")
                    print(f"    Q{q_i}: {qpt_i}")
                    print(f"    Q{q_j}: {qpt_j}")
                    print(f"    Sum: {sum_vec}")
                    print(f"    Sum mod: {sum_mod}")
                    print(f"    Sum equiv: {is_sum_equiv}")
                    print(f"    Negation equiv: {is_neg_equiv}")

    print(f"\n=== Complete pairwise overlap analysis ===")

    # Check ALL pairwise overlaps to find all high overlaps
    all_high_overlaps = []

    for i in range(len(commensurate_qpoints)):
        q_i = commensurate_qpoints[i]
        for j in range(i + 1, len(commensurate_qpoints)):
            q_j = commensurate_qpoints[j]

            displacements_i = all_commensurate_displacements[q_i]
            displacements_j = all_commensurate_displacements[q_j]

            # Check same mode index from different q-points
            projection = modes.mass_weighted_projection(
                displacements_i[0], displacements_j[0], supercell_masses
            )
            overlap = abs(projection)

            if overlap > 0.95:  # Lower threshold to catch more cases
                all_high_overlaps.append((q_i, q_j, overlap))

    print(f"Found {len(all_high_overlaps)} pairs with overlap > 0.95:")
    for q_i, q_j, overlap in all_high_overlaps:
        qpt_i = modes.qpoints[q_i]
        qpt_j = modes.qpoints[q_j]
        idx_i, idx_j = qpoint_to_ijk[q_i], qpoint_to_ijk[q_j]
        print(f"  Q{q_i} {idx_i} ↔ Q{q_j} {idx_j}: overlap = {overlap:.6f}")
        print(f"    {qpt_i} ↔ {qpt_j}")

        # Check sum equivalence
        sum_vec = qpt_i + qpt_j
        sum_mod = sum_vec - np.round(sum_vec)
        is_sum_equiv = np.allclose(sum_mod, 0.0, atol=1e-6)
        if is_sum_equiv:
            print(
                f"    --> SUM EQUIVALENT: {qpt_i} + {qpt_j} = {sum_vec} ≈ {np.round(sum_vec)}"
            )

    print(f"\nFound {len(zone_folded_pairs)} zone-folded pairs")

    # Now test specific cases that might have high overlap
    print(f"\n=== Testing specific high-overlap cases ===")

    # Generate displacements
    all_commensurate_displacements = modes.generate_all_commensurate_displacements(
        supercell_matrix, amplitude=1.0
    )

    supercell_masses = np.tile(modes.atomic_masses, 27)

    # Check some specific pairs that might have high overlap
    high_overlap_pairs = []

    # Test first few pairs to find the problematic ones
    for i in range(min(10, len(commensurate_qpoints))):
        q_i = commensurate_qpoints[i]
        for j in range(i + 1, min(10, len(commensurate_qpoints))):
            q_j = commensurate_qpoints[j]

            displacements_i = all_commensurate_displacements[q_i]
            displacements_j = all_commensurate_displacements[q_j]

            # Check same mode index from different q-points
            projection = modes.mass_weighted_projection(
                displacements_i[0], displacements_j[0], supercell_masses
            )
            overlap = abs(projection)

            if overlap > 0.9:  # High overlap threshold
                high_overlap_pairs.append((q_i, q_j, overlap))
                qpt_i = modes.qpoints[q_i]
                qpt_j = modes.qpoints[q_j]
                # q_i and q_j are already the actual qpoint indices
                idx_i, idx_j = qpoint_to_ijk[q_i], qpoint_to_ijk[q_j]

                print(f"  High overlap pair: Q{q_i} {idx_i} ↔ Q{q_j} {idx_j}")
                print(f"    Q{q_i}: {qpt_i}")
                print(f"    Q{q_j}: {qpt_j}")
                print(f"    Overlap: {overlap:.6f}")

                # Check if they are zone-folding equivalent
                diff = qpt_i - qpt_j
                diff_mod = diff - np.round(diff)
                is_zone_folded = np.allclose(diff_mod, 0.0, atol=1e-6)
                print(f"    Zone-folded: {is_zone_folded}")
                print(f"    Diff: {diff}")
                print(f"    Diff mod: {diff_mod}")

    print(f"\nFound {len(high_overlap_pairs)} high-overlap pairs")

    # Check for other types of equivalencies beyond zone-folding
    print(f"\n=== Alternative equivalence analysis ===")

    alternative_equiv_pairs = set()
    for i, q_i in enumerate(commensurate_qpoints):
        for j, q_j in enumerate(commensurate_qpoints):
            if i < j:
                qpt_i = modes.qpoints[q_i]
                qpt_j = modes.qpoints[q_j]

                # Check sum condition: q1 + q2 ≈ integer vector
                sum_vec = qpt_i + qpt_j
                sum_mod = sum_vec - np.round(sum_vec)
                is_sum_equiv = np.allclose(sum_mod, 0.0, atol=1e-6)

                if is_sum_equiv:
                    alternative_equiv_pairs.add((i, j))
                    idx_i, idx_j = qpoint_to_ijk[q_i], qpoint_to_ijk[q_j]
                    print(f"  Sum-equivalent pair: Q{q_i} {idx_i} ↔ Q{q_j} {idx_j}")
                    print(f"    Q{q_i}: {qpt_i}")
                    print(f"    Q{q_j}: {qpt_j}")
                    print(f"    Sum: {sum_vec} ≈ {np.round(sum_vec)}")

    print(f"Found {len(alternative_equiv_pairs)} sum-equivalent pairs")

    print(
        f"\nTotal equivalent pairs: {len(alternative_equiv_pairs)} (zone-folded: {len(zone_folded_pairs)})"
    )

    # Identify unique q-points for completeness test
    if len(zone_folded_pairs) > 0:
        unique_q_indices = set(commensurate_qpoints)

        # Remove higher index from each zone-folded pair
        for q_i, q_j in zone_folded_pairs:
            if q_j in unique_q_indices:
                unique_q_indices.remove(q_j)
                print(
                    f"  Removing Q{q_j} (equivalent to Q{q_i}) from completeness test"
                )

        print(f"Unique q-indices for completeness: {sorted(unique_q_indices)}")
        print(f"Total unique q-points: {len(unique_q_indices)}")
        print(
            f"Expected unique modes: {len(unique_q_indices)} × {modes._n_modes} = {len(unique_q_indices) * modes._n_modes}"
        )

    else:
        print("No zone-folded pairs found - all q-points are unique")


if __name__ == "__main__":
    debug_3x3x3_zone_folding()
