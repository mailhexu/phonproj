#!/usr/bin/env python3
"""
Debug the 3x3x3 completeness issue by testing both approaches:
1. Using unique modes only (current failing approach)
2. Using ALL modes including sum-equivalent (16x1x1 successful approach)
"""

import numpy as np
from pathlib import Path
from phonproj.modes import PhononModes

# Test data path
BATIO3_YAML_PATH = Path(__file__).parent / "data" / "BaTiO3_phonopy_params.yaml"


def test_3x3x3_completeness_debug():
    """Debug completeness for 3x3x3 supercell."""

    print("=== Debug 3x3x3 Completeness Issue ===")

    # Generate all required q-points for 3x3x3 supercell
    qpoints_3x3x3 = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                qpoints_3x3x3.append([i / 3.0, j / 3.0, k / 3.0])
    qpoints_3x3x3 = np.array(qpoints_3x3x3)

    print(f"Generated {len(qpoints_3x3x3)} q-points for 3x3x3 supercell")

    # Load BaTiO3 with all q-points
    modes = PhononModes.from_phonopy_yaml(str(BATIO3_YAML_PATH), qpoints_3x3x3)

    # Use 3x3x3 supercell
    supercell_matrix = np.array([[3, 0, 0], [0, 3, 0], [0, 0, 3]])
    N = 27  # Number of primitive cells in 3x3x3 supercell

    print(f"Supercell matrix:\n{supercell_matrix}")
    print(f"Number of primitive cells: {N}")
    print(f"Atoms per primitive cell: {modes._n_atoms}")
    print(f"Modes per q-point: {modes._n_modes}")
    print(f"Total supercell atoms: {N * modes._n_atoms}")
    print(f"Total degrees of freedom: {N * modes._n_atoms * 3}")

    # Verify we have all commensurate q-points
    commensurate_qpoints = modes.get_commensurate_qpoints(supercell_matrix)
    print(f"\nFound {len(commensurate_qpoints)} commensurate q-points")
    print(f"Commensurate q-points: {commensurate_qpoints}")

    # Generate displacements for all commensurate q-points
    print("\nGenerating displacements for all commensurate q-points...")
    all_commensurate_displacements = modes.generate_all_commensurate_displacements(
        supercell_matrix, amplitude=1.0
    )

    total_modes = 0
    for q_index, displacements in all_commensurate_displacements.items():
        print(f"  Q{q_index}: {displacements.shape[0]} modes")
        total_modes += displacements.shape[0]

    print(f"Total modes across all q-points: {total_modes}")

    # Identify sum-equivalent q-point pairs
    print(f"\n=== Identifying sum-equivalent q-point pairs ===")

    sum_equivalent_pairs = set()
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
                    sum_equivalent_pairs.add((q_i, q_j))  # Store actual q-indices

    print(f"Found {len(sum_equivalent_pairs)} sum-equivalent pairs")
    for q_i, q_j in sum_equivalent_pairs:
        qpt_i = modes.qpoints[q_i]
        qpt_j = modes.qpoints[q_j]
        print(f"  Q{q_i} + Q{q_j} = {qpt_i + qpt_j} ≈ {np.round(qpt_i + qpt_j)}")

    # Create a random displacement for testing completeness
    np.random.seed(333)  # Use 333 for 3x3x3 test
    n_supercell_atoms = N * modes._n_atoms
    random_displacement = np.random.rand(n_supercell_atoms, 3)

    # Normalize with mass-weighted norm 1
    supercell_masses = np.tile(modes.atomic_masses, N)
    current_norm = modes.mass_weighted_norm(random_displacement, supercell_masses)
    normalized_displacement = random_displacement / current_norm

    # Verify normalization
    check_norm = modes.mass_weighted_norm(normalized_displacement, supercell_masses)
    assert abs(check_norm - 1.0) < 1e-12, f"Normalization failed: norm = {check_norm}"
    print(f"\nRandom displacement normalized: norm = {check_norm:.6f}")

    # Test completeness using ALL modes (16x1x1 approach)
    print(f"\n=== Completeness Test 1: ALL modes (16x1x1 approach) ===")

    sum_projections_squared_all = 0.0
    total_all_modes = 0

    for q_index, displacements in all_commensurate_displacements.items():
        for mode_idx in range(displacements.shape[0]):
            projection = modes.mass_weighted_projection(
                normalized_displacement,
                displacements[mode_idx],
                supercell_masses,
            )
            sum_projections_squared_all += abs(projection) ** 2
            total_all_modes += 1

    theoretical_sum = 1.0
    completeness_error_all = abs(sum_projections_squared_all - theoretical_sum)

    print(f"Total modes used: {total_all_modes}")
    print(f"Sum of projections squared: {sum_projections_squared_all:.6f}")
    print(f"Expected sum: {theoretical_sum:.6f}")
    print(f"Completeness error: {completeness_error_all:.2e}")
    print(f"Ratio: {sum_projections_squared_all / theoretical_sum:.6f}")

    # Test completeness using unique modes only (current failing approach)
    print(f"\n=== Completeness Test 2: Unique modes only (current approach) ===")

    # Remove higher index from each sum-equivalent pair
    unique_q_indices = set(commensurate_qpoints)
    removed_count = 0

    for q_i, q_j in sum_equivalent_pairs:
        # Remove the higher-indexed q-point
        if q_j in unique_q_indices:
            unique_q_indices.remove(q_j)
            removed_count += 1
            print(f"  Removing Q{q_j} (equivalent to Q{q_i})")

    unique_q_indices = sorted(unique_q_indices)
    print(f"Unique q-indices: {unique_q_indices}")
    print(f"Total unique q-points: {len(unique_q_indices)}")
    print(f"Removed {removed_count} sum-equivalent duplicates")

    sum_projections_squared_unique = 0.0
    total_unique_modes = 0

    for q_index in unique_q_indices:
        displacements = all_commensurate_displacements[q_index]
        for mode_idx in range(displacements.shape[0]):
            projection = modes.mass_weighted_projection(
                normalized_displacement,
                displacements[mode_idx],
                supercell_masses,
            )
            sum_projections_squared_unique += abs(projection) ** 2
            total_unique_modes += 1

    completeness_error_unique = abs(sum_projections_squared_unique - theoretical_sum)

    print(f"Total unique modes used: {total_unique_modes}")
    print(f"Sum of projections squared: {sum_projections_squared_unique:.6f}")
    print(f"Expected sum: {theoretical_sum:.6f}")
    print(f"Completeness error: {completeness_error_unique:.2e}")
    print(f"Ratio: {sum_projections_squared_unique / theoretical_sum:.6f}")

    # Analysis of the completeness gap
    print(f"\n=== Analysis ===")
    print(
        f"ALL modes: {sum_projections_squared_all:.6f} (error: {completeness_error_all:.4f})"
    )
    print(
        f"Unique modes: {sum_projections_squared_unique:.6f} (error: {completeness_error_unique:.4f})"
    )
    print(
        f"Difference: {sum_projections_squared_all - sum_projections_squared_unique:.6f}"
    )
    print(f"Missing contribution: {(1.0 - sum_projections_squared_unique):.6f}")
    print(f"Percentage missing: {(1.0 - sum_projections_squared_unique) * 100:.1f}%")

    # Let's analyze what the sum-equivalent modes contribute
    print(f"\n=== Sum-equivalent contribution analysis ===")

    sum_equivalent_contribution = 0.0
    for q_i, q_j in sum_equivalent_pairs:
        if q_j not in unique_q_indices:  # This was removed
            displacements = all_commensurate_displacements[q_j]
            for mode_idx in range(displacements.shape[0]):
                projection = modes.mass_weighted_projection(
                    normalized_displacement,
                    displacements[mode_idx],
                    supercell_masses,
                )
                sum_equivalent_contribution += abs(projection) ** 2

    print(f"Sum-equivalent modes contribution: {sum_equivalent_contribution:.6f}")
    print(
        f"Total with sum-equiv: {sum_projections_squared_unique + sum_equivalent_contribution:.6f}"
    )

    # Conclusions
    print(f"\n=== Conclusions ===")
    if completeness_error_all < 0.05:
        print("✅ Using ALL modes (16x1x1 approach): COMPLETENESS SUCCESS")
    else:
        print("❌ Using ALL modes: still incomplete")

    if completeness_error_unique < 0.05:
        print("✅ Using unique modes only: COMPLETENESS SUCCESS")
    else:
        print("❌ Using unique modes only: incomplete (18% missing)")

    print(f"\n=== Recommendation ===")
    if completeness_error_all < completeness_error_unique:
        print("→ Use 16x1x1 methodology: ALL modes for completeness")
        print("→ This creates an over-complete basis but spans the full space")
    else:
        print("→ Further investigation needed - neither approach is fully successful")


if __name__ == "__main__":
    test_3x3x3_completeness_debug()
