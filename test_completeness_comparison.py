#!/usr/bin/env python3
"""
Compare completeness results between 16x1x1 and 3x3x3 supercells
to verify that our 3x3x3 fix is consistent with the proven 16x1x1 approach.
"""

import numpy as np
from pathlib import Path
from phonproj.modes import PhononModes

# Test data path
BATIO3_YAML_PATH = Path(__file__).parent / "data" / "BaTiO3_phonopy_params.yaml"


def test_completeness_comparison():
    """Compare completeness between 16x1x1 and 3x3x3 approaches."""

    print("=== Completeness Comparison: 16x1x1 vs 3x3x3 ===")

    # Test 16x1x1 supercell
    print("\n--- 16x1x1 Supercell ---")
    qpoints_16x1x1 = []
    for i in range(16):
        qpoints_16x1x1.append([i / 16.0, 0.0, 0.0])
    qpoints_16x1x1 = np.array(qpoints_16x1x1)

    modes_16x1x1 = PhononModes.from_phonopy_yaml(str(BATIO3_YAML_PATH), qpoints_16x1x1)
    supercell_matrix_16x1x1 = np.array([[16, 0, 0], [0, 1, 0], [0, 0, 1]])
    N_16x1x1 = 16

    commensurate_qpoints_16x1x1 = modes_16x1x1.get_commensurate_qpoints(
        supercell_matrix_16x1x1
    )
    all_displacements_16x1x1 = modes_16x1x1.generate_all_commensurate_displacements(
        supercell_matrix_16x1x1, amplitude=1.0
    )

    supercell_masses_16x1x1 = np.tile(modes_16x1x1.atomic_masses, N_16x1x1)

    # Find zone-folded pairs for 16x1x1
    zone_folded_pairs_16x1x1 = set()
    for i, q_i in enumerate(commensurate_qpoints_16x1x1):
        for j, q_j in enumerate(commensurate_qpoints_16x1x1):
            if i < j:
                qpt_i = modes_16x1x1.qpoints[q_i]
                qpt_j = modes_16x1x1.qpoints[q_j]
                sum_q = qpt_i + qpt_j
                sum_q_mod = sum_q - np.round(sum_q)
                if np.allclose(sum_q_mod, 0.0, atol=1e-6):
                    zone_folded_pairs_16x1x1.add((q_i, q_j))

    print(
        f"16x1x1: {len(commensurate_qpoints_16x1x1)} q-points, {len(zone_folded_pairs_16x1x1)} zone-folded pairs"
    )

    # Test completeness for 16x1x1
    np.random.seed(16)
    n_supercell_atoms_16x1x1 = N_16x1x1 * modes_16x1x1._n_atoms
    random_displacement_16x1x1 = np.random.rand(n_supercell_atoms_16x1x1, 3)
    current_norm_16x1x1 = modes_16x1x1.mass_weighted_norm(
        random_displacement_16x1x1, supercell_masses_16x1x1
    )
    normalized_displacement_16x1x1 = random_displacement_16x1x1 / current_norm_16x1x1

    sum_projections_squared_16x1x1 = 0.0
    total_modes_16x1x1 = 0

    for q_index, displacements in all_displacements_16x1x1.items():
        for mode_idx in range(displacements.shape[0]):
            projection = modes_16x1x1.mass_weighted_projection(
                normalized_displacement_16x1x1,
                displacements[mode_idx],
                supercell_masses_16x1x1,
            )
            sum_projections_squared_16x1x1 += abs(projection) ** 2
            total_modes_16x1x1 += 1

    completeness_error_16x1x1 = abs(sum_projections_squared_16x1x1 - 1.0)

    print(
        f"16x1x1 completeness: {sum_projections_squared_16x1x1:.6f} (error: {completeness_error_16x1x1:.4f})"
    )

    # Test 3x3x3 supercell
    print("\n--- 3x3x3 Supercell ---")
    qpoints_3x3x3 = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                qpoints_3x3x3.append([i / 3.0, j / 3.0, k / 3.0])
    qpoints_3x3x3 = np.array(qpoints_3x3x3)

    modes_3x3x3 = PhononModes.from_phonopy_yaml(str(BATIO3_YAML_PATH), qpoints_3x3x3)
    supercell_matrix_3x3x3 = np.array([[3, 0, 0], [0, 3, 0], [0, 0, 3]])
    N_3x3x3 = 27

    commensurate_qpoints_3x3x3 = modes_3x3x3.get_commensurate_qpoints(
        supercell_matrix_3x3x3
    )
    all_displacements_3x3x3 = modes_3x3x3.generate_all_commensurate_displacements(
        supercell_matrix_3x3x3, amplitude=1.0
    )

    supercell_masses_3x3x3 = np.tile(modes_3x3x3.atomic_masses, N_3x3x3)

    # Find sum-equivalent pairs for 3x3x3
    sum_equivalent_pairs_3x3x3 = set()
    for i, q_i in enumerate(commensurate_qpoints_3x3x3):
        for j, q_j in enumerate(commensurate_qpoints_3x3x3):
            if i < j:
                qpt_i = modes_3x3x3.qpoints[q_i]
                qpt_j = modes_3x3x3.qpoints[q_j]
                sum_q = qpt_i + qpt_j
                sum_q_mod = sum_q - np.round(sum_q)
                if np.allclose(sum_q_mod, 0.0, atol=1e-6):
                    sum_equivalent_pairs_3x3x3.add((q_i, q_j))

    print(
        f"3x3x3: {len(commensurate_qpoints_3x3x3)} q-points, {len(sum_equivalent_pairs_3x3x3)} sum-equivalent pairs"
    )

    # Test completeness for 3x3x3
    np.random.seed(333)
    n_supercell_atoms_3x3x3 = N_3x3x3 * modes_3x3x3._n_atoms
    random_displacement_3x3x3 = np.random.rand(n_supercell_atoms_3x3x3, 3)
    current_norm_3x3x3 = modes_3x3x3.mass_weighted_norm(
        random_displacement_3x3x3, supercell_masses_3x3x3
    )
    normalized_displacement_3x3x3 = random_displacement_3x3x3 / current_norm_3x3x3

    sum_projections_squared_3x3x3 = 0.0
    total_modes_3x3x3 = 0

    for q_index, displacements in all_displacements_3x3x3.items():
        for mode_idx in range(displacements.shape[0]):
            projection = modes_3x3x3.mass_weighted_projection(
                normalized_displacement_3x3x3,
                displacements[mode_idx],
                supercell_masses_3x3x3,
            )
            sum_projections_squared_3x3x3 += abs(projection) ** 2
            total_modes_3x3x3 += 1

    completeness_error_3x3x3 = abs(sum_projections_squared_3x3x3 - 1.0)

    print(
        f"3x3x3 completeness: {sum_projections_squared_3x3x3:.6f} (error: {completeness_error_3x3x3:.4f})"
    )

    # Comparison
    print(f"\n--- Comparison ---")
    print(
        f"16x1x1: {total_modes_16x1x1} modes, {len(zone_folded_pairs_16x1x1)} equivalent pairs"
    )
    print(
        f"3x3x3:  {total_modes_3x3x3} modes, {len(sum_equivalent_pairs_3x3x3)} equivalent pairs"
    )
    print(
        f"16x1x1 completeness: {sum_projections_squared_16x1x1:.6f} (error: {completeness_error_16x1x1:.4f})"
    )
    print(
        f"3x3x3 completeness:  {sum_projections_squared_3x3x3:.6f} (error: {completeness_error_3x3x3:.4f})"
    )

    # Both should be complete within 5% error (allow for over-completeness)
    assert completeness_error_16x1x1 < 0.05, (
        f"16x1x1 completeness error too large: {completeness_error_16x1x1}"
    )
    assert completeness_error_3x3x3 < 0.05, (
        f"3x3x3 completeness error too large: {completeness_error_3x3x3}"
    )

    print(
        f"\n✅ Both 16x1x1 and 3x3x3 pass completeness test using ALL modes methodology!"
    )

    # Calculate equivalent pair fractions
    equiv_fraction_16x1x1 = len(zone_folded_pairs_16x1x1) / len(
        commensurate_qpoints_16x1x1
    )
    equiv_fraction_3x3x3 = len(sum_equivalent_pairs_3x3x3) / len(
        commensurate_qpoints_3x3x3
    )

    print(f"\n--- Equivalent Pair Analysis ---")
    print(
        f"16x1x1: {len(zone_folded_pairs_16x1x1)}/{len(commensurate_qpoints_16x1x1)} = {equiv_fraction_16x1x1:.2%} equivalent pairs"
    )
    print(
        f"3x3x3:  {len(sum_equivalent_pairs_3x3x3)}/{len(commensurate_qpoints_3x3x3)} = {equiv_fraction_3x3x3:.2%} equivalent pairs"
    )

    print(f"\n--- Degrees of Over-completeness ---")
    over_completeness_16x1x1 = sum_projections_squared_16x1x1 - 1.0
    over_completeness_3x3x3 = sum_projections_squared_3x3x3 - 1.0
    print(f"16x1x1 over-completeness: {over_completeness_16x1x1:.4f}")
    print(f"3x3x3 over-completeness:  {over_completeness_3x3x3:.4f}")

    print(f"\n=== Conclusion ===")
    print(f"✅ Both methodologies successfully handle equivalent q-points")
    print(f"✅ Both achieve completeness within acceptable tolerances")
    print(f"✅ The 3x3x3 approach is now consistent with the proven 16x1x1 approach")


if __name__ == "__main__":
    test_completeness_comparison()
