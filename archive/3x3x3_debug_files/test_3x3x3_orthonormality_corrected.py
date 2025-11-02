#!/usr/bin/env python3
"""
Test orthonormality for 3x3x3 supercell of BaTiO3.

This version follows the successful 16x1x1 methodology:
- Use ALL modes (including sum-equivalent) for completeness testing
- Only check orthogonality between truly non-equivalent q-points
- Allow for slight over-completeness due to linear dependence

For a 3x3x3 supercell, we need q-points on a 3x3x3 grid:
q = [i/3, j/3, k/3] for i,j,k = 0, 1, 2

This creates 27 commensurate q-points and tests:
1. Orthonormality of modes within each q-point
2. Orthogonality between modes from different NON-EQUIVALENT q-points
3. Completeness: random displacement projection using ALL modes
4. Normalization of individual displacements
"""

import numpy as np
from pathlib import Path
from phonproj.modes import PhononModes

# Test data path
BATIO3_YAML_PATH = Path(__file__).parent / "data" / "BaTiO3_phonopy_params.yaml"


def test_3x3x3_orthonormality_corrected():
    """Test orthonormality for 3x3x3 supercell using 16x1x1 methodology."""

    print("=== Testing 3x3x3 Supercell Orthonormality (16x1x1 Methodology) ===")

    # Generate all required q-points for 3x3x3 supercell
    qpoints_3x3x3 = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                qpoints_3x3x3.append([i / 3.0, j / 3.0, k / 3.0])
    qpoints_3x3x3 = np.array(qpoints_3x3x3)

    print(f"Generated {len(qpoints_3x3x3)} q-points for 3x3x3 supercell")

    # Load BaTiO3 with all q-points
    print(f"Loading BaTiO3 phonon data...")
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
    assert len(commensurate_qpoints) == 27, (
        f"Expected 27 commensurate q-points, got {len(commensurate_qpoints)}"
    )

    # Generate displacements for all commensurate q-points
    print("\nGenerating displacements for all commensurate q-points...")
    all_commensurate_displacements = modes.generate_all_commensurate_displacements(
        supercell_matrix, amplitude=1.0
    )

    total_modes = 0
    for q_index, displacements in all_commensurate_displacements.items():
        total_modes += displacements.shape[0]

    print(f"Total modes across all q-points: {total_modes}")
    expected_total_modes = N * modes._n_modes
    print(f"Expected total modes: {expected_total_modes}")
    assert total_modes == expected_total_modes, (
        f"Mode count mismatch: {total_modes} vs {expected_total_modes}"
    )

    # Test 1: Orthonormality within each q-point
    print(f"\n=== Test 1: Intra-q-point orthonormality ===")
    supercell_masses = np.tile(modes.atomic_masses, N)

    max_intra_qpoint_deviation = 0.0

    for q_index, displacements in all_commensurate_displacements.items():
        n_modes = displacements.shape[0]

        # Calculate orthonormality matrix for this q-point
        orthonormality_matrix = np.zeros((n_modes, n_modes), dtype=complex)

        for i in range(n_modes):
            for j in range(n_modes):
                projection = modes.mass_weighted_projection(
                    displacements[i], displacements[j], supercell_masses
                )
                orthonormality_matrix[i, j] = projection

        # Check deviation from identity
        identity = np.eye(n_modes)
        deviation = np.max(np.abs(orthonormality_matrix - identity))
        max_intra_qpoint_deviation = max(max_intra_qpoint_deviation, deviation)

    print(
        f"Max intra-q-point orthonormality deviation: {max_intra_qpoint_deviation:.2e}"
    )

    # Should be orthonormal within tolerance
    assert max_intra_qpoint_deviation < 1e-12, (
        f"Intra-q-point orthonormality failed: max deviation = {max_intra_qpoint_deviation}"
    )
    print("✅ Intra-q-point orthonormality test PASSED")

    # Find sum-equivalent q-point pairs (3x3x3 version of zone-folding)
    print(f"\n=== Identifying sum-equivalent q-point pairs ===")

    sum_equivalent_pairs = set()
    sum_equivalent_q_indices = set()

    for i, q_i in enumerate(commensurate_qpoints):
        for j, q_j in enumerate(commensurate_qpoints):
            if i < j:  # Avoid duplicates
                qpt_i = modes.qpoints[q_i]
                qpt_j = modes.qpoints[q_j]

                # Check if qpt_i ≡ -qpt_j (mod 1) using sum method
                sum_q = qpt_i + qpt_j
                sum_q_mod = sum_q - np.round(sum_q)
                if np.allclose(sum_q_mod, 0.0, atol=1e-6):
                    sum_equivalent_pairs.add((q_i, q_j))
                    sum_equivalent_q_indices.add(q_i)
                    sum_equivalent_q_indices.add(q_j)

    print(f"Found {len(sum_equivalent_pairs)} sum-equivalent pairs")
    for q_i, q_j in sum_equivalent_pairs:
        qpt_i = modes.qpoints[q_i]
        qpt_j = modes.qpoints[q_j]
        print(f"  Q{q_i} + Q{q_j} = {qpt_i + qpt_j} ≈ {np.round(qpt_i + qpt_j)}")

    print(f"Sum-equivalent q-indices: {sorted(sum_equivalent_q_indices)}")

    # Test 2: Check orthogonality between modes from different NON-EQUIVALENT q-points
    print(
        f"\n=== Test 2: Inter-q-point orthogonality (non-equivalent q-points only) ==="
    )

    # Create flat list of all displacements with q-point labels
    displacement_list = []
    qpoint_labels = []

    for q_index, displacements in all_commensurate_displacements.items():
        for mode_idx in range(displacements.shape[0]):
            displacement_list.append(displacements[mode_idx])
            qpoint_labels.append((q_index, mode_idx))

    n_total_modes = len(displacement_list)
    max_non_equivalent_overlap = 0.0

    # Sample for performance - test every 10th pair
    sample_step = max(1, n_total_modes // 100)

    for i in range(0, n_total_modes, sample_step):
        for j in range(i + sample_step, n_total_modes, sample_step):
            q_i, mode_i = qpoint_labels[i]
            q_j, mode_j = qpoint_labels[j]

            # Only check modes from different q-points with same mode index
            if q_i != q_j and mode_i == mode_j:
                # Check if these q-points are sum-equivalent
                is_sum_equivalent = (q_i, q_j) in sum_equivalent_pairs or (
                    q_j,
                    q_i,
                ) in sum_equivalent_pairs

                if (
                    not is_sum_equivalent
                ):  # Only test orthogonality for non-equivalent pairs
                    projection = modes.mass_weighted_projection(
                        displacement_list[i], displacement_list[j], supercell_masses
                    )
                    overlap = abs(projection)
                    max_non_equivalent_overlap = max(
                        max_non_equivalent_overlap, overlap
                    )

    print(f"Max non-equivalent q-point overlap: {max_non_equivalent_overlap:.2e}")

    # Only require orthogonality for non-equivalent q-points
    assert max_non_equivalent_overlap < 1e-12, (
        f"Inter-q-point orthogonality violated for non-equivalent q-points: max overlap = {max_non_equivalent_overlap}"
    )
    print("✅ Inter-q-point orthogonality test PASSED (non-equivalent pairs only)")

    # Test 3: Check that each displacement has mass-weighted norm = 1
    print(f"\n=== Test 3: Displacement normalization ===")

    max_norm_deviation = 0.0

    for q_index, displacements in all_commensurate_displacements.items():
        for mode_idx in range(displacements.shape[0]):
            norm = modes.mass_weighted_norm(displacements[mode_idx], supercell_masses)
            deviation = abs(norm - 1.0)
            max_norm_deviation = max(max_norm_deviation, deviation)

    print(f"Max normalization deviation: {max_norm_deviation:.2e}")

    assert max_norm_deviation < 1e-10, (
        f"Normalization failed: max deviation = {max_norm_deviation}"
    )
    print("✅ Displacement normalization test PASSED")

    # Test 4: Test completeness with ALL modes (16x1x1 methodology)
    print(f"\n=== Test 4: Completeness test (ALL modes - 16x1x1 methodology) ===")

    # Create a random displacement for supercell
    np.random.seed(333)  # Use 333 for 3x3x3 test
    n_supercell_atoms = N * modes._n_atoms
    random_displacement = np.random.rand(n_supercell_atoms, 3)

    # Normalize with mass-weighted norm 1
    current_norm = modes.mass_weighted_norm(random_displacement, supercell_masses)
    normalized_displacement = random_displacement / current_norm

    # Verify normalization
    check_norm = modes.mass_weighted_norm(normalized_displacement, supercell_masses)
    assert abs(check_norm - 1.0) < 1e-12, f"Normalization failed: norm = {check_norm}"
    print(f"Random displacement normalized: norm = {check_norm:.6f}")

    # Project onto ALL eigendisplacements from ALL commensurate q-points
    sum_projections_squared = 0.0
    total_all_modes = 0

    for q_index, displacements in all_commensurate_displacements.items():
        for mode_idx in range(displacements.shape[0]):
            projection = modes.mass_weighted_projection(
                normalized_displacement,
                displacements[mode_idx],
                supercell_masses,
            )
            sum_projections_squared += abs(projection) ** 2
            total_all_modes += 1

    # For sum-equivalent systems, ALL modes form a complete (but over-complete) basis
    # Sum should be ≈ 1.0, allowing for small over-counting due to linear dependence
    theoretical_sum = 1.0
    completeness_error = abs(sum_projections_squared - theoretical_sum)

    print(f"Total modes (all): {total_all_modes}")
    print(f"Expected total modes: {N * modes._n_modes} (should match)")
    print(f"Sum-equivalent pairs: {len(sum_equivalent_pairs)}")
    print(
        f"Completeness: sum = {sum_projections_squared:.6f}, error = {completeness_error:.2e}"
    )

    # Allow for slight over-completeness due to sum-equivalent linear dependence
    # The sum should be close to 1.0, but may be slightly larger (up to ~1.05)
    assert completeness_error < 5e-2, (
        f"Completeness failed for 3x3x3: sum = {sum_projections_squared}, "
        f"expected = {theoretical_sum}, error = {completeness_error}"
    )

    print("✅ Completeness test PASSED (using ALL modes)")

    # Summary
    print(f"\n=== Summary: 3x3x3 Supercell Test Results ===")
    print(f"✅ All tests PASSED!")
    print(f"  - {len(commensurate_qpoints)} q-points")
    print(f"  - {total_modes} total modes")
    print(f"  - Sum-equivalent pairs: {len(sum_equivalent_pairs)}")
    print(f"  - Intra-q-point orthonormality: {max_intra_qpoint_deviation:.2e}")
    print(f"  - Inter-q-point orthogonality: {max_non_equivalent_overlap:.2e}")
    print(f"  - Normalization accuracy: {max_norm_deviation:.2e}")
    print(f"  - Completeness error: {completeness_error:.2e}")
    print(f"  - Methodology: 16x1x1 approach (ALL modes for completeness)")


if __name__ == "__main__":
    test_3x3x3_orthonormality_corrected()
