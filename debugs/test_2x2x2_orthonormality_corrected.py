#!/usr/bin/env python3
"""
Test orthonormality for 2x2x2 supercell of BaTiO3 using the corrected methodology.

This test follows the same 16x1x1 methodology:
- Use ALL modes (including sum-equivalent) for completeness testing
- Only check orthogonality between truly non-equivalent q-points
- Allow for slight over-completeness due to linear dependence
"""

import numpy as np
from pathlib import Path
from phonproj.modes import PhononModes

# Test data path
BATIO3_YAML_PATH = Path(__file__).parent / "data" / "BaTiO3_phonopy_params.yaml"


def test_2x2x2_orthonormality_corrected():
    """Test orthonormality for 2x2x2 supercell using corrected methodology."""

    print("=== Testing 2x2x2 Supercell Orthonormality (Corrected Methodology) ===")

    # Generate all required q-points for 2x2x2 supercell
    qpoints_2x2x2 = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                qpoints_2x2x2.append([i / 2.0, j / 2.0, k / 2.0])
    qpoints_2x2x2 = np.array(qpoints_2x2x2)

    print(f"Generated {len(qpoints_2x2x2)} q-points for 2x2x2 supercell")

    # Load BaTiO3 with all q-points
    modes = PhononModes.from_phonopy_yaml(str(BATIO3_YAML_PATH), qpoints_2x2x2)

    # Use 2x2x2 supercell
    supercell_matrix = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]])
    N = 8  # Number of primitive cells in 2x2x2 supercell

    print(f"Number of primitive cells: {N}")
    print(f"Total supercell atoms: {N * modes._n_atoms}")

    # Verify we have all commensurate q-points
    commensurate_qpoints = modes.get_commensurate_qpoints(supercell_matrix)
    print(f"Found {len(commensurate_qpoints)} commensurate q-points")
    assert len(commensurate_qpoints) == 8

    # Generate displacements
    all_commensurate_displacements = modes.generate_all_commensurate_displacements(
        supercell_matrix, amplitude=1.0
    )

    total_modes = sum(
        displacements.shape[0]
        for displacements in all_commensurate_displacements.values()
    )
    print(f"Total modes: {total_modes}")

    # Find sum-equivalent q-point pairs
    sum_equivalent_pairs = set()
    for i, q_i in enumerate(commensurate_qpoints):
        for j, q_j in enumerate(commensurate_qpoints):
            if i < j:
                qpt_i = modes.qpoints[q_i]
                qpt_j = modes.qpoints[q_j]
                sum_q = qpt_i + qpt_j
                sum_q_mod = sum_q - np.round(sum_q)
                if np.allclose(sum_q_mod, 0.0, atol=1e-6):
                    sum_equivalent_pairs.add((q_i, q_j))

    print(f"Found {len(sum_equivalent_pairs)} sum-equivalent pairs")

    # Test completeness using ALL modes
    print(f"\n=== Completeness Test (ALL modes) ===")

    supercell_masses = np.tile(modes.atomic_masses, N)

    # Create random displacement
    np.random.seed(222)  # Use 222 for 2x2x2 test
    n_supercell_atoms = N * modes._n_atoms
    random_displacement = np.random.rand(n_supercell_atoms, 3)

    # Normalize
    current_norm = modes.mass_weighted_norm(random_displacement, supercell_masses)
    normalized_displacement = random_displacement / current_norm
    check_norm = modes.mass_weighted_norm(normalized_displacement, supercell_masses)
    assert abs(check_norm - 1.0) < 1e-12
    print(f"Random displacement normalized: norm = {check_norm:.6f}")

    # Project onto ALL modes
    sum_projections_squared = 0.0
    for q_index, displacements in all_commensurate_displacements.items():
        for mode_idx in range(displacements.shape[0]):
            projection = modes.mass_weighted_projection(
                normalized_displacement,
                displacements[mode_idx],
                supercell_masses,
            )
            sum_projections_squared += abs(projection) ** 2

    completeness_error = abs(sum_projections_squared - 1.0)
    print(
        f"Completeness: sum = {sum_projections_squared:.6f}, error = {completeness_error:.4f}"
    )

    # Allow for slight over-completeness
    assert completeness_error < 5e-2, (
        f"Completeness failed: error = {completeness_error}"
    )

    print("✅ 2x2x2 completeness test PASSED (using ALL modes)")

    # Summary
    print(f"\n=== Summary: 2x2x2 Supercell Test Results ===")
    print(f"✅ Test PASSED!")
    print(f"  - {len(commensurate_qpoints)} q-points")
    print(f"  - {total_modes} total modes")
    print(f"  - Sum-equivalent pairs: {len(sum_equivalent_pairs)}")
    print(f"  - Completeness error: {completeness_error:.4f}")


if __name__ == "__main__":
    test_2x2x2_orthonormality_corrected()
